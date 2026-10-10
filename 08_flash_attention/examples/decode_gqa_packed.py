"""示例：GQA 打包的 flash-decoding（练习 08-5 的进阶版）。

练习 08-5 里每个 program 处理一个 q head。GQA 下 GROUP 个 q head 共用一个 kv head，
于是同一份 K/V 被 GROUP 个 program 各读一遍（靠 L2 命中才不至于太惨）。
更好的做法：一个 program 负责一个 kv head 的**全部** GROUP 个 q head，
把它们当成 M 维拼成 [GROUP, D] 的小矩阵，这样
  1. K/V 只从 HBM 读一次；
  2. q·K^T 变成 [GROUP, D] x [D, BLOCK_N] 的矩阵乘，可以用 tl.dot（M 维补到 16）走 Tensor Core。
解码阶段完全是 memory-bound 的，所以 (1) 才是关键。

运行：python 08_flash_attention/examples/decode_gqa_packed.py
"""
import importlib.util
import math
import pathlib

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gbps, gpu_name, report

# 复用练习 08-5 参考答案里的 combine kernel 和参考实现
_sol = pathlib.Path(__file__).resolve().parents[1] / "solutions" / "ex5_flash_decoding.py"
_spec = importlib.util.spec_from_file_location("ex5_sol", _sol)
ex5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ex5)


@triton.jit
def decode_split_gqa_kernel(
    Q, K, V, SEQLENS, O_PART, LSE_PART,
    sm_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    H, HKV, GROUP, NUM_SPLITS, CHUNK,
    HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)                 # = b * HKV + kvh
    split = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV

    seqlen = tl.load(SEQLENS + b)
    start = split * CHUNK
    end = tl.minimum(start + CHUNK, seqlen)

    offs_h = tl.arange(0, BLOCK_H)         # 组内第几个 q head（补齐到 16，多余的行 mask 掉）
    hmask = offs_h < GROUP
    heads = kvh * GROUP + offs_h
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :],
                mask=hmask[:, None], other=0.0)                      # [BLOCK_H, D]
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh
    qk_scale = sm_scale * 1.4426950408889634

    m = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        valid = cols < end
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale                        # [BLOCK_H, BLOCK_N]
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    empty = l == 0.0
    o = tl.where(empty[:, None], 0.0, acc / tl.where(empty, 1.0, l)[:, None])
    lse = tl.where(empty, float("-inf"), (m + tl.math.log2(l)) * 0.6931471805599453)
    part = (b * H + heads) * NUM_SPLITS + split                      # 和 ex5 的 o_part 布局一致
    tl.store(O_PART + part[:, None] * HEAD_DIM + offs_d[None, :], o, mask=hmask[:, None])
    tl.store(LSE_PART + part, lse, mask=hmask)


def flash_decode_gqa(q, k_cache, v_cache, seqlens, num_splits=16, BLOCK_N=64):
    B, H, D = q.shape
    _, Hkv, Nmax, _ = k_cache.shape
    group = H // Hkv
    chunk = triton.cdiv(triton.cdiv(Nmax, num_splits), BLOCK_N) * BLOCK_N
    o_part = torch.empty((B * H, num_splits, D), device=q.device, dtype=torch.float32)
    lse_part = torch.empty((B * H, num_splits), device=q.device, dtype=torch.float32)
    decode_split_gqa_kernel[(B * Hkv, num_splits)](
        q, k_cache, v_cache, seqlens, o_part, lse_part, 1.0 / math.sqrt(D),
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        H, Hkv, group, num_splits, chunk,
        HEAD_DIM=D, BLOCK_N=BLOCK_N, BLOCK_H=max(16, triton.next_power_of_2(group)), num_warps=4,
    )
    o = torch.empty_like(q)
    lse = torch.empty((B, H), device=q.device, dtype=torch.float32)
    ex5.decode_combine_kernel[(B * H,)](o_part, lse_part, o, lse, num_splits,
                                        HEAD_DIM=D, BLOCK_S=triton.next_power_of_2(num_splits))
    return o, lse


if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.bfloat16
    B, H, Hkv, N, D = 3, 32, 8, 4096, 128
    q = torch.randn(B, H, D, device="cuda", dtype=dt)
    kc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    vc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    seqlens = torch.tensor([4096, 17, 2500], device="cuda", dtype=torch.int32)
    o, lse = flash_decode_gqa(q, kc, vc, seqlens)
    o_ref, lse_ref = ex5.ref_decode(q, kc, vc, seqlens)
    check("packed GQA o", o, o_ref, atol=1e-2, rtol=1e-2)
    check("packed GQA lse", lse, lse_ref, atol=1e-3, rtol=1e-3)

    B, N = 1, 32768
    q = torch.randn(B, H, D, device="cuda", dtype=dt)
    kc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    vc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    seqlens = torch.full((B,), N, device="cuda", dtype=torch.int32)
    nbytes = 2 * B * Hkv * N * D * 2
    rows = []
    for splits in [16, 64]:
        ms = bench(lambda: ex5.flash_decode(q, kc, vc, seqlens, num_splits=splits))
        rows.append(dict(impl=f"ex5 每 q head 一个 program, splits={splits}", us=ms * 1e3, GBps=gbps(nbytes, ms)))
        ms = bench(lambda: flash_decode_gqa(q, kc, vc, seqlens, num_splits=splits))
        rows.append(dict(impl=f"GQA 打包, splits={splits}", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    ms = bench(lambda: F.scaled_dot_product_attention(q[:, :, None], kc, vc, enable_gqa=True))
    rows.append(dict(impl="torch SDPA", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"decode B={B} H={H}/{Hkv} N={N} D={D}（实测于 {gpu_name()}）")
    finish()
