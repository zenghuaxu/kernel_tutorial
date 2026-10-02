"""练习 08-5：flash-decoding（split-KV + combine）（参考答案）"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def decode_split_kernel(
    Q, K, V, SEQLENS, O_PART, LSE_PART,
    sm_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    H, GROUP, NUM_SPLITS, CHUNK,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """grid = (B*H, NUM_SPLITS)。每个 program 处理一个 (batch, q head) 在一段 KV 上的注意力。"""
    pid_bh = tl.program_id(0)
    split = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kvh = h // GROUP

    seqlen = tl.load(SEQLENS + b)
    start = split * CHUNK
    end = tl.minimum(start + CHUNK, seqlen)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)        # [D]
    k_base = K + b * stride_kb + kvh * stride_kh
    v_base = V + b * stride_vb + kvh * stride_vh
    qk_scale = sm_scale * 1.4426950408889634

    # q 只有一行，没法用 tl.dot（M 维至少 16），用广播乘 + 归约代替
    m = tl.full([], float("-inf"), dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        cols = n0 + offs_n
        valid = cols < end
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=valid[:, None], other=0.0)
        s = tl.sum(q[None, :] * k.to(tl.float32), 1) * qk_scale                   # [BLOCK_N]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 0))
        p = tl.math.exp2(s - m_new)
        alpha = tl.math.exp2(m - m_new)
        l = l * alpha + tl.sum(p, 0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m = m_new

    # 空 split（start >= seqlen）：m = -inf、l = 0。写 lse=-inf，combine 时权重 exp(-inf)=0
    empty = l == 0.0
    o = tl.where(empty, 0.0, acc / tl.where(empty, 1.0, l))
    lse = tl.where(empty, float("-inf"), (m + tl.math.log2(l)) * 0.6931471805599453)
    part = pid_bh * NUM_SPLITS + split
    tl.store(O_PART + part * HEAD_DIM + offs_d, o)
    tl.store(LSE_PART + part, lse)


@triton.jit
def decode_combine_kernel(O_PART, LSE_PART, O, LSE, NUM_SPLITS,
                          HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr):
    """grid = (B*H,)。把 NUM_SPLITS 份 (o_s, lse_s) 合并：
         lse = log Σ_s exp(lse_s)，o = Σ_s exp(lse_s - lse) * o_s
    """
    pid_bh = tl.program_id(0)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, HEAD_DIM)
    smask = offs_s < NUM_SPLITS
    lse_s = tl.load(LSE_PART + pid_bh * NUM_SPLITS + offs_s, mask=smask, other=float("-inf"))
    o_s = tl.load(O_PART + (pid_bh * NUM_SPLITS + offs_s)[:, None] * HEAD_DIM + offs_d[None, :],
                  mask=smask[:, None], other=0.0)
    m = tl.max(lse_s, 0)
    w = tl.exp(lse_s - m)                       # [BLOCK_S]
    denom = tl.sum(w, 0)
    o = tl.sum(w[:, None] * o_s, 0) / denom
    tl.store(O + pid_bh * HEAD_DIM + offs_d, o.to(O.dtype.element_ty))
    tl.store(LSE + pid_bh, m + tl.log(denom))


def flash_decode(q, k_cache, v_cache, seqlens, num_splits: int = 8, sm_scale=None, BLOCK_N: int = 64):
    """单 token 解码注意力。

    q:        [B, H, D]          当前步的 query
    k/v_cache:[B, Hkv, Nmax, D]  KV cache（每个序列只有前 seqlens[b] 个位置有效）
    seqlens:  [B] int32
    返回 (o [B, H, D], lse [B, H] fp32)
    """
    B, H, D = q.shape
    _, Hkv, Nmax, _ = k_cache.shape
    assert H % Hkv == 0 and q.stride(-1) == k_cache.stride(-1) == v_cache.stride(-1) == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    # 每个 split 负责 CHUNK 个位置（按 Nmax 划分，向上对齐到 BLOCK_N）
    chunk = triton.cdiv(triton.cdiv(Nmax, num_splits), BLOCK_N) * BLOCK_N
    o_part = torch.empty((B * H, num_splits, D), device=q.device, dtype=torch.float32)
    lse_part = torch.empty((B * H, num_splits), device=q.device, dtype=torch.float32)
    decode_split_kernel[(B * H, num_splits)](
        q, k_cache, v_cache, seqlens, o_part, lse_part, sm_scale,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        H, H // Hkv, num_splits, chunk,
        HEAD_DIM=D, BLOCK_N=BLOCK_N, num_warps=4,
    )
    o = torch.empty_like(q)
    lse = torch.empty((B, H), device=q.device, dtype=torch.float32)
    decode_combine_kernel[(B * H,)](o_part, lse_part, o, lse, num_splits,
                                    HEAD_DIM=D, BLOCK_S=triton.next_power_of_2(num_splits))
    return o, lse


def ref_decode(q, k_cache, v_cache, seqlens):
    B, H, D = q.shape
    g = H // k_cache.shape[1]
    os, lses = [], []
    for b in range(B):
        n = int(seqlens[b])
        k = k_cache[b, :, :n].repeat_interleave(g, 0).float()       # [H, n, D]
        v = v_cache[b, :, :n].repeat_interleave(g, 0).float()
        s = torch.einsum("hd,hnd->hn", q[b].float(), k) / math.sqrt(D)
        os.append(torch.einsum("hn,hnd->hd", torch.softmax(s, -1), v))
        lses.append(torch.logsumexp(s, -1))
    return torch.stack(os).to(q.dtype), torch.stack(lses)


if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.bfloat16
    for (B, H, Hkv, Nmax, D, splits, lens) in [
        (1, 8, 8, 256, 64, 1, [256]),                    # 不切分：退化成普通 online softmax
        (2, 32, 8, 4096, 128, 8, [4096, 1000]),          # GQA 4:1
        (4, 16, 2, 2048, 128, 16, [1, 63, 64, 2047]),     # 很多 split 是空的
        (3, 8, 1, 1000, 64, 5, [1000, 999, 333]),        # MQA，split 数不是 2 的幂
    ]:
        q = torch.randn(B, H, D, device="cuda", dtype=dt)
        kc = torch.randn(B, Hkv, Nmax, D, device="cuda", dtype=dt)
        vc = torch.randn(B, Hkv, Nmax, D, device="cuda", dtype=dt)
        seqlens = torch.tensor(lens, device="cuda", dtype=torch.int32)
        o, lse = flash_decode(q, kc, vc, seqlens, num_splits=splits)
        o_ref, lse_ref = ref_decode(q, kc, vc, seqlens)
        tag = f"B{B} H{H}/{Hkv} Nmax{Nmax} splits{splits} lens{lens}"
        check(f"{tag} o", o, o_ref, atol=1e-2, rtol=1e-2)
        check(f"{tag} lse", lse, lse_ref, atol=1e-3, rtol=1e-3)

    # 性能：B=1 时只有 H=32 个 (b, h) 对，不切分的话只有 32 个 program，132 个 SM 大部分闲着
    B, H, Hkv, N, D = 1, 32, 8, 32768, 128
    q = torch.randn(B, H, D, device="cuda", dtype=dt)
    kc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    vc = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    seqlens = torch.full((B,), N, device="cuda", dtype=torch.int32)
    nbytes = 2 * B * Hkv * N * D * 2       # 至少要把 K、V cache 各读一遍
    rows = []
    for splits in [1, 4, 16, 64]:
        ms = bench(lambda: flash_decode(q, kc, vc, seqlens, num_splits=splits))
        rows.append(dict(impl=f"ours splits={splits}", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    q4 = q[:, :, None, :]
    ms = bench(lambda: F.scaled_dot_product_attention(q4, kc, vc, enable_gqa=True))
    rows.append(dict(impl="torch SDPA", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"decode B={B} H={H}/{Hkv} N={N} D={D} bf16（实测于共享 H100；GBps 以 KV cache 大小计）")
    finish()
