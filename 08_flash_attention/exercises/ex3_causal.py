"""练习 08-3：causal FlashAttention + 块跳过

目标：causal attention（query i 只看 key j <= i），并且**完全在上三角的 K 块根本不访问**。

已经写好：
  - _attn_inner：处理 key 范围 [lo, hi) 的循环（和练习 2 一样），返回 (acc, l_i, m_i, 访问的块数)
  - 主 kernel 的准备工作、收尾、以及把访问块数写到 NVISIT 的代码
你要写：
  1. _attn_inner 里 DIAG=True 时的逐元素 causal mask（同时别忘了越界的 key：cols < N_CTX）
  2. 主 kernel 里两次调用 _attn_inner：
       段 1：[0, start_m)                        —— DIAG=False，不 mask
       段 2：[start_m, min(start_m + BLOCK_M, N_CTX)) —— DIAG=True
     两次调用分别返回块数 n1、n2

提示：
  - 讲义 8.4 的图。DIAG 是 tl.constexpr，`if DIAG:` 会在编译期展开，两次调用得到两份特化代码
  - 段 1 为什么不需要越界 mask？（start_m <= N_CTX 且 start_m 是 BLOCK_N 的倍数）
  - 测试会核对：Q 块 i 访问的 K 块数必须 == ceil(min((i+1)*128, N) / 64)

做完之后想一想：
  - 打印的 causal TFLOPs 是按一半 FLOPs 算的。和练习 2 的 non-causal 比，时间是一半吗？差在哪？
  - 把 BLOCK_M 改成 64（BLOCK_N 仍 64），对角线块的"浪费"比例和速度怎么变？

运行：python 08_flash_attention/exercises/ex3_causal.py
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, report, tflops


@triton.jit
def _attn_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn,
                lo, hi, offs_m, offs_n, offs_d, N_CTX, qk_scale,
                DIAG: tl.constexpr, BLOCK_N: tl.constexpr):
    """处理 key 范围 [lo, hi)。DIAG=True 表示这些块跨过对角线，需要逐元素 causal mask。"""
    nvisit = 0
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if DIAG:
            # TODO 1：逐元素 causal mask（以及 cols < N_CTX），不允许的位置置 -inf
            pass
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        nvisit += 1
    return acc, l_i, m_i, nvisit


@triton.jit
def attn_fwd_causal_kernel(
    Q, K, V, O, LSE, NVISIT,
    sm_scale,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tl.static_assert(BLOCK_M % BLOCK_N == 0)   # 保证 start_m 是 BLOCK_N 的倍数，两段之间没有缝
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    k_base = K + b * stride_kz + h * stride_kh
    v_base = V + b * stride_vz + h * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    # TODO 2：两次调用 _attn_inner（段 1 不 mask、段 2 对角线块），得到 n1、n2
    #   acc, l_i, m_i, n1 = _attn_inner(acc, l_i, m_i, q, k_base, v_base, stride_kn, stride_vn,
    #                                   lo, hi, offs_m, offs_n, offs_d, N_CTX, qk_scale,
    #                                   DIAG=..., BLOCK_N=BLOCK_N)
    n1 = 0
    n2 = 0

    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_oz + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    tl.store(LSE + pid_bh * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    tl.store(NVISIT + pid_bh * tl.num_programs(0) + pid_m, n1 + n2)


def flash_attention_causal(q, k, v, sm_scale=None, BLOCK_M=128, BLOCK_N=64, return_visits=False):
    B, H, N, D = q.shape
    assert k.shape == v.shape == q.shape and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    nvisit = torch.zeros(grid[::-1], device=q.device, dtype=torch.int32)
    attn_fwd_causal_kernel[grid](
        q, k, v, o, lse, nvisit, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, N, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )
    return (o, lse, nvisit) if return_visits else (o, lse)


def ref_attention_causal(q, k, v):
    N = q.shape[-2]
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    s = s.masked_fill(torch.ones(N, N, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype), torch.logsumexp(s, -1)


if __name__ == "__main__":
    torch.manual_seed(0)
    for (B, H, N, D) in [(1, 1, 128, 64), (2, 4, 1000, 64), (1, 8, 513, 128), (2, 2, 2048, 128), (1, 2, 40, 64)]:
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        o, lse, nv = flash_attention_causal(q, k, v, return_visits=True)
        o_ref, lse_ref = ref_attention_causal(q, k, v)
        check(f"B{B} H{H} N{N} D{D} o", o, o_ref, atol=1e-2, rtol=1e-2)
        check(f"B{B} H{H} N{N} D{D} lse", lse, lse_ref, atol=2e-3, rtol=2e-3)
        # 块跳过：Q 块 i 只应访问 ceil(min((i+1)*128, N) / 64) 个 K 块
        exp = torch.tensor([triton.cdiv(min((i + 1) * 128, N), 64) for i in range(triton.cdiv(N, 128))],
                           dtype=torch.int32, device="cuda")
        check(f"B{B} H{H} N{N} D{D} 访问的 K 块数", nv, exp.expand_as(nv), atol=0, rtol=0)

    B, H, N, D = 2, 16, 4096, 128
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flops = 4 * B * H * N * N * D / 2      # causal 只算一半
    rows = []
    for name, fn in [
        ("ours causal", lambda: flash_attention_causal(q, k, v)),
        ("torch SDPA causal", lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)),
    ]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPs=tflops(flops, ms)))
    report(rows, f"causal B={B} H={H} N={N} D={D} bf16（实测于共享 H100，有噪声）")
    finish()
