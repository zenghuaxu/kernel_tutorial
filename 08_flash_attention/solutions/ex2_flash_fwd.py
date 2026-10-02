"""练习 08-2：FlashAttention 前向（non-causal）（参考答案）"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, report, tflops


@triton.jit
def attn_fwd_kernel(
    Q, K, V, O, LSE,
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
    # ---- 我是谁：grid = (Q 块数, B*H) ----
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # ---- Q 块常驻寄存器，整个循环只读一次 ----
    q_ptrs = Q + b * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    k_base = K + b * stride_kz + h * stride_kh
    v_base = V + b * stride_vz + h * stride_vh

    # ---- online softmax 状态 ----
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)   # 行最大值（log2 单位）
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                 # 行和 Σ exp2(s - m)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)       # 未归一化的输出 Σ p·v
    qk_scale = sm_scale * 1.4426950408889634                    # 把 1/ln2 折进 scale，后面用 exp2

    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale                     # [BLOCK_M, BLOCK_N]
        qk = tl.where(cols[None, :] < N_CTX, qk, float("-inf"))     # 越界的 key 不参与 softmax

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)                           # 旧状态的缩放因子
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- 收尾：归一化、写 O 和 LSE ----
    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_oz + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453         # 换回自然对数
    tl.store(LSE + pid_bh * N_CTX + offs_m, lse, mask=offs_m < N_CTX)


def flash_attention(q, k, v, sm_scale: float | None = None, BLOCK_M: int = 128, BLOCK_N: int = 64):
    """q/k/v: [B, H, N, D]（最后一维连续）。返回 (o [B,H,N,D], lse [B,H,N] fp32)。"""
    B, H, N, D = q.shape
    assert k.shape == v.shape == q.shape and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    attn_fwd_kernel[grid](
        q, k, v, o, lse, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )
    return o, lse


def ref_attention(q, k, v, sm_scale=None):
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    return (torch.softmax(s, -1) @ v.float()).to(q.dtype), torch.logsumexp(s, -1)


def naive_attention(q, k, v):
    """PyTorch 直接写：会把 [B,H,N,N] 的分数矩阵写进 HBM。"""
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


if __name__ == "__main__":
    torch.manual_seed(0)
    for (B, H, N, D) in [(1, 1, 128, 64), (2, 4, 1000, 64), (1, 8, 513, 128), (2, 2, 2048, 128)]:
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        o, lse = flash_attention(q, k, v)
        o_ref, lse_ref = ref_attention(q, k, v)
        check(f"B{B} H{H} N{N} D{D} o", o, o_ref, atol=1e-2, rtol=1e-2)
        check(f"B{B} H{H} N{N} D{D} lse", lse, lse_ref, atol=2e-3, rtol=2e-3)
    # 大数值输入：不做减 max 的 softmax 会溢出
    q, k, v = (torch.randn(1, 2, 256, 64, device="cuda", dtype=torch.bfloat16) * 8 for _ in range(3))
    o, lse = flash_attention(q, k, v)
    o_ref, lse_ref = ref_attention(q, k, v)
    check("大数值输入 o", o, o_ref, atol=5e-2, rtol=2e-2)
    check("大数值输入 lse", lse, lse_ref, atol=1e-2, rtol=2e-3)

    B, H, N, D = 1, 16, 4096, 128   # naive 会物化 ~2GB 的 S/P，共享卡上别再加大
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flops = 4 * B * H * N * N * D
    rows = []
    for name, fn in [
        ("ours (triton)", lambda: flash_attention(q, k, v)),
        ("torch SDPA", lambda: F.scaled_dot_product_attention(q, k, v)),
        ("naive torch", lambda: naive_attention(q, k, v)),
    ]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPs=tflops(flops, ms)))
    report(rows, f"non-causal B={B} H={H} N={N} D={D} bf16（实测于共享 H100，有噪声）")
    finish()
