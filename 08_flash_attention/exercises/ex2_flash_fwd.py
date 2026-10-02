"""练习 08-2：FlashAttention 前向（non-causal）

目标：补全 attn_fwd_kernel 的后半部分。已经写好的：program 编号、offsets、Q 块的加载、K/V 的基址。
你要写：
  1. online softmax 状态的初始化：m_i [BLOCK_M]、l_i [BLOCK_M]、acc [BLOCK_M, HEAD_DIM]（全部 fp32）
  2. 沿 N 方向的 K/V 循环（讲义 8.3 的伪代码）：
       load K、V 块（[BLOCK_N, HEAD_DIM]，越界行 mask 成 0）
       qk = tl.dot(q, tl.trans(k)) * qk_scale，越界的列置 -inf
       m_new、p = exp2(qk - m_new)、alpha = exp2(m_i - m_new)
       l_i、acc 重缩放后累加；acc 的累加用 tl.dot(p.to(v.dtype), v)
  3. 收尾：acc / l_i，store 到 O；LSE（自然对数单位！）store 到 LSE + pid_bh * N_CTX + offs_m

提示：
  - qk_scale = sm_scale * log2(e)，这样循环里用 tl.math.exp2；最终 LSE = (m_i + log2(l_i)) * ln(2)
  - K 块地址：k_base + cols[:, None] * stride_kn + offs_d[None, :]（最后一维 stride 为 1）
  - 测试里 N=1000、513 不是块大小的倍数——越界的 key 必须置 -inf，越界的 query 行不能写出去

做完之后想一想：
  - 为什么 p 要先 .to(v.dtype) 再 tl.dot？改成 fp32 试试速度和精度
  - 打印的表格里 ours / SDPA / naive 的 TFLOPs 各是多少？naive 慢在哪（单元 00 的 roofline）？

运行：python 08_flash_attention/exercises/ex2_flash_fwd.py
"""
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

    qk_scale = sm_scale * 1.4426950408889634                    # 把 1/ln2 折进 scale，后面用 exp2

    # TODO 1：初始化 m_i、l_i、acc

    # TODO 2：for start_n in range(0, N_CTX, BLOCK_N): ...

    # TODO 3：归一化，写 O（offs_m < N_CTX 的行）和 LSE（自然对数）
    pass


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
