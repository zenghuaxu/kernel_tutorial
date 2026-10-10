"""练习 08-4：sliding window + GQA 的 FlashAttention 前向（参考答案）"""
import math

import torch
import triton
import triton.language as tl

from common import bench, check, finish, gpu_name, report, tflops


@triton.jit
def attn_fwd_swa_gqa_kernel(
    Q, K, V, O, LSE, NVISIT,
    sm_scale,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn,
    stride_vz, stride_vh, stride_vn,
    stride_oz, stride_oh, stride_om,
    H, N_CTX, GROUP,
    WINDOW: tl.constexpr,        # 0 = 只做 causal；>0 = 每个 query 只看最近 WINDOW 个 key（含自己）
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kvh = h // GROUP                                   # GQA：GROUP 个 q head 共用一个 kv head

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * stride_qz + h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_base = K + b * stride_kz + kvh * stride_kh
    v_base = V + b * stride_vz + kvh * stride_vh
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # m_i 用一个很小的有限值初始化而不是 -inf：sliding window 下某些行在第一个块里可能全被 mask，
    # 若 m_i = -inf，则 m_new = -inf，exp2(-inf - (-inf)) = NaN。有限初值让 p = exp2(-inf - (-1e30)) = 0。
    m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634          # 1/ln2：用 exp2 代替 exp

    # 要访问的 key 范围：[lo, hi)
    #   上界（causal）：本块最后一行是 start_m + BLOCK_M - 1，所以 hi = start_m + BLOCK_M（且不超过 N）
    #   下界（window）：本块第一行 start_m 能看到的最左 key 是 start_m - WINDOW + 1，向下对齐到 BLOCK_N
    hi = tl.minimum(start_m + BLOCK_M, N_CTX)
    if WINDOW > 0:
        lo = tl.maximum(start_m - WINDOW + 1, 0) // BLOCK_N * BLOCK_N
    else:
        lo = 0

    nvisit = 0
    for start_n in range(lo, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(k_base + cols[:, None] * stride_kn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        v = tl.load(v_base + cols[:, None] * stride_vn + offs_d[None, :], mask=cols[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale         # [BLOCK_M, BLOCK_N]，log2 单位

        mask = (offs_m[:, None] >= cols[None, :]) & (cols[None, :] < N_CTX)
        if WINDOW > 0:
            mask = mask & (offs_m[:, None] - cols[None, :] < WINDOW)
        qk = tl.where(mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        nvisit += 1

    acc = acc / l_i[:, None]
    o_ptrs = O + b * stride_oz + h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)
    # LSE（自然对数单位）：m_i、log2(l_i) 都是 log2 单位，乘 ln2 换回来
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    tl.store(LSE + pid_bh * N_CTX + offs_m, lse, mask=offs_m < N_CTX)
    tl.store(NVISIT + pid_bh * tl.num_programs(0) + pid_m, nvisit)


def attention_swa_gqa(q, k, v, window: int = 0, sm_scale: float | None = None,
                      BLOCK_M: int = 128, BLOCK_N: int = 64, return_visits: bool = False):
    """causal attention，可选 sliding window（window>0）和 GQA（k/v 的 head 数可以少于 q）。

    q: [B, H, N, D]，k/v: [B, Hkv, N, D]，H % Hkv == 0。返回 (o, lse)。
    """
    B, H, N, D = q.shape
    Hkv = k.shape[1]
    assert H % Hkv == 0 and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    nvisit = torch.zeros(grid[::-1], device=q.device, dtype=torch.int32)
    attn_fwd_swa_gqa_kernel[grid](
        q, k, v, o, lse, nvisit, sm_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, N, H // Hkv,
        WINDOW=window, HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=8, num_stages=3,
    )
    if return_visits:
        return o, lse, nvisit
    return o, lse


def ref_attention(q, k, v, causal=False, window=0, sm_scale=None):
    """fp32 参考实现（会物化 N×N 的分数矩阵，只能用于小规模）。"""
    B, H, N, D = q.shape
    g = H // k.shape[1]
    k = k.repeat_interleave(g, dim=1).float()
    v = v.repeat_interleave(g, dim=1).float()
    scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    s = (q.float() @ k.transpose(-1, -2)) * scale
    i = torch.arange(N, device=q.device)[:, None]
    j = torch.arange(N, device=q.device)[None, :]
    allowed = torch.ones(N, N, dtype=torch.bool, device=q.device)
    if causal:
        allowed &= j <= i
    if window > 0:
        allowed &= (i - j) < window
    s = s.masked_fill(~allowed, float("-inf"))
    lse = torch.logsumexp(s, dim=-1)
    o = torch.softmax(s, dim=-1) @ v
    return o.to(q.dtype), lse


def expected_visits(N, window, BLOCK_M, BLOCK_N):
    out = []
    for pid_m in range(triton.cdiv(N, BLOCK_M)):
        start_m = pid_m * BLOCK_M
        hi = min(start_m + BLOCK_M, N)
        lo = max(start_m - window + 1, 0) // BLOCK_N * BLOCK_N if window > 0 else 0
        out.append(len(range(lo, hi, BLOCK_N)))
    return torch.tensor(out, dtype=torch.int32)


if __name__ == "__main__":
    torch.manual_seed(0)
    dt = torch.bfloat16
    for (B, H, Hkv, N, D, W) in [
        (1, 4, 4, 256, 64, 0),        # 纯 causal、MHA
        (2, 8, 2, 1000, 64, 128),     # GQA 4:1，window 小于 BLOCK_M
        (1, 8, 1, 777, 128, 300),     # MQA，window 不对齐
        (2, 4, 2, 1024, 128, 2048),   # window > N，应该等价于纯 causal
        (1, 4, 1, 512, 64, 1),        # window=1：只看自己，o == v
    ]:
        q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
        k = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
        v = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
        o, lse, nv = attention_swa_gqa(q, k, v, window=W, return_visits=True)
        o_ref, lse_ref = ref_attention(q, k, v, causal=True, window=W)
        tag = f"B{B} H{H}/{Hkv} N{N} D{D} W{W}"
        check(f"{tag} o", o, o_ref, atol=1e-2, rtol=1e-2)
        check(f"{tag} lse", lse, lse_ref, atol=2e-3, rtol=2e-3)
        exp = expected_visits(N, W, 128, 64).to(nv.device)
        check(f"{tag} 访问的 K 块数", nv, exp.expand_as(nv), atol=0, rtol=0)

    # 性能：N=8192 的 causal vs window=1024（sliding window 的计算量 ∝ N*W，而不是 N^2/2）
    B, H, Hkv, N, D = 1, 16, 4, 8192, 128
    q = torch.randn(B, H, N, D, device="cuda", dtype=dt)
    k = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    v = torch.randn(B, Hkv, N, D, device="cuda", dtype=dt)
    rows = []
    for W in [0, 4096, 1024, 256]:
        ms = bench(lambda: attention_swa_gqa(q, k, v, window=W))
        # 有效 FLOPs：每个 (i, j) 对 4*D（QK^T 和 PV 各 2*D）
        pairs = sum(min(i + 1, W) if W > 0 else i + 1 for i in range(N))
        rows.append(dict(window=W if W else "causal", ms=ms, TFLOPs=tflops(4 * D * pairs * B * H, ms)))
    report(rows, f"B={B} H={H}/{Hkv} N={N} D={D} bf16（实测于 {gpu_name()}，有噪声）")
    finish()
