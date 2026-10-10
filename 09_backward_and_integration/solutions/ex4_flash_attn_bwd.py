"""练习 09-4（挑战）：FlashAttention 反向（non-causal）（参考答案）"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gpu_name, report, tflops

LOG2E = 1.4426950408889634


# ----------------------------------------------------------------------------
# 前向（已给出，就是单元 08 练习 2 的 kernel）：输出 O 和 LSE（自然对数）
# ----------------------------------------------------------------------------
@triton.jit
def attn_fwd_kernel(Q, K, V, O, LSE, sm_scale, stride_z, stride_h, stride_n, H, N_CTX,
                    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * stride_z + (pid_bh % H) * stride_h       # q/k/v/o 形状、stride 相同
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + base + offs_m[:, None] * stride_n + offs_d[None, :], mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634
    for start_n in range(0, N_CTX, BLOCK_N):
        cols = start_n + offs_n
        kv_off = base + cols[:, None] * stride_n + offs_d[None, :]
        k = tl.load(K + kv_off, mask=cols[:, None] < N_CTX, other=0.0)
        v = tl.load(V + kv_off, mask=cols[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(cols[None, :] < N_CTX, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_n + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=offs_m[:, None] < N_CTX)
    tl.store(LSE + pid_bh * N_CTX + offs_m, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, mask=offs_m < N_CTX)


# ----------------------------------------------------------------------------
# 反向
#   记 S = scale·QKᵀ，P = softmax(S)，O = PV，上游梯度 dO。
#   dV = Pᵀ dO
#   dP = dO Vᵀ
#   dS = P ⊙ (dP − Δ)，  Δ_i = Σ_j P_ij dP_ij = Σ_d dO_id O_id   （softmax 的反向）
#   dQ = scale · dS K
#   dK = scale · dSᵀ Q
#   P 不存，用前向保存的 LSE 重算：P_ij = exp(S_ij − LSE_i)
# ----------------------------------------------------------------------------
@triton.jit
def attn_bwd_preprocess_kernel(O, DO, DELTA, stride_z, stride_h, stride_n, H, N_CTX,
                               HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    """Δ = rowsum(dO ⊙ O)，形状 [B*H, N]。"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * stride_z + (pid_bh % H) * stride_h
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    off = base + offs_m[:, None] * stride_n + offs_d[None, :]
    mask = offs_m[:, None] < N_CTX
    o = tl.load(O + off, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(DO + off, mask=mask, other=0.0).to(tl.float32)
    tl.store(DELTA + pid_bh * N_CTX + offs_m, tl.sum(o * do, 1), mask=offs_m < N_CTX)


@triton.jit
def attn_bwd_dkdv_kernel(Q, K, V, DO, LSE, DELTA, DK, DV, sm_scale,
                         stride_z, stride_h, stride_n, H, N_CTX,
                         HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid = (N/BLOCK_N, B*H)：每个 program 固定一个 K/V 块，沿 Q 方向扫描，累加 dK、dV。

    在这个 kernel 里我们直接算转置的量（Sᵀ、Pᵀ、dPᵀ、dSᵀ，形状 [BLOCK_N, BLOCK_M]），
    这样 dV += Pᵀ dO、dK += dSᵀ Q 都是"行数 = BLOCK_N"的矩阵乘，结果直接累加进 [BLOCK_N, D] 的寄存器块。
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * stride_z + (pid_bh % H) * stride_h
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m0 = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    kv_off = base + offs_n[:, None] * stride_n + offs_d[None, :]
    k = tl.load(K + kv_off, mask=offs_n[:, None] < N_CTX, other=0.0)
    v = tl.load(V + kv_off, mask=offs_n[:, None] < N_CTX, other=0.0)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m = start_m + offs_m0
        q_off = base + offs_m[:, None] * stride_n + offs_d[None, :]
        q = tl.load(Q + q_off, mask=offs_m[:, None] < N_CTX, other=0.0)
        do = tl.load(DO + q_off, mask=offs_m[:, None] < N_CTX, other=0.0)
        lse = tl.load(LSE + pid_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
        delta = tl.load(DELTA + pid_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

        qkT = tl.dot(k, tl.trans(q)) * qk_scale                        # Sᵀ（log2 单位）[BN, BM]
        pT = tl.math.exp2(qkT - lse[None, :] * 1.4426950408889634)      # Pᵀ = exp(S − LSE)
        pT = tl.where(offs_m[None, :] < N_CTX, pT, 0.0)                 # 越界的 query 行不贡献
        dv += tl.dot(pT.to(do.dtype), do)                               # dV += Pᵀ dO
        dpT = tl.dot(v, tl.trans(do))                                   # dPᵀ = V dOᵀ
        dsT = pT * (dpT - delta[None, :])                               # dSᵀ
        dk += tl.dot(dsT.to(q.dtype), q)                                # dK += dSᵀ Q（scale 最后乘）

    tl.store(DK + kv_off, (dk * sm_scale).to(DK.dtype.element_ty), mask=offs_n[:, None] < N_CTX)
    tl.store(DV + kv_off, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < N_CTX)


@triton.jit
def attn_bwd_dq_kernel(Q, K, V, DO, LSE, DELTA, DQ, sm_scale,
                       stride_z, stride_h, stride_n, H, N_CTX,
                       HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """grid = (N/BLOCK_M, B*H)：每个 program 固定一个 Q 块，沿 K 方向扫描，累加 dQ。"""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    base = (pid_bh // H) * stride_z + (pid_bh % H) * stride_h
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_off = base + offs_m[:, None] * stride_n + offs_d[None, :]
    q = tl.load(Q + q_off, mask=offs_m[:, None] < N_CTX, other=0.0)
    do = tl.load(DO + q_off, mask=offs_m[:, None] < N_CTX, other=0.0)
    lse = tl.load(LSE + pid_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    delta = tl.load(DELTA + pid_bh * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + offs_n0
        kv_off = base + offs_n[:, None] * stride_n + offs_d[None, :]
        k = tl.load(K + kv_off, mask=offs_n[:, None] < N_CTX, other=0.0)
        v = tl.load(V + kv_off, mask=offs_n[:, None] < N_CTX, other=0.0)
        qk = tl.dot(q, tl.trans(k)) * qk_scale                          # [BM, BN]
        p = tl.math.exp2(qk - lse[:, None] * 1.4426950408889634)
        p = tl.where(offs_n[None, :] < N_CTX, p, 0.0)                    # 越界的 key 不贡献
        dp = tl.dot(do, tl.trans(v))                                    # dP = dO Vᵀ
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)                                 # dQ += dS K

    tl.store(DQ + q_off, (dq * sm_scale).to(DQ.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


class FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        assert q.shape == k.shape == v.shape
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        B, H, N, D = q.shape
        sm_scale = 1.0 / math.sqrt(D)
        o = torch.empty_like(q)
        lse = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        attn_fwd_kernel[(triton.cdiv(N, 128), B * H)](
            q, k, v, o, lse, sm_scale, q.stride(0), q.stride(1), q.stride(2), H, N,
            HEAD_DIM=D, BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.sm_scale = sm_scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        do = do.contiguous()
        B, H, N, D = q.shape
        strides = (q.stride(0), q.stride(1), q.stride(2))
        BLOCK_M, BLOCK_N = 64, 64
        delta = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        attn_bwd_preprocess_kernel[(triton.cdiv(N, BLOCK_M), B * H)](
            o, do, delta, *strides, H, N, HEAD_DIM=D, BLOCK_M=BLOCK_M)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        attn_bwd_dkdv_kernel[(triton.cdiv(N, BLOCK_N), B * H)](
            q, k, v, do, lse, delta, dk, dv, ctx.sm_scale, *strides, H, N,
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
        attn_bwd_dq_kernel[(triton.cdiv(N, BLOCK_M), B * H)](
            q, k, v, do, lse, delta, dq, ctx.sm_scale, *strides, H, N,
            HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
        return dq, dk, dv


def flash_attn(q, k, v):
    return FlashAttnFunction.apply(q, k, v)


def ref_attn(q, k, v):
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.softmax(s, -1) @ v


def fwd_bwd(fn, q, k, v, do):
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    o = fn(q, k, v)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


if __name__ == "__main__":
    torch.manual_seed(0)
    for (B, H, N, D) in [(1, 1, 64, 64), (2, 4, 1000, 64), (1, 4, 777, 128), (2, 2, 1024, 128)]:
        q, k, v, do = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
        ours = fwd_bwd(flash_attn, q, k, v, do)
        ref = fwd_bwd(ref_attn, q.float(), k.float(), v.float(), do.float())            # fp32 "真值"
        sdpa = fwd_bwd(F.scaled_dot_product_attention, q, k, v, do)                       # bf16 基线
        # 容差：不超过 SDPA（同为 bf16 实现）对 fp32 真值误差的 2 倍 + 一点余量
        for name, a, s, r in zip(["o", "dq", "dk", "dv"], ours, sdpa, ref):
            tol = 2 * (s.float() - r).abs().max().item() + 1e-3
            check(f"B{B} H{H} N{N} D{D} {name}", a, r.to(a.dtype), atol=tol, rtol=0)

    B, H, N, D = 2, 16, 4096, 128
    q, k, v, do = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(4))
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    flops = 4 * B * H * N * N * D * (1 + 2.5)        # 前向 4·N²·D，反向约 2.5 倍（5 个 matmul）
    rows = []
    for name, fn in [("ours (triton)", flash_attn), ("torch SDPA", F.scaled_dot_product_attention)]:
        ms = bench(lambda: torch.autograd.grad(fn(q, k, v), (q, k, v), do))
        rows.append(dict(impl=name, ms_fwd_bwd=ms, TFLOPs=tflops(flops, ms)))
    report(rows, f"attention 前向+反向 B={B} H={H} N={N} D={D} bf16（实测于 {gpu_name()}）")
    finish()
