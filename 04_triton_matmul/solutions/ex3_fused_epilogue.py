"""练习 04-3：Linear + bias + 激活 的 epilogue 融合（参考答案）"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, report, tflops


@triton.jit
def _gelu_tanh(x):
    # gelu(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))，tanh(z) = 2·sigmoid(2z) − 1
    z = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))


@triton.jit
def linear_act_kernel(
    x_ptr, w_ptr, bias_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,      # 注意：这是把 W[N, K] 当成 [K, N] 矩阵来看时的 stride
    stride_ym, stride_yn,
    HAS_BIAS: tl.constexpr, ACTIVATION: tl.constexpr,   # ACTIVATION: "none" / "silu" / "gelu_tanh"
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # ---------------- epilogue：累加器还在寄存器里，顺手做完 bias + 激活 + cast ----------------
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    if ACTIVATION == "silu":
        acc = acc * tl.sigmoid(acc)
    elif ACTIVATION == "gelu_tanh":
        acc = _gelu_tanh(acc)
    y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(y_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def linear_act(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, activation: str = "none"):
    """等价于 act(F.linear(x, weight, bias))。x: [M, K]，weight: [N, K]（nn.Linear 的布局），bias: [N]。"""
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BM, BN, BK = 128, 256, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    linear_act_kernel[grid](
        x, weight, bias if bias is not None else x, y, M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),   # W^T[k, n] = W[n, k]
        y.stride(0), y.stride(1),
        HAS_BIAS=bias is not None, ACTIVATION=activation,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8, num_warps=8, num_stages=3,
    )
    return y


def ref(x, w, b, act):
    y = F.linear(x.float(), w.float(), None if b is None else b.float())
    y = {"none": y, "silu": F.silu(y), "gelu_tanh": F.gelu(y, approximate="tanh")}[act]
    return y.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    tol = dict(atol=5e-2, rtol=2e-2)
    for act in ["none", "silu", "gelu_tanh"]:
        for M, N, K, has_bias in [(256, 512, 384, True), (77, 300, 129, True), (512, 1024, 256, False)]:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
            b = torch.randn(N, device="cuda", dtype=torch.bfloat16) if has_bias else None
            check(f"{act:9s} {M}x{N}x{K} bias={has_bias}", linear_act(x, w, b, act), ref(x, w, b, act), **tol)

    # 性能：一个 MLP 上投影的大小（8192 token，4096 -> 11008）
    M, K, N = 8192, 4096, 11008
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    f = 2 * M * N * K
    rows = []
    for name, fn in [
        ("triton 融合 epilogue", lambda: linear_act(x, w, b, "gelu_tanh")),
        ("triton matmul+bias，torch gelu", lambda: F.gelu(linear_act(x, w, b, "none"), approximate="tanh")),
        ("cuBLAS linear，torch gelu", lambda: F.gelu(F.linear(x, w, b), approximate="tanh")),
    ]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPS=tflops(f, ms)))
    report(rows, f"{M}x{K} @ W[{N},{K}]^T + bias → gelu")
    finish()
