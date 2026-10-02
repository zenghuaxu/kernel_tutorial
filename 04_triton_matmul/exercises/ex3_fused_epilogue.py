"""练习 04-3：Linear + bias + 激活 的 epilogue 融合

背景：MLP 里常见 y = act(x @ W^T + b)。不融合时 matmul 写出 [M, N] 的结果，激活 kernel 再读一遍、写一遍。
而 matmul 结束时累加器 acc 本来就在寄存器里——在写回之前顺手加 bias、过激活，就省掉了那次 HBM 往返。
这就是 "epilogue fusion"，cuBLASLt / CUTLASS / Triton 都靠它省时间。

目标（主循环已经写好了，你只写两处）：
  1. wrapper：W 是 nn.Linear 的布局 [N, K]，而 kernel 想要的是 [K, N] 的 B 矩阵。
     **不要** 调 w.t().contiguous()（会多一次拷贝），而是传对 stride_wk / stride_wn，让 kernel 直接读 W^T 视图
  2. kernel 的 epilogue：
     - HAS_BIAS 时加 bias（[N]，按列广播）
     - ACTIVATION == "silu"：x * sigmoid(x)；== "gelu_tanh"：调用写好的 _gelu_tanh；== "none"：不变
     - 转成输出 dtype，带 mask 写回
  bias 和激活都在 fp32 累加器上算。

提示：
  - HAS_BIAS / ACTIVATION 是 tl.constexpr，`if ACTIVATION == "silu":` 在编译期就决定了，不走的分支不会生成代码
  - HAS_BIAS=False 时 bias_ptr 传的是个占位 tensor，别去读它

做完之后想一想：
  - 表里三行：融合版 vs 同一个 triton matmul + 单独的 torch gelu，差多少？和省下的字节数（[M,N] bf16 读一次写一次）对得上吗？
  - cuBLAS + 单独 gelu 仍然最快——说明 Triton 主循环本身比 cuBLAS 慢。融合不能替代主循环优化，两者是叠加的。
  - 用 k = linear_act_kernel[grid](...) 拿到 handle 看 k.n_regs：gelu_tanh + 128x256 tile 时寄存器用到了多少？
    （H100 每线程上限 255，再多就 spill 到 local memory，速度暴跌）

运行：python 04_triton_matmul/exercises/ex3_fused_epilogue.py
"""
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
    # TODO: bias（HAS_BIAS 时）、激活（按 ACTIVATION）、cast、带 mask 写回 y
    pass


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
        None, None,   # TODO: stride_wk, stride_wn —— 把 W[N, K] 当作 [K, N] 读
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
