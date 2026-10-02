"""练习 10-3：FP8 (e4m3) GEMM + 行/列缩放

H100 的 Tensor Core 跑 fp8 的峰值是 bf16 的 2 倍，DeepSeek-V3 等模型已经用 fp8 训练/推理。
e4m3 只有 3 位尾数、最大值 448，所以必须配合**缩放**使用：
    x ≈ x8.float() * scale
本练习用最常见的组合：激活 A 按行（per-token）缩放，权重 W 按输出通道（W 的每一行，per-channel）缩放：
    y[m, n] = sum_k A[m,k] W[n,k] ≈ sa[m] * sw[n] * sum_k A8[m,k] W8[n,k]
所以 kernel 里只需要在 fp32 累加器上乘一次 sa[:, None] * sw[None, :]——又一个 epilogue 融合。

目标：
  1. quantize_rowwise(x)：每行 scale = 该行 |x| 的最大值 / 448；x8 = (x / scale).to(float8_e4m3fn)
     （在 fp32 里做除法；全零行别除以 0，scale 下限 clamp 到 1e-12）
  2. fp8_matmul_kernel：
     - K 循环：a_desc.load 得到 [BM, BK]；W 按 [N, K] 存，w_desc.load([pid_n * BN, ki * BK]) 得到 [BN, BK]，
       乘的时候用 w.T。（Hopper 的 fp8 wgmma 要求两个操作数都是 "K 连续"，所以权重按 [N, K] 存最自然）
     - epilogue：load sa、sw（带 mask），乘到 acc 上，c_desc.store

提示：
  - fp8 的 wgmma 一次吃 K=32，BK 取 128 正好
  - tl.dot(a, w.T, acc)：.T 只是改变 shared memory 的读取方式，不会真的搬数据

做完之后想一想：
  - 测试打印的"相对 bf16 原始矩阵乘的误差"约 3.7%。这个误差主要来自哪里？（3 位尾数的相对舍入误差 ≈ 2^-4 / √3 …）
    如果 A 的某一行里有一个特别大的离群值，per-row scale 会怎样？per-tensor scale 呢？（这就是 block-wise 缩放的动机）
  - 你的 kernel 和 torch._scaled_mm 差多少？离 fp8 峰值 1979 还差多少？差在哪？（提示：单元 10 讲义 10.4）

运行：python 10_hopper/exercises/ex3_fp8_matmul.py
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from common import bench, check, check_equal, finish, report, tflops

FP8_MAX = 448.0   # torch.finfo(torch.float8_e4m3fn).max


def quantize_rowwise(x: torch.Tensor):
    """x: [R, C] (bf16) -> (x8: [R, C] float8_e4m3fn, scale: [R] float32)，满足 x ≈ x8.float() * scale[:, None]。
    每行的 scale = 该行 |x| 的最大值 / 448，让每行都正好用满 e4m3 的范围。"""
    # TODO
    x8, scale = None, None
    return x8, scale


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (tile_id // num_pid_in_group) * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % num_pid_in_group) % group_size_m
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def fp8_matmul_kernel(a_desc, w_desc, c_desc, sa_ptr, sw_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tile_coords(tl.program_id(0), tl.cdiv(M, BM), tl.cdiv(N, BN), 8)
    # TODO:
    #   1. fp32 累加器；K 循环：a_desc.load、w_desc.load、tl.dot(a, w.T, acc)
    #   2. epilogue：sa[offs_m]、sw[offs_n]（带 mask）乘到 acc 上
    #   3. c_desc.store
    pass


def fp8_linear(a8, sa, w8, sw):
    """y = (a8 * sa[:, None]) @ (w8 * sw[:, None])^T，输出 bf16。a8: [M, K]，w8: [N, K]。"""
    M, K = a8.shape
    N = w8.shape[0]
    BM, BN, BK = 128, 256, 128
    c = torch.empty((M, N), device=a8.device, dtype=torch.bfloat16)
    a_desc = TensorDescriptor.from_tensor(a8, [BM, BK])
    w_desc = TensorDescriptor.from_tensor(w8, [BN, BK])
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN])
    fp8_matmul_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a_desc, w_desc, c_desc, sa, sw, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=8, num_stages=3)
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    # ---- 量化函数
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) * torch.logspace(-2, 2, 64, device="cuda")[:, None]
    x8, s = quantize_rowwise(x)
    check_equal("quantize: x8 dtype", x8.dtype, torch.float8_e4m3fn)
    check("quantize: scale = 行 amax / 448", s, x.abs().amax(1).float() / 448.0, atol=0, rtol=1e-6)
    check("quantize: 每行最大值映射到 ±448", x8.float().abs().amax(1), torch.full((64,), 448.0, device="cuda"), atol=0, rtol=0)
    rel = ((x8.float() * s[:, None] - x.float()).norm() / x.float().norm()).item()
    check_equal("quantize: 反量化相对误差 < 4%", rel < 0.04, True)

    # ---- GEMM
    for M, N, K in [(256, 512, 1024), (1000, 776, 528), (4096, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        a8, sa = quantize_rowwise(a)
        w8, sw = quantize_rowwise(w)
        y = fp8_linear(a8, sa, w8, sw)
        ref_deq = ((a8.float() * sa[:, None]) @ (w8.float() * sw[:, None]).T).to(torch.bfloat16)
        check(f"{M}x{N}x{K} vs 反量化参考", y, ref_deq, atol=2e-2, rtol=2e-2)
        y_bf16 = a.float() @ w.float().T
        rel = ((y.float() - y_bf16).norm() / y_bf16.norm()).item()
        print(f"    相对 bf16 原始矩阵乘的误差: {rel:.3%}")
        check_equal(f"{M}x{N}x{K} 量化误差 < 6%", rel < 0.06, True)

    # ---- 性能
    S = 8192
    a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
    a8, sa = quantize_rowwise(a)
    w8, sw = quantize_rowwise(w)
    f = 2 * S**3
    rows = [
        dict(impl="你的 fp8 kernel", TFLOPS=tflops(f, bench(lambda: fp8_linear(a8, sa, w8, sw)))),
        dict(impl="torch._scaled_mm (cuBLASLt fp8)", TFLOPS=tflops(f, bench(lambda: torch._scaled_mm(
            a8, w8.T, scale_a=sa[:, None], scale_b=sw[None, :], out_dtype=torch.bfloat16)))),
        dict(impl="cuBLAS bf16", TFLOPS=tflops(f, bench(lambda: a @ w.T))),
    ]
    report(rows, f"{S}^3（H100 dense 峰值：fp8 ≈ 1979，bf16 ≈ 989 TFLOPS）")
    finish()
