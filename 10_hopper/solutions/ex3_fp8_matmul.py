"""练习 10-3：FP8 (e4m3) GEMM + 行/列缩放（参考答案）"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from common import bench, check, check_equal, finish, gpu_spec, report, tflops

FP8_MAX = 448.0   # torch.finfo(torch.float8_e4m3fn).max


def quantize_rowwise(x: torch.Tensor):
    """x: [R, C] (bf16) -> (x8: [R, C] float8_e4m3fn, scale: [R] float32)，满足 x ≈ x8.float() * scale[:, None]。
    每行的 scale = 该行 |x| 的最大值 / 448，让每行都正好用满 e4m3 的范围。"""
    scale = x.abs().amax(dim=1).float().clamp(min=1e-12) / FP8_MAX
    x8 = (x.float() / scale[:, None]).to(torch.float8_e4m3fn)
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
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        a = a_desc.load([pid_m * BM, ki * BK])       # [BM, BK] e4m3
        w = w_desc.load([pid_n * BN, ki * BK])       # [BN, BK] e4m3（W 按 [N, K] 存）
        acc = tl.dot(a, w.T, acc)                    # fp8 x fp8 -> fp32
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    sw = tl.load(sw_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * sa[:, None] * sw[None, :]
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_desc.dtype))


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
    report(rows, f"{S}^3（{gpu_spec().mma_note()}）")
    finish()
