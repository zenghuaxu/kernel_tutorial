"""练习 04-4：split-K（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, report, tflops


@triton.jit
def matmul_splitk_kernel(
    a_ptr, b_ptr, c_ptr,          # c 是 fp32、预先清零
    M, N, K, K_PER_SPLIT,         # K_PER_SPLIT 是 BLOCK_K 的整数倍
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    pid_k = tl.program_id(1)                       # 我负责 K 的第几段
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_tile // num_pid_n
    pid_n = pid_tile % num_pid_n

    k_start = pid_k * K_PER_SPLIT
    k_end = tl.minimum(k_start + K_PER_SPLIT, K)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        k_ok = (k + tl.arange(0, BLOCK_K)) < k_end
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_ok[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_ok[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 多个 program 写同一个 tile → 原子加。fp32 原子加的顺序不确定，所以结果每次可能差最后几位
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_splitk(a: torch.Tensor, b: torch.Tensor, split_k: int) -> torch.Tensor:
    """返回 fp32 的 A @ B。K 维被切成 split_k 段，由不同 program 并行计算再原子加起来。"""
    M, K = a.shape
    N = b.shape[1]
    BM, BN, BK = 64, 64, 64
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    k_per_split = triton.cdiv(triton.cdiv(K, split_k), BK) * BK
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), triton.cdiv(K, k_per_split))
    matmul_splitk_kernel[grid](
        a, b, c, M, N, K, k_per_split,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=4, num_stages=4,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    tol = dict(atol=2e-2, rtol=1e-3)   # 输出是 fp32：只有累加顺序不同带来的误差
    for M, N, K in [(64, 64, 4096), (100, 70, 10000), (256, 256, 20000), (1, 5, 333)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        ref = a.float() @ b.float()
        for s in [1, 3, 8, 32]:
            check(f"{M}x{N}x{K} split_k={s}", matmul_splitk(a, b, s), ref, **tol)

    # 性能：skinny GEMM——输出只有 256x256（16 个 64x64 tile），K 很长
    M, N, K = 256, 256, 65536
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    f = 2 * M * N * K
    rows = []
    for s in [1, 2, 8, 32, 64]:
        ms = bench(lambda: matmul_splitk(a, b, s))
        rows.append(dict(impl=f"split_k={s}", programs=16 * s, us=ms * 1e3, TFLOPS=tflops(f, ms)))
    ms = bench(lambda: a @ b)
    rows.append(dict(impl="cuBLAS", programs="-", us=ms * 1e3, TFLOPS=tflops(f, ms)))
    report(rows, f"{M}x{N}x{K} bf16")
    finish()
