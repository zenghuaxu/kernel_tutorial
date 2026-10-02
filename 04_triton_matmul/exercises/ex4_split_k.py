"""练习 04-4：split-K —— 给"矮胖"的 GEMM 找并行度

问题：M、N 小而 K 很大的 GEMM（比如 LoRA、小 batch 的投影、某些 attention 的 dV 计算），
输出 tile 很少——256x256 的输出用 64x64 tile 只有 16 个 program，132 个 SM 里 116 个闲着。
每个 program 还要串行走完一条超长的 K 循环。

split-K：把 K 切成 split_k 段，grid 变成 (tile 数, split_k)。program (t, s) 只算 tile t 在第 s 段 K 上的部分和，
最后把各段部分和加起来。这里用最简单的做法：输出是预先清零的 fp32 buffer，每个 program 用 tl.atomic_add 加进去。

目标：
  1. wrapper：算 k_per_split（每段多长，必须是 BLOCK_K 的整数倍，各段加起来覆盖整个 K）和 grid 的第二维
  2. kernel：
     - 由 program_id(1) 算出本段的 [k_start, k_end)
     - 只在这一段上做 K 循环（mask 要以 k_end 为界，不能越过本段读到下一段的数据，否则会重复累加）
     - 用 tl.atomic_add 把 fp32 部分和加到 c 上

提示：
  - for k in range(k_start, k_end, BLOCK_K): ... 在 Triton 里可以直接用运行时边界
  - 本段最后一个 block 可能不满：mask = (k + tl.arange(0, BLOCK_K)) < k_end
  - tl.atomic_add(ptrs, val, mask=...)
  - 测试里 split_k=32 而 K=333 时，很多段是空的（k_start >= K）——你的代码要让这些 program 什么也不加（加 0 也行）

做完之后想一想：
  - 表里 split_k 从 1 到 64，速度怎么变？为什么到某个值之后不再变快甚至变慢？（原子加的冲突、每段太短主循环流水线排不满）
  - 原子加让结果不再是 bit-wise 确定的。训练里需要确定性时怎么办？（提示：先写到 [split_k, M, N] 的 workspace，再用第二个 kernel 按固定顺序归约）
  - cuBLAS 为什么还是更快？（它对这种 shape 也会选 split-K / stream-K，而且 epilogue 归约做得更好）

运行：python 04_triton_matmul/exercises/ex4_split_k.py
"""
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # TODO:
    #   1. 本段的 k_start / k_end
    #   2. A、B tile 的初始指针（注意 K 方向从 k_start 开始）
    #   3. 本段内的 K 循环（mask 以 k_end 为界）
    #   4. tl.atomic_add 写回 fp32 的 c
    pass


def matmul_splitk(a: torch.Tensor, b: torch.Tensor, split_k: int) -> torch.Tensor:
    """返回 fp32 的 A @ B。K 维被切成 split_k 段，由不同 program 并行计算再原子加起来。"""
    M, K = a.shape
    N = b.shape[1]
    BM, BN, BK = 64, 64, 64
    c = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    k_per_split = None  # TODO: 每段的长度，BK 的整数倍
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), None)  # TODO: 第二维 = 实际需要几段
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
