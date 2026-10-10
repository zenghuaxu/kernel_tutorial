"""练习 04-1：分块矩阵乘

目标：写出 C = A @ B 的 Triton kernel。
  - 二维 grid：program (pid_m, pid_n) 负责 C 的一个 [BLOCK_M, BLOCK_N] tile（wrapper 已写好）
  - 沿 K 循环：每次 load A 的 [BLOCK_M, BLOCK_K] 和 B 的 [BLOCK_K, BLOCK_N]，用 tl.dot 乘加到 fp32 累加器
  - M、N、K 都可以不是 block 的整数倍 → 三个方向都要 mask（K 方向用 other=0.0 补零，补零不影响乘加结果）
  - A、B 可以是任意 stride（测试里有转置视图和切片），全部通过 stride 寻址
  - 最后把累加器转成 C 的 dtype 再 store

提示：
  - 讲义 4.3 节的代码骨架；acc = tl.dot(a, b, acc) 等价于 acc += a @ b，但能让编译器直接累加进去
  - 循环里让指针前移：a_ptrs += BLOCK_K * stride_ak
  - K 方向的 mask：第 k 次循环剩下 K - k*BLOCK_K 个元素

做完之后想一想：
  - 你的 kernel 在 4096 上达到 cuBLAS 的百分之多少？1024 呢？为什么小矩阵差距更大？（数一数 1024 时有多少个 program，H100 有 132 个 SM，B200 有 148 个）
  - 测试里的 B 是 W.t() 视图，stride_bk=1。这时 B tile 的 load 还是合并访存吗？tl.dot 介意 B 的 tile 是哪种布局吗？

运行：python 04_triton_matmul/exercises/ex1_tiled_matmul.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, report, tflops


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # TODO:
    #   1. pid_m / pid_n，以及 offs_m、offs_n、offs_k
    #   2. A tile 和 B tile 的初始指针
    #   3. fp32 累加器 + 沿 K 的循环（带 mask 的 load、tl.dot、指针前移）
    #   4. 转 dtype、带 mask 写回 C
    pass


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B。A: [M, K]，B: [K, N]，任意 stride；C 是新的 contiguous tensor，dtype 同 A。"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2 and a.dtype == b.dtype
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=8, num_stages=3,
    )
    return c


def ref(a, b):
    return (a.float() @ b.float()).to(a.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    tol = dict(atol=5e-2, rtol=2e-2)   # 输出是 bf16/fp16：参考值也 round 到同一 dtype，差 1 个 ulp 以内
    for dtype in [torch.float16, torch.bfloat16]:
        for M, N, K in [(128, 128, 64), (256, 512, 1024), (1, 1, 1), (100, 300, 70), (1000, 777, 513)]:
            a = torch.randn(M, K, device="cuda", dtype=dtype)
            b = torch.randn(K, N, device="cuda", dtype=dtype)
            check(f"{str(dtype)[6:]} {M}x{K} @ {K}x{N}", matmul(a, b), ref(a, b), **tol)
    # 非 contiguous：B 是 nn.Linear 权重 [N, K] 的转置视图；A 是切片
    w = torch.randn(384, 640, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(300, 700, device="cuda", dtype=torch.bfloat16)[:, 30:670]
    check("A 切片 @ W.t() 视图", matmul(a, w.t()), ref(a, w.t()), **tol)

    rows = []
    for s in [1024, 4096]:
        a = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
        rows.append(dict(MNK=s, yours=tflops(2 * s**3, bench(lambda: matmul(a, b))),
                         cublas=tflops(2 * s**3, bench(lambda: a @ b))))
    report(rows, "TFLOPS（bf16）")
    finish()
