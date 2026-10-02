"""练习 04-1：分块矩阵乘（参考答案）"""
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


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
