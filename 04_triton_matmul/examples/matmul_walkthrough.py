"""示例：一个完整、带注释的 Triton 分块矩阵乘，并和 cuBLAS（torch.matmul）对比。

运行：python 04_triton_matmul/examples/matmul_walkthrough.py

C[M, N] = A[M, K] @ B[K, N]，A/B 是 bf16，累加用 fp32，输出 bf16。
"""
import torch
import triton
import triton.language as tl

from common import bench, check, gpu_spec, report, tflops


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ---- 1. program -> 输出 tile 的映射（带 GROUP_M 分组，见讲义 4.5）----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ---- 2. 这个 tile 在 A、B 里对应的起始指针 ----
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak   # [BLOCK_M, BLOCK_K]
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn   # [BLOCK_K, BLOCK_N]

    # ---- 3. 沿 K 循环：每次取 A 的一条 [BM, BK] 和 B 的一条 [BK, BN]，乘加到 fp32 累加器 ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)            # 编译成 Tensor Core 指令：H100 上是 wgmma，B200 上是 tcgen05.mma
        a_ptrs += BLOCK_K * stride_ak      # 指针前移一个 K 块
        b_ptrs += BLOCK_K * stride_bk

    # ---- 4. epilogue：转回输出 dtype，写回（带边界 mask）----
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, num_warps=8, num_stages=3):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
        num_warps=num_warps, num_stages=num_stages,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1000, 777, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(777, 513, device="cuda", dtype=torch.bfloat16)
    check("1000x777 @ 777x513", matmul(a, b), (a.float() @ b.float()).to(torch.bfloat16), atol=5e-2, rtol=2e-2)

    rows = []
    for s in [512, 1024, 2048, 4096, 8192]:
        a = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
        flops = 2 * s**3
        t_tr = bench(lambda: matmul(a, b))
        t_cb = bench(lambda: a @ b)
        rows.append(dict(MNK=s, triton_TFLOPS=tflops(flops, t_tr), cublas_TFLOPS=tflops(flops, t_cb),
                         ratio=t_cb / t_tr))
    report(rows, f"bf16 方阵 GEMM（{gpu_spec().mma_note()}；卡被别人占用时数字偏低且有噪声）")
