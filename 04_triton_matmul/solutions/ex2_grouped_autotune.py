"""练习 04-2：分组 program 排序 + autotune（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, check_equal, finish, report, tflops


# ---------------------------------------------------------------- 第 1 部分：program 排序
@triton.jit
def grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """把一维 pid 映射成 (pid_m, pid_n)：每 GROUP_M 行 tile 为一组，组内按列优先遍历。"""
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


# ---------------------------------------------------------------- 第 2 部分：autotune
CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=CONFIGS, key=["M", "N", "K"])
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_pid(pid, tl.cdiv(M, BLOCK_M), tl.cdiv(N, BLOCK_N), GROUP_M)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # grid 依赖 autotune 选中的 BLOCK_M/BLOCK_N，所以写成 lambda meta: ...
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                        c.stride(0), c.stride(1))
    return c


# ---------------------------------------------------------------- 测试用：把映射写出来
@triton.jit
def _probe_kernel(out_ptr, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    pid_m, pid_n = grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M)
    tl.store(out_ptr + 2 * pid, pid_m)
    tl.store(out_ptr + 2 * pid + 1, pid_n)


def _expected_order(num_pid_m, num_pid_n, G):
    order = []
    for m0 in range(0, num_pid_m, G):
        rows = range(m0, min(m0 + G, num_pid_m))
        for n in range(num_pid_n):
            for m in rows:
                order.append((m, n))
    return order


if __name__ == "__main__":
    torch.manual_seed(0)
    for nm, nn, G in [(8, 8, 4), (10, 7, 4), (5, 3, 8), (16, 16, 1)]:
        out = torch.full((nm * nn, 2), -1, device="cuda", dtype=torch.int32)
        _probe_kernel[(nm * nn,)](out, nm, nn, GROUP_M=G)
        got = [tuple(x) for x in out.tolist()]
        check_equal(f"grouped_pid 映射 {nm}x{nn} GROUP_M={G}", got == _expected_order(nm, nn, G), True)

    check_equal("autotune 的 key 覆盖 M、N、K", {"M", "N", "K"} <= set(matmul_kernel.keys), True)
    check_equal("CONFIGS 有 2~6 个", 2 <= len(CONFIGS) <= 6, True)

    tol = dict(atol=5e-2, rtol=2e-2)
    for M, N, K in [(512, 512, 512), (1000, 3000, 700), (4096, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        check(f"autotuned matmul {M}x{N}x{K}", matmul(a, b), (a.float() @ b.float()).to(a.dtype), **tol)
        print(f"    autotune 选中: {matmul_kernel.best_config}")

    rows = []
    for M, N, K in [(4096, 4096, 4096), (16384, 16384, 2048)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        f = 2 * M * N * K
        rows.append(dict(shape=f"{M}x{N}x{K}", yours=tflops(f, bench(lambda: matmul(a, b))),
                         cublas=tflops(f, bench(lambda: a @ b))))
        del a, b
    report(rows, "TFLOPS（bf16）")
    finish()
