"""练习 04-2：分组 program 排序 + autotune

两部分：

第 1 部分：写 grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M) -> (pid_m, pid_n)
  把一维 pid 映射到输出 tile 坐标，规则（讲义 4.5 节）：
    - 每 GROUP_M 行 tile 组成一组（最后一组可能不足 GROUP_M 行）
    - 组与组按顺序排；组内**按列优先**：先走完这一组在第 0 列的所有行，再到第 1 列……
  例：num_pid_m=4, num_pid_n=3, GROUP_M=2 时，pid 0..11 依次对应
    (0,0)(1,0)(0,1)(1,1)(0,2)(1,2) (2,0)(3,0)(2,1)(3,1)(2,2)(3,2)
  测试用一个探针 kernel 把映射写出来逐个比对。

第 2 部分：autotune
  - 在 CONFIGS 里写 3~4 个 triton.Config（块大小、GROUP_M、num_warps、num_stages）
  - 给 @triton.autotune 填上 key：哪些参数一变就要重新调优？
  - 配置别太多：每个配置第一次遇到新 shape 都要编译 + 跑几十次，这台卡被训练任务共享，越多越慢

提示：
  - kernel 里没有 Python 的 min，用 tl.minimum
  - triton.Config({"BLOCK_M": 128, ...}, num_warps=8, num_stages=4)
  - 跑完后 matmul_kernel.best_config 能看到选中了哪个

做完之后想一想：
  - 三个测试 shape 选中的配置为什么不一样？小矩阵为什么倾向于小 tile？
  - 把 GROUP_M 都改成 1，16384x16384x2048 掉多少？（examples/knobs_sweep.py 里测过）

运行：python 04_triton_matmul/exercises/ex2_grouped_autotune.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, check_equal, finish, report, tflops


# ---------------------------------------------------------------- 第 1 部分：program 排序
@triton.jit
def grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """把一维 pid 映射成 (pid_m, pid_n)：每 GROUP_M 行 tile 为一组，组内按列优先遍历。"""
    # TODO: 算出 pid_m, pid_n
    pid_m = 0
    pid_n = 0
    return pid_m, pid_n


# ---------------------------------------------------------------- 第 2 部分：autotune
CONFIGS = [
    # TODO: 3~4 个 triton.Config
]


@triton.autotune(configs=CONFIGS, key=[])  # TODO: key
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
