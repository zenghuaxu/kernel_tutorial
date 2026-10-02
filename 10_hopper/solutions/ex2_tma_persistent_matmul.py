"""练习 10-2：TMA + persistent 的 GEMM（参考答案）"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from common import bench, check, finish, report, tflops

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    """单元 04 的 GROUP_M 分组映射（已给出）。"""
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (tile_id // num_pid_in_group) * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % num_pid_in_group) % group_size_m
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_tma_persistent_kernel(
    a_desc, b_desc, c_desc, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
):
    start = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)
    # flatten=True：让编译器把"tile 循环 × K 循环"当作一条长流水线，
    # 下一个 tile 的 TMA load 可以在当前 tile 的 epilogue 时就发出去
    for tile_id in tl.range(start, num_tiles, NUM_PROGRAMS, flatten=True):
        pid_m, pid_n = tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_M)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for ki in range(k_tiles):
            a = a_desc.load([pid_m * BM, ki * BK])    # 越界部分 TMA 自动补零
            b = b_desc.load([ki * BK, pid_n * BN])
            acc = tl.dot(a, b, acc)
        c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_desc.dtype))   # 越界部分 TMA 自动丢弃


def matmul(a: torch.Tensor, b: torch.Tensor, num_programs: int | None = None) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    # TMA 要求：除最后一维外的 stride 是 16 字节的倍数；最后一维 stride = 1
    assert a.is_contiguous() and b.is_contiguous() and (K * a.element_size()) % 16 == 0 and (N * 2) % 16 == 0
    BM, BN, BK = 128, 256, 64
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a_desc = TensorDescriptor.from_tensor(a, [BM, BK])
    b_desc = TensorDescriptor.from_tensor(b, [BK, BN])
    c_desc = TensorDescriptor.from_tensor(c, [BM, BN])
    num_tiles = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    P = min(num_programs or NUM_SMS, num_tiles)
    matmul_tma_persistent_kernel[(P,)](
        a_desc, b_desc, c_desc, M, N, K,
        BM=BM, BN=BN, BK=BK, GROUP_M=8, NUM_PROGRAMS=P, num_warps=8, num_stages=3,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    tol = dict(atol=5e-2, rtol=2e-2)
    for M, N, K in [(128, 256, 64), (1000, 776, 520), (300, 264, 4104), (4096, 4096, 4096)]:
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        check(f"{M}x{N}x{K}", matmul(a, b), (a.float() @ b.float()).to(a.dtype), **tol)
    a = torch.randn(1000, 1032, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1032, 1544, device="cuda", dtype=torch.bfloat16)
    for p in [1, 5]:
        check(f"1000x1544x1032 只用 {p} 个 program", matmul(a, b, num_programs=p),
              (a.float() @ b.float()).to(a.dtype), **tol)

    rows = []
    for S in [4096, 8192]:
        a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
        rows.append(dict(MNK=S, yours=tflops(2 * S**3, bench(lambda: matmul(a, b))),
                         cublas=tflops(2 * S**3, bench(lambda: a @ b))))
        del a, b
    report(rows, "TFLOPS（bf16；单元 04 的指针版约 560~640）")
    finish()
