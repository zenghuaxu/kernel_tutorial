"""练习 10-2：TMA + persistent 的 GEMM

把单元 04 的 GEMM 升级成 Hopper 写法，两处改动：
  1. 用 TMA 搬数据：host 端用 TensorDescriptor.from_tensor(t, block_shape) 描述"这个 tensor 按多大的块读"，
     kernel 里 desc.load([行偏移, 列偏移]) 一次搬一整个 tile 进 shared memory，desc.store 写回。
     不用再算指针、不用写 mask：越界的部分 load 时硬件自动补零，store 时自动丢弃。
  2. persistent：grid = SM 个数（或更少），每个 program 循环处理 tile_id = pid, pid + P, pid + 2P, ...

目标：
  - wrapper：为 a、b、c 各建一个 TensorDescriptor（block_shape 分别是 [BM, BK]、[BK, BN]、[BM, BN]）
  - kernel：外层 tile 循环（用 tl.range(..., flatten=True)），内层 K 循环（desc.load + tl.dot），最后 c_desc.store
  - 测试里有 M、N 不是 tile 整数倍、以及只启动 1 / 5 个 program 的情况

提示：
  - tile_coords 已给出（单元 04 的 GROUP_M 映射）
  - desc.load 的偏移是**元素坐标**，不是 tile 编号：[pid_m * BM, ki * BK]
  - 输出 dtype：c_desc.dtype
  - TMA 的限制：最后一维必须连续，其他维的 stride 必须是 16 字节的倍数（所以测试里 K、N 都是 8 的倍数）

做完之后想一想：
  - 把 flatten=True 去掉、或者把 grid 改回"每个 tile 一个 program"（非 persistent），速度变多少？
  - 用 examples/hopper_features.py 的方法看看你的 kernel 的 regs：为什么比单元 04 的指针版少？
    （提示：TMA 版不需要每个线程持有一大堆地址和 mask）

运行：python 10_hopper/exercises/ex2_tma_persistent_matmul.py
"""
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
    # TODO:
    #   1. start = 本 program 编号；算 num_pid_m、num_pid_n、num_tiles、k_tiles
    #   2. for tile_id in tl.range(start, num_tiles, NUM_PROGRAMS, flatten=True):
    #        pid_m, pid_n = tile_coords(...)
    #        fp32 累加器；沿 K 用 a_desc.load / b_desc.load + tl.dot
    #        c_desc.store
    pass


def matmul(a: torch.Tensor, b: torch.Tensor, num_programs: int | None = None) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    # TMA 要求：除最后一维外的 stride 是 16 字节的倍数；最后一维 stride = 1
    assert a.is_contiguous() and b.is_contiguous() and (K * a.element_size()) % 16 == 0 and (N * 2) % 16 == 0
    BM, BN, BK = 128, 256, 64
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a_desc = None  # TODO: TensorDescriptor.from_tensor(...)
    b_desc = None  # TODO
    c_desc = None  # TODO
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
