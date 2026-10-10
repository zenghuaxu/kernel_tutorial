"""示例：扫一遍 matmul 的几个旋钮，看它们各自的影响。

运行：python 04_triton_matmul/examples/knobs_sweep.py

  1. num_stages：软件流水线深度（1 = 不预取）
  2. GROUP_M：program 排序（1 = 朴素的按行排序）
  3. BLOCK 形状
同时打印每个配置的寄存器数和 shared memory 用量，帮助理解"为什么太大的 tile 会变慢/编译失败"。
"""
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).parent))
from matmul_walkthrough import matmul_kernel  # noqa: E402

from common import bench, gpu_name, gpu_spec, report, tflops  # noqa: E402


def run(a, b, c, BM, BN, BK, G, warps, stages):
    M, K = a.shape
    N = b.shape[1]
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    return matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                               c.stride(0), c.stride(1), BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=G,
                               num_warps=warps, num_stages=stages)


if __name__ == "__main__":
    s = 8192
    a = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(s, s, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(s, s, device="cuda", dtype=torch.bfloat16)
    flops = 2 * s**3

    configs = [
        # (BM, BN, BK, GROUP_M, warps, stages)
        (128, 128, 64, 8, 8, 1),
        (128, 128, 64, 8, 8, 2),
        (128, 128, 64, 8, 8, 3),
        (128, 128, 64, 8, 8, 4),
        (128, 256, 64, 8, 8, 3),
        (64, 64, 64, 8, 4, 3),
        (32, 32, 32, 8, 4, 3),
    ]
    rows = []
    for BM, BN, BK, G, w, st in configs:
        try:
            k = run(a, b, c, BM, BN, BK, G, w, st)
            ms = bench(lambda: run(a, b, c, BM, BN, BK, G, w, st))
            rows.append(dict(tile=f"{BM}x{BN}x{BK}", GROUP_M=G, warps=w, stages=st, regs=k.n_regs,
                             smem_KB=k.metadata.shared / 1024, TFLOPS=tflops(flops, ms)))
        except Exception as e:  # shared memory 超了会在编译/launch 时报 OutOfResources
            rows.append(dict(tile=f"{BM}x{BN}x{BK}", GROUP_M=G, warps=w, stages=st, regs="-",
                             smem_KB="-", TFLOPS=f"失败: {type(e).__name__}"))
    report(rows, f"{s}x{s}x{s} bf16")
    print("\n读表：stages=1 时 Tensor Core 要等 load；stages 越多越能把访存延迟藏起来，直到 shared memory 放不下。")
    print(f"smem_KB ≈ stages × (BM×BK + BK×BN) × 2 字节；{gpu_name()} 每个 block 最多 {gpu_spec().smem_per_block_kb} KB。")

    # GROUP_M：输出很"宽"（N 大）而 K 较小时最明显——朴素排序下同时在跑的 program 要读整行 B
    del a, b, c
    M, N, K = 16384, 16384, 2048
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    rows = []
    for G in [1, 4, 8, 16]:
        ms = bench(lambda: run(a, b, c, 128, 128, 64, G, 8, 4))
        rows.append(dict(GROUP_M=G, TFLOPS=tflops(2 * M * N * K, ms)))
    report(rows, f"program 排序：{M}x{N}x{K} bf16，tile 128x128x64，stages=4")
