"""示例：同一个 GEMM 的几种写法，编译出来到底用了哪些 Hopper 特性？跑多快？

运行：python 10_hopper/examples/hopper_features.py

看 PTX 里的几种关键指令：
  wgmma.mma_async        Hopper 的 warpgroup 异步 MMA（4 个 warp 一起发一条 64xNx16 的矩阵乘）
  mma.sync               Ampere 风格的 warp 级 MMA（H100 也支持，但吞吐只有 wgmma 的一半左右）
  cp.async.cg/ca         Ampere 风格的异步拷贝：每个线程自己算地址、搬 4~16 字节到 shared memory
  cp.async.bulk.tensor   TMA：一个线程发一条指令，硬件按 descriptor 搬一整个 tile
  mbarrier               shared memory 里的硬件屏障，TMA 用它通知"数据到了"
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ref_kernels import fp8_matmul, matmul_ptr, matmul_tma, quantize_rowwise  # noqa: E402

from common import bench, report, tflops  # noqa: E402

KEYS = ["wgmma.mma_async", "mma.sync", "cp.async.cg", "cp.async.bulk.tensor", "mbarrier"]


def features(k):
    ptx = k.asm["ptx"]
    return {key: ptx.count(key) for key in KEYS}


if __name__ == "__main__":
    S = 4096
    a = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(S, S, device="cuda", dtype=torch.bfloat16)
    a32, b32 = a.float(), b.float()
    a8, sa = quantize_rowwise(a)
    b8, sb = quantize_rowwise(b.t().contiguous())   # B 按 [N, K] 存
    flops = 2 * S**3

    variants = {
        "fp32 ieee (无 TC)": lambda: matmul_ptr(a32, b32, BM=64, BN=64, BK=32, num_warps=4, ieee=True),
        "fp32 tf32": lambda: matmul_ptr(a32, b32, BM=128, BN=128, BK=32),
        "bf16 指针, stages=1": lambda: matmul_ptr(a, b, num_stages=1),
        "bf16 指针, stages=3": lambda: matmul_ptr(a, b, num_stages=3),
        "bf16 TMA": lambda: matmul_tma(a, b),
        "bf16 TMA 128x256": lambda: matmul_tma(a, b, BN=256, num_stages=3),
        "fp8 e4m3 TMA": lambda: fp8_matmul(a8, sa, b8, sb),
    }
    rows = []
    for name, fn in variants.items():
        k, _ = fn()
        f = features(k)
        ms = bench(lambda: fn())
        rows.append(dict(variant=name, wgmma=f["wgmma.mma_async"], mma_sync=f["mma.sync"],
                         cp_async=f["cp.async.cg"], tma=f["cp.async.bulk.tensor"], mbar=f["mbarrier"],
                         regs=k.n_regs, smem_KB=k.metadata.shared // 1024, TFLOPS=tflops(flops, ms)))
    rows.append(dict(variant="cuBLAS bf16", wgmma="-", mma_sync="-", cp_async="-", tma="-", mbar="-",
                     regs="-", smem_KB="-", TFLOPS=tflops(flops, bench(lambda: a @ b))))
    report(rows, f"{S}^3 GEMM：PTX 里各类指令出现的次数（静态计数，不是执行次数）")
    print("\n注意：fp8 的 FLOPS 按同样的 2·M·N·K 算；H100 dense fp8 峰值约 1979 TFLOPS，bf16 约 989。")
