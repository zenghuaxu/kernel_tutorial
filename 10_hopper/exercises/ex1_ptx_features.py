"""练习 10-1：从 PTX 里认出 Hopper 特性

写 kernel 时你以为用上了 Tensor Core / TMA，编译器未必真这么做了（dtype 不对、tile 太小、布局不支持……）。
最可靠的办法是看生成的 PTX。这个练习让你写一个小分析器，然后用它看 6 个 kernel。

目标：实现 analyze(ptx) -> dict，包含
  "wgmma_shapes": 出现过的 wgmma 形状，去重排序后的列表，如 ["m64n128k16"]；没有 wgmma 就是 []
  "mma_dtype":    wgmma 的输入类型，如 "bf16" / "tf32" / "e4m3"；没有 wgmma（或出现多种）就是 None
  "tma_load":     有没有 TMA 的 global -> shared 搬运
  "tma_store":    有没有 TMA 的 shared -> global 搬运
  "cp_async":     有没有 Ampere 风格的逐线程异步拷贝（cp.async.ca / cp.async.cg）

PTX 指令长这样（先跑一下 examples/hopper_features.py，或者自己 print(kernel.asm["ptx"]) 找找）：
  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {...}, ...;        ← 形状 m64n128k16，累加 f32，输入 bf16
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [...]   ← TMA load（目的地写在前面）
  cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [...]               ← TMA store
  cp.async.cg.shared.global [...], [...], 16;                               ← Ampere cp.async

提示：re.findall / re.search；注意 "cp.async.bulk" 也以 "cp.async" 开头，别把 TMA 误判成 cp_async。

做完之后想一想（对照测试里的期望值）：
  - 为什么 fp8 的 wgmma 是 k32，bf16 是 k16，tf32 是 k8？（提示：一条 wgmma 每行读多少字节的 K？）
  - "bf16 指针 stages=1" 为什么没有 cp.async？stages=3 时多出来的 cp.async 在干什么？
  - TMA 版本的寄存器数（examples/hopper_features.py 的 regs 列）为什么比指针版少？

运行：python 10_hopper/exercises/ex1_ptx_features.py
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from ref_kernels import fp8_matmul, matmul_ptr, matmul_tma, quantize_rowwise  # noqa: E402

from common import check_equal, finish  # noqa: E402


def analyze(ptx: str) -> dict:
    # TODO
    return {
        "wgmma_shapes": None,
        "mma_dtype": None,
        "tma_load": None,
        "tma_store": None,
        "cp_async": None,
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    a8, sa = quantize_rowwise(a)
    cases = {
        "fp32 ieee（不走 Tensor Core）": (
            lambda: matmul_ptr(a.float(), a.float(), BM=64, BN=64, BK=32, num_warps=4, ieee=True),
            dict(wgmma_shapes=[], mma_dtype=None, tma_load=False, tma_store=False, cp_async=True)),
        "fp32 默认（tf32）": (
            lambda: matmul_ptr(a.float(), a.float(), BK=32),
            dict(wgmma_shapes=["m64n128k8"], mma_dtype="tf32", tma_load=False, tma_store=False, cp_async=True)),
        "bf16 指针 stages=1": (
            lambda: matmul_ptr(a, a, num_stages=1),
            dict(wgmma_shapes=["m64n128k16"], mma_dtype="bf16", tma_load=False, tma_store=False, cp_async=False)),
        "bf16 指针 64x64x32": (
            lambda: matmul_ptr(a, a, BM=64, BN=64, BK=32, num_warps=4),
            dict(wgmma_shapes=["m64n64k16"], mma_dtype="bf16", tma_load=False, tma_store=False, cp_async=True)),
        "bf16 TMA": (
            lambda: matmul_tma(a, a),
            dict(wgmma_shapes=["m64n128k16"], mma_dtype="bf16", tma_load=True, tma_store=True, cp_async=False)),
        "fp8 TMA 128x256x128": (
            lambda: fp8_matmul(a8, sa, a8, sa),
            dict(wgmma_shapes=["m64n256k32"], mma_dtype="e4m3", tma_load=True, tma_store=True, cp_async=False)),
    }
    for name, (make, expected) in cases.items():
        kernel, _ = make()
        got = analyze(kernel.asm["ptx"])
        for key, val in expected.items():
            check_equal(f"{name}: {key}", got.get(key), val)
    finish()
