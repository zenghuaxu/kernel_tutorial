"""练习 10-1：从 PTX 里认出 Hopper / Blackwell 的 Tensor Core 与 TMA 特性

写 kernel 时你以为用上了 Tensor Core / TMA，编译器未必真这么做了（dtype 不对、tile 太小、布局不支持……）。
最可靠的办法是看生成的 PTX。这个练习让你写一个小分析器，然后用它看 6 个 kernel。
测试按当前 GPU 的架构选期望值：H100/H200（sm_90）或 B200/GB200（sm_100）。

目标：实现 analyze(ptx) -> dict，包含
  "mma_inst":     用了哪种 Tensor Core 指令："tcgen05"（Blackwell）/ "wgmma"（Hopper）/ "mma.sync"（Ampere 风格）/ None
                  多种同时出现时按这个顺序取第一个
  "wgmma_shapes": 出现过的 wgmma 形状，去重排序后的列表，如 ["m64n128k16"]；没有 wgmma 就是 []
  "mma_dtype":    wgmma 的输入类型（"bf16" / "tf32" / "e4m3"），或 tcgen05 的 kind（"f16" / "tf32" / "f8f6f4"）；
                  没有（或出现多种）就是 None
  "tmem":         有没有申请 Tensor Memory（tcgen05.alloc）
  "tma_load":     有没有 TMA 的 global -> shared 搬运
  "tma_store":    有没有 TMA 的 shared -> global 搬运
  "cp_async":     有没有 Ampere 风格的逐线程异步拷贝（cp.async.ca / cp.async.cg）

PTX 指令长这样（先跑一下 examples/hopper_features.py，或者自己 print(kernel.asm["ptx"]) 找找）：
  wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {...}, ...;        ← Hopper：形状 m64n128k16，累加 f32，输入 bf16
  tcgen05.mma.cta_group::1.kind::f16 [%r1+0], %rd1, %rd2, %r2, %p1;         ← Blackwell：kind=f16，形状在 %r2（instr desc）里
  tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [...], 128;       ← Blackwell：申请 Tensor Memory
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [...]   ← TMA load（目的地写在前面）
  cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [...]               ← TMA store
  cp.async.cg.shared.global [...], [...], 16;                               ← Ampere cp.async

提示：re.findall / re.search；注意 "cp.async.bulk" 也以 "cp.async" 开头，别把 TMA 误判成 cp_async。

做完之后想一想（对照测试里的期望值）：
  - 为什么 fp8 的 wgmma 是 k32，bf16 是 k16，tf32 是 k8？（提示：一条 wgmma 每行读多少字节的 K？）
  - "bf16 指针 stages=1" 为什么没有 cp.async？stages=3 时多出来的 cp.async 在干什么？
  - TMA 版本的寄存器数（examples/hopper_features.py 的 regs 列）为什么比指针版少？
  - Blackwell 上累加器从寄存器搬进了 Tensor Memory，这对寄存器压力、epilogue 写法各有什么影响？
    （B200 上跑 examples/hopper_features.py，对比 regs 列；结尾的 tcgen05.ld 在干什么？）

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
        "mma_inst": None,
        "wgmma_shapes": None,
        "mma_dtype": None,
        "tmem": None,
        "tma_load": None,
        "tma_store": None,
        "cp_async": None,
    }


# 每个 case：(构造 kernel, Hopper 期望, Blackwell 期望)。两代卡上 TMA / cp.async 的结论一样，只是 MMA 指令不同：
#   sm_90  (H100/H200) : wgmma.mma_async，形状和输入类型都写在指令名里
#   sm_100 (B200/GB200): tcgen05.mma，累加器在 Tensor Memory 里；形状在 instruction descriptor 寄存器里，
#                        指令名里只有 kind（f16 = bf16/fp16，tf32，f8f6f4 = fp8/fp6/fp4）
_NO_TMA = dict(tma_load=False, tma_store=False)
_TMA = dict(tma_load=True, tma_store=True, cp_async=False)


def _hopper(shape, dtype, **kw):
    return dict(mma_inst="wgmma", wgmma_shapes=[shape], mma_dtype=dtype, tmem=False, **kw)


def _blackwell(kind, **kw):
    return dict(mma_inst="tcgen05", wgmma_shapes=[], mma_dtype=kind, tmem=True, **kw)


if __name__ == "__main__":
    major = torch.cuda.get_device_capability()[0]
    if major not in (9, 10):
        print(f"这个练习的期望值只覆盖 sm_90（Hopper）和 sm_100（数据中心 Blackwell），当前是 sm_{major}x，跳过。")
        sys.exit(0)
    col = 0 if major == 9 else 1
    torch.manual_seed(0)
    a = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    a8, sa = quantize_rowwise(a)
    no_tc = dict(mma_inst=None, wgmma_shapes=[], mma_dtype=None, tmem=False, cp_async=True, **_NO_TMA)
    cases = {
        "fp32 ieee（不走 Tensor Core）": (
            lambda: matmul_ptr(a.float(), a.float(), BM=64, BN=64, BK=32, num_warps=4, ieee=True),
            no_tc, no_tc),
        "fp32 默认（tf32）": (
            lambda: matmul_ptr(a.float(), a.float(), BK=32),
            _hopper("m64n128k8", "tf32", cp_async=True, **_NO_TMA),
            _blackwell("tf32", cp_async=True, **_NO_TMA)),
        "bf16 指针 stages=1": (
            lambda: matmul_ptr(a, a, num_stages=1),
            _hopper("m64n128k16", "bf16", cp_async=False, **_NO_TMA),
            _blackwell("f16", cp_async=False, **_NO_TMA)),
        "bf16 指针 64x64x32": (
            lambda: matmul_ptr(a, a, BM=64, BN=64, BK=32, num_warps=4),
            _hopper("m64n64k16", "bf16", cp_async=True, **_NO_TMA),
            _blackwell("f16", cp_async=True, **_NO_TMA)),
        "bf16 TMA": (
            lambda: matmul_tma(a, a),
            _hopper("m64n128k16", "bf16", **_TMA),
            _blackwell("f16", **_TMA)),
        "fp8 TMA 128x256x128": (
            lambda: fp8_matmul(a8, sa, a8, sa),
            _hopper("m64n256k32", "e4m3", **_TMA),
            _blackwell("f8f6f4", **_TMA)),
    }
    print(f"按 sm_{major}0 的期望值检查")
    for name, (make, *expected) in cases.items():
        kernel, _ = make()
        got = analyze(kernel.asm["ptx"])
        for key, val in expected[col].items():
            check_equal(f"{name}: {key}", got.get(key), val)
    finish()
