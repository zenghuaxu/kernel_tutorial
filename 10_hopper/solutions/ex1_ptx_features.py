"""练习 10-1：从 PTX 里认出 Hopper / Blackwell 的 Tensor Core 与 TMA 特性（参考答案）"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from ref_kernels import fp8_matmul, matmul_ptr, matmul_tma, quantize_rowwise  # noqa: E402

from common import check_equal, finish  # noqa: E402


def analyze(ptx: str) -> dict:
    wgmma = re.findall(r"wgmma\.mma_async\.sync\.aligned\.(m\d+n\d+k\d+)\.f32\.(\w+)\.\w+", ptx)
    tcgen05 = re.findall(r"tcgen05\.mma\.cta_group::\d\.kind::(\w+)", ptx)
    if tcgen05:
        inst, dtypes = "tcgen05", set(tcgen05)
    elif wgmma:
        inst, dtypes = "wgmma", {d for _, d in wgmma}
    elif re.search(r"\bmma\.sync\.aligned", ptx):
        inst, dtypes = "mma.sync", set()
    else:
        inst, dtypes = None, set()
    return {
        "mma_inst": inst,
        "wgmma_shapes": sorted({s for s, _ in wgmma}),
        "mma_dtype": dtypes.pop() if len(dtypes) == 1 else None,
        # Blackwell：累加器放在 Tensor Memory，用之前要 tcgen05.alloc 申请
        "tmem": "tcgen05.alloc" in ptx,
        # TMA load：global -> shared（目的地在前），用 mbarrier 通知完成
        "tma_load": re.search(r"cp\.async\.bulk\.tensor\.\dd\.shared::cluster\.global", ptx) is not None,
        # TMA store：shared -> global
        "tma_store": re.search(r"cp\.async\.bulk\.tensor\.\dd\.global\.shared::cta", ptx) is not None,
        # Ampere 风格的逐线程异步拷贝
        "cp_async": re.search(r"cp\.async\.c[ag]\.shared\.global", ptx) is not None,
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
