"""练习 10-1：从 PTX 里认出 Hopper 特性（参考答案）"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
from ref_kernels import fp8_matmul, matmul_ptr, matmul_tma, quantize_rowwise  # noqa: E402

from common import check_equal, finish  # noqa: E402


def analyze(ptx: str) -> dict:
    wgmma = re.findall(r"wgmma\.mma_async\.sync\.aligned\.(m\d+n\d+k\d+)\.f32\.(\w+)\.\w+", ptx)
    shapes = sorted({s for s, _ in wgmma})
    dtypes = {d for _, d in wgmma}
    return {
        "wgmma_shapes": shapes,
        "mma_dtype": dtypes.pop() if len(dtypes) == 1 else None,
        # TMA load：global -> shared（目的地在前），用 mbarrier 通知完成
        "tma_load": re.search(r"cp\.async\.bulk\.tensor\.\dd\.shared::cluster\.global", ptx) is not None,
        # TMA store：shared -> global
        "tma_store": re.search(r"cp\.async\.bulk\.tensor\.\dd\.global\.shared::cta", ptx) is not None,
        # Ampere 风格的逐线程异步拷贝
        "cp_async": re.search(r"cp\.async\.c[ag]\.shared\.global", ptx) is not None,
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
