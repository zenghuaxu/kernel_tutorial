"""当前 GPU 的规格（理论峰值 + 运行时查到的硬件参数），让示例/练习在不同卡上都能报"占峰值多少"。

    from common import gpu_spec
    spec = gpu_spec()
    print(spec.name, spec.sm_count, spec.hbm_gbps, spec.bf16_tflops)

峰值是 NVIDIA 公布的 dense（不含稀疏）数字，查表得到；表里没有的卡对应字段是 None，
SM 个数 / shared memory / L2 这类结构参数总是从驱动实时读取。
"""
from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class GPUSpec:
    name: str                    # 简称，如 "H100" / "B200"
    full_name: str               # 驱动报告的完整名字
    arch: tuple[int, int]        # compute capability，如 (9, 0)
    sm_count: int
    smem_per_sm_kb: int          # 每个 SM 的 shared memory（含 L1 可切分部分的上限）
    smem_per_block_kb: int       # 单个 block 最多能申请的 shared memory（opt-in）
    l2_mb: int
    hbm_gbps: float | None       # HBM 峰值带宽 GB/s
    fp32_tflops: float | None    # CUDA core fp32（不走 Tensor Core）
    tf32_tflops: float | None    # Tensor Core 峰值，dense
    bf16_tflops: float | None
    fp8_tflops: float | None

    @property
    def peak_bw(self) -> float | None:
        """B/s，给 roofline 公式用"""
        return self.hbm_gbps * 1e9 if self.hbm_gbps else None

    @property
    def peak_bf16(self) -> float | None:
        """FLOP/s"""
        return self.bf16_tflops * 1e12 if self.bf16_tflops else None

    def bw_note(self) -> str:
        return f"{self.name} HBM 峰值约 {self.hbm_gbps:.0f} GB/s" if self.hbm_gbps else f"{self.name}"

    def mma_note(self) -> str:
        if not self.bf16_tflops:
            return self.name
        s = f"{self.name} dense 峰值：bf16 ≈ {self.bf16_tflops:.0f}"
        if self.fp8_tflops:
            s += f"，fp8 ≈ {self.fp8_tflops:.0f}"
        return s + " TFLOPS"


# 关键字 -> (hbm GB/s, fp32, tf32, bf16, fp8)，单位 TFLOPS；按顺序匹配，更具体的放前面
_PEAKS = [
    ("H100 PCIe", "H100", (2000, 51, 378, 756, 1513)),
    ("H100",      "H100", (3350, 67, 495, 989, 1979)),    # SXM5 (HBM3)
    ("H200",      "H200", (4800, 67, 495, 989, 1979)),
    ("H20",       "H20",  (4000, 44, 74, 148, 296)),
    ("GB200",     "GB200", (8000, 80, 1250, 2500, 5000)),
    ("B200",      "B200", (8000, 75, 1100, 2250, 4500)),  # HGX B200
    ("B300",      "B300", (8000, 75, 1100, 2250, 4500)),
    ("A100",      "A100", (2039, 19.5, 156, 312, None)),  # SXM 80GB
    ("L40S",      "L40S", (864, 91.6, 183, 362, 733)),
    ("RTX 5090",  "RTX 5090", (1792, 105, 105, 210, 419)),
    ("RTX 4090",  "RTX 4090", (1008, 82.6, 82.6, 165, 330)),
]


@lru_cache(maxsize=None)
def gpu_spec(device: int = 0) -> GPUSpec:
    p = torch.cuda.get_device_properties(device)
    short, peaks = p.name.replace("NVIDIA ", ""), (None,) * 5
    for key, nm, vals in _PEAKS:
        if key in p.name:
            short, peaks = nm, vals
            break
    return GPUSpec(
        name=short, full_name=p.name, arch=(p.major, p.minor), sm_count=p.multi_processor_count,
        smem_per_sm_kb=p.shared_memory_per_multiprocessor // 1024,
        smem_per_block_kb=p.shared_memory_per_block_optin // 1024,
        l2_mb=p.L2_cache_size // (1024 * 1024),
        hbm_gbps=peaks[0], fp32_tflops=peaks[1], tf32_tflops=peaks[2], bf16_tflops=peaks[3], fp8_tflops=peaks[4],
    )


def gpu_name() -> str:
    return gpu_spec().name
