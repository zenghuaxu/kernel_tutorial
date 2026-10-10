"""示例：看看这张卡的参数，并实测带宽和算力上限。

运行：python 00_gpu_mental_model/examples/device_info.py
"""
import torch

from common import bench, gbps, gpu_spec, report, tflops

if __name__ == "__main__":
    p = torch.cuda.get_device_properties(0)
    spec = gpu_spec()
    print(f"GPU: {p.name}")
    print(f"  SM 个数                 : {p.multi_processor_count}")
    print(f"  显存                    : {p.total_memory / 2**30:.1f} GiB")
    print(f"  L2 cache                : {p.L2_cache_size / 2**20:.0f} MiB")
    print(f"  每 SM 寄存器(32bit)     : {p.regs_per_multiprocessor}")
    print(f"  每 SM 最多线程          : {p.max_threads_per_multi_processor}")
    print(f"  每 block 可用 shared mem: {p.shared_memory_per_block_optin / 1024:.0f} KiB（需 opt-in）")
    print(f"  compute capability      : {p.major}.{p.minor}")
    free, total = torch.cuda.mem_get_info()
    print(f"  当前空闲显存            : {free / 2**30:.1f} GiB（其余被别的进程占用）")

    rows = []
    # 1) HBM 带宽：拷贝一个 1 GiB 的 tensor（读 1 GiB + 写 1 GiB）
    n = 1 << 28  # 2^28 个 fp32 = 1 GiB
    src = torch.empty(n, device="cuda")
    dst = torch.empty_like(src)
    ms = bench(lambda: dst.copy_(src))
    rows.append(dict(test="copy 1GiB fp32", ms=ms, achieved=f"{gbps(2 * n * 4, ms):.0f} GB/s", peak=f"{spec.hbm_gbps:.0f} GB/s" if spec.hbm_gbps else "?"))
    del src, dst

    # 2) Tensor Core 算力：bf16 GEMM 8192^3
    M = N = K = 8192
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    ms = bench(lambda: a @ b)
    rows.append(dict(test="bf16 GEMM 8192^3", ms=ms, achieved=f"{tflops(2 * M * N * K, ms):.0f} TFLOPS", peak=f"{spec.bf16_tflops:.0f} TFLOPS" if spec.bf16_tflops else "?"))

    # 3) 非 Tensor Core 的 fp32 算力很难用 PyTorch 直接测到，这里测 fp32 GEMM（关掉 TF32，走 SIMT FMA）
    torch.backends.cuda.matmul.allow_tf32 = False
    a32, b32 = a.float()[:4096, :4096], b.float()[:4096, :4096]
    ms = bench(lambda: a32 @ b32)
    rows.append(dict(test="fp32 GEMM 4096^3 (no TF32)", ms=ms, achieved=f"{tflops(2 * 4096**3, ms):.0f} TFLOPS", peak=f"{spec.fp32_tflops:.0f} TFLOPS" if spec.fp32_tflops else "?"))

    # 4) launch 开销：一个几乎什么都不干的 kernel
    x = torch.zeros(1, device="cuda")
    ms = bench(lambda: x.add_(1), warmup=100, rep=1000)
    rows.append(dict(test="tiny kernel (launch overhead)", ms=ms, achieved=f"{ms * 1e3:.1f} us", peak="-"))
    report(rows, f"实测于 {spec.name}（peak 列是官方 dense 峰值；卡被别人占用时数字偏保守）")
