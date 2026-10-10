"""示例：occupancy —— 每个 SM 上同时驻留多少个 warp，以及它怎么影响 memory-bound kernel。

运行：python 06_cuda_memory/examples/occupancy.py

做法：同一个简单 copy kernel（每个线程拷 1 个 float），launch 时**额外申请一块用不到的动态 shared memory**。
shared memory 申请得越多，一个 SM 能同时放下的 block 越少 → 驻留 warp 越少 → 能藏住的访存延迟越少。
"""
import torch

from common import bench, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__global__ void copy_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    extern __shared__ float unused_smem[];      // 动态 shared memory：大小在 launch 时第三个参数给
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i];
}

// 一个寄存器用得很多的 kernel：每个线程维护 64 个累加器
__global__ void heavy_regs_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    float acc[64];
#pragma unroll
    for (int k = 0; k < 64; ++k) acc[k] = 0.f;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (int64_t j = i; j < n; j += (int64_t)gridDim.x * blockDim.x) {
        float v = x[j];
#pragma unroll
        for (int k = 0; k < 64; ++k) acc[k] = acc[k] * v + (float)k;
    }
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 64; ++k) s += acc[k];
    if (i < n) out[i] = s;
}

static void set_smem_limit(int64_t smem) {
    // 动态 shared memory 超过 48 KB 必须显式 opt-in，否则 launch 失败
    C10_CUDA_CHECK(cudaFuncSetAttribute(copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
}

// 返回 [每个 SM 最多驻留的 block 数, 每线程寄存器数]
std::vector<int64_t> occupancy(int64_t threads, int64_t smem, bool heavy) {
    int blocks = 0;
    cudaFuncAttributes attr;
    if (heavy) {
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, heavy_regs_kernel, (int)threads, (size_t)smem));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, heavy_regs_kernel));
    } else {
        set_smem_limit(smem);
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, copy_kernel, (int)threads, (size_t)smem));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attr, copy_kernel));
    }
    return {blocks, attr.numRegs};
}

torch::Tensor copy_with_smem(torch::Tensor x, int64_t threads, int64_t smem) {
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    set_smem_limit(smem);
    copy_kernel<<<(n + threads - 1) / threads, threads, smem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("occupancy_demo", CUDA_SRC, ["occupancy", "copy_with_smem"])
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}: {props.multi_processor_count} SMs, 每 SM 最多 {props.max_threads_per_multi_processor} 线程, "
          f"每 SM {props.regs_per_multiprocessor} 个寄存器, 每 block shared memory 上限(opt-in) {props.shared_memory_per_block_optin // 1024} KB")

    max_warps = props.max_threads_per_multi_processor // 32   # H100 / B200 都是 64
    n = 1 << 26
    x = torch.randn(n, device="cuda")
    threads = 256
    rows = []
    for smem_kb in [0, 16, 32, 48, 64, 100, 200]:
        smem = smem_kb * 1024
        blocks_per_sm, regs = mod.occupancy(threads, smem, False)
        warps = blocks_per_sm * threads // 32
        out = mod.copy_with_smem(x, threads, smem)
        assert torch.equal(out, x)
        ms = bench(lambda: mod.copy_with_smem(x, threads, smem))
        rows.append(dict(dyn_smem_KB=smem_kb, blocks_per_SM=blocks_per_sm, warps_per_SM=warps,
                         occupancy=f"{warps / max_warps:.0%}", GBps=gbps(2 * n * 4, ms)))
    report(rows, f"copy kernel, {threads} 线程/block, 每线程 1 个 float（{props.name} 每 SM 最多 {max_warps} 个 warp）")

    blocks_per_sm, regs = mod.occupancy(threads, 0, True)
    print(f"\nheavy_regs_kernel：每线程 {regs} 个寄存器 → 每 SM 最多 {blocks_per_sm} 个 {threads} 线程的 block "
          f"（寄存器总量 65536 / ({regs}×{threads}) ≈ {65536 / (regs * threads):.1f}，还要按分配粒度向下取整）")
    print("\n结论：这种每线程只有 1 个 load 在路上的 kernel 很依赖 occupancy；")
    print("如果每个线程同时发好几个独立的 load（float4、循环展开 = ILP），低 occupancy 也能跑满带宽——GEMM 就是这么干的。")
