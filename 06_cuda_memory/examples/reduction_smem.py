"""示例：shared memory 树形归约的几个版本（经典的 Mark Harris "Optimizing Parallel Reduction"）。

运行：python 06_cuda_memory/examples/reduction_smem.py

每个版本都是两遍以上的 launch：每个 block 把自己那段归约成 1 个部分和，写到 partial[blockIdx.x]，
然后对 partial 再做一次，直到只剩 1 个数。
  v1 交错寻址 + 取模判断：if (tid % (2*s) == 0)       —— 严重的 warp 分支发散
  v2 交错寻址 + 连续线程：index = 2*s*tid               —— 不发散了，但有 bank conflict
  v3 顺序寻址：s 从 blockDim/2 往下减半，tid < s 的干活   —— 无发散、无冲突
  v4 v3 + 先在寄存器里 grid-stride 累加很多元素再进 smem —— block 数少得多，算术强度高得多
练习 06-2 会把最后几步换成 warp shuffle，并用 atomicAdd 做成一遍完成。
"""
import torch

from common import bench, check, gbps, gpu_spec, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int THREADS = 256;

template <int VERSION>
__global__ void reduce_kernel(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
    __shared__ float sdata[THREADS];
    int tid = threadIdx.x;
    float v = 0.f;
    if (VERSION < 4) {
        int64_t i = (int64_t)blockIdx.x * THREADS + tid;
        v = (i < n) ? x[i] : 0.f;                              // 每个线程 1 个元素
    } else {
        for (int64_t i = (int64_t)blockIdx.x * THREADS + tid; i < n; i += (int64_t)gridDim.x * THREADS)
            v += x[i];                                         // 每个线程先在寄存器里累加很多个
    }
    sdata[tid] = v;
    __syncthreads();

    if (VERSION == 1) {
        for (int s = 1; s < THREADS; s *= 2) {
            if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];   // 活跃线程分散在每个 warp 里
            __syncthreads();
        }
    } else if (VERSION == 2) {
        for (int s = 1; s < THREADS; s *= 2) {
            int index = 2 * s * tid;                                // 活跃线程连续，但访问跨度 2s → bank conflict
            if (index < THREADS) sdata[index] += sdata[index + s];
            __syncthreads();
        }
    } else {
        for (int s = THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) sdata[tid] += sdata[tid + s];              // 连续线程读连续地址
            __syncthreads();
        }
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

torch::Tensor reduce_sum(torch::Tensor x, int64_t version) {
    CHECK_INPUT(x);
    auto stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor cur = x;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    while (true) {
        int64_t n = cur.numel();
        int64_t blocks = (n + THREADS - 1) / THREADS;
        if (version == 4) blocks = std::min<int64_t>(blocks, num_sms * 8);
        auto partial = torch::empty({blocks}, x.options());
        switch (version) {
            case 1: reduce_kernel<1><<<blocks, THREADS, 0, stream>>>(cur.data_ptr<float>(), partial.data_ptr<float>(), n); break;
            case 2: reduce_kernel<2><<<blocks, THREADS, 0, stream>>>(cur.data_ptr<float>(), partial.data_ptr<float>(), n); break;
            case 3: reduce_kernel<3><<<blocks, THREADS, 0, stream>>>(cur.data_ptr<float>(), partial.data_ptr<float>(), n); break;
            default: reduce_kernel<4><<<blocks, THREADS, 0, stream>>>(cur.data_ptr<float>(), partial.data_ptr<float>(), n); break;
        }
        CUDA_CHECK_LAUNCH();
        cur = partial;
        if (blocks == 1) break;
    }
    return cur.reshape({});
}
"""

if __name__ == "__main__":
    mod = load_cuda("reduction_smem", CUDA_SRC, ["reduce_sum"])
    torch.manual_seed(0)
    n = 1 << 25
    x = torch.rand(n, device="cuda")
    ref = x.double().sum().float()
    rows = []
    for v in [1, 2, 3, 4]:
        check(f"v{v}", mod.reduce_sum(x, v), ref, atol=0, rtol=1e-5)
        ms = bench(lambda: mod.reduce_sum(x, v))
        rows.append(dict(version=f"v{v}", us=ms * 1e3, GBps=gbps(n * 4, ms)))
    ms = bench(lambda: x.sum())
    rows.append(dict(version="torch.sum", us=ms * 1e3, GBps=gbps(n * 4, ms)))
    report(rows, f"sum of {n} fp32（只读，{gpu_spec().bw_note()}）")
