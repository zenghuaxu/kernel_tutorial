"""示例：shared memory 的 bank conflict，用 clock64() 直接数周期。

运行：python 06_cuda_memory/examples/bank_conflicts.py

shared memory 分成 32 个 bank，每个 bank 宽 4 字节，地址 a（字节）落在 bank (a / 4) % 32。
一个 warp 的一次访问里，如果有 k 个线程访问**同一个 bank 的不同地址**，就要分 k 次完成（k-way conflict）。
（访问同一个地址不算冲突，是广播。）

kernel：线程 t 反复读 smem[(t * stride) % SIZE]（指针追逐，每次读依赖上一次），测平均每次读花多少个周期。
"""
import torch

from common import load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int SIZE = 32 * 33;   // 够放 stride=33 时 32 个线程的地址
constexpr int ITERS = 1024;

__global__ void bank_kernel(int stride, long long* cycles, int* sink) {
    // 指针追逐（pointer chasing）：smem[i] 里存的是"下一个要读的下标"。这里存的就是 i 自己，
    // 所以每个线程一直读同一个地址；但编译器不知道，只能每次老老实实发一条 load，
    // 而且下一次 load 依赖上一次的结果 —— 正好测出每次访问的延迟。
    __shared__ int smem[SIZE];
    for (int i = threadIdx.x; i < SIZE; i += blockDim.x) smem[i] = i;
    __syncthreads();
    int idx = (threadIdx.x * stride) % SIZE;
    long long t0 = clock64();
#pragma unroll 16
    for (int it = 0; it < ITERS; ++it) {
        idx = smem[idx];
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) cycles[blockIdx.x] = t1 - t0;
    sink[threadIdx.x] = idx;        // 防止整个循环被优化掉
}

double measure(int64_t stride) {
    auto cycles = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto sink = torch::zeros({32}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    // 只启动 1 个 block × 1 个 warp，排除其他 warp 的干扰
    bank_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>((int)stride, (long long*)cycles.data_ptr<int64_t>(),
                                                                sink.data_ptr<int>());
    CUDA_CHECK_LAUNCH();
    return (double)cycles.item<int64_t>() / ITERS;
}
"""

if __name__ == "__main__":
    mod = load_cuda("bank_conflicts", CUDA_SRC, ["measure"])
    mod.measure(1)   # 预热
    rows = []
    for stride in [0, 1, 2, 4, 8, 16, 32, 33]:
        banks = {(t * stride) % (32 * 33) % 32 for t in range(32)}
        addrs = {(t * stride) % (32 * 33) for t in range(32)}
        way = max(sum(1 for a in addrs if a % 32 == b) for b in banks)
        rows.append(dict(stride=stride, distinct_banks=len(banks), conflict_way=way,
                         cycles_per_read=min(mod.measure(stride) for _ in range(5))))
    report(rows, "一个 warp 读 shared memory：stride 与 bank conflict（单 warp，clock64 计时）")
    print("\n看点：stride=32 时 32 个线程全落在 bank 0 → 32-way conflict；stride=33（\"padding +1\"）又变回无冲突。")
    print("stride=0 是 32 个线程读同一个地址 → 广播，不冲突。")
    print("这里测的是**单个 warp 的延迟**：冲突每多一路，延迟多一点（k-way 冲突 = 硬件要串行做 k 次访问）。")
    print("真实 kernel 里很多 warp 同时抢 shared memory 带宽，k-way 冲突意味着 shared memory 吞吐直接降到 1/k，代价比这里看到的大得多。")
