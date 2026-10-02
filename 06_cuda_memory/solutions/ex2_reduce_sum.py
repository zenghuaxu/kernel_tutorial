"""练习 06-2：warp shuffle 归约 + block 归约 + atomicAdd，一遍求和（参考答案）"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int THREADS = 256;

// warp 内归约：5 步 shuffle，结束后 lane 0 持有 32 个线程的总和
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// block 内归约：先 warp 内归约，每个 warp 的 lane 0 把结果写进 shared memory，
// 再由第 0 个 warp 把这些部分和归约一次。结束后 threadIdx.x == 0 持有整个 block 的总和
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];            // 最多 1024/32 = 32 个 warp
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[wid] = v;
    __syncthreads();
    int nwarps = blockDim.x / 32;
    v = (threadIdx.x < nwarps) ? warp_sums[lane] : 0.f;
    if (wid == 0) v = warp_reduce_sum(v);
    return v;
}

__global__ void sum_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    // 第 1 步（已写好）：grid-stride + float4，每个线程在寄存器里累加自己负责的元素
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float v = 0.f;
    for (int64_t i = tid; i < n4; i += stride) {
        float4 t = x4[i];
        v += (t.x + t.y) + (t.z + t.w);
    }
    int64_t tail = n4 * 4 + tid;
    if (tail < n) v += x[tail];

    // 第 2 步：block 内归约，再由每个 block 的 0 号线程把结果原子地加到 out 上
    v = block_reduce_sum(v);
    if (threadIdx.x == 0) atomicAdd(out, v);
}

torch::Tensor reduce_sum(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32);
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0, "需要 16 字节对齐");
    auto out = torch::zeros({}, x.options());      // atomicAdd 的目标，必须先清零
    int64_t n = x.numel();
    if (n == 0) return out;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t blocks = std::min<int64_t>((n / 4 + THREADS - 1) / THREADS + 1, (int64_t)num_sms * 4);
    sum_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex06_2_reduce", CUDA_SRC, ["reduce_sum"])
    torch.manual_seed(0)
    for n in [1, 3, 31, 1000, 4097, 1 << 20, (1 << 25) + 3]:
        x = torch.rand(n, device="cuda")                     # 全正数：相对误差有意义
        ref = x.double().sum().float()
        check(f"rand n={n}", mod.reduce_sum(x), ref, atol=1e-6, rtol=1e-5)
    x = torch.randn(1 << 22, device="cuda")                  # 有正有负：总和接近 0，用绝对误差
    check("randn n=2^22", mod.reduce_sum(x), x.double().sum().float(), atol=1e-2, rtol=0)

    n = 1 << 25
    x = torch.rand(n, device="cuda")
    results = {mod.reduce_sum(x).item() for _ in range(10)}
    print(f"  同一个输入跑 10 次，得到 {len(results)} 种不同的结果（atomicAdd 的顺序不确定 → 浮点结果不确定）")
    rows = []
    for name, fn in [("cuda shuffle+atomic", lambda: mod.reduce_sum(x)), ("torch.sum", lambda: x.sum())]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(n * 4, ms)))
    report(rows, f"sum of {n} fp32")
    finish()
