"""练习 06-2：warp shuffle 归约 + block 归约 + atomicAdd，一遍求和

目标：out = sum(x)，x 是 fp32，n 最大 3 千多万。只启动一次 kernel。
  - 每个线程先在寄存器里累加自己负责的元素（已写好：grid-stride + float4）
  - 你写：
      warp_reduce_sum   —— 用 __shfl_down_sync 在 warp 内归约，结果在 lane 0
      block_reduce_sum  —— warp 归约 → 每个 warp 的结果放 shared memory → 第 0 个 warp 再归约，结果在 0 号线程
      kernel 的第 2 步  —— 每个 block 的 0 号线程用 atomicAdd 把 block 的和加到 out 上

提示：
  - 讲义 6.6 节；对照 examples/reduction_smem.py 的 v4（它用 shared memory 树 + 两次 launch）
  - __shfl_down_sync(mask, v, offset)：返回本 warp 中 lane + offset 号线程的 v（越界时返回自己的）
  - out 在 host 里已经用 torch::zeros 清零了

做完之后想一想：
  - 测试最后会把同一个输入跑 10 次，结果不完全一样。为什么？训练里什么时候这会成为问题？
    要做到"确定性"该怎么改？（提示：两遍 launch；或者每个 block 写自己的部分和，最后固定顺序加）
  - 和 examples/reduction_smem.py 里的 v3/v4 比，你的版本省掉了哪些 __syncthreads？
  - 把 __shfl_down_sync 换成 __shfl_xor_sync（蝶形归约）有什么不同？结束后哪些 lane 持有总和？

运行：python 06_cuda_memory/exercises/ex2_reduce_sum.py
"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int THREADS = 256;

// warp 内归约：5 步 shuffle，结束后 lane 0 持有 32 个线程的总和
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // TODO: 用 __shfl_down_sync(0xffffffff, v, offset)，offset = 16, 8, 4, 2, 1
    return v;
}

// block 内归约：先 warp 内归约，每个 warp 的 lane 0 把结果写进 shared memory，
// 再由第 0 个 warp 把这些部分和归约一次。结束后 threadIdx.x == 0 持有整个 block 的总和
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];            // 最多 1024/32 = 32 个 warp
    // TODO:
    //   1. 每个 warp 先 warp_reduce_sum
    //   2. 每个 warp 的 lane 0 把结果写进 warp_sums[warp 编号]；__syncthreads()
    //   3. 第 0 个 warp 读出 warp_sums（不足 32 个 warp 的部分补 0），再 warp_reduce_sum 一次
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
    // TODO
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
