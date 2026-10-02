"""示例：shared memory 与 __syncthreads。

运行：python 06_cuda_memory/examples/smem_basics.py
抓数据竞争：
  .tools/compute-sanitizer/compute-sanitizer --tool racecheck --print-limit 2 python 06_cuda_memory/examples/smem_basics.py

任务：把数组按 256 个一段，每段内部倒序。每个线程把自己的元素写进 shared memory，
同步之后再读"对面"线程写的元素。少了 __syncthreads，就可能读到对面还没写进去的旧值。
"""
import torch

from common import check, load_cuda

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int BLOCK = 256;

template <bool SYNC>
__global__ void block_reverse_kernel(const float* __restrict__ x, float* __restrict__ out, int64_t n) {
    // 静态 shared memory：大小编译期确定，整个 block 共享，生命周期 = block
    __shared__ float buf[BLOCK];
    int64_t base = (int64_t)blockIdx.x * BLOCK;
    int t = threadIdx.x;
    buf[t] = x[base + t];                 // 1. 每个线程写自己那格
    if (SYNC) __syncthreads();            // 2. 屏障：block 内所有线程都写完了才继续
    out[base + t] = buf[BLOCK - 1 - t];   // 3. 读对面线程写的那格
}

torch::Tensor block_reverse(torch::Tensor x, bool sync) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.numel() % BLOCK == 0, "长度要是 256 的倍数（示例简化）");
    auto out = torch::empty_like(x);
    int64_t blocks = x.numel() / BLOCK;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (sync) block_reverse_kernel<true><<<blocks, BLOCK, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    else      block_reverse_kernel<false><<<blocks, BLOCK, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), x.numel());
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("smem_basics", CUDA_SRC, ["block_reverse"])
    x = torch.arange(256 * 4096, device="cuda", dtype=torch.float32)
    ref = x.view(-1, 256).flip(1).reshape(-1)
    check("有 __syncthreads", mod.block_reverse(x, True), ref, atol=0, rtol=0)
    wrong = sum(int((mod.block_reverse(x, False) != ref).sum().item()) for _ in range(20))
    print(f"  没有 __syncthreads：20 次运行里一共 {wrong} 个元素错了"
          f"（数量每次都不一样；某些情况下甚至可能是 0——竞争不一定触发，这正是它危险的地方；用 racecheck 查）")
