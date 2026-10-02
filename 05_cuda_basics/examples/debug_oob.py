"""示例：越界写——CUDA 里最常见、也最阴险的 bug。

一个 kernel 忘了写 `if (i < n)`。它不会崩溃，而是**悄悄改写了隔壁 tensor 的数据**。

运行：
  1) 直接跑，看静默的数据损坏：
       python 05_cuda_basics/examples/debug_oob.py
  2) 用 compute-sanitizer 抓出来（要关掉 PyTorch 的显存缓存池，否则越界写落在池子内部，工具看不见）：
       PYTORCH_NO_CUDA_MEMORY_CACHING=1 .tools/compute-sanitizer/compute-sanitizer --print-limit 1 \\
           python 05_cuda_basics/examples/debug_oob.py
     会报 "Invalid __global__ write of size 4 bytes ... at fill_ones_buggy(float *, long)+0xb0 in cuda.cu:21"，
     行号能对上是因为 load_cuda 默认加了 -lineinfo。
"""
import torch

from common import load_cuda

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__global__ void fill_ones_buggy(float* x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = 1.f;            // BUG：忘了 if (i < n)
}

torch::Tensor fill_ones(torch::Tensor x) {
    const int threads = 256;
    int64_t blocks = (x.numel() + threads - 1) / threads;
    fill_ones_buggy<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(), x.numel());
    CUDA_CHECK_LAUNCH();   // 抓不到这种错误：launch 本身是合法的
    return x;
}
"""

if __name__ == "__main__":
    mod = load_cuda("debug_oob", CUDA_SRC, ["fill_ones"])
    a = torch.zeros(100, device="cuda")    # 400 字节；caching allocator 把它放进 512 字节的槽
    b = torch.zeros(100, device="cuda")    # 很可能紧挨着 a，在 512 字节之后
    print(f"a 的地址 {a.data_ptr():#x}，b 的地址 {b.data_ptr():#x}，相差 {b.data_ptr() - a.data_ptr()} 字节")
    mod.fill_ones(a)                        # 启动 1 个 block × 256 线程，写了 a[0..255]（1024 字节）
    torch.cuda.synchronize()
    print(f"a.sum() = {a.sum().item():.0f}（期望 100）")
    print(f"b.sum() = {b.sum().item():.0f}（期望 0！不是 0 就说明 b 被 a 的越界写改坏了）")
    print("没有任何报错。这就是为什么 mask / 边界检查必须写，而且要用 compute-sanitizer 查。")
