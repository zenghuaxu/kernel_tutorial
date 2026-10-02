"""练习 05-1：CUDA vector add

目标：不看 examples/hello_cuda.py，自己写出：
  - kernel 体：每个线程算一个元素 out[i] = x[i] + y[i]，注意越界保护
  - host 函数里的 blocks（需要多少个 block）

提示：
  - 全局下标 = blockIdx.x * blockDim.x + threadIdx.x；n 可能超过 2^31 的话要用 int64_t 算
  - blocks = ceil(n / threads)，整数写法：(n + threads - 1) / threads
  - 测试里有 n=255/256/257 —— 正好卡在 block 边界上

做完之后想一想：
  - 和 Triton 的 vector add 比，"一个线程处理一个元素" 和 "一个 program 处理一个 BLOCK" 有什么对应关系？
    Triton 的 BLOCK=1024, num_warps=4 时，每个线程实际处理几个元素？
  - 把 threads 改成 32、1024、1025 分别会怎样？

运行：python 05_cuda_basics/exercises/ex1_vector_add.py   （第一次编译 30~60 秒）
"""
import torch

from common import check, finish, load_cuda

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                           float* __restrict__ out, int64_t n) {
    // TODO: 算出全局下标 i，越界保护，写 out[i]
}

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x); CHECK_INPUT(y);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32, "只支持 fp32");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape 不一致");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;
    int64_t blocks = 0;  // TODO: 需要多少个 block？
    TORCH_CHECK(blocks > 0, "blocks 还没算（TODO）");
    add_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex05_1_add", CUDA_SRC, ["add"])
    torch.manual_seed(0)
    for n in [1, 17, 255, 256, 257, 98432, 1 << 22]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        check(f"n={n}", mod.add(x, y), x + y)
    x = torch.randn(3, 5, 7, device="cuda")
    check("3D shape (3,5,7)", mod.add(x, x), 2 * x)
    finish()
