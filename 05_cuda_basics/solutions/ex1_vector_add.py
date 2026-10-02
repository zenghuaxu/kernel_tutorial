"""练习 05-1：CUDA vector add（参考答案）"""
import torch

from common import check, finish, load_cuda

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__global__ void add_kernel(const float* __restrict__ x, const float* __restrict__ y,
                           float* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x); CHECK_INPUT(y);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 && y.scalar_type() == torch::kFloat32, "只支持 fp32");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape 不一致");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
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
