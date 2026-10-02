"""练习 05-2：二维 block / grid（参考答案）"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

// out[i][j] = x[i][j] * scale + bias[j]，x/out 是 contiguous 的 [M, N]
// 线程映射：threadIdx.x -> 列 j（内存连续的方向），threadIdx.y -> 行 i
__global__ void bias_scale_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                  float* __restrict__ out, int M, int N, float scale) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        int64_t idx = (int64_t)i * N + j;
        out[idx] = x[idx] * scale + bias[j];
    }
}

// 对照组（已写好）：把映射反过来，threadIdx.x -> 行 i。结果一样，但……跑一下看看带宽。
__global__ void bias_scale_swapped_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                          float* __restrict__ out, int M, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        int64_t idx = (int64_t)i * N + j;
        out[idx] = x[idx] * scale + bias[j];
    }
}

torch::Tensor bias_scale(torch::Tensor x, torch::Tensor bias, double scale) {
    CHECK_INPUT(x); CHECK_INPUT(bias);
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat32 && bias.size(0) == x.size(1));
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    dim3 block(32, 8);                                   // 256 个线程：x 方向 32，y 方向 8
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    TORCH_CHECK(grid.x > 0 && grid.y > 0, "grid 还没算（TODO）");
    bias_scale_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), M, N, (float)scale);
    CUDA_CHECK_LAUNCH();
    return out;
}

torch::Tensor bias_scale_swapped(torch::Tensor x, torch::Tensor bias, double scale) {
    CHECK_INPUT(x); CHECK_INPUT(bias);
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    dim3 block(32, 8);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    bias_scale_swapped_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), M, N, (float)scale);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex05_2_bias", CUDA_SRC, ["bias_scale", "bias_scale_swapped"])
    torch.manual_seed(0)
    for M, N in [(1, 1), (7, 33), (100, 300), (1000, 1024), (4097, 31)]:
        x = torch.randn(M, N, device="cuda")
        b = torch.randn(N, device="cuda")
        check(f"{M}x{N}", mod.bias_scale(x, b, 0.5), x * 0.5 + b)

    M = N = 8192
    x = torch.randn(M, N, device="cuda")
    b = torch.randn(N, device="cuda")
    check("swapped 对照组也是对的", mod.bias_scale_swapped(x, b, 0.5), x * 0.5 + b)
    rows = []
    for name, fn in [("threadIdx.x->列 (你的)", lambda: mod.bias_scale(x, b, 0.5)),
                     ("threadIdx.x->行 (对照)", lambda: mod.bias_scale_swapped(x, b, 0.5))]:
        ms = bench(fn)
        rows.append(dict(mapping=name, us=ms * 1e3, GBps=gbps(2 * M * N * 4, ms)))
    report(rows, "8192x8192 fp32")
    finish()
