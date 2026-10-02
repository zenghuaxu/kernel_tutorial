"""练习 05-2：二维 block / grid

目标：out[i][j] = x[i][j] * scale + bias[j]，x 是 contiguous 的 [M, N] fp32。
  - 用二维 block：dim3 block(32, 8)，threadIdx.x 对应列 j，threadIdx.y 对应行 i
  - 你要写：kernel 体（算 i、j、越界保护、地址）和 host 里的二维 grid

提示：
  - dim3 grid(gx, gy)：gx 管列方向（N），gy 管行方向（M）
  - 行主序下 (i, j) 的线性下标 = i * N + j；M*N 可能超过 int 范围，乘之前转 int64_t
  - 注意 gridDim.y / gridDim.z 上限只有 65535，gridDim.x 上限 2^31-1。所以"大的那一维"通常放 x

做完之后想一想：
  - 文件里还有一个已经写好的对照 kernel（threadIdx.x 对应行），结果相同，但带宽差了多少？
    一个 warp 是 threadIdx.x 连续的 32 个线程。两种映射下，一个 warp 的一次 load 分别访问几段连续内存？
    （单元 06 会正式讲"合并访存"）

运行：python 05_cuda_basics/exercises/ex2_bias_2d.py
"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

// out[i][j] = x[i][j] * scale + bias[j]，x/out 是 contiguous 的 [M, N]
// 线程映射：threadIdx.x -> 列 j（内存连续的方向），threadIdx.y -> 行 i
__global__ void bias_scale_kernel(const float* __restrict__ x, const float* __restrict__ bias,
                                  float* __restrict__ out, int M, int N, float scale) {
    // TODO
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
    dim3 grid(0, 0);  // TODO
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
