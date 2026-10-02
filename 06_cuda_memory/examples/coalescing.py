"""示例：合并访存（coalescing）。同样读 64 MB 有用数据，换个访问模式，带宽能差一个数量级。

运行：python 06_cuda_memory/examples/coalescing.py

kernel：out[i] = x[i * stride]（i < n）。stride=1 时一个 warp 读连续的 128 字节；
stride=s 时一个 warp 的 32 个地址相距 4s 字节，要跨好多条 cache line / 32 字节 sector，
大部分搬进来的字节都被浪费了。
"""
import torch

from common import bench, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__global__ void strided_gather_kernel(const float* __restrict__ x, float* __restrict__ out,
                                      int64_t n, int64_t stride) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i * stride];
}

// 地址整体偏移 offset 个 float：访问仍然连续，只是不再和 128 字节边界对齐
__global__ void offset_copy_kernel(const float* __restrict__ x, float* __restrict__ out,
                                   int64_t n, int64_t offset) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i + offset];
}

torch::Tensor strided_gather(torch::Tensor x, int64_t n, int64_t stride) {
    TORCH_CHECK(x.numel() >= (n - 1) * stride + 1);
    auto out = torch::empty({n}, x.options());
    strided_gather_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n, stride);
    CUDA_CHECK_LAUNCH();
    return out;
}

torch::Tensor offset_copy(torch::Tensor x, int64_t n, int64_t offset) {
    TORCH_CHECK(x.numel() >= n + offset);
    auto out = torch::empty({n}, x.options());
    offset_copy_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n, offset);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("coalescing_demo", CUDA_SRC, ["strided_gather", "offset_copy"])
    n = 1 << 23                       # 每次都读 n 个有用的 float = 32 MB，写 32 MB
    max_stride = 32
    x = torch.randn(n * max_stride, device="cuda")   # 1 GB 源数据，stride=32 时也够用
    rows = []
    for stride in [1, 2, 4, 8, 16, 32]:
        out = mod.strided_gather(x, n, stride)
        assert torch.equal(out, x[::stride][:n])
        ms = bench(lambda: mod.strided_gather(x, n, stride))
        # 实际从 HBM 搬进来的量：stride>=8 时每个 float 都独占一个 32 字节 sector
        rows.append(dict(stride=stride, us=ms * 1e3, useful_GBps=gbps(2 * n * 4, ms),
                         sectors_per_warp=min(32, max(4, 4 * stride))))
    report(rows, "out[i] = x[i*stride]：有效带宽（只算有用的字节）")

    rows = []
    for offset in [0, 1, 8, 32]:
        assert torch.equal(mod.offset_copy(x, n, offset), x[offset:offset + n])
        ms = bench(lambda: mod.offset_copy(x, n, offset))
        rows.append(dict(offset_floats=offset, us=ms * 1e3, GBps=gbps(2 * n * 4, ms)))
    report(rows, "out[i] = x[i+offset]：未对齐但连续")
    print("\n结论：连续性是第一位的；首地址不对齐只多碰一条 cache line，影响很小（L2 也会帮忙）。")
