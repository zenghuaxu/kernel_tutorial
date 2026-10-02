"""练习 05-3：grid-stride 循环 + float4 向量化访存（参考答案）"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

// 标量版本（已写好，作为对照组，也作为"指针没对齐"时的回退路径）
__global__ void axpb_scalar_kernel(const float* __restrict__ x, float* __restrict__ out,
                                   int64_t n, float alpha, float beta) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        out[i] = x[i] * alpha + beta;
    }
}

// 向量化版本：每次读写一个 float4（16 字节）
__global__ void axpb_vec4_kernel(const float* __restrict__ x, float* __restrict__ out,
                                 int64_t n, float alpha, float beta) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n4 = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* out4 = reinterpret_cast<float4*>(out);
    // 主体：grid-stride 循环处理 n4 个 float4
    for (int64_t i = tid; i < n4; i += stride) {
        float4 v = x4[i];
        v.x = v.x * alpha + beta;
        v.y = v.y * alpha + beta;
        v.z = v.z * alpha + beta;
        v.w = v.w * alpha + beta;
        out4[i] = v;
    }
    // 尾巴：最后 n % 4 个元素，交给前几个线程逐个处理
    int64_t tail = n4 * 4 + tid;
    if (tail < n) out[tail] = x[tail] * alpha + beta;
}

torch::Tensor axpb(torch::Tensor x, double alpha, double beta, bool vectorized) {
    CHECK_CUDA(x);
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    // grid-stride：block 数不随 n 增长，封顶为"每个 SM 若干个 block"
    int64_t want = (n + threads - 1) / threads;
    int blocks = (int)std::min<int64_t>(want, (int64_t)num_sms * 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    // float4 load 要求地址 16 字节对齐。torch 新分配的 tensor 一定对齐，但切片视图（如 x[1:]）不一定
    bool aligned = reinterpret_cast<uintptr_t>(x.data_ptr<float>()) % 16 == 0 &&
                   reinterpret_cast<uintptr_t>(out.data_ptr<float>()) % 16 == 0;
    if (vectorized && aligned) {
        axpb_vec4_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), n,
                                                         (float)alpha, (float)beta);
    } else {
        axpb_scalar_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), out.data_ptr<float>(), n,
                                                           (float)alpha, (float)beta);
    }
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex05_3_vec", CUDA_SRC, ["axpb"])
    torch.manual_seed(0)
    for n in [1, 3, 4, 5, 1023, 1_000_003, 1 << 24, (1 << 24) + 3]:
        x = torch.randn(n, device="cuda")
        check(f"vec4 n={n}", mod.axpb(x, 2.0, -1.0, True), x * 2.0 - 1.0)
    base = torch.randn(4097, device="cuda")
    check("未对齐视图 x[1:]（走标量回退）", mod.axpb(base[1:], 2.0, 1.0, True), base[1:] * 2.0 + 1.0)
    check("scalar n=4097", mod.axpb(base, 2.0, 1.0, False), base * 2.0 + 1.0)

    rows = []
    for n in [1 << 20, 1 << 24, 1 << 27]:
        x = torch.randn(n, device="cuda")
        for name, fn in [("scalar", lambda: mod.axpb(x, 2.0, 1.0, False)),
                         ("float4", lambda: mod.axpb(x, 2.0, 1.0, True))]:
            ms = bench(fn)
            rows.append(dict(n=n, impl=name, us=ms * 1e3, GBps=gbps(2 * n * 4, ms)))
    report(rows, "axpb fp32（H100 HBM3 峰值约 3350 GB/s）")
    finish()
