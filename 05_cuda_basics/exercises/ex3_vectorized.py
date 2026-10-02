"""练习 05-3：grid-stride 循环 + float4 向量化访存

目标：out = alpha * x + beta（fp32），写出向量化版本的 kernel axpb_vec4_kernel：
  - 每个线程一次读写一个 float4（16 字节，4 个 float），而不是一个 float
  - grid-stride 循环：host 端的 block 数封顶在 SM 数 × 8，所以每个线程要循环处理多个 float4
  - n 不一定是 4 的倍数：最后 n % 4 个元素要单独处理（测试里有 n=1、3、5、2^24+3）

已经给你的：
  - 标量 grid-stride 版本 axpb_scalar_kernel（照着它写循环）
  - host 函数，包括"指针 16 字节对齐才走向量化路径，否则回退到标量"的判断

提示：
  - reinterpret_cast<const float4*>(x) 把指针当成 float4 数组；v.x / v.y / v.z / v.w 访问分量
  - 尾巴的下标从 (n / 4) * 4 开始；全局线程号 tid < n % 4 的线程各处理一个

做完之后想一想：
  - 最后打印 scalar 和 float4 两个版本的带宽。在这张卡上 fp32 的差距可能并不大（编译器 + 硬件已经
    把相邻线程的 4 字节访问合并成整条 cache line 了）。那什么情况下向量化收益会明显？
    （想想 bf16：每个线程一次只读 2 字节；以及每个元素要做很多整数下标计算的 kernel）
  - 为什么 x[1:] 这种视图不能直接走 float4？如果硬走会怎样？（试试把对齐检查去掉）
  - grid-stride 的 block 数封顶在 SM 数 × 8。改成 × 1、× 32 或者干脆不封顶，带宽有变化吗？

运行：python 05_cuda_basics/exercises/ex3_vectorized.py
"""
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
    // TODO:
    //   1. grid-stride 循环：把 x / out 当作 float4 数组，处理前 n/4 个 float4
    //   2. 尾巴：最后 n % 4 个元素逐个处理（只需要前 n % 4 个线程干活）
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
