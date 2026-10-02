"""练习 06-3：CUDA 按行 softmax，bf16 进出、fp32 计算

目标：out = softmax(x, dim=-1)，x 是 contiguous 的 [M, N] bf16。两种 kernel，host 按 N 自动选：

A. softmax_kernel —— 长行（N > 1024）：一个 block 处理一行
   三遍扫描：求 max → 求 sum(exp(x - max)) → 写 exp(x - max) / sum。
   block 内归约用已经给你的 block_allreduce（练习 06-2 的"所有线程都拿到结果"版本）。
B. softmax_warp_kernel<COLS> —— 短行（N <= 1024）：一个 warp 处理一行
   整行放进寄存器（每个 lane COLS 个），只读一遍；归约只需要 warp_allreduce，不用 shared memory、不用 __syncthreads。

已经给你的：MaxOp / SumOp、warp_allreduce、block_allreduce、host 函数（选 kernel、选线程数）。

提示：
  - 为什么要减 max：bf16 的 x*50 可能到几百，exp(300) 直接溢出成 inf。测试里有这个用例
  - 讲义 6.6 节解释了 xor 蝶形 all-reduce
  - 测试覆盖 N = 1, 7, 32, 33, 1000, 1024, 1025, 4096, 32000, 100000，以及 M 不是 4 的倍数（warp kernel 一个 block 4 行）

做完之后想一想：
  - 长行版本把一行读了三遍，为什么带宽没有掉到 1/3？（这一行在 L1/L2 里吗？32000 个 bf16 是多少字节？）
  - 如果先把 A 的第 1、2 步合并成一遍（online softmax：边扫边更新 max，并把旧的 sum 乘上 exp(旧max - 新max)），
    能少读一遍。单元 08 的 FlashAttention 就建立在这个技巧上
  - host 里 N >= 16384 时用 1024 个线程。改回固定 256 个线程，512x32768 那一行的时间会怎样？为什么？

运行：python 06_cuda_memory/exercises/ex3_softmax.py
"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cfloat>

struct MaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); } };
struct SumOp { __device__ __forceinline__ float operator()(float a, float b) const { return a + b; } };

// ---------- 已经写好的工具（练习 06-2 的"all-reduce"版本）----------
// warp 内 all-reduce：用 xor 蝶形交换，结束后 **32 个 lane 都**拿到结果
template <typename Op>
__device__ __forceinline__ float warp_allreduce(float v, Op op) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = op(v, __shfl_xor_sync(0xffffffff, v, offset));
    return v;
}

// block 内 all-reduce：结束后 block 里**所有线程**都拿到结果
template <typename Op>
__device__ __forceinline__ float block_allreduce(float v, Op op, float identity) {
    __shared__ float warp_vals[32];
    __shared__ float result;
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nwarps = blockDim.x / 32;
    v = warp_allreduce(v, op);
    __syncthreads();                 // 防止上一次调用还有线程在读 warp_vals / result
    if (lane == 0) warp_vals[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nwarps) ? warp_vals[lane] : identity;
        v = warp_allreduce(v, op);
        if (lane == 0) result = v;
    }
    __syncthreads();
    return result;
}

// ---------- 一个 block 处理一行 ----------
__global__ void softmax_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int N) {
    const __nv_bfloat16* row = x + (int64_t)blockIdx.x * N;
    __nv_bfloat16* orow = out + (int64_t)blockIdx.x * N;

    // TODO: 三遍扫描这一行（每个线程负责 j = threadIdx.x, +blockDim.x, ...）
    //   1. 行最大值 m：线程内 fmaxf，再 block_allreduce(m, MaxOp(), -FLT_MAX)
    //   2. 分母 s = sum(exp(x - m))：线程内累加，再 block_allreduce(s, SumOp(), 0.f)
    //   3. 写出 exp(x - m) / s
    //   读写 bf16 用 __bfloat162float / __float2bfloat16；exp 用 __expf
}

// ---------- 短行（N <= 1024）：一个 warp 处理一行，整行放在寄存器里，只读一次 HBM ----------
// 每个 lane 负责 j = lane, lane+32, lane+64, ... 共 COLS 个元素（COLS 是编译期常量，数组才能放进寄存器）
template <int COLS>
__global__ void softmax_warp_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                                    int M, int N) {
    int lane = threadIdx.x % 32;
    int64_t r = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // 本 warp 负责的行
    if (r >= M) return;                    // 整个 warp 一起退出，不影响后面的 shuffle
    const __nv_bfloat16* row = x + r * N;
    __nv_bfloat16* orow = out + r * N;

    // TODO:
    //   1. float v[COLS]：把 row[lane + k*32]（k = 0..COLS-1）读进寄存器，越界的位置填 -FLT_MAX；同时求 lane 内最大值
    //   2. warp_allreduce 求整行最大值 m
    //   3. v[k] = exp(v[k] - m)（越界位置填 0），累加，warp_allreduce 求和
    //   4. 写出 v[k] / sum（越界的不写）
    //   循环加 #pragma unroll，否则 v[] 可能被放到 local memory（显存）而不是寄存器
}

torch::Tensor softmax(torch::Tensor x) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16, "要求 2D bf16");
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty_like(x);
    if (M == 0 || N == 0) return out;
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (N <= 1024) {
        const int rows_per_block = 4;                       // 4 个 warp = 128 线程，一个 warp 一行
        int blocks = (M + rows_per_block - 1) / rows_per_block;
        int cols = (N + 31) / 32;
        if      (cols <= 1)  softmax_warp_kernel<1><<<blocks, 128, 0, stream>>>(xp, op, M, N);
        else if (cols <= 2)  softmax_warp_kernel<2><<<blocks, 128, 0, stream>>>(xp, op, M, N);
        else if (cols <= 4)  softmax_warp_kernel<4><<<blocks, 128, 0, stream>>>(xp, op, M, N);
        else if (cols <= 8)  softmax_warp_kernel<8><<<blocks, 128, 0, stream>>>(xp, op, M, N);
        else if (cols <= 16) softmax_warp_kernel<16><<<blocks, 128, 0, stream>>>(xp, op, M, N);
        else                 softmax_warp_kernel<32><<<blocks, 128, 0, stream>>>(xp, op, M, N);
    } else {
        // 长行：行越长，给一个 block 的线程越多（最多 1024），否则行数少时 SM 吃不饱
        int threads = N >= 16384 ? 1024 : (N >= 4096 ? 512 : 256);
        softmax_kernel<<<M, threads, 0, stream>>>(xp, op, N);
    }
    CUDA_CHECK_LAUNCH();
    return out;
}
"""


def ref_softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


if __name__ == "__main__":
    mod = load_cuda("ex06_3_softmax", CUDA_SRC, ["softmax"])
    torch.manual_seed(0)
    for M, N in [(1, 1), (3, 7), (1001, 32), (1001, 33), (128, 1000), (5, 1024), (7, 1025), (64, 4096),
                 (32, 32000), (4, 100_000)]:
        x = torch.randn(M, N, device="cuda").to(torch.bfloat16)
        check(f"{M}x{N}", mod.softmax(x), ref_softmax(x), atol=1e-5, rtol=1.6e-2)
    x = (torch.randn(16, 2048, device="cuda") * 50).to(torch.bfloat16)   # 大数值：不减 max 会 exp 溢出
    check("16x2048 大数值 (x*50)", mod.softmax(x), ref_softmax(x), atol=1e-5, rtol=1.6e-2)

    rows = []
    for M, N in [(4096, 4096), (512, 32768), (65536, 128)]:
        x = torch.randn(M, N, device="cuda").to(torch.bfloat16)
        for name, fn in [("cuda", lambda: mod.softmax(x)), ("torch", lambda: torch.softmax(x, -1))]:
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, us=ms * 1e3, GBps=gbps(2 * M * N * 2, ms)))
    report(rows, "softmax bf16（读 + 写各一遍）")
    finish()
