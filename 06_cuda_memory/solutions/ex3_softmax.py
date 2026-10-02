"""练习 06-3：CUDA 按行 softmax，bf16 进出、fp32 计算（参考答案）"""
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

    // 1. 行最大值（数值稳定：exp(x - max) 不会上溢）
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, __bfloat162float(row[j]));
    m = block_allreduce(m, MaxOp(), -FLT_MAX);

    // 2. 分母 sum(exp(x - max))
    float s = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) s += __expf(__bfloat162float(row[j]) - m);
    s = block_allreduce(s, SumOp(), 0.f);

    // 3. 写出（第三次读这一行：通常还在 L1/L2 里，不会再走 HBM）
    float inv = 1.f / s;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        orow[j] = __float2bfloat16(__expf(__bfloat162float(row[j]) - m) * inv);
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

    float v[COLS];
    float m = -FLT_MAX;
#pragma unroll
    for (int k = 0; k < COLS; ++k) {
        int j = lane + k * 32;
        v[k] = (j < N) ? __bfloat162float(row[j]) : -FLT_MAX;
        m = fmaxf(m, v[k]);
    }
    m = warp_allreduce(m, MaxOp());
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < COLS; ++k) {
        int j = lane + k * 32;
        v[k] = (j < N) ? __expf(v[k] - m) : 0.f;
        s += v[k];
    }
    s = warp_allreduce(s, SumOp());
    float inv = 1.f / s;
#pragma unroll
    for (int k = 0; k < COLS; ++k) {
        int j = lane + k * 32;
        if (j < N) orow[j] = __float2bfloat16(v[k] * inv);
    }
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
