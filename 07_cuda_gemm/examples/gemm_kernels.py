"""示例：GEMM 从 naive 到 Tensor Core 的完整演进 + 和 cuBLAS 对比。

运行：
    python 07_cuda_gemm/examples/gemm_kernels.py            # 默认 4096^3
    python 07_cuda_gemm/examples/gemm_kernels.py 2048       # 换个尺寸

约定：全部是行主序（row-major），C[M,N] = A[M,K] @ B[K,N]。
SIMT 版本（kernel 1~6）是 fp32；Tensor Core 版本（kernel 7）是 bf16 输入、fp32 累加和输出。
为了让代码短一点，kernel 4~7 要求 M/N/K 是 tile 大小的整数倍（wrapper 里有 assert）。
"""
import sys

import torch

from common import bench, check, load_cuda, report, tflops

SRC = r"""
#include <mma.h>
using namespace nvcuda;

// ---------------------------------------------------------------------------
// kernel 1: naive。一个线程算 C 的一个元素。
// 故意把 threadIdx.x 映射到"行"：同一个 warp 的 32 个线程读 A 的 32 个不同行（地址相距 K），
// 写 C 时也相距 N —— 完全不合并。
// ---------------------------------------------------------------------------
__global__ void sgemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// kernel 2: 合并访存。只改了一件事：threadIdx.x 映射到"列"。
// 现在一个 warp 读 B 的同一行里连续 32 个 float（合并），读 A 时 32 个线程读同一个地址（广播）。
// ---------------------------------------------------------------------------
__global__ void sgemm_coalesced(int M, int N, int K, const float* A, const float* B, float* C) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// kernel 3: shared memory 分块。block 是 32x32 个线程，负责 C 的一个 32x32 tile。
// 每轮把 A 的 32x32 和 B 的 32x32 搬进 smem，块内 1024 个线程共享 -> 每个 global 元素被复用 32 次。
// ---------------------------------------------------------------------------
#define T3 32
__global__ void sgemm_smem(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[T3][T3];
    __shared__ float Bs[T3][T3];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * T3 + ty, col = blockIdx.x * T3 + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += T3) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < T3; ++kk) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// kernel 4: 1D 寄存器分块。BM=BN=64, BK=8，每个线程算同一列上的 TM=8 个结果。
// 内循环里一个 Bs 值（放寄存器）被复用 TM 次 -> smem 读取次数降为 1/TM 左右。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM>
__global__ void sgemm_1d_blocktile(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;          // 0 .. BM/TM-1
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;   // 512 线程正好铺满 64x8
    const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;   // 512 线程正好铺满 8x64
    float res[TM] = {0.f};
    for (int k0 = 0; k0 < K; k0 += BK) {
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            float b = Bs[dot * BN + threadCol];
            #pragma unroll
            for (int r = 0; r < TM; ++r) res[r] += As[(threadRow * TM + r) * BK + dot] * b;
        }
        __syncthreads();
    }
    for (int r = 0; r < TM; ++r) C[(threadRow * TM + r) * N + threadCol] = res[r];
}

// ---------------------------------------------------------------------------
// kernel 5: 2D 寄存器分块。BM=BN=128, BK=8，每个线程算 TMxTN=8x8 的小块（外积）。
// 每个 dot 步：从 smem 读 TM+TN=16 个数，做 TM*TN=64 次 FMA。
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_2d_blocktile(int M, int N, int K, const float* A, const float* B, float* C) {
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    constexpr int strideA = NUM_THREADS / BK;            // 一轮能搬 A 的多少行
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
    constexpr int strideB = NUM_THREADS / BN;
    float res[TM * TN] = {0.f};
    float regM[TM], regN[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int off = 0; off < BM; off += strideA)
            As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
        for (int off = 0; off < BK; off += strideB)
            Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + dot];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) res[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(threadRow * TM + i) * N + threadCol * TN + j] = res[i * TN + j];
}

// ---------------------------------------------------------------------------
// kernel 6: 在 kernel 5 基础上
//   (a) global -> smem 用 float4（128-bit）读写
//   (b) As 在 smem 里转置存放（As[k][m]），这样内循环读 regM 时也是连续地址，可以向量化
//   (c) 写回 C 用 float4
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_vectorized(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[BK * BM];   // 转置：As[k * BM + m]
    __shared__ float Bs[BK * BN];
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;
    // A tile 128x8 = 256 个 float4，256 线程每人一个
    const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
    // B tile 8x128 = 256 个 float4
    const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
    float res[TM * TN] = {0.f};
    float regM[TM], regN[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        float4 t = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
        As[(innerColA * 4 + 0) * BM + innerRowA] = t.x;
        As[(innerColA * 4 + 1) * BM + innerRowA] = t.y;
        As[(innerColA * 4 + 2) * BM + innerRowA] = t.z;
        As[(innerColA * 4 + 3) * BM + innerRowA] = t.w;
        reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
            reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
        __syncthreads();
        A += BK;
        B += BK * N;
        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regM[i] = As[dot * BM + threadRow * TM + i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regN[j] = Bs[dot * BN + threadCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) res[i * TN + j] += regM[i] * regN[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; j += 4) {
            float4 v = make_float4(res[i * TN + j], res[i * TN + j + 1], res[i * TN + j + 2], res[i * TN + j + 3]);
            reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j])[0] = v;
        }
}

// ---------------------------------------------------------------------------
// kernel 7: WMMA（Tensor Core），bf16 输入，fp32 累加。
// block tile 128x128x32，8 个 warp 排成 4x2，每个 warp 算 32x64 = 2x4 个 16x16 fragment。
// A/B tile 先用 16 字节向量 load 搬进 smem（行尾 pad 8 个元素，减少 bank conflict），
// 再用 load_matrix_sync 从 smem 装进 fragment。
// ---------------------------------------------------------------------------
#define WB_M 128
#define WB_N 128
#define WB_K 32
#define WPAD 8
__global__ void __launch_bounds__(256) gemm_wmma_smem(int M, int N, int K, const __nv_bfloat16* A,
                                                      const __nv_bfloat16* B, float* C) {
    __shared__ __align__(128) __nv_bfloat16 As[WB_M][WB_K + WPAD];
    __shared__ __align__(128) __nv_bfloat16 Bs[WB_K][WB_N + WPAD];
    const int warp = threadIdx.x / 32;
    const int wm = warp / 2, wn = warp % 2;
    const int bm = blockIdx.y * WB_M, bn = blockIdx.x * WB_N;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) wmma::fill_fragment(acc[i][j], 0.f);

    for (int k0 = 0; k0 < K; k0 += WB_K) {
        // A tile 128x32 bf16 = 512 个 uint4（每个 8 个 bf16），256 线程各搬 2 个
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = threadIdx.x + i * 256;
            int r = idx / (WB_K / 8), c = (idx % (WB_K / 8)) * 8;
            *reinterpret_cast<uint4*>(&As[r][c]) = *reinterpret_cast<const uint4*>(&A[(size_t)(bm + r) * K + k0 + c]);
        }
        // B tile 32x128 bf16 = 512 个 uint4
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = threadIdx.x + i * 256;
            int r = idx / (WB_N / 8), c = (idx % (WB_N / 8)) * 8;
            *reinterpret_cast<uint4*>(&Bs[r][c]) = *reinterpret_cast<const uint4*>(&B[(size_t)(k0 + r) * N + bn + c]);
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < WB_K; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[4];
            #pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], WB_K + WPAD);
            #pragma unroll
            for (int j = 0; j < 4; ++j) wmma::load_matrix_sync(b[j], &Bs[kk][wn * 64 + j * 16], WB_N + WPAD);
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                #pragma unroll
                for (int j = 0; j < 4; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float* cptr = C + (size_t)(bm + wm * 32 + i * 16) * N + bn + wn * 64 + j * 16;
            wmma::store_matrix_sync(cptr, acc[i][j], N, wmma::mem_row_major);
        }
}

// ============================ host 端 wrapper ============================
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor run_sgemm(torch::Tensor A, torch::Tensor B, int64_t version) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    auto C = torch::empty({M, N}, A.options());
    const float *a = A.data_ptr<float>(), *b = B.data_ptr<float>();
    float* c = C.data_ptr<float>();
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (version == 1) {
        sgemm_naive<<<dim3(cdiv(M, 32), cdiv(N, 32)), dim3(32, 32), 0, stream>>>(M, N, K, a, b, c);
    } else if (version == 2) {
        sgemm_coalesced<<<dim3(cdiv(N, 32), cdiv(M, 32)), dim3(32, 32), 0, stream>>>(M, N, K, a, b, c);
    } else if (version == 3) {
        sgemm_smem<<<dim3(cdiv(N, T3), cdiv(M, T3)), dim3(T3, T3), 0, stream>>>(M, N, K, a, b, c);
    } else if (version == 4) {
        TORCH_CHECK(M % 64 == 0 && N % 64 == 0 && K % 8 == 0, "kernel 4 要求 M,N%64==0, K%8==0");
        sgemm_1d_blocktile<64, 64, 8, 8><<<dim3(N / 64, M / 64), 64 * 64 / 8, 0, stream>>>(M, N, K, a, b, c);
    } else if (version == 5) {
        TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0, "kernel 5 要求 M,N%128==0, K%8==0");
        sgemm_2d_blocktile<128, 128, 8, 8, 8><<<dim3(N / 128, M / 128), 256, 0, stream>>>(M, N, K, a, b, c);
    } else if (version == 6) {
        TORCH_CHECK(M % 128 == 0 && N % 128 == 0 && K % 8 == 0, "kernel 6 要求 M,N%128==0, K%8==0");
        sgemm_vectorized<128, 128, 8, 8, 8><<<dim3(N / 128, M / 128), 256, 0, stream>>>(M, N, K, a, b, c);
    } else {
        TORCH_CHECK(false, "未知 version");
    }
    CUDA_CHECK_LAUNCH();
    return C;
}

torch::Tensor run_wmma(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(M % WB_M == 0 && N % WB_N == 0 && K % WB_K == 0, "WMMA kernel 要求 M,N%128==0, K%32==0");
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    gemm_wmma_smem<<<dim3(N / WB_N, M / WB_M), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        M, N, K, reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()), C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""

NAMES = {
    1: "1 naive",
    2: "2 coalesced",
    3: "3 smem tiling",
    4: "4 1D blocktile",
    5: "5 2D blocktile",
    6: "6 vectorized",
}

if __name__ == "__main__":
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    torch.backends.cuda.matmul.fp32_precision = "ieee"   # 关掉 TF32，让 cuBLAS fp32 做真正的 fp32
    mod = load_cuda("gemm_kernels", SRC, ["run_sgemm", "run_wmma"])

    torch.manual_seed(0)
    print("== 正确性（M=N=256, K=512）==")
    A = torch.randn(256, 512, device="cuda")
    B = torch.randn(512, 256, device="cuda")
    ref = A @ B
    for v, name in NAMES.items():
        check(name, mod.run_sgemm(A, B, v), ref, atol=1e-3, rtol=1e-3)
    Ab, Bb = A.bfloat16(), B.bfloat16()
    check("7 wmma bf16", mod.run_wmma(Ab, Bb), Ab.float() @ Bb.float(), atol=1e-2, rtol=1e-3)

    M = N = K = size
    flops = 2 * M * N * K
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    rows = []
    for v, name in NAMES.items():
        ms = bench(lambda: mod.run_sgemm(A, B, v))
        rows.append(dict(kernel=name, dtype="fp32", ms=ms, TFLOPs=tflops(flops, ms)))
    ms = bench(lambda: A @ B)
    rows.append(dict(kernel="cuBLAS fp32", dtype="fp32", ms=ms, TFLOPs=tflops(flops, ms)))
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    ms = bench(lambda: A @ B)
    rows.append(dict(kernel="cuBLAS tf32", dtype="tf32", ms=ms, TFLOPs=tflops(flops, ms)))
    Ab, Bb = A.bfloat16(), B.bfloat16()
    ms = bench(lambda: mod.run_wmma(Ab, Bb))
    rows.append(dict(kernel="7 wmma (smem)", dtype="bf16", ms=ms, TFLOPs=tflops(flops, ms)))
    ms = bench(lambda: Ab @ Bb)
    rows.append(dict(kernel="cuBLAS bf16", dtype="bf16", ms=ms, TFLOPs=tflops(flops, ms)))
    report(rows, f"GEMM {M}x{N}x{K}（H100 SXM 峰值：fp32 SIMT ~67、tf32 TC ~495、bf16 TC ~989 TFLOPs）")
