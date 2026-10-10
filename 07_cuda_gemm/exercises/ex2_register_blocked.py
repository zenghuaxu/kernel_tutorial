"""练习 07-2：2D 寄存器分块 SGEMM

目标：写出讲义 7.4 节的 kernel 5。block tile 128x128，K 方向每轮 BK=8，每个线程算 8x8 = 64 个结果。
  - 线程映射、指针偏移、host 函数都写好了，你要填三段：
      (1) 把 A 的 [128 x 8] tile、B 的 [8 x 128] tile 搬进 smem（每个线程要搬好几次，用循环）
      (2) 内积：对每个 dot = 0..BK-1，先把需要的 TM 个 A 值、TN 个 B 值读进寄存器 regM / regN，
          再做 TM x TN 的外积累加到 res
      (3) 把 res 写回 C
  - 只要求 M、N 是 128 的倍数、K 是 8 的倍数（host 函数里有检查）

提示：
  - smem 布局：As[m * BK + k]，Bs[k * BN + n]
  - 搬 A：线程 (innerRowA, innerColA) 搬第 innerRowA + off 行、第 innerColA 列，off = 0, strideA, 2*strideA, ...
  - 本线程负责 C tile 的行 threadRow*TM + i（i < TM）、列 threadCol*TN + j（j < TN）
  - 别忘了每轮结束后把 A 指针右移 BK 列、B 指针下移 BK 行（模板里已经写了，看看在哪）

做完之后想一想：
  - 每个 dot 步，一个线程读 smem 多少次、做多少次 FMA？和练习 1 比，"FMA / smem 读取"的比值提高了几倍？
  - 把 TM、TN 改成 4 会怎样？改成 16 呢？（给 load_cuda 加 extra_cuda_cflags=["-Xptxas=-v"], verbose=True，编译输出里会有寄存器数和 spill）
  - 这个 kernel 离 cuBLAS fp32 还差多少？examples/gemm_kernels.py 里的 kernel 6 又做了什么？

运行：python 07_cuda_gemm/exercises/ex2_register_blocked.py
"""
import torch

from common import bench, check, finish, gpu_name, load_cuda, report, tflops

SRC = r"""
// block tile BM x BN，K 方向每轮 BK；每个线程算 TM x TN 个结果。
// 线程数 = (BM/TM) * (BN/TN) = 16 * 16 = 256。
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
#define NUM_THREADS ((BM / TM) * (BN / TN))

__global__ void __launch_bounds__(NUM_THREADS)
sgemm_2d_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[BM * BK];      // As[m * BK + k]
    __shared__ float Bs[BK * BN];      // Bs[k * BN + n]

    // 本线程在 block tile 里负责的 TM x TN 小块：行 threadRow*TM ..，列 threadCol*TN ..
    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    // 把指针挪到本 block 的起点
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += blockIdx.y * BM * N + blockIdx.x * BN;

    // 搬运时的线程映射：256 个线程一次搬 A 的 (256/BK)=32 行 x BK 列，要搬 BM/32=4 次；
    //                   一次搬 B 的 (256/BN)=2 行 x BN 列，要搬 BK/2=4 次。相邻线程搬相邻列 -> 合并访存
    const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
    const int strideA = NUM_THREADS / BK;
    const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;
    const int strideB = NUM_THREADS / BN;

    float res[TM * TN] = {0.f};
    float regM[TM], regN[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // TODO (1): 把 A、B 的 tile 搬进 As、Bs
        __syncthreads();
        A += BK;
        B += BK * N;

        #pragma unroll
        for (int dot = 0; dot < BK; ++dot) {
            // TODO (2): 读 regM[0..TM)、regN[0..TN)，然后 res[i * TN + j] += regM[i] * regN[j]
        }
        __syncthreads();
    }

    // TODO (3): 把 res 写回 C
}

torch::Tensor sgemm_2d(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(M % BM == 0 && N % BN == 0 && K % BK == 0, "要求 M、N 是 128 的倍数，K 是 8 的倍数");
    auto C = torch::empty({M, N}, A.options());
    dim3 grid(N / BN, M / BM);
    sgemm_2d_kernel<<<grid, NUM_THREADS, 0, c10::cuda::getCurrentCUDAStream()>>>(
        M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""

if __name__ == "__main__":
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    mod = load_cuda("ex07_2_reg2d", SRC, ["sgemm_2d"])
    torch.manual_seed(0)
    ok = True
    for M, N, K in [(128, 128, 8), (256, 384, 72), (128, 256, 1000), (1024, 1024, 1024)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ok &= check(f"M={M} N={N} K={K}", mod.sgemm_2d(A, B), A @ B, atol=1e-3, rtol=1e-3)
    if not ok:
        print("\n正确性没通过，跳过性能测试。")
        finish()

    S = 4096
    A = torch.randn(S, S, device="cuda")
    B = torch.randn(S, S, device="cuda")
    rows = []
    for name, fn in [("2D blocktile (yours)", lambda: mod.sgemm_2d(A, B)), ("cuBLAS fp32", lambda: A @ B)]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPs=tflops(2 * S**3, ms)))
    report(rows, f"SGEMM {S}^3（{gpu_name()}，数字有噪声）")
    finish()
