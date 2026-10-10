"""练习 07-1：shared memory 分块 SGEMM

目标：写出讲义 7.3 节的 kernel 3。C[M,N] = A[M,K] @ B[K,N]，全部行主序 fp32。
  - block = 32x32 个线程，负责 C 的一个 32x32 tile；threadIdx.x 对应列（为了合并访存）
  - 沿 K 方向循环：每轮把 A 的 [32 x 32] 和 B 的 [32 x 32] 搬进 shared memory，然后做 32 步内积
  - **M、N、K 可以是任意值**（测试里有 100x77x130、1x1000x3 这种），越界的地方搬 0 进 smem
  - host 函数（grid/block 配置）已经写好

提示：
  - 每个线程每轮只搬 A、B 各 1 个元素：As[ty][tx] 和 Bs[ty][tx]
  - 两个 __syncthreads() 各防的是什么？少了哪一个会出错？
  - 最后写 C 时也要判断越界

做完之后想一想：
  - 和 kernel 2（coalesced naive）相比，每个 A 元素、B 元素分别要从 global memory 读多少次？
  - 内循环 acc += As[ty][kk] * Bs[kk][tx]：一个 warp 里 32 个线程读 As 是同一个地址（广播），读 Bs 是连续 32 个地址。
    每次 FMA 要读 2 次 smem —— 这就是 kernel 3 的新瓶颈，kernel 4/5 用寄存器分块解决（练习 2）。

运行：python 07_cuda_gemm/exercises/ex1_smem_tiled.py
"""
import torch

from common import bench, check, finish, gpu_name, load_cuda, report, tflops

SRC = r"""
#define TILE 32

// C[M,N] = A[M,K] @ B[K,N]，全部行主序 fp32。block = TILE x TILE 个线程，负责 C 的一个 TILE x TILE tile。
__global__ void sgemm_smem_kernel(int M, int N, int K, const float* A, const float* B, float* C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;   // 本线程负责的 C 元素
    const int col = blockIdx.x * TILE + tx;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // TODO: 1) 把 A[row, k0+tx] 搬到 As[ty][tx]，B[k0+ty, col] 搬到 Bs[ty][tx]（越界填 0）
        //       2) 同步
        //       3) acc += sum_kk As[ty][kk] * Bs[kk][tx]
        //       4) 同步
    }
    // TODO: 把 acc 写到 C[row, col]（注意越界）
}

torch::Tensor sgemm_smem(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    auto C = torch::empty({M, N}, A.options());
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_smem_kernel<<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(
        M, N, K, A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""

if __name__ == "__main__":
    torch.backends.cuda.matmul.fp32_precision = "ieee"   # 参考答案用真正的 fp32（不用 TF32）
    mod = load_cuda("ex07_1_smem", SRC, ["sgemm_smem"])
    torch.manual_seed(0)
    ok = True
    for M, N, K in [(32, 32, 32), (64, 96, 128), (100, 77, 130), (1, 1000, 3), (257, 129, 65), (512, 512, 1024)]:
        A = torch.randn(M, K, device="cuda")
        B = torch.randn(K, N, device="cuda")
        ok &= check(f"M={M} N={N} K={K}", mod.sgemm_smem(A, B), A @ B, atol=1e-3, rtol=1e-3)
    if not ok:
        print("\n正确性没通过，跳过性能测试。")
        finish()

    S = 4096
    A = torch.randn(S, S, device="cuda")
    B = torch.randn(S, S, device="cuda")
    rows = []
    for name, fn in [("smem tiled (yours)", lambda: mod.sgemm_smem(A, B)), ("cuBLAS fp32", lambda: A @ B)]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPs=tflops(2 * S**3, ms)))
    report(rows, f"SGEMM {S}^3（{gpu_name()}，数字有噪声）")
    finish()
