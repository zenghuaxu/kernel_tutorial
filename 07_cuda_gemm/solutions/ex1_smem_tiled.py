"""练习 07-1：shared memory 分块 SGEMM（参考答案）"""
import torch

from common import bench, check, finish, load_cuda, report, tflops

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
        // 每个线程搬 A、B 各一个元素；越界的位置填 0，这样后面的内积不用再判断
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
        __syncthreads();                       // 等所有人都搬完再读
        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();                       // 等所有人都读完再进入下一轮覆盖 smem
    }
    if (row < M && col < N) C[row * N + col] = acc;
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
    report(rows, f"SGEMM {S}^3（共享 H100，数字有噪声）")
    finish()
