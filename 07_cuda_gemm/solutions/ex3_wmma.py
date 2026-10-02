"""练习 07-3：用 WMMA API 调用 Tensor Core（参考答案）"""
import torch

from common import bench, check, finish, load_cuda, report, tflops

SRC = r"""
#include <mma.h>
using namespace nvcuda;

// C[M,N](fp32) = A[M,K](bf16) @ B[K,N](bf16)，行主序。
// 每个 warp 算 C 的一个 16x16 tile；一个 block 4 个 warp 排成 2x2，覆盖 32x32。
// fragment 直接从 global memory 装（没有 smem 复用 —— 简单但慢，见"想一想"）。
__global__ void wmma_gemm_kernel(int M, int N, int K, const __nv_bfloat16* A, const __nv_bfloat16* B, float* C) {
    const int warp = threadIdx.x / 32;
    const int tile_m = (blockIdx.y * 2 + warp / 2) * 16;   // 本 warp 负责的 C tile 左上角
    const int tile_n = (blockIdx.x * 2 + warp % 2) * 16;
    if (tile_m >= M || tile_n >= N) return;                // 整个 warp 一起退出（WMMA 要求 warp 内一致）

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        // A 的 16x16 子块从 (tile_m, k0) 开始，行距（leading dimension）是 K
        wmma::load_matrix_sync(a_frag, A + (size_t)tile_m * K + k0, K);
        // B 的 16x16 子块从 (k0, tile_n) 开始，行距是 N
        wmma::load_matrix_sync(b_frag, B + (size_t)k0 * N + tile_n, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);    // c += a @ b
    }
    wmma::store_matrix_sync(C + (size_t)tile_m * N + tile_n, c_frag, N, wmma::mem_row_major);
}

torch::Tensor wmma_gemm(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "要求 M、N、K 都是 16 的倍数");
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    wmma_gemm_kernel<<<grid, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(
        M, N, K, reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B.data_ptr<at::BFloat16>()), C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex07_3_wmma", SRC, ["wmma_gemm"])
    torch.manual_seed(0)
    ok = True
    for M, N, K in [(16, 16, 16), (48, 80, 32), (16, 128, 1024), (256, 512, 1024), (1008, 496, 2048)]:
        A = torch.randn(M, K, device="cuda").bfloat16()
        B = torch.randn(K, N, device="cuda").bfloat16()
        ref = A.float() @ B.float()          # bf16 -> fp32 是精确的，只差累加顺序
        ok &= check(f"M={M} N={N} K={K}", mod.wmma_gemm(A, B), ref, atol=1e-2 * (K / 1024) ** 0.5 + 1e-3, rtol=1e-3)
    if not ok:
        print("\n正确性没通过，跳过性能测试。")
        finish()

    S = 4096
    A = torch.randn(S, S, device="cuda").bfloat16()
    B = torch.randn(S, S, device="cuda").bfloat16()
    rows = []
    for name, fn in [("wmma, 从 global 装 (yours)", lambda: mod.wmma_gemm(A, B)), ("cuBLAS bf16", lambda: A @ B)]:
        ms = bench(fn)
        rows.append(dict(impl=name, ms=ms, TFLOPs=tflops(2 * S**3, ms)))
    report(rows, f"bf16 GEMM {S}^3（共享 H100，数字有噪声）")
    finish()
