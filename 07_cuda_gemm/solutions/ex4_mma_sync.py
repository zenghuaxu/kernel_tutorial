"""练习 07-4：手写 mma.sync.m16n8k16 + ldmatrix（参考答案）"""
import torch

from common import check, finish, load_cuda

SRC = r"""
// bf16 在这里一律当成 uint16_t 处理（只搬运比特，不做运算），避免和 torch 的编译宏打架。

// ---------------- fragment 布局（PTX ISA: mma.m16n8k16, .bf16）----------------
// lane = 0..31，g = lane / 4（groupID），t = lane % 4（threadID_in_group）

// A 是 16x16（m x k）。寄存器 r = 0..3 各装 2 个 bf16：(row, col) 和 (row, col + 1)
__device__ __forceinline__ void a_frag_coord(int lane, int r, int& row, int& col) {
    int g = lane >> 2, t = lane & 3;
    row = g + ((r & 1) ? 8 : 0);
    col = 2 * t + ((r & 2) ? 8 : 0);
}
// B 是 16x8（k x n）。寄存器 r = 0..1 各装 2 个 bf16：(k, n) 和 (k + 1, n)
__device__ __forceinline__ void b_frag_coord(int lane, int r, int& k, int& n) {
    int g = lane >> 2, t = lane & 3;
    k = 2 * t + (r ? 8 : 0);
    n = g;
}
// C/D 是 16x8（m x n）。c[i]，i = 0..3，各 1 个 fp32
__device__ __forceinline__ void c_frag_coord(int lane, int i, int& row, int& col) {
    int g = lane >> 2, t = lane & 3;
    row = g + (i >= 2 ? 8 : 0);
    col = 2 * t + (i & 1);
}

__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
    return (uint32_t)lo | ((uint32_t)hi << 16);     // 下标小的元素放低 16 位
}

__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 每个 warp 算 C 的一个 16x8 tile；一个 block 4 个 warp 排成 2x2 -> block 覆盖 32x16
#define WARP_TILE(tile_m, tile_n)                                   \
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;     \
    const int tile_m = (blockIdx.y * 2 + warp / 2) * 16;            \
    const int tile_n = (blockIdx.x * 2 + warp % 2) * 8;             \
    if (tile_m >= M || tile_n >= N) return;

// ---------------- 版本 A：fragment 直接从 global 装 ----------------
__global__ void mma_manual_kernel(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C) {
    WARP_TILE(tile_m, tile_n)
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 16) {
        uint32_t a[4], b[2];
        for (int r = 0; r < 4; ++r) {
            int row, col;
            a_frag_coord(lane, r, row, col);
            const uint16_t* p = A + (size_t)(tile_m + row) * K + k0 + col;
            a[r] = pack2(p[0], p[1]);
        }
        for (int r = 0; r < 2; ++r) {
            int k, n;
            b_frag_coord(lane, r, k, n);
            const uint16_t* p = B + (size_t)(k0 + k) * N + tile_n + n;
            b[r] = pack2(p[0], p[N]);
        }
        mma_16816(c, a, b);
    }
    for (int i = 0; i < 4; ++i) {
        int row, col;
        c_frag_coord(lane, i, row, col);
        C[(size_t)(tile_m + row) * N + tile_n + col] = c[i];
    }
}

// ---------------- 版本 B：先搬进 smem，再用 ldmatrix 装 fragment ----------------
__global__ void mma_ldmatrix_kernel(int M, int N, int K, const uint16_t* A, const uint16_t* B, float* C) {
    __shared__ __align__(16) uint16_t As[4][16][16];   // 每个 warp 一块，互不干扰
    __shared__ __align__(16) uint16_t Bs[4][16][8];
    WARP_TILE(tile_m, tile_n)
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 16) {
        // 搬运：A 子块 16x16 = 32 个 16 字节，B 子块 16x8 = 32 个 8 字节，每个 lane 各搬一份
        {
            int r = lane / 2, cc = (lane % 2) * 8;
            *reinterpret_cast<uint4*>(&As[warp][r][cc]) =
                *reinterpret_cast<const uint4*>(A + (size_t)(tile_m + r) * K + k0 + cc);
            cc = (lane % 2) * 4;
            *reinterpret_cast<uint2*>(&Bs[warp][r][cc]) =
                *reinterpret_cast<const uint2*>(B + (size_t)(k0 + r) * N + tile_n + cc);
        }
        __syncwarp();

        uint32_t a[4], b[2];
        // ldmatrix.x4：lane i 提供第 (i/8) 个 8x8 矩阵第 (i%8) 行的地址。
        // 4 个矩阵对应 a0..a3 = (行0-7,列0-7) (行8-15,列0-7) (行0-7,列8-15) (行8-15,列8-15)
        const uint16_t* a_row_ptr = &As[warp][lane % 16][(lane / 16) * 8];
        // ldmatrix.x2.trans：两个矩阵 = B 的 k 0-7 行和 k 8-15 行；lane 0..15 提供第 lane 行（k = lane）。
        // .trans 让每个 lane 拿到"同一列 n、相邻两个 k"——正好是 B fragment 要的打包方式
        const uint16_t* b_row_ptr = &Bs[warp][lane % 16][0];

        uint32_t a_addr = __cvta_generic_to_shared(a_row_ptr);
        uint32_t b_addr = __cvta_generic_to_shared(b_row_ptr);
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                     : "=r"(b[0]), "=r"(b[1]) : "r"(b_addr));
        mma_16816(c, a, b);
        __syncwarp();          // 下一轮要覆盖 smem，先确保本轮所有 lane 都读完
    }
    for (int i = 0; i < 4; ++i) {
        int row, col;
        c_frag_coord(lane, i, row, col);
        C[(size_t)(tile_m + row) * N + tile_n + col] = c[i];
    }
}

torch::Tensor mma_gemm(torch::Tensor A, torch::Tensor B, bool use_ldmatrix) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16 && B.scalar_type() == torch::kBFloat16);
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K);
    TORCH_CHECK(M % 16 == 0 && N % 8 == 0 && K % 16 == 0, "要求 M%16==0, N%8==0, K%16==0");
    auto C = torch::empty({M, N}, A.options().dtype(torch::kFloat32));
    dim3 grid((N + 15) / 16, (M + 31) / 32);
    auto a = reinterpret_cast<const uint16_t*>(A.data_ptr<at::BFloat16>());
    auto b = reinterpret_cast<const uint16_t*>(B.data_ptr<at::BFloat16>());
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (use_ldmatrix)
        mma_ldmatrix_kernel<<<grid, 128, 0, stream>>>(M, N, K, a, b, C.data_ptr<float>());
    else
        mma_manual_kernel<<<grid, 128, 0, stream>>>(M, N, K, a, b, C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""

if __name__ == "__main__":
    mod = load_cuda("ex07_4_mma", SRC, ["mma_gemm"])
    torch.manual_seed(0)
    for use_ldm, tag in [(False, "manual"), (True, "ldmatrix")]:
        for M, N, K in [(16, 8, 16), (16, 8, 64), (48, 24, 32), (128, 256, 512), (1008, 520, 1024)]:
            A = torch.randn(M, K, device="cuda").bfloat16()
            B = torch.randn(K, N, device="cuda").bfloat16()
            ref = A.float() @ B.float()
            check(f"{tag:8s} M={M} N={N} K={K}", mod.mma_gemm(A, B, use_ldm), ref, atol=1e-2, rtol=1e-3)
    finish()
