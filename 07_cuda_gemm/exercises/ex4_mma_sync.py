"""练习 07-4：手写 mma.sync.m16n8k16 + ldmatrix

不用 WMMA，直接写 PTX 的 mma.sync 指令。这是 CUTLASS、FlashAttention-2、Triton（在 Ampere 上）生成的代码真正在用的东西。
C[M,N](fp32) = A[M,K](bf16) @ B[K,N](bf16)，行主序；每个 warp 算 C 的一个 16x8 tile，沿 K 每次 16。

你要填（全部标了 TODO）：
  (1) a_frag_coord / b_frag_coord / c_frag_coord：fragment 布局 —— lane 的第 r 个寄存器对应矩阵里哪个位置
      （讲义 7.8 节的表；也可以先跑 examples/mma_sync_demo.py 看布局图）
  (2) 版本 A 里 B fragment 的打包：b[r] 由 B 里哪两个元素拼成？它们在内存里相距多远？
  (3) 版本 B 里 ldmatrix 的地址：每个 lane 要提供 smem 里哪一行的地址
      - A 用 ldmatrix.x4（4 个 8x8 矩阵，按 a0..a3 的顺序）
      - B 用 ldmatrix.x2.trans（2 个 8x8 矩阵，装载时转置）
  测试分别跑版本 A（manual）和版本 B（ldmatrix）。两个版本共用 (1)，所以先把 manual 做对。

提示：
  - g = lane / 4，t = lane % 4。A 的 a0 = A[g][2t, 2t+1]，C 的 c0,c1 = C[g][2t, 2t+1] —— 剩下的照着讲义推
  - ldmatrix：lane i 提供"第 i/8 个矩阵的第 i%8 行"的起始地址（.x2 只看 lane 0..15）；
    返回时每个 lane 拿到每个矩阵第 lane/4 行、第 2*(lane%4) 和 2*(lane%4)+1 列的两个元素
  - .trans 时，"行"变成了"列"：lane 拿到的是原矩阵第 lane/4 列上、第 2*(lane%4) 和 +1 行的两个元素

做完之后想一想：
  - 为什么 B 用 .trans？如果 B 在内存里本来就是列主序（[N, K] 行主序，比如 nn.Linear 的 weight），还需要 .trans 吗？
  - 一条 mma.sync.m16n8k16 是 16*8*16*2 = 4096 FLOP。H100 每个 SM 每周期能做多少？要多少个 warp 同时发 mma 才能喂饱？
    （单元 10：Hopper 上 mma.sync 已经不是最快的路径了，wgmma 一次是 64xNx16，而且是异步的；
     Blackwell 上换成了 tcgen05.mma，单线程发射、累加器在 Tensor Memory 里）

运行：python 07_cuda_gemm/exercises/ex4_mma_sync.py
"""
import torch

from common import check, finish, load_cuda

SRC = r"""
// bf16 在这里一律当成 uint16_t 处理（只搬运比特，不做运算），避免和 torch 的编译宏打架。

// ---------------- fragment 布局（PTX ISA: mma.m16n8k16, .bf16）----------------
// lane = 0..31，g = lane / 4（groupID），t = lane % 4（threadID_in_group）

// A 是 16x16（m x k）。寄存器 r = 0..3 各装 2 个 bf16：(row, col) 和 (row, col + 1)
__device__ __forceinline__ void a_frag_coord(int lane, int r, int& row, int& col) {
    int g = lane >> 2, t = lane & 3;
    // TODO (1)
    row = 0;
    col = 0;
}
// B 是 16x8（k x n）。寄存器 r = 0..1 各装 2 个 bf16：(k, n) 和 (k + 1, n)
__device__ __forceinline__ void b_frag_coord(int lane, int r, int& k, int& n) {
    int g = lane >> 2, t = lane & 3;
    // TODO (1)
    k = 0;
    n = 0;
}
// C/D 是 16x8（m x n）。c[i]，i = 0..3，各 1 个 fp32
__device__ __forceinline__ void c_frag_coord(int lane, int i, int& row, int& col) {
    int g = lane >> 2, t = lane & 3;
    // TODO (1)
    row = 0;
    col = 0;
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
            b[r] = 0;   // TODO (2): 用 pack2 把 (k, n) 和 (k + 1, n) 两个元素拼起来
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
        const uint16_t* a_row_ptr = &As[warp][0][0];   // TODO (3)
        // ldmatrix.x2.trans：两个 8x8 矩阵依次对应 b0、b1。b0 覆盖 B 的哪几行（k）？b1 呢？
        // .trans 让每个 lane 拿到"同一列 n、相邻两个 k"——正好是 B fragment 要的打包方式
        const uint16_t* b_row_ptr = &Bs[warp][0][0];   // TODO (3)

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
