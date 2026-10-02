"""示例：mma.sync.m16n8k16 的 fragment 布局，一个 warp 算一个 16x8 的 tile。

运行：python 07_cuda_gemm/examples/mma_sync_demo.py

做两件事：
  1. 画出 fragment 布局：A(16x16)、B(16x8)、C(16x8) 的每个元素分别在哪个 lane 的寄存器里
     （kernel 让每个 lane 把自己的 lane id 写到它持有的元素位置上）
  2. 按这个布局从 global 取 fragment，执行一次 mma.sync，和 torch 对比
"""
import torch

from common import check, finish, load_cuda

SRC = r"""
// 一个 warp：每个 lane 按 PTX ISA 里 m16n8k16 的 fragment 布局，把自己的 lane id 写到它持有的位置上
__global__ void fragment_map(int* amap, int* bmap, int* cmap) {
    int lane = threadIdx.x;
    int g = lane >> 2;      // groupID：0..7
    int t = lane & 3;       // threadID_in_group：0..3
    // A：4 个 32-bit 寄存器，每个装 2 个 bf16
    //   a0 = A[g   ][2t, 2t+1]   a1 = A[g+8][2t, 2t+1]
    //   a2 = A[g   ][2t+8, 2t+9] a3 = A[g+8][2t+8, 2t+9]
    for (int r = 0; r < 4; ++r) {
        int row = g + ((r & 1) ? 8 : 0);
        int col = 2 * t + ((r & 2) ? 8 : 0);
        amap[row * 16 + col] = lane;
        amap[row * 16 + col + 1] = lane;
    }
    // B（逻辑上 16x8，k 行 n 列）：2 个寄存器，每个装 k 方向上相邻的 2 个元素
    //   b0 = B[2t, 2t+1][g]   b1 = B[2t+8, 2t+9][g]
    for (int r = 0; r < 2; ++r) {
        int k = 2 * t + (r ? 8 : 0);
        bmap[k * 8 + g] = lane;
        bmap[(k + 1) * 8 + g] = lane;
    }
    // C/D：4 个 fp32
    //   c0,c1 = C[g][2t, 2t+1]   c2,c3 = C[g+8][2t, 2t+1]
    for (int i = 0; i < 4; ++i) {
        int row = g + (i >= 2 ? 8 : 0);
        int col = 2 * t + (i & 1);
        cmap[row * 8 + col] = lane;
    }
}

// 一个 warp 算 C[16x8] = A[16x16] @ B[16x8]（bf16 输入，fp32 累加）。A、B 都是行主序。
// 这里直接按 fragment 布局从 global memory 取数（和上面 fragment_map 的下标完全一致）。
// 真实 kernel 会先搬到 smem 再用 ldmatrix 装 fragment —— 见讲义 7.8 和练习 ex4。
__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
    // 低 16 位放下标小的那个元素
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

__global__ void mma_one_tile(const uint16_t* A, const uint16_t* B, float* C) {
    int lane = threadIdx.x;
    int g = lane >> 2, t = lane & 3;
    uint32_t a[4], b[2];
    for (int r = 0; r < 4; ++r) {
        int row = g + ((r & 1) ? 8 : 0);
        int col = 2 * t + ((r & 2) ? 8 : 0);
        a[r] = pack2(A[row * 16 + col], A[row * 16 + col + 1]);   // 行主序：两个元素本来就相邻
    }
    for (int r = 0; r < 2; ++r) {
        int k = 2 * t + (r ? 8 : 0);
        b[r] = pack2(B[k * 8 + g], B[(k + 1) * 8 + g]);           // k 方向相邻 -> 在内存里隔了一行
    }
    float c[4] = {0.f, 0.f, 0.f, 0.f};
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    C[g * 8 + 2 * t] = c[0];
    C[g * 8 + 2 * t + 1] = c[1];
    C[(g + 8) * 8 + 2 * t] = c[2];
    C[(g + 8) * 8 + 2 * t + 1] = c[3];
}

std::vector<torch::Tensor> get_maps() {
    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    auto a = torch::full({16, 16}, -1, opt), b = torch::full({16, 8}, -1, opt), c = torch::full({16, 8}, -1, opt);
    fragment_map<<<1, 32>>>(a.data_ptr<int>(), b.data_ptr<int>(), c.data_ptr<int>());
    CUDA_CHECK_LAUNCH();
    return {a, b, c};
}

torch::Tensor mma_tile(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A); CHECK_INPUT(B);
    auto C = torch::empty({16, 8}, A.options().dtype(torch::kFloat32));
    mma_one_tile<<<1, 32>>>(reinterpret_cast<const uint16_t*>(A.data_ptr<at::BFloat16>()),
                            reinterpret_cast<const uint16_t*>(B.data_ptr<at::BFloat16>()), C.data_ptr<float>());
    CUDA_CHECK_LAUNCH();
    return C;
}
"""


def show(name, m):
    print(f"\n{name}（每格 = 持有该元素的 lane id）")
    for row in m.tolist():
        print("   " + " ".join(f"{v:2d}" for v in row))


if __name__ == "__main__":
    mod = load_cuda("mma_sync_demo", SRC, ["get_maps", "mma_tile"])
    amap, bmap, cmap = mod.get_maps()
    show("A 16x16（行 = m，列 = k）", amap)
    show("B 16x8（行 = k，列 = n）", bmap)
    show("C 16x8（行 = m，列 = n）", cmap)
    print("\n看点：A 里 lane 0..3 共享第 0 行，每人连续 2 个；B 里同一个 lane 持有一列上 k 相邻的两个元素。")
    print("      每个元素恰好属于一个 lane（没有 -1）：A 32x8=256=16x16，B、C 都是 32x4=128=16x8。\n")

    torch.manual_seed(0)
    A = torch.randn(16, 16, device="cuda").bfloat16()
    B = torch.randn(16, 8, device="cuda").bfloat16()
    check("mma.sync 16x8x16", mod.mma_tile(A, B), A.float() @ B.float(), atol=1e-4, rtol=1e-4)
    finish()
