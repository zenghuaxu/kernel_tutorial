"""练习 06-1：shared memory 分块转置 + padding 消除 bank conflict（参考答案）"""
import torch

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int TILE = 32;      // 每个 block 负责一个 32x32 的 tile
constexpr int ROWS = 8;       // block = (32, 8)：每个线程搬 TILE/ROWS = 4 行

// 对照组（已写好）：直接转置。读 in 是合并的（threadIdx.x 沿列），写 out 时相邻线程相距 M 个 float —— 不合并
__global__ void transpose_naive_kernel(const float* __restrict__ in, float* __restrict__ out, int M, int N) {
    int x = blockIdx.x * TILE + threadIdx.x;   // in 的列
    int y = blockIdx.y * TILE + threadIdx.y;   // in 的行
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < N && y + j < M) out[(int64_t)x * M + (y + j)] = in[(int64_t)(y + j) * N + x];
    }
}

// in: [M, N] -> out: [N, M]
template <int PAD>
__global__ void transpose_tiled_kernel(const float* __restrict__ in, float* __restrict__ out, int M, int N) {
    __shared__ float tile[TILE][TILE + PAD];

    // 1. 合并地读：线程 (tx, ty) 读 in 的 tile 里第 ty, ty+8, ty+16, ty+24 行、第 tx 列
    int x = blockIdx.x * TILE + threadIdx.x;   // in 的列
    int y = blockIdx.y * TILE + threadIdx.y;   // in 的行
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < N && y + j < M) tile[threadIdx.y + j][threadIdx.x] = in[(int64_t)(y + j) * N + x];
    }
    __syncthreads();

    // 2. 合并地写：out 的 tile 在 (blockIdx.x, blockIdx.y) 互换后的位置；
    //    线程按列读 shared memory（tile[tx][...]），这一步才是 bank conflict 的来源
    x = blockIdx.y * TILE + threadIdx.x;       // out 的列（= in 的行）
    y = blockIdx.x * TILE + threadIdx.y;       // out 的行（= in 的列）
    for (int j = 0; j < TILE; j += ROWS) {
        if (x < M && y + j < N) out[(int64_t)(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

torch::Tensor transpose(torch::Tensor in, int64_t mode) {
    CHECK_INPUT(in);
    TORCH_CHECK(in.dim() == 2 && in.scalar_type() == torch::kFloat32);
    int M = in.size(0), N = in.size(1);
    auto out = torch::empty({N, M}, in.options());
    dim3 block(TILE, ROWS);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* ip = in.data_ptr<float>();
    float* op = out.data_ptr<float>();
    if (mode == 0)      transpose_naive_kernel<<<grid, block, 0, stream>>>(ip, op, M, N);
    else if (mode == 1) transpose_tiled_kernel<0><<<grid, block, 0, stream>>>(ip, op, M, N);
    else                transpose_tiled_kernel<1><<<grid, block, 0, stream>>>(ip, op, M, N);
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

MODES = {0: "naive", 1: "tiled PAD=0", 2: "tiled PAD=1"}

if __name__ == "__main__":
    mod = load_cuda("ex06_1_transpose", CUDA_SRC, ["transpose"])
    torch.manual_seed(0)
    for M, N in [(1, 1), (33, 65), (1000, 3), (2047, 4099), (4096, 4096)]:
        x = torch.randn(M, N, device="cuda")
        for mode in [1, 2]:
            check(f"{MODES[mode]} {M}x{N}", mod.transpose(x, mode), x.t().contiguous(), atol=0, rtol=0)

    M = N = 8192
    x = torch.randn(M, N, device="cuda")
    nbytes = 2 * M * N * 4
    rows = []
    for mode, name in MODES.items():
        ms = bench(lambda: mod.transpose(x, mode))
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    ms = bench(lambda: x.t().contiguous())
    rows.append(dict(impl="torch .t().contiguous()", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    ms = bench(lambda: x.clone())
    rows.append(dict(impl="torch clone (上限参考)", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "转置 8192x8192 fp32")
    finish()
