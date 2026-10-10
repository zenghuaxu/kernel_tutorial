"""示例：第一个 CUDA kernel —— 带详细注释的 vector add，外加"线程是怎么编号的"演示。

运行：python 05_cuda_basics/examples/hello_cuda.py
（第一次运行要编译 30~60 秒）
"""
import torch

from common import bench, check, gbps, gpu_spec, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream()

// ---------------- device 代码：在 GPU 上跑 ----------------
// __global__：kernel 入口，由 host 用 <<<grid, block>>> 启动，返回值必须是 void。
// 每个线程都执行这整个函数，靠 blockIdx / threadIdx 区分"我是谁"。
__global__ void add_kernel(const float* __restrict__ x,
                           const float* __restrict__ y,
                           float* __restrict__ out,
                           int64_t n) {
    // 全局线程编号 = 第几个 block * 每个 block 的线程数 + block 内编号
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                 // 尾部保护：最后一个 block 可能有多余的线程
        out[i] = x[i] + y[i];    // 每个线程只处理一个元素（Triton 里一个 program 处理一整块）
    }
}

// 记录每个线程看到的内建变量，用来直观理解编号
__global__ void whoami_kernel(int* out) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    out[gid * 4 + 0] = blockIdx.x;
    out[gid * 4 + 1] = threadIdx.x;
    out[gid * 4 + 2] = threadIdx.x / 32;   // warp 编号（block 内）
    out[gid * 4 + 3] = threadIdx.x % 32;   // lane 编号（warp 内）
}

// ---------------- host 代码：在 CPU 上跑，负责检查参数、分配输出、启动 kernel ----------------
torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x); CHECK_INPUT(y);
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "这个示例只支持 fp32");
    TORCH_CHECK(x.sizes() == y.sizes(), "shape 不一致");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    if (n == 0) return out;
    const int threads = 256;                                  // 每个 block 256 个线程（上限 1024）
    const int64_t blocks = (n + threads - 1) / threads;       // ceil(n / threads)
    // 在 PyTorch 当前的 stream 上启动，这样和前后的 torch 算子顺序正确
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    add_kernel<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    CUDA_CHECK_LAUNCH();   // launch 配置错误（比如 threads=2048）会在这里立刻报出来
    return out;
}

torch::Tensor whoami(int64_t blocks, int64_t threads) {
    auto out = torch::empty({blocks * threads, 4}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    whoami_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(out.data_ptr<int>());
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("hello_cuda", CUDA_SRC, ["add", "whoami"])

    print("== 线程编号：2 个 block × 64 个线程，每隔 16 个线程打印一次 ==")
    ids = mod.whoami(2, 64).cpu()
    print("  gid  blockIdx  threadIdx  warp  lane")
    for gid in range(0, 128, 16):
        b, t, w, l = ids[gid].tolist()
        print(f"  {gid:3d}  {b:8d}  {t:9d}  {w:4d}  {l:4d}")

    print("\n== 正确性 ==")
    for n in [1, 1000, 98432, 1 << 20]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        check(f"n={n}", mod.add(x, y), x + y)

    print("\n== launch 配置错误会怎样：每个 block 2048 个线程（上限 1024） ==")
    try:
        mod.whoami(1, 2048)
    except RuntimeError as e:
        print("  捕获到 RuntimeError:", str(e).splitlines()[0])

    rows = []
    for n in [1 << 16, 1 << 20, 1 << 24, 1 << 26]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        for name, fn in [("cuda", lambda: mod.add(x, y)), ("torch", lambda: x + y)]:
            ms = bench(fn)
            rows.append(dict(n=n, impl=name, us=ms * 1e3, GBps=gbps(3 * n * 4, ms)))
    report(rows, f"vector add 带宽（{gpu_spec().bw_note()}）")
