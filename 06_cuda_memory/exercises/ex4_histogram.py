"""练习 06-4：shared memory 私有化直方图

目标：hist[b] = x 中等于 b 的元素个数。x 是 int32，取值在 [0, bins)，bins 最多 12288。
  - 对照组 hist_global_kernel 已写好：每个元素直接对全局显存做 atomicAdd
  - 你写 hist_smem_kernel：每个 block 在 shared memory 里攒一份私有直方图，最后每个 bin 只往全局加一次

已经给你的：host 函数（动态 shared memory 大小 = bins * 4 字节，作为 <<<>>> 的第三个参数传入）、测试、benchmark。

提示：
  - extern __shared__ int local[]; 是动态 shared memory 的声明方式，大小由 launch 决定
  - 三个阶段之间都要 __syncthreads()：清零没完成就开始加，或者没加完就开始合并，都会错
  - 测试里有"所有元素都相同"的极端情况（所有线程抢同一个 bin）

做完之后想一想：
  - 表格里 global atomics 在 256 个 bin 时比 8192 个 bin 慢得多。为什么 bin 越少反而越慢？
  - 私有化之后，全局 atomicAdd 的次数从 n 降到了多少？
  - all-same 这种最坏情况，你的 smem 版本为什么还能这么快？（提示：同一个 warp 里对同一地址的原子操作，
    硬件/编译器会先在 warp 内合并；可以用 ncu 看 shared atomic 的指令数）
  - 如果 bins 是 100 万（比如统计 token id 的频次），shared memory 放不下，怎么办？

运行：python 06_cuda_memory/exercises/ex4_histogram.py
"""
import torch

from common import bench, check, finish, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

constexpr int THREADS = 256;

// 对照组（已写好）：每个元素直接 atomicAdd 到全局显存里的直方图
__global__ void hist_global_kernel(const int* __restrict__ x, int* __restrict__ hist, int64_t n) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        atomicAdd(&hist[x[i]], 1);
    }
}

// 私有化：每个 block 先在 shared memory 里攒一份自己的直方图，最后再合并到全局
__global__ void hist_smem_kernel(const int* __restrict__ x, int* __restrict__ hist, int64_t n, int bins) {
    extern __shared__ int local[];               // 动态 shared memory，大小 = bins * sizeof(int)

    // TODO:
    //   1. 把 local[0..bins) 清零（所有线程分工，循环步长 blockDim.x）；__syncthreads()
    //   2. grid-stride 扫 x，atomicAdd(&local[x[i]], 1)；__syncthreads()
    //   3. 合并：每个线程负责若干个 bin，把非 0 的 local[b] atomicAdd 到全局 hist[b]
}

torch::Tensor histogram(torch::Tensor x, int64_t bins, bool use_smem) {
    CHECK_INPUT(x);
    TORCH_CHECK(x.scalar_type() == torch::kInt32, "输入要 int32，取值在 [0, bins)");
    TORCH_CHECK(bins > 0 && bins * 4 <= 48 * 1024, "bins 太多：动态 shared memory 超过 48KB 要 cudaFuncSetAttribute");
    auto hist = torch::zeros({bins}, x.options());
    int64_t n = x.numel();
    if (n == 0) return hist;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int64_t blocks = std::min<int64_t>((n + THREADS - 1) / THREADS, (int64_t)num_sms * 8);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (use_smem) {
        hist_smem_kernel<<<blocks, THREADS, bins * sizeof(int), stream>>>(x.data_ptr<int>(), hist.data_ptr<int>(), n, (int)bins);
    } else {
        hist_global_kernel<<<blocks, THREADS, 0, stream>>>(x.data_ptr<int>(), hist.data_ptr<int>(), n);
    }
    CUDA_CHECK_LAUNCH();
    return hist;
}
"""


def ref_hist(x, bins):
    return torch.bincount(x, minlength=bins).to(torch.int32)


if __name__ == "__main__":
    mod = load_cuda("ex06_4_hist", CUDA_SRC, ["histogram"])
    torch.manual_seed(0)
    cases = [
        ("uniform n=1 bins=4", torch.randint(0, 4, (1,)), 4),
        ("uniform n=1000 bins=10", torch.randint(0, 10, (1000,)), 10),
        ("uniform n=1M bins=256", torch.randint(0, 256, (1 << 20,)), 256),
        ("uniform n=1M bins=10000", torch.randint(0, 10000, (1 << 20,)), 10000),
        ("all-same n=1M bins=256", torch.full((1 << 20,), 7), 256),
        ("skewed n=4M bins=4096", (torch.randn(1 << 22).abs() * 30).long().clamp(max=4095), 4096),
    ]
    for name, x, bins in cases:
        x = x.to(device="cuda", dtype=torch.int32)
        check(name, mod.histogram(x, bins, True), ref_hist(x, bins), atol=0, rtol=0)

    n = 1 << 24
    rows = []
    for dist, x, bins in [("uniform", torch.randint(0, 256, (n,), device="cuda"), 256),
                          ("all-same", torch.full((n,), 7, device="cuda"), 256),
                          ("uniform", torch.randint(0, 8192, (n,), device="cuda"), 8192)]:
        x = x.to(torch.int32)
        for name, fn in [("global atomics", lambda: mod.histogram(x, bins, False)),
                         ("smem privatized", lambda: mod.histogram(x, bins, True)),
                         ("torch.bincount", lambda: torch.bincount(x, minlength=bins))]:
            ms = bench(fn)
            rows.append(dict(data=dist, bins=bins, impl=name, us=ms * 1e3, Gelem_s=n / ms / 1e6))
    report(rows, f"直方图 n=2^24 int32")
    finish()
