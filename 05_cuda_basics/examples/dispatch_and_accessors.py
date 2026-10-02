"""示例：模板 + AT_DISPATCH 支持多种 dtype；packed_accessor 访问多维 tensor；__device__ 辅助函数。

运行：python 05_cuda_basics/examples/dispatch_and_accessors.py
"""
import torch

from common import check, load_cuda

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

// __device__：只能在 GPU 上被别的 device/global 函数调用。编译器通常会把它内联。
// __forceinline__ 强制内联。
__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// ---------- 1. 模板 kernel：scalar_t 由 AT_DISPATCH 决定 ----------
// scalar_t 可能是 float、c10::Half、c10::BFloat16。
// c10::Half / c10::BFloat16 在 device 上可以 static_cast<float>(v) 转成 float，
// 也可以用 static_cast<scalar_t>(f) 从 float 构造 —— 所以一份代码三种 dtype 都能用。
template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* __restrict__ x, scalar_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = static_cast<float>(x[i]);          // 低精度读入，转 fp32 计算
        out[i] = static_cast<scalar_t>(gelu_tanh(v)); // 写回时转回原 dtype
    }
}

torch::Tensor gelu(torch::Tensor x) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    const int threads = 256;
    const int64_t blocks = (n + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();
    // AT_DISPATCH_FLOATING_TYPES_AND2(额外类型1, 额外类型2, 运行时dtype, 名字, lambda)
    // 宏根据 x.scalar_type() 展开成 switch，每个分支里 scalar_t 被 typedef 成对应的 C++ 类型。
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gelu", [&] {
        gelu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    CUDA_CHECK_LAUNCH();
    return out;
}

// ---------- 2. packed_accessor：按 [i][j] 下标访问，自动用 stride 算地址 ----------
// 适合写原型；性能敏感的地方一般还是自己用 data_ptr + stride 算，控制得更细。
__global__ void row_sum_kernel(
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> x,   // 2 维 float
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> out) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < x.size(0)) {
        float s = 0.f;
        for (int j = 0; j < x.size(1); ++j) s += x[row][j];   // 每个线程串行加一整行（很慢，只是演示）
        out[row] = s;
    }
}

torch::Tensor row_sum(torch::Tensor x) {
    CHECK_CUDA(x);   // 不要求 contiguous：accessor 带着 stride
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kFloat32);
    auto out = torch::empty({x.size(0)}, x.options());
    const int threads = 128;
    row_sum_kernel<<<(x.size(0) + threads - 1) / threads, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 1, torch::RestrictPtrTraits>());
    CUDA_CHECK_LAUNCH();
    return out;
}
"""

if __name__ == "__main__":
    mod = load_cuda("dispatch_demo", CUDA_SRC, ["gelu", "row_sum"])
    torch.manual_seed(0)
    x = torch.randn(10_000, device="cuda")
    for dt, tol in [(torch.float32, 1e-5), (torch.float16, 1e-3), (torch.bfloat16, 1e-2)]:
        xd = x.to(dt)
        ref = torch.nn.functional.gelu(xd.float(), approximate="tanh").to(dt)
        check(f"gelu {dt}", mod.gelu(xd), ref, atol=tol, rtol=tol)
    try:
        mod.gelu(x.to(torch.int32))
    except RuntimeError as e:
        print("  int32 输入 -> AT_DISPATCH 报错:", str(e).splitlines()[0])

    m = torch.randn(300, 77, device="cuda")
    check("row_sum contiguous", mod.row_sum(m), m.sum(1), atol=1e-4, rtol=1e-4)
    check("row_sum 转置视图", mod.row_sum(m.t()), m.t().sum(1), atol=1e-4, rtol=1e-4)
