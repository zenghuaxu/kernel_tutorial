"""练习 05-4：模板 + AT_DISPATCH 的 SwiGLU（参考答案）"""
import torch
import torch.nn.functional as F

from common import bench, check, finish, gbps, load_cuda, report

CUDA_SRC = r"""
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ float silu(float x) {
    return x / (1.f + __expf(-x));   // __expf：快速版 exp（精度略低，对激活函数足够）
}

// out = silu(g) * u。scalar_t ∈ {float, c10::Half, c10::BFloat16}
template <typename scalar_t>
__global__ void swiglu_kernel(const scalar_t* __restrict__ g, const scalar_t* __restrict__ u,
                              scalar_t* __restrict__ out, int64_t n) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float gv = static_cast<float>(g[i]);
        float uv = static_cast<float>(u[i]);
        out[i] = static_cast<scalar_t>(silu(gv) * uv);
    }
}

// bf16 专用的向量化版本：每个线程一次读 16 字节 = 8 个 bf16（一个 uint4），
// 再把它看成 4 个 __nv_bfloat162，两个两个地转成 float2 计算。
__global__ void swiglu_bf16x8_kernel(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
                                     __nv_bfloat16* __restrict__ out, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    int64_t n8 = n / 8;
    for (int64_t i = tid; i < n8; i += stride) {
        uint4 gv = reinterpret_cast<const uint4*>(g)[i];
        uint4 uv = reinterpret_cast<const uint4*>(u)[i];
        uint4 ov;
        const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(&gv);
        const __nv_bfloat162* u2 = reinterpret_cast<const __nv_bfloat162*>(&uv);
        __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(&ov);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 gf = __bfloat1622float2(g2[k]);
            float2 uf = __bfloat1622float2(u2[k]);
            o2[k] = __floats2bfloat162_rn(silu(gf.x) * uf.x, silu(gf.y) * uf.y);
        }
        reinterpret_cast<uint4*>(out)[i] = ov;
    }
    int64_t tail = n8 * 8 + tid;
    if (tail < n) {
        out[tail] = __float2bfloat16(silu(__bfloat162float(g[tail])) * __bfloat162float(u[tail]));
    }
}

torch::Tensor swiglu_bf16_vec(torch::Tensor g, torch::Tensor u) {
    CHECK_INPUT(g); CHECK_INPUT(u);
    TORCH_CHECK(g.scalar_type() == torch::kBFloat16 && u.scalar_type() == torch::kBFloat16 && g.sizes() == u.sizes());
    auto out = torch::empty_like(g);
    int64_t n = g.numel();
    if (n == 0) return out;
    // at::BFloat16 和 __nv_bfloat16 内存布局相同（都是 2 字节），可以直接 reinterpret_cast
    auto gp = reinterpret_cast<const __nv_bfloat16*>(g.data_ptr<at::BFloat16>());
    auto up = reinterpret_cast<const __nv_bfloat16*>(u.data_ptr<at::BFloat16>());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    TORCH_CHECK(((uintptr_t)gp | (uintptr_t)up | (uintptr_t)op) % 16 == 0, "指针要 16 字节对齐");
    const int threads = 256;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int blocks = (int)std::min<int64_t>((n / 8 + threads - 1) / threads + 1, (int64_t)num_sms * 16);
    swiglu_bf16x8_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(gp, up, op, n);
    CUDA_CHECK_LAUNCH();
    return out;
}

torch::Tensor swiglu(torch::Tensor g, torch::Tensor u) {
    CHECK_INPUT(g); CHECK_INPUT(u);
    TORCH_CHECK(g.sizes() == u.sizes() && g.scalar_type() == u.scalar_type(), "g/u 的 shape、dtype 要一致");
    auto out = torch::empty_like(g);
    int64_t n = g.numel();
    if (n == 0) return out;
    const int threads = 256;
    int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, (int64_t)num_sms * 16);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, g.scalar_type(), "swiglu", [&] {
        swiglu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
    });
    CUDA_CHECK_LAUNCH();
    return out;
}
"""


def ref_swiglu(g, u):
    return (F.silu(g.float()) * u.float()).to(g.dtype)


if __name__ == "__main__":
    mod = load_cuda("ex05_4_swiglu", CUDA_SRC, ["swiglu", "swiglu_bf16_vec"])
    torch.manual_seed(0)
    for dt, tol in [(torch.float32, 1e-5), (torch.float16, 5e-3), (torch.bfloat16, 1.6e-2)]:
        for shape in [(7,), (3, 1000), (4, 512, 1408)]:
            g = (torch.randn(shape, device="cuda") * 3).to(dt)
            u = torch.randn(shape, device="cuda").to(dt)
            check(f"{dt} {shape}", mod.swiglu(g, u), ref_swiglu(g, u), atol=tol, rtol=tol)

    for n in [1, 7, 8, 9, 1000, 4096 * 1408 + 5]:
        g = (torch.randn(n, device="cuda") * 3).to(torch.bfloat16)
        u = torch.randn(n, device="cuda").to(torch.bfloat16)
        check(f"bf16x8 n={n}", mod.swiglu_bf16_vec(g, u), ref_swiglu(g, u), atol=1.6e-2, rtol=1.6e-2)

    n = 8 * 1024 * 1024
    g = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    rows = []
    for name, fn, nbytes in [("cuda 标量", lambda: mod.swiglu(g, u), 3 * n * 2),
                             ("cuda bf16x8", lambda: mod.swiglu_bf16_vec(g, u), 3 * n * 2),
                             ("torch eager", lambda: F.silu(g) * u, 5 * n * 2)]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, bytes_MB=nbytes / 1e6, GBps=gbps(nbytes, ms)))
    report(rows, "SwiGLU bf16, n=8M")
    finish()
