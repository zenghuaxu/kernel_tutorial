"""示例：自定义 kernel 遇到 torch.compile —— 断图、custom_op、triton_op、CUDA graph。

运行：python 09_backward_and_integration/examples/compile_graph_break.py（第一次要编译 CUDA 扩展，约 1 分钟）

对比四种接入方式：
  A. load_inline 编出来的 CUDA 扩展，直接调用        → Dynamo 不认识这个 C 函数 → 断图
  B. 同一个扩展，包成 torch.library.custom_op         → 不断图（编译器把它当黑盒算子）
  C. 直接在 Python 里 launch Triton kernel            → 不断图（Dynamo 能追踪"用户定义的 Triton kernel"）
  D. torch.library.triton_op + wrap_triton            → 不断图，且 Inductor 能看到 kernel 本身
最后用 mode="reduce-overhead"（CUDA graph）跑 B，确认 custom_op 和 CUDA graph 兼容。
"""
import torch
import triton
import triton.language as tl

from common import load_cuda

CUDA_SRC = r"""
__global__ void softcap_kernel(const float* x, float* y, int n, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = cap * tanhf(x[i] / cap);
}
torch::Tensor softcap_cuda(torch::Tensor x, double cap) {
    CHECK_INPUT(x);
    auto y = torch::empty_like(x);
    int n = x.numel();
    softcap_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n, (float)cap);
    CUDA_CHECK_LAUNCH();
    return y;
}
"""
ext = load_cuda("softcap_ext", CUDA_SRC, ["softcap_cuda"])


# ---- B：custom_op 包一层 ----
@torch.library.custom_op("kt_demo::softcap", mutates_args=())
def softcap_op(x: torch.Tensor, cap: float) -> torch.Tensor:
    return ext.softcap_cuda(x.contiguous(), cap)


@softcap_op.register_fake
def _(x, cap):
    return torch.empty_like(x)


# ---- C / D：Triton ----
@triton.jit
def softcap_triton_kernel(x_ptr, y_ptr, n, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, cap * (2.0 * tl.sigmoid(2.0 * x / cap) - 1.0), mask=mask)


def softcap_triton(x, cap):
    y = torch.empty_like(x)
    softcap_triton_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), cap, BLOCK=1024)
    return y


@torch.library.triton_op("kt_demo::softcap_t", mutates_args=())
def softcap_triton_op(x: torch.Tensor, cap: float) -> torch.Tensor:
    y = torch.empty_like(x)
    torch.library.wrap_triton(softcap_triton_kernel)[(triton.cdiv(x.numel(), 1024),)](
        x, y, x.numel(), cap, BLOCK=1024)
    return y


def make_model(op):
    def model(x):
        return op(x * 2.0, 30.0).sum(-1)
    return model


if __name__ == "__main__":
    x = torch.randn(256, 1024, device="cuda")
    ref = (30.0 * torch.tanh(x * 2.0 / 30.0)).sum(-1)
    for name, op in [("A. 裸 CUDA 扩展", ext.softcap_cuda), ("B. custom_op 包装", softcap_op),
                     ("C. 裸 Triton launch", softcap_triton), ("D. triton_op", softcap_triton_op)]:
        torch._dynamo.reset()
        exp = torch._dynamo.explain(make_model(op))(x)
        out = torch.compile(make_model(op))(x)
        err = (out - ref).abs().max().item()
        print(f"{name:20s} graph_break_count={exp.graph_break_count}  图数={exp.graph_count}  max_err={err:.1e}")
        for r in exp.break_reasons[:1]:
            print("    断图原因:", str(r.reason).splitlines()[0][:110])

    print("\nCUDA graph（mode='reduce-overhead'）+ custom_op：")
    torch._dynamo.reset()
    f = torch.compile(make_model(softcap_op), mode="reduce-overhead", fullgraph=True)
    for _ in range(3):           # 前几次是 warmup + 录制 graph
        out = f(x)
    print(f"    max_err={(out - ref).abs().max().item():.1e}  —— OK")
    print("    CUDA graph 的要求：kernel 里不能有 host 同步（.item()、.cpu()、根据数据决定 shape），")
    print("    launch 要用当前 stream（上面 CUDA 代码里的 at::cuda::getCurrentCUDAStream()）。")
