"""练习 03-2：RMSNorm 前向

LLaMA / Qwen / GLM 都用 RMSNorm：y = x / sqrt(mean(x^2) + eps) * w，对最后一维（hidden）归一化。

目标：写 kernel 体（wrapper 已给出）。
  - 输入输出 bf16（也要支持 fp32），权重 w 是 [N]
  - **平方和必须用 fp32 累加**：bf16 只有 8 位尾数，4096 个数直接在 bf16 里加会严重失真
  - hidden size 不一定是 2 的幂（5120、7168、1000……），注意 mask 和分母

提示：tl.rsqrt；讲义 3.4 节

做完之后想一想：
  - torch eager（HuggingFace 写法）比你慢了约 10 倍（实测 8192x4096：~530us vs ~52us）。用单元 02 的
    gpu_kernels() 数一数它发了几个 kernel，每个读写多少字节，能不能解释这个 10 倍？
  - torch.compile 在 4096 上和你一样快，在 7168 上慢一截（实测 ~70us vs ~46us）。猜猜为什么？（提示：7168 不是 2 的幂）

运行：python 03_triton_reductions/exercises/ex2_rmsnorm.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, stride_x, stride_out, eps, BLOCK: tl.constexpr):
    # TODO: 一个 program 处理一行
    #   1. load 这一行（转 fp32），越界填 0
    #   2. 均方 ms = sum(x*x) / N（注意分母是 N 不是 BLOCK），rstd = rsqrt(ms + eps)
    #   3. load 权重 w，y = x * rstd * w，转回输出 dtype 存回去
    pass

def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.shape[-1] == w.shape[0] and x.stride(-1) == 1
    x2 = x.reshape(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
    rmsnorm_kernel[(M,)](x2, w, out, N, x2.stride(0), out.stride(0), eps, BLOCK=BLOCK, num_warps=num_warps)
    return out.view_as(x)


def ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def eager_rmsnorm(x, w, eps=1e-6):
    # HuggingFace LlamaRMSNorm 的写法
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(1, 8), (5, 100), (64, 1024), (33, 4096), (16, 5120), (8, 7168), (2, 3, 2048)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        check(f"bf16 {shape}", rmsnorm(x, w), ref_rmsnorm(x, w), atol=2e-2, rtol=2e-2)
    x = torch.randn(37, 1000, device="cuda")
    w = torch.rand(1000, device="cuda")
    check("fp32 (37, 1000)", rmsnorm(x, w), ref_rmsnorm(x, w), atol=1e-5, rtol=1e-5)
    x = torch.randn(4, 999, device="cuda") * 1e-4      # 很小的值：eps 起作用
    w = torch.ones(999, device="cuda")
    check("小幅值 + eps", rmsnorm(x, w, eps=1e-5), ref_rmsnorm(x, w, eps=1e-5), atol=1e-5, rtol=1e-4)

    rows = []
    compiled = torch.compile(eager_rmsnorm)
    for M, N in [(8192, 4096), (4096, 7168)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        nbytes = 2 * x.numel() * 2 + N * 2
        for name, fn in [("triton", lambda: rmsnorm(x, w)),
                         ("torch eager", lambda: eager_rmsnorm(x, w)),
                         ("torch.compile", lambda: compiled(x, w))]:
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "RMSNorm bf16（GBps 按最少字节：读 x + 写 out）")
    finish()
