"""练习 03-2：RMSNorm 前向（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, N, stride_x, stride_out, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # 越界位置填 0：不影响平方和；但求均值时分母必须是 N 而不是 BLOCK
    ms = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(ms + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * stride_out + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


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
