"""示例：LayerNorm 前向（完整实现）。RMSNorm 练习可以参考这里的结构。

LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * w + b
和 RMSNorm 比多了一次"减均值"，所以要两个归约：先求 mean，再求 var = mean((x - mean)^2)。
（也可以一遍同时求 sum(x) 和 sum(x^2)，用 var = E[x^2] - E[x]^2，但这个公式在 fp32 里会有灾难性抵消，
  均值大、方差小时误差很大——这里用更稳的两步法，因为整行已经在寄存器里了，第二步不需要再读内存。）

运行：python 03_triton_reductions/examples/layernorm_fwd.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, gbps, report


@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, mean_ptr, rstd_ptr,
                     N, stride_x, stride_out, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    # 越界位置 x=0，但 x - mean 不是 0 了！必须再 mask 一次，否则 var 会被 (BLOCK-N)*mean^2 污染
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(out_ptr + row * stride_out + offs, y.to(out_ptr.dtype.element_ty), mask=mask)
    # 反向传播要用 mean 和 rstd，前向顺手存下来（每行 2 个 fp32，几乎不占带宽）
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)


def layernorm(x, w, b, eps=1e-5):
    x2 = x.reshape(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    mean = torch.empty(M, device=x.device, dtype=torch.float32)
    rstd = torch.empty(M, device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
    layernorm_kernel[(M,)](x2, w, b, out, mean, rstd, N, x2.stride(0), out.stride(0), eps,
                           BLOCK=BLOCK, num_warps=num_warps)
    return out.view_as(x), mean, rstd


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N in [(7, 100), (64, 1000), (128, 4096)]:
        x = torch.randn(M, N, device="cuda") * 3 + 5        # 均值不为 0：能发现忘了 mask xc 的 bug
        w = torch.randn(N, device="cuda")
        b = torch.randn(N, device="cuda")
        out, mean, rstd = layernorm(x, w, b)
        check(f"fp32 {M}x{N}", out, F.layer_norm(x, (N,), w, b, 1e-5), atol=1e-4, rtol=1e-4)
        check(f"mean {M}x{N}", mean, x.mean(-1), atol=1e-4, rtol=1e-5)

    rows = []
    M, N = 8192, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    check("bf16 8192x4096", layernorm(x, w, b)[0], F.layer_norm(x.float(), (N,), w.float(), b.float()).to(x.dtype),
          atol=5e-2, rtol=2e-2)
    for name, fn in [("triton", lambda: layernorm(x, w, b)), ("torch F.layer_norm", lambda: F.layer_norm(x, (N,), w, b))]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(2 * x.numel() * 2, ms)))
    report(rows, f"LayerNorm bf16 {M}x{N}")
