"""练习 01-3：任意 stride 的 2D kernel（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def bias_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)   # [BLOCK_N]
    y = (x + b[None, :]) * scale

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(o_ptrs, y.to(out_ptr.dtype.element_ty), mask=mask)


def bias_scale(x: torch.Tensor, bias: torch.Tensor, scale: float, block_m: int = 32, block_n: int = 128) -> torch.Tensor:
    """out = (x + bias[None, :]) * scale。x 可以是任意 stride 的 2D 视图，out 是 contiguous 的。"""
    assert x.dim() == 2 and bias.shape == (x.shape[1],) and bias.is_contiguous()
    M, N = x.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N = block_m, block_n
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    bias_scale_kernel[grid](
        x, bias, out, M, N,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def ref(x, bias, scale):
    return ((x.float() + bias.float()[None, :]) * scale).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = {
        "contiguous 100x300": lambda: torch.randn(100, 300, device="cuda"),
        "transposed view 257x129": lambda: torch.randn(129, 257, device="cuda").t(),
        "column slice x[:, ::3]": lambda: torch.randn(64, 600, device="cuda")[:, ::3],
        "row slice x[5:70]": lambda: torch.randn(80, 128, device="cuda")[5:70],
        "bf16 1000x1000": lambda: torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16),
    }
    for name, make in cases.items():
        x = make()
        bias = torch.randn(x.shape[1], device="cuda", dtype=x.dtype)
        tol = 1e-2 if x.dtype == torch.bfloat16 else 1e-5
        check(name, bias_scale(x, bias, 0.5), ref(x, bias, 0.5), atol=tol, rtol=tol)

    # 性能：同样是 8192x8192 fp32，contiguous vs 转置视图
    base = torch.randn(8192, 8192, device="cuda")
    bias = torch.randn(8192, device="cuda")
    nbytes = 2 * base.numel() * 4
    rows = []
    for name, x in [("contiguous", base), ("transposed", base.t())]:
        for bm, bn in [(32, 128), (1, 1024)]:
            ms = bench(lambda: bias_scale(x, bias, 0.5, bm, bn))
            rows.append(dict(input=name, tile=f"{bm}x{bn}", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "8192x8192 fp32")
    finish()
