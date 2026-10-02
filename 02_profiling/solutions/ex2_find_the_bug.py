"""练习 02-2：找 bug（参考答案）

三个 bug 的位置：
  1. mask 只管了行方向，没管列方向 —— N 不是 BLOCK_N 的倍数时，会读写到下一行（甚至 buffer 之外）
  2. x 的列方向 stride 写成了 stride_xm —— contiguous 输入时 x[m, n] 被读成 x[m + n, 0] 之类的位置
  3. scale 是按行的（长度 M），却用 offs_n 去 load —— 只有 BLOCK_M == BLOCK_N 才"碰巧"能编译通过
"""
import torch
import triton
import triton.language as tl

from common import check, finish


@triton.jit
def scale_bias_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)                                     # 修复 1
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask)  # 修复 2
    s = tl.load(scale_ptr + offs_m, mask=offs_m < M)                                         # 修复 3
    b = tl.load(bias_ptr + offs_n, mask=offs_n < N)
    out = x * s[:, None] + b[None, :]
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=mask)


def scale_bias(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """out[m, n] = x[m, n] * scale[m] + bias[n]。x 可以是任意 stride 的 2D fp32 tensor。"""
    M, N = x.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M = BLOCK_N = 16
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    scale_bias_kernel[grid](x, scale, bias, out, M, N, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
                            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = {
        "16x16": lambda: torch.randn(16, 16, device="cuda"),
        "5x7": lambda: torch.randn(5, 7, device="cuda"),
        "33x70": lambda: torch.randn(33, 70, device="cuda"),
        "转置视图 40x24": lambda: torch.randn(24, 40, device="cuda").t(),
    }
    for name, make in cases.items():
        x = make()
        M, N = x.shape
        scale = torch.randn(M, device="cuda")
        bias = torch.randn(N, device="cuda")
        check(name, scale_bias(x, scale, bias), x * scale[:, None] + bias[None, :])
    finish()
