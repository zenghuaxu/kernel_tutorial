"""练习 01-1：vector add（参考答案）"""
import torch
import triton
import triton.language as tl

from common import check, finish


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [1, 17, 1024, 1025, 98432, 1 << 22]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        check(f"fp32 n={n}", add(x, y), x + y)
    x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    check("bf16 n=4096", add(x, y), x + y, atol=0, rtol=0)
    finish()
