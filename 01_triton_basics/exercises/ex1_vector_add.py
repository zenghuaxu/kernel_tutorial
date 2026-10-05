"""练习 01-1：vector add

目标：不看 examples/vector_add.py，自己写出 kernel 体和 grid。
  - kernel：每个 program 处理 BLOCK 个元素，out[i] = x[i] + y[i]
  - wrapper：算出 grid（program 个数）并 launch

提示：
  - tl.program_id(axis=0)、tl.arange(0, BLOCK)、tl.load / tl.store 的 mask 参数
  - grid 是一个 tuple，triton.cdiv(a, b) = ceil(a / b)
  - 测试里有 n=1、n=17 这种不是 BLOCK 整数倍的情况——mask 写错就会挂（或者更糟：静默写坏别的内存）

运行：python 01_triton_basics/exercises/ex1_vector_add.py
"""
import torch
import triton
import triton.language as tl

from common import check, finish


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # TODO: 算出本 program 负责的下标 offs、越界 mask，load x 和 y，store x + y
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    x = tl.load(x_ptr + off, mask=mask)
    y = tl.load(y_ptr + off, mask=mask)
    tl.store(out_ptr + off, x + y, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (n + BLOCK - 1) // BLOCK, # TODO: 一共需要多少个 program？
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
