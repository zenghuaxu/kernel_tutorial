"""示例：用 Triton 解释器调试 kernel。

对比两种运行方式：
    python 02_profiling/examples/interpret_debug.py                     # 正常编译到 GPU
    TRITON_INTERPRET=1 python 02_profiling/examples/interpret_debug.py  # 在 CPU 上用 numpy 逐个 program 解释执行

解释器模式下：
  - kernel 里的 Python print() 能直接打印 tensor 的值（是 numpy 数组）
  - 可以用 breakpoint() / pdb 单步，查看 offs、mask 等中间变量
  - 越界访问会直接报错，而不是在 GPU 上静默读到垃圾
  - 很慢：只用小输入、小 grid
注意：TRITON_INTERPRET 必须在 import triton 之前设置（所以用环境变量，不要在代码里改）。
"""
import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def row_max_kernel(x_ptr, out_ptr, N, stride, BLOCK: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, axis=0)
    if DEBUG:
        # 解释器：普通 print 就行，打印的是 numpy 数组
        print("row", row, "offs", offs, "mask", mask, "x", x, "max", m)
    tl.store(out_ptr + row, m)


@triton.jit
def device_print_kernel(x_ptr, N, BLOCK: tl.constexpr):
    # GPU 上：tl.device_print 会让每个线程打印它持有的元素（输出量 = 元素个数，只在 grid 很小时用）
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < N, other=0.0)
    tl.device_print("x=", x)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(3, 5, device="cuda")
    out = torch.empty(3, device="cuda")
    print(f"=== TRITON_INTERPRET={'1' if INTERPRET else '0'} ===")
    row_max_kernel[(3,)](x, out, 5, x.stride(0), BLOCK=8, DEBUG=INTERPRET)
    print("kernel:", out.tolist())
    print("torch :", x.max(dim=1).values.tolist())

    if not INTERPRET:
        print("\n--- tl.device_print（GPU 上打印，1 个 warp、BLOCK=32：每个线程打印一个元素）---")
        # 注意：如果 BLOCK 比线程数还少（比如 BLOCK=4、4 个 warp），同一个元素会被多个线程各打印一遍
        device_print_kernel[(1,)](x, 6, BLOCK=32, num_warps=1)
        torch.cuda.synchronize()   # device_print 的输出在 kernel 结束后才刷出来
        print("\n现在用 TRITON_INTERPRET=1 再跑一次，看看 kernel 内部的 print。")
