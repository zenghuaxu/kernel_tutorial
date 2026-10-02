"""示例：CuTe 的核心概念 —— Layout = (Shape, Stride)，以及用它切 tile。

运行：python 10_hopper/examples/cute_layouts.py

CuTe DSL（nvidia-cutlass-dsl）是 CUTLASS 4 的 Python 前端：用 Python 写，JIT 成和 C++ CuTe 一样的代码。
它的一切都围绕 Layout：一个从"逻辑坐标"到"内存偏移"的函数。

在 @cute.jit 函数里，普通的 Python print 在**编译期**执行，打印静态的 layout；
cute.printf 在**运行期**（GPU 上）打印。
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack


@cute.jit
def show_layouts():
    # 1. 行主序 4x8：坐标 (i, j) -> i*8 + j
    row_major = cute.make_layout((4, 8), stride=(8, 1))
    # 2. 列主序 4x8：坐标 (i, j) -> i + j*4
    col_major = cute.make_layout((4, 8), stride=(1, 4))
    print("行主序 4x8          :", row_major)
    print("列主序 4x8          :", col_major)
    print("  row_major((1, 2)) =", row_major((1, 2)), "   (1*8 + 2)")
    print("  col_major((1, 2)) =", col_major((1, 2)), "   (1 + 2*4)")

    # 3. 分层 shape：把 8 列拆成 (2, 4)。Layout 可以嵌套 —— 这是 CuTe 描述"线程 × 每线程元素"的方式
    nested = cute.make_layout((4, (2, 4)), stride=(8, (4, 1)))
    print("嵌套 (4,(2,4))      :", nested, " size =", cute.size(nested))

    # 4. zipped_divide：按 tile 切。结果的第 0 个 mode 是 tile 内坐标，第 1 个 mode 是"第几个 tile"
    big = cute.make_layout((16, 64), stride=(64, 1))
    tiled = cute.zipped_divide(big, (1, 8))
    print("16x64 按 (1,8) 切   :", tiled)
    print("   -> 每个 tile 8 个连续元素，一共", cute.size(tiled, mode=[1]), "个 tile；一个线程拿一个 tile = 一次 16B 向量访存（bf16）")


@cute.kernel
def print_kernel(gA: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("GPU 上: gA[0, 1] = %f, gA[1, 0] = %f\n", gA[0, 1], gA[1, 0])


@cute.jit
def show_tensor(mA: cute.Tensor):
    print("torch tensor 进入 CuTe 后的 layout:", mA.layout)
    print_kernel(mA).launch(grid=(1, 1, 1), block=(32, 1, 1))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)   # 让编译期 print 和 GPU printf 的输出顺序一致
    show_layouts()
    a = torch.arange(12, device="cuda", dtype=torch.float32).reshape(3, 4)
    show_tensor(from_dlpack(a))
    torch.cuda.synchronize()
    print("\n对照 torch：a[0,1] =", a[0, 1].item(), " a[1,0] =", a[1, 0].item(), "  a.stride() =", a.stride())
