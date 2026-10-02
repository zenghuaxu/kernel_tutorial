"""练习 02-3：检查编译产物，让 load 向量化

第 1 部分：实现 load_width_bits(compiled)：从 Triton 编译出的 PTX 里找出所有 global load 指令，
          返回单条指令最多读多少 bit（ld.global.v4.b32 = 128，ld.global.b16 = 16）。
          测试会用不同 dtype / BLOCK 编译同一个 copy kernel 来检验你的解析。
第 2 部分：row_scale_kernel 处理每行 1000 个 bf16 的矩阵，load 却是一次 16 bit。
          找出原因并修好，让 load 变成 128-bit（测试会用你第 1 部分的函数检查）。

提示：
  - launch 的返回值就是 CompiledKernel：k = kernel[grid](...)；k.asm["ptx"] 是 PTX 文本
  - Triton 会自动对"能被 16 整除"的整数参数做特化（讲义 2.4 节）。1000 % 16 = 8……
  - mask = offs < N：编译器要能证明"连续 8 个元素的 mask 要么全真要么全假"，才敢合成一条 128-bit load
  - 可以先打印 k.asm["ptx"] 里含 ld.global 的行看看

做完之后想一想：
  - 修好之后带宽提高了多少？（这台被占用的 H100 上实测 bf16 约 2430 → 2620 GB/s）为什么不是好几倍？
  - 除了在 kernel 里"告诉"编译器，还有什么办法？（比如把 N 和 stride 都声明成 tl.constexpr——只改 N 不够，试试看为什么；或者分配时把行长 pad 到 16 的倍数）
    各自的代价是什么？

运行：python 02_profiling/exercises/ex3_vectorize.py
"""
import re

import torch
import triton
import triton.language as tl

from common import bench, check, check_equal, finish, gbps, report


# ---------------- 第 1 部分：从 PTX 里读出 global load 的最大位宽 ----------------
def load_width_bits(compiled) -> int:
    """返回 kernel 里所有 ld.global 指令中，单条指令读取的最大位数。

    ld.global.b16 -> 16；ld.global.b32 -> 32；ld.global.v2.b32 -> 64；ld.global.v4.b32 -> 128
    """
    # TODO: 遍历 compiled.asm["ptx"] 的每一行，找出 ld.global 指令，
    #       从后缀里解析向量宽度（.v2 / .v4，没有就是 1）和元素位数（.b16 / .b32 / .f32 / .u8 ...），
    #       返回 max(向量宽度 * 位数)。提示：re.search(r"\bld\.global(\.[\w:]+)*", line)
    return 0


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


# ---------------- 第 2 部分：让 row_scale 的 load 变成 128-bit ----------------
@triton.jit
def row_scale_kernel(x_ptr, s_ptr, out_ptr, N, stride, BLOCK: tl.constexpr):
    """out[r, :N] = x[r, :N] * s[r]，一个 program 处理一行。"""
    row = tl.program_id(0)
    # TODO: 现在 N=1000 时这里的 load 是 ld.global.b16（一次一个 bf16）。
    #       wrapper 保证了 N 和 stride 都是 8 的倍数，想办法把这个事实告诉编译器（见讲义 2.4 节），
    #       让 load 变成 128-bit（一次 8 个 bf16）。只加代码，不要改下面的计算逻辑。
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    s = tl.load(s_ptr + row).to(tl.float32)
    x = tl.load(x_ptr + row * stride + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + row * stride + offs, (x * s).to(out_ptr.dtype.element_ty), mask=mask)


def row_scale(x: torch.Tensor, s: torch.Tensor):
    M, N = x.shape
    assert x.stride(1) == 1 and N % 8 == 0 and x.stride(0) % 8 == 0
    out = torch.empty_like(x)
    k = row_scale_kernel[(M,)](x, s, out, N, x.stride(0), BLOCK=triton.next_power_of_2(N), num_warps=4)
    return out, k


if __name__ == "__main__":
    torch.manual_seed(0)
    print("第 1 部分：load_width_bits")
    f32 = torch.randn(1 << 16, device="cuda")
    b16 = torch.randn(1 << 16, device="cuda", dtype=torch.bfloat16)
    for name, x, block, expect in [("fp32 BLOCK=1024", f32, 1024, 128),
                                   ("bf16 BLOCK=128", b16, 128, 16),
                                   ("bf16 BLOCK=256", b16, 256, 32),
                                   ("bf16 BLOCK=1024", b16, 1024, 128),
                                   ("fp32 n=1001(mask 不对齐)", f32[:1001], 1024, 32)]:
        out = torch.empty_like(x)
        k = copy_kernel[(triton.cdiv(x.numel(), block),)](x, out, x.numel(), BLOCK=block, num_warps=4)
        check_equal(name, load_width_bits(k), expect)

    print("第 2 部分：row_scale 的 load 要是 128-bit")
    for M, N in [(64, 1000), (37, 2008), (8, 4096)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        s = torch.randn(M, device="cuda", dtype=torch.bfloat16)
        out, k = row_scale(x, s)
        check(f"row_scale {M}x{N}", out, (x.float() * s.float()[:, None]).to(x.dtype), atol=1e-2, rtol=1e-2)
        check_equal(f"row_scale {M}x{N} load 位宽", load_width_bits(k), 128)

    x = torch.randn(32768, 1000, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(32768, device="cuda", dtype=torch.bfloat16)
    ms = bench(lambda: row_scale(x, s))
    report([dict(shape="32768x1000 bf16", us=ms * 1e3, GBps=gbps(2 * x.numel() * 2, ms))], "row_scale")
    finish()
