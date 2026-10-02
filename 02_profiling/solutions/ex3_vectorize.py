"""练习 02-3：检查编译产物，让 load 向量化（参考答案）"""
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
    best = 0
    for line in compiled.asm["ptx"].splitlines():
        m = re.search(r"\bld\.global(\.[\w:]+)*", line)
        if m is None:
            continue
        instr = m.group(0)
        vec = re.search(r"\.v(\d)\b", instr)
        bits = re.search(r"\.[bfsu](\d+)\b", instr)
        if bits is None:
            continue
        best = max(best, (int(vec.group(1)) if vec else 1) * int(bits.group(1)))
    return best


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
    # wrapper 保证了 N 和 stride 都是 8 的倍数。"// 8 * 8" 在数值上什么都没改，
    # 但编译器由此知道它们能被 8 整除：mask 每 8 个元素一组要么全真要么全假，
    # 行首地址按 16 字节对齐 -> 可以用 128-bit load（8 个 bf16）
    N = N // 8 * 8
    stride = stride // 8 * 8
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
