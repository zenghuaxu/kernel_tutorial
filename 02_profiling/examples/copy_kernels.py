"""示例：给 Nsight Compute (ncu) 用的靶子——三个"同样是 copy、快慢差好几倍"的 Triton kernel。

单独运行只打印带宽：
    python 02_profiling/examples/copy_kernels.py
用 ncu 看原因（见 README 2.6 节）。加 --once：每个 kernel 只跑一次，不做 benchmark，ncu 只会抓到 3 次 launch：
    bash tools/ncu.sh --section SpeedOfLight -k regex:copy python 02_profiling/examples/copy_kernels.py --once

  copy_coalesced : 每个 warp 读一段连续内存，fp32 向量化 128-bit load         —— 理想情况
  copy_column    : 按列读一个 row-major 矩阵（tile 是 1 x BLOCK 的"竖条"）   —— 不合并访存
  copy_scalar    : bf16，每个线程只拿 1 个元素（16-bit load），block 又很小   —— 指令/调度开销大
"""
import sys

import torch
import triton
import triton.language as tl

from common import bench, gbps, report


@triton.jit
def copy_coalesced(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


@triton.jit
def copy_column(x_ptr, out_ptr, M, N, BLOCK: tl.constexpr):
    # program (pid_m, n) 负责第 n 列的 BLOCK 个元素：相邻 lane 的地址相差 N*4 字节
    pid_m = tl.program_id(0)
    n = tl.program_id(1)
    offs_m = pid_m * BLOCK + tl.arange(0, BLOCK)
    mask = offs_m < M
    ptrs = offs_m * N + n
    tl.store(out_ptr + ptrs, tl.load(x_ptr + ptrs, mask=mask), mask=mask)


@triton.jit
def copy_scalar(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # 和 copy_coalesced 一模一样，只是单独起个名字，方便在 ncu 里区分
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def run_coalesced(x, out):
    n = x.numel()
    copy_coalesced[(triton.cdiv(n, 4096),)](x, out, n, BLOCK=4096, num_warps=8)


def run_column(x, out):
    M, N = x.shape
    copy_column[(triton.cdiv(M, 1024), N)](x, out, M, N, BLOCK=1024, num_warps=4)


def run_scalar(x, out):
    n = x.numel()
    copy_scalar[(triton.cdiv(n, 128),)](x, out, n, BLOCK=128, num_warps=4)


if __name__ == "__main__":
    once = "--once" in sys.argv
    M = N = 4096
    x32 = torch.randn(M, N, device="cuda")
    o32 = torch.empty_like(x32)
    x16 = torch.randn(M, N * 2, device="cuda", dtype=torch.bfloat16)   # 字节数和 fp32 版一样：64MB
    o16 = torch.empty_like(x16)

    rows = []
    for name, fn, x, o in [("copy_coalesced", run_coalesced, x32, o32),
                           ("copy_column", run_column, x32, o32),
                           ("copy_scalar(bf16,BLOCK=128)", run_scalar, x16, o16)]:
        fn(x, o)
        assert torch.equal(o, x), name
        if once:
            continue
        ms = bench(lambda: fn(x, o))
        rows.append(dict(kernel=name, us=ms * 1e3, GBps=gbps(2 * x.numel() * x.element_size(), ms)))
    if once:
        print("每个 kernel 各跑了一次（--once）")
    else:
        report(rows, "同样搬 64MB → 64MB")
