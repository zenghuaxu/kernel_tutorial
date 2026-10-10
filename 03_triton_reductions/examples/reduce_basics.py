"""示例：按行求和的两种写法——"整行一个 block" vs "在行内循环"。

运行：python 03_triton_reductions/examples/reduce_basics.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, gbps, report


@triton.jit
def row_sum_single(x_ptr, out_ptr, N, stride, BLOCK: tl.constexpr):
    """BLOCK >= N：整行一次 load 进寄存器，一次 tl.sum。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(x, axis=0))


@triton.jit
def row_sum_loop(x_ptr, out_ptr, N, stride, BLOCK: tl.constexpr):
    """BLOCK 固定（比如 1024），在行内循环。累加器是一个 [BLOCK] 向量，最后再归约一次。"""
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + row * stride + offs, mask=offs < N, other=0.0).to(tl.float32)
    tl.store(out_ptr + row, tl.sum(acc, axis=0))


def run_single(x, out):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    return row_sum_single[(M,)](x, out, N, x.stride(0), BLOCK=BLOCK, num_warps=min(16, max(4, BLOCK // 1024)))


def run_loop(x, out, block=2048):
    M, N = x.shape
    return row_sum_loop[(M,)](x, out, N, x.stride(0), BLOCK=block, num_warps=8)


if __name__ == "__main__":
    torch.manual_seed(0)
    rows = []
    for M, N in [(8192, 1000), (8192, 4096), (512, 65536), (64, 524288)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(M, device="cuda", dtype=torch.float32)
        ref = x.float().sum(-1)
        nbytes = x.numel() * 2
        for name, fn in [("single", lambda: run_single(x, out)), ("loop", lambda: run_loop(x, out))]:
            k = fn()
            check(f"{name} {M}x{N}", out, ref, atol=1e-2, rtol=1e-4)
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, regs=k.n_regs, spills=k.n_spills,
                             us=ms * 1e3, GBps=gbps(nbytes, ms)))
        ms = bench(lambda: x.sum(-1, dtype=torch.float32))
        rows.append(dict(shape=f"{M}x{N}", impl="torch", regs="-", spills="-", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "按行求和 bf16 → fp32")
    print("\n看点 1：N=524288 时 single 版本 BLOCK=524288，却没有 spill——因为 x 只被用了一次（求和），")
    print("        编译器可以边 load 边累加，不必把整行同时放在寄存器里。softmax 要用 x 两次，就没这么幸运了")
    print("        （见 examples/online_softmax.py：N=131072 时 spill 几百个寄存器，带宽掉到 ~360 GB/s）。")
    print("看点 2：M=64 时只有 64 个 program，132~148 个 SM（H100 / B200）有一半以上闲着；loop 版本每轮只发出 BLOCK 个 load 就要等，")
    print("        在途的访存请求太少，带宽掉得很厉害。'行数少、行很长'时要把一行拆给多个 program。")
