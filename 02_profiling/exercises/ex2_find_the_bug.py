"""练习 02-2：找 bug

下面的 kernel 要算 out[m, n] = x[m, n] * scale[m] + bias[n]（x 是任意 stride 的 2D fp32 tensor）。
它能编译、能运行，但结果不对。**里面一共有 3 个 bug**，找出来并修好（只需要改 kernel）。

推荐的调试流程（讲义 2.5 节）：
  1. 先直接跑，看哪些用例挂了、误差什么样 —— 只有非方阵挂？只有 N 不是 16 的倍数挂？连 16x16 都挂？
  2. 用解释器跑：TRITON_INTERPRET=1 python 02_profiling/exercises/ex2_find_the_bug.py
     在 kernel 里加 print("offs_m", offs_m, "x", x) 之类的语句，对照 torch 的结果看
     解释器把 tensor 拷到 CPU 上执行，越界**写**会写坏 CPU 的堆内存：如果进程直接崩了
     （比如 "free(): invalid size"），这本身就是线索——有 store 写出界了。先只看 load，把 tl.store 注释掉试试
  3. 想不明白时，把用例缩到最小（比如 M=2, N=3），手算每个地址

做完之后想一想：
  - 3 个 bug 里，哪个在 GPU 上可能"偶尔对、偶尔错"或者破坏别的 tensor？为什么这种最危险？
  - 第 3 个 bug 为什么能编译通过？如果把 BLOCK_M 改成 32，会发生什么？

运行：python 02_profiling/exercises/ex2_find_the_bug.py
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
    # TODO: 这个 kernel 里有 3 个 bug
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_m[:, None] < M
    x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xm, mask=mask)
    s = tl.load(scale_ptr + offs_n, mask=offs_n < N)
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
