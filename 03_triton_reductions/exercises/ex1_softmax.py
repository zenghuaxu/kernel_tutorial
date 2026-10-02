"""练习 03-1：按行 softmax

目标：对 [M, N] 的每一行做 softmax：y = exp(x - max(x)) / sum(exp(x - max(x)))
  - 一个 program 处理一整行，BLOCK = 不小于 N 的最小 2 的幂（N 最大到 16384）
  - 输入可以是 fp32 或 bf16；fp32 计算
  - 行之间可能有间隔（x[:, :1000] 这种视图），所以用 stride_x 定位每一行的起点

提示（讲义 3.1、3.2 节）：
  - tl.max(x, axis=0)、tl.sum(x, axis=0) 把一个 [BLOCK] 向量归约成标量
  - tl.load(..., other=float("-inf"))
  - triton.next_power_of_2(N)

做完之后想一想：
  - 表格里 4096x4096 时你的 kernel 比 torch.softmax 快将近一倍（实测 ~2290 vs ~1150 GB/s）。
    torch 的 softmax 读了几遍输入？
  - N=16384 时每个 program 要在寄存器里放 16384 个 fp32 = 64KB。如果 N=262144 呢？（看 3.3 节的循环写法）

运行：python 03_triton_reductions/exercises/ex1_softmax.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, stride_x, stride_out, BLOCK: tl.constexpr):
    # TODO: 一个 program 处理一行（row = tl.program_id(0)），BLOCK >= N
    #   1. load 这一行（越界位置填什么值，才能既不影响 max 也不影响 sum？），转 fp32
    #   2. 减去行最大值再 exp（数值稳定），求和，相除
    #   3. 转回输出 dtype 并 store
    pass

def softmax(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做 softmax。x: [..., N]，最后一维连续。"""
    assert x.stride(-1) == 1
    x2 = x.reshape(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    BLOCK = None  # TODO: BLOCK 必须是 2 的幂，且能一次装下整行
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    softmax_kernel[(M,)](x2, out, N, x2.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=num_warps)
    return out.view_as(x)


def ref_softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(1, 1), (3, 7), (64, 128), (100, 1000), (17, 4096), (8, 8192), (2, 3, 513)]:
        x = torch.randn(shape, device="cuda")
        check(f"fp32 {shape}", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    check("bf16 (64, 2048)", softmax(x), ref_softmax(x), atol=1e-2, rtol=1e-2)
    # 数值稳定性：logit 很大时，不减 max 会 exp 溢出成 inf
    x = torch.randn(16, 1000, device="cuda") * 100 + 500
    check("大 logit（数值稳定性）", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)
    # 视图：行之间有间隔
    x = torch.randn(32, 1024, device="cuda")[:, :1000]
    check("行 stride != N 的视图", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)

    rows = []
    compiled = torch.compile(lambda t: torch.softmax(t, dim=-1))
    for M, N in [(4096, 1024), (4096, 4096), (1024, 16384)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        nbytes = 2 * x.numel() * 2
        for name, fn in [("triton", lambda: softmax(x)),
                         ("torch", lambda: torch.softmax(x, dim=-1)),
                         ("torch.compile", lambda: compiled(x))]:
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "softmax bf16")
    finish()
