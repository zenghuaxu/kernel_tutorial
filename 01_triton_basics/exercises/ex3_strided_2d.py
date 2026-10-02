"""练习 01-3：任意 stride 的 2D kernel

目标：out = (x + bias[None, :]) * scale
  - x 是 [M, N] 的 2D tensor，**可能不是 contiguous**（转置视图、按列切片……），所以必须用 stride 算地址
  - out 是新分配的 contiguous tensor（wrapper 里已经建好），也通过 stride 访问
  - 二维 grid：program (pid_m, pid_n) 负责 [BLOCK_M, BLOCK_N] 的一个 tile
  - 计算用 fp32

提示：
  - 讲义 1.3 节：offs_m[:, None] * stride_m + offs_n[None, :] * stride_n 生成 tile 的地址
  - mask 也是二维的：(offs_m[:, None] < M) & (offs_n[None, :] < N)
  - bias 是一维的 [N]，load 出来是 [BLOCK_N]，用 b[None, :] 广播到 tile 上
  - wrapper 里的 grid 也要你填

做完之后想一想：最后会用两种 tile 形状（32x128 和 1x1024）分别测 contiguous 和转置视图的带宽。
  - 1x1024 的 tile 读转置视图时，带宽为什么掉得这么厉害？（转置视图里相邻的 n 在内存里相距多远？
    一个 warp 的 32 个线程一次读到的地址落在多少个 128 字节的 cache line 里？）
  - 那 32x128 的 tile 读转置视图为什么几乎不掉？（提示：沿 m 方向的 32 个元素在内存里是连续的，
    整个 program 读完之后，每条 cache line 里的数据是不是都用上了？L1/L2 帮了什么忙？）

运行：python 01_triton_basics/exercises/ex3_strided_2d.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def bias_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # TODO
    pass


def bias_scale(x: torch.Tensor, bias: torch.Tensor, scale: float, block_m: int = 32, block_n: int = 128) -> torch.Tensor:
    """out = (x + bias[None, :]) * scale。x 可以是任意 stride 的 2D 视图，out 是 contiguous 的。"""
    assert x.dim() == 2 and bias.shape == (x.shape[1],) and bias.is_contiguous()
    M, N = x.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N = block_m, block_n
    grid = None  # TODO: 二维 grid
    bias_scale_kernel[grid](
        x, bias, out, M, N,
        x.stride(0), x.stride(1), out.stride(0), out.stride(1),
        scale, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )
    return out


def ref(x, bias, scale):
    return ((x.float() + bias.float()[None, :]) * scale).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = {
        "contiguous 100x300": lambda: torch.randn(100, 300, device="cuda"),
        "transposed view 257x129": lambda: torch.randn(129, 257, device="cuda").t(),
        "column slice x[:, ::3]": lambda: torch.randn(64, 600, device="cuda")[:, ::3],
        "row slice x[5:70]": lambda: torch.randn(80, 128, device="cuda")[5:70],
        "bf16 1000x1000": lambda: torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16),
    }
    for name, make in cases.items():
        x = make()
        bias = torch.randn(x.shape[1], device="cuda", dtype=x.dtype)
        tol = 1e-2 if x.dtype == torch.bfloat16 else 1e-5
        check(name, bias_scale(x, bias, 0.5), ref(x, bias, 0.5), atol=tol, rtol=tol)

    # 性能：同样是 8192x8192 fp32，contiguous vs 转置视图
    base = torch.randn(8192, 8192, device="cuda")
    bias = torch.randn(8192, device="cuda")
    nbytes = 2 * base.numel() * 4
    rows = []
    for name, x in [("contiguous", base), ("transposed", base.t())]:
        for bm, bn in [(32, 128), (1, 1024)]:
            ms = bench(lambda: bias_scale(x, bias, 0.5, bm, bn))
            rows.append(dict(input=name, tile=f"{bm}x{bn}", us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "8192x8192 fp32")
    finish()
