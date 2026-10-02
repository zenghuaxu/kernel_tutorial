"""示例：带详细注释的 vector add。

运行：python 01_triton_basics/examples/vector_add.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, gbps, report


@triton.jit
def add_kernel(
    x_ptr,               # *fp32：x 的首元素指针（传 torch.Tensor 时 Triton 自动取 data_ptr）
    y_ptr,
    out_ptr,
    n,                   # 运行时整数：元素总数
    BLOCK: tl.constexpr, # 编译期常量：每个 program 处理多少元素（必须是 2 的幂）
):
    # 1. 我是谁：grid 是一维的，所以只看 axis=0
    pid = tl.program_id(axis=0)
    # 2. 我负责哪些元素：一个长度为 BLOCK 的整数向量
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # 3. 尾部保护：n 不一定是 BLOCK 的整数倍
    mask = offs < n
    # 4. 指针向量 -> 一次 load 一整块。被 mask 掉的 lane 不会访问内存
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    # 5. 写回
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    assert x.is_cuda and x.shape == y.shape and x.is_contiguous() and y.is_contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, block),)    # program 个数 = ceil(n / BLOCK)
    add_kernel[grid](x, y, out, n, BLOCK=block)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [1, 1000, 98432, 1 << 20]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        check(f"n={n}", add(x, y), x + y)

    # 带宽：读 x、读 y、写 out，每个元素 3 * 4 字节
    rows = []
    for n in [1 << 16, 1 << 20, 1 << 24, 1 << 26]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        nbytes = 3 * n * 4
        for name, fn in [("triton", lambda: add(x, y)), ("torch", lambda: x + y)]:
            ms = bench(fn)
            rows.append(dict(n=n, impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "vector add 带宽（H100 HBM3 峰值约 3350 GB/s）")
    print("\n观察：n 小的时候带宽很低——数据太少，时间被 launch 开销（几微秒）主导。")
