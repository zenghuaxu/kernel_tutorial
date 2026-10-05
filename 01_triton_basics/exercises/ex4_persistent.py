"""练习 01-4：persistent kernel

目标：out = alpha * x + beta，但**只启动固定个数的 program**（默认 = SM 个数 132），
每个 program 用循环处理多个 block。

  - wrapper 已写好：grid = (num_programs,)
  - 你写 kernel：program pid 依次处理 block pid, pid + nprog, pid + 2*nprog, ...（grid-stride 循环）
  - 测试会用 grid=1、3、7 这种很小的值来确认你的循环覆盖了所有 block、且没有重复

提示：
  - tl.num_programs(0) 拿到 grid 大小；tl.cdiv(n, BLOCK) 在 kernel 里算 block 数
  - Triton 里 `for i in range(start, end, step):` 的三个参数都可以是运行时值
  - 循环体内就是一个普通的 elementwise block：offs、mask、load、store

做完之后想一想：最后的表格里 grid 取 1x/2x/4x/8x SM 数时带宽有差别吗？
为什么 elementwise 用 persistent 收益不大，但 matmul 会有用（单元 04 再回来看）？

运行：python 01_triton_basics/exercises/ex4_persistent.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def axpb_persistent_kernel(x_ptr, out_ptr, n, alpha, beta, BLOCK: tl.constexpr):
    sm_id = tl.program_id(0)
    sm_num = tl.num_programs(0)
    loop = (n + sm_num * BLOCK - 1) // (sm_num * BLOCK)
    for i in range(0, loop):
        offs = i * sm_num * BLOCK + sm_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, alpha * x + beta, mask=mask)


def axpb(x: torch.Tensor, alpha: float, beta: float, num_programs: int | None = None) -> torch.Tensor:
    """out = alpha * x + beta，只启动 num_programs 个 program（默认 = SM 个数）。"""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    if num_programs is None:
        num_programs = NUM_SMS
    num_programs = max(1, min(num_programs, triton.cdiv(n, BLOCK)))
    axpb_persistent_kernel[(num_programs,)](x, out, n, alpha, beta, BLOCK=BLOCK, num_warps=8)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [5, 4096, 4097, 1_000_003, 1 << 24]:
        x = torch.randn(n, device="cuda")
        check(f"n={n} grid=SMs", axpb(x, 2.0, -1.0), x * 2.0 - 1.0)
    x = torch.randn(123_457, device="cuda")
    for p in [1, 3, 7]:
        check(f"n=123457 grid={p}", axpb(x, 0.5, 3.0, num_programs=p), x * 0.5 + 3.0)

    n = 1 << 26
    x = torch.randn(n, device="cuda")
    rows = []
    for mult in [1, 2, 4, 8]:
        ms = bench(lambda: axpb(x, 2.0, 1.0, num_programs=NUM_SMS * mult))
        rows.append(dict(grid=f"{mult}x SMs", us=ms * 1e3, GBps=gbps(2 * n * 4, ms)))
    ms = bench(lambda: x * 2.0 + 1.0)
    rows.append(dict(grid="torch(2 kernels)", us=ms * 1e3, GBps=gbps(4 * n * 4, ms)))
    report(rows, f"persistent axpb, n=2^26 fp32, SM 数={NUM_SMS}")
    finish()
