"""练习 02-4：用 torch.profiler 数 kernel

一段 PyTorch 代码会启动几个 kernel？这是判断"值不值得写融合 kernel"的第一步。

目标：实现 gpu_kernels(fn)，返回调用一次 fn() 时 GPU 上执行的 kernel 名字列表。
  - 不算 record_function 在 GPU 时间线上留下的区间（它们也是 device_type == CUDA 的事件）
  - 不算 Memcpy / Memset（它们是拷贝引擎的操作，不是 kernel）
测试里的期望值都是"一个 eager 算子 = 一个 kernel"推出来的，你可以先自己猜每个用例是几个再跑。

提示：讲义 2.3 节、examples/torch_profiler_demo.py

做完之后想一想：
  - eager RMSNorm 6 个 kernel，每个都要把 [1024, 1024] 的 fp32 读写一遍 HBM。融合成 1 个能省多少字节？
    （单元 03 你会亲手写这个 kernel）
  - 把 rmsnorm_eager 用 torch.compile 包一下再数，会是几个？名字长什么样？

运行：python 02_profiling/exercises/ex4_count_kernels.py
"""
import warnings

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile, record_function

from common import check_equal, finish

warnings.filterwarnings("ignore")


def gpu_kernels(fn) -> list[str]:
    """返回调用一次 fn() 时 GPU 上执行的 kernel 名字列表（按时间顺序）。

    不算：record_function 在 GPU 时间线上留下的区间、Memcpy、Memset。
    """
    # TODO:
    #   1. 先调用一次 fn() 并同步（预热）
    #   2. with profile(activities=[ProfilerActivity.CUDA]) as prof: 里调用一次 fn()，再同步
    #   3. 从 prof.events() 里挑出 GPU 上的事件（e.device_type == torch.autograd.DeviceType.CUDA），
    #      去掉 e.is_user_annotation 为真的（record_function 区间），以及名字以 "Memcpy"/"Memset" 开头的
    #   4. 按开始时间（e.time_range.start）排序，返回名字列表
    return []


@triton.jit
def my_fused_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * tl.sigmoid(x) * 2.0 + 1.0, mask=mask)


def rmsnorm_eager(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(1024, 1024, device="cuda")
    b = torch.randn(1024, 1024, device="cuda")
    w = torch.randn(1024, device="cuda")
    out = torch.empty_like(a)
    cpu = torch.randn(1024, 1024)

    def annotated():
        with record_function("my_region"):
            return a + b

    cases = [
        ("a + b", lambda: a + b, 1),
        ("F.silu(a) * b", lambda: F.silu(a) * b, 2),
        ("(a + b) * a - b", lambda: (a + b) * a - b, 3),
        ("eager RMSNorm", lambda: rmsnorm_eager(a, w), 6),
        ("record_function 包着 a + b", annotated, 1),
        ("CPU→GPU 拷贝（是 Memcpy 不是 kernel）", lambda: out.copy_(cpu), 0),
        ("一个 Triton kernel", lambda: my_fused_kernel[(1024,)](a, out, a.numel(), BLOCK=1024), 1),
    ]
    for name, fn, expect in cases:
        names = gpu_kernels(fn)
        check_equal(f"{name}: kernel 个数", len(names), expect)
        for n in names:
            print(f"      {n[:100]}")
    names = gpu_kernels(lambda: my_fused_kernel[(1024,)](a, out, a.numel(), BLOCK=1024))
    check_equal("Triton kernel 的名字就是函数名", names[0] if names else None, "my_fused_kernel")
    finish()
