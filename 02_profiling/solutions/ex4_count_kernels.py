"""练习 02-4：用 torch.profiler 数 kernel（参考答案）"""
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
    fn()                       # 预热：第一次调用可能有一次性的初始化 kernel
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    events = [e for e in prof.events()
              if e.device_type == torch.autograd.DeviceType.CUDA
              and not e.is_user_annotation
              and not e.name.startswith(("Memcpy", "Memset"))]
    events.sort(key=lambda e: e.time_range.start)
    return [e.name for e in events]


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
