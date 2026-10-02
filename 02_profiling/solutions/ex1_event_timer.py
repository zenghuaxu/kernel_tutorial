"""练习 02-1：自己写一个正确的 GPU 计时器（参考答案）"""
import time

import torch
from triton.testing import do_bench

from common import check, finish, report

_L2_FLUSH = None


def time_cuda(fn, warmup: int = 10, rep: int = 50, flush_l2: bool = True) -> float:
    """返回 fn() 在 GPU 上的平均耗时（毫秒）。"""
    global _L2_FLUSH
    if flush_l2 and _L2_FLUSH is None:
        _L2_FLUSH = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        if flush_l2:
            _L2_FLUSH.zero_()
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / rep


def naive_time(fn) -> float:
    """反面教材：不预热、不同步、用 CPU 时钟。"""
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1e3


class SlowFirstCall:
    """模拟一个第一次调用要 JIT 编译 300ms 的函数。"""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls == 1:
            time.sleep(0.3)
        torch.cuda._sleep(200_000)


if __name__ == "__main__":
    rows = []
    # 1. 纯 GPU 耗时的 kernel：不同步就会测成几微秒
    sleep = lambda: torch.cuda._sleep(1_000_000)
    ref = do_bench(sleep)
    mine = time_cuda(sleep)
    rows.append(dict(case="sleep 1M cycles", naive=naive_time(sleep), mine=mine, do_bench=ref))
    check("sleep kernel ≈ do_bench", mine, ref, atol=0, rtol=0.1)

    # 2. 第一次调用很慢：必须预热
    slow = SlowFirstCall()
    mine = time_cuda(slow)
    ref = do_bench(lambda: torch.cuda._sleep(200_000))
    rows.append(dict(case="slow first call", naive=float("nan"), mine=mine, do_bench=ref))
    check("首次调用被预热排除", mine, ref, atol=0, rtol=0.1)

    # 3. 16MB copy：能放进 50MB 的 L2，不清 L2 会偏快
    x = torch.randn(4 * 1024 * 1024, device="cuda")
    y = torch.empty_like(x)
    copy = lambda: y.copy_(x)
    ref = do_bench(copy)                     # do_bench 每次都会清 L2
    mine = time_cuda(copy)
    hot = time_cuda(copy, flush_l2=False)
    rows.append(dict(case="copy 16MB", naive=naive_time(copy), mine=mine, do_bench=ref))
    rows.append(dict(case="copy 16MB (不清L2)", naive=float("nan"), mine=hot, do_bench=ref))
    check("L2 已清空 ≈ do_bench", mine, ref, atol=0, rtol=0.12)

    report(rows, "毫秒")
    finish()
