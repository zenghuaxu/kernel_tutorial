"""练习 02-1：自己写一个正确的 GPU 计时器

目标：实现 time_cuda(fn, warmup, rep, flush_l2)，返回 fn() 在 GPU 上的平均耗时（毫秒）。
测试会拿它和 triton.testing.do_bench 对比三种情况：
  1. 一个 GPU 上空转 ~0.5ms 的 kernel —— 不同步的话你会测到几微秒
  2. 第一次调用要额外花 300ms 的函数（模拟 JIT 编译）—— 不预热的话平均值会被拉高几十倍
  3. 16MB 的 copy —— 数据能放进 L2，不清 L2 会比 do_bench 快 ~20%，超出容差

提示（讲义 2.2 节）：
  - torch.cuda.Event(enable_timing=True)、event.record()、start.elapsed_time(end)
  - event 记录的是 GPU 时间线上的时刻，所以 CPU 端排队多少 kernel 都没关系，最后同步一次即可
  - 每次迭代都 synchronize 也能得到正确结果，但会多出 CPU→GPU 的空隙；想想为什么 event 方式不需要

做完之后想一想：
  - do_bench 返回的是中位数，你返回的是平均数。哪种对偶发的干扰（比如这台机器上其他训练任务）更稳？
  - 表格里 naive 那一列测到的到底是什么？

运行：python 02_profiling/exercises/ex1_event_timer.py
"""
import time

import torch
from triton.testing import do_bench

from common import check, finish, report

_L2_FLUSH = None


def time_cuda(fn, warmup: int = 10, rep: int = 50, flush_l2: bool = True) -> float:
    """返回 fn() 在 GPU 上的平均耗时（毫秒）。"""
    # TODO:
    #   1. 预热 warmup 次（第一次调用可能包含编译/加载）
    #   2. 计时 rep 次：每次用一对 torch.cuda.Event(enable_timing=True) 夹住 fn()
    #      flush_l2=True 时，在每次计时开始前写一遍一个 256MB 的 buffer（比 50MB 的 L2 大）把 L2 挤掉
    #      （提示：buffer 只分配一次，放在模块级变量 _L2_FLUSH 里；清 L2 的那个 kernel 不能在两个 event 之间）
    #   3. torch.cuda.synchronize() 之后用 start.elapsed_time(end) 取毫秒数，返回平均值
    return None


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
