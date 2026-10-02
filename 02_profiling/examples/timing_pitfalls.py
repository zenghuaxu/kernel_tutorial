"""示例：GPU 计时的 4 个坑，以及正确的做法。

运行：python 02_profiling/examples/timing_pitfalls.py
"""
import time

import torch
import triton
import triton.language as tl
from triton.testing import do_bench


@triton.jit
def copy_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def triton_copy(x, out, block):
    copy_kernel[(triton.cdiv(x.numel(), block),)](x, out, x.numel(), BLOCK=block)


def events_timer(fn, warmup=10, rep=50, flush_l2=True):
    """正确的计时：预热 + CUDA event + （可选）每次清 L2。返回平均毫秒。"""
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device="cuda")  # 256MB > 50MB L2
    for _ in range(warmup):
        fn()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        if flush_l2:
            cache.zero_()          # 把 L2 里的数据挤掉；这个 kernel 不在 event 区间里
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()       # 等所有 event 都真正发生
    return sum(s.elapsed_time(e) for s, e in zip(starts, ends)) / rep


if __name__ == "__main__":
    torch.manual_seed(0)

    print("== 坑 1：kernel launch 是异步的 ==")
    sleep = lambda: torch.cuda._sleep(1_000_000)   # GPU 上空转一百万个时钟周期（约 0.5ms）
    sleep()                                         # 先调一次：第一次调用会加载 kernel 模块，本身就慢（坑 2）
    torch.cuda.synchronize()
    t = time.perf_counter(); sleep(); t_nosync = (time.perf_counter() - t) * 1e3
    torch.cuda.synchronize()
    t = time.perf_counter(); sleep(); torch.cuda.synchronize(); t_sync = (time.perf_counter() - t) * 1e3
    print(f"  不 synchronize: {t_nosync:.3f} ms   ← 只测到了 CPU 把 kernel 塞进队列的时间")
    print(f"  synchronize 后: {t_sync:.3f} ms")
    print(f"  do_bench      : {do_bench(sleep):.3f} ms")

    print("\n== 坑 2：第一次调用包含 JIT 编译 / 加载 ==")
    x = torch.randn(1 << 22, device="cuda")
    out = torch.empty_like(x)
    block = [1024, 2048, 4096][int(time.time()) % 3]   # 换个 constexpr，尽量逼它重新特化
    torch.cuda.synchronize()
    t = time.perf_counter(); triton_copy(x, out, block); torch.cuda.synchronize()
    print(f"  第一次调用: {(time.perf_counter() - t) * 1e3:8.3f} ms  (BLOCK={block}，含编译或读磁盘缓存)")
    t = time.perf_counter(); triton_copy(x, out, block); torch.cuda.synchronize()
    print(f"  第二次调用: {(time.perf_counter() - t) * 1e3:8.3f} ms  (含 Python launch 开销 + sync 往返)")
    print(f"  do_bench  : {do_bench(lambda: triton_copy(x, out, block)):8.3f} ms")

    print("\n== 坑 3：数据还在 L2 里（H100 L2 = 50MB）==")
    for mb in [16, 128]:
        n = mb * 1024 * 1024 // 4
        a = torch.randn(n, device="cuda")
        b = torch.empty_like(a)
        fn = lambda: b.copy_(a)
        hot = events_timer(fn, flush_l2=False)
        cold = events_timer(fn, flush_l2=True)
        print(f"  copy {mb:4d}MB: 不清 L2 {hot * 1e3:7.1f} us ({2 * mb / 1024 / hot * 1e3:5.0f} GB/s)   "
              f"清 L2 {cold * 1e3:7.1f} us ({2 * mb / 1024 / cold * 1e3:5.0f} GB/s)   do_bench {do_bench(fn) * 1e3:7.1f} us")
    print("  → 能装进 L2 的数据反复跑，会命中 L2，测出偏乐观的数字（16MB 时快 ~20%）；")
    print("    128MB 远大于 L2，清不清都一样。真实模型里上一层的输出未必还在 L2，所以 benchmark 默认要清。")

    print("\n== 坑 4：太小的 kernel，测到的是 launch 开销 ==")
    tiny = torch.randn(1024, device="cuda")
    print(f"  1024 个元素的 copy: {do_bench(lambda: triton_copy(tiny, tiny, 1024)) * 1e3:.1f} us（实际 GPU 干活不到 1us）")
    print("  → 小 kernel 优化方向不是 kernel 本身，而是融合、CUDA Graph（triton.testing.do_bench_cudagraph）")
