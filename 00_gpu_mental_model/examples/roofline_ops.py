"""示例：把几个常见算子放到 roofline 上，看它们离上限有多远。

运行：python 00_gpu_mental_model/examples/roofline_ops.py

对每个算子手算：FLOPs、最少要搬运的字节数 → 算术强度 AI = FLOPs / bytes
roofline 下界时间 = max(FLOPs / 峰值算力, bytes / 峰值带宽)
"""
import torch
import torch.nn.functional as F

from common import bench, report

PEAK_BW = 3.35e12        # B/s，H100 SXM HBM3
PEAK_BF16 = 989e12       # FLOP/s，bf16 dense Tensor Core
RIDGE = PEAK_BF16 / PEAK_BW

if __name__ == "__main__":
    torch.manual_seed(0)
    bf = torch.bfloat16
    E = 2  # bf16 字节数
    T, D, H = 8192, 4096, 14336   # tokens, hidden, MLP intermediate

    x = torch.randn(T, D, device="cuda", dtype=bf)
    y = torch.randn(T, D, device="cuda", dtype=bf)
    w = torch.randn(D, device="cuda", dtype=bf)
    W1 = torch.randn(D, H, device="cuda", dtype=bf)
    v = torch.randn(1, D, device="cuda", dtype=bf)

    def rmsnorm():
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(bf) * w

    ops = [
        # name, fn, flops, bytes
        ("x + y  (elementwise)", lambda: x + y, T * D, 3 * T * D * E),
        ("softmax(x, -1)", lambda: torch.softmax(x, -1), 5 * T * D, 2 * T * D * E),
        ("RMSNorm (eager, 多个kernel)", rmsnorm, 4 * T * D, 2 * T * D * E + D * E),
        ("GEMV 1x4096 @ 4096x14336 (decode)", lambda: v @ W1, 2 * D * H, (D + D * H + H) * E),
        ("GEMM 8192x4096 @ 4096x14336 (prefill)", lambda: x @ W1, 2 * T * D * H, (T * D + D * H + T * H) * E),
    ]
    rows = []
    for name, fn, flops, nbytes in ops:
        ms = bench(fn)
        ai = flops / nbytes
        t_bound = max(flops / PEAK_BF16, nbytes / PEAK_BW) * 1e3  # ms
        rows.append(dict(op=name, AI=ai, bound="compute" if ai > RIDGE else "memory",
                         roofline_us=t_bound * 1e3, measured_us=ms * 1e3, pct_of_roofline=100 * t_bound / ms))
    report(rows, f"roofline（ridge point = {RIDGE:.0f} FLOP/B）")
    print("""
怎么读这张表：
  - AI 远小于 ridge 的都是 memory-bound：优化目标是"少搬字节"（融合、低精度），而不是"少算"。
  - pct_of_roofline 接近 100% 说明已经贴着硬件上限，再优化这个 kernel 本身没意义。
  - eager RMSNorm 的百分比很低：它被拆成好几个 kernel，每个都把 [T, D] 读写一遍。
    单元 03 你会写一个融合版本，把它拉到 80%+。
  - decode 时的 GEMV（batch=1）AI≈1，比 ridge 低两个数量级：Tensor Core 基本闲着，
    时间全花在读权重上。这就是 speculative decoding 能加速的根本原因（练习 ex2）。""")
