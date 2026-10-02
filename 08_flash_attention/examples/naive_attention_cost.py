"""示例：朴素 attention 的时间和显存随 N 怎么涨，和 SDPA（FlashAttention 后端）对比。

运行：python 08_flash_attention/examples/naive_attention_cost.py

朴素实现：S = QK^T 写进 HBM（B*H*N*N 个元素）→ softmax 读 S 写 P → P·V 再读 P。
FlashAttention：S/P 只存在于片上（寄存器 / shared memory），HBM 流量只有 Q、K、V、O。
"""
import math

import torch
import torch.nn.functional as F

from common import bench, report, tflops


def naive_attention(q, k, v):
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    p = torch.softmax(s.float(), -1).to(q.dtype)
    return p @ v


def peak_mem_mb(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, D = 1, 8, 128
    rows = []
    for N in [512, 1024, 2048, 4096]:   # N=8192 时 naive 要 ~5GB 额外显存，共享卡上放不下
        q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        flops = 4 * B * H * N * N * D
        s_mb = B * H * N * N * 2 / 2**20
        for name, fn in [("naive", lambda: naive_attention(q, k, v)),
                         ("SDPA", lambda: F.scaled_dot_product_attention(q, k, v))]:
            ms = bench(fn)
            rows.append(dict(N=N, impl=name, ms=ms, TFLOPs=tflops(flops, ms),
                             peak_extra_MB=peak_mem_mb(fn), S_matrix_MB=s_mb))
    report(rows, f"non-causal B={B} H={H} D={D} bf16（实测于共享 H100）")
    print("\n看点：naive 的额外显存 ≈ S 矩阵的若干倍（S、fp32 的 softmax 中间结果、P），且随 N² 增长；")
    print("SDPA 的额外显存只有输出 O 那么大。naive 的 TFLOPs 也远低于 SDPA：时间都花在搬 S/P 上了。")
