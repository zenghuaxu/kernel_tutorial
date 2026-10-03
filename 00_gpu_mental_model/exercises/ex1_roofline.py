"""练习 00-1：roofline 计算器

纯 Python 练习，不写 kernel。把讲义 0.5 节的公式写成函数，后面每个单元都会用它来判断"我的 kernel 离上限多远"。

要实现：
  - elementwise_cost(n, n_inputs, n_outputs, flops_per_elem, dtype_bytes) -> (flops, bytes)
  - matmul_cost(M, N, K, dtype_bytes) -> (flops, bytes)          [M,K] @ [K,N]，一次乘加算 2 FLOPs
  - attention_cost(B, H, S, D, dtype_bytes, causal) -> (flops, bytes)
        只算 QK^T 和 PV 两个矩阵乘的 FLOPs；bytes 按 FlashAttention 的理想情况：读 Q、K、V，写 O，各一次
        causal=True 时 FLOPs 减半（整数除法），bytes 不变
  - roofline_time_us(flops, bytes, peak_flops, peak_bw) -> 下界时间（微秒）
  - is_memory_bound(flops, bytes, peak_flops, peak_bw) -> bool    AI 严格小于 ridge point 时为 True
  - min_m_for_compute_bound(N, K, dtype_bytes, peak_flops) -> 使 [M,K]@[K,N] 不再 memory-bound 的最小 M
        （从 M=1 往上试就行）

bytes 一律按"每个输入读一次、每个输出写一次"算（理想下界）。函数返回 Python int / float / bool / tuple。

做完之后想一想：
  - 最后一个测试：FP8（字节减半、峰值算力翻倍）下，临界 M 竟然和 bf16 一样是 345。为什么？
    这对"用 FP8 加速 decode"意味着什么（decode 的 M 很小）？FP8 对 decode 其实是靠什么加速的？
  - attention 的 AI 和 S 成正比。S 多大时变成 compute-bound？sliding window（窗口 W）会怎么改变这个结论？

运行：python 00_gpu_mental_model/exercises/ex1_roofline.py
"""
from common import check, check_equal, finish

PEAK_BW = 3.35e12      # B/s
PEAK_BF16 = 989e12     # FLOP/s


def elementwise_cost(n: int, n_inputs: int, n_outputs: int, flops_per_elem: int, dtype_bytes: int):
    return (flops_per_elem * n, (n_inputs + n_outputs) * n * dtype_bytes)

def matmul_cost(M: int, N: int, K: int, dtype_bytes: int):
    return (2 * M * N * K, dtype_bytes * (M * N + N * K + M * K))

def attention_cost(B: int, H: int, S: int, D: int, dtype_bytes: int, causal: bool = False):
    if not causal:
        return (4 * B * H * S**2 * D, 4 * dtype_bytes * B * H * S * D)
    else:
        return (2 * B * H * S**2 * D, 4 * dtype_bytes * B * H * S * D)

def roofline_time_us(flops: float, nbytes: float, peak_flops: float = PEAK_BF16, peak_bw: float = PEAK_BW) -> float:
    return max(flops / peak_flops, nbytes / peak_bw) * 1e6


def is_memory_bound(flops: float, nbytes: float, peak_flops: float = PEAK_BF16, peak_bw: float = PEAK_BW) -> bool:
    return flops / peak_flops < nbytes / peak_bw


def min_m_for_compute_bound(N: int, K: int, dtype_bytes: int = 2, peak_flops: float = PEAK_BF16) -> int:
    """[M, K] @ [K, N]：M 至少多大，这个 GEMM 才是 compute-bound？"""
    M = 1
    while 1:
        if not is_memory_bound(*matmul_cost(M, N, K, dtype_bytes), peak_flops):
            return M
        M = M + 1


if __name__ == "__main__":
    # elementwise
    check_equal("add fp32 1M: (flops, bytes)", elementwise_cost(1 << 20, 2, 1, 1, 4), (1048576, 12582912))
    check_equal("gelu bf16 1M", elementwise_cost(1 << 20, 1, 1, 8, 2), (8388608, 4194304))
    # matmul
    check_equal("matmul 4096^3 bf16", matmul_cost(4096, 4096, 4096, 2), (137438953472, 100663296))
    check_equal("matmul decode 1x4096 @ 4096x14336", matmul_cost(1, 14336, 4096, 2), (117440512, 117477376))
    # attention
    check_equal("attn B2 H32 S4096 D128", attention_cost(2, 32, 4096, 128, 2), (549755813888, 268435456))
    check_equal("attn causal", attention_cost(2, 32, 4096, 128, 2, causal=True), (274877906944, 268435456))
    # roofline
    f, b = matmul_cost(4096, 4096, 4096, 2)
    check("roofline matmul 4096^3 (us)", roofline_time_us(f, b), 138.96760, atol=1e-3, rtol=0)
    f, b = elementwise_cost(1 << 26, 2, 1, 1, 2)
    check("roofline add bf16 64M (us)", roofline_time_us(f, b), 120.19498, atol=1e-3, rtol=0)
    check_equal("add 是 memory-bound", is_memory_bound(f, b), True)
    check_equal("matmul 4096^3 不是 memory-bound", is_memory_bound(*matmul_cost(4096, 4096, 4096, 2)), False)
    check_equal("attn S=4096 不是 memory-bound", is_memory_bound(*attention_cost(1, 1, 4096, 128, 2)), False)
    check_equal("attn S=256 是 memory-bound", is_memory_bound(*attention_cost(1, 1, 256, 128, 2)), True)
    # 临界 M
    check_equal("min M, [M,4096]@[4096,14336] bf16", min_m_for_compute_bound(14336, 4096), 326)
    check_equal("min M, [M,4096]@[4096,4096] bf16", min_m_for_compute_bound(4096, 4096), 345)
    check_equal("min M, [M,4096]@[4096,4096] fp8 (1 字节, 峰值翻倍)",
                min_m_for_compute_bound(4096, 4096, dtype_bytes=1, peak_flops=2 * PEAK_BF16), 345)
    finish()
