"""练习 00-1：roofline 计算器（参考答案）"""
from common import check, check_equal, finish

# 纸笔题统一用 H100 SXM 的峰值，测试答案按它算，不随你的卡变化。
# B200 是 8e12 B/s、2250e12 FLOP/s（ridge ≈ 281 FLOP/B，和 H100 的 295 差不多）：
# 做完后可以换成 B200 的数字再跑一遍表格看看结论变不变（检查会失败，正常）。
PEAK_BW = 3.35e12      # B/s
PEAK_BF16 = 989e12     # FLOP/s


def elementwise_cost(n: int, n_inputs: int, n_outputs: int, flops_per_elem: int, dtype_bytes: int):
    flops = n * flops_per_elem
    nbytes = n * (n_inputs + n_outputs) * dtype_bytes
    return flops, nbytes


def matmul_cost(M: int, N: int, K: int, dtype_bytes: int):
    flops = 2 * M * N * K
    nbytes = (M * K + K * N + M * N) * dtype_bytes
    return flops, nbytes


def attention_cost(B: int, H: int, S: int, D: int, dtype_bytes: int, causal: bool = False):
    # QK^T: 2*S*S*D，PV: 2*S*S*D，每个 (batch, head) 一份
    flops = 4 * B * H * S * S * D
    if causal:
        flops //= 2
    nbytes = 4 * B * H * S * D * dtype_bytes   # 读 Q、K、V，写 O
    return flops, nbytes


def roofline_time_us(flops: float, nbytes: float, peak_flops: float = PEAK_BF16, peak_bw: float = PEAK_BW) -> float:
    return max(flops / peak_flops, nbytes / peak_bw) * 1e6


def is_memory_bound(flops: float, nbytes: float, peak_flops: float = PEAK_BF16, peak_bw: float = PEAK_BW) -> bool:
    return flops / nbytes < peak_flops / peak_bw


def min_m_for_compute_bound(N: int, K: int, dtype_bytes: int = 2, peak_flops: float = PEAK_BF16) -> int:
    """[M, K] @ [K, N]：M 至少多大，这个 GEMM 才是 compute-bound？"""
    M = 1
    while True:
        flops, nbytes = matmul_cost(M, N, K, dtype_bytes)
        if not is_memory_bound(flops, nbytes, peak_flops=peak_flops):
            return M
        M += 1


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
