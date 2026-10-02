"""练习 00-2：decode 的性能上限与 speculative decoding（参考答案）"""
from common import check, check_equal, finish

PEAK_BW = 3.35e12      # B/s
PEAK_BF16 = 989e12     # FLOP/s

# Llama-3-8B 量级的配置
N_PARAMS = 8.0e9
N_LAYERS = 32
N_KV_HEADS = 8
HEAD_DIM = 128


def kv_cache_bytes(batch: int, seq_len: int, kv_bytes: int = 2) -> float:
    """batch 条序列、每条 seq_len 个 token 的 KV cache 总字节数（所有层，K 和 V 都算）。"""
    return 2 * N_LAYERS * N_KV_HEADS * HEAD_DIM * seq_len * batch * kv_bytes


def step_time_us(batch: int, seq_len: int, new_tokens: int = 1, weight_bytes: int = 2, kv_bytes: int = 2) -> float:
    """一次前向的 roofline 下界：batch 条序列，每条处理 new_tokens 个新 token，已有 seq_len 的 KV cache。

    FLOPs = 2 * N_PARAMS * (batch * new_tokens)                （忽略 attention 本身的 FLOPs）
    bytes = 权重读一遍 + KV cache 读一遍                          （忽略激活值）
    """
    flops = 2 * N_PARAMS * batch * new_tokens
    nbytes = N_PARAMS * weight_bytes + kv_cache_bytes(batch, seq_len, kv_bytes)
    return max(flops / PEAK_BF16, nbytes / PEAK_BW) * 1e6


def tokens_per_sec(batch: int, seq_len: int) -> float:
    """普通 decode 的吞吐上限（每秒生成的 token 总数，所有序列加起来）。"""
    return batch / (step_time_us(batch, seq_len) * 1e-6)


def min_batch_for_compute_bound(seq_len: int, max_batch: int = 100_000):
    """decode 一步变成 compute-bound 的最小 batch；到 max_batch 还不行就返回 None。"""
    for b in range(1, max_batch + 1):
        flops = 2 * N_PARAMS * b
        nbytes = N_PARAMS * 2 + kv_cache_bytes(b, seq_len)
        if flops / nbytes >= PEAK_BF16 / PEAK_BW:
            return b
    return None


def expected_tokens_per_verify(alpha: float, k: int) -> float:
    """draft 一次猜 k 个 token，每个被接受的概率独立为 alpha。
    一次 verify 平均产出多少个 token（含 target 自己补的那 1 个）？= 1 + alpha + ... + alpha^k"""
    if alpha == 1.0:
        return float(k + 1)
    return (1 - alpha ** (k + 1)) / (1 - alpha)


def spec_decode_speedup(alpha: float, k: int, draft_cost: float, batch: int = 1, seq_len: int = 2048) -> float:
    """相对普通 decode 的加速比。

    普通 decode：每个 token 花 t1 = step_time_us(batch, seq_len, 1)
    speculative：一轮 = draft 跑 k 步（每步 draft_cost * t1）+ target 一次验证 k+1 个 token
                 （step_time_us(batch, seq_len, k + 1)），平均产出 expected_tokens_per_verify 个 token
    """
    t1 = step_time_us(batch, seq_len, 1)
    t_round = k * draft_cost * t1 + step_time_us(batch, seq_len, k + 1)
    return expected_tokens_per_verify(alpha, k) * t1 / t_round


if __name__ == "__main__":
    check_equal("KV cache: 1 条序列 4096 token", kv_cache_bytes(1, 4096), 536870912)
    check("step_time batch=1 seq=0 (us)", step_time_us(1, 0), 4776.1194, atol=1e-3, rtol=0)
    check("step_time batch=64 seq=4096 (us)", step_time_us(64, 4096), 15032.7577, atol=1e-3, rtol=0)
    check("tokens/s batch=1 seq=2048", tokens_per_sec(1, 2048), 205.9202, atol=1e-3, rtol=0)
    check("tokens/s batch=64 seq=2048", tokens_per_sec(64, 2048), 6461.7494, atol=1e-3, rtol=0)
    check_equal("compute-bound 的最小 batch (seq=0)", min_batch_for_compute_bound(0), 296)
    check_equal("compute-bound 的最小 batch (seq=4096)", min_batch_for_compute_bound(4096), None)
    check("期望 token 数 alpha=0.8 k=4", expected_tokens_per_verify(0.8, 4), 3.3616, atol=1e-6, rtol=0)
    check("期望 token 数 alpha=1 k=4", expected_tokens_per_verify(1.0, 4), 5.0, atol=1e-9, rtol=0)
    check("加速比 alpha=0.8 k=4 c=0.05 batch=1", spec_decode_speedup(0.8, 4, 0.05), 2.80133, atol=1e-4, rtol=0)
    check("加速比 alpha=0.6 k=8 c=0.05 batch=1", spec_decode_speedup(0.6, 8, 0.05), 1.76772, atol=1e-4, rtol=0)
    check("加速比 batch=256 seq=128（短上下文）", spec_decode_speedup(0.8, 4, 0.05, batch=256, seq_len=128),
          0.92910, atol=1e-4, rtol=0)
    check("加速比 batch=256 seq=2048（长上下文）", spec_decode_speedup(0.8, 4, 0.05, batch=256, seq_len=2048),
          2.80133, atol=1e-4, rtol=0)

    print("\n参考：alpha=0.8, k=4, draft_cost=0.05 时，加速比随 batch 的变化")
    for seq in [128, 2048]:
        cells = [f"b={b}:{spec_decode_speedup(0.8, 4, 0.05, batch=b, seq_len=seq):.2f}" for b in [1, 32, 64, 128, 256]]
        print(f"  seq_len={seq:5d}  " + "  ".join(cells))
    print("\n参考：batch=1 时不同 k 的加速比（alpha=0.8, draft_cost=0.05）")
    for k in [1, 2, 4, 6, 8, 12]:
        print(f"  k={k:2d}  speedup={spec_decode_speedup(0.8, k, 0.05):.2f}")
    finish()
