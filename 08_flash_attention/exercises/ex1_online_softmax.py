"""练习 08-1：分块 online softmax

FlashAttention 的数学核心就是 online softmax。先在一维上把它写对。

目标：一个 program 处理一行 x[row, :N]（N 可能有十几万，放不进一个 block），输出
  - out[row] = softmax(x[row])
  - lse[row] = logsumexp(x[row])（fp32）
做法：两遍扫描
  - pass 1：按 BLOCK 分块扫一遍，在线维护 m（目前的最大值）和 l（Σ exp(x - m)）。
           新块到来：m' = max(m, max(块))；l' = l * exp(m - m') + Σ exp(块 - m')
           扫完：lse = m + log(l)
  - pass 2：再扫一遍，写 exp(x - lse)

提示：
  - 标量状态：m = tl.full([], float("-inf"), dtype=tl.float32)、l = tl.zeros([], dtype=tl.float32)
  - 越界位置 load 成 -inf（other=float("-inf")），exp(-inf) = 0，不影响 max 和 sum
  - 输入可能是 bf16：load 后 .to(tl.float32)
  - 第一块时 m = -inf，exp(m - m') = exp(-inf) = 0，正好把空的 l 清零——不需要特判

做完之后想一想：
  - 测试里有 x*300 和一个 1000 的尖峰。不减 max 直接 exp 会怎样？
  - 和 torch.softmax 比，你的版本多读了一遍输入（3 次访存 vs 2 次）。为什么 torch 能只读一遍？
    （提示：一行 15 万个 bf16 = 300KB，放得进 shared memory 吗？H100 / B200 每 SM 都是 228KB）
  - 单元 03 的 cross-entropy 练习，如果你做过，和这里是什么关系？

运行：python 08_flash_attention/exercises/ex1_online_softmax.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, gpu_name, report


@triton.jit
def online_softmax_kernel(x_ptr, out_ptr, lse_ptr, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o

    # TODO pass 1：分块扫描，在线维护 (m, l)，算出 lse 并写到 lse_ptr + row

    # TODO pass 2：再扫一遍，写 exp(x - lse)（转成输出 dtype）
    pass


def online_softmax(x: torch.Tensor, BLOCK: int = 2048):
    """x: [M, N]，最后一维连续。返回 (softmax(x, -1), logsumexp(x, -1) fp32)。"""
    assert x.dim() == 2 and x.stride(-1) == 1
    M, N = x.shape
    out = torch.empty_like(x)
    lse = torch.empty(M, device=x.device, dtype=torch.float32)
    online_softmax_kernel[(M,)](x, out, lse, N, x.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=8)
    return out, lse


if __name__ == "__main__":
    torch.manual_seed(0)
    for (M, N, scale) in [(4, 1000, 1.0), (64, 32768, 1.0), (8, 100_003, 1.0), (16, 50_000, 300.0)]:
        x = torch.randn(M, N, device="cuda") * scale
        out, lse = online_softmax(x)
        check(f"M{M} N{N} x*{scale} softmax", out, torch.softmax(x, -1), atol=1e-6, rtol=1e-4)
        check(f"M{M} N{N} x*{scale} lse", lse, torch.logsumexp(x, -1), atol=1e-4, rtol=1e-5)
    # 极端：每行只有一个元素特别大（scale=300 时 exp(x) 早就溢出 fp32 了）
    x = torch.zeros(4, 10_000, device="cuda")
    x[:, 1234] = 1000.0
    out, lse = online_softmax(x)
    check("one-hot 1000", out, torch.softmax(x, -1), atol=1e-6, rtol=1e-4)
    # bf16 输入：内部 fp32 计算
    x = torch.randn(32, 151_936, device="cuda", dtype=torch.bfloat16) * 5   # Qwen 词表大小
    out, lse = online_softmax(x)
    check("bf16 vocab=151936 softmax", out, torch.softmax(x.float(), -1).to(torch.bfloat16), atol=1e-4, rtol=1e-2)
    check("bf16 vocab=151936 lse", lse, torch.logsumexp(x.float(), -1), atol=1e-3, rtol=1e-4)

    M, N = 256, 151_936
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rows = []
    for name, fn, nbytes in [
        ("online (2 次读 + 1 次写)", lambda: online_softmax(x), 3 * M * N * 2),
        ("torch.softmax", lambda: torch.softmax(x, -1), 2 * M * N * 2),
    ]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"softmax M={M} N={N} bf16（实测于 {gpu_name()}）")
    finish()
