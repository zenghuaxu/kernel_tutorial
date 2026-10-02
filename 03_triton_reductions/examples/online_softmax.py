"""示例：online softmax —— 一边扫一边更新 max 和 sum。

第 1 部分（纯 PyTorch，CPU）：验证 online 递推和普通的"先求 max 再求 sum"结果一样。
第 2 部分（Triton）：行很长时（N=131072），"整行一个 block"的 softmax 会寄存器溢出；
                    改成在行内循环 + online 递推后恢复正常。

运行：python 03_triton_reductions/examples/online_softmax.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, gbps, report


# ---------------- 第 1 部分：递推公式 ----------------
def online_logsumexp(x: torch.Tensor, chunk: int):
    """把 x 切成若干块，逐块更新 (m, s)：m = 目前为止的最大值，s = sum(exp(x_i - m))。"""
    m = torch.tensor(float("-inf"), dtype=torch.float64)
    s = torch.tensor(0.0, dtype=torch.float64)
    for start in range(0, x.numel(), chunk):
        blk = x[start:start + chunk].double()
        m_new = torch.maximum(m, blk.max())
        # 旧的 s 是以 m 为基准累加的：sum(exp(x_i - m)) * exp(m - m_new) = sum(exp(x_i - m_new))
        s = s * torch.exp(m - m_new) + torch.exp(blk - m_new).sum()
        m = m_new
    return m, s


# ---------------- 第 2 部分：两种 Triton softmax ----------------
@triton.jit
def softmax_single(x_ptr, out_ptr, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=float("-inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))         # x 被用了两次：整行都得留在寄存器里
    tl.store(out_ptr + row * stride + offs, (e / tl.sum(e, axis=0)).to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def softmax_online(x_ptr, out_ptr, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = x_ptr + row * stride
    # 第一遍：online 求 (m, s)。这里用"逐元素"的向量 m_i / s_i，最后再合并——比每轮都做一次标量归约更省
    m_i = tl.full([BLOCK], float("-inf"), tl.float32)
    s_i = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m_i, x)
        # m_new 可能还是 -inf（这个 lane 到目前为止全被 mask 掉），exp(-inf - -inf) = nan，要防一下
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        s_i = s_i * alpha + tl.where(m_new == float("-inf"), 0.0, tl.exp(x - m_new))
        m_i = m_new
    m = tl.max(m_i, axis=0)
    s = tl.sum(s_i * tl.exp(m_i - m), axis=0)
    # 第二遍：写出 exp(x - m) / s
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(out_ptr + row * stride + offs, (tl.exp(x - m) / s).to(out_ptr.dtype.element_ty), mask=mask)


if __name__ == "__main__":
    torch.manual_seed(0)
    print("第 1 部分：online 递推 vs 直接计算")
    x = torch.randn(10000) * 10
    for chunk in [1, 7, 1000, 10000]:
        m, s = online_logsumexp(x, chunk)
        print(f"  chunk={chunk:5d}: logsumexp online = {(m + s.log()).item():.10f}   "
              f"torch = {torch.logsumexp(x.double(), 0).item():.10f}")

    print("\n第 2 部分：长行 softmax")
    rows = []
    for M, N in [(4096, 4096), (512, 32768), (128, 131072)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        out = torch.empty_like(x)
        ref = torch.softmax(x.float(), -1).to(x.dtype)
        B = triton.next_power_of_2(N)
        variants = [("single", lambda: softmax_single[(M,)](x, out, N, N, BLOCK=B, num_warps=16)),
                    ("online", lambda: softmax_online[(M,)](x, out, N, N, BLOCK=2048, num_warps=8))]
        for name, fn in variants:
            k = fn()
            check(f"{name} {M}x{N}", out, ref, atol=1e-2, rtol=1e-2)
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, regs=k.n_regs, spills=k.n_spills,
                             us=ms * 1e3, GBps=gbps(2 * x.numel() * 2, ms)))
        ms = bench(lambda: torch.softmax(x, -1))
        rows.append(dict(shape=f"{M}x{N}", impl="torch", regs="-", spills="-", us=ms * 1e3,
                         GBps=gbps(2 * x.numel() * 2, ms)))
    report(rows, "softmax bf16（GBps 按 读一遍 + 写一遍 算）")
    print("\n看点：online 版本要读两遍 x（第一遍求 m、s，第二遍写结果），所以短行时比 single 慢；")
    print("      但它的寄存器用量和 N 无关，长行不会 spill。FlashAttention 把第二遍也省掉了——")
    print("      因为它要的不是 softmax 本身，而是 softmax @ V，可以边扫边把 V 也按同样的比例缩放（单元 08）。")
    print("      128x131072 时 online 也只有 ~730 GB/s，torch 更快：只有 128 个 program、每个串行扫 13 万个元素，")
    print("      并行度不够（同 reduce_basics.py 的看点 2）。解决办法是把一行拆给多个 program，再合并各自的 (m, s)。")
