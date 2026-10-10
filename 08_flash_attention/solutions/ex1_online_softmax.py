"""练习 08-1：分块 online softmax（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, gpu_name, report


@triton.jit
def online_softmax_kernel(x_ptr, out_ptr, lse_ptr, N, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    o_row = out_ptr + row * stride_o

    # pass 1：一遍扫完整行，在线维护 (m, l)
    #   m = 目前见过的最大值，l = Σ exp(x - m)
    #   新块到来：m' = max(m, max(块))，l' = l * exp(m - m') + Σ exp(块 - m')
    m = tl.full([], float("-inf"), dtype=tl.float32)
    l = tl.zeros([], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        l = l * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(l)
    tl.store(lse_ptr + row, lse)

    # pass 2：softmax(x) = exp(x - lse)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < N, other=float("-inf")).to(tl.float32)
        tl.store(o_row + offs, tl.exp(x - lse).to(out_ptr.dtype.element_ty), mask=offs < N)


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
