"""练习 03-1：按行 softmax（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, stride_x, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # 越界位置填 -inf：exp(-inf) = 0，不影响 max 也不影响 sum
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    denom = tl.sum(e, axis=0)
    y = e / denom
    tl.store(out_ptr + row * stride_out + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """对最后一维做 softmax。x: [..., N]，最后一维连续。"""
    assert x.stride(-1) == 1
    x2 = x.reshape(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    softmax_kernel[(M,)](x2, out, N, x2.stride(0), out.stride(0), BLOCK=BLOCK, num_warps=num_warps)
    return out.view_as(x)


def ref_softmax(x):
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(1, 1), (3, 7), (64, 128), (100, 1000), (17, 4096), (8, 8192), (2, 3, 513)]:
        x = torch.randn(shape, device="cuda")
        check(f"fp32 {shape}", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    check("bf16 (64, 2048)", softmax(x), ref_softmax(x), atol=1e-2, rtol=1e-2)
    # 数值稳定性：logit 很大时，不减 max 会 exp 溢出成 inf
    x = torch.randn(16, 1000, device="cuda") * 100 + 500
    check("大 logit（数值稳定性）", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)
    # 视图：行之间有间隔
    x = torch.randn(32, 1024, device="cuda")[:, :1000]
    check("行 stride != N 的视图", softmax(x), ref_softmax(x), atol=1e-6, rtol=1e-5)

    rows = []
    compiled = torch.compile(lambda t: torch.softmax(t, dim=-1))
    for M, N in [(4096, 1024), (4096, 4096), (1024, 16384)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        nbytes = 2 * x.numel() * 2
        for name, fn in [("triton", lambda: softmax(x)),
                         ("torch", lambda: torch.softmax(x, dim=-1)),
                         ("torch.compile", lambda: compiled(x))]:
            ms = bench(fn)
            rows.append(dict(shape=f"{M}x{N}", impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, "softmax bf16")
    finish()
