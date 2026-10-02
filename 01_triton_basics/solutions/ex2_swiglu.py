"""练习 01-2：融合 SwiGLU 激活（参考答案）"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def swiglu_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(u_ptr + offs, mask=mask).to(tl.float32)
    # silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    out = g * tl.sigmoid(g) * u
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.shape == u.shape and g.is_contiguous() and u.is_contiguous()
    out = torch.empty_like(g)
    n = g.numel()
    BLOCK = 2048
    swiglu_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, BLOCK=BLOCK, num_warps=8)
    return out


def ref_swiglu(g, u):
    return (F.silu(g.float()) * u.float()).to(g.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(7,), (3, 1000), (4, 512, 1408)]:
        g = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 3
        u = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        check(f"bf16 {shape}", swiglu(g, u), ref_swiglu(g, u), atol=1e-2, rtol=1e-2)
    g = torch.randn(1000, device="cuda") * 3
    u = torch.randn(1000, device="cuda")
    check("fp32 (1000,)", swiglu(g, u), ref_swiglu(g, u), atol=1e-5, rtol=1e-5)

    # 性能：bf16，8M 个元素
    n = 8 * 1024 * 1024
    g = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    rows = []
    for name, fn, nbytes in [
        ("triton fused", lambda: swiglu(g, u), 3 * n * 2),        # 读 g、u，写 out
        ("torch eager", lambda: F.silu(g) * u, 5 * n * 2),         # silu: 读g写tmp；mul: 读tmp、u，写out
    ]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, bytes_MB=nbytes / 1e6, GBps=gbps(nbytes, ms)))
    report(rows, "SwiGLU（GBps 按各自实际搬运的字节算）")
    finish()
