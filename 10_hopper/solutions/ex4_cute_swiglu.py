"""练习 10-4：用 CuTe DSL 写向量化 SwiGLU（参考答案）"""
import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass.cute.runtime import from_dlpack

from common import bench, check, finish, gbps, report

VEC = 8          # 每个线程处理 8 个连续元素（bf16 时 = 16 字节 = 一次 128-bit 访存）
THREADS = 256    # 每个 block 的线程数


@cute.kernel
def swiglu_kernel(gG: cute.Tensor, gU: cute.Tensor, gO: cute.Tensor):
    # gG 等的 layout 是 ((1, VEC), (M, N/VEC))：mode 0 = 一个线程的 VEC 个元素，mode 1 = 第几个"向量"
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    m, n = gG.shape[1]
    mi = i // n
    ni = i % n
    # gG[(None, (mi, ni))]：mode 0 全取（None = 切片里的 ":"），mode 1 取坐标 (mi, ni) → 一个 VEC 长的子 tensor
    g = gG[(None, (mi, ni))].load().to(cutlass.Float32)     # .load() 把它读进寄存器，得到 TensorSSA
    u = gU[(None, (mi, ni))].load().to(cutlass.Float32)
    o = g / (1.0 + cute.math.exp(-g)) * u
    gO[(None, (mi, ni))] = o.to(gO.element_type)


@cute.jit
def swiglu_host(mG: cute.Tensor, mU: cute.Tensor, mO: cute.Tensor):
    tiler = (1, VEC)
    gG = cute.zipped_divide(mG, tiler)
    gU = cute.zipped_divide(mU, tiler)
    gO = cute.zipped_divide(mO, tiler)
    num_vec = cute.size(gO, mode=[1])
    swiglu_kernel(gG, gU, gO).launch(grid=(num_vec // THREADS, 1, 1), block=(THREADS, 1, 1))


_compiled = {}


def swiglu(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    assert g.shape == u.shape and g.dim() == 2 and g.is_contiguous() and u.is_contiguous()
    M, N = g.shape
    assert N % VEC == 0 and (M * N // VEC) % THREADS == 0, "为了简单，要求能整除（不做边界处理）"
    out = torch.empty_like(g)
    args = [from_dlpack(t, assumed_align=16) for t in (g, u, out)]
    key = (M, N, g.dtype)
    if key not in _compiled:                 # cute.compile 按静态 shape 特化，换 shape 要重新编译
        _compiled[key] = cute.compile(swiglu_host, *args)
    _compiled[key](*args)
    return out


def ref(g, u):
    return (F.silu(g.float()) * u.float()).to(g.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N in [(1, 2048), (128, 2048), (1024, 11008)]:
        g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 3
        u = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        check(f"bf16 {M}x{N}", swiglu(g, u), ref(g, u), atol=1e-2, rtol=1e-2)
    g = torch.randn(64, 1024, device="cuda") * 3
    u = torch.randn(64, 1024, device="cuda")
    check("fp32 64x1024", swiglu(g, u), ref(g, u), atol=1e-5, rtol=1e-5)

    M, N = 4096, 8192
    g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    swiglu(g, u)
    rows = []
    for name, fn, nbytes in [("CuTe DSL", lambda: swiglu(g, u), 3 * g.numel() * 2),
                             ("torch eager", lambda: F.silu(g) * u, 5 * g.numel() * 2)]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"SwiGLU {M}x{N} bf16（含 Python 侧 launch 开销）")
    finish()
