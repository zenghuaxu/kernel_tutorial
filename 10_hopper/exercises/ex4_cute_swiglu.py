"""练习 10-4：用 CuTe DSL 写向量化 SwiGLU

CuTe DSL 是 CUTLASS 4 的 Python 前端，写法比 Triton 底层：你直接管每个**线程**做什么（像 CUDA），
但用 Layout 代数来描述数据怎么切、怎么分给线程（这是 CUTLASS/FA3 的核心抽象）。
先跑 examples/cute_layouts.py 看看 Layout 是什么。

本练习做一个最小但完整的 CuTe kernel：out = silu(g) * u（单元 01 练习 2 的 CuTe 版），每个线程处理 VEC=8 个连续元素。

目标：
  1. host 函数 swiglu_host（@cute.jit）：用 cute.zipped_divide(mX, (1, VEC)) 把三个 [M, N] tensor 切成
     ((1, VEC), (M, N/VEC)) 的形状——mode 0 是一个线程负责的向量，mode 1 是向量的编号；
     num_vec = cute.size(gO, mode=[1])，然后 launch（launch 那行已给出）
  2. kernel swiglu_kernel（@cute.kernel）：
     - 全局线程号 i = block_idx * block_dim + thread_idx（cute.arch.thread_idx() 等返回三元组）
     - 把 i 拆成 mode 1 里的二维坐标 (mi, ni)：gG.shape[1] 就是 (M, N/VEC)
     - gG[(None, (mi, ni))].load() 读出 VEC 个元素（TensorSSA），.to(cutlass.Float32) 转 fp32
     - silu(g) * u，用 cute.math.exp
     - 结果 .to(gO.element_type) 后赋值给 gO[(None, (mi, ni))]

提示：
  - @cute.jit 函数里的 Python print 在编译期执行：print(gG.layout) 可以看到切完的 layout
  - 编译错误信息可能很长，从下往上找第一条 DSL 相关的报错
  - 运行时报 "CUDA error code 9" = cudaErrorInvalidConfiguration（比如 grid 是 0——还没填 num_vec 时就是这样）。
    它后面附带的 "Target SM ARCH unknown is not compatible" 是 DSL 的诊断信息，和真正的错误无关，可以忽略

做完之后想一想：
  - 把 VEC 改成 1、2、4，带宽怎么变？（每个线程一次访存的宽度）
  - 和单元 01 的 Triton 版比：同样的事情，你在 CuTe 里显式决定了什么，而 Triton 替你决定了？

运行：python 10_hopper/exercises/ex4_cute_swiglu.py
"""
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
    # TODO: 见文件头"目标 2"
    pass


@cute.jit
def swiglu_host(mG: cute.Tensor, mU: cute.Tensor, mO: cute.Tensor):
    # TODO: 见文件头"目标 1"，得到 gG、gU、gO 和 num_vec
    gG, gU, gO = mG, mU, mO
    num_vec = 0
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
