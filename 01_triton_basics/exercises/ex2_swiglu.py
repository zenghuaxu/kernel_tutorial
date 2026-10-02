"""练习 01-2：融合 SwiGLU 激活

LLaMA 类模型的 MLP：down( silu(gate(x)) * up(x) )。中间那个 silu(g) * u 在 PyTorch eager 下是两个 kernel，
中间结果要写回 HBM 再读出来。融合成一个 kernel 能省掉这次往返。

目标：
  - kernel：out = silu(g) * u，其中 silu(g) = g * sigmoid(g)
  - 输入输出可以是 bf16 也可以是 fp32；**load 后转 fp32 计算，store 前转回输出 dtype**
  - wrapper 已经写好了，只需要写 kernel 体

提示：
  - x.to(tl.float32)、tl.sigmoid
  - 输出指针的元素类型：out_ptr.dtype.element_ty

做完之后想一想：
  - 跑完会打印 fused 和 eager 的时间。两者的时间比，和它们搬运字节数的比（3:5）接近吗？
  - 如果在 bf16 里直接算 g * sigmoid(g) * u，误差会大多少？（可以改一下试试）

运行：python 01_triton_basics/exercises/ex2_swiglu.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def swiglu_kernel(g_ptr, u_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # TODO
    pass


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
