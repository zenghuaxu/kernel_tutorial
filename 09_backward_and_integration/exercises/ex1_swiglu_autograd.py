"""练习 09-1：SwiGLU 的 autograd.Function（前向 + 反向 kernel）

out = silu(g) * u。前向 kernel 和 autograd.Function.forward 已写好，你要写：
  1. swiglu_bwd_kernel：读 g、u、dout，写 dg、du
       du = dout * silu(g)
       dg = dout * u * silu'(g)，  silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g)))
     计算 dtype 用 ACC（fp64 输入时是 tl.float64，否则 tl.float32），store 时转回输出 dtype
  2. SwiGLUFunction.backward：取出保存的 tensor，launch 反向 kernel，返回 (dg, du)

测试包括：
  - fp32 / bf16 对 PyTorch autograd 的结果
  - y.sum().backward()：这时 dout 是 stride 全 0 的 expand 视图（讲义 9.1）
  - torch.autograd.gradcheck（fp64 有限差分）

提示：
  - 先看 examples/autograd_basics.py
  - 前向用的 launch 写法可以照抄：swiglu_bwd_kernel[(triton.cdiv(n, BLOCK),)](..., ACC=_acc_dtype(g), BLOCK=BLOCK, num_warps=8)

做完之后想一想：前向存的是 (g, u) 而不是 silu(g)。反向多算一次 sigmoid 的代价是什么？省下了什么？

运行：python 09_backward_and_integration/exercises/ex1_swiglu_autograd.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, check_equal, finish, gpu_name, report


@triton.jit
def swiglu_fwd_kernel(g_ptr, u_ptr, out_ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask).to(ACC)
    u = tl.load(u_ptr + offs, mask=mask).to(ACC)
    out = g * tl.sigmoid(g) * u
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def swiglu_bwd_kernel(g_ptr, u_ptr, dout_ptr, dg_ptr, du_ptr, n, ACC: tl.constexpr, BLOCK: tl.constexpr):
    # TODO 1
    pass


def _acc_dtype(t):
    # fp64 输入（gradcheck 用）就在 fp64 里算，其余一律 fp32
    return tl.float64 if t.dtype == torch.float64 else tl.float32


BLOCK = 2048


class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, u):
        assert g.shape == u.shape
        g, u = g.contiguous(), u.contiguous()
        out = torch.empty_like(g)
        n = g.numel()
        swiglu_fwd_kernel[(triton.cdiv(n, BLOCK),)](g, u, out, n, ACC=_acc_dtype(g), BLOCK=BLOCK, num_warps=8)
        # 存输入而不是存 silu(g)：反向时重算 sigmoid 很便宜（memory-bound kernel 里算力是免费的），省一份激活显存
        ctx.save_for_backward(g, u)
        return out

    @staticmethod
    def backward(ctx, dout):
        # TODO 2：取出 saved tensors；处理 dout 的 contiguity；分配 dg、du；launch 反向 kernel
        raise NotImplementedError


def swiglu(g, u):
    return SwiGLUFunction.apply(g, u)


def ref_swiglu(g, u):
    return F.silu(g) * u


def grads(fn, g, u, dout):
    g = g.detach().requires_grad_()
    u = u.detach().requires_grad_()
    out = fn(g, u)
    out.backward(dout)
    return out.detach(), g.grad, u.grad


if __name__ == "__main__":
    torch.manual_seed(0)
    # 1. fp32：和 PyTorch autograd 的结果比
    for shape in [(7,), (3, 1000), (4, 256, 1408)]:
        g = torch.randn(shape, device="cuda") * 3
        u = torch.randn(shape, device="cuda")
        dout = torch.randn(shape, device="cuda")
        o, dg, du = grads(swiglu, g, u, dout)
        o_r, dg_r, du_r = grads(ref_swiglu, g, u, dout)
        check(f"fp32 {shape} out", o, o_r, atol=1e-5, rtol=1e-5)
        check(f"fp32 {shape} dg", dg, dg_r, atol=1e-5, rtol=1e-5)
        check(f"fp32 {shape} du", du, du_r, atol=1e-5, rtol=1e-5)

    # 2. bf16：和 fp32 参考比（参考在 fp32 里算，再转 bf16）
    shape = (8, 512, 1408)
    g = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 3
    u = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    o, dg, du = grads(swiglu, g, u, dout)
    o_r, dg_r, du_r = grads(ref_swiglu, g.float(), u.float(), dout.float())
    check("bf16 out", o, o_r.bfloat16(), atol=2e-2, rtol=1e-2)
    check("bf16 dg", dg, dg_r.bfloat16(), atol=2e-2, rtol=1e-2)
    check("bf16 du", du, du_r.bfloat16(), atol=2e-2, rtol=1e-2)

    # 3. 非 contiguous 的 grad_output：y.sum() 的梯度是 expand 出来的 stride-0 张量
    g = torch.randn(64, 100, device="cuda", requires_grad=True)
    u = torch.randn(64, 100, device="cuda", requires_grad=True)
    swiglu(g, u).sum().backward()
    g2 = g.detach().clone().requires_grad_()
    u2 = u.detach().clone().requires_grad_()
    ref_swiglu(g2, u2).sum().backward()
    check("expand 的 grad_output: dg", g.grad, g2.grad, atol=1e-5, rtol=1e-5)
    check("expand 的 grad_output: du", u.grad, u2.grad, atol=1e-5, rtol=1e-5)

    # 4. gradcheck：fp64 下用有限差分验证解析梯度
    g = torch.randn(37, device="cuda", dtype=torch.float64, requires_grad=True)
    u = torch.randn(37, device="cuda", dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(swiglu, (g, u), eps=1e-6, atol=1e-5)
    check_equal("gradcheck fp64", ok, True)

    # 性能：前向 + 反向
    shape = (8, 2048, 1408)
    g = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    u = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dout = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    rows = []
    for name, fn in [("triton autograd.Function", swiglu), ("torch eager", ref_swiglu)]:
        ms = bench(lambda: torch.autograd.grad(fn(g, u), (g, u), dout))
        rows.append(dict(impl=name, us_fwd_bwd=ms * 1e3))
    report(rows, f"SwiGLU 前向+反向 {shape} bf16（实测于 {gpu_name()}）")
    finish()
