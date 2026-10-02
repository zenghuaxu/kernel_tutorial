"""练习 09-3：把 Triton kernel 注册成 torch.library.custom_op，跑在 torch.compile(fullgraph=True) 里

softcap(x) = cap * tanh(x / cap)。前向、反向 kernel 已给出。你要把它变成一个完整的 PyTorch 算子：
  1. softcap(x, cap)：用 @torch.library.custom_op("kt::softcap", mutates_args=()) 注册，函数体 launch 前向 kernel
  2. 给它注册 fake 实现（@softcap.register_fake）
  3. softcap_backward(y, dy, cap)：同样注册成 custom_op "kt::softcap_backward" + fake 实现
  4. _setup_context / _backward，然后 softcap.register_autograd(_backward, setup_context=_setup_context)

测试会：
  - 比较 eager 前向/反向
  - torch.library.opcheck 两个算子（schema、fake、autograd 注册、AOT 追踪四项都要 SUCCESS）
  - torch._dynamo.explain 统计断图数（必须为 0），并在 torch.compile(fullgraph=True) 下前向+反向

提示：
  - 讲义 9.6 有完整模板
  - custom_op 的函数签名必须带类型注解（x: torch.Tensor, cap: float) -> torch.Tensor
  - 反向只需要 y（dy/dx = 1 - (y/cap)^2），所以 setup_context 里保存 output 即可
  - _backward 返回值个数 = 前向输入个数（x 和 cap），cap 的梯度是 None
  - dy 可能不 contiguous

做完之后想一想：
  - 把 register_fake 删掉，opcheck 和 compile 会报什么错？
  - 直接在 Python 里 launch Triton kernel（不包 custom_op）也不会断图（见 examples/compile_graph_break.py）。
    那 custom_op 还有什么用？（反向、opcheck、CUDA 扩展、torch.export……）

运行：python 09_backward_and_integration/exercises/ex3_custom_op.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, check_equal, finish, report


# ----------------------------------------------------------------------------
# kernel（已给出）：logit soft-capping，y = cap * tanh(x / cap)（Gemma 2 的 attention / 输出 logits 用它）
#   tanh(z) = 2 * sigmoid(2z) - 1
#   dy/dx = 1 - tanh^2 = 1 - (y / cap)^2   —— 反向只需要 y，不需要 x
# ----------------------------------------------------------------------------
@triton.jit
def softcap_fwd_kernel(x_ptr, y_ptr, n, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = cap * (2.0 * tl.sigmoid(2.0 * x / cap) - 1.0)
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def softcap_bwd_kernel(y_ptr, dy_ptr, dx_ptr, n, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
    dy = tl.load(dy_ptr + offs, mask=mask).to(tl.float32)
    t = y / cap
    tl.store(dx_ptr + offs, (dy * (1.0 - t * t)).to(dx_ptr.dtype.element_ty), mask=mask)


BLOCK = 1024


# ----------------------------------------------------------------------------
# 注册成 PyTorch 算子
# ----------------------------------------------------------------------------
# TODO 1 + 2：注册 kt::softcap 和它的 fake 实现
def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    raise NotImplementedError


# TODO 3：注册 kt::softcap_backward 和它的 fake 实现
def softcap_backward(y: torch.Tensor, dy: torch.Tensor, cap: float) -> torch.Tensor:
    raise NotImplementedError


# TODO 4：_setup_context、_backward、softcap.register_autograd(...)


def ref_softcap(x, cap):
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


if __name__ == "__main__":
    torch.manual_seed(0)
    cap = 30.0

    # 1. eager 前向 + 反向
    for dt, tol in [(torch.float32, 1e-5), (torch.bfloat16, 2e-2)]:
        x = (torch.randn(4, 1000, device="cuda") * 50).to(dt).requires_grad_()
        dy = torch.randn(4, 1000, device="cuda", dtype=dt)
        y = softcap(x, cap)
        (dx,) = torch.autograd.grad(y, x, dy)
        xr = x.detach().float().requires_grad_()
        yr = ref_softcap(xr, cap)
        (dxr,) = torch.autograd.grad(yr, xr, dy.float())
        check(f"{dt} y", y, yr.to(dt), atol=tol, rtol=tol)
        check(f"{dt} dx", dx, dxr.to(dt), atol=tol, rtol=tol)

    # 2. opcheck：PyTorch 官方的算子注册自检（schema、fake 实现、autograd 注册、AOT 追踪）
    x = torch.randn(3, 77, device="cuda", requires_grad=True)
    res = torch.library.opcheck(softcap, (x, cap))
    check_equal("opcheck(softcap)", res, {k: "SUCCESS" for k in res})
    res = torch.library.opcheck(softcap_backward, (torch.randn(3, 77, device="cuda"), torch.randn(3, 77, device="cuda"), cap))
    check_equal("opcheck(softcap_backward)", res, {k: "SUCCESS" for k in res})

    # 3. torch.compile(fullgraph=True)：一处断图就会直接报错
    W = torch.randn(512, 1024, device="cuda", requires_grad=True)

    def model(h):
        logits = h @ W                       # 编译器会把 matmul 交给 cuBLAS
        return softcap(logits, cap).float().logsumexp(-1).mean()

    h = torch.randn(64, 512, device="cuda")
    torch._dynamo.reset()
    explain = torch._dynamo.explain(model)(h)
    check_equal("graph_break_count", explain.graph_break_count, 0)

    compiled = torch.compile(model, fullgraph=True)
    loss_c = compiled(h)
    (gW_c,) = torch.autograd.grad(loss_c, W)
    loss_e = model(h)
    (gW_e,) = torch.autograd.grad(loss_e, W)
    check("compiled loss == eager loss", loss_c, loss_e, atol=1e-5, rtol=1e-5)
    check("compiled dW == eager dW", gW_c, gW_e, atol=1e-5, rtol=1e-4)

    # 性能：前向+反向（单独的算子调用）
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn_like(x)
    rows = []
    for name, fn in [("custom_op (triton)", lambda t: softcap(t, cap)),
                     ("torch eager", lambda t: cap * torch.tanh(t / cap))]:
        ms = bench(lambda: torch.autograd.grad(fn(x), x, dy))
        rows.append(dict(impl=name, us_fwd_bwd=ms * 1e3))
    report(rows, "softcap 前向+反向 [8192, 8192] bf16（实测于共享 H100）")
    finish()
