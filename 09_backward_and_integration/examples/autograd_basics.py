"""示例：torch.autograd.Function 的解剖。

运行：python 09_backward_and_integration/examples/autograd_basics.py

演示：
  1. forward / backward 的签名约定：backward 返回值个数 = forward 输入个数，非 tensor 参数返回 None
  2. ctx.save_for_backward（只用于 tensor）vs ctx.xxx = ...（非 tensor）
  3. ctx.needs_input_grad：不需要的梯度可以不算
  4. grad_output 不一定 contiguous（最常见的 kernel 反向 bug）
  5. gradcheck：fp64 有限差分验证
"""
import torch

VERBOSE = True


class ScaledTanh(torch.autograd.Function):
    """y = tanh(a * x) * b，a 是 python float，b 是 tensor（可学习）。用纯 torch 写，重点看结构。"""

    @staticmethod
    def forward(ctx, x, a: float, b):
        t = torch.tanh(a * x)
        ctx.save_for_backward(t, b)     # 存 tensor：autograd 会检查它们在反向前没被 in-place 改过
        ctx.a = a                       # 存非 tensor：直接挂在 ctx 上
        return t * b

    @staticmethod
    def backward(ctx, dy):
        t, b = ctx.saved_tensors
        if VERBOSE:
            print(f"    backward 收到 dy: shape={tuple(dy.shape)} stride={dy.stride()} contiguous={dy.is_contiguous()}")
        dx = db = None
        if ctx.needs_input_grad[0]:                     # 对应 forward 的第 0 个输入 x
            dx = dy * b * ctx.a * (1 - t * t)
        if ctx.needs_input_grad[2]:                     # 第 2 个输入 b
            db = (dy * t).sum_to_size(b.shape)          # b 被广播过，梯度要按广播规则求和回去
        return dx, None, db                             # 3 个输入 -> 3 个返回值，a 是 float 返回 None


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 8, device="cuda", requires_grad=True)
    b = torch.randn(8, device="cuda", requires_grad=True)

    print("1) y.sum().backward()：")
    ScaledTanh.apply(x, 0.7, b).sum().backward()
    print("   ↑ stride 全是 0！sum 的反向把一个标量 expand 成 [4, 8]。")
    print("     如果你的 Triton kernel 假设 dy 是 contiguous、直接按 row*N+col 取地址，就会读错。")
    print("     对策：backward 开头 dy = dy.contiguous()（对 stride-0 张量会真的拷贝一份，代价很小）\n")

    print("2) (y * w).sum().backward()：")
    x.grad = b.grad = None
    w = torch.randn(4, 8, device="cuda")
    (ScaledTanh.apply(x, 0.7, b) * w).sum().backward()

    print("\n3) y.t() 之后再用：")
    x.grad = b.grad = None
    (ScaledTanh.apply(x, 0.7, b).t() * torch.randn(8, 4, device="cuda")).sum().backward()
    print("   ↑ 输出被转置后再参与逐元素运算，梯度回传时也是转置视图（stride=(1, 4)），非 contiguous。")

    print("\n4) 只对 x 求导时 needs_input_grad =", end=" ")
    x2 = torch.randn(3, device="cuda", requires_grad=True)
    b2 = torch.randn(3, device="cuda")              # 不需要梯度

    class Peek(ScaledTanh):
        @staticmethod
        def backward(ctx, dy):
            print(ctx.needs_input_grad)
            return ScaledTanh.backward(ctx, dy)

    Peek.apply(x2, 1.0, b2).sum().backward()

    print("\n5) gradcheck（fp64，有限差分 vs 解析梯度）：")
    VERBOSE = False
    xd = torch.randn(5, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    bd = torch.randn(3, device="cuda", dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(lambda x_, b_: ScaledTanh.apply(x_, 0.7, b_), (xd, bd))
    print("   gradcheck:", ok)
    print("   注意：kernel 内部若把 fp64 降成 fp32 算，gradcheck 会因为精度不够而失败。")
    print("   练习 1 用一个 constexpr 的 ACC dtype 解决：fp64 输入就用 fp64 算。")
