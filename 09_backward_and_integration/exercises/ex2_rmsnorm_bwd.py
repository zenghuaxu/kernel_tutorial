"""练习 09-2：RMSNorm 反向（dx 和 dweight）

前向：rstd = (mean(x^2) + eps)^(-1/2)，y = x * rstd * w。前向 kernel、autograd.Function 都已写好，
前向为每行保存了一个 fp32 的 rstd。你要写 rmsnorm_bwd_kernel：

  grid = (P,)，P = min(M, 4 * SM 数)。program pid 处理行 pid, pid+P, pid+2P, ...
  对每一行：
      g  = dy * w
      dx = rstd * g - rstd^3 * x * mean(g * x)          （推导见讲义 9.4）
      写出 dx 这一行
      dw_acc += dy * x * rstd                            （寄存器里的 [BLOCK_N] 部分和，fp32）
  循环结束后把 dw_acc 写到 DW_PART[pid, :N]（DW_PART 是 [P, N] 的 fp32 buffer）
  （第二步 dw_part.sum(0) 已在 backward 里写好）

提示：
  - 和前向一样一个 program 一次处理一整行：cols = tl.arange(0, BLOCK_N)，mask = cols < N
  - w 对所有行相同，循环外 load 一次
  - mean 的分母是 N，不是 BLOCK_N（被 mask 的位置 load 成 0，求和不受影响）
  - 测试会检查：同一输入跑两次，dw 必须 bitwise 相同（不要用 atomic_add）

做完之后想一想：
  - 表格里 triton / eager / torch.compile 各是多少？编译器已经很强了——自己写 kernel 的价值在哪？
  - 如果 N = 16384（BLOCK_N 很大），一行的 x、dy、dx、dw_acc 都放在寄存器里还放得下吗？会发生什么？

运行：python 09_backward_and_integration/exercises/ex2_rmsnorm_bwd.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gpu_name, report

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def rmsnorm_fwd_kernel(X, W, Y, RSTD, stride_x, stride_y, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, 0) / N + eps)
    tl.store(RSTD + row, rstd)                      # 存下来给反向用：每行一个 fp32
    tl.store(Y + row * stride_y + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


@triton.jit
def rmsnorm_bwd_kernel(X, W, DY, RSTD, DX, DW_PART,
                       stride_x, stride_dy, stride_dx, M, N,
                       BLOCK_N: tl.constexpr):
    """grid = (P,)，P 个 program 各自处理行 pid, pid+P, pid+2P, ...

    dx 每行独立；dw 要对所有行求和 —— 每个 program 先在寄存器里累加自己那些行的贡献，
    最后把 [N] 的部分和写到 DW_PART[pid]，再由第二步把 P 份部分和加起来。
    """
    # TODO
    pass


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2.shape
        y = torch.empty_like(x2)
        rstd = torch.empty(M, device=x.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        rmsnorm_fwd_kernel[(M,)](x2, w, y, rstd, x2.stride(0), y.stride(0), N, eps, BLOCK_N=BLOCK_N,
                                 num_warps=8 if BLOCK_N >= 4096 else 4)
        ctx.save_for_backward(x2, w, rstd)
        ctx.shape = shape
        return y.reshape(shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        M, N = x2.shape
        dy2 = dy.reshape(-1, N).contiguous()
        dx = torch.empty_like(x2)
        P = min(M, 4 * NUM_SMS)               # program 数：几倍 SM 即可，太多会让部分和 buffer 变大
        dw_part = torch.empty((P, N), device=x2.device, dtype=torch.float32)
        BLOCK_N = triton.next_power_of_2(N)
        rmsnorm_bwd_kernel[(P,)](x2, w, dy2, rstd, dx, dw_part,
                                 x2.stride(0), dy2.stride(0), dx.stride(0), M, N,
                                 BLOCK_N=BLOCK_N, num_warps=8 if BLOCK_N >= 4096 else 4)
        # 第二步归约：P 份部分和 -> 1 份。顺序固定，所以结果是确定性的（bitwise 可复现）
        dw = dw_part.sum(0).to(w.dtype)
        return dx.reshape(ctx.shape), dw, None


def rmsnorm(x, w, eps=1e-6):
    return RMSNormFunction.apply(x, w, eps)


def ref_rmsnorm(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def grads(fn, x, w, dy):
    x = x.detach().requires_grad_()
    w = w.detach().requires_grad_()
    y = fn(x, w)
    y.backward(dy)
    return y.detach(), x.grad, w.grad


if __name__ == "__main__":
    torch.manual_seed(0)
    for (M, N) in [(1, 64), (3, 1000), (1000, 4096), (4097, 768)]:
        x = torch.randn(M, N, device="cuda")
        w = torch.rand(N, device="cuda") + 0.5
        dy = torch.randn(M, N, device="cuda")
        y, dx, dw = grads(rmsnorm, x, w, dy)
        y_r, dx_r, dw_r = grads(ref_rmsnorm, x, w, dy)
        check(f"fp32 M{M} N{N} y", y, y_r, atol=1e-5, rtol=1e-5)
        check(f"fp32 M{M} N{N} dx", dx, dx_r, atol=1e-4, rtol=1e-4)
        check(f"fp32 M{M} N{N} dw", dw, dw_r, atol=1e-3, rtol=1e-4)

    # bf16，3D 输入（[batch, seq, hidden]）
    x = torch.randn(4, 1024, 2048, device="cuda", dtype=torch.bfloat16)
    w = (torch.rand(2048, device="cuda") + 0.5).bfloat16()
    dy = torch.randn_like(x)
    y, dx, dw = grads(rmsnorm, x, w, dy)
    y_r, dx_r, dw_r = grads(ref_rmsnorm, x.float(), w.float(), dy.float())
    check("bf16 [4,1024,2048] y", y, y_r.bfloat16(), atol=2e-2, rtol=1e-2)
    check("bf16 [4,1024,2048] dx", dx, dx_r.bfloat16(), atol=2e-2, rtol=1e-2)
    check("bf16 [4,1024,2048] dw", dw, dw_r.bfloat16(), atol=0.5, rtol=1e-2)   # 4096 行求和，量级 ~100

    # 确定性：同样的输入跑两次，dw 必须 bitwise 相同
    _, _, dw1 = grads(rmsnorm, x, w, dy)
    _, _, dw2 = grads(rmsnorm, x, w, dy)
    check("dw 确定性（两次 bitwise 相同）", dw1, dw2, atol=0, rtol=0)

    # 性能
    x = torch.randn(16384, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    rows = []
    for name, fn in [("triton", rmsnorm), ("torch eager", ref_rmsnorm),
                     ("torch.compile", torch.compile(ref_rmsnorm))]:
        xr = x.detach().requires_grad_()
        wr = w.detach().requires_grad_()
        ms = bench(lambda: torch.autograd.grad(fn(xr, wr), (xr, wr), dy))
        rows.append(dict(impl=name, us_fwd_bwd=ms * 1e3))
    report(rows, f"RMSNorm 前向+反向 [16384, 4096] bf16（实测于 {gpu_name()}）")
    finish()
