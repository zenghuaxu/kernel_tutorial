"""练习 03-3：融合 残差相加 + RMSNorm

Pre-norm Transformer 的每一层开头都是：
    residual = x + residual          # x 是上一个子层（attention / MLP）的输出
    h = rmsnorm(residual) * w        # 送进下一个子层
eager 下是"一个 add kernel + 一串 RMSNorm kernel"，residual 要写回 HBM 再读出来。
vLLM / SGLang 里都有一个 fused_add_rms_norm 把它们合成一个 kernel。

目标：写 kernel 体，返回 (out, res_out)：
  - res_out = x + residual（输出 dtype，和 torch 的 bf16 加法逐位一致：先在 fp32 里加，再舍入）
  - out = rmsnorm(x + residual) * w（用 fp32 的 h 归一化）

做完之后想一想：
  - 最少要搬多少字节？（读 x、residual、w，写 out、res_out）你的 kernel 达到了 HBM 峰值的百分之几？
  - 推理框架里通常是 in-place 版本：直接把 x + residual 写回 residual 的 buffer。改成 in-place 有什么好处？
    在这个 kernel 里，in-place 会不会有读写冲突？（一个 program 只读写自己那一行）

运行：python 03_triton_reductions/exercises/ex3_add_rmsnorm.py
"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def add_rmsnorm_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, res_out_ptr,
    N, stride_x, stride_res, stride_out, stride_res_out, eps,
    BLOCK: tl.constexpr,
):
    # TODO: 一个 program 处理一行
    #   1. load x 和 residual 的这一行（fp32），h = x + residual
    #   2. 把 h 存到 res_out（新的残差流，下一层还要用）
    #   3. 对 h 做 RMSNorm（同练习 2），结果存到 out
    pass

def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = 1e-6):
    """返回 (rmsnorm(x + residual) * w, x + residual)。"""
    assert x.shape == residual.shape and x.shape[-1] == w.shape[0]
    x2 = x.reshape(-1, x.shape[-1])
    r2 = residual.reshape(-1, x.shape[-1])
    M, N = x2.shape
    out = torch.empty_like(x2)
    res_out = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)
    add_rmsnorm_kernel[(M,)](
        x2, r2, w, out, res_out, N,
        x2.stride(0), r2.stride(0), out.stride(0), res_out.stride(0), eps,
        BLOCK=BLOCK, num_warps=num_warps,
    )
    return out.view_as(x), res_out.view_as(x)


def ref_add_rmsnorm(x, residual, w, eps=1e-6):
    h = x.float() + residual.float()
    out = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return out.to(x.dtype), h.to(x.dtype)


def eager_add_rmsnorm(x, residual, w, eps=1e-6):
    h = x + residual
    hf = h.float()
    hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    return w * hf.to(x.dtype), h


if __name__ == "__main__":
    torch.manual_seed(0)
    for shape in [(1, 16), (7, 300), (64, 4096), (16, 5120), (2, 8, 2048)]:
        x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        res = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 4
        w = torch.randn(shape[-1], device="cuda", dtype=torch.bfloat16)
        out, res_out = add_rmsnorm(x, res, w)
        ref_out, ref_res = ref_add_rmsnorm(x, res, w)
        check(f"out {shape}", out, ref_out, atol=2e-2, rtol=2e-2)
        check(f"residual_out {shape}", res_out, ref_res, atol=0, rtol=0)
    x = torch.randn(33, 1000, device="cuda")
    res = torch.randn(33, 1000, device="cuda")
    w = torch.rand(1000, device="cuda")
    out, res_out = add_rmsnorm(x, res, w)
    ref_out, ref_res = ref_add_rmsnorm(x, res, w)
    check("fp32 out (33, 1000)", out, ref_out, atol=1e-5, rtol=1e-5)
    check("fp32 residual_out (33, 1000)", res_out, ref_res, atol=1e-6, rtol=1e-6)

    rows = []
    compiled = torch.compile(eager_add_rmsnorm)
    M, N = 8192, 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    nbytes = 4 * x.numel() * 2          # 读 x、res，写 out、res_out
    for name, fn in [("triton fused", lambda: add_rmsnorm(x, res, w)),
                     ("torch eager", lambda: eager_add_rmsnorm(x, res, w)),
                     ("torch.compile", lambda: compiled(x, res, w))]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"add + RMSNorm bf16 {M}x{N}（GBps 按最少字节）")
    finish()
