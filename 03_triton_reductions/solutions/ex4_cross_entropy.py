"""练习 03-4：大词表 cross-entropy 前向（参考答案）"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_ptr,
    V, stride_row, ignore_index,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    target = tl.load(target_ptr + row)
    if target == ignore_index:
        tl.store(loss_ptr + row, 0.0)
        return
    base = logits_ptr + row.to(tl.int64) * stride_row     # T*V 可能超过 2^31，地址用 int64 算

    # online logsumexp：一边扫一边维护 当前最大值 m 和 sum(exp(x - m))
    m = float("-inf")
    s = 0.0
    for start in range(0, V, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(base + offs, mask=offs < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # 旧的和是以 m 为基准的，换成以 m_new 为基准要乘 exp(m - m_new)
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(s)
    x_t = tl.load(base + target).to(tl.float32)
    tl.store(loss_ptr + row, lse - x_t)


def cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """逐行的 loss（reduction='none'），fp32。logits: [T, V]，target: [T] int64。"""
    assert logits.dim() == 2 and logits.stride(-1) == 1
    T, V = logits.shape
    loss = torch.empty(T, device=logits.device, dtype=torch.float32)
    cross_entropy_kernel[(T,)](logits, target, loss, V, logits.stride(0), ignore_index, BLOCK=4096, num_warps=8)
    return loss


def ref_cross_entropy(logits, target, ignore_index=-100):
    return F.cross_entropy(logits.float(), target, reduction="none", ignore_index=ignore_index)


if __name__ == "__main__":
    torch.manual_seed(0)
    for T, V in [(1, 10), (5, 1000), (32, 32000), (16, 128256), (8, 151936)]:
        logits = (torch.randn(T, V, device="cuda") * 3).to(torch.bfloat16)
        target = torch.randint(0, V, (T,), device="cuda")
        check(f"bf16 T={T} V={V}", cross_entropy(logits, target), ref_cross_entropy(logits, target),
              atol=1e-4, rtol=1e-4)
    # ignore_index：被忽略的位置 loss 为 0
    logits = torch.randn(64, 50000, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, 50000, (64,), device="cuda")
    target[::3] = -100
    check("ignore_index=-100", cross_entropy(logits, target), ref_cross_entropy(logits, target), atol=1e-4, rtol=1e-4)
    # 数值稳定性：logit 很大
    logits = (torch.randn(4, 128256, device="cuda") * 20 + 1000).to(torch.bfloat16)
    target = torch.randint(0, 128256, (4,), device="cuda")
    check("大 logit", cross_entropy(logits, target), ref_cross_entropy(logits, target), atol=1e-3, rtol=1e-4)
    # 行之间有间隔的视图
    big = torch.randn(16, 40000, device="cuda", dtype=torch.bfloat16)
    logits = big[:, :32000]
    target = torch.randint(0, 32000, (16,), device="cuda")
    check("行 stride != V 的视图", cross_entropy(logits, target), ref_cross_entropy(logits, target),
          atol=1e-4, rtol=1e-4)

    T, V = 1024, 128256
    logits = torch.randn(T, V, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, V, (T,), device="cuda")
    compiled = torch.compile(lambda l, t: F.cross_entropy(l.float(), t, reduction="none"))
    nbytes = logits.numel() * 2
    rows = []
    for name, fn in [("triton", lambda: cross_entropy(logits, target)),
                     ("torch eager (bf16 logits)", lambda: F.cross_entropy(logits, target, reduction="none")),
                     ("torch eager (.float())", lambda: F.cross_entropy(logits.float(), target, reduction="none")),
                     ("torch.compile (.float())", lambda: compiled(logits, target))]:
        ms = bench(fn)
        rows.append(dict(impl=name, us=ms * 1e3, GBps=gbps(nbytes, ms)))
    report(rows, f"cross-entropy T={T} V={V} bf16（GBps 按只读一遍 logits 算）")
    finish()
