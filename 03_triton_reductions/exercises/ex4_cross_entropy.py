"""练习 03-4：大词表 cross-entropy 前向

LLM 的最后一步：logits [T, V] → 每个 token 的 loss = logsumexp(logits[t]) - logits[t, target[t]]。
V 是 12.8 万（LLaMA-3）到 15 万（Qwen），一行 bf16 就有 ~250KB，整行放不进一个 program 的寄存器，
所以要在 V 上**循环**，用 online logsumexp 一边扫一边更新 max 和 sum——只读一遍 logits，
也不需要像 eager 那样先物化一个 [T, V] 的 fp32 log_softmax。

目标：写 kernel 体（wrapper 已给出），返回逐行 loss（fp32，reduction='none'）。
  - target == ignore_index（-100）的行 loss = 0（和 F.cross_entropy 一致）
  - 数值稳定：logit 可能很大（测试里有 ~1000 的）

提示：讲义 3.3 节（online softmax）、3.6 节

做完之后想一想：
  - eager 的 F.cross_entropy(logits.float(), ...) 慢了 10 倍（实测 ~1120us vs ~106us）。它额外分配了多少显存？
    T=8192（一个训练 micro-batch 常见的 token 数）时呢？
  - 如果 T 很小（比如 decode 时 T=8），只有 8 个 program，132 个 SM 大部分闲着。怎么改？
    （提示：把一行拆给多个 program，各自算局部的 (m, s)，再用第二个小 kernel 合并——FlashDecoding 也是这个思路）
  - 反向传播 dlogits = softmax(logits) - onehot(target)。能不能在前向这个 kernel 里顺便算出来，直接覆盖 logits？
    （Liger Kernel 就是这么省显存的；单元 09 会做反向）

运行：python 03_triton_reductions/exercises/ex4_cross_entropy.py
"""
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
    # TODO: 一个 program 处理一行（一个 token），在词表维度上循环，每次处理 BLOCK 个 logit
    #   1. 读 target；如果等于 ignore_index，loss 写 0 然后 return
    #   2. 这一行的起始地址：logits_ptr + row * stride_row（T*V 可能超过 2^31，先把 row 转成 int64）
    #   3. online logsumexp：维护 m（目前为止的最大值）和 s（sum(exp(x - m))）
    #        for start in range(0, V, BLOCK): 读一块 x（越界填 -inf，转 fp32）
    #            m_new = max(m, max(x));  s = s * exp(m - m_new) + sum(exp(x - m_new));  m = m_new
    #   4. lse = m + log(s)；loss = lse - logits[row, target]
    pass

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
