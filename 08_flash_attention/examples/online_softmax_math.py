"""示例：用纯 PyTorch（CPU）验证 online softmax 的三个等式。

运行：python 08_flash_attention/examples/online_softmax_math.py

  1. 分块扫描 + 重缩放 (m, l) 得到的 logsumexp 与一次性计算完全一致
  2. 把 P·V 也一起分块累加（FlashAttention 的 acc），最后除以 l，等于 softmax(s)·V
  3. 两个独立处理的分段 (m1, l1, acc1)、(m2, l2, acc2) 可以合并 —— flash-decoding 的 combine 就靠这个
"""
import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)   # 用 fp64，误差只剩舍入，方便看出"数学上相等"

N, D, BLOCK = 1000, 16, 128
s = torch.randn(N) * 3 + torch.linspace(0, 12, N)   # 一行注意力分数（带上升趋势，让 m 不断被刷新）
v = torch.randn(N, D)              # 对应的 V

# ---- 一次性计算（需要整行都在手里）----
p_full = torch.softmax(s, 0)
o_full = p_full @ v
lse_full = torch.logsumexp(s, 0)

# ---- 1 + 2：分块、只扫一遍 ----
m = torch.tensor(float("-inf"))
l = torch.tensor(0.0)
acc = torch.zeros(D)
for start in range(0, N, BLOCK):
    sb, vb = s[start:start + BLOCK], v[start:start + BLOCK]
    m_new = torch.maximum(m, sb.max())
    alpha = torch.exp(m - m_new)           # 旧状态要乘的因子：把"以 m 为基准"换成"以 m_new 为基准"
    p = torch.exp(sb - m_new)              # 本块未归一化的概率
    l = l * alpha + p.sum()
    acc = acc * alpha + p @ vb
    m = m_new
    print(f"块 {start // BLOCK}: m={m.item():8.4f}  l={l.item():10.4f}")

print(f"\nlogsumexp: 分块={(m + l.log()).item():.12f}  一次性={lse_full.item():.12f}")
print(f"output   : max|分块 - 一次性| = {(acc / l - o_full).abs().max().item():.2e}")


# ---- 3：两段分别处理，再合并 ----
def partial(sb, vb):
    m = sb.max()
    p = torch.exp(sb - m)
    return m, p.sum(), p @ vb


cut = 637
m1, l1, a1 = partial(s[:cut], v[:cut])
m2, l2, a2 = partial(s[cut:], v[cut:])
mm = torch.maximum(m1, m2)
l12 = l1 * torch.exp(m1 - mm) + l2 * torch.exp(m2 - mm)
a12 = a1 * torch.exp(m1 - mm) + a2 * torch.exp(m2 - mm)
print(f"合并两段: max|o - 一次性| = {(a12 / l12 - o_full).abs().max().item():.2e}")

# 等价的"用 lse 合并"写法（flash-decoding 的 combine kernel 用的就是这个）：
#   每段存 o_s = acc_s / l_s（已归一化）和 lse_s = m_s + log l_s
#   lse = logsumexp(lse_1, lse_2)，o = Σ exp(lse_s - lse) * o_s
lse1, lse2 = m1 + l1.log(), m2 + l2.log()
lse = torch.logsumexp(torch.stack([lse1, lse2]), 0)
o = torch.exp(lse1 - lse) * (a1 / l1) + torch.exp(lse2 - lse) * (a2 / l2)
print(f"用 lse 合并: max|o - 一次性| = {(o - o_full).abs().max().item():.2e}   lse 误差 = {(lse - lse_full).abs().item():.2e}")

# ---- 不减 max 会怎样 ----
s_big = s * 100
print(f"\n不减 max 直接 exp：exp(s*100).sum() = {torch.exp(s_big.float()).sum().item()}  (fp32 溢出成 inf)")
print(f"减 max 后：logsumexp = {torch.logsumexp(s_big, 0).item():.4f}  (正常)")
