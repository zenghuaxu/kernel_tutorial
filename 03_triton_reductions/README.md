# 03 · 归约与归一化：softmax、RMSNorm、cross-entropy

> 前置：单元 01（program / block / mask）、单元 02（会用 `bench`、看 `n_spills`）。
> 本单元的数字都是在**这台被训练任务占用的 H100** 上实测的。

单元 01 的 kernel 里，每个输出元素只依赖一个输入元素。这一单元开始，一个输出依赖**一整行**输入：
求和、求最大值、softmax、归一化。LLM 里除了 GEMM 和 attention，剩下的大部分算子都是这种"按行归约"——
而且它们全是 memory-bound 的，eager PyTorch 往往把它们拆成好几个 kernel，融合起来就是几倍的加速。

## 3.1 Triton 里的归约

```python
x = tl.load(...)                 # [BLOCK]
s = tl.sum(x, axis=0)            # 标量
m = tl.max(x, axis=0)            # 也有 tl.min、tl.argmax、tl.argmin、tl.reduce(自定义组合函数)

t = tl.load(...)                 # [BLOCK_M, BLOCK_N]
row_sums = tl.sum(t, axis=1)     # [BLOCK_M]：每行一个
```

一个 block 内的归约，编译器会替你生成"线程内累加 → warp 内 shuffle → warp 间经 shared memory"的完整树形归约
（单元 06 会用 CUDA 手写一遍，那时你会感激 Triton）。

**mask 掉的位置要填归约的单位元**，否则它们会参与计算：

| 归约 | `other=` | 原因 |
|---|---|---|
| sum | `0.0` | x + 0 = x |
| max | `float("-inf")` | max(x, -inf) = x |
| min | `float("inf")` | |
| softmax 的 exp | 先填 `-inf`，exp(-inf) = 0 | 一次同时满足 max 和 sum |

还有一个容易漏的：**求均值时分母是 N 不是 BLOCK**。`BLOCK = next_power_of_2(N)`，N=5120 时 BLOCK=8192，
用错分母均值直接差 37.5%。

## 3.2 最简单的套路：一行一个 program，整行进寄存器

```python
@triton.jit
def softmax_kernel(x_ptr, out_ptr, N, stride_x, stride_out, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)                     # BLOCK = next_power_of_2(N) >= N
    mask = offs < N
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float("-inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))              # 减 max：数值稳定
    tl.store(out_ptr + row * stride_out + offs, e / tl.sum(e, axis=0), mask=mask)
```

- **数值稳定**：softmax(x) = softmax(x - c) 对任意常数 c 成立。取 c = max(x)，exp 的参数全 ≤ 0，不会溢出。
  fp32 的 exp 在参数 > 88.7 时就溢出成 inf 了，而 logit 上百并不罕见。
- **每行只读一遍、写一遍**：整行 load 进寄存器后，max、exp、sum、除法全在寄存器里完成。
  这正是比 eager 快的根本原因（见下表）。
- **num_warps 随 BLOCK 增大**：BLOCK=4096、8 个 warp → 每线程 16 个元素。经验上每线程 8~32 个元素比较合适。

实测（`exercises/ex1_softmax.py` 的参考答案，bf16，GB/s 按读一遍 + 写一遍算）：

| shape | Triton（本节写法） | torch.softmax | torch.compile |
|---|---|---|---|
| 4096×1024 | 1464 | 1074 | 1421 |
| 4096×4096 | **2292** | 1148 | 1131 |
| 1024×16384 | **2315** | 1558 | 1217 |

## 3.3 行太长怎么办：循环 + online 递推

整行进寄存器的前提是放得下。H100 每个 SM 只有 64K 个 32-bit 寄存器，每个线程最多 255 个。
当 N 太大，编译器只能把寄存器**溢出**（spill）到 local memory（其实就是走 L1/L2 的显存），性能雪崩。
实测 softmax（`examples/online_softmax.py`）：

| shape | 整行一个 block | 每线程寄存器 | spill | GB/s |
|---|---|---|---|---|
| 4096×4096 | BLOCK=4096 | 25 | 0 | 2185 |
| 512×32768 | BLOCK=32768 | 128 | 0 | 1879 |
| 128×131072 | BLOCK=131072 | 32 | **506** | **365** |

（注意：如果 x 只被用一次，比如单纯求和，编译器可以边 load 边累加，即使 BLOCK=524288 也不 spill——
见 `examples/reduce_basics.py`。softmax 要用 x 两次——先求 max、再算 exp——整行必须同时活着。）

解决办法是**在行内循环**，每次只处理 BLOCK 个元素。sum 好办（累加就行），max 也好办，
难的是 softmax 的分母 `sum(exp(x - max))` —— 你得先知道全局 max 才能算 exp。
朴素做法是扫三遍（求 max、求 sum、写结果）。**online softmax** 把前两遍合成一遍：

```
维护两个量：m = 目前见过的最大值，s = sum(exp(x_i - m))（以当前 m 为基准）
读到新的一块 x_blk：
    m_new = max(m, max(x_blk))
    s     = s * exp(m - m_new) + sum(exp(x_blk - m_new))     ← 旧的和"换基准"
    m     = m_new
最后：logsumexp = m + log(s)，softmax_i = exp(x_i - m) / s
```

为什么对：s 原来是 Σ exp(x_i − m)，乘以 exp(m − m_new) 就变成 Σ exp(x_i − m_new)。
`examples/online_softmax.py` 第 1 部分用 PyTorch 验证了：chunk 取 1、7、1000、10000，结果都和 `torch.logsumexp` 一致到 10 位小数。

online 版本在 128×131072 上从 365 GB/s 回到 730 GB/s（只有 128 个 program，并行度仍然不够——见 3.7 节）。
**这个递推是 FlashAttention 的核心**（单元 08）：attention 的 softmax 行长 = 序列长度，动辄几万到几十万。

两个实现细节：
- 循环里可以维护**逐元素**的 `m_i`、`s_i` 向量（形状 [BLOCK]），循环结束后再做一次归约合并，比每轮都做标量归约省事；
  但要小心某个 lane 一直是 -inf 的情况：`exp(-inf - (-inf)) = nan`，要用 `tl.where` 挡掉。
- Triton 的 `for start in range(0, N, BLOCK)` 中 N 是运行时值，循环不会被展开。

## 3.4 归一化：RMSNorm 和 LayerNorm

```
RMSNorm:   y = x * rsqrt(mean(x^2) + eps) * w
LayerNorm: y = (x - mean(x)) * rsqrt(var(x) + eps) * w + b
```

要点：

1. **fp32 累加**。bf16 只有 8 位有效尾数（≈ 2~3 位十进制），4096 个数的平方和在 bf16 里累加，后面加上的小数会被直接舍掉。
   load 进来就 `.to(tl.float32)`，store 时再转回去。
2. **LayerNorm 的 mask 陷阱**：越界位置 x 填了 0，但 `x - mean` 不是 0！求方差前必须
   `xc = tl.where(mask, x - mean, 0.0)`，否则方差会被 (BLOCK−N)·mean² 污染。`examples/layernorm_fwd.py` 的测试特意用了均值为 5 的输入来暴露这个 bug。
3. **方差用两步法**：E[x²] − E[x]² 只需一遍，但均值大、方差小时会发生灾难性抵消。整行已经在寄存器里，第二步不读内存，两步法几乎免费。
4. **为反向存中间量**：LayerNorm 前向顺手把每行的 mean、rstd 存下来（每行 8 字节），反向就不用再算一遍归约（单元 09）。

eager 的 RMSNorm 为什么慢？HuggingFace 的写法 `x.float()` → `pow` → `mean` → `+eps` → `rsqrt` → `mul` → `.to(bf16)` → `mul w`
是 8 个 kernel，其中好几个要读写整个 [T, D] 的 fp32 中间结果。实测 8192×4096 bf16：

| 实现 | 时间 | 有效带宽 |
|---|---|---|
| Triton（练习 2） | 52 us | 2578 GB/s |
| torch.compile | 51 us | 2610 GB/s |
| torch eager（HF 写法） | 531 us | 253 GB/s |

torch.compile 生成的其实就是和你一样的 Triton kernel。但在 D=7168（不是 2 的幂）时实测 compile 版 70us、手写版 46us——
自动生成的 kernel 在边角 shape 上不一定选得好配置，这也是手写 kernel 的价值所在。

## 3.5 融合：残差相加 + RMSNorm

Pre-norm Transformer 每个子层的入口：

```python
residual = x + residual        # x = 上一个子层的输出
h = rmsnorm(residual) * w
```

融合后一个 kernel 读 x、residual、w，写 h 和新 residual——最少字节 = 4 × T × D × 2。
实测 8192×4096 bf16：融合 98us（2744 GB/s，HBM 峰值的 82%），eager 600us，torch.compile 98us。
vLLM / SGLang 里的 `fused_add_rms_norm` 就是这个 kernel（通常是 in-place 版本：直接把结果写回 residual 和 x 的 buffer）。

## 3.6 大词表 cross-entropy

```
loss_t = logsumexp(logits[t, :]) - logits[t, target_t]
```

V = 128256（LLaMA-3）/ 151936（Qwen）。一行 bf16 ~250KB，放不进一个 program 的寄存器 → 用 3.3 节的 online logsumexp 在 V 上循环。
eager 的 `F.cross_entropy(logits.float(), target)` 先物化一个 [T, V] 的 fp32 副本，再做 log_softmax 又是一个 [T, V] fp32——
T=8192、V=128256 时就是两个 4.2GB 的临时 tensor。

实现细节：
- **int64 地址**：T × V 很容易超过 2³¹ ≈ 21.5 亿（8192 × 151936 = 12.4 亿已经过半，T=16384 时就溢出了）。
  Triton 里 `program_id` 是 int32，`row * stride_row` 也是 int32 乘法。写 `row.to(tl.int64) * stride_row`。
- **ignore_index**：padding 位置 target = −100，loss 记 0（和 `F.cross_entropy(reduction="none")` 一致）。
  Triton 里可以在 kernel 开头 `if target == ignore_index: ...; return`。
- **只读 target 处的一个 logit**：`tl.load(base + target)` 是一个标量 load。

实测 T=1024、V=128256 bf16（GB/s 按只读一遍 logits 算）：

| 实现 | 时间 | GB/s |
|---|---|---|
| Triton（练习 4） | 106 us | 2479 |
| torch.compile(`.float()` 版) | 186 us | 1412 |
| eager，bf16 logits 直接传 | 367 us | 716 |
| eager，`.float()` 之后 | 1120 us | 235 |

## 3.7 并行度：行少而长、行多而短

"一行一个 program" 有两种退化情况：

- **行少而长**（decode 时的 cross-entropy：T=8；`reduce_basics.py` 里的 64×524288）：program 数 < SM 数（132），
  大部分 SM 闲着。实测 64×524288 求和，循环版本只有 645 GB/s。解法：一行拆给多个 program，各自算局部结果
  （sum 直接加；softmax/logsumexp 则是局部的 (m, s)，合并公式和 online 递推一样），再用一个小 kernel 或 `tl.atomic_add` 合并。
  FlashDecoding（单元 08）就是这个思路。
- **行多而短**（比如 N=64 的 head_dim 归一化）：一个 program 只处理 64 个元素，128 个线程大部分闲着。
  解法：一个 program 处理多行，用 2D block `[ROWS, BLOCK_N]` 加 `tl.sum(x, axis=1)`。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/reduce_basics.py` | 按行求和：整行一个 block vs 行内循环；寄存器、spill、并行度 |
| `examples/online_softmax.py` | online 递推的 PyTorch 验证；长行 softmax 的 spill 与 online 解法 |
| `examples/layernorm_fwd.py` | 完整的 LayerNorm 前向（两步方差、mask 陷阱、为反向存 mean/rstd） |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_softmax.py` | 按行 softmax，fp32/bf16，数值稳定，支持行 stride | tl.max、tl.sum、-inf 填充 |
| `ex2_rmsnorm.py` | RMSNorm 前向，bf16 进出，hidden 不是 2 的幂 | fp32 累加、分母 N |
| `ex3_add_rmsnorm.py` | 残差相加 + RMSNorm 融合，输出两个 tensor | 融合、多输出 |
| `ex4_cross_entropy.py` | V=128k/152k 的 cross-entropy 前向，online logsumexp，ignore_index | 行内循环、int64 地址 |

做完后回答（不检查）：
1. softmax 练习里，如果 x 一整行都是 −inf（比如 attention 里被完全 mask 掉的一行），你的 kernel 输出什么？torch 呢？
   单元 08 写 attention 时这会成为真问题。
2. RMSNorm 的 BLOCK = next_power_of_2(7168) = 8192，有 1024 个 lane 是被 mask 掉的空转。浪费了多少？
   可以怎么避免？（提示：BLOCK=1024 循环 7 次 / 非 2 的幂的 tile 拆成两段）
3. 交叉熵练习中，如果把 BLOCK 从 4096 改成 1024 或 16384，带宽怎么变？为什么？
4. 3.7 节说"一行拆给多个 program，各自算 (m, s) 再合并"。写出两个局部结果 (m₁, s₁)、(m₂, s₂) 合并成 (m, s) 的公式。
