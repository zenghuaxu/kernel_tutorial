# 08 · FlashAttention：从 online softmax 到 sliding window、GQA 和 decode

> 前置：单元 03（行归约、softmax 的数值稳定写法）、单元 04（`tl.dot`、分块 GEMM）。
> 本单元所有 kernel 都是 Triton。布局统一为 `q: [B, H, N, D]`，`k/v: [B, Hkv, N, D]`，最后一维连续。

这是整个教程的"主菜"。学完你应该能：读懂 FlashAttention-2 前向的每一行；
给它加上 causal、sliding window、GQA；写出解码阶段的 split-KV kernel；
并且知道自己的 kernel 离 SDPA（cuDNN / FA2 后端）还差在哪。

---

## 8.1 朴素 attention 为什么慢

```
S = Q Kᵀ · scale        [N, N]     ← 2·N²·D FLOPs
P = softmax(S, dim=-1)  [N, N]
O = P V                 [N, D]     ← 2·N²·D FLOPs
```

PyTorch 直接写，就是三个（或更多）kernel，每个之间通过 HBM 交换一个 `N×N` 的矩阵：

```
       HBM                          片上（SM）
  Q,K ───────────▶ matmul ──▶ S ───┐
  S   ◀──────────────────────────── ┘   写 N² 个元素
  S   ───────────▶ softmax ─▶ P ───┐
  P   ◀──────────────────────────── ┘   读+写 N²
  P,V ───────────▶ matmul ──▶ O         再读 N²
```

算一笔账（单 head，N=4096，D=128，bf16）：
- 计算：`4·N²·D ≈ 8.6 GFLOP`，H100 bf16 dense ~990 TFLOP/s → **~9 μs**
- 搬 S/P：至少 4 次 `N²×2B = 32 MB` → `128 MB / 3.35 TB/s` → **~40 μs**

计算强度（FLOPs / 字节）只有 `4·N²·D / (8·N²) = D/2 = 64`，远低于 H100 的 ridge point（~300）。
**朴素 attention 是 memory-bound 的，瓶颈是 N² 的中间矩阵。** 而且显存也是 O(N²)。

`examples/naive_attention_cost.py` 实测（共享 H100，B=1 H=8 D=128）：

| N | naive ms | naive TFLOPs | SDPA ms | SDPA TFLOPs | naive 额外显存 |
|---|---|---|---|---|---|
| 1024 | 0.098 | 44 | 0.023 | 189 | 80 MB |
| 4096 | 1.535 | 45 | 0.112 | 614 | 1280 MB |

FlashAttention 的核心思路：**S 和 P 永远不落 HBM**。把 Q 分成行块，每个 program 拿一个 Q 块，
依次把 K/V 块搬进片上，在寄存器里算出这一小块的 S、P，乘上 V 累加进输出。
困难只有一个——softmax 需要整行的 max 和 sum，而我们一次只看到一行的一小段。

## 8.2 online softmax：一遍扫描就能算 softmax

对一行分数 `s_1..s_N`，数值稳定的 softmax：

```
m = max_j s_j
l = Σ_j exp(s_j − m)
softmax_j = exp(s_j − m) / l
```

**在线版**：按块扫描，维护"到目前为止"的 `(m, l)`。处理完前 t 个块后：
`m⁽ᵗ⁾ = 前 t 块的最大值`，`l⁽ᵗ⁾ = Σ_{见过的 j} exp(s_j − m⁽ᵗ⁾)`。新块 `B` 到来：

```
m⁽ᵗ⁺¹⁾ = max(m⁽ᵗ⁾, max_{j∈B} s_j)
l⁽ᵗ⁺¹⁾ = l⁽ᵗ⁾ · exp(m⁽ᵗ⁾ − m⁽ᵗ⁺¹⁾)  +  Σ_{j∈B} exp(s_j − m⁽ᵗ⁺¹⁾)
          └─── 把旧和"换基准" ───┘
```

为什么对：`l⁽ᵗ⁾·exp(m⁽ᵗ⁾ − m') = Σ exp(s_j − m⁽ᵗ⁾)·exp(m⁽ᵗ⁾ − m') = Σ exp(s_j − m')`，
即旧的和被精确地改写成以新最大值为基准。最后 `logsumexp = m + log l`，`softmax_j = exp(s_j − lse)`。

**把输出也一起在线算**。我们要的其实不是 P，而是 `O = Σ_j softmax_j · v_j`。维护未归一化的累加器：

```
acc⁽ᵗ⁾ = Σ_{见过的 j} exp(s_j − m⁽ᵗ⁾) · v_j          （形状 [D]）

新块到来：  α   = exp(m⁽ᵗ⁾ − m⁽ᵗ⁺¹⁾)
            p_j = exp(s_j − m⁽ᵗ⁺¹⁾),  j ∈ B
            acc⁽ᵗ⁺¹⁾ = α · acc⁽ᵗ⁾ + Σ_{j∈B} p_j v_j     ← 这一步就是 tl.dot(p, v)
            l⁽ᵗ⁺¹⁾   = α · l⁽ᵗ⁾   + Σ_{j∈B} p_j

全部扫完：  O = acc / l
```

同样的换基准论证，`acc/l` 恰好等于 `softmax(s)·V`。`examples/online_softmax_math.py` 用 fp64 验证了这些等式（误差 ~1e-16）。

**可合并性**（flash-decoding 的基础）：两段独立算出的 `(m₁,l₁,acc₁)`、`(m₂,l₂,acc₂)` 可以合并：

```
m = max(m₁, m₂)
l = l₁·exp(m₁−m) + l₂·exp(m₂−m)
acc = acc₁·exp(m₁−m) + acc₂·exp(m₂−m)
```

等价地，每段存已归一化的 `o_s = acc_s/l_s` 和 `lse_s = m_s + log l_s`，则
`lse = log Σ_s exp(lse_s)`，`O = Σ_s exp(lse_s − lse)·o_s`。

## 8.3 FlashAttention-2 前向

```
grid = (ceil(N / BLOCK_M), B·H)      每个 program：一个 (batch, head) 的一个 Q 块

            K/V 块 0   K/V 块 1   K/V 块 2  ...
          ┌─────────┬─────────┬─────────┬───
 Q 块 i   │ S_i0    │ S_i1    │ S_i2    │      ← 一个 program 从左到右扫这一行块
 [BM, D]  │ [BM,BN] │         │         │        S、P 只在寄存器里
          └─────────┴─────────┴─────────┴───
```

伪代码（就是 `solutions/ex2_flash_fwd.py` 的结构）：

```python
q = load(Q 块)                         # [BM, D]，常驻寄存器，只读一次
m = -inf; l = 0; acc = 0               # [BM], [BM], [BM, D]，fp32
for start_n in range(0, N, BN):
    k, v = load(K 块), load(V 块)       # [BN, D]
    s = tl.dot(q, tl.trans(k)) * qk_scale         # [BM, BN]，Tensor Core
    s = where(col < N, s, -inf)                   # 尾块的越界 key
    m_new = maximum(m, max(s, 1))
    p = exp2(s - m_new[:, None])
    alpha = exp2(m - m_new)
    l = l * alpha + sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(bf16), v)   # P 转成 bf16 喂 Tensor Core
    m = m_new
O = acc / l[:, None];  LSE = (m + log2(l)) * ln2
```

几个工程细节：

1. **`exp2` 代替 `exp`**。GPU 的特殊函数单元（MUFU）原生只算 `2^x`，`exp(x)` 会被编译成 `exp2(x·log₂e)` 多一次乘法。
   把 `log₂e = 1.4427` 提前折进 `qk_scale = sm_scale · log₂e`，循环里直接 `exp2`，此时 `m`、`l` 都在"log₂ 单位"下，
   最后 `LSE = (m + log₂ l) · ln2` 换回自然对数。
2. **为什么输出 LSE**。反向传播要重新算 `P = exp(S − LSE)`（单元 09 练习 4），存每行一个标量就够，而不用存 N² 的 P。
3. **精度**。`acc`、`m`、`l` 全是 fp32；`p.to(bf16)` 喂给第二个 `tl.dot` 是标准做法（FA2/FA3 都这样），误差约 1e-2 量级，和 SDPA 一致。
4. **块大小**。H100 上 `D=128` 用 `BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3` 是不错的起点：
   Q 块 32 KB 常驻，K/V 每块 16 KB，`num_stages=3` 让 Triton 把下一块的加载和当前块的计算重叠（软件流水）。
5. **FA2 vs FA1**：FA1 外层循环是 K/V、内层是 Q，每次都要读写 O；FA2 把 Q 放外层（每个 program 一个 Q 块），
   O 只写一次，而且不同 Q 块完全独立，天然并行。

实测（共享 H100，B=1 H=16 N=4096 D=128 non-causal）：我们的 Triton 版 ~430 TFLOPs，SDPA ~635 TFLOPs。
差距主要来自 Hopper 专属特性（wgmma 异步、TMA、warp specialization、ping-pong 调度），这是单元 10 和 FA3 的内容。

## 8.4 causal：跳过整块，只在对角线上 mask

causal：query i 只能看 key `j ≤ i`。最省事的写法是每个块都算完再 `where(i >= j, s, -inf)`，
但这样一半的计算是白做的。按块看：

```
              K 块 →
           0    1    2    3    4    5    6    7
 Q 块 0  [ ◢ ][ ✗ ][ ✗ ][ ✗ ] ...
 (BM=128 [ ■ ][ ◢ ][ ✗ ][ ✗ ]                   ◢ = 对角线块：需要逐元素 mask
  BN=64) Q 块 1                                 ■ = 完全在下三角：不需要 mask
         [ ■ ][ ■ ][ ◢ ][ ◢ ][ ✗ ] ...          ✗ = 完全在上三角：根本不用算
         Q 块 2
         [ ■ ][ ■ ][ ■ ][ ■ ][ ◢ ][ ◢ ][ ✗ ]
```

于是把 K 循环拆成两段（`solutions/ex3_causal.py`）：

```python
# 段 1：[0, start_m)          全在对角线左下，不加 mask（省掉比较和 where）
# 段 2：[start_m, start_m+BM)  对角线块，逐元素 mask：offs_m[:, None] >= cols[None, :]
# start_m + BM 之后：跳过
```

要求 `BLOCK_M % BLOCK_N == 0`，这样 `start_m` 是 `BLOCK_N` 的倍数，两段之间没有缝。
把"段 1 / 段 2"写成同一个 `@triton.jit` 辅助函数、用 `constexpr` 开关决定是否 mask，编译器会生成两份特化代码。

Q 块 i 访问的 K 块数 = `ceil(min((i+1)·BM, N) / BN)`。练习 3 会让 kernel 把实际访问的块数写出来，测试精确比对。
实测 causal（N=4096）约 370 TFLOPs（按一半 FLOPs 计），时间约为 non-causal 的 58%。

**一个坑**：对角线块里，若某一行在**它第一次见到的块**里全被 mask 掉，`m_new = -inf`，
于是 `exp2(-inf − (-inf)) = NaN`。causal 两段写法下第一块总包含 `j=0 ≤ i`，所以没事；sliding window 就会踩到（见下）。

## 8.5 sliding window

LLM 里的 sliding window attention（Mistral、Gemma 2/3、以及你的 dspark SWA 层）：
query i 只看最近 W 个 key（含自己）：

```
允许：  i − W < j ≤ i        （与 HF 的 `kv_idx > q_idx − sliding_window` 一致）
```

```
              K 块 →
           0    1    2    3    4    5    6    7
 Q 块 2  [ ✗ ][ ✗ ][ ◣ ][ ■ ][ ◢ ][ ◢ ][ ✗ ][ ✗ ]      W=200，BM=128，BN=64
                     ↑ lo                 ↑ hi
           窗口左边界块（部分 mask）       对角线块
```

Q 块起点 `start_m`：
- 上界 `hi = min(start_m + BM, N)`（和 causal 一样）
- 下界 `lo = floor((start_m − W + 1) / BN) · BN`，再和 0 取 max —— **块里第一行能看到的最左 key**，向下对齐到块边界

于是每个 Q 块只访问约 `(W + BM) / BN` 个 K 块，**计算量从 O(N²) 降到 O(N·W)**。
实测（共享 H100，B=1 H=16 Hkv=4 N=8192 D=128）：

| window | ms | TFLOPs（按有效 FLOPs） |
|---|---|---|
| causal | 0.85 | 324 |
| 4096 | 0.69 | 298 |
| 1024 | 0.26 | 250 |
| 256 | 0.12 | 139 |

窗口越小，每个 Q 块只做几个 K 块，循环的固定开销（加载 Q、写 O、边界块的 mask）占比上升，TFLOPs 下降——但绝对时间仍然大幅下降。

**NaN 坑**：W 小于 BM 时，Q 块里靠后的行在 `lo` 那个块里可能一个能看的 key 都没有（全被 mask）。
解决：`m_i` 初始化成一个很小的**有限值**（如 `-1e30`）而不是 `-inf`：
被 mask 的位置是 `-inf`，`p = exp2(-inf − (-1e30)) = 0`，`alpha = exp2(-1e30 − (-1e30)) = 1`，状态保持不变，没有 NaN。
（FA 的 CUDA 实现是另一种写法：`m == -inf` 时用 0 当基准。）

参考答案为了简单，在 `[lo, hi)` 的每个块上都做三重 mask（causal、window、越界）。进一步的优化：
和 causal 一样把"完全在窗口内"的中间块拆出来不加 mask——留给你做。

## 8.6 GQA / MQA

GQA：`H` 个 q head 共用 `Hkv` 个 kv head，`GROUP = H / Hkv`，q head `h` 用 kv head `h // GROUP`。
前向（prefill）里最简单的支持方式就是在 kernel 里改一行地址：

```python
kvh = h // GROUP
k_base = K + b * stride_kb + kvh * stride_kh
```

不要在 Python 里 `repeat_interleave` 把 K/V 复制 GROUP 份——那会多出 GROUP 倍的显存和 HBM 流量。
prefill 是 compute-bound 的，同一个 kv head 被 GROUP 个 program 各读一遍问题不大（多数命中 L2）；
但 decode 是 memory-bound 的，这里就要把一组 q head 打包到一个 program 里（见 8.7 末尾）。

## 8.7 decode：q 只有一行时怎么办（flash-decoding）

自回归生成时每步只有 1 个新 token：`q: [B, H, D]`，对长度 N 的 KV cache 做注意力。
如果照搬 prefill 的并行方式（每个 (b, h) 一个 program），B=1、H=32 时只有 32 个 program，
而 H100 有 132 个 SM——大部分 SM 闲着，而每个 program 还要顺序扫完整个 N。

**flash-decoding**（split-KV）：把 KV 序列再切成 `S` 段，并行度变成 `B·H·S`：

```
                KV cache 长度 N
  ┌──────────┬──────────┬──────────┬──────────┐
  │ split 0  │ split 1  │ split 2  │ split 3  │   kernel 1：grid=(B·H, S)
  └────┬─────┴────┬─────┴────┬─────┴────┬─────┘   每段输出 (o_s, lse_s)
       ▼          ▼          ▼          ▼
     (o₀,lse₀)  (o₁,lse₁)  (o₂,lse₂)  (o₃,lse₃)    fp32 临时 buffer
       └──────────┴────┬─────┴──────────┘
                       ▼                          kernel 2（combine）：grid=(B·H,)
          lse = log Σ exp(lse_s)                  就是 8.2 的"可合并性"
          o   = Σ exp(lse_s − lse) · o_s
```

细节：
- q 只有一行，`tl.dot` 要求 M ≥ 16，所以 `s = tl.sum(q[None, :] * k, 1)`（逐元素乘 + 归约），`acc += tl.sum(p[:, None] * v, 0)`。
- 序列长度各不相同（`seqlens[b]`），超过的部分 mask 掉；某些 split 可能完全为空（`start ≥ seqlen`），
  这时写 `lse = -inf`、`o = 0`，combine 时权重 `exp(-inf) = 0` 自动忽略。
- `S` 怎么选：让 `B·H·S` 达到 SM 数的几倍即可，太大则 combine 和 partial buffer 的开销上升。

实测（共享 H100，B=1 H=32 Hkv=8 N=32768 D=128，GBps 以 KV cache 字节计）：

| 实现 | μs | GB/s |
|---|---|---|
| 不切分（splits=1） | 1741 | 77 |
| splits=16 | 168 | 800 |
| splits=64 | 162 | 831 |
| **GQA 打包 + splits=16**（`examples/decode_gqa_packed.py`） | **64** | **2093** |
| torch SDPA | 66 | 2047 |

切分带来 10 倍；但每个 q head 一个 program 时，同一个 kv head 被 GROUP=4 个 program 各读一遍，卡在 ~800 GB/s。
**GQA 打包**：一个 program 负责一个 kv head 下的全部 GROUP 个 q head，把它们拼成 `[GROUP→16, D]` 的小矩阵，
K/V 只读一次，`q·Kᵀ` 也能用 `tl.dot` 了 —— 直接追平 SDPA。decode 是纯 memory-bound，"每个字节只读一次"是第一原则。

（MLA 的 decode 更极端：所有 q head 共享同一份 latent KV（`kv_lora_rank + rope_dim`），
打包的收益更大——FlashMLA 就是这么做的，`[H, 576]` 的 q 对 `[N, 576]` 的 latent cache。）

## 8.8 和 SDPA 的差距在哪

`F.scaled_dot_product_attention` 在 H100 上会选 FlashAttention-2 / cuDNN / efficient 后端。我们的 Triton kernel 落后约 30%，原因：
1. **wgmma 异步**：Hopper 的 warpgroup MMA 可以和 softmax 计算重叠；Triton 自动用 wgmma，但重叠程度有限。
2. **TMA + warp specialization**：FA3 用专门的 producer warp 搬数据、consumer warp 计算，两组 consumer 交替（ping-pong）。
3. **exp 的吞吐**：softmax 的 `exp2` 走 MUFU，H100 上 MUFU 吞吐只有 Tensor Core 的零头，D=64/128 时会成为瓶颈；FA3 把 softmax 和 GEMM 交错隐藏它。

这些是单元 10 的内容。对研究代码来说，**能快速写出一个正确、70% SOTA 性能、且支持你的奇怪变体（窗口、mask、GQA、MLA 维度）的 kernel**，就已经很有价值了。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/online_softmax_math.py` | CPU fp64 验证 online softmax 的重缩放、累加器、两段合并 |
| `examples/naive_attention_cost.py` | 朴素 attention vs SDPA：时间、TFLOPs、额外显存随 N 的变化 |
| `examples/decode_gqa_packed.py` | GQA 打包的 flash-decoding，对比练习 5 和 SDPA（做完练习 5 再看） |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_online_softmax.py` | 一个 program 一行，两遍扫描算 softmax 和 logsumexp：第一遍在线维护 (m, l)，第二遍写结果。处理 15 万词表、×300 的大数值 | online (m, l)、数值稳定 |
| `ex2_flash_fwd.py` | FlashAttention 前向（non-causal）：写循环体和收尾，输出 O 和 LSE，对比 SDPA 和朴素实现 | tl.dot、exp2、acc 重缩放 |
| `ex3_causal.py` | causal + 块跳过：写带 constexpr mask 开关的内层函数，主 kernel 分两段调用；测试会核对每个 Q 块访问的 K 块数 | 块级跳过、对角线 mask |
| `ex4_swa_gqa.py` | sliding window + GQA：算出 `[lo, hi)`、kv head 映射、三重 mask，并处理全 mask 行的 NaN | 窗口边界、GQA |
| `ex5_flash_decoding.py` | decode：split-KV kernel（变长 seqlens、空 split）+ combine kernel | flash-decoding、lse 合并 |

做完后回答（不检查）：
1. ex2 里，为什么 `p` 要先 `.to(v.dtype)` 再 `tl.dot`？如果保持 fp32 会怎样（试试看速度）？
2. ex3 里把 `BLOCK_M` 从 128 改成 64，causal 的"浪费"（对角线块里被 mask 掉的那一半）占比怎么变？速度呢？
3. ex4：W=256、N=8192 时 TFLOPs 只有 causal 的一半不到。用 ncu 或者直接数：每个 Q 块访问几个 K 块，其中几个需要 mask？怎么把边界块的开销降下来？
4. ex5：`num_splits` 太大时为什么不再变快？partial buffer `o_part` 有多大，和 KV cache 比呢？
5. 你的 dspark 草稿模型的 MLA 层，decode 时 q head 数、latent 维度是多少？照 `decode_gqa_packed.py` 的思路，一个 program 应该打包几个 head？
