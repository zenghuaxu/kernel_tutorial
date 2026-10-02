# 04 · Triton 矩阵乘

> 前置：单元 01（2D block、stride、mask）、单元 02（benchmark、看 PTX）、单元 00（roofline）。
>
> 学完你能：写出达到 cuBLAS 70~90% 的 bf16 GEMM；知道每个旋钮（tile、warps、stages、GROUP_M）在干什么；
> 会把 bias/激活融合进 epilogue；知道"矮胖"矩阵该怎么办（split-K）。

LLM 里 70% 以上的 FLOPs 都在矩阵乘里。你大概不会去手写一个比 cuBLAS 快的通用 GEMM，
但你**一定**会需要写"GEMM + 点别的东西"的 kernel：融合 epilogue、量化 GEMM、grouped GEMM（MoE）、
attention（本质上是两个 GEMM 夹一个 softmax，单元 08）。这些都建立在本单元的骨架上。

---

## 4.1 GEMM 为什么可以是 compute-bound

`C[M,N] = A[M,K] @ B[K,N]`：FLOPs = `2·M·N·K`（每个乘加算 2 次），最少要搬的数据 = `(M·K + K·N + M·N) × 每元素字节`。

以 4096³、bf16 为例：

| | 数值 |
|---|---|
| FLOPs | 2 × 4096³ ≈ 137 GFLOP |
| 最少字节 | 3 × 4096² × 2 B ≈ 100 MB |
| 算术强度 | ≈ 1365 FLOP/B |
| H100 的"拐点" | 989 TFLOPS ÷ 3.35 TB/s ≈ 295 FLOP/B |

1365 ≫ 295，所以**只要数据复用做得好**，GEMM 是 compute-bound 的，上限是 Tensor Core 的峰值
（H100 SXM dense bf16 ≈ 989 TFLOPS；本机实测 cuBLAS 在 4096³ 上约 730 TFLOPS，卡上还跑着训练任务）。

"数据复用"是关键。最朴素的写法——每个输出元素各自读 A 的一行和 B 的一列——每 2 FLOP 要读 2 个元素，
算术强度只有 0.5 FLOP/B（bf16），比拐点低 600 倍。

### 分块（tiling）

让一个 program 负责 C 的一个 `BM × BN` tile。它沿 K 走，每步读 A 的 `BM × BK` 和 B 的 `BK × BN`，
做 `2·BM·BN·BK` FLOP。每读一个元素能做的 FLOP 数：

```
2·BM·BN·BK / ((BM + BN)·BK) = 2·BM·BN / (BM + BN)
BM=BN=128  →  128 FLOP / 元素  =  64 FLOP/B（bf16）
```

tile 越大、越"方"，复用越好。再加上 L2 缓存（50 MB）里相邻 program 共享同一条 A/B（见 4.5），
实际从 HBM 读的量还会再低很多。tile 的上限由**寄存器**（累加器 `BM×BN` 个 fp32 放在寄存器里）
和 **shared memory**（流水线里缓存的 A/B tile）决定。

```
            A [M, K]                    B [K, N]                 C [M, N]
      ┌────────────────────┐      ┌──────────┬──┬──────┐   ┌──────────┬──┬──────┐
      │                    │      │          │░░│      │   │          │  │      │
      ├────────────────────┤      │          │▓▓│ ← 第k步│   │          │  │      │
 BM → │░░░░▓▓░░░░░░░░░░░░░░│      │          │░░│ 向下走 │   ├──────────┼██┼──────┤ ← 本 program
      ├────────────────────┤      │          │░░│      │   │          │  │      │   负责的 tile
      │   ↑ 第k步，向右走    │      └──────────┴──┴──────┘   └──────────┴──┴──────┘   (acc 在寄存器里)
      └────────────────────┘                  BN
   每步：读 A 的 [BM, BK] 和 B 的 [BK, BN]（▓），acc += a @ b，然后各自前进 BK
```

## 4.2 Tensor Core 与 `tl.dot`

`tl.dot(a, b, acc)` 计算 `acc + a @ b`，`a: [BM, BK]`、`b: [BK, BN]`。在 H100 上它被编译成
**wgmma**（warpgroup MMA，4 个 warp 协作的异步 Tensor Core 指令，单元 10 细讲）；在 A100 上是 `mma.sync`。

需要记住的规矩：

- 形状：BM、BN、BK 都 ≥ 16，且是 2 的幂。wgmma 的 M 方向以 64 为单位，所以 H100 上 BM 常取 64/128/256。
- dtype：bf16/fp16 → fp32 累加是主力。fp8（e4m3/e5m2）在 H100 上也走 Tensor Core（单元 10）。
- **fp32 输入的坑**：`tl.dot` 对 fp32 输入默认 `input_precision="tf32"`——只有 10 位尾数，误差约 1e-3。
  需要真 fp32 时写 `tl.dot(a, b, acc, input_precision="ieee")`，但那就不走 Tensor Core 了，慢一个数量级。
- 累加器永远用 fp32（`tl.zeros(..., dtype=tl.float32)`），只在最后写回时转成 bf16。
  在 bf16 里累加 K=4096 项，误差会大到不可用。

## 4.3 Kernel 骨架

完整代码见 `examples/matmul_walkthrough.py`，核心部分：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
offs_k = tl.arange(0, BLOCK_K)
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak    # [BM, BK]
b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn    # [BK, BN]

acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
for k in range(0, tl.cdiv(K, BLOCK_K)):
    k_remaining = K - k * BLOCK_K
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
    acc = tl.dot(a, b, acc)
    a_ptrs += BLOCK_K * stride_ak
    b_ptrs += BLOCK_K * stride_bk

c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

几个细节：

- **K 方向越界用 `other=0.0` 补零**：多乘几个 0 不改变结果，这样 tl.dot 的形状始终是完整的 block。
- **指针前移而不是每次重算**：`a_ptrs += BLOCK_K * stride_ak`，编译器更容易识别出这是一个规整的流水线。
- **stride 让布局自由**：`nn.Linear` 的权重是 `[N, K]`，`y = x @ W^T` 时直接把 `W.stride(1), W.stride(0)` 当作
  `stride_bk, stride_bn` 传进去就行，不用 `.t().contiguous()`。Triton 会在 shared memory 里处理转置。

## 4.4 三个旋钮：tile、num_warps、num_stages

`examples/knobs_sweep.py` 在 8192³ bf16 上扫了一遍（本机实测，卡被训练任务共享，±10% 噪声）：

| tile (BM×BN×BK) | warps | stages | 寄存器/线程 | smem | TFLOPS |
|---|---|---|---|---|---|
| 128×128×64 | 8 | 1 | 124 | 32 KB | 349 |
| 128×128×64 | 8 | 2 | 107 | 64 KB | 500 |
| 128×128×64 | 8 | 3 | 109 | 96 KB | 543 |
| 128×128×64 | 8 | 4 | 110 | 128 KB | 603 |
| 128×256×64 | 8 | 3 | 210 | 144 KB | 623 |
| 64×64×64 | 4 | 3 | 79 | 48 KB | 298 |
| 32×32×32 | 4 | 3 | 42 | 8 KB | 130 |

- **tile 大小**：小 tile 复用差（32×32 只有 130 TFLOPS）。大 tile 复用好，但累加器吃寄存器：
  128×256 的 fp32 累加器 = 32768 个数 / 256 线程 = 每线程 128 个寄存器，加上地址等已经到 210。
  H100 每线程最多 255 个寄存器，超出就 **spill** 到 local memory（实际上是 HBM），性能暴跌。
- **num_warps**：一个 program 有多少 warp 来分担这个 tile。wgmma 以 4 个 warp（一个 warpgroup）为单位，
  所以常见是 4 或 8。tile 大就多给 warp，否则每线程的寄存器压力太大。
- **num_stages = 软件流水线深度**。Tensor Core 很快，从 HBM 读数据很慢（几百个周期）。
  stages=1 时：读 → 等 → 算 → 读 → 等 → 算……Tensor Core 大部分时间在等。
  stages=s 时，Triton 会提前发出后面 s-1 步的异步拷贝（H100 上是 `cp.async` 或 TMA），数据先进 shared memory，
  算第 k 步的同时第 k+1..k+s-1 步的数据在路上：

  ```
  stages=1:  [load0][mma0][load1][mma1][load2][mma2]...
  stages=3:  [load0][load1][load2]
                          [mma0][mma1][mma2]...     ← load 和 mma 重叠
                                [load3][load4]...
  ```
  代价是 shared memory：`stages × (BM·BK + BK·BN) × 2 B`。H100 每个 SM 最多 227 KB；用得越多，
  一个 SM 上能同时驻留的 program 越少。表里 stages 1→4 从 349 涨到 603 TFLOPS。

## 4.5 program 排序：为 L2 分组（GROUP_M）

grid 里的 program 并不是同时跑的——132 个 SM，每个同时跑 1~2 个 program，其余排队。
**同时在跑的那一批 program 读的数据，越能在 L2 里共享越好。**

朴素的"按行"排序：pid 0,1,2,… 依次对应 C 的第 0 行 tile 的第 0,1,2,… 列。
同时运行的 ~200 个 program 覆盖 C 的一两行 tile：它们共享 A 的同一条，但要读 B 的**几乎所有列**。
如果 B 大于 L2，这些列在下一行 tile 再用到时早被挤出去了。

分组排序：把 GROUP_M 行 tile 编成一组，**组内按列优先**：

```
朴素 (GROUP_M=1)              分组 (GROUP_M=2)
pid:  0  1  2  3              pid:  0  2  4  6
      4  5  6  7                    1  3  5  7
      8  9 10 11                    8 10 12 14
     12 13 14 15                    9 11 13 15
```

假设同时只能跑 4 个 program（pid 0~3）：
朴素排序覆盖 C 的第 0 行、第 0~3 列，要读 A 的 1 条 + B 的 4 条 = 5 条；
分组排序覆盖 2×2 的方块，只要 A 的 2 条 + B 的 2 条 = 4 条。
一般地，同时活跃 P 个 program 时，朴素排序要读 ~(1 + P) 条，方块形状只要 ~2√P 条。
H100 上 P ≈ 132~264，差别是几倍的 L2 工作集。
代码见 `examples/matmul_walkthrough.py` 的第 1 段，练习 2 你会自己写一遍。

本机实测（`examples/knobs_sweep.py`，16384×16384×2048 bf16，128×128×64 tile）：

| GROUP_M | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| TFLOPS | 351 | 530 | 569 | 564 |

N 很大（B 每一"行带"有 16384 列 × 2048 × 2 B = 64 MB，大于 50 MB 的 L2）时效果最明显；
8192³ 上只差约 2%。GROUP_M=8 是安全的默认值。

## 4.6 Autotune

最优的 tile/warps/stages 依赖 shape、dtype 和硬件。`@triton.autotune` 在第一次遇到某个 key 时把每个配置都跑一遍，记住最快的：

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],          # 这些参数的值变了就重新调
)
@triton.jit
def matmul_kernel(..., BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, ...): ...

grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
matmul_kernel[grid](a, b, c, M, N, K, ...)   # 不传 BLOCK_*，由 autotune 填
```

注意：

- 配置越多，第一次调用越慢（每个都要编译 + 跑几十次）。生产里常用 `prune_configs_by` 先按 shape 剪枝。
- key 里放什么：会改变最优配置的参数。放太多（比如放了 batch size 而它每步都变）会导致频繁重调。
- kernel 有副作用（原子加、原地修改）时，autotune 反复运行会把结果加好几遍：用 `reset_to_zero=["c_ptr"]` 或 `restore_value`。
- `TRITON_PRINT_AUTOTUNING=1` 打印每个 key 选中的配置；`kernel.best_config` 也能看。

## 4.7 Epilogue 融合

累加器在 K 循环结束时还在寄存器里。在 `tl.store` 前做的任何 elementwise 运算几乎都是免费的：
加 bias、激活函数、残差相加、乘缩放因子（量化 GEMM）、转 dtype、甚至写两个输出。

```python
acc = tl.dot(...)  # 循环结束
if HAS_BIAS:                                  # constexpr 分支：编译期就决定，不生成多余代码
    acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :]
if ACTIVATION == "silu":
    acc = acc * tl.sigmoid(acc)
tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=...)
```

能省多少？不融合时，激活要把 `[M, N]` 再读一遍写一遍。8192×4096 @ 4096×11008 时那是 2×180 MB ≈ 0.1 ms，
而 GEMM 本身约 1.3 ms。练习 3 本机实测：

| 实现 | ms |
|---|---|
| Triton matmul + 融合 bias/GELU | 1.37 |
| 同一个 Triton matmul + 单独的 torch GELU | 1.55 |
| cuBLAS linear + 单独的 torch GELU | 1.19 |

结论很实在：融合省下了预期的 ~0.15 ms；但 cuBLAS 的主循环本身比我们快 ~20%，所以对这种大 GEMM，
"cuBLAS + 单独激活"仍然最快。融合真正决定胜负的场景：GEMM 本身不大（decode）、epilogue 很重（量化/反量化、多个输出）、
或者干脆没有现成库能做（attention 里的 softmax 夹在两个 GEMM 之间——单元 08）。

## 4.8 矮胖矩阵：split-K

M、N 小、K 大时（256×256 输出、K=65536），按输出分块只有 16 个 tile，132 个 SM 大多闲着。
**split-K**：把 K 切成 s 段，grid = (tile 数, s)，每个 program 算一段的部分和，最后加起来。
加法用 fp32 `tl.atomic_add`（简单，但结果不再 bit 级确定）或者写到 workspace 再归约（确定）。

本机实测（练习 4，256×256×65536 bf16）：

| split_k | 1 | 2 | 8 | 32 | 64 | cuBLAS |
|---|---|---|---|---|---|---|
| program 数 | 16 | 32 | 128 | 512 | 1024 | - |
| TFLOPS | 32 | 58 | 139 | 143 | 124 | 208 |

更进一步的做法是 **stream-K**：把"所有 tile 的所有 K 步"拉成一条线均分给 SM 个 persistent program，
同时解决尾部浪费（tile 数不是 SM 数整数倍时，最后一波只有部分 SM 在干活）。

## 4.9 和 cuBLAS 比一比

`examples/matmul_walkthrough.py` 的固定配置（128×128×64，8 warps，3 stages）本机实测：

| MNK | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| Triton TFLOPS | 25 | 137 | 511 | 598 | 556 |
| cuBLAS TFLOPS | 32 | 199 | 595 | 729 | 704 |

小矩阵差距大：1024² 只有 64 个 128×128 tile，填不满 132 个 SM；cuBLAS 对小 shape 有专门的小 tile/split-K 内核。
大矩阵上剩下的 15~25% 主要差在 Hopper 专属特性：TMA 搬运、warp specialization、persistent 调度、
更深的异步流水线——单元 10 会用 Triton 的 TMA descriptor 把这部分差距补回一些。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/matmul_walkthrough.py` | 带注释的完整 GEMM（含 GROUP_M），和 cuBLAS 对比 5 个尺寸 |
| `examples/knobs_sweep.py` | 扫 stages、tile、GROUP_M，打印寄存器/smem 用量和 TFLOPS |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_tiled_matmul.py` | 从零写分块 GEMM：任意 M/N/K、任意 stride（含转置视图），fp16/bf16 | tl.dot、K 循环、三向 mask |
| `ex2_grouped_autotune.py` | 写 GROUP_M 分组映射（探针 kernel 逐个比对），给 kernel 配 autotune | L2 复用、autotune key |
| `ex3_fused_epilogue.py` | `act(x @ W^T + b)`：W 用 stride 当转置读，epilogue 里做 bias + SiLU/GELU | epilogue 融合、constexpr 分支 |
| `ex4_split_k.py` | 256×256×65536 这种矮胖 GEMM 的 split-K，原子加归约 | 并行度、atomic_add |

做完后回答（不检查）：
1. 你的 ex1 kernel 用的是 fp32 输入会怎样？试试把测试里的 dtype 换成 float32，看误差变成多少，为什么？
2. 4.5 节说"同时运行的 program 覆盖 C 的一两行 tile"——132 个 SM、每 SM 同时跑几个 128×128 的 program？用 ex2 里打印的 smem 用量算一算。
3. split-K 用 atomic_add 时，如果把 autotune 加在这个 kernel 上，会出什么错？怎么修？
