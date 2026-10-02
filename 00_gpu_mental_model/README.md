# 00 · GPU 硬件与性能模型

写 kernel 之前，先要能回答一个问题：**这个算子在这张卡上最快能跑多快？**
回答不了这个问题，就不知道自己的 kernel 是"已经很好了"还是"还差十倍"。
这一单元不写 kernel，只建立心智模型，并学会用纸笔估算。

## 0.1 一个 PyTorch 算子背后发生了什么

```python
y = torch.nn.functional.gelu(x * a + b)      # x: [8192, 4096] bf16
```

eager 模式下这一行会**依次启动 3 个 kernel**：`mul`、`add`、`gelu`。每个 kernel 都：
1. 从 HBM（显存）把输入整个读进来，
2. 算一点点东西，
3. 把结果整个写回 HBM。

GPU 算一次乘加只要零点几纳秒，但从 HBM 搬一个字节的"摊销成本"比这贵得多。
这三个 kernel 一共读写了 6 份 64MB 的数据，真正需要的只有"读 x、写 y"两份——
**写一个融合 kernel 能快 ~3 倍，而一个计算都没省**。这就是大部分自定义 kernel 的价值来源。

## 0.2 H100 SXM 的关键数字

| 资源 | 数值 | 备注 |
|---|---|---|
| SM（流式多处理器） | **132** 个 | 一个 thread block 只能在一个 SM 上跑 |
| 每 SM：线程上限 | 2048 线程 = 64 warps | warp = 32 个线程，GPU 调度的基本单位 |
| 每 SM：寄存器 | 64K 个 32-bit（256 KB） | 每线程最多 255 个；用得越多，同时驻留的线程越少 |
| 每 SM：shared memory | 最多 228 KB（每 block 最多 227 KB） | 和 L1 共用 256 KB 的 SRAM，程序员可控的"手动 cache" |
| L2 cache | 50 MB | 所有 SM 共享 |
| HBM3 显存 | 80 GB，**3.35 TB/s** | |
| BF16/FP16 Tensor Core | **989 TFLOPS**（dense） | 只有矩阵乘能用 |
| FP8 Tensor Core | 1979 TFLOPS | |
| TF32 Tensor Core | 495 TFLOPS | |
| FP32 普通 CUDA core | 67 TFLOPS | elementwise、归约走这里 |
| 一次 kernel launch | ~3–5 μs | 从 CPU 发起到 GPU 开始执行 |

**本机实测**（`examples/device_info.py`，卡上同时有训练任务，偏保守）：

| 测试 | 实测 | 峰值 | 比例 |
|---|---|---|---|
| copy 1 GiB fp32 | 2996 GB/s | 3350 GB/s | 89% |
| bf16 GEMM 8192³ | 760 TFLOPS | 989 TFLOPS | 77% |
| fp32 GEMM 4096³（关 TF32） | 51 TFLOPS | 67 TFLOPS | 76% |
| 空 kernel | 5.1 μs | — | — |

经验：带宽能跑到峰值的 ~90%，Tensor Core 能跑到 ~75–80%，这就是"实际天花板"。

## 0.3 执行模型：grid → block → warp → thread

```
kernel<<<grid, block>>>         Triton: kernel[grid](...)，block 内部由编译器安排
  grid  = 很多个 block           ← 被调度到 132 个 SM 上，一个 SM 可以同时驻留多个 block
  block = 若干 warp（≤1024线程） ← 同一 block 内可以共享 shared memory、可以 __syncthreads()
  warp  = 32 个线程               ← 同一条指令同时作用于 32 个线程（SIMT）
```

为什么一个 SM 要同时驻留那么多线程？**为了藏延迟**。一次 HBM 读取要 ~500+ 个时钟周期，
一个 warp 发出 load 后就在等，调度器立刻切到另一个准备好的 warp 去执行，切换零开销。
所以 GPU 需要大量"在飞"的工作（并行度）来把延迟填满。
驻留 warp 数 / 最大 warp 数（64）叫 **occupancy**，它受寄存器、shared memory 用量限制（单元 06 细讲）。

另一个推论：**grid 太小填不满 132 个 SM**。比如只启动 32 个 block，其余 100 个 SM 就闲着。
这在 decode（batch 小）的 attention 里很常见，单元 08 的 flash-decoding 就是为了解决它。

## 0.4 内存层次

```
               容量           带宽（全卡）       延迟
寄存器         256 KB/SM      —                 ~0 周期
shared/L1      256 KB/SM      ~30+ TB/s         ~30 周期
L2             50 MB          ~10 TB/s          ~200 周期
HBM            80 GB          3.35 TB/s         ~500+ 周期
```

所有高性能 kernel 都是同一个套路：**把数据从 HBM 搬进 SM 内部（shared memory / 寄存器）后尽可能多次复用，
再写回去**。复用次数越多，对 HBM 带宽的需求越低。GEMM 和 FlashAttention 都是这个思路的具体化。

## 0.5 Roofline 模型

对一个 kernel，数两样东西：
- **FLOPs**：总浮点运算数（一次乘加 = 2 FLOPs）
- **bytes**：最少要在 HBM 和芯片之间搬运的字节数（每个输入读一次、每个输出写一次）

**算术强度** `AI = FLOPs / bytes`（单位 FLOP/B）。执行时间下界：

```
t ≥ max( FLOPs / 峰值算力 ,  bytes / 峰值带宽 )
```

两项相等时的 AI 叫 **ridge point**：
- bf16 Tensor Core：989e12 / 3.35e12 ≈ **295 FLOP/B**
- fp32 CUDA core：67e12 / 3.35e12 ≈ **20 FLOP/B**

AI 低于 ridge 的算子是 **memory-bound**（瓶颈是带宽），高于的是 **compute-bound**。

```
 性能
 (FLOP/s)
   ^             ______________________  ← 峰值算力（compute-bound 区）
   |            /
   |           /   ← 斜率 = 带宽（memory-bound 区）
   |          /
   |         /
   +--------+-----------------------------> AI (FLOP/B)
         ridge≈295
```

常见算子的 AI（bf16，元素 2 字节）：

| 算子 | FLOPs | bytes | AI | 类型 |
|---|---|---|---|---|
| `x + y`，n 个元素 | n | 3n·2 | 0.17 | memory |
| softmax / RMSNorm，[T, D] | ~5TD | 2TD·2 | ~1 | memory |
| GEMM M×K @ K×N | 2MNK | (MK+KN+MN)·2 | M=N=K=4096 时 1365 | compute |
| GEMV（decode，M=1） | 2KN | ≈KN·2 | ≈1 | memory |
| attention（flash，序列 S，head dim d） | 4S²d | 4Sd·2 | S/2 | S 大时 compute |

两个重要结论：
1. **LLM 里除了矩阵乘和 attention，几乎所有算子都是 memory-bound**。优化它们 = 少搬字节（融合、低精度存储），不是少算。
2. **矩阵乘的 AI 取决于最小的那个维度**。M（token 数）很小时，GEMM 退化成 memory-bound——decode 阶段就是这样。

本机实测（`examples/roofline_ops.py`，T=8192, D=4096, H=14336，bf16）：

| 算子 | AI | 类型 | roofline 下界 | 实测 | 达到下界的 % |
|---|---|---|---|---|---|
| x + y | 0.17 | memory | 60 μs | 74 μs | 82% |
| softmax(x, -1) | 1.25 | memory | 40 μs | 111 μs | 36% |
| RMSNorm（eager） | 1.0 | memory | 40 μs | 531 μs | **7.5%** |
| GEMV 1×4096 @ 4096×14336 | 1.0 | memory | 35 μs | 56 μs | 62% |
| GEMM 8192×4096 @ 4096×14336 | 2294 | compute | 973 μs | 1209 μs | 80% |

eager RMSNorm 只达到下界的 7.5%——它被拆成 ~6 个 kernel，每个都完整读写一遍 [T, D]，
而且中间还转了 fp32（字节数翻倍）。单元 03 会写一个融合版本。

## 0.6 和你的工作的联系：decode 为什么慢，speculative decoding 为什么快

decode 一步（batch=B，每条序列生成 1 个 token）对每个权重矩阵做的是 [B, K] @ [K, N]：
- FLOPs ≈ 2·B·(参数量)
- bytes ≈ 权重字节数（每步都要把全部权重从 HBM 读一遍）+ KV cache 字节数

B 很小时 AI ≈ B，远低于 295，**时间 ≈ 权重字节数 / 带宽，跟 B 几乎无关**。
所以验证 k+1 个 token（speculative decoding 的 verify 步骤）和生成 1 个 token 几乎一样贵——
draft 模型猜得越准，每次读权重摊到的 token 越多。练习 ex2 会把这件事算清楚。

## 0.7 launch 开销与 CUDA Graph

一次 kernel launch ~5 μs。一个 32 层的模型 decode 一步要启动上千个小 kernel，光 launch 就可能几毫秒，
比实际计算还长。两个对策：**融合**（减少 kernel 个数）和 **CUDA Graph**（把一串 launch 录制下来一次性重放）。
sglang/vLLM 的 decode 默认都开 CUDA Graph。写 kernel 时要注意兼容 CUDA Graph：不能在 kernel 外做 CPU 同步、
不能依赖每次变化的 Python 值决定 grid（单元 09 再讲）。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/device_info.py` | 打印这张卡的参数，实测带宽、Tensor Core 算力、launch 开销 |
| `examples/roofline_ops.py` | 把 5 个常见算子放到 roofline 上，算出离上限多远 |

## 练习

| 文件 | 内容 |
|---|---|
| `ex1_roofline.py` | 实现 FLOPs/bytes 计算器（elementwise、GEMM、attention）和 roofline 时间估计；求 GEMM 变成 compute-bound 的最小 M |
| `ex2_decode_bound.py` | 估算 8B 模型 decode 一步的时间下界、吞吐上限；推导 speculative decoding 的加速比 |
| `ex3_fusion.py` | 数清楚 eager 和融合版本各搬了多少字节，用 torch.compile 验证预测的加速比 |

做完之后想一想：
1. RMSNorm 的 roofline 下界是 40 μs，eager 用了 531 μs。按 0.5 节的方法，你能估算出 eager 实际搬了多少字节吗？大概对应几个 kernel？
2. 一个 [8192, 4096] 的 bf16 tensor 是 64 MB，比 L2（50 MB）大。如果是 [2048, 4096] 呢？连续两个 elementwise kernel 之间，中间结果有可能留在 L2 里吗？这会让 roofline 估算偏保守还是偏乐观？
3. ex2 里，batch 增大到多少时 decode 变成 compute-bound？那时候 speculative decoding 还有用吗？
