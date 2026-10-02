# 10 · Hopper 专属特性：TMA、wgmma、warp specialization、FP8、CuTe

> 前置：单元 04（Triton GEMM）、单元 06/07（shared memory、Tensor Core 的 CUDA 视角）、单元 08（FlashAttention）。
>
> 学完你能：说清楚 H100 比 A100 多了什么、FA3 为什么快；在 Triton 里用 TMA 写出和 cuBLAS 持平的 GEMM；
> 写 FP8 GEMM；从 PTX 里确认编译器真的用上了这些特性；读懂 CuTe 的 Layout 并写一个 CuTe DSL kernel。

到单元 04 为止，我们的 Triton GEMM 在 4096³ 上大约是 cuBLAS 的 80%。剩下的差距几乎都来自 Hopper 新增的硬件能力。
本单元的一句话总结：**Hopper 把"搬数据"和"算矩阵乘"都变成了异步的、由专门硬件完成的操作，
kernel 的工作从"自己动手干活"变成"给硬件排流水线"。**

---

## 10.1 H100 比 A100 多了什么

| 特性 | A100 (sm_80) | H100 (sm_90a) | 为什么重要 |
|---|---|---|---|
| Tensor Core 指令 | `mma.sync`：一个 warp 同步地算 16×8×16 | **`wgmma.mma_async`**：4 个 warp（一个 *warpgroup*）异步地算 64×N×16（N 最大 256） | 单指令的活多得多，而且发出去之后线程可以干别的；操作数直接从 shared memory 读 |
| global → shared 搬运 | `cp.async`：每个线程自己算地址、搬 4~16 B | **TMA**（Tensor Memory Accelerator）：一个线程发一条指令，硬件按 descriptor 搬整个 tile（最多 5 维） | 不占线程、不占寄存器算地址；边界自动处理；支持 multicast |
| 同步 | `__syncthreads`、`cp.async.wait` | **mbarrier** + "期望字节数"（transaction count）：TMA 搬完自动到达 | 生产者/消费者之间的细粒度异步握手 |
| 协作范围 | 一个 block（CTA） | **thread block cluster**：多个 CTA 一组（可移植上限 8 个，opt-in 可到 16），可互相读写 shared memory（**DSMEM**），TMA 可以把一份数据 **multicast** 给整个 cluster | 相邻 tile 共享 A/B 的读取 |
| 寄存器 | 每线程固定 | **`setmaxnreg`**：运行时在 warpgroup 之间重新分配寄存器 | 搬数据的 warp 只要 ~24 个，算 MMA 的 warp 可以拿到 240 个 |
| 低精度 | bf16/fp16/tf32/int8 | + **FP8**（e4m3、e5m2），Tensor Core 吞吐是 bf16 的 2 倍 | 训练和推理都在往 fp8 走 |
| 规格 | 108 SM，164 KB smem/SM，40 MB L2，2.0 TB/s | 132 SM，**228 KB** smem/SM，**50 MB** L2，**3.35 TB/s** HBM3 | |
| 峰值（dense） | bf16 312 TFLOPS | bf16 **989** / fp8 **1979** TFLOPS | |

注意 `sm_90a` 里的 `a`：wgmma、setmaxnreg 这些是"架构专属"特性，只在 sm_90a 目标下可用，代码不能向前兼容到 Blackwell（sm_100 又换成了 tcgen05 + Tensor Memory）。

## 10.2 "全异步"的编程模型，以及 FA3 为什么需要它

单元 04 的 GEMM 里，每个线程都在做三件事：算地址 + 发 load、等数据、发 MMA。它们挤在同一组寄存器、同一个指令流里。
Hopper 的理想形态是**分工**（warp specialization）：

```
          ┌──────────── 一个 CTA ────────────────────────────────────────────┐
          │ producer warpgroup（setmaxnreg 降到 ~24-40 个寄存器）               │
          │   for k: 等 smem 槽位空 → 发 TMA load A[k], B[k] → (硬件搬，完成时 arrive mbarrier) │
          │                                                                  │
          │ consumer warpgroup 0/1（setmaxnreg 升到 ~232-240 个寄存器）          │
          │   for k: 等 mbarrier（数据到了）→ 发 wgmma（异步）→ wgmma.wait → 释放槽位   │
          │   epilogue：acc → 寄存器里做融合 → TMA store                          │
          └──────────────────────────────────────────────────────────────────┘
          shared memory: [ 槽 0 | 槽 1 | 槽 2 | 槽 3 ]  ← 环形缓冲区，mbarrier 管理"满/空"
```

FlashAttention-3（Shah et al., 2024）在此基础上又做了两件事：

1. **ping-pong 调度**：两个 consumer warpgroup 错开——一个在做 softmax（指数运算，走 SFU，慢）时，另一个在做 GEMM（走 Tensor Core）。
   H100 的 exp 吞吐只有矩阵乘的约 1/256，不重叠的话 softmax 会把 Tensor Core 饿死。
2. **warpgroup 内的流水**：当前块的 softmax 和下一块的 `QK^T` wgmma 重叠（wgmma 是异步的，所以能做到）。

结果：FA2 在 H100 上只有约 35% 的峰值利用率，FA3 的 bf16 前向达到约 75%（论文报告最高约 740 TFLOPS），FP8 接近 1.2 PFLOPS。
这些都依赖 wgmma 的异步性、TMA 释放出来的寄存器、以及 setmaxnreg——全是 Hopper 才有的。

## 10.3 在 Triton 里用 TMA：tensor descriptor

Triton 3.6 提供两种写法（本机已验证）：

**host 端 descriptor**（推荐，练习 2/3 用这个）：

```python
from triton.tools.tensor_descriptor import TensorDescriptor

a_desc = TensorDescriptor.from_tensor(a, block_shape=[BM, BK])     # 描述"按 [BM, BK] 的块读 a"
b_desc = TensorDescriptor.from_tensor(b, block_shape=[BK, BN])
c_desc = TensorDescriptor.from_tensor(c, block_shape=[BM, BN])
kernel[grid](a_desc, b_desc, c_desc, M, N, K, ...)

@triton.jit
def kernel(a_desc, b_desc, c_desc, M, N, K, BM: tl.constexpr, ...):
    ...
    a = a_desc.load([pid_m * BM, k * BK])            # 偏移是元素坐标；越界部分自动补零
    ...
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_desc.dtype))   # 越界部分自动丢弃
```

**device 端 descriptor**（在 kernel 里建，适合 shape 在 kernel 内才知道的情况，比如 MoE 的 grouped GEMM）：

```python
triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))  # descriptor 要一块 global 内存

@triton.jit
def kernel(a_ptr, ..., M, K, BM: tl.constexpr, BK: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BM, BK])
    a = a_desc.load([pid_m * BM, k * BK])
```

限制：最后一维必须连续（stride=1）；其余维的 stride 必须是 16 字节的倍数；block 每一维 ≤ 256。

对比单元 04 的指针版：没有地址计算、没有 mask，每个线程省下一大堆寄存器（4096³、128×128×64 tile：指针版 109 个/线程，TMA 版 90 个），
可以把 tile 做到 128×256 而不 spill。

## 10.4 Persistent + 流水线，以及 warp specialization 在 Triton 里的现状

练习 2 把 TMA 和单元 01 学过的 persistent 写法结合：grid = 132，每个 program 循环处理多个 tile，
`tl.range(start, num_tiles, NUM_SMS, flatten=True)` 让编译器把"tile 循环 × K 循环"当作一条长流水线，
下一个 tile 的 TMA load 能和当前 tile 的 epilogue 重叠。

本机实测（bf16，卡被训练任务共享，±5%）：

| 写法 | 4096³ | 8192³ |
|---|---|---|
| 单元 04 指针版 128×128×64 | ~600 | ~560 |
| TMA，非 persistent，128×128×64 | ~630 | ~660 |
| TMA + persistent + flatten，128×256×64 | **~730~765** | **~700~720** |
| cuBLAS | ~750 | ~680 |

**关于 warp specialization**：`tl.range(..., warp_specialize=True)` 这个参数在 Triton 3.6 里存在，
但在 H100 上实测它**没有生效**：ttgir 里没有生成 `ttg.warp_specialize` 操作，PTX 与不加时完全相同，也没有 `setmaxnreg`。
Triton 的自动 warp specialization 目前主要面向 Blackwell。在 Hopper 上要拿到 FA3 那种显式的 producer/consumer 分工，现实的选择是：

- **CUTLASS / CuTe**（C++ 或 CuTe DSL）——FA3 本身就是用 CUTLASS 写的；
- **Triton Gluon**（`triton.experimental.gluon`，本机可 import，有 `hopper.tma`、`mbarrier`、`warpgroup_mma`）——Triton 的"低一层"方言，布局、同步都要自己管；
- **ThunderKittens**（C++ 模板库，以 tile 为基本单位，比 CUTLASS 好上手）。

这条经验比具体数字更重要：**新特性是否生效，要去 IR / PTX 里确认**（练习 1）。

## 10.5 FP8

两种格式：**e4m3**（4 位指数、3 位尾数，最大 448，精度高，用于前向的权重和激活）和
**e5m2**（5 位指数、2 位尾数，最大 57344，范围大，常用于反向的梯度）。

3 位尾数意味着单个元素的相对误差约 6%（2⁻⁴），而且范围窄，所以**必须配合缩放**：`x ≈ x8 · s`。缩放粒度：

| 粒度 | 做法 | 对离群值 |
|---|---|---|
| per-tensor | 整个 tensor 一个 s | 一个大离群值就把所有其他值压到 fp8 的低端 |
| per-row / per-channel | 激活每行（token）一个、权重每个输出通道一个 | 练习 3 用这个，epilogue 里乘 `sa[:,None] * sw[None,:]` |
| block-wise | 激活 1×128、权重 128×128 一个 s（DeepSeek-V3） | scale 随 K 变化，要在主循环里每个 K 块乘一次，不能只放在 epilogue |

两个 Hopper 特有的坑：

- **操作数布局**：fp8 的 wgmma 要求 A、B 都是 "K 连续"（K-major）。所以权重按 `[N, K]` 存、kernel 里用 `tl.dot(a, w.T)`。
  如果 B 按 `[K, N]` 行主序存，编译器只能在寄存器里转置 B 再喂给 wgmma。本机实测（8192³）：TMA 版从 ~1080 掉到 47~80 TFLOPS
  （寄存器顶到 255），指针版约 180 TFLOPS——差一个数量级。
- **累加精度**：DeepSeek-V3 技术报告指出 H800 上 fp8 wgmma 的累加器实际只有约 14 位有效精度，K 很大时误差会累积；
  他们的做法是每 128 个 K 就把部分和提升到 CUDA core 上的 fp32 寄存器里累加。Triton 的 `tl.dot(..., max_num_imprecise_acc=...)` 就是控制这个的。

本机实测（练习 3，8192³，per-row/per-col 缩放）：

| 实现 | TFLOPS |
|---|---|
| Triton fp8 + TMA，128×256×128 | ~1080~1150 |
| `torch._scaled_mm`（cuBLASLt） | ~1160~1240 |
| cuBLAS bf16 | ~690 |

和 bf16 原始矩阵乘相比的相对误差约 3.7%（随机高斯数据）。

## 10.6 全景：同一个 GEMM 的 7 种写法

`examples/hopper_features.py` 编译同一个 4096³ GEMM 的不同写法，统计 PTX 里的关键指令（本机实测）：

| 写法 | wgmma | cp.async | TMA | mbarrier | 寄存器 | smem | TFLOPS |
|---|---|---|---|---|---|---|---|
| fp32, `input_precision="ieee"` | 0 | 24 | 0 | 0 | 128 | 32 KB | 43 |
| fp32 默认（tf32） | 4 | 12 | 0 | 0 | 128 | 96 KB | 82 |
| bf16 指针, stages=1 | 4 | 0 | 0 | 0 | 124 | 32 KB | 335 |
| bf16 指针, stages=3 | 4 | 24 | 0 | 0 | 109 | 96 KB | 587 |
| bf16 TMA 128×128 | 4 | 0 | 9 | 21 | 90 | 128 KB | 626 |
| bf16 TMA 128×256 | 4 | 0 | 7 | 16 | 154 | 144 KB | 718 |
| fp8 TMA 128×256×128 | 4 | 0 | 7 | 16 | 178 | 144 KB | 1078 |
| cuBLAS bf16 | | | | | | | 751 |

读法：stages=1 没有 cp.async（同步 load，没有预取）；stages=3 用 cp.async 预取；TMA 版完全没有 cp.async，
取而代之的是 `cp.async.bulk.tensor` 和管理它的 mbarrier；wgmma 的形状写在指令名里（`m64n128k16.f32.bf16.bf16`）。

## 10.7 CuTe DSL 入门

CUTLASS 4 带了一个 Python 前端 **CuTe DSL**（`nvidia-cutlass-dsl`，本机 4.5.2，已验证可编译运行）。
写法介于 CUDA 和 Triton 之间：像 CUDA 一样以线程为单位写 kernel，但数据划分全部用 **Layout 代数**描述。

**Layout = (Shape, Stride)**，一个从逻辑坐标到偏移的函数：

```
(4,8):(8,1)          行主序 4x8：(i, j) -> 8i + j
(4,8):(1,4)          列主序 4x8：(i, j) -> i + 4j
(4,(2,4)):(8,(4,1))  嵌套：把第二维拆成 2x4。CuTe 用这种嵌套描述"线程 × 每线程的值"
```

`zipped_divide(layout, tiler)` 把 layout 切成 `(tile 内坐标, 第几个 tile)`。例如 16×64 行主序按 (1, 8) 切：
`((1,8),(16,8)):((0,1),(64,8))`——每个 tile 是 8 个连续元素，一共 16×8 个 tile。
一个线程拿一个 tile，就是一次 16 字节的向量访存。练习 4 用这个写向量化的 SwiGLU。

```python
@cute.kernel
def kernel(gA: cute.Tensor, ...):
    tidx, _, _ = cute.arch.thread_idx()
    ...
    v = gA[(None, (mi, ni))].load()          # 一个线程读 VEC 个元素到寄存器（TensorSSA）
    gC[(None, (mi, ni))] = (v * 2).to(gC.element_type)

@cute.jit
def host(mA: cute.Tensor, ...):
    gA = cute.zipped_divide(mA, (1, 8))
    kernel(gA, ...).launch(grid=(...), block=(256, 1, 1))

compiled = cute.compile(host, from_dlpack(a, assumed_align=16), ...)   # JIT，约 0.5 秒
compiled(...)
```

真正的 Hopper GEMM 在 CuTe 里用 `TiledMMA`（描述 wgmma 怎么铺）、`TiledCopy` / TMA atom（描述搬运）、
`PipelineTmaAsync`（mbarrier 环形缓冲）组合出来。CUTLASS 仓库的 `examples/python/CuTeDSL/hopper/` 下有完整的 Hopper dense GEMM 等示例，
是本单元之后最好的读物——你会发现 10.2 节那张图在代码里一一对应。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/ref_kernels.py` | 指针版 / TMA 版 / FP8 版 GEMM，供示例和练习 1 共用 |
| `examples/hopper_features.py` | 7 种写法的 PTX 特性统计 + TFLOPS（10.6 节的表） |
| `examples/cute_layouts.py` | CuTe Layout、嵌套 shape、zipped_divide；GPU 上 cute.printf |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_ptx_features.py` | 写一个 PTX 分析器：wgmma 形状和 dtype、TMA load/store、cp.async；用它检查 6 个 kernel | 验证编译器真的用了新特性 |
| `ex2_tma_persistent_matmul.py` | TMA descriptor + persistent + flatten 的 GEMM，追平 cuBLAS | TensorDescriptor、tl.range |
| `ex3_fp8_matmul.py` | per-row 量化 + FP8 e4m3 GEMM，epilogue 里反缩放 | fp8、K-major、缩放 |
| `ex4_cute_swiglu.py` | 用 CuTe DSL 写向量化 SwiGLU | Layout、zipped_divide、TensorSSA |

做完后回答（不检查）：
1. 练习 2 的 kernel 在 1024³ 上和 cuBLAS 比怎么样？persistent 在小矩阵上有优势吗？为什么？
2. 把练习 3 的权重改成 `[K, N]` 存（kernel 里不再 `.T`），用练习 1 的分析器看 PTX 有什么变化，速度掉多少？
3. 画出 FA3 的 ping-pong 时间线：两个 consumer warpgroup、每个做 GEMM0 → softmax → GEMM1，怎么错开能让 Tensor Core 一直忙？

## 延伸阅读

- **FlashAttention-3**：Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao. *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, 2024. arXiv:2407.08608
- **CUTLASS / CuTe 文档**：https://github.com/NVIDIA/cutlass （`media/docs` 下的 CuTe 教程；`examples/python/CuTeDSL/`）
- **Colfax Research 教程**：WGMMA、TMA、CUTLASS 流水线、FA3 实现细节的系列文章 https://research.colfax-intl.com/
- **ThunderKittens**：Spector et al., *ThunderKittens: Simple, Fast, and Adorable AI Kernels*, 2024；以及博客 *GPUs Go Brrr*（hazyresearch.stanford.edu）
- **DeepSeek-V3 技术报告**的 FP8 训练一节（block-wise 缩放、累加精度提升）；**DeepGEMM** 仓库是可读性很好的 Hopper FP8 GEMM 实现
- **Triton**：官方教程 `09-persistent-matmul.py`（TMA / persistent / warp specialization 各版本对比）；`python/tutorials/gluon/` 下的 Gluon 教程
- **NVIDIA 文档**：H100 架构白皮书；PTX ISA 中 `wgmma`、`cp.async.bulk.tensor`、`mbarrier` 三节；CUDA C++ Programming Guide 的 Thread Block Clusters、Distributed Shared Memory 两节
