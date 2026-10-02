# 07 · CUDA GEMM 一步步优化：从 naive 到 Tensor Core

> 前置：单元 05（CUDA 基础、load_inline）、单元 06（合并访存、shared memory、bank conflict）。
> 单元 04 用 Triton 写过 GEMM —— 这一单元是"拆开 `tl.dot` 看里面"。

本单元的蓝本是 Simon Boehm 的 *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance*，
改成了 H100 + 行主序 + 中文，并且往后多走了一步：Tensor Core（WMMA 和 mma.sync）。

约定：全部行主序，`C[M,N] = A[M,K] @ B[K,N]`。所有 kernel 的完整代码在 `examples/gemm_kernels.py`。

## 7.1 先算账：GEMM 是 compute-bound 的

| 量 | 公式 | 4096³ 的值 |
|---|---|---|
| 计算量 | 2·M·N·K FLOP（一次乘加 = 2 FLOP） | 1.37×10¹¹ FLOP |
| 最少 HBM 流量 | (MK + KN + MN)·bytes | fp32: 201 MB；bf16: 101 MB |
| 算术强度 | 上面两者之比 | fp32: 683 FLOP/B；bf16: 1365 FLOP/B |

H100 SXM 的峰值和 ridge point（峰值算力 / 3.35 TB/s）：

| 计算单元 | 峰值 | ridge point |
|---|---|---|
| FP32 CUDA core（SIMT，普通 FMA 指令） | ~67 TFLOPs | ~20 FLOP/B |
| TF32 Tensor Core | ~495 TFLOPs | ~150 FLOP/B |
| BF16/FP16 Tensor Core | ~989 TFLOPs | ~295 FLOP/B |
| FP8 Tensor Core | ~1979 TFLOPs | ~590 FLOP/B |

大方阵 GEMM 的算术强度远高于 ridge point，所以**理论上是 compute-bound**。
但这是"每个元素只从 HBM 读一次"的理想值；naive kernel 会把同一个元素读成千上万次，
把算术强度降到 ~0.25 FLOP/B，于是变成严重的 memory-bound。
**整个 GEMM 优化史，就是一层层提高数据复用（global → smem → 寄存器），直到计算单元成为瓶颈的历史。**

## 7.2 kernel 1 → 2：线程映射决定合并访存

一个线程算 C 的一个元素：`acc += A[row*K + k] * B[k*N + col]`。

kernel 1 把 `threadIdx.x` 映射到 **row**：同一个 warp 的 32 个线程读 A 的 32 个不同行（地址相距 K×4 字节），
读 B 是同一个地址；写 C 时 32 个地址相距 N×4 字节。一个 warp 的每次访存都要碰 32 条 cache line。

kernel 2 只改一件事：`threadIdx.x` 映射到 **col**。现在读 B 是 32 个连续 float（一次 128 字节事务），
读 A 是同一个地址（广播），写 C 也是连续的。**实测快了 12 倍，一行算法都没改。**

> 规则：让 warp 内相邻线程访问相邻地址。看到 `threadIdx.x` 时，先问"它对应的是内存里最内层（stride=1）的那个维度吗？"

## 7.3 kernel 3：shared memory 分块

kernel 2 里，A 的每个元素被 N 个线程各读一次，B 的每个元素被 M 个线程各读一次 —— 全靠 L1/L2 兜着。
分块的想法：block 负责 C 的一个 T×T tile，沿 K 方向每次把 A 的 `[T, T]` 和 B 的 `[T, T]` 搬进 smem，
块内 T² 个线程共享。

```
for k0 in 0, T, 2T, ...:
    As[ty][tx] = A[row, k0+tx]       # 每个线程搬 1 个 A、1 个 B
    Bs[ty][tx] = B[k0+ty, col]
    __syncthreads()                  # ① 等所有人搬完（否则会读到还没写进来的数）
    for kk in 0..T-1: acc += As[ty][kk] * Bs[kk][tx]
    __syncthreads()                  # ② 等所有人算完（否则快的线程会覆盖别人还在读的 smem）
```

- global 访问次数降为原来的 1/T（T=32 时是 1/32）。
- 越界处理：搬运时越界的位置填 0，内积就不用判断了。这是处理任意尺寸最省事的办法。
- 新的瓶颈：内循环每做 1 次 FMA 要读 2 次 smem。smem 带宽虽高（每 SM 每周期 128 字节），
  但和 FMA 吞吐比还是不够 —— kernel 3 实测只比 kernel 2 快 1.4 倍。

## 7.4 kernel 4、5：寄存器分块（register blocking）

**让每个线程算多个结果**，把从 smem 读出来的值放在寄存器里反复用。

kernel 4（1D）：每个线程算同一列上 TM=8 个结果。内循环读一个 `Bs` 值到寄存器，和 8 个 `As` 值相乘：

```
for dot in 0..BK-1:
    b = Bs[dot][threadCol]                     # 1 次 smem 读
    for r in 0..TM-1:
        res[r] += As[threadRow*TM + r][dot] * b   # TM 次 smem 读，TM 次 FMA
```

kernel 5（2D）：每个线程算 TM×TN = 8×8 的小块，做**外积**：

```
for dot in 0..BK-1:
    regM[0..TM) = As[threadRow*TM + i][dot]   # TM 次 smem 读
    regN[0..TN) = Bs[dot][threadCol*TN + j]   # TN 次 smem 读
    res[i][j] += regM[i] * regN[j]            # TM*TN 次 FMA
```

每个 dot 步：16 次 smem 读、64 次 FMA，"FMA / smem 读" 从 kernel 3 的 0.5 提到 4。
代价是寄存器：`res` 64 个 + `regM/regN` 16 个 + 地址等，每线程 100+ 个寄存器 → 每个 SM 能驻留的线程变少（occupancy 降低）。
这是 GPU 优化里最典型的取舍：**用寄存器换复用，用 occupancy 换 ILP**。

Block tile 128×128、BK=8、256 线程。搬运 tile 时线程映射和计算时不同：搬运要的是"相邻线程搬相邻地址"（合并），
计算要的是"每个线程拿到自己那 8×8 需要的数"，两者都在 kernel 里分别算好下标。

## 7.5 kernel 6：向量化访存 + 转置 As

- global → smem 用 `float4`（128 位）：指令数 ÷4，而且一次 16 字节的事务更高效（需要地址 16 字节对齐，所以要求 K、N 是 4 的倍数）。
- As 在 smem 里**转置**存成 `As[k][m]`：这样内循环读 `regM[0..8)` 是连续地址，编译器可以用 `LDS.128` 一条指令读 4 个。
- 写回 C 也用 `float4`。

实测从 kernel 5 的 22.6 → 37.5 TFLOPs，到了 cuBLAS fp32 的 ~72%。
剩下的差距：smem bank conflict（`regN` 那行，相邻线程地址相距 8 个 float → 4 路冲突）、
没有双缓冲（搬下一块 tile 时计算单元在空等）、warp tiling、autotune 出的 tile 大小……Boehm 的原文继续走了这几步，有兴趣可以接着做。

## 7.6 SIMT 的天花板

即使把 fp32 SIMT GEMM 做到完美，上限也就是 67 TFLOPs。H100 的 Tensor Core 做 bf16 是 989 TFLOPs ——
**15 倍**。原因：一条 FFMA 指令是 1 次乘加；一条 Tensor Core 指令是一整个小矩阵乘（比如 16×8×16 = 2048 次乘加），
取指、译码、读寄存器的开销被摊到 2048 次乘加上，硅片面积几乎全花在乘法器上。

所以现实中 LLM 的 GEMM 全部跑在 Tensor Core 上；前面 6 个 kernel 的价值在于：
**分块、复用、合并、向量化、寄存器压力这些概念，在 Tensor Core kernel 里一个都不少**，只是最内层的乘加换成了矩阵指令。

## 7.7 Tensor Core 之一：WMMA API

CUDA C++ 里最简单的入口是 `<mma.h>` 里的 `nvcuda::wmma`。一个 **warp** 协作完成一个 16×16×16 的矩阵乘加：

```cpp
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;  // A 的 16x16 子块
wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;  // B 的 16x16 子块
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;                         // C 的 16x16，fp32 累加
wmma::fill_fragment(c, 0.f);
for (k0 ...) {
    wmma::load_matrix_sync(a, A + tile_m * K + k0, K);   // ptr = 子块左上角，ldm = 整个矩阵的行距
    wmma::load_matrix_sync(b, B + k0 * N + tile_n, N);
    wmma::mma_sync(c, a, b, c);                          // c = a @ b + c
}
wmma::store_matrix_sync(C + tile_m * N + tile_n, c, N, wmma::mem_row_major);
```

- `fragment` 是**分布在 warp 32 个线程寄存器里**的矩阵，每个线程只持有一部分；哪个线程持有哪个元素是"不透明"的（WMMA 不保证布局）。
- 所有 `*_sync` 函数必须由整个 warp 一起调用（不能有 warp 内分支只让部分线程调用）。
- 限制：指针要 256 位（32 字节）对齐；16 位类型的 ldm 要是 8 的倍数。
- 支持的形状/类型：bf16/fp16 是 m16n16k16、m32n8k16、m8n32k16；tf32 是 m16n16k8 等。

"直接从 global 装 fragment"的版本（练习 3）很慢：同一个 A 子块被同一行的所有 warp 各从 global 读一次，没有任何 smem 复用。
`examples/gemm_kernels.py` 的 kernel 7 先把 128×32 / 32×128 的 tile 用 16 字节向量 load 搬进 smem（行尾 pad 8 个元素防 bank conflict），
每个 warp 再从 smem 装 2×4 个 fragment —— 和 kernel 5 的寄存器分块是同一个思路，只是"寄存器里的小块"变成了 fragment。

## 7.8 Tensor Core 之二：mma.sync PTX 与 ldmatrix

WMMA 隐藏了 fragment 布局，这让你没法在寄存器里直接对结果做事（比如 FlashAttention 要在 S = QKᵀ 的 fragment 上就地做 softmax）。
CUTLASS、FlashAttention-2、Triton 生成的代码用的是更底层的 PTX 指令 **`mma.sync`**，它的布局是**文档化的**。

### m16n8k16（bf16 输入，fp32 累加）的 fragment 布局

一个 warp 计算 `D[16×8] = A[16×16] @ B[16×8] + C[16×8]`。记 `g = lane / 4`（0..7），`t = lane % 4`（0..3）：

| 操作数 | 每个 lane 的寄存器 | 内容 |
|---|---|---|
| A (16×16, row) | 4 个 .b32，每个装 2 个 bf16 | a0 = A[g][2t, 2t+1]，a1 = A[g+8][2t, 2t+1]，a2 = A[g][2t+8, 2t+9]，a3 = A[g+8][2t+8, 2t+9] |
| B (16×8, col) | 2 个 .b32，每个装 2 个 bf16 | b0 = B[2t, 2t+1][g]，b1 = B[2t+8, 2t+9][g]（**k 方向相邻**的两个） |
| C/D (16×8) | 4 个 fp32 | c0, c1 = C[g][2t, 2t+1]，c2, c3 = C[g+8][2t, 2t+1] |

打包规则：下标小的元素在 32 位寄存器的低 16 位。

```cpp
asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
             "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
             : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
             : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
```

`examples/mma_sync_demo.py` 会把这三张布局图画出来（每格写的是持有该元素的 lane id），建议先跑一遍对着看。

### ldmatrix：从 smem 一次装好 fragment

按上表逐个元素从 smem 读太慢（每个 lane 好多次 2 字节的读取）。`ldmatrix` 是专门为此设计的 warp 级指令：

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];
```

- 一次装 1/2/4 个（`.x1/.x2/.x4`）**8×8 的 16 位矩阵**。
- **地址**：lane i 提供第 `i/8` 个矩阵第 `i%8` 行的起始 smem 地址（每行 16 字节，必须 16 字节对齐）。`.x2` 只用 lane 0..15 的地址，`.x1` 只用 0..7。
- **结果**：对第 j 个矩阵，lane 拿到它第 `lane/4` 行、第 `2*(lane%4)` 和 `+1` 列的两个元素，放进 rj。
  —— 这正好就是 mma 的 A fragment 里一个寄存器的布局！
- 于是 A 的 16×16 子块 = 4 个 8×8 矩阵，按 a0..a3 的顺序是（行 0-7, 列 0-7）、（行 8-15, 列 0-7）、（行 0-7, 列 8-15）、（行 8-15, 列 8-15），
  lane i 提供的地址就是 `&As[i % 16][(i / 16) * 8]`。
- **`.trans`**：装载时把每个 8×8 矩阵转置。B 在 smem 里是 `[k][n]` 行主序，而 B fragment 要的是"同一列 n 上 k 相邻的两个" ——
  用 `.trans` 就能直接拿到。怎么给 B 算地址是练习 4 的内容。

地址要先用 `__cvta_generic_to_shared(ptr)` 转成 32 位的 shared 空间地址再传给 asm。

## 7.9 离 cuBLAS 还差什么

实测（下表）kernel 7 只有 cuBLAS bf16 的 ~20%。差距不在"会不会用 Tensor Core"，而在**怎么一直喂饱它**：

1. **异步搬运 + 多级流水**：kernel 7 搬 tile 时 Tensor Core 在等，算的时候内存在闲。Ampere 起有 `cp.async`（global → smem 不经过寄存器），
   可以做 2~4 级 buffer：算第 i 块时在搬第 i+1、i+2 块。
2. **smem swizzle**：用 XOR 打乱 smem 布局，让 ldmatrix 和写入都没有 bank conflict，而且不浪费 padding。
3. **Hopper 的 wgmma**：`mma.sync` 是一个 warp、同步、操作数在寄存器里；H100 上它拿不到全部峰值。
   `wgmma.mma_async` 由 4 个 warp（一个 warpgroup）一起发，形状到 64×256×16，**操作数直接从 smem 读，而且是异步的**。
4. **TMA**：一条指令把一个多维 tile 从 global 搬到 smem（硬件算地址、处理越界、做 swizzle），搬运几乎不占线程。
5. **Warp specialization**：一部分 warp 只负责发 TMA（producer），另一部分只算 wgmma（consumer），用 mbarrier 同步。
6. **Thread block cluster / TMA multicast**、persistent kernel + tile scheduler、epilogue 融合……

3~6 是 Hopper 专属的，放在**单元 10**。到那时你会发现 CUTLASS/CuTe 或 Triton 帮你做掉了大部分，但知道它们在做什么才能在出问题时看懂。

## 实测结果（4096³，这台共享 H100 上跑的，数字有噪声）

`python 07_cuda_gemm/examples/gemm_kernels.py` 的输出：

| kernel | dtype | ms | TFLOPs | 占 cuBLAS 同精度 |
|---|---|---|---|---|
| 1 naive | fp32 | 273.8 | 0.50 | 1% |
| 2 coalesced | fp32 | 21.6 | 6.4 | 12% |
| 3 smem tiling | fp32 | 15.1 | 9.1 | 18% |
| 4 1D blocktile | fp32 | 7.75 | 17.7 | 34% |
| 5 2D blocktile | fp32 | 6.09 | 22.6 | 43% |
| 6 vectorized | fp32 | 3.66 | 37.5 | 72% |
| cuBLAS fp32 | fp32 | 2.65 | 51.9 | 100% |
| cuBLAS tf32 | tf32 | 0.37 | 374 | — |
| 7 wmma + smem | bf16 | 0.92 | 150 | 20% |
| cuBLAS bf16 | bf16 | 0.18 | 753 | 100% |

（练习 3 那种"fragment 直接从 global 装"的 WMMA 版本实测约 28 TFLOPs。）

换成 `python ... 1024` 试试：kernel 5、6 反而比 kernel 4 慢。1024² 的 C 按 128×128 分块只有 64 个 block，
填不满 132 个 SM，一半的 SM 在闲着（**wave quantization**）。tile 大小没有通用最优，要按问题尺寸选 —— 这就是 autotune 存在的原因（单元 04）。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/gemm_kernels.py` | kernel 1~7 的完整实现 + 正确性检查 + 和 cuBLAS 的对比表。`python ... 2048` 换尺寸 |
| `examples/mma_sync_demo.py` | 画出 m16n8k16 的 A/B/C fragment 布局图；手动装 fragment 跑一次 mma.sync 并验证 |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_smem_tiled.py` | shared memory 分块 SGEMM，支持任意 M/N/K | `__syncthreads`、越界填 0 |
| `ex2_register_blocked.py` | 2D 寄存器分块 SGEMM（每线程 8×8） | 外积、搬运 vs 计算的两套线程映射 |
| `ex3_wmma.py` | 用 WMMA 写 bf16 GEMM（每个 warp 一个 16×16 tile） | fragment、load/mma/store_matrix_sync |
| `ex4_mma_sync.py` | 手写 mma.sync.m16n8k16：填 fragment 布局，再用 ldmatrix（含 `.trans`）从 smem 装 | PTX inline asm、fragment 布局 |

练习 1~3 的正确性全部通过后会自动跑一次 4096³ 的 benchmark 并和 cuBLAS 对比。

做完后回答（不检查）：
1. 用 7.1 节的方法估算 kernel 2 的实际 HBM 流量（假设 L2 完全没帮忙），它对应的算术强度是多少？为什么实测比这个估算快得多？
2. kernel 5 每个线程用了多少寄存器？每个 SM 能驻留几个 block？（`extra_cuda_cflags=["-Xptxas=-v"]` 看寄存器；H100 每 SM 65536 个寄存器）
3. 在 kernel 7 里把 block tile 从 128×128 改成 64×64，或者把 smem 的 pad 去掉，TFLOPs 怎么变？
4. FlashAttention 为什么必须用 mma.sync（或 wgmma）而不能用 WMMA？（提示：softmax 要按行做 max 和 sum，你需要知道 S 的哪一行在哪个 lane 上）
