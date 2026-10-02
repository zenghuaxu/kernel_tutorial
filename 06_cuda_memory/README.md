# 06 · CUDA 内存层次：合并访存、shared memory、bank conflict、occupancy、归约

> 前置：单元 05（会写、会编译、会调试一个 CUDA kernel）。
> 绝大多数 kernel 的瓶颈都是"数据搬运"，这一单元讲 GPU 上数据在哪、怎么搬才快。
> 本单元所有数字都是在**这台共享的 H100 上实测**的（同卡上有训练任务在跑，有噪声，看量级和趋势）。

## 6.1 内存层次一览（H100 SXM / sm_90）

| 层级 | 容量 | 谁能看见 | 访问方式 | 量级 |
|---|---|---|---|---|
| 寄存器 | 每 SM 64K 个 32 位（256 KB）；每线程最多 255 个 | 单个线程 | 局部变量 | ~1 周期 |
| shared memory | 和 L1 共用每 SM 256 KB，shared 最多划 228 KB；每 block 最多 227 KB | 同一 block 的线程 | `__shared__` | 实测 ~29 周期（单 warp 指针追逐，见 6.4） |
| L1 cache | 同上（剩下的部分） | 单个 SM | 自动 | |
| L2 cache | 50 MB（两个分区） | 所有 SM | 自动 | 几百周期 |
| HBM3（global memory） | 80 GB，峰值约 3.35 TB/s | 所有 SM + host | 指针 | 五百周期以上 |
| local memory | 物理上在 HBM（经过 L1/L2） | 单个线程 | 寄存器溢出、动态下标的局部数组 | 和 global 一样慢 |

两个要记住的结论：
1. **离 SM 越近越快越小**。优化的本质就是：数据从 HBM 只搬一次，然后在寄存器 / shared memory 里尽量多用几次。
2. **local memory 是个陷阱**。`float v[COLS]` 这种局部数组，只有下标在编译期能确定（循环被完全展开）时才放寄存器；
   否则放 local memory，也就是显存。练习 3 的 warp softmax 里那个 `#pragma unroll` 不是可有可无的。

## 6.2 合并访存（coalescing）

一个 warp 执行一条 load 指令时，32 个线程的地址被合起来，按 **32 字节的 sector** 向内存系统请求（一条 cache line 是 128 字节 = 4 个 sector）。
- 32 个线程读连续的 32 个 float（128 字节）→ 4 个 sector，每个字节都有用。**这就是合并访存**。
- 相邻线程的地址相距 s 个 float → 要碰 min(32, 4s) 个 sector，大部分搬进来的字节被浪费。

`examples/coalescing.py`：`out[i] = x[i * stride]`，每次都只用到 32 MB 有用数据：

| stride | 每个 warp 碰的 sector 数 | 有效带宽（实测） |
|---|---|---|
| 1 | 4 | 1970 GB/s |
| 2 | 8 | 1494 GB/s |
| 4 | 16 | 1047 GB/s |
| 8 | 32 | 616 GB/s |
| 16 | 32 | 339 GB/s |
| 32 | 32 | 225 GB/s |

stride ≥ 8 之后 sector 数不再增加，带宽却还在掉——访问跨度越大，DRAM 的行缓冲、TLB、L2 的局部性越差。
同一个例子里，**首地址偏移**（`x[i + offset]`，仍然连续）几乎没有影响（1925~1980 GB/s）。

**规则**：让 `threadIdx.x` 沿内存连续的维度走。单元 05 练习 2 已经见过：反过来映射，带宽从 2210 掉到 465 GB/s。
如果算法天然要"按列读"（转置、某些归约），就先合并地读进 shared memory，再在 shared memory 里按任意顺序访问——6.4。

## 6.3 Shared memory 与 `__syncthreads`

shared memory 是程序员手动管理的片上缓存（"scratchpad"），一个 block 的所有线程共享，生命周期等于 block。

```cpp
__shared__ float tile[32][33];               // 静态：大小编译期确定，总共 ≤ 48 KB
extern __shared__ float buf[];               // 动态：大小由 <<<grid, block, smem_bytes, stream>>> 的第三个参数决定
// 动态 shared memory 超过 48 KB 要先 opt-in：
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
```

典型用法：**协作加载 → 同步 → 复用**。

```cpp
buf[t] = x[base + t];          // 1. 每个线程搬一部分（合并访存）
__syncthreads();               // 2. 屏障：block 里所有线程都到这里才继续
y = buf[BLOCK - 1 - t];        // 3. 读别的线程搬进来的数据
```

`__syncthreads()` 的规则：
- block 里**所有线程都必须执行到同一个** `__syncthreads()`。把它放在只有部分线程进入的 `if` 里 → 死锁或未定义行为。
- 它既是执行屏障，也是内存屏障（之前对 shared/global 的写对 block 内其他线程可见）。
- 只同步一个 block。不同 block 之间没有（便宜的）同步办法——要么拆成两次 launch，要么用原子操作。

`examples/smem_basics.py` 去掉 `__syncthreads()` 后，20 次运行错了几百万个元素（每次数量不同）；
用 `compute-sanitizer --tool racecheck` 能直接定位到读写冲突的两行源码：

```
========= Error: Race reported between Write access at void block_reverse_kernel<(bool)0>(...)+0x150 in cuda.cu:27
=========     and Read access at void block_reverse_kernel<(bool)0>(...)+0x160 in cuda.cu:30 [4194304 hazards]
```

## 6.4 Bank conflict

shared memory 被分成 **32 个 bank**，每个 bank 4 字节宽，字节地址 a 落在 bank `(a / 4) % 32`。
一个 warp 的一次访问中，若有 k 个线程访问**同一个 bank 的不同地址**，硬件要分 k 次完成（k-way conflict）。
多个线程访问**同一个地址**不算冲突（广播）。

`examples/bank_conflicts.py`：一个 warp，线程 t 反复读 `smem[t * stride]`（指针追逐，测延迟）：

| stride | 用到的 bank 数 | 冲突路数 | 每次读的周期（实测） |
|---|---|---|---|
| 0（同一地址） | 1 | 1（广播） | 29.1 |
| 1 | 32 | 1 | 29.1 |
| 2 | 16 | 2 | 31.1 |
| 8 | 4 | 8 | 43.1 |
| 32 | 1 | 32 | 91.0 |
| **33** | 32 | 1 | **29.1** |

这里只有一个 warp，测到的是延迟，每多一路多 ~2 周期。真实 kernel 里几十个 warp 抢 shared memory 带宽，
k-way 冲突意味着吞吐降到 1/k，代价大得多（看下面的转置）。

### 经典案例：矩阵转置

```
in [M, N]（行主序）                     out [N, M]
         tx →                                    tx →
    ┌────────────┐  ① 合并地按行读进 tile   ┌────────────┐
 ty │ in tile    │ ───────────────────────► │ smem tile  │ tile[ty][tx] = in[row][col]
 ↓  └────────────┘                          └────────────┘
                                                  │ ② __syncthreads
                                                  ▼
                       ③ 按列从 tile 里读，     out[row'][col'] = tile[tx][ty]
                          合并地按行写 out      ← 一个 warp 读 tile 的一列！
```

第 ③ 步一个 warp 读 `tile[0..31][c]`：`tile[32][32]` 时这 32 个地址相距 32 个 float = 128 字节，**全在同一个 bank**，32-way 冲突。
声明成 `tile[32][33]`（每行多一个没用的 float，"padding"），相邻行错开一个 bank，冲突消失。练习 1 实测（8192×8192 fp32）：

| 实现 | 带宽 |
|---|---|
| naive（读合并，写跨 M 个 float） | 455 GB/s |
| shared memory 分块，`tile[32][32]` | 1707 GB/s |
| shared memory 分块，`tile[32][33]` | **2753 GB/s** |
| `x.t().contiguous()`（PyTorch） | 966 GB/s |
| `x.clone()`（同样字节数的纯拷贝，参考上限） | 2931 GB/s |

另一种消除冲突的办法是 **swizzle**：不加 padding，而是把列下标和行下标做异或 `tile[r][c ^ r]`，不浪费空间。
Triton 和 CUTLASS 在 shared memory 里布局 tile 时用的就是 swizzle（单元 07、10 会再见到）。

## 6.5 Occupancy：每个 SM 上能同时驻留多少 warp

H100 每个 SM 最多驻留 64 个 warp（2048 线程）、32 个 block。实际能放多少，取决于**最紧的那个资源**：

| 资源 | 每 SM 总量 | 一个 block 用多少 |
|---|---|---|
| 线程 | 2048 | blockDim |
| 寄存器 | 65536 | 每线程寄存器数 × blockDim（按分配粒度向上取整） |
| shared memory | 最多 228 KB | 静态 + 动态 shared memory |
| block 数 | 32 | 1 |

`examples/occupancy.py` 用一个每线程拷一个 float 的 copy kernel，额外申请用不到的动态 shared memory 来人为降低 occupancy：

| 动态 smem/block | block/SM | warp/SM | occupancy | 带宽（实测） |
|---|---|---|---|---|
| 0 KB | 8 | 64 | 100% | 2353 GB/s |
| 32 KB | 6 | 48 | 75% | 1884 GB/s |
| 48 KB | 4 | 32 | 50% | 1380 GB/s |
| 100 KB | 2 | 16 | 25% | 787 GB/s |
| 200 KB | 1 | 8 | 12% | 438 GB/s |

原理（Little 定律）：要跑满带宽，"在路上"的字节数 = 带宽 × 延迟。每个线程只有 1 个 4 字节的 load 在路上时，
只能靠更多的驻留线程来凑。另一条路是 **ILP**：每个线程同时发好几个独立的 load（float4、循环展开）。
GEMM、FlashAttention 这类 kernel 通常 occupancy 很低（每 SM 一两个 block），靠的就是 ILP + 异步拷贝。
所以 **occupancy 不是越高越好**，它只是藏延迟的手段之一。

怎么查 kernel 用了多少资源：
- `cudaFuncGetAttributes(&attr, kernel)` → `attr.numRegs`、`attr.sharedSizeBytes`；
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, threads, smem)` 直接算出每 SM 的 block 数（见 occupancy 示例）。
- 编译时：`load_cuda(..., extra_cuda_cflags=["--resource-usage"], verbose=True)`，ptxas 会打印
  `Used 10 registers, used 1 barriers, 256 bytes smem`。
- 限制寄存器：`__global__ void __launch_bounds__(256, 2) k(...)` 告诉编译器"block 最多 256 线程，希望每 SM 至少 2 个 block"，
  它会相应压缩寄存器（可能导致溢出到 local memory）。
- ncu 的 Occupancy 一节（单元 02）。

## 6.6 归约：从 shared memory 树到 warp shuffle

求和、求最大值、softmax、LayerNorm 的核心都是归约。分三层：**线程内 → block 内 → grid 内**。

**线程内**：grid-stride 循环在寄存器里累加尽可能多的元素（便宜，而且减少后面层级的工作量）。

**block 内，方法一：shared memory 树**（`examples/reduction_smem.py`）

```cpp
sdata[tid] = v; __syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {     // 每轮活跃线程减半
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
}
```

实测 sum of 2²⁵ fp32（只读 128 MB）：

| 版本 | 要点 | 带宽 |
|---|---|---|
| v1 | `if (tid % (2*s) == 0)`：活跃线程散落在各个 warp，分支发散 | 814 GB/s |
| v2 | `index = 2*s*tid`：活跃线程连续，但访问跨度 2s → bank conflict | 674 GB/s |
| v3 | 顺序寻址 `tid < s`：无发散、无冲突 | 832 GB/s |
| v4 | v3 + 先 grid-stride 在寄存器里累加，block 数 = SM×8 | 2070 GB/s |
| torch.sum | | 2106 GB/s |

在 H100 上 v1→v3 的差别远小于 v3→v4：每个线程只处理 1 个元素时，瓶颈是 8 轮 `__syncthreads` 和巨量的 block，
发散和 bank conflict 是次要的（v2 比 v1 还慢，说明 bank conflict 的代价是实打实的）。
**先让每个线程干足够多的活**，几乎总是归约优化的第一步。

**block 内，方法二：warp shuffle**。同一个 warp 的线程可以直接读彼此的寄存器，不经过 shared memory、不需要 `__syncthreads`：

```cpp
// 归约到 lane 0（down）：offset = 16, 8, 4, 2, 1，5 步完成 32 个数的求和
for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
// all-reduce（xor 蝶形）：结束后 32 个 lane 都拿到总和
for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffff, v, offset);
```

`0xffffffff` 是参与的 lane 的掩码（全部 32 个）。block 归约 = 每个 warp 先 shuffle 归约 → 每个 warp 的 lane 0 写进 `__shared__ float warp_sums[32]`
→ `__syncthreads()` → 第 0 个 warp 再 shuffle 一次。只需要 1 次（all-reduce 版本 2~3 次）`__syncthreads`。

**grid 内**：block 之间不能同步，两种办法：
1. **两遍 launch**：每个 block 写一个部分和到 `partial[blockIdx.x]`，再 launch 一次归约 partial。结果**确定**（加法顺序固定）。
2. **atomicAdd**：每个 block 的 0 号线程 `atomicAdd(out, block_sum)`，一遍完成。快，但浮点加法不满足结合律，
   而 block 完成的顺序每次不同 → **结果每次都可能不一样**（练习 2 里同一输入跑 10 次得到 7 种结果）。
   训练中需要逐位复现时（`torch.use_deterministic_algorithms`），这类 kernel 必须换成方法 1。

练习 2 的实现（shuffle + block 归约 + atomicAdd）实测 2205 GB/s，略快于 torch.sum 的 2117 GB/s。

## 6.7 Warp 发散

同一个 warp 里的线程走不同分支时，硬件把两条路径串行执行（不走的线程被屏蔽）。要点：
- 发散只在 **warp 内部**有代价。`if (threadIdx.x < 64)` 这种按 warp 边界切开的分支没有代价；`if (threadIdx.x % 2)` 才有。
- 边界检查 `if (i < n)` 只让最后一个 warp 发散，可以忽略。
- 归约 v1 那种"活跃线程散布在所有 warp 里"的写法，是最常见的发散来源：活跃线程越来越少，却每个 warp 都还得跑。
- sm_70 之后每个线程有独立的程序计数器（independent thread scheduling），发散的线程可能不会在你以为的地方重新汇合。
  所以 shuffle 一定要用 `_sync` 版本并给对 mask，不能依赖"warp 天然同步"的老假设。

## 6.8 原子操作与私有化

`atomicAdd(&addr, v)` 保证读-改-写不被打断。global memory 上的原子在 L2 执行；同一地址的原子会串行化。
**私有化（privatization）**：每个 block 先在 shared memory 里做原子（快、只有 block 内竞争），最后每个地址往全局只加一次。
练习 4 的直方图实测（n = 2²⁴）：

| 数据 | bins | 全局原子 | shared memory 私有化 | torch.bincount |
|---|---|---|---|---|
| 均匀 | 256 | 4328 µs | 54 µs | 209 µs |
| 全部相同 | 256 | 12320 µs | 48 µs | 2311 µs |
| 均匀 | 8192 | 264 µs | 82 µs | 364 µs |

bins 越少，同一地址上的竞争越激烈，全局原子越慢。私有化把全局原子的次数从 n 降到 (block 数 × bins)。

## 6.9 CUDA ↔ Triton 对照（本单元的部分）

| CUDA（本单元） | Triton | 说明 |
|---|---|---|
| 合并访存：`threadIdx.x` 沿连续维度 | 自动：编译器按"最内层维度连续"给线程分元素 | 但你的 tile 形状仍然决定能不能合并（单元 01 练习 3 的 1×1024 vs 32×128） |
| `__shared__` + `__syncthreads` | 看不到。编译器在需要时（`tl.dot` 的操作数、布局转换、`tl.trans`）自动用 shared memory 并插入屏障 | |
| padding / swizzle 消除 bank conflict | 自动 swizzle | `tl.dot` 的操作数在 `k.asm["ttgir"]` 里是 `#ttg.nvmma_shared<{swizzlingByteWidth = 128, ...}>`（128 字节 swizzle）；`tl.trans` 只显示为 `ttg.convert_layout`，shared memory 在更后面的 lowering 里才分配，`k.metadata.shared` 能看到用了多少（64×64 fp32 转置：16384 字节） |
| block 归约（shuffle + smem） | `tl.sum(x, axis=0)`、`tl.max` | 编译器生成的正是 shuffle + shared memory 的代码 |
| `atomicAdd` | `tl.atomic_add(ptr, val)` | 同样不确定 |
| occupancy：blockDim、寄存器、smem | `num_warps`、`num_stages`、BLOCK 大小 | `k.n_regs`、`k.metadata.shared` 查资源 |
| 两遍 launch 做 grid 归约 | 一样要两遍（或原子） | program 之间也不能同步 |
| warp 级编程（每个 warp 一行） | 没有 warp 的概念；用"一个 program 处理一行 + 小 num_warps"近似 | 这是 Triton 表达不了、CUDA 能做得更好的典型场景之一 |

## 示例

| 文件 | 内容 |
|---|---|
| `examples/coalescing.py` | 不同 stride / 偏移的 gather，量化合并访存的影响 |
| `examples/smem_basics.py` | shared memory 倒序；去掉 `__syncthreads` 的数据竞争；racecheck |
| `examples/bank_conflicts.py` | 用 `clock64()` 数一个 warp 在不同 stride 下读 shared memory 的周期 |
| `examples/occupancy.py` | 用多余的动态 shared memory 压低 occupancy，看带宽怎么掉；查询寄存器数和每 SM block 数 |
| `examples/reduction_smem.py` | shared memory 树形归约 v1~v4，和 torch.sum 对比 |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_transpose.py` | shared memory 分块转置，PAD=0/1 两个版本 | 合并访存、`__syncthreads`、bank conflict、padding |
| `ex2_reduce_sum.py` | warp shuffle → block 归约 → atomicAdd，一遍求和 | `__shfl_down_sync`、block reduce、原子、确定性 |
| `ex3_softmax.py` | 按行 softmax：长行一个 block 一行（三遍扫描）；短行一个 warp 一行（寄存器里只读一遍） | 数值稳定、all-reduce、`__shfl_xor_sync`、寄存器数组 |
| `ex4_histogram.py` | shared memory 私有化直方图，和全局原子对比 | 动态 shared memory、原子、私有化 |

做完后回答（不检查）：
1. 转置时为什么只有"从 tile 里按列读"那一步有 bank conflict，"按行写进 tile"那一步没有？如果 tile 元素是 bf16（2 字节），`[32][33]` 的 padding 还对吗？
2. 练习 2 里 block 数取 SM×4。改成 SM×1 或者"每 256 个 float4 一个 block"会怎样？对 atomicAdd 的竞争有什么影响？
3. 练习 3 的 warp softmax 在 N=128 时比 torch 快，在 N=4096 时用的是 block 版本。如果把 warp 版本的 COLS 提到 128（N=4096），寄存器会怎样？
   用 `--resource-usage` 编译看看，或者用 occupancy 示例里的方法查。
4. 读完 6.9 的表：哪些优化 Triton 会替你做？哪些它做不了？
