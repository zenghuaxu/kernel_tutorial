# 02 · 计时与 profiling：先学会量，再谈优化

> 前置：单元 01（会写 elementwise Triton kernel）。
> 本单元所有数字都是在**这台被训练任务占用的 H100** 上实测的，你自己跑会有 ±10% 的出入。

写 kernel 的工作循环是：**写对 → 量准 → 找瓶颈 → 改 → 再量**。"量准"这一步错了，后面全白干。
这一单元讲四件工具：

| 工具 | 回答的问题 | 成本 |
|---|---|---|
| CUDA event / `do_bench` | 这个 kernel 多快？比 baseline 快多少？ | 一行代码 |
| `torch.profiler` | 这段 PyTorch 代码启动了哪些 kernel？时间花在哪？ | 几行代码 |
| Triton 编译产物（PTX、`n_regs`） | 编译器把我的代码变成了什么？向量化了吗？溢出寄存器了吗？ | 一行代码 |
| Nsight Compute (`ncu`) | 这个 kernel **为什么**慢？卡在 DRAM、L2、还是指令发射上？ | 要 sudo，每个 kernel 重放十几遍 |

再加一个调试工具：`TRITON_INTERPRET=1`，在 CPU 上逐个 program 解释执行 kernel，可以 `print`、可以下断点。

---

## 2.1 GPU 计时为什么容易测错

`examples/timing_pitfalls.py` 把四个坑都演示了一遍。实测输出（节选）：

```
== 坑 1：kernel launch 是异步的 ==
  不 synchronize: 0.021 ms   ← 只测到了 CPU 把 kernel 塞进队列的时间
  synchronize 后: 0.514 ms
  do_bench      : 0.507 ms
== 坑 2：第一次调用包含 JIT 编译 / 加载 ==
  第一次调用:  419.943 ms
  第二次调用:    0.070 ms  (含 Python launch 开销 + sync 往返)
  do_bench  :    0.016 ms
== 坑 3：数据还在 L2 里（H100 L2 = 50MB）==
  copy   16MB: 不清 L2    13.8 us   清 L2    17.4 us   do_bench    17.4 us
  copy  128MB: 不清 L2    93.9 us   清 L2    94.7 us   do_bench    94.7 us
== 坑 4：太小的 kernel，测到的是 launch 开销 ==
  1024 个元素的 copy: 5.4 us（实际 GPU 干活不到 1us）
```

1. **异步**。`kernel[grid](...)` 和 PyTorch 的 CUDA 算子都只是把命令塞进 stream 的队列就返回了。
   用 `time.perf_counter()` 不同步，测到的是 CPU 排队时间（几到几十微秒），和 kernel 真正的执行时间毫无关系。
2. **第一次调用**。Triton 第一次遇到一组新的（constexpr, dtype, 对齐特化）组合会编译（几百毫秒，或者从磁盘缓存加载），
   cuBLAS 第一次调用会选算法、分配 workspace。永远要**预热**。
3. **L2 缓存**。H100 的 L2 有 50MB。同一份 16MB 数据反复 copy，第二次起大部分命中 L2，快了 ~20%。
   但真实模型里，上一层的输出等你用的时候多半已经被挤出 L2 了。所以 benchmark 默认在每次计时前**写一个 256MB 的 buffer 把 L2 冲掉**。
4. **launch 开销**。一次 kernel launch 本身在 CPU+GPU 上就要几微秒。几微秒级别的 kernel，优化 kernel 本身意义不大，
   该做的是**融合**（少发几个 kernel）或者 **CUDA Graph**（把一串 launch 录下来一次性重放）。

## 2.2 正确的计时：CUDA event 与 `do_bench`

CUDA event 是插在 stream 里的"时间戳命令"：GPU 执行到它时记下当前时刻。
两个 event 夹住 kernel，差值就是 GPU 上的执行时间，**和 CPU 什么时候提交无关**：

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(10):          # 预热
    fn()
cache.zero_()                # 冲 L2（不在计时区间内）
start.record()               # 往 stream 里插一个时间戳
fn()
end.record()
torch.cuda.synchronize()     # 等 GPU 真正执行到 end
ms = start.elapsed_time(end)
```

`triton.testing.do_bench(fn)` 就是这个套路的完整版：先估一次耗时，据此决定预热和重复次数；每次重复前清 L2；
默认返回**中位数**（这台机器上有别人的训练任务，中位数比平均数抗干扰）。本教程的 `common.bench` 就是它。
还有一个 `triton.testing.do_bench_cudagraph`：把 fn 录成 CUDA Graph 再测，能扣掉大部分 launch 开销，适合测很小的 kernel。

**有效带宽**和**有效算力**是比较时最好用的两个归一化指标：

```
有效带宽  GB/s  = 这个运算"最少"要读写的字节数 / 时间        → 和 HBM 峰值 3.35 TB/s 比
有效算力 TFLOPS = 这个运算的浮点运算数 / 时间               → 和 bf16 dense 峰值 ~990 TFLOPS 比
```

注意"最少"二字：分母用的是理论下限，而不是 kernel 实际搬了多少（实际搬得多恰恰说明 kernel 写得差）。

## 2.3 torch.profiler：看清一段代码发了哪些 kernel

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("mlp_block"):          # 自定义区间，会出现在表格和时间线里
        y = mlp_block(x, ...)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
prof.export_chrome_trace("trace.json")          # 拖进 https://ui.perfetto.dev 看时间线
```

`examples/torch_profiler_demo.py` 跑一个 LLaMA 式的 MLP 块（RMSNorm → gate/up GEMM → SwiGLU → down GEMM → 残差）。
实测每次迭代 **16 个 GPU kernel**：3 个 GEMM（`nvjet_sm90_...`，cuBLAS 的 Hopper kernel）每个 ~60-70us，
外加 RMSNorm 拆成的 8 个小 kernel（`to(float)`、`pow`、`mean`、`+eps`、`rsqrt`、`mul`、`to(bf16)`、`mul`）、
SwiGLU 的 2 个、残差 1 个，以及 cuBLAS 自己发的 Memset。GEMM 之外的这些小 kernel 就是**融合**的目标。

想只拿 GPU kernel：

```python
kernels = [e for e in prof.events()
           if e.device_type == torch.autograd.DeviceType.CUDA   # GPU 上的事件
           and not e.is_user_annotation]                         # 去掉 record_function 在 GPU 上的区间
```

Triton kernel 在 profiler 里的名字就是 `@triton.jit` 函数名；`torch.compile` 生成的 kernel 叫
`triton_per_fused_add_mean_mul_pow_rsqrt_0` 这种（实测：eager RMSNorm 6 个 kernel，compile 之后 1 个）。

## 2.4 看 Triton 的编译产物

launch 的返回值是一个 `CompiledKernel`，身上有编译器的全部产出：

```python
k = my_kernel[grid](...)
k.n_regs          # 每个线程用了多少寄存器（H100 每个 SM 64K 个，用得多 → 同时能驻留的 warp 少）
k.n_spills        # 寄存器溢出到 local memory 的数量。> 0 通常是性能灾难，要减小 BLOCK 或拆分计算
k.metadata.shared # 用了多少字节 shared memory
k.asm["ttgir"]    # TritonGPU IR：能看到每个 tensor 被分配的 layout（每个线程拿几个元素）
k.asm["ptx"]      # PTX 汇编
```

最常用的检查是**访存有没有向量化**。在 PTX 里搜 `ld.global`：

| PTX | 一条指令读 | 说明 |
|---|---|---|
| `ld.global.v4.b32` | 128 bit（4 个 fp32 或 8 个 bf16） | 理想 |
| `ld.global.b32` | 32 bit | fp32 标量，或 2 个 bf16 |
| `ld.global.b16` | 16 bit | bf16 标量——最差 |

要生成 128-bit load，编译器必须**证明**三件事：

1. **每个线程拿到连续的 ≥ 4 个（fp32）/ 8 个（bf16）元素**。每线程元素数 = BLOCK / (32 × num_warps)。
   实测 bf16 copy：BLOCK=128、4 warps → 每线程 1 个 → `b16`；BLOCK=256 → `b32`；BLOCK=1024 → `v4.b32`。
2. **地址 16 字节对齐**。Triton 对指针参数会检查是否 16 字节对齐，对整数参数会检查**是否能被 16 整除**（以及是否等于 1），
   并据此特化编译出不同版本。所以 `row * stride` 里的 stride=1024 没问题，stride=1000 就不行。
3. **mask 在每组内一致**。`mask = offs < n`：n 能被 16 整除时，连续 16 个元素的 mask 必然一样；
   n=1001 时编译器无法证明，就只能逐元素 load。实测 fp32 copy 只因为 n=1001，就从 `v4.b32` 退化成 `b32`。

当**你知道**而编译器不知道时，可以告诉它：

```python
# wrapper 已经保证 N % 8 == 0。数值上什么也没改，但编译器由此推出 N 能被 8 整除
N = N // 8 * 8
offs = tl.max_contiguous(tl.multiple_of(offs, 8), 8)   # 对 tensor 的提示：offs 每 8 个连续、起点是 8 的倍数
```

注意：实测 Triton 3.6 中 `tl.multiple_of` 用在**标量参数**上（`N = tl.multiple_of(N, 16)`）不起作用，
`N // 8 * 8` 这个写法有效。另一条路是把 N、stride 声明成 `tl.constexpr`（代价：每个不同的值都要重新编译一次）。

向量化值多少？实测 32768×1000 的 bf16 逐行缩放：`b16` load 2430 GB/s → `v4.b32` load 2620 GB/s，约 8%。
纯 memory-bound 的 kernel 里，硬件的合并访存已经帮了大忙，所以差距不是几倍；
但在计算密集的 kernel 里，load 指令条数直接占发射带宽，差距会更大。

## 2.5 调试：解释器模式

```bash
TRITON_INTERPRET=1 python my_script.py
```

Triton 会把 kernel 在 CPU 上用 numpy 逐个 program 执行（GPU tensor 会被拷过去再拷回来）：

- kernel 里可以直接写 Python `print(...)`，打印出来的是 numpy 数组；可以 `breakpoint()` 单步
- 越界**读**往往直接报错；越界**写**会写坏 CPU 堆内存，进程可能以 `free(): invalid size` 之类的错误崩溃——这本身就是线索
- 很慢，只用小输入、小 grid；`TRITON_INTERPRET` 必须在 import triton **之前**设置，所以用环境变量

`examples/interpret_debug.py` 正常跑和加 `TRITON_INTERPRET=1` 各跑一次对比一下。实测解释器下的输出：

```
row [0] offs [0 1 2 3 4 5 6 7] mask [ True  True  True  True  True False False False] x [-0.92 -0.43 -2.64 0.15 -0.12 -inf -inf -inf] max [0.145]
```

在 GPU 上打印用 `tl.device_print("x=", x)`：每个线程打印它持有的元素，输出量巨大，只在 grid=(1,) 时用。
注意 BLOCK 小于线程数时，同一个元素会被好几个线程各打一遍（数据在线程间是复制的）。

其他好用的：`tl.static_assert(cond)` 编译期断言；`tl.device_assert(cond, "msg")` 运行期断言
（只在 `TRITON_DEBUG=1` 时生效）；把可疑的中间结果 `tl.store` 到一个额外的输出 buffer 里拿回 Python 看。

## 2.6 Nsight Compute：kernel 为什么慢

`ncu` 会把指定的 kernel 重放十几遍（每遍收集一组硬件计数器），然后告诉你硬件各单元的忙碌程度。
这台机器上读计数器要 root，而且 GPU 上有别人的训练任务，所以统一用包装脚本（见 `tools/ncu.sh` 里的说明）：

```bash
# 只看名字匹配 copy 的 kernel；--once 让脚本每个 kernel 只跑一次，免得 ncu 抓到几百次 benchmark 调用
bash tools/ncu.sh --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    -k regex:copy python 02_profiling/examples/copy_kernels.py --once
```

常用参数：`-k regex:名字` 过滤 kernel；`-c N` 最多抓 N 个；`--launch-skip N` 跳过前 N 个；
`--section X` 选择要看的部分（`--list-sections` 列出全部）；`--set full` 全部收集（慢）。

`examples/copy_kernels.py` 里三个 kernel 都是搬 64MB → 64MB，实测：

| kernel | 时间 | 有效带宽 | ncu: DRAM Throughput | L2 Throughput | Achieved Occupancy |
|---|---|---|---|---|---|
| `copy_coalesced`（fp32，BLOCK=4096） | 44 us | ~2.6 TB/s | **77%** | 78% | 81% |
| `copy_column`（按列读 row-major 矩阵） | 417 us | ~0.32 TB/s | 13% | **87%** | 88% |
| `copy_scalar`（bf16，BLOCK=128） | 160 us | ~0.82 TB/s | 22% | 26% | **38%** |

怎么读：

- **GPU Speed Of Light (SOL)**：`Memory Throughput` 和 `Compute (SM) Throughput` 是"离硬件上限还有多远"的百分比，
  取各个内存/计算单元里**最忙**的那个。copy_coalesced 的 DRAM 77% → 已经接近 HBM 能给的上限，这个 kernel 写好了。
- **copy_column**：DRAM 只有 13%，但 L2 87% —— 瓶颈不在 HBM 而在 L2。每个线程读 4 字节，但相邻线程地址差 16KB，
  每次都要从 L2 搬一整个 32 字节的 sector。ncu 直接告诉你了（`--set full` 后看 Memory Workload Analysis Tables）：
  ```
  OPT   Est. Speedup: 76.17%
        The memory access pattern for global loads from L2 might not be optimal. On average, only 4.0 of the 32 bytes
        transmitted per sector are utilized by each thread. ... This could possibly be caused by a stride between threads.
  ```
- **copy_scalar**：DRAM、L2 都闲着（22%、26%），Occupancy 只有 38%（理论 100%）。每个 program 只有 128 个线程、
  每线程只搬 1 个 bf16，program 刚启动就结束，SM 大部分时间在调度新的 block 而不是在发 load。
  这种"什么都不忙"的 kernel，问题是**并行度/每线程工作量**不够。
- **Occupancy**：每个 SM 上同时驻留的 warp 数 / 最大值（H100 是 64）。`Block Limit Registers / Shared Mem / Warps`
  告诉你是哪个资源限制了理论 occupancy。occupancy 不是越高越好（单元 04、07 的 GEMM 故意用低 occupancy 换大 tile），
  但对 memory-bound kernel，太低意味着没有足够的 in-flight load 来掩盖延迟。

保存报告，之后慢慢看：

```bash
bash tools/ncu.sh --set full -k regex:copy_column -c 1 -o 02_profiling/out/copy_column -f \
    python 02_profiling/examples/copy_kernels.py --once
# 读报告不需要 root，直接调用 ncu 本体（包装脚本加的 --clock-control 和 -i 冲突）
$KT_ROOT/.tools/nsight-compute/ncu -i 02_profiling/out/copy_column.ncu-rep --page details
$KT_ROOT/.tools/nsight-compute/ncu -i 02_profiling/out/copy_column.ncu-rep --page raw --csv \
    --metrics dram__bytes_read.sum,lts__t_sectors_srcunit_tex_op_read.sum
```

实测后一条输出：DRAM 读了 100.6 MB（理论只需 64MB），L2 为 load 搬了 1658 万个 sector = 530MB —— 8 倍放大。

注意事项：
- `.ncu-rep` 是 sudo 写的，属主是 root；要删就 `sudo rm`，或者 `sudo chown $USER 文件`。
- 包装脚本用了 `--clock-control none`（不锁频，避免拖慢别人的训练），所以 ncu 报告里的绝对时间会随 GPU 频率浮动，
  看百分比和计数器，别看绝对时间。
- 每个 kernel 要重放十几遍：一定用 `-k` / `-c` 限定范围，不然一个训练 step 能跑一个小时。

## 2.7 一套够用的工作流

1. 写完先和 PyTorch 参考实现比对正确性（多种 shape，包括非 2 的幂、非 16 的倍数）
2. `bench` 测时间，换算成有效带宽 / TFLOPS，和峰值比 —— 到了 80%+ 就收手
3. 差得远：先看 `k.n_spills`、PTX 的 load 宽度这些"便宜"的信息
4. 还看不出来：`ncu --section SpeedOfLight`，看是哪个单元在忙（或者谁都不忙）
5. 在模型层面：`torch.profiler` 找出最值得融合 / 替换的 kernel 序列

## 示例

| 文件 | 内容 |
|---|---|
| `examples/timing_pitfalls.py` | 计时的 4 个坑：异步、首次调用、L2、launch 开销 |
| `examples/torch_profiler_demo.py` | profile 一个 MLP 块：表格、GPU kernel 列表、导出 Perfetto 时间线 |
| `examples/copy_kernels.py` | 三个快慢不同的 copy kernel，给 ncu 当靶子（`--once` 模式） |
| `examples/interpret_debug.py` | 解释器模式下在 kernel 里 `print`；GPU 上的 `tl.device_print` |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_event_timer.py` | 用 CUDA event 实现计时器：预热、清 L2，和 `do_bench` 对上 | event、异步、L2 |
| `ex2_find_the_bug.py` | 一个能跑但算错的 2D kernel，藏了 3 个 bug，用解释器找出来 | `TRITON_INTERPRET`、mask、stride |
| `ex3_vectorize.py` | 解析 PTX 求 load 位宽；修好一个因 N=1000 而没向量化的 kernel | PTX、特化、对齐 |
| `ex4_count_kernels.py` | 用 `torch.profiler` 数一段 PyTorch 代码发了几个 kernel | profiler、融合机会 |

做完后回答（不检查）：
1. `do_bench` 为什么用中位数而不用平均数或最小值？在这台共享的机器上，三者分别会有什么偏差？
2. copy_column 的 DRAM 实际读了 100MB 而不是 530MB（L2 搬了 530MB），中间差的部分去哪了？
3. 一个 kernel 的 SOL 显示 Memory 30%、Compute 20%，你下一步看什么？
4. 用 `bash tools/ncu.sh --section SpeedOfLight -k regex:swiglu --launch-skip 4 -c 1 python 01_triton_basics/solutions/ex2_swiglu.py` 去 profile 单元 01 的 SwiGLU（跳过前 4 次小测试用例的 launch），它的 DRAM Throughput 是多少？和你算的有效带宽一致吗？
