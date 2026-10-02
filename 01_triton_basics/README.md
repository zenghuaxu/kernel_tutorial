# 01 · Triton 入门：program、block、mask、指针

> 前置：单元 00（知道什么是 memory-bound，知道 H100 HBM 带宽约 3.35 TB/s）。

## 1.1 一个 kernel 在 GPU 上是怎么跑的（Triton 视角）

写 CUDA 时你要管"每个线程做什么"；写 Triton 时你管的是 **"每个 program 做什么"**，
一个 program 处理一整块数据（一个 *block*），块内的并行由编译器替你分给 32×num_warps 个线程。

```
kernel[grid](args...)          grid = (G,)  → 启动 G 个 program，编号 0..G-1
                               每个 program 是同一段代码，只是 tl.program_id(0) 不一样
```

最小例子（完整版见 `examples/vector_add.py`）：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)                 # 我是第几个 program
    offs = pid * BLOCK + tl.arange(0, BLOCK)    # 我负责的 BLOCK 个元素的下标（一个向量！）
    mask = offs < n                             # 最后一个 block 可能越界
    x = tl.load(x_ptr + offs, mask=mask)        # 指针 + 下标向量 = 指针向量，一次加载一整块
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, BLOCK=1024)      # 传 torch.Tensor 会自动变成指向首元素的指针
```

要点：

| 概念 | 说明 |
|---|---|
| `x_ptr` | 传进来的 tensor 在 kernel 里就是一个**指向首元素的指针**，类型由 dtype 决定（`*fp32`、`*bf16`...）。Triton **不知道 shape 和 stride**，要你自己传 |
| `tl.arange(0, BLOCK)` | 编译期长度的向量。BLOCK **必须是 2 的幂** |
| `tl.constexpr` | 编译期常量。不同取值会编译出不同的 kernel（并被缓存）。块大小、开关类参数都用它 |
| `mask` | 越界的 lane 不读不写。`tl.load(..., mask=m, other=0.0)` 给被 mask 掉的位置一个默认值 |
| grid | 一维、二维、三维都行：`grid=(a, b)` 后用 `tl.program_id(0)`、`tl.program_id(1)`。也可以写成 `lambda meta: (triton.cdiv(n, meta["BLOCK"]),)` |

## 1.2 数据类型：低精度存储，fp32 计算

LLM 里 tensor 一般是 bf16。规范做法：**load 进来立即 `.to(tl.float32)`，算完 store 时再转回去**
（`tl.store` 会自动转换成指针的元素类型，但显式写 `.to(out_ptr.dtype.element_ty)` 更清楚）。
bf16 只有 8 位尾数，在 bf16 里做中间计算（尤其是累加、exp）会明显掉精度。
elementwise kernel 是 memory-bound 的，fp32 计算几乎是免费的。

常用数学：`tl.exp`、`tl.log`、`tl.sqrt`、`tl.rsqrt`、`tl.sigmoid`、`tl.maximum`、`tl.where(cond, a, b)`；
`tl.math` / `tl.extra.cuda.libdevice` 里还有 `tanh`、`erf` 等。

## 1.3 二维 block 与 stride

Triton 不懂 shape，所以多维 tensor 要自己用 stride 算地址。
对一个 `[M, N]` 的 tensor，元素 `(i, j)` 的地址 = `base + i*stride_m + j*stride_n`。
借助广播，可以一次生成一个二维 tile 的地址：

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)            # [BLOCK_M]
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)            # [BLOCK_N]
ptrs = x_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n   # [BLOCK_M, BLOCK_N]
mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
tile = tl.load(ptrs, mask=mask)
```

传 stride：`x.stride(0), x.stride(1)`。这样转置视图（`x.t()`）、切片（`x[:, ::2]`）都能直接处理，不用先 `.contiguous()`。
**但**：最后一维 stride 不是 1 时，相邻 lane 读的地址不相邻，访存不再合并（coalesced），带宽会掉很多（单元 06 会细讲）。
所以 kernel 一般按"最内层维度连续"来设计，必要时在 wrapper 里检查 `x.stride(-1) == 1`。

## 1.4 BLOCK 和 num_warps 怎么选

- `num_warps`（默认 4）：每个 program 用多少个 warp（×32 线程）。block 大，就多给几个 warp。
- 对 elementwise：每个 program 处理 1024~8192 个元素、4~8 个 warp 一般就够了。
  太小 → program 太多、每个干的活太少，调度开销占比大；太大 → program 太少填不满 132 个 SM，或者寄存器不够。
- 经验目标：elementwise kernel 的有效带宽达到 HBM 峰值的 **80~90%**（H100 上 ~2.8 TB/s）就算写好了。

有效带宽 = (最少需要读写的字节数) / 时间。`common.gbps(nbytes, ms)` 帮你算。

## 1.5 Persistent kernel：program 数 = SM 数

默认做法是"一个 block 一个 program"，数据大时会启动几十万个 program，由硬件排队调度。
另一种写法是**只启动 ~SM 个数的 program，每个 program 在循环里处理多个 block**：

```python
pid = tl.program_id(0)
nprog = tl.num_programs(0)
for block_id in range(pid, num_blocks, nprog):     # Triton 里 range 的边界可以是运行时值
    offs = block_id * BLOCK + tl.arange(0, BLOCK)
    ...
```

对 elementwise 来说收益不大，但这个模式在 matmul（单元 04）、attention（单元 08）、Hopper 的 warp specialization（单元 10）里非常重要：
它让一个 program 可以跨 tile 复用资源、做流水线。

## 1.6 调试小技巧

- `TRITON_INTERPRET=1 python xxx.py`：在 CPU 上用 numpy 解释执行 kernel，**可以在 kernel 里 `print`、下断点**。慢，只用于小输入。
- `tl.device_print("x", x)`：在 GPU 上打印（输出很多，只配小 grid 用）。
- `tl.static_assert(BLOCK % 16 == 0)`：编译期断言。
- 改了 kernel 不生效？Triton 按源码哈希缓存，一般不会；真怀疑就删 `$TRITON_CACHE_DIR`。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/vector_add.py` | 带详细注释的 vector add，和 torch 对比正确性与带宽 |
| `examples/inspect_kernel.py` | 看看一次 launch 之后 Triton 生成了什么：寄存器数、PTX 片段 |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_vector_add.py` | 自己从零写 vector add 的 kernel 体 | program_id、arange、mask |
| `ex2_swiglu.py` | 融合 `silu(gate) * up`，bf16 进出、fp32 计算；和 PyTorch eager 比带宽 | 融合、dtype |
| `ex3_strided_2d.py` | 给任意 stride 的 2D tensor 加行偏置 + 缩放，支持转置视图 | 2D block、stride、广播 |
| `ex4_persistent.py` | 把 ex1 改写成 persistent kernel（grid = SM 数），并把 grid-stride 循环写对 | num_programs、循环 |

做完后回答（不检查）：
1. ex2 里 PyTorch eager 版本 `F.silu(g) * u` 一共读写了多少字节？你的融合版本呢？速度比和字节比接近吗？
2. ex3 里同一个转置视图，1x1024 的 tile 和 32x128 的 tile 带宽差了好几倍。为什么 tile 形状能"救回"非连续访存？
3. ex4 里如果 grid 设成 SM 数的 2 倍、4 倍会怎样？
