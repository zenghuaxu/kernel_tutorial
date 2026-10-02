# 05 · CUDA C++ 入门：线程、block、grid，以及怎么接进 PyTorch

> 前置：单元 01（Triton 的 program / block / mask）、单元 00（memory-bound 的概念）。
> 这一单元开始写 CUDA C++。目的不是替代 Triton，而是看清 Triton 在底下替你做了什么——
> 以及 Triton 表达不了的东西（warp 级原语、精确控制 shared memory、Hopper 的新指令）怎么写。

## 5.1 Host 与 device：两段代码，两块内存，异步执行

一个 CUDA 程序有两部分：

| | host（CPU） | device（GPU） |
|---|---|---|
| 代码 | 普通 C++ 函数 | `__global__` / `__device__` 函数 |
| 内存 | 主存（`malloc`） | 显存 HBM（`cudaMalloc`，或者 torch 在 GPU 上分配的 tensor） |
| 谁调用 | Python → pybind → host 函数 | host 用 `kernel<<<grid, block>>>(...)` 启动 |

**kernel 启动是异步的**：`<<<>>>` 只是把任务塞进一个 *stream*（GPU 的任务队列）就立刻返回，CPU 继续往下跑。
所以：
- 计时必须用 CUDA event 或者先 `torch.cuda.synchronize()`（单元 02 细讲）；
- kernel 里的错误（比如非法地址）**不会在 launch 那一行报出来**，而是在之后某个同步点冒出来，经常指错位置。

函数修饰符：

| 修饰符 | 在哪执行 | 谁能调用 | 说明 |
|---|---|---|---|
| `__global__` | device | host（`<<<>>>`） | kernel 入口，返回 `void` |
| `__device__` | device | device | 辅助函数，通常被内联。加 `__forceinline__` 强制内联 |
| `__host__ __device__` | 两边都编一份 | 两边 | 写一次两边用（比如小的数学函数） |

## 5.2 线程层次：thread → warp → block → grid

```
grid  (gridDim.x × gridDim.y × gridDim.z 个 block)
 └── block (blockDim.x × blockDim.y × blockDim.z 个 thread，同一个 block 一定在同一个 SM 上)
      └── warp (32 个 threadIdx 连续的线程，硬件真正的调度单位)
           └── thread
```

每个线程执行**同一份** kernel 代码，靠内建变量区分自己：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;      // 一维全局下标
int j = blockIdx.y * blockDim.y + threadIdx.y;      // 二维时第二个坐标
```

`examples/hello_cuda.py` 会把每个线程的 `blockIdx / threadIdx / warp / lane` 打印出来，跑一遍感受一下。

**二维映射的惯例**：`threadIdx.x` 永远对应**内存里连续的那一维**（行主序矩阵的列 j）。
原因在 5.4：一个 warp 是 `threadIdx.x` 连续的 32 个线程，让它们访问连续地址，一次内存事务就能喂饱整个 warp。
练习 2 里有一个反着映射的对照 kernel，在这台 H100 上实测：正确映射 **2210 GB/s**，反过来 **465 GB/s**。

### 启动配置的硬限制（H100 / sm_90）

| 项目 | 上限 |
|---|---|
| 每个 block 的线程数 | **1024**（blockDim.x×y×z ≤ 1024；单维 x,y ≤ 1024，z ≤ 64） |
| gridDim.x | 2³¹−1 |
| gridDim.y、gridDim.z | **65535**（所以"大的那一维"放 x） |
| 每个 SM 同时驻留 | 最多 2048 个线程、32 个 block、64K 个 32 位寄存器 |
| 每个线程的寄存器 | 最多 255 个 |
| 每个 block 的 shared memory | 默认 48 KB，opt-in 最多 227 KB（单元 06） |
| SM 个数 | 132 |

超过上限 → launch 失败（`invalid argument` / `invalid configuration argument`）。
`CUDA_CHECK_LAUNCH()` 宏（= `C10_CUDA_KERNEL_LAUNCH_CHECK()`）会在 launch 后立刻把这类错误变成 Python 的 `RuntimeError`。

### block 大小怎么选

- 选 32 的倍数（不满一个 warp 的部分是浪费）。128 / 256 是最常用的默认值。
- block 太小（如 32）：每个 SM 最多 32 个 block → 只有 32×32 = 1024 个线程驻留，填不满 2048 的上限，延迟藏不住。
- block 太大（1024）：一个 block 的寄存器/shared memory 需求大，可能一个 SM 只能放一个，而且 `__syncthreads` 等得更久。
- block 数：至少要让 132 个 SM 都有活干，最好每个 SM 有好几个 block 轮换（"多波"）。

## 5.3 Warp 与 SIMT

GPU 的执行单位是 warp：32 个线程**同一时刻执行同一条指令**，各自用自己的寄存器（Single Instruction, Multiple Threads）。

- **分支发散（divergence）**：`if (threadIdx.x % 2) A(); else B();` 时，同一 warp 里一半线程走 A 一半走 B，
  硬件只能先跑 A（B 那一半闲着）、再跑 B。代价是两边时间之和。边界检查 `if (i < n)` 这种只在最后一个 warp 发散，无所谓。
- **延迟隐藏**：一次 HBM 读要几百个周期。SM 不会傻等，而是切到另一个已经就绪的 warp（切换零开销）。
  所以需要**足够多的驻留 warp**（occupancy，单元 06）或者**每个线程有多个独立的 load 在路上**（ILP，比如 float4、循环展开）。

和 Triton 对比：Triton 里你看不到 warp，`num_warps=4` 只是告诉编译器"这个 program 用 128 个线程"，
它自动把 BLOCK 个元素分给这些线程（每个线程若干个，而且尽量连续、向量化）。

## 5.4 把 kernel 接进 PyTorch：load_inline 做了什么

`common.load_cuda(name, SRC, ["fn"])` 是 `torch.utils.cpp_extension.load_inline` 的薄封装：

1. 在 SRC 前面拼上 `<torch/extension.h>`、`cuda_fp16.h`、`cuda_bf16.h` 和几个宏（`CHECK_INPUT`、`CUDA_CHECK_LAUNCH`）；
2. 从源码里抽出 `fn` 的声明，生成 pybind11 绑定代码（这样 Python 里能 `mod.fn(tensor)`）；
3. 用 ninja 调 nvcc（`.cu`）和 g++（绑定代码）编译，链接成 `.so`，`import` 进来；
4. 结果缓存在 `$TORCH_EXTENSIONS_DIR`，名字里带源码哈希，源码不变第二次就是秒加载。

我们传给 nvcc 的选项：`-O3 -lineinfo --expt-relaxed-constexpr -std=c++17`，外加 `TORCH_CUDA_ARCH_LIST=9.0` → `-gencode=arch=compute_90,code=sm_90`。
**torch 还会自动加** `-D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__`，
意思是 `__half` / `__nv_bfloat16` 不能隐式和 float 互转、不能直接用 `+ *` 运算——必须用显式的 intrinsic（5.6 节）。

一个典型的 host 函数：

```cpp
#include <ATen/cuda/CUDAContext.h>

torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    CHECK_INPUT(x); CHECK_INPUT(y);                   // 在 GPU 上、contiguous
    TORCH_CHECK(x.sizes() == y.sizes(), "shape 不一致");  // 失败 -> Python RuntimeError
    auto out = torch::empty_like(x);                  // 用 torch 分配输出（走 caching allocator）
    int64_t n = x.numel();
    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    add_kernel<<<blocks, threads, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                               out.data_ptr<float>(), n);
    CUDA_CHECK_LAUNCH();
    return out;
}
```

**为什么一定要传 stream**：`<<<blocks, threads>>>` 不写第 4 个参数时用的是默认 stream（legacy stream 0）。
PyTorch 允许用户切换 stream（`with torch.cuda.stream(s):`），如果你的 kernel 不在 PyTorch 的当前 stream 上，
和前后算子之间就没有顺序保证——读到还没算完的输入。第三个参数是动态 shared memory 字节数（单元 06）。

## 5.5 访问 tensor 的数据

| 方式 | 写法 | 说明 |
|---|---|---|
| 裸指针 | `x.data_ptr<float>()` | 最常用。指向**第一个元素**（已经算上 storage_offset）。你自己负责 stride |
| 只读裸指针 + 限定 | `const float* __restrict__` | `__restrict__` 告诉编译器指针之间没有别名，可以更激进地优化（比如走只读缓存） |
| accessor | `x.packed_accessor32<float, 2, torch::RestrictPtrTraits>()` | kernel 里用 `x[i][j]`，自动按 stride 算地址；`32` 表示下标用 int32。写原型方便 |

要点：
- **下标溢出**：`int` 最大 2³¹−1 ≈ 21 亿。一个 `[65536, 32768]` 的 tensor 就超了。
  `blockIdx.x * blockDim.x` 是 `unsigned int` 乘法，先转 `(int64_t)` 再乘。
- 非 contiguous 输入：要么 `TORCH_CHECK(x.is_contiguous())` 拒绝、要么在 Python 侧 `.contiguous()`、要么把 stride 传进 kernel（像单元 01 练习 3 那样）。

## 5.6 数据类型：float / half / bf16 和向量类型

| 类型 | 字节 | 说明 |
|---|---|---|
| `float` | 4 | |
| `__half` / `c10::Half` | 2 | `cuda_fp16.h`；c10 版本是 torch 的包装 |
| `__nv_bfloat16` / `c10::BFloat16` | 2 | `cuda_bf16.h`；两者内存布局相同，可以 `reinterpret_cast` |
| `float2` / `float4` | 8 / 16 | 内建向量类型，`.x .y .z .w` |
| `__nv_bfloat162` / `__half2` | 4 | 两个打包的 bf16 / fp16 |
| `uint4` | 16 | 常被当作"16 字节的原始数据"来做向量化 load（= 8 个 bf16） |

转换（因为 torch 关掉了隐式转换，这些要背下来）：

```cpp
float f = __bfloat162float(b);            __nv_bfloat16 b = __float2bfloat16(f);   // 最近偶数舍入
float2 f2 = __bfloat1622float2(b2);       __nv_bfloat162 b2 = __floats2bfloat162_rn(f2.x, f2.y);
float f = __half2float(h);                __half h = __float2half(f);
// c10 类型更省事：static_cast<float>(c10_bf16_value)、static_cast<c10::BFloat16>(f)
```

和 Triton 一样的原则：**低精度存储，fp32 计算**。

### 模板 + AT_DISPATCH：一份 kernel 多种 dtype

```cpp
template <typename scalar_t>
__global__ void gelu_kernel(const scalar_t* x, scalar_t* out, int64_t n) {
    ...  float v = static_cast<float>(x[i]);  out[i] = static_cast<scalar_t>(gelu(v));
}

AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "gelu", [&] {
    gelu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), n);
});
```

宏会展开成一个 `switch (x.scalar_type())`，每个分支里把 `scalar_t` typedef 成对应 C++ 类型
（`float`、`double`、`c10::Half`、`c10::BFloat16`），然后执行 lambda。不支持的 dtype 会抛
`"gelu" not implemented for 'Int'`。完整例子：`examples/dispatch_and_accessors.py`。

## 5.7 错误检查与调试

两类错误：

1. **launch 错误**（配置非法、kernel 镜像不匹配架构）：launch 后立刻 `CUDA_CHECK_LAUNCH()` 就能拿到。
2. **执行错误**（非法地址、misaligned address、kernel 里 `assert` 失败）：异步的，在后续某个同步点才冒出来，
   而且一旦发生，这个进程的 CUDA context 就坏了，后面所有 CUDA 调用都失败，只能重启进程。
   调试时用 `CUDA_LAUNCH_BLOCKING=1 python xxx.py` 让每次 launch 都同步，报错位置就准了。

更阴险的第三类：**越界写但没有报错**。PyTorch 的 caching allocator 会从大块显存里切小块给 tensor，
越界写很可能落在隔壁 tensor 里——不崩溃，只是数据悄悄错了。`examples/debug_oob.py` 现场演示：
一个忘了 `if (i < n)` 的 kernel 把隔壁 tensor `b` 从全 0 改成了全 1，没有任何报错。

抓这类 bug 的工具是 **compute-sanitizer**（从 sglang 容器里拷到了 `.tools/compute-sanitizer/`）：

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1 .tools/compute-sanitizer/compute-sanitizer --print-limit 1 \
    python 05_cuda_basics/examples/debug_oob.py
# ========= Invalid __global__ write of size 4 bytes
# =========     at fill_ones_buggy(float *, long)+0xb0 in cuda.cu:21
# =========     by thread (100,0,0) in block (0,0,0)
# =========     Access to 0x7f48d8800190 is out of bounds
# =========     and is 1 bytes after the nearest allocation at 0x7f48d8800000 of size 400 bytes
```

`PYTORCH_NO_CUDA_MEMORY_CACHING=1` 很关键：不关缓存池，越界写落在池子内部，sanitizer 认为是合法地址，报 0 个错误。
其他工具：`--tool racecheck`（shared memory 数据竞争，单元 06 用得上）、`--tool initcheck`（读了未初始化的显存）。
kernel 里也可以 `printf(...)`（输出在同步时刷出，只配小 grid 用）和 `assert(cond)`。

## 5.8 Grid-stride 循环

"一个线程一个元素"要求 block 数随 n 增长。另一种写法是 block 数固定（比如 SM 数 × 若干），每个线程循环处理多个元素：

```cpp
int64_t stride = (int64_t)gridDim.x * blockDim.x;          // 整个 grid 的线程总数
for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    out[i] = x[i] * alpha + beta;
}
```

注意每一轮里，整个 grid 的线程访问的仍然是**连续的**一段 `[k*stride, (k+1)*stride)`——访存依旧合并。
（反面例子：让每个线程处理连续的一段 `[tid*chunk, (tid+1)*chunk)`，同一时刻一个 warp 的 32 个线程地址相距 chunk，完全不合并。）

好处：任意 n 都能用同一个 grid；可以在循环外做一次性的初始化（比如把常量读进寄存器）；
一个 kernel 的 block 数可控（这对 reduction、persistent kernel 很重要）。这就是单元 01 练习 4 的 persistent kernel 的 CUDA 版。

## 5.9 向量化访存

每个线程一次读 16 字节（`float4` / `uint4`）而不是 4 字节或 2 字节：
- load 指令数减为 1/4（fp32）或 1/8（bf16），下标计算也少了；
- 每个线程同时有更多字节"在路上"，更容易喂满带宽。

```cpp
const float4* x4 = reinterpret_cast<const float4*>(x);
float4 v = x4[i];                      // 编译成一条 ld.global.v4.f32（Triton 单元 01 里看到的 ld.global.v4.b32 就是它）
```

条件：**地址必须 16 字节对齐**，否则 `misaligned address` 错误。torch 新分配的 tensor 至少 256 字节对齐，
但切片视图不一定（`x[1:]` 的首地址偏移 4 字节）。所以 host 里要检查 `ptr % 16 == 0`，不满足就走标量回退；
另外 n 不是 4 的倍数时，剩下的尾巴要单独处理。

在这张 H100 上实测（训练任务同时在跑，数字有噪声）：

| kernel | 数据 | 标量 | 向量化 |
|---|---|---|---|
| fp32 axpb（练习 3），n=2²⁷ | 读写各 512 MB | 2715 GB/s | 2803 GB/s |
| bf16 SwiGLU（练习 4），n=8M | 读 32 MB 写 16 MB | 1499 GB/s | 2053 GB/s |

fp32 时差别不大——相邻线程的 4 字节访问本来就被合并成整条 cache line 了。bf16 每个线程只读 2 字节，
标量版本的指令开销占比大得多，向量化收益明显。Triton 会自动帮你做这件事（在 PTX 里看到的 `.v4`），CUDA 里得自己写。

## 5.10 CUDA ↔ Triton 对照

| CUDA | Triton（单元 01） | 说明 |
|---|---|---|
| block（`blockIdx`） | program（`tl.program_id`） | 都是"一个 SM 上一起调度的一组工作" |
| `blockDim.x` 个线程，每个处理 1 个（或几个）元素 | `num_warps*32` 个线程，一共处理 `BLOCK` 个元素 | Triton 自动决定每个线程管哪几个元素 |
| `int i = blockIdx.x*blockDim.x + threadIdx.x` | `offs = pid*BLOCK + tl.arange(0, BLOCK)` | 标量下标 vs 下标向量 |
| `if (i < n)` | `mask = offs < n` | |
| `x[i]`、`float4` 手写向量化 | `tl.load(x_ptr + offs, mask)` 自动向量化 | |
| `gridDim`、`<<<grid, block, smem, stream>>>` | `kernel[grid](..., num_warps=)` | Triton 自动用当前 torch stream |
| 模板 + `AT_DISPATCH` | 指针的 dtype 自动特化 | Triton 对每种 dtype 组合 JIT 一份 |
| `__device__` 函数 | 另一个 `@triton.jit` 函数 | 都会被内联 |
| grid-stride 循环 | `for i in range(pid, nblk, tl.num_programs(0))` | |
| `threadIdx`、warp、lane、shuffle、shared memory | **看不到** | Triton 替你管；单元 06 开始讲的东西大多在这一层 |

一句话：Triton 的抽象层级是"block"，CUDA 的是"thread"。block 内部的事（线程分工、向量化、shared memory、同步）
Triton 编译器替你做了；CUDA 里全部自己来，换来的是完全的控制权。

## 示例

| 文件 | 内容 |
|---|---|
| `examples/hello_cuda.py` | 带注释的 vector add；打印线程编号；故意用 2048 线程触发 launch 错误；和 torch 比带宽 |
| `examples/dispatch_and_accessors.py` | `__device__` 函数、模板 + `AT_DISPATCH`（fp32/fp16/bf16）、`packed_accessor` 处理非 contiguous 输入 |
| `examples/debug_oob.py` | 越界写悄悄改坏隔壁 tensor；用 compute-sanitizer 定位到源码行 |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_vector_add.py` | 自己写 kernel 体和 block 数 | 全局下标、边界检查、int64 |
| `ex2_bias_2d.py` | `out = x*scale + bias[None,:]`，二维 block/grid；和反向映射的对照组比带宽 | dim3、二维下标、合并访存预告 |
| `ex3_vectorized.py` | grid-stride + float4 的 `alpha*x+beta`，处理 n%4 的尾巴，未对齐时回退 | grid-stride、float4、对齐 |
| `ex4_swiglu_dispatch.py` | A：模板 + AT_DISPATCH 的 SwiGLU（fp32/fp16/bf16）；B：bf16 专用的 16 字节向量化版本 | 模板、dtype、`__nv_bfloat162` |

做完后回答（不检查）：
1. 练习 2 里两种线程映射的带宽差了约 5 倍。一个 warp 的一次 load，在两种映射下分别涉及多少条 128 字节的 cache line？
2. `examples/debug_oob.py` 里，如果 `a` 有 128 个元素（正好 512 字节），越界写还会改坏 `b` 吗？用 sanitizer 验证你的猜测。
3. 练习 4 的 bf16x8 版本，n 的尾巴是怎么处理的？如果 n 很大且是 8 的倍数，尾巴代码会有开销吗？
4. 什么时候你会选 CUDA 而不是 Triton？（读完单元 06、07 再回来回答一次）
