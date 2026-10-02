# GPU Kernel 教程（H100 / Triton / CUDA）

目标：从"知道 GPU 很快"走到"能为自己的模型写出接近硬件上限的 kernel"，
最后一站是能写、能改 FlashAttention 类的 kernel（包括 sliding window / GQA / decode）。

每个单元 = 一篇讲义（`README.md`）+ 可运行的示例（`examples/`）+ 练习（`exercises/`）+ 参考答案（`solutions/`）。
练习都是"填空 + 自动测试"：打开文件找 `TODO`，写完直接运行，看到 `全部 N 个检查通过 ✓` 就算过关。

## 0. 环境（已经装好了）

```bash
cd ~/kernel-tutorial          # 实际位置 /mnt/HD/workspace/kernel-tutorial（根分区太满）
source env.sh                 # 每个新终端执行一次：激活 venv、设置 CUDA_HOME、自动挑最空的 GPU
python check.py 01 --solutions   # 确认环境：参考答案应该全部 PASS
```

- Python 3.11 + torch 2.11 (cu130) + triton 3.6 + nvcc 13.0（pip 版，`.cuda_home/` 是拼出来的 CUDA_HOME）
- Nsight Compute 2025.3.1 在 `.tools/`，通过 `tools/ncu.sh` 调用（需要 sudo，见单元 02）
- 8 张 H100 上都在跑训练，每张只剩 ~5GB 显存。练习都很小，但 **benchmark 数字会有噪声**，
  看相对快慢、看占峰值的百分比量级即可。想固定用某张卡：`export CUDA_VISIBLE_DEVICES=3` 后再 `source env.sh`。
- 环境坏了：`bash setup.sh` 重建。

## 1. 怎么做练习

```bash
python 01_triton_basics/exercises/ex1_vector_add.py   # 单个练习，完整输出
python check.py 01                                    # 单元 01 所有练习，汇总 PASS/FAIL
python check.py 01 ex2                                # 只跑 ex2
python check.py --solutions                           # 跑所有参考答案
```

建议节奏：先读讲义 → 跑一遍 `examples/` 里的代码 → 做练习 → 卡住超过 30 分钟再看 `solutions/`。
每个练习文件顶部写了目标、提示和"做完之后想一想"的问题，后者没有自动检查，但比通过测试更重要。

## 2. 路线图

| 单元 | 主题 | 你会写出的东西 |
|---|---|---|
| [00](00_gpu_mental_model/) | （选读，不写 kernel）GPU 硬件与性能模型 | roofline 估算器；decode / speculative decoding 上限分析 |
| [01](01_triton_basics/) | Triton 入门：program、block、mask、指针 | vector add、融合 SwiGLU、2D strided kernel、persistent kernel |
| [02](02_profiling/) | 计时与 profiling | 正确的 benchmark；读 ncu 报告；看 Triton 生成的 PTX；interpreter 调试 |
| [03](03_triton_reductions/) | 归约与归一化 | softmax、RMSNorm、残差+RMSNorm 融合、大词表 cross-entropy |
| [04](04_triton_matmul/) | Triton 矩阵乘 | 分块 GEMM、L2 友好的 program 排序、autotune、epilogue 融合 |
| [05](05_cuda_basics/) | CUDA C++ 入门 | thread/block/grid、load_inline、向量化访存、bf16 |
| [06](06_cuda_memory/) | CUDA 内存层次 | 合并访存、shared memory、bank conflict、warp shuffle 归约 |
| [07](07_cuda_gemm/) | CUDA GEMM 一步步优化 | 从 naive 到寄存器分块，再到 Tensor Core (WMMA / mma.sync) |
| [08](08_flash_attention/) | FlashAttention | online softmax、FA 前向、causal、sliding window + GQA、decode |
| [09](09_backward_and_integration/) | 反向与框架集成 | autograd.Function、RMSNorm 反向、custom_op + torch.compile |
| [10](10_hopper/) | Hopper 专属特性 | TMA、wgmma、warp specialization、FP8、CuTe DSL |

**直接从 01 开始写 kernel**；00 是纸笔估算，讲"kernel 最快能多快"，写 kernel 时遇到看不懂的性能数字再回头看。

依赖关系：01 → 02 → 03 → 04 → 08 → 09 是 **Triton 主线**，学完就能干活；
05 → 06 → 07 是 **CUDA 主线**，帮你理解 Triton 在底下替你做了什么、以及 Triton 做不到时怎么办；
10 需要两条线都走过。只想尽快上手的话：01、02、03、04、08、09。

## 3. 目录约定

```
NN_topic/
  README.md            讲义
  examples/*.py        讲义里引用的完整示例，直接 python 运行
  exercises/exK_*.py   练习（有 TODO）
  solutions/exK_*.py   参考答案（和练习同名，测试代码完全一样）
common/                check() / bench() / load_cuda() 等公用工具
tools/ncu.sh           Nsight Compute 包装
check.py               批量跑练习
```

## 4. 参考资料（按推荐顺序）

- Triton 官方教程：https://triton-lang.org/main/getting-started/tutorials/index.html
- *Programming Massively Parallel Processors*（PMPP，第 4 版）—— CUDA 基础的教科书
- Simon Boehm, *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance* —— 单元 07 的蓝本
- Tri Dao, FlashAttention 1/2/3 论文 —— 单元 08、10
- NVIDIA H100 白皮书 / CUDA C++ Programming Guide / PTX ISA —— 查手册用
- GPU MODE 讲座（YouTube）和 https://github.com/gpu-mode/lectures
