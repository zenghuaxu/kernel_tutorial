# 09 · 反向传播与框架集成

> 前置：单元 01、03（RMSNorm 前向）、08（FlashAttention 前向，练习 4 用到）。

前面写的 kernel 都只有前向。训练要反向；放进模型要和 autograd、`torch.compile`、CUDA graph 和平相处。
本单元把一个"能跑的 kernel"变成一个"能放进训练代码的算子"。

---

## 9.1 torch.autograd.Function

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps: float):          # 输入可以混着 tensor 和普通 Python 值
        y, rstd = launch_fwd_kernel(x, w, eps)
        ctx.save_for_backward(x, w, rstd)        # tensor 用 save_for_backward
        ctx.eps = eps                            # 非 tensor 直接挂在 ctx 上
        return y

    @staticmethod
    def backward(ctx, dy):                       # 每个输出一个梯度
        x, w, rstd = ctx.saved_tensors
        dy = dy.contiguous()                     # ← 一定要！见下
        dx, dw = launch_bwd_kernel(x, w, rstd, dy)
        return dx, dw, None                      # 每个输入一个返回值；eps 不可导 → None

y = MyOp.apply(x, w, 1e-6)
```

`examples/autograd_basics.py` 逐条演示了下面这些坑：

| 坑 | 说明 |
|---|---|
| **grad_output 不 contiguous** | `y.sum().backward()` 传进来的 `dy` 是 stride 全 0 的 expand 视图；`y.t()` 之后做逐元素运算，`dy` 是转置视图。kernel 若按 `row * N + col` 取地址就读错了——而且**不会报错**。backward 开头 `dy = dy.contiguous()`。 |
| 存什么 | `save_for_backward` 存的 tensor 会一直活到反向结束，占激活显存。能重算就别存：SwiGLU 存 `g, u` 而不是 `silu(g)`；RMSNorm 存每行一个 `rstd`（fp32 标量）而不是归一化后的 `x̂`；FlashAttention 存 `LSE` 而不是 `P`。 |
| `ctx.needs_input_grad` | 一个 bool tuple，对应 forward 的每个输入。冻结的权重不用算梯度。 |
| 返回值个数 | backward 返回值个数必须等于 forward 输入个数（不含 ctx）。 |
| dtype | 梯度的 dtype 要和对应输入一致（bf16 输入返回 bf16 梯度），内部累加用 fp32。 |

## 9.2 怎么验证反向是对的

三层验证，从严到松：

1. **gradcheck**（fp64 有限差分）：`torch.autograd.gradcheck(fn, inputs)` 对每个输入扰动 ±eps，数值梯度和解析梯度比。
   最严格，但要求 kernel **在 fp64 下计算**——如果 kernel 里一律 `.to(tl.float32)`，fp64 的输入被降精度，gradcheck 会失败。
   练习 1 的做法：传一个 `ACC: tl.constexpr` dtype，fp64 输入用 `tl.float64`，否则 `tl.float32`。
2. **和 PyTorch 参考实现的 autograd 比**（fp32）：参考实现用 torch 原生算子写，梯度靠 autograd 自动求。容差 1e-5 量级。
3. **bf16 和"fp32 真值"比**：bf16 输入，参考在 fp32 里算。容差怎么定？一个稳妥的办法是**和同类实现的误差比**：
   练习 4 里，我们的 FlashAttention 反向误差不能超过 `SDPA (bf16) vs fp32 真值` 误差的 2 倍。
   这比拍脑袋的 `atol=1e-2` 更有说服力：既不会松到放过 bug（实测：漏掉 Δ 项时 dQ 误差 0.14，而容差约 0.004），
   也不会严到连 cuDNN 都过不了。

另外，**测试形状要刁钻**：长度 1、非 2 的幂、不是块大小整数倍、多维输入、非 contiguous 输入/梯度。

## 9.3 逐元素反向：SwiGLU

`out = silu(g) · u`，`silu(g) = g·σ(g)`：

```
∂out/∂u = silu(g)
∂out/∂g = u · silu'(g),   silu'(g) = σ(g) + g·σ(g)(1−σ(g)) = σ(g)·(1 + g·(1−σ(g)))
```

反向 kernel 读 `g, u, dout`，写 `dg, du`：5 次访存。前向 3 次。实测（共享 H100，`[8, 2048, 1408]` bf16）
前向+反向 ~132 μs，PyTorch eager ~232 μs。

## 9.4 有归约的反向：RMSNorm

前向：`r = (mean(x²) + eps)^(-1/2)`，`y = x · r · w`（逐行，`x` 是一行 `[N]`）。

**dx**（逐行独立）：记 `g = dy ⊙ w`（对 `x̂ = x·r` 的梯度），

```
y_j = w_j · x_j · r,     ∂r/∂x_k = −r³ · x_k / N

dx_k = Σ_j g_j · ∂(x_j r)/∂x_k = g_k · r + Σ_j g_j x_j · (−r³ x_k / N)

dx = r · g − r³ · x · mean(g ⊙ x)
```

一行内需要一次归约（`mean(g ⊙ x)`），和前向一样，一个 program 处理一整行。

**dw**（跨行归约）：`dw = Σ_rows dy ⊙ x̂`。这是一个**跨 program**的归约——M 行由不同 program 处理，结果要加到同一个 `[N]` 向量上。三种做法：

| 做法 | 说明 | 确定性 |
|---|---|---|
| `tl.atomic_add` | 每行直接原子加到 `dw`。简单，但 M 很大时同一地址竞争严重；fp32 加法顺序不固定 | ✗ 每次结果可能差最后几位 |
| **部分和 + 二次归约** | grid = P 个 program（几倍 SM 数），program p 处理行 p, p+P, p+2P…，在寄存器里累加 `[N]` 的部分和，写到 `DW_PART[p]`；之后 `DW_PART.sum(0)` | ✓ 顺序固定 |
| 先存每行贡献再 sum | 需要 `[M, N]` 的临时 buffer，太大 | ✓ |

参考答案用第二种（Liger-Kernel、Apex 的做法）。**确定性**在训练里很重要：
不确定的梯度让 loss 曲线无法复现，debug 时你分不清"改动的影响"和"噪声"。练习 2 会检查两次反向的 `dw` 是否 bitwise 相同。

实测（共享 H100，`[16384, 4096]` bf16 前向+反向）：Triton ~269 μs，torch.compile ~266 μs，PyTorch eager ~3670 μs。
`torch.compile` 对这种"逐行归约 + 逐元素"的模式已经能生成很好的 kernel——自己写的价值在于融合更多东西
（比如残差加 + RMSNorm + 量化），或者编译器不支持的场景。

## 9.5 FlashAttention 反向（练习 4，挑战）

记 `S = scale·QKᵀ`，`P = softmax(S)`，`O = PV`。链式法则：

```
dV = Pᵀ dO                                   [N, D]
dP = dO Vᵀ                                   [N, N]
dS_ij = P_ij · (dP_ij − Δ_i),  Δ_i = Σ_j P_ij dP_ij       （softmax 的反向）
dQ = scale · dS K
dK = scale · dSᵀ Q
```

关键技巧：

1. **Δ 不用 P 算**：`Δ_i = Σ_j P_ij (dO_i · v_j) = dO_i · (Σ_j P_ij v_j) = dO_i · O_i`。
   所以先跑一个预处理 kernel：`Δ = rowsum(dO ⊙ O)`，O(N·D) 的代价。
2. **P 不存，重算**：前向保存了 `LSE_i = log Σ_j exp(S_ij)`，反向 `P_ij = exp(S_ij − LSE_i)`，一步到位（不需要再做 online softmax）。
3. **两个 kernel，避免原子操作**：
   ```
   dK/dV kernel：grid = (N/BN, B·H)，每个 program 固定一个 K/V 块，沿 Q 方向扫描      ┌──────────┐
                 dK、dV 只由这个 program 写，不用原子加                               │ Q 块 ↓   │ K 块 j 固定
   dQ kernel：   grid = (N/BM, B·H)，每个 program 固定一个 Q 块，沿 K 方向扫描        └──────────┘
   ```
   代价：`QKᵀ` 和 `dO Vᵀ` 在两个 kernel 里各算一次。FA2 官方实现是一个 kernel + dQ 原子累加，省计算但不确定。
4. **在 dK/dV kernel 里算转置**：直接算 `Sᵀ = K Qᵀ`（`[BN, BM]`），这样 `dV += Pᵀ dO`、`dK += dSᵀ Q` 都是行数为 BN 的矩阵乘，不用显式转置大矩阵。

反向的 FLOPs 约是前向的 2.5 倍（5 个矩阵乘 vs 2 个）。实测（共享 H100，B=2 H=16 N=4096 D=128，前向+反向）：
我们 ~2.44 ms（~394 TFLOPs），SDPA ~1.79 ms（~539 TFLOPs）。

## 9.6 torch.library.custom_op：让 torch.compile 认识你的 kernel

`torch.compile` 的前端 Dynamo 逐字节码追踪 Python。遇到它不认识的东西，就**断图（graph break）**：
把图切成两段，中间回到 Python 解释执行。断图多了，编译器能做的融合就少了，`fullgraph=True` 时直接报错。

`examples/compile_graph_break.py` 实测（torch 2.11）：

| 接入方式 | 断图数 |
|---|---|
| A. `load_inline` 编的 CUDA 扩展，直接调用 | **1**（"Attempted to call function marked as skipped"） |
| B. 同一个扩展，包成 `torch.library.custom_op` | 0 |
| C. Python 里直接 launch Triton kernel | 0（Dynamo 支持追踪"用户定义的 Triton kernel"） |
| D. `torch.library.triton_op` + `wrap_triton` | 0，且 Inductor 能看到 kernel 本身 |

C 虽然能用，但 **custom_op 是官方推荐的稳妥方式**：它对 CUDA 扩展、Triton、甚至任意 Python 代码都适用；
有 `opcheck` 自检；能和 `torch.export`、AOTAutograd、CUDA graph 配合；autograd 也注册在算子上，而不是靠 `autograd.Function`。

```python
@torch.library.custom_op("mylib::softcap", mutates_args=())      # 名字 = 命名空间::算子名
def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:          # 必须写类型注解，PyTorch 据此生成 schema
    y = torch.empty_like(x)
    softcap_fwd_kernel[grid](x, y, x.numel(), cap, BLOCK=1024)
    return y                                                       # 不能返回输入本身或它的视图

@softcap.register_fake                                             # "fake" 实现：只算输出的 shape/dtype/device
def _(x, cap):
    return torch.empty_like(x)

def setup_context(ctx, inputs, output):                            # 前向之后调用，决定存什么
    x, cap = inputs
    ctx.save_for_backward(output)
    ctx.cap = cap

def backward(ctx, dy):
    (y,) = ctx.saved_tensors
    return softcap_backward(y, dy, ctx.cap), None                  # 反向也调用 custom_op，反向图也不断

softcap.register_autograd(backward, setup_context=setup_context)
```

要点：
- **`register_fake` 必须写**。compile 时用 FakeTensor（没有真实数据的 tensor）追踪，遇到你的算子就调用 fake 实现推断输出形状。
  输出形状依赖于**数据内容**（比如 `nonzero`）时要用 `torch.library.get_ctx().new_dynamic_size()`。
- **`mutates_args`**：如果算子原地修改了某个输入（比如 KV cache 写入），必须在这里声明，否则编译器可能重排或消掉它。
- **`torch.library.opcheck(op, args)`**：检查 schema、fake 实现、autograd 注册、AOT 追踪是否一致。写完必跑。
- 命名空间在进程内全局唯一：同一个名字注册两次会报错（所以别在会被 import 两次的模块里注册）。

## 9.7 CUDA graph 友好

`torch.compile(mode="reduce-overhead")` 或手动 `torch.cuda.CUDAGraph` 会把一串 kernel launch 录下来重放，省掉 CPU launch 开销
（decode 阶段每步几百个小 kernel，这个优化很关键）。要求：

- **不能有 host 同步**：kernel wrapper 里不能 `.item()`、`.cpu()`、`.tolist()`，也不能根据 tensor 的值决定 grid 或分配大小。
  比如 flash-decoding 里 `seqlens` 必须作为 device tensor 传进 kernel，而不是在 Python 里读出来算 grid。
- **用当前 stream**：Triton 自动用当前 stream；CUDA 扩展要用 `at::cuda::getCurrentCUDAStream()`，否则录不进图。
- **地址固定**：重放时输入输出的地址不变，所以 graph 外面要把新数据 copy 到固定 buffer 里（`reduce-overhead` 自动做）。
- Triton 的 autotune 第一次调用会跑 benchmark，要在录制之前 warmup。

## 9.8 测试策略小结

```
单 kernel 正确性：  fp32 vs 参考（严） → bf16 vs fp32 真值（容差参照同类实现）
反向：              gradcheck（fp64） + 对参考实现的 autograd
边界：              长度 1、非 2 的幂、非块整数倍、非 contiguous 输入和梯度、空 batch
确定性：            同输入跑两次 bitwise 比较（需要的话）
集成：              opcheck；torch.compile(fullgraph=True)；compile 前后结果一致
端到端：            小模型训练几百步，loss 曲线和纯 PyTorch 版本重合
```

## 示例

| 文件 | 内容 |
|---|---|
| `examples/autograd_basics.py` | autograd.Function 的结构、needs_input_grad、非 contiguous 的 grad_output、gradcheck |
| `examples/compile_graph_break.py` | 四种接入方式在 torch.compile 下是否断图；custom_op + CUDA graph |

## 练习

| 文件 | 内容 | 关键词 |
|---|---|---|
| `ex1_swiglu_autograd.py` | 写 SwiGLU 的反向 kernel 和 autograd.Function.backward；过 gradcheck（fp64）和 expand 梯度 | 逐元素反向、ACC dtype |
| `ex2_rmsnorm_bwd.py` | RMSNorm 反向：dx 公式 + dw 的部分和归约；检查 bitwise 确定性 | 跨 program 归约、确定性 |
| `ex3_custom_op.py` | 把 softcap 的 Triton kernel 注册成 custom_op：fake 实现、反向 op、register_autograd；过 opcheck 和 fullgraph compile | torch.library |
| `ex4_flash_attn_bwd.py` | **挑战**：FlashAttention 反向——预处理 Δ、dK/dV kernel、dQ kernel；误差不超过 SDPA 的 2 倍 | 重算 P、Δ = rowsum(dO⊙O) |

做完后回答（不检查）：
1. ex2 里把 P（program 数）从 4×SM 改成 M（每行一个 program），`dw_part` 有多大？速度怎么变？
2. ex2 改用 `tl.atomic_add` 累加 dw，确定性测试还能过吗？多跑几次试试。
3. ex4 里 dK/dV kernel 用 BLOCK_M=BLOCK_N=64、num_warps=4。为什么反向的块比前向（128×64、8 warps）小？
   （提示：dK/dV kernel 要同时在寄存器里放 k、v、dk、dv 四个 `[BN, D]` 的块）
4. ex4 支持 causal 需要改哪几处？sliding window 呢？（dK/dV kernel 的 Q 扫描范围：哪些 Q 块能看到这个 K 块？）
