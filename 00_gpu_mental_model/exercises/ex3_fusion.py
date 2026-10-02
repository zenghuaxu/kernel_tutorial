"""练习 00-3：融合到底省了什么

y = gelu(x * a + b) * c，x 是 [T, D]，a/b/c 是 [D]（广播到每一行），都是 bf16。

要做：
  1. predict_eager_kernels()：先**猜**PyTorch eager 会启动几个 kernel——测试会用 torch.profiler 实际数一遍来对答案
  2. predict_eager_bytes(T, D, elem_bytes)：eager 一共读写了多少字节（只算 [T, D] 大小的 tensor，忽略 a/b/c）
  3. predict_fused_bytes(T, D, elem_bytes)：理想的单个融合 kernel 读写多少字节

torch.compile 会把这串操作融合成一个 Triton kernel（测试会数 kernel 个数确认）。
最后打印预测加速比（字节比）和实测加速比。

做完之后想一想：
  - 实测加速比和字节比差多少？如果实测比字节比还大，说明 eager 的某个 kernel 连带宽都没跑满——是哪个？
    （提示：用 02 单元会学的 torch.profiler 看每个 kernel 的耗时；或者现在就看 count_kernels 里的 prof.key_averages()）
  - torch.compile 生成的那个 Triton kernel 长什么样？跑一下
    TORCH_LOGS=output_code python 00_gpu_mental_model/solutions/ex3_fusion.py 2>&1 | less
    看看它和你在单元 01 写的 kernel 有多像。

运行：python 00_gpu_mental_model/exercises/ex3_fusion.py
"""
import warnings

import torch
import torch.nn.functional as F

from common import bench, check, check_equal, finish, gbps, report


def chain_eager(x, a, b, c):
    """x: [T, D]，a/b/c: [D]，全是 bf16。"""
    return F.gelu(x * a + b) * c


def predict_eager_kernels() -> int:
    """chain_eager 会启动几个 CUDA kernel？"""
    raise NotImplementedError  # TODO


def predict_eager_bytes(T: int, D: int, elem_bytes: int) -> int:
    """chain_eager 一共读写多少字节 HBM（忽略 [D] 大小的 a/b/c，只算 [T, D] 大小的 tensor）。"""
    raise NotImplementedError  # TODO


def predict_fused_bytes(T: int, D: int, elem_bytes: int) -> int:
    """理想的融合 kernel：读 x 一次、写 y 一次。"""
    raise NotImplementedError  # TODO


chain_fused = torch.compile(chain_eager)


def count_kernels(fn, *args) -> int:
    """用 torch.profiler 数 fn(*args) 实际启动了多少个 CUDA kernel。"""
    warnings.filterwarnings("ignore", message=".*Profiler clears events.*")
    fn(*args)  # 预热（torch.compile 第一次调用会编译）
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn(*args)
        torch.cuda.synchronize()
    return sum(1 for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA)


if __name__ == "__main__":
    torch.manual_seed(0)
    T, D = 8192, 4096
    x = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    a, b, c = (torch.randn(D, device="cuda", dtype=torch.bfloat16) for _ in range(3))

    check_equal("eager kernel 个数（预测 vs profiler 实测）", predict_eager_kernels(), count_kernels(chain_eager, x, a, b, c))
    check_equal("融合后 kernel 个数（torch.compile，profiler 实测）", count_kernels(chain_fused, x, a, b, c), 1)
    check_equal("eager 字节数 T=8192 D=4096 bf16", predict_eager_bytes(T, D, 2), 536870912)
    check_equal("融合字节数 T=8192 D=4096 bf16", predict_fused_bytes(T, D, 2), 134217728)
    check("torch.compile 结果正确", chain_fused(x, a, b, c), chain_eager(x, a, b, c), atol=2e-2, rtol=2e-2)

    ms_eager = bench(lambda: chain_eager(x, a, b, c))
    ms_fused = bench(lambda: chain_fused(x, a, b, c))
    rows = [
        dict(impl="eager", us=ms_eager * 1e3, MB=predict_eager_bytes(T, D, 2) / 1e6,
             GBps=gbps(predict_eager_bytes(T, D, 2), ms_eager)),
        dict(impl="torch.compile", us=ms_fused * 1e3, MB=predict_fused_bytes(T, D, 2) / 1e6,
             GBps=gbps(predict_fused_bytes(T, D, 2), ms_fused)),
    ]
    report(rows, "gelu(x * a + b) * c，[8192, 4096] bf16")
    print(f"\n预测加速比（字节比）: {predict_eager_bytes(T, D, 2) / predict_fused_bytes(T, D, 2):.2f}x"
          f"   实测: {ms_eager / ms_fused:.2f}x")
    finish()
