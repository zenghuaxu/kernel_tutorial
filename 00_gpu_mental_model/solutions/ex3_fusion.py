"""练习 00-3：融合到底省了什么（参考答案）"""
import warnings

import torch
import torch.nn.functional as F

from common import bench, check, check_equal, finish, gbps, report


def chain_eager(x, a, b, c):
    """x: [T, D]，a/b/c: [D]，全是 bf16。"""
    return F.gelu(x * a + b) * c


def predict_eager_kernels() -> int:
    """chain_eager 会启动几个 CUDA kernel？"""
    return 4   # mul、add、gelu、mul


def predict_eager_bytes(T: int, D: int, elem_bytes: int) -> int:
    """chain_eager 一共读写多少字节 HBM（忽略 [D] 大小的 a/b/c，只算 [T, D] 大小的 tensor）。"""
    # 每个 kernel 读一个 [T, D]、写一个 [T, D]
    return predict_eager_kernels() * 2 * T * D * elem_bytes


def predict_fused_bytes(T: int, D: int, elem_bytes: int) -> int:
    """理想的融合 kernel：读 x 一次、写 y 一次。"""
    return 2 * T * D * elem_bytes


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
