"""示例：用 torch.profiler 看一段 PyTorch 代码到底启动了哪些 kernel。

运行：python 02_profiling/examples/torch_profiler_demo.py
产物：02_profiling/out/trace.json —— 用 https://ui.perfetto.dev 打开（拖进去即可），能看到 CPU 端的
      op 调用和 GPU 端的 kernel 时间线，以及两者之间的对应关系。
"""
import os
import warnings

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "out")


def rmsnorm_eager(x, w, eps=1e-6):
    # 一个"教科书式"的 RMSNorm：每个算子都是一个独立的 kernel
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * w


def mlp_block(x, w_norm, w_gate, w_up, w_down):
    h = rmsnorm_eager(x, w_norm)
    g = h @ w_gate
    u = h @ w_up
    return (F.silu(g) * u) @ w_down + x


if __name__ == "__main__":
    torch.manual_seed(0)
    d, ff, T = 2048, 5632, 2048
    dt = torch.bfloat16
    x = torch.randn(T, d, device="cuda", dtype=dt)
    w_norm = torch.ones(d, device="cuda", dtype=dt)
    w_gate = torch.randn(d, ff, device="cuda", dtype=dt) * 0.02
    w_up = torch.randn(d, ff, device="cuda", dtype=dt) * 0.02
    w_down = torch.randn(ff, d, device="cuda", dtype=dt) * 0.02

    for _ in range(3):   # 预热：别把 cuBLAS 初始化之类的一次性开销算进去
        mlp_block(x, w_norm, w_gate, w_up, w_down)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(5):
            with record_function("mlp_block"):   # 自定义区间，会出现在表格和时间线里
                mlp_block(x, w_norm, w_gate, w_up, w_down)
        torch.cuda.synchronize()

    print("== 按 GPU 总时间排序的前 12 项（5 次迭代累计）==")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))

    # 只看 GPU 上的事件：device_type == CUDA；再去掉 record_function 在 GPU 时间线上留下的区间
    kernels = [e for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation]
    print(f"5 次迭代一共 {len(kernels)} 个 GPU kernel，每次 {len(kernels) // 5} 个（Memset 是 cuBLAS 自己发的）：")
    for e in kernels[: len(kernels) // 5]:
        print(f"  {e.device_time:8.1f} us  {e.name[:90]}")

    os.makedirs(OUT, exist_ok=True)
    path = os.path.abspath(os.path.join(OUT, "trace.json"))
    prof.export_chrome_trace(path)
    print(f"\n时间线已导出: {path}（用 https://ui.perfetto.dev 打开）")
    print("看点：3 个 GEMM 占了绝大部分时间；RMSNorm 和 SwiGLU 被拆成一串小 elementwise kernel——这就是融合的机会。")
