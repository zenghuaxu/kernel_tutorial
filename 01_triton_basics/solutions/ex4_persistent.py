"""练习 01-4：persistent kernel（参考答案）"""
import torch
import triton
import triton.language as tl

from common import bench, check, finish, gbps, report

NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count


@triton.jit
def axpb_persistent_kernel(x_ptr, out_ptr, n, alpha, beta, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    num_blocks = tl.cdiv(n, BLOCK)
    for block_id in range(pid, num_blocks, nprog):
        offs = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * alpha + beta, mask=mask)


def axpb(x: torch.Tensor, alpha: float, beta: float, num_programs: int | None = None) -> torch.Tensor:
    """out = alpha * x + beta，只启动 num_programs 个 program（默认 = SM 个数）。"""
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    if num_programs is None:
        num_programs = NUM_SMS
    num_programs = max(1, min(num_programs, triton.cdiv(n, BLOCK)))
    axpb_persistent_kernel[(num_programs,)](x, out, n, alpha, beta, BLOCK=BLOCK, num_warps=8)
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [5, 4096, 4097, 1_000_003, 1 << 24]:
        x = torch.randn(n, device="cuda")
        check(f"n={n} grid=SMs", axpb(x, 2.0, -1.0), x * 2.0 - 1.0)
    x = torch.randn(123_457, device="cuda")
    for p in [1, 3, 7]:
        check(f"n=123457 grid={p}", axpb(x, 0.5, 3.0, num_programs=p), x * 0.5 + 3.0)

    n = 1 << 26
    x = torch.randn(n, device="cuda")
    rows = []
    for mult in [1, 2, 4, 8]:
        ms = bench(lambda: axpb(x, 2.0, 1.0, num_programs=NUM_SMS * mult))
        rows.append(dict(grid=f"{mult}x SMs", us=ms * 1e3, GBps=gbps(2 * n * 4, ms)))
    ms = bench(lambda: x * 2.0 + 1.0)
    rows.append(dict(grid="torch(2 kernels)", us=ms * 1e3, GBps=gbps(4 * n * 4, ms)))
    report(rows, f"persistent axpb, n=2^26 fp32, SM 数={NUM_SMS}")
    finish()
