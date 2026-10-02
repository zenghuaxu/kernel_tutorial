"""正确性检查 + 计时。

约定：每个练习文件在 `if __name__ == "__main__":` 里调用若干次 check(...)，
最后调用 finish()。只要有一个 check 失败，进程就以退出码 1 结束，
check.py 据此判断 PASS / FAIL。
"""
import sys

import torch

_failures: list[str] = []
_passes = 0


def _fmt_err(out: torch.Tensor, ref: torch.Tensor) -> str:
    diff = (out.float() - ref.float()).abs()
    return f"max_abs_err={diff.max().item():.3e}  mean_abs_err={diff.mean().item():.3e}"


def check(name: str, out, ref, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    """比较 out 与 ref。打印 PASS/FAIL，不抛异常（这样一次能看到所有用例的结果）。"""
    global _passes
    if out is None:
        print(f"  [FAIL] {name}: 返回了 None（还没实现？）")
        _failures.append(name)
        return False
    if not isinstance(out, torch.Tensor):
        out = torch.as_tensor(out)
    if not isinstance(ref, torch.Tensor):
        ref = torch.as_tensor(ref)
    if out.shape != ref.shape:
        print(f"  [FAIL] {name}: shape 不一致 out={tuple(out.shape)} ref={tuple(ref.shape)}")
        _failures.append(name)
        return False
    if out.dtype != ref.dtype:
        print(f"  [warn] {name}: dtype 不一致 out={out.dtype} ref={ref.dtype}（按 float32 比较）")
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    if ok and torch.isnan(out.float()).any() != torch.isnan(ref.float()).any():
        ok = False
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {_fmt_err(out, ref)}  (atol={atol}, rtol={rtol})")
    if ok:
        _passes += 1
    else:
        _failures.append(name)
    return ok


def check_equal(name: str, out, expected) -> bool:
    """精确相等（用于整数、字符串、bool、tuple 这类非 tensor 的答案）。"""
    global _passes
    ok = out == expected
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got={out!r}" + ("" if ok else f"  expected={expected!r}"))
    if ok:
        _passes += 1
    else:
        _failures.append(name)
    return ok


def assert_close(out, ref, atol: float = 1e-5, rtol: float = 1e-5, name: str = "assert_close"):
    """和 check 一样，但失败时直接抛 AssertionError。"""
    if not check(name, out, ref, atol, rtol):
        raise AssertionError(f"{name} 不匹配")


def bench(fn, warmup: int = 25, rep: int = 100) -> float:
    """返回 fn() 的耗时（毫秒，中位数）。

    用的是 triton.testing.do_bench：每次调用前清 L2，用 CUDA event 计时。
    注意：这台机器上其他卡/同一张卡上可能在跑训练，数字会有噪声，看相对快慢即可。
    """
    import triton.testing

    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def gbps(nbytes: float, ms: float) -> float:
    """有效带宽 GB/s。nbytes = 这次 kernel 最少需要读+写的字节数。"""
    return nbytes / (ms * 1e-3) / 1e9


def tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def report(rows: list[dict], title: str | None = None):
    """把一组 dict 打成对齐的表格。"""
    if not rows:
        return
    if title:
        print(f"\n== {title} ==")
    keys = list(rows[0].keys())
    cells = [[_cell(r.get(k)) for k in keys] for r in rows]
    widths = [max(len(k), *(len(c[i]) for c in cells)) for i, k in enumerate(keys)]
    print("  " + "  ".join(k.rjust(w) for k, w in zip(keys, widths)))
    for c in cells:
        print("  " + "  ".join(v.rjust(w) for v, w in zip(c, widths)))


def _cell(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1e4 else f"{v:.3e}"
    return str(v)


def finish():
    """打印汇总并设置退出码。放在练习文件 __main__ 的最后一行。"""
    total = _passes + len(_failures)
    if _failures:
        print(f"\n结果: {_passes}/{total} 通过。失败: {', '.join(_failures)}")
        sys.exit(1)
    print(f"\n结果: 全部 {total} 个检查通过 ✓")
