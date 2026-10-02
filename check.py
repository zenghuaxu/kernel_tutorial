#!/usr/bin/env python
"""跑练习并汇总结果。

    python check.py                 # 所有单元的所有练习
    python check.py 01              # 只跑单元 01
    python check.py 01 ex2          # 只跑单元 01 里文件名以 ex2 开头的练习
    python check.py 03 --solutions  # 跑参考答案（用来确认环境没问题）

每个练习是一个独立脚本，退出码 0 = 通过。也可以直接 python 某个练习文件，看完整输出。
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_units(prefix: str | None) -> list[Path]:
    units = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name[:2].isdigit())
    if prefix:
        units = [u for u in units if u.name.startswith(prefix)]
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit", nargs="?", help="单元编号前缀，如 01")
    ap.add_argument("ex", nargs="?", help="练习文件名前缀，如 ex2")
    ap.add_argument("--solutions", action="store_true", help="跑 solutions/ 而不是 exercises/")
    ap.add_argument("-v", "--verbose", action="store_true", help="打印每个练习的完整输出")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    if "CUDA_HOME" not in os.environ:
        print("先执行: source env.sh")
        sys.exit(2)

    sub = "solutions" if args.solutions else "exercises"
    results = []
    for unit in find_units(args.unit):
        d = unit / sub
        if not d.is_dir():
            continue
        files = sorted(d.glob("ex*.py"))
        if args.ex:
            files = [f for f in files if f.name.startswith(args.ex)]
        for f in files:
            rel = f.relative_to(ROOT)
            print(f"▶ {rel} ... ", end="", flush=True)
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, str(f)], cwd=ROOT, capture_output=True, text=True,
                                   timeout=args.timeout)
                ok, out = p.returncode == 0, p.stdout + p.stderr
            except subprocess.TimeoutExpired:
                ok, out = False, f"超时（>{args.timeout}s）"
            dt = time.time() - t0
            print(("PASS" if ok else "FAIL") + f"  ({dt:.1f}s)")
            if args.verbose or not ok:
                tail = out.strip().splitlines()[-15:]
                print("    " + "\n    ".join(tail))
            results.append((str(rel), ok))

    if not results:
        print("没找到练习文件。")
        sys.exit(2)
    n_ok = sum(ok for _, ok in results)
    print(f"\n总计 {n_ok}/{len(results)} 通过")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
