"""把一段 CUDA C++ 源码即时编译成 Python 模块（torch.utils.cpp_extension.load_inline 的薄封装）。

用法：
    mod = load_cuda("my_add", CUDA_SRC, ["add"])
    c = mod.add(a, b)

CUDA_SRC 里要同时包含 __global__ kernel 和一个接收/返回 torch::Tensor 的 host 函数；
functions 列出要暴露给 Python 的 host 函数名。我们自动从源码里抽出这些函数的声明，
所以不用自己写 cpp_sources。

第一次编译要 30~60 秒，之后命中缓存（$TORCH_EXTENSIONS_DIR）几乎瞬间完成。
源码一改，名字相同也会自动重新编译。
"""
import hashlib
import os
import re

from torch.utils.cpp_extension import load_inline

_PRELUDE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " 必须在 GPU 上")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " 必须是 contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
// 在 kernel 启动后调用，立刻暴露 launch 配置错误（比如 block 太大）
#define CUDA_CHECK_LAUNCH() C10_CUDA_KERNEL_LAUNCH_CHECK()
"""


def _extract_decls(src: str, functions: list[str]) -> str:
    decls = []
    for fn in functions:
        # 匹配 "返回类型 fn(参数...) {"，参数可以跨行
        m = re.search(r"([\w:<>,\s\*&]+?)\b" + re.escape(fn) + r"\s*\(([^)]*)\)\s*\{", src)
        if m is None:
            raise ValueError(f"在 CUDA 源码里找不到 host 函数 {fn}(...) {{ ... }} 的定义")
        ret = m.group(1).strip().split("\n")[-1].strip()
        decls.append(f"{ret} {fn}({m.group(2)});")
    return "\n".join(decls)


def load_cuda(name: str, cuda_src: str, functions: list[str], extra_cuda_cflags: list[str] | None = None,
              verbose: bool = False):
    src = _PRELUDE + cuda_src
    flags = ["-O3", "-lineinfo", "--expt-relaxed-constexpr", "-std=c++17"]
    if extra_cuda_cflags:
        flags += extra_cuda_cflags
    # 源码/参数变化 -> 名字变化 -> 不会误用旧的缓存
    digest = hashlib.md5((src + " ".join(flags)).encode()).hexdigest()[:8]
    if "CUDA_HOME" not in os.environ:
        raise RuntimeError("没找到 CUDA_HOME。先在仓库根目录执行: source env.sh")
    return load_inline(
        name=f"{name}_{digest}",
        cpp_sources=_extract_decls(cuda_src, functions),
        cuda_sources=src,
        functions=functions,
        extra_cuda_cflags=flags,
        verbose=verbose,
    )
