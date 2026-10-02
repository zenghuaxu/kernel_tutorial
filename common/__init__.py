"""教程公用工具：正确性检查、计时、CUDA 内联编译。

练习文件里一般这样用：
    from common import check, bench, load_cuda, report
"""
from .testing import check, check_equal, assert_close, bench, report, gbps, tflops, finish
from .cuda_ext import load_cuda

__all__ = ["check", "check_equal", "assert_close", "bench", "report", "gbps", "tflops", "finish", "load_cuda"]
