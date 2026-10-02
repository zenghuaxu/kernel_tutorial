#!/usr/bin/env bash
# Nsight Compute 包装脚本。
#
#   tools/ncu.sh -k regex:my_kernel python 02_profiling/examples/copy_kernels.py
#   tools/ncu.sh --set full -k regex:matmul -c 1 -o out/matmul python xxx.py   # 存成 .ncu-rep
#   tools/ncu.sh -i out/matmul.ncu-rep --page details                         # 在终端里看报告
#
# 为什么要包一层：
#   1. 这台机器 RmProfilingAdminOnly=1，读性能计数器必须 root -> 用 sudo 并保留环境变量
#   2. 默认 ncu 会把 GPU 锁到 base clock（--clock-control base）。卡上有别人的训练任务，
#      锁频会拖慢它们，所以这里默认 --clock-control none。代价：数字会随频率浮动。
#   3. ncu 是从 sglang 容器里拷出来的（2025.3.1），不在系统 PATH 里。
set -euo pipefail
if [ -z "${KT_ROOT:-}" ]; then
  echo "先执行: source env.sh" >&2; exit 2
fi
NCU="$KT_ROOT/.tools/nsight-compute/ncu"
# 读已保存的报告（-i xxx.ncu-rep）不需要 root，也不能带 --clock-control
for a in "$@"; do
  if [ "$a" = "-i" ] || [ "$a" = "--import" ]; then exec "$NCU" "$@"; fi
done
exec sudo -E env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" "PYTHONPATH=${PYTHONPATH:-}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" \
  "$NCU" --clock-control none "$@"
