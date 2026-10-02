#!/usr/bin/env bash
# 从零重建本教程的环境（.venv / .cuda_home / ncu）。正常情况下已经建好了，不用跑。
# 用法: bash setup.sh
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/mnt/HD/.cache/uv UV_PYTHON_INSTALL_DIR=/mnt/HD/.uv-python

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
# CUDA 工具链全部钉在 13.0：要和 torch 2.11 自带的 cu130 runtime 头文件一致，
# 否则 nvcc 会报 "CUDA compiler and CUDA toolkit headers are incompatible"
uv pip install --python .venv/bin/python torch==2.11.0 ninja numpy pytest \
  "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*"

# pip 版 CUDA 没有标准 CUDA_HOME 目录结构（缺 lib64/libcudart.so），用软链拼一个
N="$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13"
mkdir -p .cuda_home/lib64 .cache
for d in bin include nvvm; do ln -sfn "$N/$d" ".cuda_home/$d"; done
for f in "$N"/lib/*; do ln -sfn "$f" ".cuda_home/lib64/$(basename "$f")"; done
ln -sfn "$N/lib/libcudart.so.13" .cuda_home/lib64/libcudart.so

# ncu：从 sglang 容器里拷（宿主机 apt 源里没有 nsight-compute）
if [ ! -x .tools/nsight-compute/ncu ]; then
  mkdir -p .tools
  docker cp glm53:/opt/nvidia/nsight-compute/2025.3.1 .tools/nsight-compute || echo "!! 拷 ncu 失败（容器 glm53 不在？），单元 02 的 ncu 部分不可用"
fi
# compute-sanitizer（越界 / race 检查，单元 05、06 用）
if [ ! -x .tools/compute-sanitizer/compute-sanitizer ]; then
  mkdir -p .tools
  docker cp glm53:/usr/local/cuda/compute-sanitizer .tools/compute-sanitizer || echo "!! 拷 compute-sanitizer 失败"
fi
echo "完成。执行: source env.sh && python check.py 01 --solutions"
