#!/usr/bin/env bash
# 从零重建本教程的环境（.venv / .cuda_home / ncu / compute-sanitizer）。换机器后跑一次即可。
# 用法: bash setup.sh
#
# 不依赖具体机器：
#   - 没有 uv 就先在 .tools/ 下建个小 venv 装一个
#   - uv 的缓存默认用 uv 自己的位置；根分区紧张时可以先 export UV_CACHE_DIR=/大盘/xxx
#   - ncu / compute-sanitizer 优先用系统里已有的；没有就从任意一个带 CUDA 的 docker 容器里拷
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
  # 装进仓库里的一个小 venv：系统 python 常常是 externally-managed / 只读，pip --user 不一定能用
  echo ">> 安装 uv 到 .tools/uv-venv"
  [ -x .tools/uv-venv/bin/uv ] || { python3 -m venv .tools/uv-venv && .tools/uv-venv/bin/pip install -q uv; }
  export PATH="$PWD/.tools/uv-venv/bin:$PATH"
fi

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
# CUDA 工具链全部钉在 13.0：要和 torch 2.11 自带的 cu130 runtime 头文件一致，
# 否则 nvcc 会报 "CUDA compiler and CUDA toolkit headers are incompatible"。
# cu130 的 torch / triton / nvcc 同时支持 sm_80 / sm_90 / sm_100 / sm_120（A100 / H100 / B200 / RTX 50）。
uv pip install --python .venv/bin/python torch==2.11.0 ninja numpy pytest \
  "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*"
# 单元 10 的 CuTe DSL（可选，装不上不影响其他单元）
uv pip install --python .venv/bin/python nvidia-cutlass-dsl || echo "!! nvidia-cutlass-dsl 安装失败，单元 10 ex4 不可用"

# pip 版 CUDA 没有标准 CUDA_HOME 目录结构（缺 lib64/libcudart.so），用软链拼一个
N="$PWD/.venv/lib/python3.11/site-packages/nvidia/cu13"
mkdir -p .cuda_home/lib64 .cache
for d in bin include nvvm; do ln -sfn "$N/$d" ".cuda_home/$d"; done
for f in "$N"/lib/*; do ln -sfn "$f" ".cuda_home/lib64/$(basename "$f")"; done
ln -sfn "$N/lib/libcudart.so.13" .cuda_home/lib64/libcudart.so

# 找一个装了 CUDA 13 工具的运行中容器，用来拷 ncu / compute-sanitizer（宿主机 apt 源里通常没有）
find_container() {  # $1 = 容器里要存在的路径
  command -v docker >/dev/null || return 1
  for c in $(docker ps --format '{{.Names}}' 2>/dev/null); do
    if docker exec "$c" test -e "$1" 2>/dev/null; then echo "$c"; return 0; fi
  done
  return 1
}

mkdir -p .tools
if [ ! -x .tools/nsight-compute/ncu ]; then
  if command -v ncu >/dev/null; then
    ln -sfn "$(dirname "$(readlink -f "$(command -v ncu)")")" .tools/nsight-compute
  elif c=$(find_container /opt/nvidia/nsight-compute); then
    v=$(docker exec "$c" ls /opt/nvidia/nsight-compute | sort -V | tail -1)
    echo ">> 从容器 $c 拷 ncu $v"
    docker cp "$c:/opt/nvidia/nsight-compute/$v" .tools/nsight-compute
  else
    echo "!! 没找到 ncu，单元 02 的 ncu 部分不可用"
  fi
fi
if [ ! -x .tools/compute-sanitizer/compute-sanitizer ]; then
  if command -v compute-sanitizer >/dev/null; then
    ln -sfn "$(dirname "$(readlink -f "$(command -v compute-sanitizer)")")" .tools/compute-sanitizer
  elif c=$(find_container /usr/local/cuda/compute-sanitizer); then
    echo ">> 从容器 $c 拷 compute-sanitizer"
    docker cp -L "$c:/usr/local/cuda/compute-sanitizer" .tools/compute-sanitizer
  else
    echo "!! 没找到 compute-sanitizer，单元 05、06 的越界检查部分不可用"
  fi
fi
echo "完成。执行: source env.sh && python check.py 01 --solutions"
