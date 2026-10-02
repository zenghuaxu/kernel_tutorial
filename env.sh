# 用法: source env.sh   （每开一个新终端执行一次）
_KT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_KT_ROOT/.venv/bin/activate"
export KT_ROOT="$_KT_ROOT"
export CUDA_HOME="$_KT_ROOT/.cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# 编译缓存放大盘（根分区几乎满了）
export TORCH_EXTENSIONS_DIR="$_KT_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$_KT_ROOT/.cache/triton"
export TORCH_CUDA_ARCH_LIST="9.0"
export PYTHONPATH="$_KT_ROOT:${PYTHONPATH:-}"
# 机器上在跑训练：默认挑显存占用最少的卡；想固定就提前 export CUDA_VISIBLE_DEVICES=3
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
fi
echo "[kernel-tutorial] python=$(which python)  GPU=$CUDA_VISIBLE_DEVICES  CUDA_HOME=$CUDA_HOME"
