# 用法: source env.sh   （每开一个新终端执行一次）
_KT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$_KT_ROOT/.venv/bin/activate"
export KT_ROOT="$_KT_ROOT"
export CUDA_HOME="$_KT_ROOT/.cuda_home"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# 编译缓存放在仓库里（.gitignore 已忽略）
export TORCH_EXTENSIONS_DIR="$_KT_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$_KT_ROOT/.cache/triton"
# load_inline 只为本机的 GPU 架构编译（H100 = 9.0，B200 = 10.0 ...），换卡不用改
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | paste -sd';')
fi
export PYTHONPATH="$_KT_ROOT:${PYTHONPATH:-}"
_KT_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "[kernel-tutorial] python=$(which python)  GPU=$_KT_GPU (sm ${TORCH_CUDA_ARCH_LIST})  CUDA_HOME=$CUDA_HOME"
# 多人共享的机器上用 gpu-run 拿锁再跑（它会设置 CUDA_VISIBLE_DEVICES），不要自己 export
if command -v gpu-run >/dev/null; then
  echo "[kernel-tutorial] 共享机器：先 gpu-status 挑空卡，再 gpu-run <id> -- python check.py 01"
fi
unset _KT_GPU
