"""单元 10 的几个参考 kernel，供示例和练习 1（PTX 分析）共用。

  matmul_ptr   : 单元 04 的指针版 GEMM（Ampere 风格的 cp.async 预取 + wgmma）
  matmul_tma   : 用 host 端 TensorDescriptor 的 GEMM（TMA 搬运 + wgmma）
  fp8_matmul   : FP8 e4m3 GEMM，B 按 [N, K] 存，带 per-row / per-col 缩放
"""
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _tile_coords(tile_id, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (tile_id // num_pid_in_group) * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (tile_id % num_pid_in_group) % group_size_m
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_ptr_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                      stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr):
    pid_m, pid_n = _tile_coords(tl.program_id(0), tl.cdiv(M, BM), tl.cdiv(N, BN), 8)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_rem = K - k * BK
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_rem), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_rem) & (offs_n[None, :] < N), other=0.0)
        if IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def matmul_ptr(a, b, BM=128, BN=128, BK=64, num_warps=8, num_stages=3, ieee=False):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16 if a.dtype != torch.float32 else torch.float32)
    return matmul_ptr_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BM=BM, BN=BN, BK=BK, IEEE=ieee, num_warps=num_warps, num_stages=num_stages), c


@triton.jit
def matmul_tma_kernel(a_desc, b_desc, c_desc, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = _tile_coords(tl.program_id(0), tl.cdiv(M, BM), tl.cdiv(N, BN), 8)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        acc = tl.dot(a_desc.load([pid_m * BM, ki * BK]), b_desc.load([ki * BK, pid_n * BN]), acc)
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_desc.dtype))


def matmul_tma(a, b, BM=128, BN=128, BK=64, num_warps=8, num_stages=4):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    a_d = TensorDescriptor.from_tensor(a, [BM, BK])
    b_d = TensorDescriptor.from_tensor(b, [BK, BN])
    c_d = TensorDescriptor.from_tensor(c, [BM, BN])
    return matmul_tma_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a_d, b_d, c_d, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages), c


@triton.jit
def fp8_matmul_kernel(a_desc, b_desc, c_desc, sa_ptr, sb_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = _tile_coords(tl.program_id(0), tl.cdiv(M, BM), tl.cdiv(N, BN), 8)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        a = a_desc.load([pid_m * BM, ki * BK])          # [BM, BK]
        b = b_desc.load([pid_n * BN, ki * BK])          # [BN, BK]（B 按 [N, K] 存，K 连续）
        acc = tl.dot(a, b.T, acc)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    sa = tl.load(sa_ptr + offs_m, mask=offs_m < M, other=0.0)
    sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc * sa[:, None] * sb[None, :]
    c_desc.store([pid_m * BM, pid_n * BN], acc.to(c_desc.dtype))


def fp8_matmul(a8, sa, b8_nk, sb, BM=128, BN=256, BK=128, num_warps=8, num_stages=3):
    M, K = a8.shape
    N = b8_nk.shape[0]
    c = torch.empty((M, N), device=a8.device, dtype=torch.bfloat16)
    a_d = TensorDescriptor.from_tensor(a8, [BM, BK])
    b_d = TensorDescriptor.from_tensor(b8_nk, [BN, BK])
    c_d = TensorDescriptor.from_tensor(c, [BM, BN])
    return fp8_matmul_kernel[(triton.cdiv(M, BM) * triton.cdiv(N, BN),)](
        a_d, b_d, c_d, sa, sb, M, N, K, BM=BM, BN=BN, BK=BK, num_warps=num_warps, num_stages=num_stages), c


def quantize_rowwise(x: torch.Tensor):
    """每行一个 scale：x ≈ x8.float() * scale[:, None]。e4m3 能表示的最大值是 448。"""
    scale = x.abs().amax(dim=1).float().clamp(min=1e-12) / 448.0
    return (x.float() / scale[:, None]).to(torch.float8_e4m3fn), scale
