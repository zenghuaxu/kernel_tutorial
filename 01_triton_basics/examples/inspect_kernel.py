"""示例：一次 launch 之后，看看 Triton 编译出了什么。

运行：python 01_triton_basics/examples/inspect_kernel.py

Triton 的编译流水线：Python AST → Triton IR (ttir) → TritonGPU IR (ttgir，带布局信息)
→ LLVM IR → PTX → cubin (SASS)。launch 返回的对象上能拿到每一级。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * alpha, mask=mask)


if __name__ == "__main__":
    n = 1 << 20
    x = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    for block, warps in [(1024, 4), (4096, 8)]:
        k = scale_kernel[(triton.cdiv(n, block),)](x, out, n, 2.0, BLOCK=block, num_warps=warps)
        print(f"===== BLOCK={block} num_warps={warps} =====")
        print(f"寄存器/线程: {k.n_regs}   寄存器溢出(spill): {k.n_spills}   shared mem: {k.metadata.shared} B")
        ptx = k.asm["ptx"]
        # 每个线程负责 BLOCK / (32*num_warps) 个元素；看它用了多宽的向量 load
        loads = [l.strip() for l in ptx.splitlines() if "ld.global" in l]
        print(f"PTX 里的 global load 指令（共 {len(loads)} 条），前 4 条：")
        for l in loads[:4]:
            print("   ", l)
        print()
    print("看点：ld.global.v4.b32 = 一条指令读 4 个 fp32（16 字节）——向量化访存，带宽利用率高的前提。")
    print("想看全部：k.asm['ttgir'] / k.asm['ptx'] / k.asm['sass']（sass 需要 cuobjdump，可能不可用）")
