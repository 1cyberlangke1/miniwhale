"""融合 Sinkhorn（triton）—— 把 mHC 的 20 次串行归一融成 1 个 kernel。

背景：`mhc.sinkhorn_normalize` 是「softmax(-1)+eps → 列归一 → (行,列)×(repeat-1)」的
**串行迭代**；torch.compile 因归约的串行依赖无法融合，每次迭代切成独立 kernel。参考实现用
tilelang 融成 1 个（`hc_split_sinkhorn_kernel`）；本文件用 **triton** 独立实现同效果。

**维度参数化**：kernel 支持任意方阵 N×N（N 为 2 的幂，≤16，见 `can_fuse`）。归约顺序分两档：
  - **N == 4**：用 torch 小维度 `.sum` **实测的精确顺序**（行和 `(a0+a2)+(a1+a3)`、
    列和 `((a0+a1)+a2)+a3`）→ 与 torch **逐位一致**（已对拍 gdiff=0.00e+00）。
  - **N ≠ 4**：用 `tl.sum`（树形归约）→ **数学正确（式子对）**，但归约顺序与 torch 未必
    相同，**不保证逐位一致**。用于消融等非生产配置，只求式子没错、不崩。

**数值红线**：N == 4（生产唯一配置 `hc_mult=4`）与 torch eager **逐位一致**。关键（勿动）：
  1. 除法必须 `libdevice.div_rn`——triton 默认 `/` 是近似 `div.full.f32`，差 1 ULP。
  2. exp 必须 `libdevice.exp`——triton 默认 `tl.exp` 是近似 `__expf`。
  3. 反向用 torch autograd 公式：`g = g/den + Σ(gden)`（gden 已含负号）；
     softmax 反向 `y*gy - y*Σ(gy*y)`（**不是** `y*(gy-Σ(gy*y))`，二者差 1 ULP）。

以 `torch.library.custom_op` 注册（带 fake impl + autograd）——**避免 torch.compile
graph break**（裸 autograd.Function 会被 dynamo 切断编译图）。

triton / CUDA / shape 不满足时由调用方**回退** torch 实现（行为不变）。
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:  # 旧版 triton 的导入路径
        from triton.language.extra.cuda import libdevice

    TRITON_OK = True
except Exception:
    TRITON_OK = False

# 支持的方阵维度上限：一个 program 装 N*N 个 fp32，N>16 时寄存器压力过大。
_MAX_N = 16


def _is_pow2(v: int) -> bool:
    """v 是否为 2 的幂（triton `tl.arange` 要求长度为 2 的幂）。"""
    return v >= 1 and (v & (v - 1)) == 0


def can_fuse(x: Tensor) -> bool:
    """判断能否走融合 kernel。

    输入：x（sinkhorn 输入张量）。
    输出：bool。
    预期行为：仅当 triton 可用、x 在 CUDA、fp32、末两维为**方阵**、维度是 2 的幂且在
              [1, _MAX_N] 内、非空时返回 True；其余回退 torch 实现（行为不变）。
              注：非 2 的幂的维度无法用 `tl.arange` 表达，只能回退。
    """
    if not (TRITON_OK and x.is_cuda and x.dtype == torch.float32 and x.dim() >= 2):
        return False
    n = x.shape[-1]
    return (
        x.shape[-2] == n
        and _is_pow2(n)
        and n <= _MAX_N
        and x.numel() > 0
    )


if TRITON_OK:

    @triton.jit
    def _row_sum(x, N: tl.constexpr):
        """行和（axis=1）。

        输入：x [N,N]。
        输出：[N]。
        预期行为：N==4 用 torch 实测顺序 `(a0+a2)+(a1+a3)`（保逐位一致）；
                  其余 N 用 tl.sum 树形归约（式子对，顺序不保证同 torch）。
        """
        if N == 4:
            a0 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 0, x, 0.0), axis=1)
            a1 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 1, x, 0.0), axis=1)
            a2 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 2, x, 0.0), axis=1)
            a3 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 3, x, 0.0), axis=1)
            return (a0 + a2) + (a1 + a3)
        else:
            return tl.sum(x, axis=1)

    @triton.jit
    def _col_sum(x, N: tl.constexpr):
        """列和（axis=0）。

        输入：x [N,N]。
        输出：[N]。
        预期行为：N==4 用 torch 实测顺序 `((a0+a1)+a2)+a3`（保逐位一致）；
                  其余 N 用 tl.sum 树形归约（式子对，顺序不保证同 torch）。
        """
        if N == 4:
            a0 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 0, x, 0.0), axis=0)
            a1 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 1, x, 0.0), axis=0)
            a2 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 2, x, 0.0), axis=0)
            a3 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 3, x, 0.0), axis=0)
            return ((a0 + a1) + a2) + a3
        else:
            return tl.sum(x, axis=0)

    def _ns(iters: int) -> int:
        """S 每矩阵的槽数 = 4 + 2*iters（前向存的中间量个数）。"""
        return 4 + 2 * iters

    @triton.jit
    def _sink_fwd(X, Y, S, K: tl.constexpr, EPS: tl.constexpr, NS: tl.constexpr, N: tl.constexpr):
        """前向：softmax(-1)+eps → 列归一 → (行,列)×K；中间量写 S 供反向。

        输入：X [M*N*N] 展平输入；K=repeat-1；NS=每矩阵槽数；N=方阵维度。
        输出：Y [M*N*N]；S [M*NS*N*N]。
        预期行为：与 torch 版逐位一致（N==4；div_rn / libdevice.exp / 归约顺序见模块 docstring）。
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, N)[:, None] * N + tl.arange(0, N)[None, :]
        nn = N * N
        x = tl.load(X + pid * nn + offs)
        sbase = pid * NS * nn
        mx = tl.max(x, axis=1)
        e = libdevice.exp(x - mx[:, None])
        tl.store(S + sbase + 0 * nn + offs, e)
        y1 = libdevice.div_rn(e, _row_sum(e, N)[:, None])
        tl.store(S + sbase + 1 * nn + offs, y1)
        a = y1 + EPS
        tl.store(S + sbase + 2 * nn + offs, a)
        cur = libdevice.div_rn(a, _col_sum(a, N)[None, :] + EPS)
        tl.store(S + sbase + 3 * nn + offs, cur)
        idx = 4
        for _ in tl.static_range(K):
            cur = libdevice.div_rn(cur, _row_sum(cur, N)[:, None] + EPS)
            tl.store(S + sbase + idx * nn + offs, cur)
            idx += 1
            cur = libdevice.div_rn(cur, _col_sum(cur, N)[None, :] + EPS)
            tl.store(S + sbase + idx * nn + offs, cur)
            idx += 1
        tl.store(Y + pid * nn + offs, cur)

    @triton.jit
    def _sink_bwd_kernel(GY, S, GX, K: tl.constexpr, EPS: tl.constexpr, NS: tl.constexpr, N: tl.constexpr):
        """反向：完整逆序链 + softmax 反向。

        输入：GY [M*N*N] 输出梯度；S 前向中间量；K=repeat-1；NS=每矩阵槽数；N=方阵维度。
        输出：GX [M*N*N]。
        预期行为：g = g/den + Σ(gden)（gden=-g·num/den²）；softmax 反向 y·gy - y·Σ(gy·y)。
                  反向第 i 步的 num 取**输入** S[idx-1]，不是输出 S[idx]。
        """
        pid = tl.program_id(0)
        offs = tl.arange(0, N)[:, None] * N + tl.arange(0, N)[None, :]
        nn = N * N
        sbase = pid * NS * nn
        g = tl.load(GY + pid * nn + offs)
        idx = 3 + 2 * K
        for _ in tl.static_range(K):
            num = tl.load(S + sbase + (idx - 1) * nn + offs)
            den = _col_sum(num, N)[None, :] + EPS
            gden = -g * libdevice.div_rn(libdevice.div_rn(num, den), den)
            g = libdevice.div_rn(g, den) + _col_sum(gden, N)[None, :]
            idx -= 1
            num = tl.load(S + sbase + (idx - 1) * nn + offs)
            den = _row_sum(num, N)[:, None] + EPS
            gden = -g * libdevice.div_rn(libdevice.div_rn(num, den), den)
            g = libdevice.div_rn(g, den) + _row_sum(gden, N)[:, None]
            idx -= 1
        a = tl.load(S + sbase + 2 * nn + offs)
        den = _col_sum(a, N)[None, :] + EPS
        gden = -g * libdevice.div_rn(libdevice.div_rn(a, den), den)
        g = libdevice.div_rn(g, den) + _col_sum(gden, N)[None, :]
        y1 = tl.load(S + sbase + 1 * nn + offs)
        g = y1 * g - y1 * _row_sum(g * y1, N)[:, None]
        tl.store(GX + pid * nn + offs, g)

    @torch.library.custom_op("miniwhale::sinkhorn_fused", mutates_args=())
    def sinkhorn_fused(x: Tensor, iters: int, eps: float, n: int) -> tuple[Tensor, Tensor]:
        """融合 sinkhorn 前向（custom_op：黑盒注册，避免 compile graph break）。

        输入：x [...,N,N] fp32 CUDA；iters=repeat-1；eps；n=N（方阵维度）。
        输出：(y 同 x 形状, S 中间量 [M,NS,N*N])——S 仅供反向用。
        预期行为：跑 triton 前向 kernel；S 缓存每步中间量。
        """
        nn = n * n
        m = x.numel() // nn
        ns = _ns(iters)
        xf = x.reshape(m, n, n).contiguous()
        y = torch.empty_like(xf)
        S = torch.empty(m, ns, nn, device=x.device, dtype=torch.float32)
        _sink_fwd[(m,)](xf, y, S, iters, eps, ns, n)
        return y.reshape(x.shape), S

    @sinkhorn_fused.register_fake
    def _(x: Tensor, iters: int, eps: float, n: int) -> tuple[Tensor, Tensor]:
        """fake tensor 传播（compile 追踪用）。"""
        return torch.empty_like(x), x.new_empty((x.numel() // (n * n), _ns(iters), n * n))

    def _sink_setup(ctx, inputs, output):
        """保存反向所需上下文：iters/eps/n/形状 + 中间量 S。"""
        x, iters, eps, n = inputs
        ctx.iters, ctx.eps, ctx.n, ctx.shape = iters, eps, n, x.shape
        ctx.save_for_backward(output[1])

    def _sink_backward(ctx, gy, gS):
        """反向：跑 triton 逆链 kernel；返回 (dx, None, None, None)。"""
        (S,) = ctx.saved_tensors
        n = ctx.n
        m = gy.numel() // (n * n)
        gx = torch.empty(m, n, n, device=gy.device, dtype=torch.float32)
        _sink_bwd_kernel[(m,)](
            gy.reshape(m, n, n).contiguous(), S, gx, ctx.iters, ctx.eps, _ns(ctx.iters), n
        )
        return gx.reshape(ctx.shape), None, None, None

    torch.library.register_autograd(
        "miniwhale::sinkhorn_fused", _sink_backward, setup_context=_sink_setup
    )


def triton_sinkhorn(x: Tensor, repeat: int = 20, eps: float = 1e-6) -> Tensor:
    """融合 sinkhorn（triton 实现）。仅当 `can_fuse(x)` 为真时调用。

    输入：x [...,N,N] fp32 CUDA（N 为 2 的幂、≤16）；repeat（迭代次数）；eps。
    输出：同 x 形状。
    预期行为：返回融合前向结果（N==4 与 torch eager 逐位一致；其余 N 式子对）；
              反向由注册的 autograd 处理。
    """
    y, _ = sinkhorn_fused(x, repeat - 1, eps, x.shape[-1])
    return y
