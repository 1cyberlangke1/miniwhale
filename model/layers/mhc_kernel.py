"""融合 mHC 前向（triton）—— split_mixes(pre/post/comb) + sinkhorn + hc_pre 融成 1 个 kernel。

动机（对齐参考实现 TileKernels `mhc_pre_big_fuse`）：mHC 是每层几十个 elementwise/reduction 小
kernel（launch-bound，CPU 发射跟不上 GPU）。参考实现用一个 kernel 干完「split_mixes + sinkhorn +
hc_pre」。本文件用 **triton** 独立实现同效果（非抄参考实现 tilelang）。

对应 `mhc.py` 的 `split_mixes` + `sinkhorn_normalize` + `hc_pre` 三者的**层内**计算：
    mixes → pre/post/comb（含 sinkhorn 归一）
    h = Σ_h pre_mix[h]·x[h,:]   （hc_pre，collapse）
注：`Block.forward` 里 `hc_mixes(x,·)` 与 `hc_pre(x, pre_mix)` 用**同一个 x**，故可融成一次调用。

**数值**：与参考实现（TileKernels/tile_kernels/torch/mhc.py）对拍 ~1e-7（fp32 舍入量级）。
关键（勿动）：除法 `libdevice.div_rn`、exp `libdevice.exp`、N==4 用 torch 实测归约顺序。

以 `torch.library.custom_op` 注册（带 fake + autograd）——避免 torch.compile graph break。
triton/CUDA/shape 不满足时由调用方**回退** torch 实现（行为不变）。
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:
        from triton.language.extra.cuda import libdevice

    TRITON_OK = True
except Exception:
    TRITON_OK = False

_MAX_H = 4096         # hidden_size 上限（一个 program 装 4*H 个 fp32）


def _ns(iters: int) -> int:
    """S 每矩阵的槽数 = 4 + 2*iters（前向存的 sinkhorn 中间量个数）。"""
    return 4 + 2 * iters


def can_fuse(x: Tensor, hc_mult: int, hidden: int) -> bool:
    """判断能否走融合 kernel。

    输入：x（残差流 [...,hc_mult,hidden]）；hc_mult；hidden。
    输出：bool。
    预期行为：仅当 triton 可用、x 在 CUDA、hc_mult==4、hidden 是 2 的幂且在 [1,_MAX_H]
              内时返回 True；其余回退 torch（行为不变）。
    """
    return (
        TRITON_OK
        and x.is_cuda
        and hc_mult == 4
        and 1 <= hidden <= _MAX_H
        and (hidden & (hidden - 1)) == 0
        and x.numel() > 0
    )


if TRITON_OK:

    @triton.jit
    def _rs(x):
        """行和（4 元素），顺序对齐 torch 小维度 .sum。"""
        a0 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 0, x, 0.0), axis=1)
        a1 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 1, x, 0.0), axis=1)
        a2 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 2, x, 0.0), axis=1)
        a3 = tl.sum(tl.where(tl.arange(0, 4)[None, :] == 3, x, 0.0), axis=1)
        return (a0 + a2) + (a1 + a3)

    @triton.jit
    def _cs(x):
        """列和（4 元素）。"""
        a0 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 0, x, 0.0), axis=0)
        a1 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 1, x, 0.0), axis=0)
        a2 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 2, x, 0.0), axis=0)
        a3 = tl.sum(tl.where(tl.arange(0, 4)[:, None] == 3, x, 0.0), axis=0)
        return ((a0 + a1) + a2) + a3

    @triton.jit
    def _mhc_fwd(MIXES, SCALE, BASE, RES, PREIN, PRE, POST, COMB, OUT, S,
                 K: tl.constexpr, EPS: tl.constexpr, NS: tl.constexpr, HH: tl.constexpr):
        """一个 program 处理一个 token：split_mixes + sinkhorn + hc_pre。

        输入：MIXES [N*24]；SCALE [3]；BASE [24]；RES [N*4*HH]；PREIN [N*4]；K=repeat-1。
        输出：PRE [N*4]；POST [N*4]；COMB [N*16]；OUT [N*HH]；S [N*NS*16]。
        预期行为：pre=sigmoid(m[:4]*s0+b)+eps；post=2*sigmoid(m[4:8]*s1+b)；
                  comb=Sinkhorn(m[8:24]*s2+b)；out[d]=Σ_h prein[h]·res[h,d]。
        """
        pid = tl.program_id(0)
        r = tl.arange(0, 4)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        s2 = tl.load(SCALE + 2)
        mp = tl.load(MIXES + pid * 24 + r) * s0 + tl.load(BASE + r)
        tl.store(PRE + pid * 4 + r, tl.sigmoid(mp) + EPS)
        mo = tl.load(MIXES + pid * 24 + 4 + r) * s1 + tl.load(BASE + 4 + r)
        tl.store(POST + pid * 4 + r, 2.0 * tl.sigmoid(mo))
        offs = r[:, None] * 4 + r[None, :]
        x = tl.load(MIXES + pid * 24 + 8 + offs) * s2 + tl.load(BASE + 8 + offs)
        sbase = pid * NS * 16
        mx = tl.max(x, axis=1)
        e = libdevice.exp(x - mx[:, None])
        tl.store(S + sbase + 0 * 16 + offs, e)
        y1 = libdevice.div_rn(e, _rs(e)[:, None])
        tl.store(S + sbase + 1 * 16 + offs, y1)
        a = y1 + EPS
        tl.store(S + sbase + 2 * 16 + offs, a)
        cur = libdevice.div_rn(a, _cs(a)[None, :] + EPS)
        tl.store(S + sbase + 3 * 16 + offs, cur)
        idx = 4
        for _ in tl.static_range(K):
            cur = libdevice.div_rn(cur, _rs(cur)[:, None] + EPS)
            tl.store(S + sbase + idx * 16 + offs, cur)
            idx += 1
            cur = libdevice.div_rn(cur, _cs(cur)[None, :] + EPS)
            tl.store(S + sbase + idx * 16 + offs, cur)
            idx += 1
        tl.store(COMB + pid * 16 + offs, cur)
        d = tl.arange(0, HH)
        pin = tl.load(PREIN + pid * 4 + r)
        res = tl.load(RES + pid * 4 * HH + r[:, None] * HH + d[None, :])
        tl.store(OUT + pid * HH + d, tl.sum(pin[:, None] * res, axis=0))

    @triton.jit
    def _mhc_bwd(GOUT, DPRE, DPOST, DCOMB, MIXES, SCALE, BASE, RES, PREIN, S,
                 DM, DRES, DPREIN, K: tl.constexpr, EPS: tl.constexpr, NS: tl.constexpr,
                 HH: tl.constexpr):
        """反向：hc_pre 反向 + sinkhorn 反向 + sigmoid 反向；输出 dm=∂L/∂m（缩放前）。

        输入：GOUT [N*HH]；DPRE/DPOST [N*4]；DCOMB [N*16]；MIXES/SCALE/BASE/RES/PREIN；S。
        输出：DM [N*24]（缩放前）；DRES [N*4*HH]；DPREIN [N*4]。
        预期行为：dprein[h]=Σ_d gout·res；dres=prein·gout；dm 分段=σ'·dpre/dpost/comb_bwd。
        """
        pid = tl.program_id(0)
        r = tl.arange(0, 4)
        s0 = tl.load(SCALE + 0)
        s1 = tl.load(SCALE + 1)
        d = tl.arange(0, HH)
        gout = tl.load(GOUT + pid * HH + d)
        res = tl.load(RES + pid * 4 * HH + r[:, None] * HH + d[None, :])
        pin = tl.load(PREIN + pid * 4 + r)
        tl.store(DPREIN + pid * 4 + r, tl.sum(gout[None, :] * res, axis=1))
        tl.store(DRES + pid * 4 * HH + r[:, None] * HH + d[None, :], pin[:, None] * gout[None, :])
        offs = r[:, None] * 4 + r[None, :]
        sbase = pid * NS * 16
        g = tl.load(DCOMB + pid * 16 + offs)
        idx = 3 + 2 * K
        for _ in tl.static_range(K):
            num = tl.load(S + sbase + (idx - 1) * 16 + offs)
            den = _cs(num)[None, :] + EPS
            gden = -g * libdevice.div_rn(libdevice.div_rn(num, den), den)
            g = libdevice.div_rn(g, den) + _cs(gden)[None, :]
            idx -= 1
            num = tl.load(S + sbase + (idx - 1) * 16 + offs)
            den = _rs(num)[:, None] + EPS
            gden = -g * libdevice.div_rn(libdevice.div_rn(num, den), den)
            g = libdevice.div_rn(g, den) + _rs(gden)[:, None]
            idx -= 1
        a = tl.load(S + sbase + 2 * 16 + offs)
        den = _cs(a)[None, :] + EPS
        gden = -g * libdevice.div_rn(libdevice.div_rn(a, den), den)
        g = libdevice.div_rn(g, den) + _cs(gden)[None, :]
        y1 = tl.load(S + sbase + 1 * 16 + offs)
        g = y1 * g - y1 * _rs(g * y1)[:, None]
        sig_p = tl.sigmoid(tl.load(MIXES + pid * 24 + r) * s0 + tl.load(BASE + r))
        tl.store(DM + pid * 24 + r, tl.load(DPRE + pid * 4 + r) * sig_p * (1.0 - sig_p))
        sig_o = tl.sigmoid(tl.load(MIXES + pid * 24 + 4 + r) * s1 + tl.load(BASE + 4 + r))
        tl.store(DM + pid * 24 + 4 + r, tl.load(DPOST + pid * 4 + r) * 2.0 * sig_o * (1.0 - sig_o))
        tl.store(DM + pid * 24 + 8 + offs, g)

    @torch.library.custom_op("miniwhale::mhc_fused", mutates_args=())
    def mhc_fused(mixes: Tensor, hc_scale: Tensor, hc_base: Tensor, residual: Tensor,
                  pre_mix: Tensor, iters: int, eps: float) -> tuple[Tensor, Tensor, Tensor,
                                                                     Tensor, Tensor]:
        """融合 mHC 前向（custom_op：黑盒注册，避免 compile graph break）。

        输入：mixes [B,L,24] fp32；hc_scale [3]；hc_base [24]；residual [B,L,4,H] fp32；
              pre_mix [B,L,4] fp32；iters=repeat；eps。
        输出：(pre [B,L,4], post [B,L,4], comb [B,L,4,4], out [B,L,H], S 中间量)。
        预期行为：跑 triton 前向 kernel；S 缓存每步 sinkhorn 中间量供反向。
        """
        B, L, _ = mixes.shape
        H = residual.shape[-1]
        n = B * L
        ns = _ns(iters - 1)
        dev = mixes.device
        pre = torch.empty(n, 4, device=dev, dtype=torch.float32)
        post = torch.empty(n, 4, device=dev, dtype=torch.float32)
        comb = torch.empty(n, 16, device=dev, dtype=torch.float32)
        out = torch.empty(n, H, device=dev, dtype=torch.float32)
        S = torch.empty(n, ns, 16, device=dev, dtype=torch.float32)
        _mhc_fwd[(n,)](mixes.reshape(n, 24).contiguous(), hc_scale, hc_base,
                       residual.reshape(n, 4 * H).contiguous(), pre_mix.reshape(n, 4).contiguous(),
                       pre, post, comb, out, S, iters - 1, eps, ns, H)
        return (pre.reshape(B, L, 4), post.reshape(B, L, 4), comb.reshape(B, L, 4, 4),
                out.reshape(B, L, H), S)

    @mhc_fused.register_fake
    def _(mixes: Tensor, hc_scale: Tensor, hc_base: Tensor, residual: Tensor,
          pre_mix: Tensor, iters: int, eps: float):
        """fake tensor 传播（compile 追踪用）。"""
        B, L, _ = mixes.shape
        H = residual.shape[-1]
        n = B * L
        return (mixes.new_empty((B, L, 4)), mixes.new_empty((B, L, 4)),
                mixes.new_empty((B, L, 4, 4)), mixes.new_empty((B, L, H)),
                mixes.new_empty((n, _ns(iters - 1), 16)))

    @torch.library.custom_op("miniwhale::mhc_bwd", mutates_args=())
    def mhc_bwd(gout: Tensor, gpre: Tensor, gpost: Tensor, gcomb: Tensor, mixes: Tensor,
                hc_scale: Tensor, hc_base: Tensor, residual: Tensor, pre_mix: Tensor, S: Tensor,
                iters: int, eps: float) -> tuple[Tensor, Tensor, Tensor]:
        """融合 mHC 反向 kernel（custom_op：黑盒，避免 compile 追踪进 triton kernel）。

        输入：gout [B,L,H]；gpre/gpost [B,L,4]；gcomb [B,L,4,4]；mixes [B,L,24]；hc_scale [3]；
              hc_base [24]；residual [B,L,4,H]；pre_mix [B,L,4]；S；iters；eps。
        输出：(dm [B,L,24] 缩放前, dres [B,L,4,H], dprein [B,L,4])。
        预期行为：跑 triton 反向 kernel；scale/base 梯度由调用方在 torch 里从 dm 补。
        """
        B, L, _ = mixes.shape
        H = residual.shape[-1]
        n = B * L
        dev = mixes.device
        dm = torch.empty(n, 24, device=dev, dtype=torch.float32)
        dres = torch.empty(n, 4 * H, device=dev, dtype=torch.float32)
        dprein = torch.empty(n, 4, device=dev, dtype=torch.float32)
        _mhc_bwd[(n,)](gout.reshape(n, H).contiguous(), gpre.reshape(n, 4).contiguous(),
                       gpost.reshape(n, 4).contiguous(), gcomb.reshape(n, 16).contiguous(),
                       mixes.reshape(n, 24).contiguous(), hc_scale, hc_base,
                       residual.reshape(n, 4 * H).contiguous(), pre_mix.reshape(n, 4).contiguous(),
                       S, dm, dres, dprein, iters - 1, eps, _ns(iters - 1), H)
        return dm.reshape(B, L, 24), dres.reshape(B, L, 4, H), dprein.reshape(B, L, 4)

    @mhc_bwd.register_fake
    def _(gout: Tensor, gpre: Tensor, gpost: Tensor, gcomb: Tensor, mixes: Tensor,
          hc_scale: Tensor, hc_base: Tensor, residual: Tensor, pre_mix: Tensor, S: Tensor,
          iters: int, eps: float):
        """fake tensor 传播（compile 追踪用）。"""
        B, L, _ = mixes.shape
        H = residual.shape[-1]
        return (mixes.new_empty((B, L, 24)), mixes.new_empty((B, L, 4, H)),
                mixes.new_empty((B, L, 4)))

    def _mhc_setup(ctx, inputs, output):
        """保存反向上下文：iters/eps + 全部输入 + S（都是张量，save_for_backward 直接存）。"""
        mixes, hc_scale, hc_base, residual, pre_mix, iters, eps = inputs
        ctx.iters, ctx.eps = iters, eps
        ctx.save_for_backward(mixes, hc_scale, hc_base, residual, pre_mix, output[4])

    def _mhc_backward(ctx, gpre, gpost, gcomb, gout, gS):
        """反向：调 mhc_bwd custom_op（黑盒出 dm），torch 补 scale/base 梯度。

        返回：(dmixes, dscale, dbase, dres, dprein, None, None)。
        """
        mixes, hc_scale, hc_base, residual, pre_mix, S = ctx.saved_tensors
        dm, dres, dprein = mhc_bwd(gout, gpre, gpost, gcomb, mixes, hc_scale, hc_base,
                                   residual, pre_mix, S, ctx.iters, ctx.eps)
        # scale/base 梯度在 torch 里从 dm 补（确定、无 atomic）：dscale=Σ dm·mixes、dbase=Σ dm
        n = mixes.shape[0] * mixes.shape[1]
        mf = mixes.reshape(n, 24)
        dmf = dm.reshape(n, 24)
        dscale = torch.stack([(dmf[:, :4] * mf[:, :4]).sum(),
                              (dmf[:, 4:8] * mf[:, 4:8]).sum(),
                              (dmf[:, 8:24] * mf[:, 8:24]).sum()])
        dbase = dmf.sum(0)
        scale = torch.cat([hc_scale[0].expand(4), hc_scale[1].expand(4), hc_scale[2].expand(16)])
        return (dm * scale).reshape(mixes.shape), dscale, dbase, dres, dprein, None, None

    torch.library.register_autograd(
        "miniwhale::mhc_fused", _mhc_backward, setup_context=_mhc_setup
    )


def mhc_fused_call(mixes: Tensor, hc_scale: Tensor, hc_base: Tensor, residual: Tensor,
                   pre_mix: Tensor, repeat: int = 20, eps: float = 1e-6):
    """融合 mHC：split_mixes + sinkhorn + hc_pre。仅当 `can_fuse` 为真时调用。

    输入：mixes [B,L,24]；hc_scale [3]；hc_base [24]；residual [B,L,4,H]；pre_mix [B,L,4]。
    输出：(pre [B,L,4], post [B,L,4], comb [B,L,4,4], h [B,L,H])。
    """
    pre, post, comb, out, _S = mhc_fused(mixes, hc_scale, hc_base, residual, pre_mix, repeat, eps)
    return pre, post, comb, out
