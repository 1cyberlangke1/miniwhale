"""QAT 假量化（fake quant）—— 逐行对按参考实现 kernel.py 的公式。

参考实现 `act_quant` / `fp4_act_quant` 是 tilelang kernel，做的是"量化→反量化"（inplace 写回）。
本文件用 torch 复现**同一套公式**，供训练时 QAT（前向量化-反量化、反向直通梯度）。

参考实现公式（参考模型的 act_quant / fp4_act_quant）：
- act_quant：amax 逐块、clamp_min(1e-4)；pow2 时 scale = 2^ceil(log2(amax/448))，否则 amax/448；
  clamp(x/s, -448, 448) → fp8(e4m3) → *s。
- fp4_act_quant：e4m3 scale 时 amax clamp_min(6*2^-9)、s=amax/6；
  e8m0 scale 时 amax clamp_min(6*2^-126)、s=2^ceil(log2(amax/6))；
  clamp(x/s, -6, 6) → fp4(e2m1) → *s。
"""

from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0

# e2m1 可表示的正值网格
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """参考实现 fast_round_scale：scale = 2^ceil(log2(amax * max_inv))。"""
    return torch.exp2(torch.ceil(torch.log2((amax * max_inv).clamp_min(1e-30))))


def _round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """舍入到最近 e2m1 网格值（**round-half-to-even**，对齐 CUDA `cudaRoundNearest`）。

    输入：x 任意浮点（调用方已 clamp 到 [-6, 6]）。
    输出：同形状，取值为 ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}。
    预期行为：恰在两相邻网格值**中点**时取偶数尾数那个——e2m1 里下标为偶数 ⟺ 尾数位为 0，
              即 round-half-to-even。参考实现 kernel `T.Cast(FP4, ...)` 用 CUDA `cudaRoundNearest`，
              若此处用「取较小者」会在中点翻错整格（例如 -1.75 应为 -2.0 而非 -1.5）。
    """
    vals = _E2M1.to(x.device)
    a = x.abs()
    # a 落在 vals[lo, hi] 之间；hi = 首个 >= a 的下标
    hi = torch.searchsorted(vals, a.contiguous()).clamp(1, vals.numel() - 1)
    lo = hi - 1
    vlo, vhi = vals[lo], vals[hi]
    dlo, dhi = a - vlo, vhi - a
    # 严格更近取上界；中点时取下标偶数（偶数尾数）那个——相邻两下标恰一奇一偶
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (hi % 2 == 0))
    return torch.where(pick_hi, vhi, vlo) * x.sign()


def _act_quant_fp8(x, block_size, pow2):
    """参考实现 act_quant：逐块 FP8 量化再反量化（返回 float）。

    注：除法一律写成 `x * (1/scale)`（显式倒数乘法），因为 torch.compile/inductor 会把
    `x / scale` 优化成 `x * (1/scale)`——fp32 下两者差 1ulp。显式写乘法可让 eager 与
    compile **逐位一致**（否则 1ulp 差会在 e2m1 网格舍入处翻转成整格跳变）。
    """
    xf = x.float()
    n = xf.size(-1)
    blocks = xf.reshape(*xf.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = _fast_round_scale(amax, 1.0 / FP8_MAX) if pow2 else amax / FP8_MAX
    q = (blocks * (1.0 / scale)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).reshape(xf.shape)


def _act_quant_fp4(x, block_size, e4m3_scale):
    """参考实现 fp4_act_quant：逐块 FP4 量化再反量化（返回 float）。

    注：同 `_act_quant_fp8`——除法写显式倒数乘法，保证 eager 与 compile 逐位一致。
    """
    xf = x.float()
    n = xf.size(-1)
    blocks = xf.reshape(*xf.shape[:-1], n // block_size, block_size)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    if e4m3_scale:  # 压缩 KV：E4M3 scale，全零块也保非零
        amax = amax.clamp_min(FP4_MAX * (2 ** -9))
        # 参考实现 `T.Cast(FP8, amax / fp4_max)`：scale 本身先舍入到 e4m3 精度（不是留 fp32）
        scale = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:           # indexer：E8M0（2 的幂）
        amax = amax.clamp_min(FP4_MAX * (2 ** -126))
        scale = _fast_round_scale(amax, 1.0 / FP4_MAX)
    q = _round_to_e2m1((blocks * (1.0 / scale)).clamp(-FP4_MAX, FP4_MAX))
    return (q * scale).reshape(xf.shape)


class _FakeQuant(torch.autograd.Function):
    """假量化：前向 量化→反量化，反向直通（straight-through）。

    输入：x；kind（"fp8"/"fp4_e4m3"/"fp4_e8m0"）；block_size；pow2。
    输出：与 x 同形状、同 dtype 的近似值。
    预期行为：前向调参考实现公式；反向把梯度原样回传。
    """

    @staticmethod
    def forward(ctx, x, kind, block_size, pow2):
        if kind == "fp8":
            out = _act_quant_fp8(x, block_size, pow2)
        else:
            out = _act_quant_fp4(x, block_size, e4m3_scale=(kind == "fp4_e4m3"))
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None, None


def fake_quant_fp8(x: torch.Tensor, block_size: int = 32, pow2: bool = True) -> torch.Tensor:
    """FP8 假量化（e4m3，block=32，scale 取 2 的幂）——用于 SWA KV。"""
    return _FakeQuant.apply(x, "fp8", block_size, pow2)


def fake_quant_fp4(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """FP4 假量化（e2m1，±6，block=32，E8M0 scale）——用于 indexer Q/K。"""
    return _FakeQuant.apply(x, "fp4_e8m0", block_size, False)


def fake_quant_compressed_kv(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """压缩 KV 的 FP4 假量化（block=16，E4M3 scale）——用于 Compressor 的 latent。"""
    return _FakeQuant.apply(x, "fp4_e4m3", block_size, False)
