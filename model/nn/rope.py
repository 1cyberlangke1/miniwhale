"""RoPE（旋转位置编码）。

只旋转末段 rope_head_dim 维；前段 nope 维不动。
输出 cos/sin 表按位置索引。
original_seq_len>0 时对压缩层启用 YaRN 外推（按参考实现 precompute_freqs_cis）。
"""

from __future__ import annotations

import math

import torch


def precompute_freqs_cis(dim: int, seqlen: int, base: float = 10000.0,
                         original_seq_len: int = 0, factor: float = 40.0,
                         beta_fast: int = 32, beta_slow: int = 1) -> torch.Tensor:
    """预计算旋转频率（复数形式）。

    输入：dim 旋转维度；seqlen 最大长度；base 底数；original_seq_len>0 时启用 YaRN；
          factor 外推倍率；beta_fast/beta_slow 平滑带。
    输出：[seqlen, dim//2] 的复数张量。
    预期行为：inv_freq = 1 / base^(arange(0,dim,2)/dim)；
              YaRN 时按波长是否落在训练上下文内，分「保留/除以 factor/线性渐变」三段。
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # 波长在训练上下文内完成的维保留；远超的除以 factor；中间带线性渐变（参考实现公式）
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        inv_freq = inv_freq / factor * (1 - smooth) + inv_freq * smooth

    t = torch.arange(seqlen).float()
    freqs = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """对 x 的末维施加 RoPE（返回新张量，不原地改）。

    输入：x [B, L, H, dim]，dim 为偶数；freqs_cis [L, dim//2]；inverse=True 施加共轭（反向旋转）。
    输出：同形状新张量。
    预期行为：按复数乘法旋转；freqs 广播成 [1, L, 1, dim//2]。
              返回新张量（不原地改），避免与自定义 autograd Function 冲突。
    """
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # [B,L,...,dim//2]
    f = freqs_cis.conj() if inverse else freqs_cis  # [L, dim//2]
    # 序列维是 dim 1：补成 [1, L, *1, dim//2] 与 xc 对齐
    f = f.reshape(1, f.size(0), *([1] * (xc.dim() - 3)), f.size(1))
    out = torch.view_as_real(xc * f).flatten(-2)
    return out.to(dtype)
