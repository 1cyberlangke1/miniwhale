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


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int | None = None,
                     inverse: bool = False) -> torch.Tensor:
    """对 x 的末 `rd` 维施加 RoPE（返回新张量，**不原地改**）。

    输入：x [B, L, H, dim] 或 [B, L, dim]，末维为偶数；freqs_cis [L, rd//2]（rd=None 时为 dim//2）；
          rd 旋转维度（None=全部）；inverse=True 施加共轭（反向旋转）。
    输出：同形状**新张量**（前 dim-rd 段原样拷贝）。
    预期行为：把末 rd 维按相邻元素对当复数相乘，与参考实现数值一致。
              调用方式：`q = apply_rotary_emb(q, freqs, rd)`（须赋回，不能原地改）。
              注意：**不能原地改**——参考实现推理是 inference-only 的原地版（`y.copy_`），但本仓库
              要训练，原地改（含切片赋值 `x[..., -rd:] = ...`）会与 autograd 冲突报 inplace 错。
    """
    head = None
    if rd is not None:
        head, x = x[..., :-rd], x[..., -rd:]
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))  # [B,L,...,dim//2]
    f = freqs_cis.conj() if inverse else freqs_cis                # [L, dim//2]
    # 序列维是 dim 1：补成 [1, L, *1, dim//2] 与 xc 对齐
    f = f.reshape(1, f.size(0), *([1] * (xc.dim() - 3)), f.size(1))
    out = torch.view_as_real(xc * f).flatten(-2).to(x.dtype)
    return out if head is None else torch.cat([head, out], dim=-1)
