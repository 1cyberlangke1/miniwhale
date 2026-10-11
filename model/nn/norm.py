"""RMSNorm。

与参考实现一致：内部以 fp32 计算，再转回输入 dtype。
参考实现 `rms_norm_eps` 极小（1e-20），bf16 直接算会失精度，故必须上转。
"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm。

    输入：x，任意前导维度 + 末维 = dim，任意浮点 dtype。
    输出：同形状张量，dtype 与输入一致。
    预期行为：fp32 下计算 rsqrt(mean(x^2) + eps)，乘可学习权重后转回原 dtype。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * y).to(dtype)
