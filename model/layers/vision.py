"""ViT：从零训练的视觉编码器 + Aligner。

- 线性 patch embed（非卷积，兼容 Muon）
- 2D-RoPE（按 patch 的行列位置）
- RMSNorm + SwiGLU
- Aligner：3×3 pixel-unshuffle 降采样后映射到语言骨干维度
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from model.nn.linear import Linear


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    """按 patch 行列位置生成 2D-RoPE 的 cos/sin。

    输入：n_h/n_w patch 网格；dim 旋转维度；theta 底数。
    输出：(cos, sin) 各 [n_h*n_w, 1, dim]。
    预期行为：行、列各占 dim/2 的频率，拼成 dim 维。
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x 施加 2D-RoPE（前后半维配对旋转）。

    输入：x [n, h, d]；cos/sin [n, 1, d]。
    输出：同形状。
    预期行为：x1*cos - x2*sin 与 x2*cos + x1*sin 拼接。
    """
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class VisionRMSNorm(nn.Module):
    """ViT 用 RMSNorm：权重 fp32，计算 fp32 后转回。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(dtype)


class PatchEmbed(nn.Module):
    """线性 patch 嵌入（按参考实现，用 Linear 而非卷积）。"""

    def __init__(self, vision_cfg):
        super().__init__()
        self.proj = Linear(3 * vision_cfg.patch_size**2, vision_cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [n_patches, 3, ps, ps] → [n_patches, dim]。"""
        return self.proj(x.flatten(1))


class VisionAttention(nn.Module):
    """ViT 双向注意力（2D-RoPE + SDPA）。"""

    def __init__(self, vision_cfg):
        super().__init__()
        self.n_heads = vision_cfg.num_attention_heads
        self.head_dim = vision_cfg.hidden_size // self.n_heads
        self.wqkv = Linear(vision_cfg.hidden_size, 3 * vision_cfg.hidden_size)
        self.wo = Linear(vision_cfg.hidden_size, vision_cfg.hidden_size)

    def forward(self, x, cos, sin):
        n = x.size(0)
        q, k, v = (t.view(n, self.n_heads, self.head_dim) for t in self.wqkv(x).chunk(3, dim=-1))
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class VisionMLP(nn.Module):
    """SwiGLU MLP（w1 输出 2×inter，切半做 gate/up）。"""

    def __init__(self, vision_cfg):
        super().__init__()
        self.w1 = Linear(vision_cfg.hidden_size, 2 * vision_cfg.intermediate_size)
        self.w2 = Linear(vision_cfg.intermediate_size, vision_cfg.hidden_size)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    """ViT block：attn + mlp，各带残差。"""

    def __init__(self, vision_cfg):
        super().__init__()
        self.norm1 = VisionRMSNorm(vision_cfg.hidden_size)
        self.attn = VisionAttention(vision_cfg)
        self.norm2 = VisionRMSNorm(vision_cfg.hidden_size)
        self.mlp = VisionMLP(vision_cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """ViT：对单张图像做全双向注意力 + 2D-RoPE。

    输入：patches [n_h*n_w, 3, ps, ps]；n_h/n_w patch 网格。
    输出：[n_h*n_w, dim]。
    预期行为：线性 patch embed → N 层 block → 末端 RMSNorm。
    """

    def __init__(self, vision_cfg):
        super().__init__()
        self.rope_dim = vision_cfg.hidden_size // vision_cfg.num_attention_heads // 2
        self.rope_theta = vision_cfg.rope_theta
        self.patch_embed = PatchEmbed(vision_cfg)
        self.blocks = nn.ModuleList([VisionBlock(vision_cfg) for _ in range(vision_cfg.num_hidden_layers)])
        self.norm = VisionRMSNorm(vision_cfg.hidden_size)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
        cos, sin = cos.to(x.device), sin.to(x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class Aligner(nn.Module):
    """视觉特征对齐：3×3 pixel-unshuffle 降采样后映射到语言维度。

    输入：x [n_h*n_w, vision_dim]；n_h/n_w。
    输出：[n_h/r * n_w/r, lang_dim]。
    预期行为：unfold 重排通道 → gelu(w1) → w2。
    """

    def __init__(self, vision_cfg, lang_dim: int):
        super().__init__()
        self.downsample_ratio = vision_cfg.downsample_ratio
        r = self.downsample_ratio
        self.w1 = Linear(vision_cfg.hidden_size * r**2, lang_dim)
        self.w2 = Linear(lang_dim, lang_dim)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return self.w2(F.gelu(self.w1(x)))
