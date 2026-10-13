"""SigLIP 对比预训练（ViT 独立训练的第一阶段，报告 §4.2.2）。

参考实现流程：视觉编码器先做 SigLIP sigmoid 对比预训练（约 47B 图文对，分辨率限到 224×224），
再自回归微调（接进 LLM 做 next-token，见 `train_vlm.py`）。本模块实现**第一阶段**：
一个轻量图文双塔，用 SigLIP 的 pairwise sigmoid loss 把图像与文本嵌入对齐到同一空间。

为什么单独一个文本塔：对比阶段视觉编码器尚未接进语言骨干，需要一个文本编码器把 caption
编到同一空间。参考实现用 4B MoE LLM 当文本塔，这里用缩水的小文本塔。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.vision import ViT
from model.nn.linear import Linear
from model.nn.norm import RMSNorm


class TextBlock(nn.Module):
    """文本塔一层：因果自注意力 + SwiGLU MLP，各带 RMSNorm 残差。

    输入：x [B, L, D]；pad_mask [B, L]（bool，True=有效 token）。
    输出：[B, L, D]。
    预期行为：注意力为因果 + padding 屏蔽；padded query 行给全 0 掩码（softmax 均匀，
              避免整行 -inf 产生 NaN；其输出随后被池化的 mask 丢掉）。
    """

    def __init__(self, dim: int, n_heads: int, eps: float = 1e-6):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm1 = RMSNorm(dim, eps)
        self.qkv = Linear(dim, 3 * dim)
        self.wo = Linear(dim, dim)
        self.norm2 = RMSNorm(dim, eps)
        self.w1 = Linear(dim, 2 * 4 * dim)   # SwiGLU：2×（4·dim）中间维
        self.w2 = Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, L, d = x.shape
        h = self.norm1(x)
        q, k, v = (t.view(b, L, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in self.qkv(h).chunk(3, dim=-1))
        # 因果 + padding 的加性掩码（0 保留 / -inf 屏蔽）
        neg = torch.finfo(x.dtype).min
        causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
        attn = torch.zeros(b, 1, L, L, dtype=x.dtype, device=x.device)
        attn = attn.masked_fill(~causal[None, None], neg)
        attn = attn.masked_fill(~pad_mask[:, None, None, :], neg)
        attn = attn.masked_fill(~pad_mask[:, None, :, None], 0.0)  # padded query 行给 0
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn)
        x = x + self.wo(o.transpose(1, 2).reshape(b, L, d))
        gate, up = self.w1(self.norm2(x)).chunk(2, dim=-1)
        return x + self.w2(F.silu(gate) * up)


class TextEncoder(nn.Module):
    """对比阶段的小文本塔：embed → N 层 → RMSNorm → mask 平均池化 → 投影 → L2 归一化。

    输入：ids [B, L]（6400 词表）；pad_mask [B, L]（bool）。
    输出：[B, proj_dim]（L2 归一化，fp32）。
    预期行为：只对有效 token 做平均池化，得到整段文本的单一嵌入。
    """

    def __init__(self, vocab_size: int, dim: int, n_layers: int, n_heads: int,
                 proj_dim: int, eps: float = 1e-6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, n_heads, eps) for _ in range(n_layers)])
        self.norm = RMSNorm(dim, eps)
        self.proj = Linear(dim, proj_dim)

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        x = self.norm(x)
        m = pad_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return F.normalize(self.proj(pooled).float(), dim=-1)


class ImageEncoder(nn.Module):
    """对比阶段的图像塔：ViT → 全 patch 平均池化 → 投影 → L2 归一化。

    输入：patches_list list[[n,3,p,p]]；grid_list list[(n_vit_h, n_vit_w)]（每张图一个）。
    输出：[B, proj_dim]（L2 归一化，fp32）。
    预期行为：每张图独立过 ViT（patch 数随分辨率不同，无法 stack），池化后拼成 batch。
    """

    def __init__(self, vision_cfg, proj_dim: int):
        super().__init__()
        self.vit = ViT(vision_cfg)
        self.proj = Linear(vision_cfg.hidden_size, proj_dim)

    def forward_one(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """单图：ViT [n,dim] → 平均池化 [dim] → 投影 [proj_dim]。"""
        return self.proj(self.vit(patches, n_h, n_w).mean(dim=0))

    def forward(self, patches_list, grid_list) -> torch.Tensor:
        outs = [self.forward_one(p, h, w) for p, (h, w) in zip(patches_list, grid_list)]
        return F.normalize(torch.stack(outs).float(), dim=-1)


class SigLIP(nn.Module):
    """图文双塔 + SigLIP pairwise sigmoid loss（报告 §4.2.2 第一阶段）。

    输入：patches_list / grid_list（图）；text_ids [B,L] / text_mask [B,L]（文）。
    输出：logits [B, B]（图 i · 文 j）。
    预期行为：对角为配对正样本；loss = -logsigmoid(配对符号 × logits) 的均值。
    """

    def __init__(self, vocab_size: int, vision_cfg, proj_dim: int = 256,
                 text_dim: int = 256, text_layers: int = 4, text_heads: int = 4):
        super().__init__()
        self.image = ImageEncoder(vision_cfg, proj_dim)
        self.text = TextEncoder(vocab_size, text_dim, text_layers, text_heads, proj_dim)
        # SigLIP 的可学习温度与偏置（初始 bias=-10 是 SigLIP 论文的经验值）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.logit_bias = nn.Parameter(torch.tensor(-10.0))

    def forward(self, patches_list, grid_list, text_ids, text_mask) -> torch.Tensor:
        img = self.image(patches_list, grid_list)      # [B, proj]
        txt = self.text(text_ids, text_mask)           # [B, proj]
        return img @ txt.t() * self.logit_scale.exp() + self.logit_bias

    @staticmethod
    def loss(logits: torch.Tensor) -> torch.Tensor:
        """pairwise sigmoid loss。

        输入：logits [B, B]（图 i · 文 j，已含温度与偏置）。
        输出：标量。
        预期行为：对角（i==j）为 +1 正样本，其余 -1 负样本；-logsigmoid(sign×logits) 求均值。
        """
        b = logits.size(0)
        sign = 2 * torch.eye(b, device=logits.device) - 1
        return -F.logsigmoid(sign * logits).mean()
