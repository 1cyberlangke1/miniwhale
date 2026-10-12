"""Engram：n-gram 条件记忆。

用哈希 id 取表行，投影成「每个 hc 副本一个 key + 一个共享 value」，
门控 = 残差流与 key 的归一化点积，加性写回全部 hc 副本。
表很大但只有命中的行有梯度（稀疏更新）。
表按参考实现 ParallelEngramEmbedding 以 **FP8 存储 + 块 scale**（查表时反量化）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.nn.norm import RMSNorm  # noqa: F401  (保持与包内其它模块一致的导入习惯)

FP8_BLOCK = 32  # 参考实现 fp8_block_size：一个 scale 管 32 个元素


class EngramTable(nn.Module):
    """n-gram 哈希表：FP8 存储 + 逐行 e8m0 scale（按参考实现 ParallelEngramEmbedding）。

    输入：indices [..., ]（哈希行号）。
    输出：[..., dim]（bf16，反量化后的行）。
    预期行为：权重存 **fp8_e4m3fn**、每行一个 **e8m0** scale（与参考实现 checkpoint 一致）；
              查表取 fp8 行与 scale，按参考实现公式反量化。梯度经 autograd 直通回 master
              （fp8 参数本身不可反传，故用 fp32 master 存值、前向量化成 fp8 存储语义）。
    """

    def __init__(self, num_embeddings: int, dim: int, block_size: int = FP8_BLOCK):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        # master 权重 fp32（可训练）；前向按参考实现 fp8+scale 反量化
        self.weight = nn.Parameter(torch.empty(num_embeddings, dim, dtype=torch.float32))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """初始化：正态小值。"""
        self.weight.copy_(torch.randn(self.num_embeddings, self.dim) * 0.02)

    def _quantize_rows(self, rows: torch.Tensor):
        """按参考实现公式把 fp32 行量化成 fp8 + e8m0 scale（**每 block_size 个元素一个 scale**）。

        输入：rows [..., dim] fp32。
        输出：(fp8 [..., dim], scale [..., dim // block_size] e8m0)。
        预期行为：amax 按 **block**（dim 方向切 block_size）取、clamp_min(1e-4)；
                  scale = 2^ceil(log2(amax/448))（e8m0）。按参考实现 ParallelEngramEmbedding：
                  scale 是 block-wise（dim // fp8_block_size），不是每行一个。
        """
        bs = self.block_size
        blocks = rows.unflatten(-1, (-1, bs))                           # [..., dim//bs, bs]
        amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)  # [..., dim//bs, 1]
        scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))        # e8m0
        quant = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return quant.flatten(-2), scale.squeeze(-1)                     # [...,dim], [...,dim//bs]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """查表（训练/推理**同一条**路径）。

        输入：indices [...]。
        输出：[..., dim] bf16。
        预期行为：master 权重（fp32）直接查表转 bf16。参考实现推理存 fp8、查表时反量化；
                  猫猫 master 存反量化后的 fp32，查表即等价（无额外量化）。
                  **不在训练时做 QAT**：技术报告的 QAT 只用于 indexer 与主 KV cache，
                  Engram 表是「存 fp8、查表反量化」的推理格式，不是训练机制。
                  若训练时再量化一次会造成训练/eval 数值分叉（训练路径从未被对拍验证）。
        """
        return F.embedding(indices, self.weight).to(torch.bfloat16)


class Engram(nn.Module):
    """一层 Engram。

    输入：x [B, L, hc_mult, dim]；hash_ids [B, L, n_hash_cols]；
          token_mask [B, L]（False 处门控置 0，残差原样透传）。
    输出：同形状张量，dtype 与输入一致；不改写也不别名 x。
    预期行为：门控按 (token, hc 副本) 各自归一化（非跨副本联合）；
              带符号平方根后 sigmoid；fp32 计算，末步转回输入 dtype。
    """

    def __init__(self, text_cfg, layer_id: int):
        super().__init__()
        self.layer_hash_index = tuple(text_cfg.engram_layer_ids).index(layer_id)
        self.dim = text_cfg.hidden_size
        self.hc_mult = text_cfg.hc_mult
        self.eps = text_cfg.rms_norm_eps
        self.clamp_value = 1e-6

        num_embeddings = text_cfg.engram_num_embeddings[self.layer_hash_index]
        n_hash_cols = (text_cfg.engram_max_ngram_size - 1) * text_cfg.engram_n_heads
        self.head_dim = text_cfg.engram_head_dim

        self.embed = EngramTable(num_embeddings, self.head_dim)
        self.wkv = nn.Linear(n_hash_cols * self.head_dim, self.dim * (self.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))

    def forward(
        self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # 只以乘积形式出现
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # sigmoid 前带符号平方根，与训练 kernel 对齐
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
