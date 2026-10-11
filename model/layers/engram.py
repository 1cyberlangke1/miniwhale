"""Engram：n-gram 条件记忆。

用哈希 id 取表行，投影成「每个 hc 副本一个 key + 一个共享 value」，
门控 = 残差流与 key 的归一化点积，加性写回全部 hc 副本。
表很大但只有命中的行有梯度（稀疏更新）。
"""

from __future__ import annotations

import torch
from torch import nn

from model.nn.norm import RMSNorm  # noqa: F401  (保持与包内其它模块一致的导入习惯)


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

        self.embed = nn.Embedding(num_embeddings, self.head_dim)
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
