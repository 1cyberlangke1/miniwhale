"""Compressor：把 compress_ratio 个 token 池化成一个 KV latent。

- ratio == 1：退化成普通投影（bf16，无门控、无池化）
- ratio > 1：学习式 softmax 门控池化（fp32）
- 输出 latent 在 RoPE 之前（indexer 需要未旋转形式，由 Attention 之后再旋转）

decode 时按步喂入，组没凑满则返回 None（部分组存在 kv_state/score_state）。
"""

from __future__ import annotations

import torch
from torch import nn

from model.nn.norm import RMSNorm


class Compressor(nn.Module):
    """KV 压缩器。

    输入：x [B, L, dim]；start_pos 起始位置（0=prefill，>0=decode 单步）。
    输出：latent [B, L', head_dim]（L' = L/ratio，或 decode 时为 1）；组未满时返回 None。
    预期行为：ratio==1 走纯投影（bf16）；ratio>1 走 fp32 门控池化；
              尾组不满时缓存进 state 等待后续步。
    """

    def __init__(self, text_cfg, layer_id: int, max_batch_size: int = 1):
        super().__init__()
        ratio = text_cfg.compress_ratios[layer_id]
        self.compress_ratio = ratio
        self.head_dim = text_cfg.head_dim
        self.norm = RMSNorm(self.head_dim, text_cfg.rms_norm_eps)

        # ratio 1 是纯投影，留 bf16；ratio>1 的池化在 fp32 做，权重也升 fp32
        wdtype = torch.float32 if ratio > 1 else torch.bfloat16
        self.wkv = nn.Linear(text_cfg.hidden_size, self.head_dim, bias=False, dtype=wdtype)
        if ratio > 1:
            self.wgate = nn.Linear(text_cfg.hidden_size, self.head_dim, bias=False, dtype=torch.float32)
            state_shape = (max_batch_size, ratio, self.head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "score_state", torch.full(state_shape, -torch.inf, dtype=torch.float32), persistent=False
            )

    def reset_state(self):
        """清空跨步缓存（新序列开始时调用）。

        输入：无。
        输出：无。
        预期行为：kv_state 归零、score_state 复位为 -inf；ratio==1 时无状态可清。
        """
        if self.compress_ratio > 1:
            self.kv_state.zero_()
            self.score_state.fill_(-torch.inf)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor | None:
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype

        if ratio == 1:  # 一个 token 一组，无需池化
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # 尾组不满，存起来
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:  # decode：填一格，组满才池化
            should_compress = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should_compress:
                kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return None
        return self.norm(kv.to(dtype))
