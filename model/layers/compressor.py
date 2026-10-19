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

    def forward(self, x: torch.Tensor, start_pos=0, is_prefill: bool | None = None,
                emit_at=None) -> torch.Tensor | None:
        """把输入压成 KV latent。

        输入：x [B,L,dim]；start_pos（prefill 为 int 0，decode 可为 0 维张量）；
              is_prefill（None=按 `start_pos` 推断；显式传可避免按值特化）；
              emit_at（decode 专用：本步内**凑满整组**的 token 下标元组，如 (1, 3)；
                       None=按单 token 语义推断；空元组=本步不产出 latent）。
        输出：latent [B,m,head_dim]（m=len(emit_at)），m==0 时返回 None。
        预期行为：prefill 把整段按 ratio 分组池化，尾组缓存；
                  decode 逐 token 写进环形状态，遇到 emit_at 里的下标就池化产出一行。
                  **按 token 展开（Python 循环）而非沿 T 维并行**：池化必须按组边界
                  顺序发生；展开后对给定 T 形状与控制流完全静态，可被 CUDA Graph 捕获。
        """
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype

        if is_prefill is None:
            is_prefill = (not torch.is_tensor(start_pos)) and start_pos == 0
        if emit_at is None:
            assert not torch.is_tensor(start_pos), "张量 start_pos 必须显式传 emit_at"
            emit_at = (0,) if (start_pos + 1) % ratio == 0 else ()

        if ratio == 1:  # 一个 token 一组，无需池化
            return self.norm(self.wkv(x))

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if is_prefill:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:  # 尾组不满，存起来
                kv, self.kv_state[:bsz, :remainder] = kv.split([cutoff, remainder], dim=1)
                score, self.score_state[:bsz, :remainder] = score.split([cutoff, remainder], dim=1)
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            if not should_compress:
                return None
            return self.norm(kv.to(dtype))

        # decode：逐 token 填格，凑满整组时池化
        out = []
        for j in range(seqlen):
            slot = (start_pos + j) % ratio
            kv_j, score_j = kv[:, j:j + 1], score[:, j:j + 1]
            if torch.is_tensor(slot):
                # 张量槽位用 scatter（形状固定，可进编译图）
                self.kv_state.scatter_(
                    1, slot.long().view(1, 1, 1).expand(bsz, 1, self.head_dim), kv_j)
                self.score_state.scatter_(
                    1, slot.long().view(1, 1, 1).expand(bsz, 1, self.head_dim), score_j)
            else:
                self.kv_state[:bsz, slot] = kv_j[:, 0]
                self.score_state[:bsz, slot] = score_j[:, 0]
            if j in emit_at:
                out.append((self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1))
                           .sum(dim=1, keepdim=True))
        if not out:
            return None
        return self.norm(torch.cat(out, dim=1).to(dtype))
