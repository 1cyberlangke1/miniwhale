"""层级稀疏 indexer：两级 Top-K。

一级（候选块）：把压缩位置按 block_size 分块，每块取块内最大分，选 topk_blocks 块，
              返回与位置等长的 bool mask。最新那个（未填满的）块强制保留。
二级（精选）：在候选 mask 内按自己的分数取 index_topk 个压缩位置。

index key 由 kv_source 层的 compressor latent 派生（不是各自对 hidden 再压一遍）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.nn.norm import RMSNorm
from model.nn.quant import fake_quant_fp4
from model.nn.rope import apply_rotary_emb


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """一级：每个 query 保留得分最高的 topk_blocks 个块。

    输入：logits [..., n_positions]（不可达位置已置 -inf）；compress_lens 每 query 可见的压缩位置数；
          topk_blocks 保留块数；block_size 块大小。
    输出：与 logits 同形状的 bool mask。
    预期行为：块分 = 块内最大；最新（部分填充）块强制保留；不可达块（分=-inf）被丢弃。
    """
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # 最新块可能只填了一部分，却存着最近的 token，强制钉住
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Indexer(nn.Module):
    """稀疏 indexer 打分与精选。

    输入：x [B, L, dim]（本层注意力输入）；qr [B, L, q_lora_rank]（Q 的低秩投影）；
          index_k [B, T, index_head_dim]（共享的 index key 缓存）；start_pos；offset（压缩位置相对窗口的偏移）。
    输出：topk 索引 [B, L, topk]，int32；不可达位置填 -1，其余加 offset。
    预期行为：score = sum_h ReLU(q·k) * weight；候选 mask 由 candidate_source 层发布；
              在 mask 内取 index_topk 个并按位置排序；不可达（>= compress_len）填 -1。
    """

    def __init__(self, text_cfg, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.dim = text_cfg.hidden_size
        self.n_heads = text_cfg.index_n_heads
        self.index_head_dim = text_cfg.index_head_dim
        self.rope_head_dim = text_cfg.qk_rope_head_dim
        self.q_lora_rank = text_cfg.q_lora_rank
        self.index_topk = text_cfg.index_topk
        self.softmax_scale = self.index_head_dim**-0.5
        self.compress_ratio = text_cfg.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == text_cfg.candidate_source_layer_id
        self.uses_candidates = 0 <= text_cfg.candidate_source_layer_id < layer_id
        self.candidate_topk_blocks = text_cfg.candidate_topk_blocks
        self.candidate_block_size = text_cfg.candidate_block_size

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.index_head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.wk = nn.Linear(text_cfg.head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16)
        self.k_norm = RMSNorm(self.index_head_dim, text_cfg.rms_norm_eps)
        # 参考实现：indexer 的 q 用 attention 层的 freqs_cis 施加 RoPE（Indexer.freqs_cis = Attention.freqs_cis）
        self.freqs_cis = None

    def score(self, x: torch.Tensor, qr: torch.Tensor, index_k: torch.Tensor,
              start_pos: int = 0) -> torch.Tensor:
        """算每个 query 对每个压缩位置的分数。x [B,L,dim] → [B,L,T]。"""
        dtype = x.dtype
        q = self.wq_b(qr.to(self.wq_b.weight.dtype)).unflatten(-1, (self.n_heads, self.index_head_dim)).to(dtype)
        # 参考实现：对 indexer 的 q 施加 RoPE（末 rope_head_dim 维）
        if self.freqs_cis is not None:
            seqlen = q.size(1)
            apply_rotary_emb(q[..., -self.rope_head_dim:], self.freqs_cis[start_pos:start_pos + seqlen])
        # 参考实现对 indexer 的 q/k 做 FP4 量化（fp4_act_quant, block=32, E8M0 scale）
        q = fake_quant_fp4(q, block_size=32)
        weights = self.weights_proj(x.to(self.weights_proj.weight.dtype)).to(dtype) \
            * (self.softmax_scale * self.n_heads**-0.5)
        s = torch.einsum("bshd,btd->bsht", q, fake_quant_fp4(index_k.to(dtype), block_size=32))
        return (s.relu() * weights.unsqueeze(-1)).sum(dim=2)

    def indexer_loss(self, main_attn_dist: torch.Tensor, index_score: torch.Tensor,
                     topk_indices: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """DSA indexer 蒸馏 loss：让 indexer 分数逼近主注意力分布。

        输入：main_attn_dist [B,L,T] 主注意力分布（由 detach 的 q/kv 算出）；
              index_score [B,L,T] indexer 原始分；topk_indices [B,L,K]（-1 填充）。
        输出：标量 loss。
        预期行为：主分布按 topk 索引取出（K 个），indexer 分数也在同样的 K 个位置上
                  softmax；两者算 KL。indexer 参数不经主 loss 梯度，由本 loss 单独驱动。
        """
        max_valid = main_attn_dist.size(-1) - 1
        idx = topk_indices.masked_fill(topk_indices == -1, max_valid).clamp(0, max_valid)
        # 主分布（Q）：归一化后按 topk 取，再归一
        tgt = torch.nn.functional.normalize(main_attn_dist.float(), p=1, dim=-1)
        sel_tgt = torch.gather(tgt, -1, idx)
        sel_tgt = torch.nn.functional.normalize(sel_tgt, p=1, dim=-1)
        # indexer 分布（P）：同样在 topk 位置取分数后 softmax
        sel_score = torch.gather(index_score.float(), -1, idx)
        y = torch.softmax(sel_score, dim=-1)
        logp = torch.clamp(sel_tgt, min=eps).log()
        log_y = (y + eps).log()
        return ((logp - log_y) * sel_tgt).sum(-1).mean()


    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor,
        compress_lens: torch.Tensor | int,
        candidates: torch.Tensor | None = None,
        offset: int = 0,
        start_pos: int = 0,
    ) -> torch.Tensor:
        index_score = self.score(x, qr, index_k, start_pos)  # [B, L, T]
        self._last_score = index_score  # 供 indexer_loss 使用（不参与主 loss 梯度）
        end = index_k.size(1)

        # 不可达位置屏蔽：prefill 时 compress_lens 逐 query（[L,1]），decode 时是标量
        pos = torch.arange(end, device=x.device)
        if isinstance(compress_lens, torch.Tensor) and compress_lens.dim() > 0:
            # compress_lens [L,1] → 广播到 [1,L,T]
            reach = pos.view(1, 1, -1) >= compress_lens.unsqueeze(0)
        else:
            reach = pos.view(1, 1, -1) >= compress_lens
        index_score = index_score.masked_fill(reach, -torch.inf)

        if self.uses_candidates and candidates is not None:
            index_score = index_score.masked_fill(~candidates, -torch.inf)

        topk = min(self.index_topk, end)
        idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < compress_lens, idxs + offset, -1).int()


class IndexerLossAutoScaler(torch.autograd.Function):
    """把 indexer loss 挂到某个输出张量上，不改变前向值，只在反向注入它的梯度。

    参考实现做法（DSA indexer）：indexer 的 KL loss 不能直接 backward（会与主 loss 重复计入），
    而是通过本 Function 挂在 attention 输出上——前向透传，反向把 indexer_loss 的梯度按
    `main_loss_backward_scale` 缩放后回传，从而用同一个 backward 同时驱动主 loss 和 indexer loss。
    """

    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (indexer_loss,) = ctx.saved_tensors
        loss_scale = IndexerLossAutoScaler.main_loss_backward_scale
        if loss_scale is None:
            loss_scale = torch.ones((), dtype=indexer_loss.dtype, device=indexer_loss.device)
        else:
            loss_scale = loss_scale.to(device=indexer_loss.device, dtype=indexer_loss.dtype)
        return grad_output, torch.ones_like(indexer_loss) * loss_scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """设置反向时施加在 indexer loss 上的缩放（通常取主 loss 的 detach）。"""
        IndexerLossAutoScaler.main_loss_backward_scale = scale.detach()
