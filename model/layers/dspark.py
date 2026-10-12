"""DSpark 投机解码（按参考实现 DSparkBlock / DSparkAttention 结构）。

参考实现（DeepSeek-V4.1-Flash 推理实现里的 DSpark* 部分）：
- `DSparkBlock(Block)`：每个 draft stage 是一个**完整 Block**（MLA + MoE + mHC），
  子类化普通 Block，只覆写 attention 与追加 draft 头。
- 首块有 `main_proj`/`main_norm`：把主干 target 层的 hidden 拼起来投成 draft 输入。
- 末块有 `norm`/`markov_head`/`confidence_head`。
- 多 stage 共享主干的 `embed`/`head`。
- 一次前向**并行**出 block_size 个 draft 位置；markov head 建块内依赖，confidence head 出接受概率。

本文件按上述结构实现（纯 torch、可训练）。
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.attention import Attention, get_window_topk_idxs
from model.nn.linear import Head, Linear
from model.nn.norm import RMSNorm
from model.nn.quant import fake_quant_fp8
from model.nn.rope import apply_rotary_emb


@lru_cache(1)
def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int,
                         device=None):
    """draft block 每个 query 可见的 KV 槽位（按参考实现 get_dspark_topk_idxs）。

    输入：window_size；bsz；block_size；start_pos（>0）；device（**直接在目标设备建张量**）。
    输出：[bsz, block_size, min(win, start_pos+1)+block_size]，int32。
    预期行为：每个 draft query 看 [窗口内历史槽 … | 本块 block_size 个槽]。
    """
    assert start_pos > 0
    matrix = torch.cat([
        torch.arange(min(window_size, start_pos + 1), device=device),
        window_size + torch.arange(block_size, device=device),
    ])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


def sample_anchor_positions(seq_len: int, loss_mask: torch.Tensor, num_anchors: int,
                            device: torch.device):
    """采样训练用的 anchor 位置（按参考实现 sample_anchor_positions）。

    输入：seq_len；loss_mask [B,S]（1=可算 loss）；num_anchors；device。
    输出：(anchors [B,N] long, block_keep_mask [B,N] bool)。
    预期行为：只在“自身与下一 token 都有效”的位置里随机取 N 个；不足则补 dummy（keep=False）。
    """
    bsz = loss_mask.shape[0]
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, num_anchors, dtype=torch.long, device=device)
        return anchors, torch.zeros_like(anchors, dtype=torch.bool)
    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1:num_candidates + 1] > 0.5
    valid = anchor_valid & first_target_valid
    valid_counts = valid.sum(dim=1)

    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(bsz, -1)
    masked_indices = torch.where(valid, indices, torch.full_like(indices, seq_len + 1))
    random_vals = torch.rand(bsz, num_candidates, device=device)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    if num_candidates < num_anchors:
        pad = torch.full((bsz, num_anchors - num_candidates), seq_len + 1,
                         dtype=gathered.dtype, device=device)
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :num_anchors].sort(dim=1).values
    keep = torch.arange(num_anchors, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=num_anchors)
    anchors = torch.where(keep, anchors, torch.zeros_like(anchors))
    return anchors, keep


def build_eval_mask(seq_len: int, loss_mask: torch.Tensor, label_indices: torch.Tensor,
                    safe_label_indices: torch.Tensor, block_keep_mask: torch.Tensor) -> torch.Tensor:
    """有效前缀掩码（按参考实现 build_eval_mask）。

    输入：seq_len；loss_mask [B,S]；label_indices [B,N,K]；safe_label_indices（clamp 后）；
          block_keep_mask [B,N]。
    输出：[B,N,K] bool。
    预期行为：目标越界 / loss_mask=0 / 非保留块 → 无效；再沿块内做前缀累积（连续有效前缀）。
    """
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1), 2, safe_label_indices)
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()


def _dspark_train_mask(anchor_positions, block_keep_mask, seq_len, block_size, device):
    """训练注意力掩码（按参考实现 create_dspark_attention_mask 的 dspark_mask_mod）。

    输入：anchor_positions [B,N]；block_keep_mask [B,N]；seq_len；block_size；device。
    输出：[B, N*K, S+N*K] bool。
    预期行为：query 块 j 位置 k 可见 —— context 位置 < anchor[b,j]，或**本块内全部** draft 位置
              （整块可见，无块内因果；按参考实现 `q_block_id == kv_block_id`）；仅当该块 keep 时。
    """
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size
    kv_len = seq_len + q_len
    q_block = torch.arange(q_len, device=device).div(block_size, rounding_mode="floor")
    q_block = q_block.view(1, num_blocks, block_size, 1)            # [1,N,K,1]
    anchor_pos = anchor_positions.view(bsz, num_blocks, 1, 1)       # [B,N,1,1]
    kv_idx = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)  # [1,1,1,KV]
    mask_context = (kv_idx < seq_len) & (kv_idx < anchor_pos)       # [B,N,1,KV]
    kv_block = (kv_idx - seq_len).div(block_size, rounding_mode="floor")
    mask_draft = (kv_idx >= seq_len) & (q_block == kv_block)        # [1,N,K,KV]
    keep = block_keep_mask.view(bsz, num_blocks, 1, 1)              # [B,N,1,1]
    mask = (mask_context | mask_draft) & keep                       # [B,N,K,KV]
    return mask.reshape(bsz, q_len, kv_len)


def _sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """采样（按参考实现 sample）：temperature=0 贪心，否则 Gumbel-max。

    输入：logits [B, V]；temperature。
    输出：[B] token id。
    预期行为：Gumbel-max = argmax(softmax(logits/T) / Exp(1))，等价多项采样但无 GPU-CPU 同步。
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def _rope_with_freqs(x: torch.Tensor, freqs: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """对 x 末 rd 维施加逐位置 RoPE（支持 [B,L,...] 的 batch 化 freq）。

    输入：x [B,L,H,D]；freqs [B,L,rd//2,2]（末维=(cos,sin)，每样本每位置不同）；rd 旋转维度；inverse 施加共轭。
    输出：同形状新张量。
    预期行为：只旋转末 rd 维（前段 nope 维不动）；freqs 广播到 head 维。参考实现训练各样本 anchor 不同，故需 batch 化。
              纯实数实现（不用复数算子），与 `apply_rotary_emb` 同款，便于 torch.compile。
    """
    dtype = x.dtype
    head = x[..., :-rd]
    tail = x[..., -rd:].float()
    # cos/sin 广播到 [B, L, *1, rd//2]，与 tail 的 [..., rd//2, 2] 对齐
    cos = freqs[..., 0].reshape(freqs.size(0), freqs.size(1), *([1] * (tail.dim() - 3)), freqs.size(-2))
    sin = freqs[..., 1].reshape(freqs.size(0), freqs.size(1), *([1] * (tail.dim() - 3)), freqs.size(-2))
    if inverse:
        sin = -sin
    x0, x1 = tail[..., 0::2], tail[..., 1::2]
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    tail = torch.stack([o0, o1], dim=-1).flatten(-2).to(dtype)
    return torch.cat([head, tail], dim=-1)


class DSparkAttention(Attention):
    """draft stage 的注意力：main_x 写窗口 KV，draft block 并行 attend（按参考实现 DSparkAttention）。"""

    def __init__(self, text_cfg, layer_id: int):
        super().__init__(text_cfg, layer_id, is_backbone=False)
        self.block_size = text_cfg.dspark_block_size

    def forward_train(self, x, context_hidden, position_ids, anchor_positions,
                      block_keep_mask, seq_len):
        """训练前向：teacher-forced，每个 draft block 并行 attend 上下文 + 本块。

        输入：x [B,Q,D]（Q=N*K，draft 输入）；context_hidden [B,S,D]（主干 target hidden）；
              position_ids [B,Q]；anchor_positions [B,N]；block_keep_mask [B,N]；seq_len。
        输出：[B,Q,D]。
        预期行为：按参考实现 create_dspark_attention_mask —— query(b,j,k) 看 context 位置 < anchor[b,j]，
                  以及本块 draft 位置 k'<=k；dummy 块整体屏蔽。
        """
        bsz, q_len, _ = x.shape
        rd = self.rope_head_dim
        q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        k_ctx = self.kv_norm(self.wkv(context_hidden))          # [B,S,head_dim]
        k_draft = self.kv_norm(self.wkv(x))                     # [B,Q,head_dim]

        ctx_pos = torch.arange(seq_len, device=x.device)
        # freqs 为 [seq, rd//2, 2]（实数 (cos,sin) 存储）；上下文位置对所有样本相同，
        # 故 unsqueeze(0) 后广播到 batch —— 保持末维 2，得到 [B,S,rd//2,2]。
        ctx_freqs = self.freqs_cis[ctx_pos].unsqueeze(0).expand(bsz, -1, -1, -1)   # [B,S,rd//2,2]
        draft_freqs = self.freqs_cis[position_ids]              # [B,Q,rd//2,2]
        k_ctx = _rope_with_freqs(k_ctx.unsqueeze(2), ctx_freqs, rd).squeeze(2)
        q = _rope_with_freqs(q, draft_freqs, rd)
        k_draft = _rope_with_freqs(k_draft.unsqueeze(2), draft_freqs, rd).squeeze(2)

        k = torch.cat([k_ctx, k_draft], dim=1)                  # [B,S+Q,head_dim]
        scores = torch.einsum("bqhd,bkd->bqhk", q.float(), k.float()) * self.softmax_scale
        mask = _dspark_train_mask(anchor_positions, block_keep_mask, seq_len,
                                  self.block_size, x.device)     # [B,Q,S+Q]
        scores = scores.masked_fill(~mask.unsqueeze(2), -torch.inf)
        sink = self.attn_sink.view(1, 1, self.n_heads, 1).expand(bsz, q_len, self.n_heads, 1)
        scores = torch.cat([scores, sink], dim=-1)
        probs = torch.nan_to_num(scores.softmax(dim=-1)[..., :-1], nan=0.0)
        o = torch.einsum("bqhk,bkd->bqhd", probs, k.float()).to(q.dtype)

        o = o.reshape(bsz, q_len, self.n_groups, -1).to(self.wo_a.weight.dtype)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bqgd,grd->bqgr", o, wo_a)
        return self.wo_b(o.flatten(2).to(self.wo_b.weight.dtype))

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        assert self.compress_ratio == 0, "DSpark 层必须是纯 SWA（compress_ratio=0）"
        bsz, seqlen, _ = main_x.size()
        win, rd = self.window_size, self.rope_head_dim

        # main_x（主干 target hidden）→ 窗口 KV；参考实现对 KV 做 FP8 量化
        main_freqs = self.freqs_cis[start_pos:start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        main_kv = apply_rotary_emb(main_kv, main_freqs, rd)
        main_kv = fake_quant_fp8(main_kv, block_size=32, pow2=True)

        self._ensure_cache(bsz)
        if start_pos == 0:  # prefill：只把 main_x 灌进窗口缓存，不产出 draft
            if seqlen <= win:
                with torch.no_grad():
                    self.window_kv_cache[:bsz, :seqlen] = main_kv.detach()
            else:
                cutoff = seqlen % win
                with torch.no_grad():
                    self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = \
                        main_kv[:, -win:].detach().split([win - cutoff, cutoff], dim=1)
            return x

        # decode：draft block 并行前向
        bsz, block_size, _ = x.size()
        freqs = self.freqs_cis[start_pos + seqlen:start_pos + seqlen + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary_emb(q, freqs, rd)
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rotary_emb(kv, freqs, rd)
        kv = fake_quant_fp8(kv, block_size=32, pow2=True)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos, x.device)
        with torch.no_grad():
            self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1).detach()
        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)

        o = self._sparse_attn(q, kv, topk_idxs)
        o = apply_rotary_emb(o, freqs, rd, inverse=True)
        o = o.reshape(bsz, block_size, self.n_groups, -1).to(self.wo_a.weight.dtype)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2).to(self.wo_b.weight.dtype))


class DSparkMarkovHead(nn.Module):
    """块内 token 依赖：由上一个 token 出低秩 logits 偏置（按参考实现 DSparkMarkovHead）。

    输入：token_ids [B]（上一个 token）。
    输出：(logits [B, vocab], embed [B, rank])。
    """

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank)
        self.head = Head(rank, vocab_size)

    def forward(self, token_ids: torch.Tensor):
        embed = self.embed(token_ids)
        return self.head(embed), embed

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """取上一 token 的 markov 嵌入（按参考实现 VanillaMarkov.get_prev_embeddings）。"""
        return self.embed(token_ids.long())

    def project_bias(self, latent_states: torch.Tensor) -> torch.Tensor:
        """把 markov 隐状态投成 logits 偏置（按参考实现 project_bias）。"""
        return self.head(latent_states)

    def apply_block_logits(self, base_logits, *, token_ids, hidden_states=None):
        """对整块各位置加 markov 偏置（按参考实现 VanillaMarkov.apply_block_logits）。

        输入：base_logits [B,N,K,V]；token_ids [B,N,K]（各位置真实前驱）；hidden_states 未用。
        输出：[B,N,K,V]。
        预期行为：偏置 = head(embed(prev_token))，与 base_logits 相加。
        """
        if base_logits.size(2) == 0:
            return base_logits
        return base_logits + self.project_bias(self.get_prev_embeddings(token_ids))


class DSparkConfidenceHead(nn.Module):
    """每位置接受概率（fp32，按参考实现 DSparkConfidenceHead）。

    输入：hidden [B, block, dim]；markov_embed [B, block, rank]。
    输出：[B, block]。
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor) -> torch.Tensor:
        h = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(h.float()).squeeze(-1)


class DSparkBlock(nn.Module):
    """一个 DSpark draft stage（按参考实现 DSparkBlock 的结构）。

    与普通 Block 同构：mHC 残差流 + attention + MoE；首块加 main_proj/main_norm，
    末块加 norm/markov_head/confidence_head。

    输入：x [B, block, hc, dim]；start_pos；pre_mix；main_x [B, S, dim]（主干 target hidden）。
    输出：(x, pre_mix)；末块另可调 forward_head 出草稿。
    """

    def __init__(self, text_cfg, layer_id: int, stage_id: int):
        super().__init__()
        from model.layers.engram import Engram  # 避免循环导入
        from model.layers.mhc import HyperConnection
        from model.layers.moe import MoE

        self.layer_id = layer_id
        self.stage_id = stage_id
        self.hc_mult = text_cfg.hc_mult
        self.block_size = text_cfg.dspark_block_size
        self.noise_token_id = text_cfg.dspark_noise_token_id

        self.attn = DSparkAttention(text_cfg, layer_id)
        # 参考实现 get_moe_config：DSpark 层用 dspark_n_routed_experts（0 则回退主干）
        from dataclasses import replace
        dspark_n = getattr(text_cfg, "dspark_n_routed_experts", 0) or text_cfg.n_routed_experts
        dspark_k = getattr(text_cfg, "dspark_num_experts_per_tok", 0) or text_cfg.num_experts_per_tok
        moe_cfg = replace(text_cfg, n_routed_experts=dspark_n, num_experts_per_tok=dspark_k)
        self.ffn = MoE(moe_cfg)
        self.attn_norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        self.hc = HyperConnection(text_cfg)
        self.engram = None  # draft 是文本，无 VL bias

        if stage_id == 0:
            assert len(text_cfg.dspark_target_layer_ids) > 0, "DSpark 需要 target 层"
            self.main_proj = Linear(
                text_cfg.hidden_size * len(text_cfg.dspark_target_layer_ids),
                text_cfg.hidden_size, bias=False)
            self.main_norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        if stage_id == text_cfg.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
            self.markov_head = DSparkMarkovHead(text_cfg.vocab_size, text_cfg.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(
                text_cfg.hidden_size + text_cfg.dspark_markov_rank)

    def forward_train(self, x, pre_mix, context_hidden, position_ids,
                      anchor_positions, block_keep_mask, seq_len):
        """训练前向：与普通 Block 同构，但 attention 走 teacher-forced 并行路径。

        输入：x [B,Q,hc,dim]；pre_mix；context_hidden [B,S,D]；position_ids [B,Q]；
              anchor_positions [B,N]；block_keep_mask [B,N]；seq_len。
        输出：(x, ffn_pre)。
        预期行为：结构按参考实现 DSparkBlock.forward（train 分支）——mHC + attn + MoE。
        """
        residual = x
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(x, "attn")
        h = self.hc.hc_pre(x, pre_mix)
        h = self.attn_norm(h)
        h = self.attn.forward_train(h, context_hidden, position_ids,
                                    anchor_positions, block_keep_mask, seq_len)
        x = self.hc.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(x, "ffn")
        h = self.hc.hc_pre(x, attn_pre)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = self.hc.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre

    def forward(self, x, start_pos, pre_mix, main_x):
        residual = x
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(x, "attn")
        h = self.hc.hc_pre(x, pre_mix)
        h = self.attn_norm(h)
        h = self.attn(h, start_pos, main_x)
        x = self.hc.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(x, "ffn")
        h = self.hc.hc_pre(x, attn_pre)
        h = self.ffn_norm(h)
        h = self.ffn(h)
        x = self.hc.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class DSparkHead(nn.Module):
    """DSpark 多 stage drafter（按参考实现 Transformer.mtp + forward_spec）。

    输入：main_hidden [B, S, D]（主干 target 层 hidden 拼接）；input_ids [B]（seed token）；
          embed/head（共享主干）。
    输出：(output_ids [B, block+1], logits [B, block, V], confidence [B, block])。
    预期行为：main_proj 投出 draft 输入 → 逐 stage 前向 → 末块 markov 偏置 + 置信度。
    """

    def __init__(self, text_cfg, embed: nn.Module, head: nn.Module):
        super().__init__()
        self.cfg = text_cfg
        self.block_size = text_cfg.dspark_block_size
        self.hc_mult = text_cfg.hc_mult
        self.embed = embed
        self.head = head
        n = text_cfg.num_nextn_predict_layers
        self.mtp = nn.ModuleList([
            DSparkBlock(text_cfg, text_cfg.num_hidden_layers + i, i) for i in range(n)])

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        """把 target hidden 投成 draft 输入，并铺 block_size 个 noise token（首位置为 seed）。"""
        first = self.mtp[0]
        main_x = first.main_norm(first.main_proj(main_hidden))
        draft_ids = input_ids.new_full((input_ids.size(0), self.block_size),
                                       self.cfg.dspark_noise_token_id)
        draft_ids[:, 0] = input_ids
        x = self.embed(draft_ids)
        return x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1), main_x

    def forward_head(self, x, pre_mix, input_ids, temperature: float = 0.0):
        """末块出草稿：base logits + markov 偏置 → 自回归采样；confidence head 出接受概率。

        temperature=0 为贪心（argmax）；>0 用 Gumbel-max 采样（参考实现 sample 同款）。
        """
        last = self.mtp[-1]
        x = last.hc.hc_pre(x, pre_mix)
        base_logits = self.head(last.norm(x))                  # [B, block, V]
        # 逐位置用 markov 偏置修正并采样；id 用列表收集（避免原地写入破坏计算图）
        ids_list = [input_ids]
        logits_list, markov_embeds = [], []
        for i in range(self.block_size):
            bias, me = last.markov_head(ids_list[i])
            step_logits = base_logits[:, i] + bias             # 非原地
            logits_list.append(step_logits)
            markov_embeds.append(me)
            ids_list.append(_sample(step_logits, temperature))
        logits = torch.stack(logits_list, dim=1)               # [B, block, V]
        output_ids = torch.stack(ids_list, dim=1)              # [B, block+1]
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = last.confidence_head(x, markov_embed)
        return output_ids, logits, confidence

    def forward(self, main_hidden, input_ids, start_pos: int = 0, temperature: float = 0.0):
        """draft 前向（prefill 只填缓存，decode 出草稿）。

        输入：main_hidden [B, S, D]；input_ids [B]；start_pos；temperature（0=贪心）。
        输出：start_pos==0 时 None；否则 (output_ids, logits, confidence)。
        """
        x, main_x = self.forward_embed(main_hidden, input_ids)
        pre_mix = x.new_zeros(x.size(0), x.size(1), self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        for layer in self.mtp:
            x, pre_mix = layer(x, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.forward_head(x, pre_mix, input_ids, temperature)

    def forward_train(self, input_ids, context_hidden, loss_mask, aligned_target_logits=None):
        """训练前向：teacher-forced 并行（按参考实现 Qwen3DSparkModel.forward）。

        输入：input_ids [B,S]；context_hidden [B,S,D]（主干 target 层 hidden 拼接）；
              loss_mask [B,S]（1=可算 loss）；aligned_target_logits [B,S,V]（主干对每位置的 logits，
              用于 L1 蒸馏与 accept_rate；None 时关闭 L1/confidence）。
        输出：DSparkForwardOutput（见 model/train/dspark_loss.py）。
        预期行为：采 anchor → 噪声嵌入 → 并行 draft 前向 → markov 偏置 → 组 logits 与目标 id。
        """
        from model.train.dspark_loss import DSparkForwardOutput

        bsz, seq_len = input_ids.shape
        device = input_ids.device
        K = self.block_size
        first = self.mtp[0]

        anchors, keep = sample_anchor_positions(seq_len, loss_mask, self.cfg.dspark_num_anchors, device)
        num_blocks = anchors.size(1)

        # 噪声嵌入：每块首位置为 anchor token，其余为 noise_token
        noise_ids = input_ids.new_full((bsz, num_blocks * K), self.cfg.dspark_noise_token_id)
        block_starts = (torch.arange(num_blocks, device=device) * K).unsqueeze(0).expand(bsz, -1)
        anchor_tokens = torch.gather(input_ids, 1, anchors)
        batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, num_blocks)
        noise_ids[batch_idx, block_starts] = torch.where(keep, anchor_tokens,
                                                         torch.full_like(anchor_tokens, self.cfg.dspark_noise_token_id))
        x = self.embed(noise_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        # 位置 id：context 0..S-1，draft 为 anchor + 块内偏移
        offsets = torch.arange(K, device=device).view(1, 1, -1)
        position_ids = (anchors.unsqueeze(-1) + offsets).reshape(bsz, num_blocks * K)

        pre_mix = x.new_zeros(bsz, num_blocks * K, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        ctx = first.main_norm(first.main_proj(context_hidden))
        for layer in self.mtp:
            x, pre_mix = layer.forward_train(x, pre_mix, ctx, position_ids, anchors, keep, seq_len)

        # 目标 id：anchor 后 1..K 个 token
        label_offsets = torch.arange(1, K + 1, device=device).view(1, 1, -1)
        label_indices = anchors.unsqueeze(-1) + label_offsets
        safe_label = label_indices.clamp(max=seq_len - 1)
        safe_label = torch.where(keep.unsqueeze(-1), safe_label, torch.zeros_like(safe_label))
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_label)

        last = self.mtp[-1]
        h = last.hc.hc_pre(x, pre_mix)                                  # [B,Q,dim] 折叠后
        hidden = last.norm(h)                                          # 参考实现：logits 与 confidence 都用归一后的 hidden
        h4 = hidden.reshape(bsz, num_blocks, K, -1)
        base_logits = self.head(hidden).reshape(bsz, num_blocks, K, -1)

        # teacher-forced markov 偏置：位置 k 用真实前驱 token（anchor 或 target[:,k-1]）
        anchor_ids = torch.gather(input_ids, 1, anchors)
        prev_ids = torch.cat([anchor_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1)
        base_logits = last.markov_head.apply_block_logits(
            base_logits, token_ids=prev_ids, hidden_states=h4)

        eval_mask = build_eval_mask(seq_len, loss_mask, label_indices, safe_label, keep)

        conf = None
        if last.confidence_head is not None:
            me = last.markov_head.get_prev_embeddings(prev_ids).to(hidden.dtype)
            conf = last.confidence_head(h4, me)

        # aligned target logits：位置 s 的目标分布用主干在 s-1 的 logits
        aligned = None
        if aligned_target_logits is not None:
            pred_idx = (safe_label - 1).clamp(min=0)
            aligned = torch.gather(
                aligned_target_logits.unsqueeze(1).expand(-1, num_blocks, -1, -1), 2,
                pred_idx.unsqueeze(-1).expand(-1, -1, -1, aligned_target_logits.size(-1)))

        return DSparkForwardOutput(
            draft_logits=base_logits, target_ids=target_ids, eval_mask=eval_mask,
            block_keep_mask=keep, confidence_pred=conf, aligned_target_logits=aligned)
