"""DSpark 投机解码（按参考实现 DSparkBlock / DSparkAttention 结构）。

参考实现（参考模型的 DSpark* 部分）：
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
from model.nn.linear import Linear
from model.nn.norm import RMSNorm
from model.nn.quant import fake_quant_fp8
from model.nn.rope import apply_rotary_emb


@lru_cache(1)
def get_dspark_topk_idxs(window_size: int, bsz: int, block_size: int, start_pos: int):
    """draft block 每个 query 可见的 KV 槽位（按参考实现 get_dspark_topk_idxs）。

    输入：window_size；bsz；block_size；start_pos（>0）。
    输出：[bsz, block_size, min(win, start_pos+1)+block_size]，int32。
    预期行为：每个 draft query 看 [窗口内历史槽 … | 本块 block_size 个槽]。
    """
    assert start_pos > 0
    matrix = torch.cat([
        torch.arange(min(window_size, start_pos + 1)),
        window_size + torch.arange(block_size),
    ])
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    """draft stage 的注意力：main_x 写窗口 KV，draft block 并行 attend（按参考实现 DSparkAttention）。"""

    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor) -> torch.Tensor:
        assert self.compress_ratio == 0, "DSpark 层必须是纯 SWA（compress_ratio=0）"
        bsz, seqlen, _ = main_x.size()
        win, rd = self.window_size, self.rope_head_dim

        # main_x（主干 target hidden）→ 窗口 KV；参考实现对 KV 做 FP8 量化
        main_freqs = self.freqs_cis[start_pos:start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs)
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
        apply_rotary_emb(q[..., -rd:], freqs)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs)
        kv = fake_quant_fp8(kv, block_size=32, pow2=True)

        topk_idxs = get_dspark_topk_idxs(win, bsz, block_size, start_pos).to(x.device)
        with torch.no_grad():
            self.window_kv_cache[:bsz, start_pos % win] = main_kv.squeeze(1).detach()
        kv = torch.cat([self.window_kv_cache[:bsz], kv], dim=1)

        o = self._sparse_attn(q, kv, topk_idxs)
        apply_rotary_emb(o[..., -rd:], freqs, inverse=True)
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
        self.head = Linear(rank, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor):
        embed = self.embed(token_ids)
        return self.head(embed), embed


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
        self.ffn = MoE(text_cfg)
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

    def forward_head(self, x, pre_mix, input_ids):
        """末块出草稿：base logits + markov 偏置 → 自回归采样；confidence head 出接受概率。"""
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
            ids_list.append(step_logits.argmax(dim=-1))
        logits = torch.stack(logits_list, dim=1)               # [B, block, V]
        output_ids = torch.stack(ids_list, dim=1)              # [B, block+1]
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = last.confidence_head(x, markov_embed)
        return output_ids, logits, confidence

    def forward(self, main_hidden, input_ids, start_pos: int = 0):
        """draft 前向（prefill 只填缓存，decode 出草稿）。

        输入：main_hidden [B, S, D]；input_ids [B]；start_pos。
        输出：start_pos==0 时 None；否则 (output_ids, logits, confidence)。
        """
        x, main_x = self.forward_embed(main_hidden, input_ids)
        pre_mix = x.new_zeros(x.size(0), x.size(1), self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0
        for layer in self.mtp:
            x, pre_mix = layer(x, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.forward_head(x, pre_mix, input_ids)
