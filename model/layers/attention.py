"""MLA + CSA2 + 跨层共享。

每层对两个 KV 源做稀疏注意力并拼接：滑窗原始 KV + （compress_ratio>0 时）index_topk 个压缩位置。

跨层共享：只有 kv_source 层用自己的输入算压缩 KV 并发布；后续层**直接读同一份缓存**（无投影）。
同理 index 源层发布 topk 索引，后续层复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.compressor import Compressor
from model.layers.indexer import Indexer, IndexerLossAutoScaler, select_candidate_blocks
from model.nn.norm import RMSNorm
from model.nn.quant import fake_quant_compressed_kv, fake_quant_fp4, fake_quant_fp8
from model.nn.rope import apply_rotary_emb


@dataclass
class SharedAttentionRuntime:
    """跨层共享槽（同一份张量被源层写、后续层读）。

    compress_kv：源层发布的压缩 KV 缓存
    index_k：源层发布的 index key 缓存
    topk_idxs：index 源发布的稀疏索引
    candidates：候选块 mask
    """

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int,
                         device=None) -> torch.Tensor:
    """每个 query 可见的滑窗缓存槽位；-1 表示空槽。

    输入：window_size；bsz；seqlen；start_pos；device（**直接在目标设备建张量，避免 CPU→GPU 同步**）。
    输出：[bsz, seqlen, min(seqlen,window_size)]（prefill）或 [bsz,1,window_size]（decode），int32。
    预期行为：缓存是 window_size 个槽的环形缓冲。prefill 每 query 一行、各自因果窗口；
              decode 单 query 看整个环，最旧在前。
    """
    if start_pos == 0:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        oldest = start_pos % window_size + 1
        idxs = torch.cat([torch.arange(oldest, window_size, device=device),
                          torch.arange(oldest, device=device)])
        idxs = torch.where(idxs > start_pos, -1, idxs)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Attention(nn.Module):
    """一层 MLA 注意力（滑窗 + 压缩 KV 拼接）。

    输入：x [B, L, dim]；start_pos；shared（跨层共享槽）。
    输出：同形状张量。
    预期行为：Q 走低秩投影 + RoPE；滑窗 KV 每层自算；压缩 KV 由 kv_source 层发布、
              其余层读同一缓存；O 走分组低秩投影。
    """

    def __init__(self, text_cfg, layer_id: int, is_backbone: bool = True):
        super().__init__()
        self.layer_id = layer_id
        self.dim = text_cfg.hidden_size
        self.n_heads = text_cfg.num_attention_heads
        self.q_lora_rank = text_cfg.q_lora_rank
        self.o_lora_rank = text_cfg.o_lora_rank
        self.head_dim = text_cfg.head_dim
        self.rope_head_dim = text_cfg.qk_rope_head_dim
        self.n_groups = text_cfg.o_groups
        self.window_size = text_cfg.sliding_window
        # 超出主干层号的（DSpark draft 层）默认纯 SWA（ratio=0，同样结构）
        cr = text_cfg.compress_ratios
        self.compress_ratio = cr[layer_id] if layer_id < len(cr) else 0
        self.eps = text_cfg.rms_norm_eps
        self.softmax_scale = self.head_dim**-0.5

        # attn sink 初值置零（参考实现 checkpoint 载入前无意义；torch.empty 脏内存会污染 softmax）
        self.attn_sink = nn.Parameter(torch.zeros(self.n_heads, dtype=torch.float32))
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups,
                              self.n_groups * self.o_lora_rank, bias=False, dtype=torch.bfloat16)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)

        self.is_kv_source = is_backbone and layer_id in text_cfg.kv_source_layer_ids
        self.is_index_source = is_backbone and layer_id in text_cfg.index_source_layer_ids
        self.is_candidate_source = layer_id == text_cfg.candidate_source_layer_id
        self.uses_candidates = 0 <= text_cfg.candidate_source_layer_id < layer_id

        self.compressor = Compressor(text_cfg, layer_id) if self.is_kv_source else None
        self.indexer = Indexer(text_cfg, layer_id) if self.is_index_source else None

        max_batch = getattr(text_cfg, "max_batch_size", 1)
        max_seq = text_cfg.max_position_embeddings
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(max_batch, self.window_size, self.head_dim),
            persistent=False,
        )
        if self.is_kv_source:
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(max_batch, max_seq // self.compress_ratio, self.head_dim),
                persistent=False,
            )
        if self.is_index_source and self.is_kv_source:
            self.register_buffer(
                "index_k_cache",
                torch.zeros(max_batch, max_seq // self.compress_ratio, text_cfg.index_head_dim),
                persistent=False,
            )

        if self.compress_ratio:
            rope_base = text_cfg.compress_rope_theta
            original_seq_len = getattr(text_cfg, "original_seq_len", 0)
        else:
            # 纯滑窗层：禁用 YaRN，用 base rope_theta（同样结构）
            rope_base = text_cfg.rope_theta
            original_seq_len = 0
        self.register_buffer(
            "freqs_cis",
            _precompute(text_cfg, rope_base, original_seq_len),
            persistent=False,
        )

    def _ensure_cache(self, bsz: int):
        """按需扩容 KV / index 缓存到能容纳 bsz 个样本。

        输入：bsz 当前 batch 大小。
        输出：无（必要时重建 buffer）。
        预期行为：缓存首维 >= bsz 时不动；否则重建为零张量。
        """
        if self.window_kv_cache.size(0) >= bsz:
            return
        dev = self.window_kv_cache.device
        self.window_kv_cache = torch.zeros(bsz, self.window_size, self.head_dim, device=dev)
        if self.is_kv_source:
            self.compress_kv_cache = torch.zeros(
                bsz, self.compress_kv_cache.size(1), self.head_dim, device=dev)
        if self.is_index_source and self.is_kv_source:
            self.index_k_cache = torch.zeros(
                bsz, self.index_k_cache.size(1), self.index_k_cache.size(2), device=dev)

    def reset_cache(self):
        """清空本层的 KV / index 缓存（新序列开始时调用）。

        输入：无。
        输出：无。
        预期行为：window_kv_cache 归零；kv_source 层的 compress_kv_cache / index_k_cache 归零。
        """
        self.window_kv_cache.zero_()
        if self.is_kv_source:
            self.compress_kv_cache.zero_()
        if self.is_index_source and self.is_kv_source:
            self.index_k_cache.zero_()
        if self.compressor is not None:
            self.compressor.reset_state()

    def forward(self, x: torch.Tensor, start_pos: int, shared: SharedAttentionRuntime) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        self._ensure_cache(bsz)
        rd = self.rope_head_dim
        freqs = self.freqs_cis[start_pos:start_pos + seqlen]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary_emb(q, freqs, rd)
        # 滑窗 KV（每层自算）；参考实现对 KV 做 FP8 量化（act_quant, block=32, pow2）
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rotary_emb(kv, freqs, rd)
        kv = fake_quant_fp8(kv, block_size=32, pow2=True)
        win = self.window_size
        if start_pos == 0:
            # 训练/prefill 直接用 kv（不写缓存，避免破坏计算图）；decode 才需要缓存
            if seqlen <= win:
                with torch.no_grad():
                    self.window_kv_cache[:bsz, :seqlen] = kv.detach()
            else:
                cutoff = seqlen % win
                with torch.no_grad():
                    self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = \
                        kv[:, -win:].detach().split([win - cutoff, cutoff], dim=1)
            window_kv = kv
        else:
            self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            window_kv = self.window_kv_cache[:bsz]
        win_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device)

        kv_all = window_kv
        idxs_all = win_idxs
        if self.compress_ratio:
            # 压缩位置索引以窗口实际长度为偏移（kv_all = 窗口 ‖ 压缩）
            comp_kv, comp_idxs = self._compress_kv(x, qr, start_pos, window_kv.size(1), shared)
            kv_all = torch.cat([kv_all, comp_kv], dim=1)
            idxs_all = torch.cat([idxs_all, comp_idxs], dim=-1)

        o = self._sparse_attn(q, kv_all, idxs_all)
        o = apply_rotary_emb(o, freqs, rd, inverse=True)
        # indexer 训练：让 indexer 打分对齐主注意力对压缩位置的分布（DSA 做法）。
        # indexer 的 topk 选择不可微，故用单独的 KL loss 驱动其参数，经 AutoScaler 挂到输出。
        # 注意：挂在 rope 输出**之后**——custom Function 的输出不能再被后续原地操作改写。
        if (self.is_index_source and self.indexer is not None
                and getattr(self.indexer, "_last_score", None) is not None
                and self.compress_ratio and start_pos == 0):
            comp_attn = kv_all[:, window_kv.size(1):]          # [B,T,D] 压缩部分
            score = self.indexer._last_score                   # [B,L,T]
            if comp_attn.size(1) > 0 and comp_attn.size(1) == score.size(-1):
                qd = q.detach().float()
                kd = comp_attn.detach().float()
                s = torch.einsum("blhd,btd->blht", qd, kd) * self.softmax_scale
                main_dist = s.mean(dim=2).softmax(dim=-1)      # [B,L,T] 主注意力分布
                idx_loss = self.indexer.indexer_loss(main_dist, score, idxs_all)
                o = IndexerLossAutoScaler.apply(o, idx_loss)

        o = o.reshape(bsz, seqlen, self.n_groups, -1).to(self.wo_a.weight.dtype)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2).to(self.wo_b.weight.dtype))

    def _sparse_attn(self, q: torch.Tensor, kv: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
        """按 idxs 选中的 KV 做因果稀疏注意力。

        输入：q [B,L,H,D]；kv [B,T,D]（单 KV 头）；idxs [B,L,K]（-1 表示无效）。
        输出：[B,L,H,D]。
        预期行为：对每个 query 只对其 idxs 指向的 KV 做 softmax 注意力；
              -1 位置被屏蔽；拼接 attn_sink 参与 softmax。
        """
        b, L, H, D = q.shape
        T = kv.size(1)
        qf = q.float()
        kf = kv.float()
        # gather KV: [B,L,K,D]
        valid = idxs >= 0
        safe = idxs.clamp_min(0)
        k_sel = kf.unsqueeze(1).expand(b, L, T, D).gather(2, safe.unsqueeze(-1).expand(-1, -1, -1, D))
        scores = torch.einsum("blhd,blkd->blhk", qf, k_sel) * self.softmax_scale
        # 参考实现用有限下界 -1e30（非 -inf）：全 -1 行会得到 exp(-inf-(-inf))=NaN，
        # 有限下界下该行输出全 0，与训练 kernel 约定一致。
        scores = scores.masked_fill(~valid.unsqueeze(2), -1e30)
        # attn sink：每个 head 一个额外 logit
        sink = self.attn_sink.view(1, 1, H, 1).expand(b, L, H, 1)
        scores = torch.cat([scores, sink], dim=-1)
        probs = scores.softmax(dim=-1)[..., :-1]
        out = torch.einsum("blhk,blkd->blhd", probs, k_sel)
        return out.to(q.dtype)

    def _compress_kv(self, x, qr, start_pos, offset, shared):
        """压缩 KV + 压缩位置索引（跨层共享 共享在此发生）。

        prefill（训练）：源层算出的 latent 直接作为共享张量发布（带梯度，供后续层读）。
        decode：写入缓存供跨步复用。
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            # 参考实现：源层把**整个共享 cache** 发布给后续层（非源层按自己的 compress_len 截取）。
            # 推理走此路（逐位一致）；训练用 latent 保留梯度（见下）。
            if not self.training:
                shared.compress_kv = self.compress_kv_cache
        # 参考实现：shared_attn.index_k 只在「本层 owns_k 且 latent 非 None」时发布
        # （Indexer.forward 内 if self.owns_k and latent is not None）。组未满的 decode
        # 非压缩步 latent=None 不发布，此处读到的是上一层 kv_source 遗留的槽——按参考实现。
        # prefill 时每个 query 可见的压缩位置数不同（query t 只能看 (t+1)//ratio 个）
        if start_pos == 0:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        else:
            compress_lens = compress_len

        # index 源层从自己的 latent 派生 index key（owns_k），供后续层复用
        if self.is_index_source and latent is not None and self.is_kv_source:
            rd = self.rope_head_dim
            k = self.indexer.k_norm(self.indexer.wk(latent))
            if start_pos == 0:
                freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            else:
                freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            k = apply_rotary_emb(k, freqs, rd)
            # 参考实现：index k 写缓存前做 FP4 量化（fp4_act_quant, block=32, E8M0 scale）
            k = fake_quant_fp4(k, block_size=32)
            if self.training:
                shared.index_k = k          # 训练：带梯度共享（张量）
            else:
                # 推理：写缓存并发布**整个缓存**（按参考实现 shared_attn.index_k = self.k_cache）；
                # 后续 indexer 按 end_pos//ratio 切片，故缓存长度须为 max_seq//ratio。
                self.index_k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
                shared.index_k = self.index_k_cache

        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_lens, shared)
        if latent is not None:
            rd = self.rope_head_dim
            if start_pos == 0:
                freqs = self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            else:
                freqs = self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
            latent = apply_rotary_emb(latent, freqs, rd)
            # 参考实现对压缩 latent 做 FP4 量化（fp4_act_quant, block=16, E4M3 scale）
            latent = fake_quant_compressed_kv(latent, block_size=16)
            if self.training:
                shared.compress_kv = latent   # 训练：带梯度共享给后续层
            else:
                # 推理：写进 cache（shared.compress_kv 已指向 cache，按参考实现）
                self.compress_kv_cache[:bsz, start_pos // ratio : start_pos // ratio + latent.size(1)] = latent
        if shared.compress_kv is None:  # 尚无源层发布（如单独调用本层）
            return x.new_zeros(bsz, 0, self.head_dim), idxs
        return shared.compress_kv[:bsz, :compress_len], idxs

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_lens, shared):
        """压缩位置的 topk 索引；非 index 源层复用共享结果。

        compress_lens：prefill 时逐 query [L,1]，decode 时标量。
        """
        bsz, seqlen, _ = x.size()
        if not self.is_index_source:
            if shared.topk_idxs is None:  # 尚无源层发布 → 本层无压缩位置可看
                return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
            return shared.topk_idxs
        # 参考实现：index 源层无条件读共享槽 shared_attn.index_k（Indexer.forward 的
        # `index_k = shared_attn.index_k[:bsz, : end_pos // ratio]`）。该槽只由「owns_k 且
        # latent 非 None」的层更新（见 _compress_kv），所以组未满的 decode 非压缩步读到的是
        # 上一层 kv_source 遗留的缓存——猫猫照抄此语义，不直读自己的缓存。
        end_pos = start_pos + seqlen
        index_k = shared.index_k
        if index_k is None:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            # 参考实现：indexer 的 q 用 attention 层的 freqs_cis（Indexer.freqs_cis = Attention.freqs_cis）
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            # 参考实现：index_k 按本层自己的 end_pos//ratio 切片（非源层读的缓存比它自己的压缩位置多）
            index_k = index_k[:bsz, : end_pos // self.compress_ratio]
            idxs = self.indexer(x, qr, index_k, compress_lens,
                                offset, start_pos, shared)
        shared.topk_idxs = idxs
        return idxs


def _precompute(text_cfg, rope_base, original_seq_len=0):
    from model.nn.rope import precompute_freqs_cis

    return precompute_freqs_cis(
        text_cfg.qk_rope_head_dim, text_cfg.max_position_embeddings, rope_base,
        original_seq_len=original_seq_len,
        factor=getattr(text_cfg, "rope_factor", 40.0),
        beta_fast=getattr(text_cfg, "beta_fast", 32),
        beta_slow=getattr(text_cfg, "beta_slow", 1))
