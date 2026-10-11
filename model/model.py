"""miniwhale 模型组装。

embed → 复制成 hc_mult 份残差流 → N 层 Block（含 Engram 注入）→ 折叠 → RMSNorm → LM Head。

单个 Block：attention 与 FFN 各夹在 hc_pre（折叠成 1 份喂子层）与 hc_post（展开回 hc 份、
用 comb 混残差）之间。子层产出的 pre_mix 供**下一个**子层用（single-pass 错位）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.layers.attention import Attention, SharedAttentionRuntime
from model.layers.engram import Engram
from model.layers.engram_hash import EngramHash
from model.nn.linear import Head, Linear
from model.layers.mhc import HyperConnection
from model.layers.moe import MoE
from model.nn.norm import RMSNorm


class Block(nn.Module):
    """一层：mHC 残差流 + 注意力 + MoE（+ 可选 Engram）。

    输入：x [B, L, hc_mult, dim]；start_pos；pre_mix [B, L, hc_mult]；shared；engram_hash_ids。
    输出：(x [B, L, hc_mult, dim], ffn_pre [B, L, hc_mult])。
    预期行为：attn 用上一层传的 pre_mix，FFN 用本层 attn 产的 pre_mix（错位）。
    """

    def __init__(self, text_cfg, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn = Attention(text_cfg, layer_id)
        self.ffn = MoE(text_cfg)
        self.attn_norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        self.ffn_norm = RMSNorm(text_cfg.hidden_size, text_cfg.rms_norm_eps)
        self.hc = HyperConnection(text_cfg)
        self.engram = Engram(text_cfg, layer_id) if layer_id in text_cfg.engram_layer_ids else None

    def forward(self, x, start_pos, pre_mix, *attn_args, engram_hashes=None, token_mask=None,
                image_mask=None):
        """attn_args 透传给 attention（普通块传 shared 运行时；DSpark 块传 main_x）。

        输入：x [B,L,hc,dim]；start_pos；pre_mix；attn_args；engram_hashes；token_mask；
              image_mask [B,L]（True=图像 span token，MoE 用单独路由 bias）。
        输出：(x, ffn_pre)。
        预期行为：attn 用上一层传的 pre_mix，FFN 用本层 attn 产的 pre_mix（single-pass 错位）。
        """
        if self.engram is not None and engram_hashes is not None:
            x = self.engram(x, engram_hashes, token_mask)
        residual = x
        attn_pre, attn_post, attn_comb = self.hc.hc_mixes(x, "attn")
        h = self.hc.hc_pre(x, pre_mix)
        h = self.attn_norm(h)
        h = self.attn(h, start_pos, *attn_args)
        x = self.hc.hc_post(h, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(x, "ffn")
        h = self.hc.hc_pre(x, attn_pre)
        h = self.ffn_norm(h)
        h = self.ffn(h, image_mask)
        x = self.hc.hc_post(h, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class Model(nn.Module):
    """miniwhale 主干。

    输入：input_ids [B, L]（6400 词表）；start_pos。
    输出：(logits [B, L, vocab], hidden [B, L, dim])。
    预期行为：Engram 哈希在层间共享（按层索引取），attention 的跨层状态经 shared 传递。
    """

    def __init__(self, cfg, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        tc = cfg.text
        self.hc_mult = tc.hc_mult
        self.embed = nn.Embedding(tc.vocab_size, tc.hidden_size)
        self.layers = nn.ModuleList([Block(tc, i) for i in range(tc.num_hidden_layers)])
        self.norm = RMSNorm(tc.hidden_size, tc.rms_norm_eps)
        self.head = Head(tc.hidden_size, tc.vocab_size)
        has_engram = bool(tc.engram_layer_ids) and len(tc.engram_num_embeddings) > 0
        self.engram_hash = EngramHash(tc, tokenizer) if (tokenizer is not None and has_engram) else None
        self.shared = SharedAttentionRuntime()

        # DSpark draft 头（参考实现：主干预训练不含它，单独阶段训练）
        self.target_layer_ids = tuple(tc.dspark_target_layer_ids)
        self.dspark = None
        if tc.num_nextn_predict_layers > 0 and self.target_layer_ids:
            from model.layers.dspark import DSparkHead
            # 参考实现：draft 共享主干的 embed / head（权重同一份）
            self.dspark = DSparkHead(tc, self.embed, self.head)

    def forward(self, input_ids, start_pos=0, token_mask=None, image_embeds=None):
        """主干前向。

        输入：input_ids [B,L]；start_pos；token_mask [B,L]（False=图像位置，Engram 跳过）；
              image_embeds list[Tensor]（每个 [n_tokens, dim]），非 None 时覆盖图像占位位置的 embedding。
        输出：(logits [B,L,V], hidden [B,L,dim])。
        预期行为：先把 `<image>` 位置的 embedding 替换为视觉特征，再走主干。
        """
        tc = self.cfg.text
        b, L = input_ids.shape
        # 每次前向重置跨层共享槽（避免复用上一次前向的张量）
        self.shared = SharedAttentionRuntime()

        engram_hashes = None
        if self.engram_hash is not None:
            engram_hashes = self.engram_hash(input_ids, token_mask)  # [B,L,n_layers,n_hash_cols]

        # 图像 span 掩码：MoE 对图像 token 用单独路由 bias（参考实现 noaux_tc_for_vl）
        img_id = getattr(tc, "image_token_id", None)
        image_mask = (input_ids == img_id) if (img_id is not None and image_embeds is not None) else None

        h = self.embed(input_ids)                       # [B,L,dim]
        if image_embeds is not None:
            h = self.merge_image_embeds(input_ids, h, image_embeds)
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()  # [B,L,hc,dim]

        # 初始 one-hot pre_mix（第 0 份复制为 1）
        pre_mix = h.new_zeros(b, L, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        main_hiddens = []  # DSpark 读目标层的注意力输入
        for i, layer in enumerate(self.layers):
            eh = None
            if engram_hashes is not None:
                idx = tuple(tc.engram_layer_ids).index(i) if i in tc.engram_layer_ids else None
                if idx is not None:
                    eh = engram_hashes[:, :, idx, :]
            # 参考实现：在层处理**之前**取 hc 副本的均值（MTP 读的是注意力输入）
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(h, start_pos, pre_mix, self.shared,
                               engram_hashes=eh, token_mask=token_mask, image_mask=image_mask)

        h = self.layers[-1].hc.hc_pre(h, pre_mix)  # 折叠成 1 份
        hidden = self.norm(h)
        logits = self.head(hidden)
        self._last_main_hidden = (
            torch.cat(main_hiddens, dim=-1) if main_hiddens else None)
        return logits, hidden

    def forward_spec(self, input_ids: torch.Tensor, start_pos: int = 0):
        """DSpark draft 前向（参考实现 forward_spec 同构）。

        输入：input_ids [B]（seed token）；start_pos。
        输出：start_pos==0 时 None；否则 (output_ids [B, block+1], logits [B, block, V], confidence [B, block])。
        预期行为：用主干 target 层 hidden 投影出 draft 输入，逐 stage 前向出草稿 + 置信度。
        """
        assert self.dspark is not None, "模型未启用 DSpark"
        main_hidden = getattr(self, "_last_main_hidden", None)
        assert main_hidden is not None, "先调用 forward 以获得主干 target 层 hidden"
        # decode（start_pos>0）：主干只对上一步 token 前向，取最后一个位置的 hidden
        if start_pos > 0:
            main_hidden = main_hidden[:, -1:]
        return self.dspark(main_hidden, input_ids, start_pos)

    def merge_image_embeds(self, input_ids, h, image_embeds):
        """把视觉特征覆盖到 `<image>` 占位位置。

        输入：input_ids [B,L]；h [B,L,dim]；image_embeds list[Tensor]（每个 [n_tokens,dim]）。
        输出：新的 h [B,L,dim]。
        预期行为：找到每个样本的 image_token_id 连续段，把对应的视觉 token 按顺序写入。
        """
        tc = self.cfg.text
        img_id = getattr(tc, "image_token_id", None)
        if img_id is None:
            return h
        h = h.clone()
        for i, emb in enumerate(image_embeds):
            if emb is None:
                continue
            pos = (input_ids[i] == img_id).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            n = min(pos.numel(), emb.size(0))
            h[i, pos[:n]] = emb[:n].to(h.dtype)
        return h

    @torch.no_grad()
    def update_router_bias(self, rate: float = 1e-3):
        """更新所有 MoE 层的纠正 bias（负载均衡，非梯度）。

        输入：rate 更新步长。
        输出：无。
        预期行为：遍历每层 ffn.gate，按 tokens_per_expert 调整 bias（参考实现 noaux_tc）。
        """
        for layer in self.layers:
            gate = getattr(layer.ffn, "gate", None)
            if gate is not None:
                gate.update_bias(rate)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens: int = 16, temperature: float = 0.0):
        """自回归生成（贪心或采样）。

        输入：input_ids [B, L0]；max_new_tokens；temperature（0=贪心）。
        输出：[B, L0+max_new_tokens]。
        预期行为：逐 token 前向，取最后一个位置的 logits 采样，拼到序列尾。
        """
        self.eval()
        ids = input_ids
        for _ in range(max_new_tokens):
            logits, _ = self.forward(ids)          # 每次前向重置 shared，重新算
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                nxt = next_logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(next_logits.softmax(-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
        return ids
