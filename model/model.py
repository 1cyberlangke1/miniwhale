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

    def forward(self, x, start_pos, pre_mix, *attn_args, image_mask=None):
        """attn_args 透传给 attention（普通块传 shared 运行时；DSpark 块传 main_x）。

        输入：x [B,L,hc,dim]；start_pos；pre_mix；attn_args；image_mask [B,L]。
        输出：(x, ffn_pre)。
        预期行为：attn 用上一层传的 pre_mix，FFN 用本层 attn 产的 pre_mix（single-pass 错位）。
                  Engram 注入由 Model 在调本块**之前**完成（按参考实现）。
        """
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

        # 视觉塔 + Aligner + 图像 span 的分隔符嵌入（参考实现 Transformer 同款）
        self.vision = None
        self.aligner = None
        if getattr(tc, "vision_enabled", False):
            from model.layers.vision import Aligner, ViT
            self.vision = ViT(cfg.vision)
            self.aligner = Aligner(cfg.vision, tc.hidden_size)
            # 图像分隔符嵌入：正态小值初始化（参考实现 checkpoint 载入前 torch.empty 会留脏内存）
            self.image_start = nn.Parameter(torch.randn(tc.hidden_size) * 0.02)
            self.image_end = nn.Parameter(torch.randn(tc.hidden_size) * 0.02)
            self.image_newline = nn.Parameter(torch.randn(tc.hidden_size) * 0.02)

    def forward(self, input_ids, start_pos=0, token_mask=None, images=None, token_types=None,
                full_logits=True):
        """主干前向（按参考实现 Transformer.forward 的 VL 接口）。

        输入：input_ids [B,L]；start_pos；token_mask [B,L]（False=图像位置，Engram 跳过）；
              images list[list[ImageInput]]（每样本的图像 span，参考实现 prepare_vl_inputs 产物）；
              token_types [B,L]（TEXT=-1，非负为图像位置类型）；
              full_logits=False 时只算末位置（按参考实现 ParallelHead，推理/对拍用；训练需全序列）。
        输出：(logits [B,L,V] 或 full_logits=False 时 [B,V], hidden [B,L,dim])。
        预期行为：图像 span 的占位位置被替换为 ViT/Aligner 特征与分隔符嵌入，再走主干。
        """
        tc = self.cfg.text
        b, L = input_ids.shape
        # 跨层共享槽**跨前向持久**（按参考实现：shared_attn 是模块级全局，forward 之间不重置）。
        # 源层每步都会在自己消费者之前写入，故无需重置；decode 的非压缩步正是靠上一前向
        # 遗留的槽（如 index_k）才能逐位一致——重置会破坏这一语义。

        # 图像 span 掩码：参考实现 image_mask = token_types >= 0（TEXT=-1）
        image_mask = None if token_types is None else (token_types >= 0)
        # Engram 跳过图像 token：参考实现 engram_mask = ~image_mask
        engram_mask = token_mask
        if engram_mask is None and image_mask is not None:
            engram_mask = ~image_mask

        engram_hashes = None
        if self.engram_hash is not None:
            engram_hashes = self.engram_hash(input_ids, engram_mask)  # [B,L,n_layers,n_hash_cols]

        h = self.embed(input_ids)                       # [B,L,dim]
        if images is not None:
            h = self.merge_image_embeddings(images, h)
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()  # [B,L,hc,dim]

        # 初始 one-hot pre_mix（第 0 份复制为 1）
        pre_mix = h.new_zeros(b, L, self.hc_mult, dtype=torch.float32)
        pre_mix[:, :, 0] = 1.0

        main_hiddens = []  # DSpark 读目标层的注意力输入
        for i, layer in enumerate(self.layers):
            # 参考实现：Engram 注入在层处理**之前**（且先于 main_hiddens 取值）
            if layer.engram is not None and engram_hashes is not None:
                idx = tuple(tc.engram_layer_ids).index(i)
                h = layer.engram(h, engram_hashes[:, :, idx, :], engram_mask)
            # 参考实现：在层处理**之前**取 hc 副本的均值（MTP 读的是注意力输入）
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(h, start_pos, pre_mix, self.shared, image_mask=image_mask)

        h = self.layers[-1].hc.hc_pre(h, pre_mix)  # 折叠成 1 份
        hidden = self.norm(h)
        logits = self.head(hidden, full_logits=full_logits)
        self._last_main_hidden = (
            torch.cat(main_hiddens, dim=-1) if main_hiddens else None)
        return logits, hidden

    def forward_spec(self, input_ids: torch.Tensor, start_pos: int = 0, temperature: float = 0.0):
        """DSpark draft 前向（参考实现 forward_spec 同构）。

        输入：input_ids [B]（seed token）；start_pos；temperature（0=贪心）。
        输出：start_pos==0 时 None；否则 (output_ids [B, block+1], logits [B, block, V], confidence [B, block])。
        预期行为：用主干 target 层 hidden 投影出 draft 输入，逐 stage 前向出草稿 + 置信度。
        """
        assert self.dspark is not None, "模型未启用 DSpark"
        main_hidden = getattr(self, "_last_main_hidden", None)
        assert main_hidden is not None, "先调用 forward 以获得主干 target 层 hidden"
        # decode（start_pos>0）：主干只对上一步 token 前向，取最后一个位置的 hidden
        if start_pos > 0:
            main_hidden = main_hidden[:, -1:]
        return self.dspark(main_hidden, input_ids, start_pos, temperature)

    def encode_image(self, patches, n_vit_h: int, n_vit_w: int):
        """ViT + Aligner：把 patch 网格编码成 Aligner 特征行。

        输入：patches [n_vit_h*n_vit_w, 3, p, p]；n_vit_h；n_vit_w。
        输出：[n_llm_h*n_llm_w, dim]。
        """
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: torch.Tensor):
        """把每张图的 token span 覆盖成 ViT/Aligner 特征（按参考实现 merge_image_embeddings）。

        输入：images list[list[ImageInput]]（每样本的图像 span）；h [B,L,dim]。
        输出：新的 h [B,L,dim]。
        预期行为：IMAGE 槽按行主序填入 aligner 行；start/end/newline 填学习嵌入。
        """
        from model.image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START

        h = h.clone()
        for i, sample in enumerate(images):
            for img in sample or ():
                types = img.types.to(h.device)
                span = h[i, img.start:img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = self.encode_image(img.patches.to(h.device), img.n_vit_h, img.n_vit_w)
                span[types == IMAGE] = embeds.to(h.dtype)
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
