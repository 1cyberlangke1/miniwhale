"""MLA + CSA2 跨层共享。

每层对两个 KV 源做稀疏注意力并拼接：滑窗原始 KV + （compress_ratio>0 时）index_topk 个压缩位置。

CSA：只有 kv_source 层用自己的输入算压缩 KV 并发布；后续层**直接读同一份缓存**（无投影）。
同理 index 源层发布 topk 索引，后续层复用。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from model.layers.compressor import Compressor
from model.layers.indexer import Indexer, IndexerLossAutoScaler
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


def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos,
                         device=None, is_prefill: bool | None = None) -> torch.Tensor:
    """每个 query 可见的滑窗缓存槽位；-1 表示空槽。

    输入：window_size；bsz；seqlen；start_pos；device（**直接在目标设备建张量，避免 CPU→GPU 同步**）。
    输出：[bsz, seqlen, min(seqlen,window_size)]（prefill）或 [bsz,1,window_size]（decode），int32。
    预期行为：缓存是 window_size 个槽的环形缓冲。prefill 每 query 一行、各自因果窗口；
              decode 单 query 看整个环（槽 0..window_size-1，>start_pos 的填 -1），
              与参考实现 `arange(window_size)` / `F.pad(arange(start_pos+1), ..., -1)` 同序。
              **decode 固定返回 window_size 个**：长度若随 start_pos 变化，`torch.compile`
              会因张量形状每步不同而反复重编译（实测生成 60 token 要 200+ 秒）；
              未写入的槽填 -1，在 softmax 里被 mask 掉，不改变结果。
    """
    if is_prefill is None:
        is_prefill = (not torch.is_tensor(start_pos)) and start_pos == 0
    if is_prefill:
        end = torch.arange(seqlen, device=device).unsqueeze(1)
        idxs = (end - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=device)
        idxs = torch.where(idxs > end, -1, idxs)
    else:
        assert seqlen == 1, "多 token decode 走线性窗口路径，不经本函数"
        idxs = torch.arange(window_size, device=device)
        idxs = torch.where(idxs <= start_pos, idxs, -1)
    return idxs.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _pos_freqs(freqs_cis: torch.Tensor, start_pos, seqlen: int, device):
    """取 [start_pos, start_pos+seqlen) 的 RoPE 频率行。

    输入：freqs_cis [max_seq, rd]；start_pos（int 或 0 维张量）；seqlen；device。
    输出：[seqlen, rd]。
    预期行为：张量 start_pos 用 `index_select`（形状固定，可进编译图）；
              int 用切片（训练/对拍路径，行为不变）。
    """
    if torch.is_tensor(start_pos):
        return freqs_cis.index_select(0, torch.arange(seqlen, device=device) + start_pos.long())
    return freqs_cis[start_pos:start_pos + seqlen]


def _cache_write(buf: torch.Tensor, slot, value: torch.Tensor):
    """把 value [B,1,D] 写进缓存第 `slot` 个槽。

    输入：buf [B,W,D]；slot（int 或 0 维张量）；value [B,1,D]。
    输出：无（原地写）。
    预期行为：int 用直接下标；张量用 `scatter_`（形状固定，可进编译图）。
    """
    if torch.is_tensor(slot):
        buf.scatter_(1, slot.long().view(1, 1, 1).expand(value.size(0), 1, buf.size(-1)), value)
    else:
        buf[:, slot] = value[:, 0]


def _cache_write_at(buf: torch.Tensor, start, value: torch.Tensor):
    """把 value [B,T,D] 写进缓存从第 `start` 行起的连续 T 行。

    输入：buf [B,N,D]；start（int 或 0 维张量）；value [B,T,D]。
    输出：无（原地写）。
    预期行为：int 用切片；张量用 `scatter_`（形状固定，可进编译图）。
    """
    t = value.size(1)
    if torch.is_tensor(start):
        idx = (start.long() + torch.arange(t, device=value.device)).view(1, -1, 1)
        buf.scatter_(1, idx.expand(value.size(0), -1, buf.size(-1)), value)
    else:
        buf[:, start:start + t] = value


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

    def forward(self, x: torch.Tensor, start_pos, shared: SharedAttentionRuntime,
                is_prefill: bool | None = None, emit_at=None,
                pos=None) -> torch.Tensor:
        """一层注意力前向。

        输入：x [B,L,dim]；start_pos（prefill 为 int 0；decode 为位置张量）；
              shared（跨层共享槽）；is_prefill（None=按 start_pos 推断）；
              emit_at（本步内凑满压缩组的 token 下标元组；None=按单 token 语义推断）；
              pos（缓存 scatter 写用的位置张量；None=用 start_pos）。
        输出：[B,L,dim]。
        预期行为：显式 is_prefill/emit_at 让分支不依赖 start_pos 的**值**，
                  否则 torch.compile 会按每个位置的取模结果重复编译；
                  decode 下形状全程静态（压缩缓存按整块参与，靠 mask 裁剪）。
        """
        bsz, seqlen, _ = x.size()
        self._ensure_cache(bsz)
        rd = self.rope_head_dim
        if is_prefill is None:
            is_prefill = (not torch.is_tensor(start_pos)) and start_pos == 0
        if pos is None:
            pos = start_pos
        if emit_at is None and not is_prefill:
            assert not torch.is_tensor(start_pos), "位置为张量时必须显式传 emit_at"
            cr = self.compress_ratio
            emit_at = tuple(j for j in range(seqlen)
                            if cr > 0 and (start_pos + j + 1) % cr == 0)
        freqs = _pos_freqs(self.freqs_cis, pos, seqlen, x.device)

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rotary_emb(q, freqs, rd)
        # 滑窗 KV（每层自算）；参考实现对 KV 做 FP8 量化（act_quant, block=32, pow2）
        kv = self.kv_norm(self.wkv(x))
        kv = apply_rotary_emb(kv, freqs, rd)
        kv = fake_quant_fp8(kv, block_size=32, pow2=True)
        win = self.window_size
        if is_prefill:
            # 训练/prefill 计算走 kv 本身，同时把 detached 副本写进窗口缓存供后续 decode 读
            if seqlen <= win:
                with torch.no_grad():
                    self.window_kv_cache[:bsz, :seqlen] = kv.detach()
            else:
                cutoff = seqlen % win
                with torch.no_grad():
                    self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = \
                        kv[:, -win:].detach().split([win - cutoff, cutoff], dim=1)
            window_kv = kv
            win_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device,
                                            is_prefill=True)
        else:
            if seqlen == 1:
                # 单 token：槽位用张量 scatter（形状固定，可进编译图）
                with torch.no_grad():
                    _cache_write(self.window_kv_cache[:bsz], pos % win,
                                 kv[:, 0:1].detach())
                window_kv = self.window_kv_cache[:bsz]
                win_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos, x.device,
                                                is_prefill=False)
            else:
                # 多 token：一次写 T 行会覆盖块内**更早 query 仍需读取**的旧槽位
                # （环形缓冲里位置 q 与 q+win 共用槽）。故先按位置顺序把旧窗口取出来，
                # 再写新行；注意力在「旧窗口 ‖ 本块」的线性缓冲上做因果滑窗。
                order = (torch.arange(win, device=x.device) + (pos - win)) % win
                hist = self.window_kv_cache[:bsz].index_select(1, order)
                with torch.no_grad():
                    for j in range(seqlen):
                        _cache_write(self.window_kv_cache[:bsz], (pos + j) % win,
                                     kv[:, j:j + 1].detach())
                window_kv = torch.cat([hist, kv], dim=1)
                # 线性下标 i 对应绝对位置 pos-win+i；query j 的窗口是 i ∈ [j+1, j+win]，
                # 只需屏蔽 i < win-pos（位置 <0 的越界项）。
                low = (win - pos).clamp_min(0)
                base = (torch.arange(win, device=x.device).view(1, 1, win)
                        + (torch.arange(seqlen, device=x.device) + 1).view(1, seqlen, 1))
                win_idxs = torch.where(base >= low, base, torch.full_like(base, -1))
                win_idxs = win_idxs.expand(bsz, -1, -1).contiguous().int()

        kv_all = window_kv
        idxs_all = win_idxs
        if self.compress_ratio:
            # 压缩位置索引以窗口实际长度为偏移（kv_all = 窗口 ‖ 压缩）
            comp_kv, comp_idxs = self._compress_kv(x, qr, start_pos, window_kv.size(1),
                                                    shared, is_prefill, emit_at, pos)
            kv_all = torch.cat([kv_all, comp_kv], dim=1)
            idxs_all = torch.cat([idxs_all, comp_idxs], dim=-1)

        o = self._sparse_attn(q, kv_all, idxs_all)
        o = apply_rotary_emb(o, freqs, rd, inverse=True)
        # indexer 训练：让 indexer 打分对齐主注意力对压缩位置的分布（DSA 做法）。
        # indexer 的 topk 选择不可微，故用单独的 KL loss 驱动其参数，经 AutoScaler 挂到输出。
        # 注意：挂在 rope 输出**之后**——custom Function 的输出不能再被后续原地操作改写。
        if (self.is_index_source and self.indexer is not None
                and getattr(self.indexer, "_last_score", None) is not None
                and self.compress_ratio and is_prefill):
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

        数值：**逐算子按参考实现 `sparse_attn_kernel` 复刻**（tilelang）：
        - QK^T 与 PV 都是 bf16 输入、fp32 累加（kernel 的 `T.gemm` 输出 fp32）。
        - softmax 的 exp 结果先转 bf16（kernel 的 `acc_s_cast` 是 BF16）再算 PV。
        - sink 作为额外一项加在分母（`sum_exp += exp(sink - max)`），不参与 PV。
        若把 exp 留 fp32、或先归一化再算 PV，会与参考实现差 ~1e-2（bf16 舍入点不同）。
        """
        b, L, H, D = q.shape
        K = idxs.size(-1)
        qf = q.float()
        # gather KV: [B,L,K,D]。用沿 T 维的 gather（index 直接是 [B,L*K]），**不造**
        # [B,L,T,D] 中间张量——后者在 L=1024/T=1152 时是 6 亿元素（fwd+bwd 爆显存）。
        # gather 是纯复制无算术，与旧写法逐位一致。
        # 等价改写：先在 **kv 的原 dtype（bf16）** 上 gather，再 `.float()`。dtype cast 是逐元素、
        # 与 gather（纯复制）可交换，故数值逐位相同；但中间张量从 fp32 [B,L,K,D] 减半为 bf16，
        # 拷贝量减半（profiler 里 copy_ 是大头）。
        valid = idxs >= 0
        safe = idxs.clamp_min(0)
        k_sel = kv.gather(1, safe.reshape(b, L * K, 1).expand(b, L * K, D)).reshape(b, L, K, D).float()
        # QK^T：bf16 输入、fp32 累加（此处输入已是 bf16 上转的 fp32，等价 fp32 累加）
        scores = torch.einsum("blhd,blkd->blhk", qf, k_sel) * self.softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -torch.inf)
        # 在线 softmax（单块）：减 topk 上的最大（**不含 sink**，按参考实现）。
        # max 下界取 -1e30（按参考实现 `T.fill(scores_max, -1e30)`）：全 -1 行（无有效 KV）时
        # 避免 exp(-inf - (-inf))=NaN，该行输出退化为全 0。
        mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
        e = (scores - mx).exp()
        # sink 只进分母：sum_exp = Σ exp(x-max) + exp(sink-max)
        sum_exp = e.sum(dim=-1) + (self.attn_sink.view(1, 1, H) - mx.squeeze(-1)).exp()
        # PV：exp 先转 bf16（acc_s_cast），再 bf16 输入、fp32 累加；最后除分母
        out = torch.einsum("blhk,blkd->blhd", e.to(q.dtype).float(), k_sel) / sum_exp.unsqueeze(-1)
        return out.to(q.dtype)

    def _compress_kv(self, x, qr, start_pos, offset, shared, is_prefill: bool = False,
                     emit_at=None, pos=None):
        """压缩 KV + 压缩位置索引（CSA 共享在此发生）。

        输入：x [B,L,dim]；qr；start_pos；offset；shared；is_prefill；
              emit_at（本步凑满压缩组的 token 下标元组）；pos（位置张量，写缓存用）。
        输出：(comp_kv [B,T,head_dim], idxs [B,L,K])。
        预期行为：prefill（训练）源层算出的 latent 直接作为共享张量发布（带梯度，供后续层读）；
                  decode 写入缓存供跨步复用，并返回**整个缓存**（定长，便于 CUDA Graph）。
                  多 token 时一次可产出多行 latent（`emit_at` 多项），行号连续。
        """
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        if pos is None:
            pos = start_pos
        if is_prefill:
            compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        else:
            # 每个 query 可见的压缩位置数 = (绝对位置 + 1) // ratio
            compress_lens = ((pos + 1 + torch.arange(seqlen, device=x.device)) // ratio
                             ).unsqueeze(-1)
        latent = None
        if self.is_kv_source:
            latent = self.compressor(x, pos, is_prefill=is_prefill, emit_at=emit_at)
            # 参考实现：源层把**整个共享 cache** 发布给后续层（非源层按自己的 compress_len 截取）。
            # 推理走此路（逐位一致）；训练用 latent 保留梯度（见下）。
            if not self.training:
                shared.compress_kv = self.compress_kv_cache
        # 参考实现：shared_attn.index_k 只在「本层 owns_k 且 latent 非 None」时发布
        # （Indexer.forward 内 if self.owns_k and latent is not None）。组未满的 decode
        # 非压缩步 latent=None 不发布，此处读到的是上一层 kv_source 遗留的槽——按参考实现。

        # index 源层从自己的 latent 派生 index key（owns_k），供后续层复用。
        # 发射行的绝对位置：第 j 个 token 凑满组时，缓存行 = (pos+j+1)//ratio - 1，
        # 多个发射行彼此连续，故可用一次区间写。
        freqs = None
        row0 = None
        if latent is not None and not is_prefill:
            # 下标张量**缓存**在模块上：捕获期做 H2D 拷贝是非法的（除非 pinned），
            # 故热身第一次就建好，捕获时直接复用同一块设备内存。
            if getattr(self, "_emit_idx_key", None) != emit_at:
                self._emit_idx = torch.tensor(emit_at, device=x.device, dtype=torch.long)
                self._emit_idx_key = emit_at
            j_idx = self._emit_idx
            freqs = self.freqs_cis.index_select(0, (pos + 1 - ratio + j_idx).long())
            row0 = ((pos + 1 - ratio + j_idx[0]) // ratio)
        if self.is_index_source and latent is not None and self.is_kv_source:
            rd = self.rope_head_dim
            k = self.indexer.k_norm(self.indexer.wk(latent))
            k_freqs = freqs if not is_prefill else self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            k = apply_rotary_emb(k, k_freqs, rd)
            # 参考实现：index k 写缓存前做 FP4 量化（fp4_act_quant, block=32, E8M0 scale）
            k = fake_quant_fp4(k, block_size=32)
            if self.training:
                shared.index_k = k          # 训练：带梯度共享（张量）
            else:
                # 推理：写缓存并发布**整个缓存**（按参考实现 shared_attn.index_k = self.k_cache）；
                # 后续 indexer 按位置切片，故缓存长度须为 max_seq//ratio。
                _cache_write_at(self.index_k_cache[:bsz],
                                row0 if not is_prefill else pos // ratio, k)
                shared.index_k = self.index_k_cache

        idxs = self._compress_topk_idxs(x, qr, latent, start_pos, offset, compress_lens,
                                        shared, is_prefill)
        if latent is not None:
            rd = self.rope_head_dim
            l_freqs = freqs if not is_prefill else self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            latent = apply_rotary_emb(latent, l_freqs, rd)
            # 参考实现对压缩 latent 做 FP4 量化（fp4_act_quant, block=16, E4M3 scale）
            latent = fake_quant_compressed_kv(latent, block_size=16)
            if self.training:
                shared.compress_kv = latent   # 训练：带梯度共享给后续层
            else:
                # 推理：写进 cache（shared.compress_kv 已指向 cache，按参考实现）
                _cache_write_at(self.compress_kv_cache[:bsz],
                                row0 if not is_prefill else pos // ratio, latent)
        if shared.compress_kv is None:  # 尚无源层发布（如单独调用本层）
            return x.new_zeros(bsz, 0, self.head_dim), idxs
        if is_prefill:
            return shared.compress_kv[:bsz, :compress_len], idxs
        # decode：返回**整个压缩缓存**（定长）。CUDA Graph 要求形状全程静态，
        # 按 compress_len 切片会让长度随位置变化；越界位置由 indexer 的
        # `compress_lens` 掩成 -inf，在 softmax 里 exp=0，对结果无贡献。
        return shared.compress_kv[:bsz], idxs

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, compress_lens, shared,
                            is_prefill: bool = False):
        """压缩位置的 topk 索引；非 index 源层复用共享结果。

        输入：x；qr；latent；start_pos；offset；compress_lens（prefill [L,1]，decode 标量）；
              shared；is_prefill。
        输出：idxs [B,L,index_topk]，int32（不可达填 -1，其余加 offset）。
        预期行为：index 源层调用 indexer 打分并发布到 shared.topk_idxs；其余层直接复用。
        """
        bsz, seqlen, _ = x.size()
        if not self.is_index_source:
            if shared.topk_idxs is None:  # 尚无源层发布 → 本层无压缩位置可看
                return torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
            return shared.topk_idxs
        # 参考实现：index 源层无条件读共享槽 shared_attn.index_k（Indexer.forward 的
        # `index_k = shared_attn.index_k[:bsz, : end_pos // ratio]`）。该槽只由「owns_k 且
        # latent 非 None」的层更新（见 _compress_kv），所以组未满的 decode 非压缩步读到的是
        # 上一层 kv_source 遗留的缓存——猫猫照这个语义实现，不直读自己的缓存喵。
        end_pos = start_pos + seqlen
        index_k = shared.index_k
        if index_k is None:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            # 参考实现：indexer 的 q 用 attention 层的 freqs_cis（Indexer.freqs_cis = Attention.freqs_cis）
            if self.indexer.freqs_cis is None:
                self.indexer.freqs_cis = self.freqs_cis
            # 参考实现：index_k 按本层自己的 end_pos//ratio 切片（非源层读的缓存比它自己的压缩位置多）。
            # decode 用**整个缓存**（定长）：切片长度随位置变化会破坏 CUDA Graph 的静态形状；
            # 越界位置由 `compress_lens` 掩成 -inf，对 top-k 无影响。
            if is_prefill:
                index_k = index_k[:bsz, : end_pos // self.compress_ratio]
            else:
                index_k = index_k[:bsz]
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
