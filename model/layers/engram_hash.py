"""Engram n-gram 哈希。

把每个位置映射到它结尾的 n-gram 的哈希桶 id。
哈希在「压缩词表」上进行（大小写/空白归一后折叠），因此 " The"/"the"/"THE" 同哈希。
每个 (n-gram 阶数, head) 占独立素数桶区间，互不干扰。
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

DEAD = -1  # 图像/padding 位置：阻断 n-gram 回看


def find_next_prime(start: int, seen: set[int]) -> int:
    """取大于 start 且未分配过的最小素数。

    输入：start 起点；seen 已分配的素数集合。
    输出：素数。
    预期行为：纯 Python 判定（不依赖 sympy）；分配后由调用方加入 seen 保证区间不相交。
    """
    candidate = start + 1
    while True:
        if candidate >= 2 and candidate not in seen:
            is_p = candidate == 2 or (
                candidate % 2 != 0
                and all(candidate % d for d in range(3, math.isqrt(candidate) + 1, 2))
            )
            if is_p:
                return candidate
        candidate += 1


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """把每个 token id 映射到折叠后的压缩 id 空间。

    输入：HF fast tokenizer。
    输出：(lookup 表，压缩词表大小)。lookup[tid] = 压缩 id。
    预期行为：归一化 = NFKC→NFD→去重音→小写→空白归一；无效字节 token 用原始形式作 key。
              压缩词表大小是哈希乘子的依赖项，必须稳定。
    """
    from tokenizers import Regex, normalizers

    sentinel = ""  # 单个空格 token 用它保命，避免 Strip 后塌成空串
    normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), sentinel),
        normalizers.Strip(),
        normalizers.Replace(sentinel, " "),
    ])
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for tid in range(len(tokenizer)):
        text = backend.decode([tid], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(tid)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_new:
            key_to_new[key] = len(key_to_new)
        lookup[tid] = key_to_new[key]
    return lookup, len(key_to_new)


def compute_hash_multipliers(
    layer_ids: tuple[int, ...], max_ngram_size: int, tokenizer_vocab_size: int
) -> torch.Tensor:
    """每个 (层, 回看位) 一个乘子，按层独立 RNG，保证各层哈希不同。

    输入：层 id 元组、最大 n-gram 阶数、压缩词表大小。
    输出：[num_layers, max_ngram_size] 的 int64 乘子。
    预期行为：乘子取奇数，并限幅保证 token_id * multiplier 不溢出 int64。
    """
    max_long = np.iinfo(np.int64).max
    bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        values = g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


class EngramHash(nn.Module):
    """把输入序列映射为 Engram 哈希 id。

    输入：input_ids [B, L]（原始 token id）；token_mask [B, L]（False=图像/padding，阻断回看）。
    输出：哈希 id [B, L, n_layers, (max_ngram-1)*n_heads]，int64，含每层桶偏移。
    预期行为：序列头/padding/图像位置用 pad_id 填充；n-gram 不跨这些边界；
              所有派生量（token_map/primes/offsets/multipliers）为无梯度 buffer。
    """

    def __init__(self, text_cfg, tokenizer):
        super().__init__()
        self.max_ngram_size = text_cfg.engram_max_ngram_size
        self.n_heads = text_cfg.engram_n_heads
        self.layer_ids = tuple(text_cfg.engram_layer_ids)

        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != text_cfg.engram_compressed_vocab_size:
            raise ValueError(
                f"压缩词表大小不符：得 {vocab_size}，config 期望 {text_cfg.engram_compressed_vocab_size}；"
                "哈希乘子依赖此值，不符会静默重哈希整表"
            )
        self.pad_id = token_map[text_cfg.engram_pad_token_id]

        # 每 (层, n-gram 阶, head) 一个全局不重复素数
        primes, seen = [], set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes, current = [], text_cfg.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))

        flat = [[p for per_ngram in layer for p in per_ngram] for layer in primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        multipliers = compute_hash_multipliers(self.layer_ids, self.max_ngram_size, vocab_size)

        self.register_buffer("primes", torch.tensor(primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, L = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, torch.full_like(compressed, DEAD))

        # 向量化（等价替换原先的 shift 循环）：一次性建 [1,L,M] 的偏移索引，
        # 沿 L 维一次 gather 出全部回看位，再对「越界 / DEAD」做**累积 OR** 得到屏蔽位。
        # 原循环里 `blocked = blocked | ...` 逐 shift 累积，等价于对 bad 做前缀 OR；
        # 每步 tokens[shift] = where(blocked_shift, pad, source_shift)，故此处直接得到
        # 同形状 [B,L,M] 的 tokens（m 维顺序与原来 stack 一致：m=0 为最近位）。
        M = self.max_ngram_size
        positions = torch.arange(L, device=input_ids.device).view(1, L, 1)   # [1, L, 1]
        shifts = torch.arange(M, device=input_ids.device).view(1, 1, M)      # [1, 1, M]
        src_pos = positions - shifts                                         # [1, L, M]
        # source[b,l,m] = compressed[b, clamp(src_pos[l,m])]：expand 出 [B,L,M] 后沿 L 维 gather
        source = compressed.unsqueeze(-1).expand(b, L, M).gather(
            1, src_pos.expand(b, L, M).clamp_min(0))
        bad = (src_pos < 0) | (source == DEAD)                               # [B, L, M]
        blocked = bad.int().cumsum(dim=-1) > 0                               # 前缀 OR
        tokens = torch.where(blocked, self.pad_id, source)                   # [B, L, M]

        products = tokens.unsqueeze(2) * self.multipliers  # [B, L, layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets
