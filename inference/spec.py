"""DSpark 投机解码主循环。

每轮：用草稿头并行出 block_size 个草稿 → 主干**一次多 token 前向**验证
→ 接受最长匹配前缀 + 1 个修正 token。

缓存回滚：**不需要**。环形窗口与压缩缓存的槽位由绝对位置决定，被拒绝位置留下的
内容会在下一轮同样的位置被真 token 覆写，且覆写发生在注意力读取之前
（compress 路径在同一 forward 内先写后读，窗口路径同理）。因此拒绝不会污染后续。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from inference.engine import Engine, GenerationConfig


@dataclass
class SpeculativeConfig(GenerationConfig):
    """投机解码配置。"""

    use_dspark: bool = True
    confidence_threshold: float = 0.0   # sigmoid(conf) < 阈值 → 从该位置截断草稿


@dataclass
class SpecStats:
    """投机解码统计。"""

    drafted: int = 0        # 共起草多少 token
    accepted: int = 0       # 共接受多少 token
    rounds: int = 0         # 验证轮数

    @property
    def accept_rate(self) -> float:
        """接受率 = accepted / drafted。"""
        return self.accepted / max(self.drafted, 1)

    @property
    def tau(self) -> float:
        """每次验证平均产出 token 数。"""
        return 1.0 + self.accepted / max(self.rounds, 1)


def _confident_prefix(confidence: torch.Tensor, block_size: int, threshold: float) -> int:
    """按置信度截断草稿前缀。

    输入：confidence [block]（sigmoid 前的 logits）；block_size；threshold。
    输出：可接受的草稿长度（0..block_size）。
    预期行为：找到第一个 sigmoid(conf) < 阈值的位置，返回它之前的位置数；
              全高则返回 block_size；阈值 <=0 时不截断（不产生 GPU→CPU 同步）。
    """
    if threshold <= 0.0:
        return block_size
    below = confidence.sigmoid() < threshold
    if not bool(below.any()):
        return block_size
    return int(torch.nonzero(below, as_tuple=False)[0].item())


@torch.no_grad()
def generate_speculative(engine: Engine, input_ids: torch.Tensor,
                         cfg: SpeculativeConfig) -> tuple[torch.Tensor, SpecStats]:
    """投机解码生成。

    输入：engine；input_ids [1, L0]；cfg。
    输出：([1, L0 + 生成数], SpecStats)。
    预期行为：草稿头（有图则用图）出 block 个候选 → `decode_step_multi` 一次验证
              T=block+1 个 token → 接受最长前缀 + 1 个修正 token。
              位置记账：`q` 是"主干 hidden 所在位置"，seed token 在 q+1；
              验证块覆盖 q+1 .. q+1+k；验证后 `_last_hidden[:, a]` 正好是
              新 seed（位置 q+a+2）所需的前一位置 hidden。
    """
    assert engine.model.dspark is not None, "模型未启用 DSpark"
    block_size = engine.model.dspark.block_size
    eos = cfg.eos_token_id
    stats = SpecStats()

    engine.model.reset_cache()
    ids = input_ids
    logits = engine.prefill(ids)                      # hidden 在 L0-1，logits 预测 L0
    seed = engine.sample(logits, cfg.temperature)     # 位置 L0
    ids = torch.cat([ids, seed], dim=1)
    stats.accepted += 1
    stats.rounds += 1
    q = ids.size(1) - 2                               # = L0-1（hidden 所在位置）
    hidden = engine._last_hidden[:, -1:]
    need_step = eos is not None and int(seed.item()) == eos

    while not need_step and ids.size(1) - input_ids.size(1) < cfg.max_new_tokens:
        out_ids, _, conf = engine.model.forward_spec(
            seed, start_pos=q, temperature=0.0, main_hidden=hidden)
        drafts = out_ids[:, 1:]                       # [1, block]
        k = _confident_prefix(conf[0], block_size, cfg.confidence_threshold)
        stats.rounds += 1
        if k == 0:                                    # 没草稿可用：走一次普通解码
            q = ids.size(1) - 1
            logits = engine.decode_step(ids[:, -1:], q, ids[:, :-1])
            nxt = engine.sample(logits, cfg.temperature)
            ids = torch.cat([ids, nxt], dim=1)
            hidden = engine._last_hidden[:, -1:]
            seed = nxt
            stats.accepted += 1
            need_step = eos is not None and int(nxt.item()) == eos
            continue

        stats.drafted += k
        verify = torch.cat([seed, drafts[:, :k]], dim=1)          # [1, k+1]
        v_logits = engine.decode_step_multi(verify, q + 1, ids[:, :-1])
        # 一次采样所有验证位置，只在最后取一次 accepted（避免逐 token 同步）
        preds = engine.sample(v_logits[0, :k], cfg.temperature)   # [k,1]
        match = (preds.reshape(-1) == drafts[0]).to(torch.int32)
        accepted = int(match.cumprod(0).sum().item())
        fix = engine.sample(v_logits[0, accepted:accepted + 1], cfg.temperature)
        ids = torch.cat([ids, drafts[:, :accepted], fix], dim=1)
        stats.accepted += accepted + 1
        # 新 seed = fix（位置 q+accepted+2），它前面的 hidden 在 q+accepted+1 = 下标 accepted
        hidden = engine._last_hidden[:, accepted:accepted + 1]
        seed = fix
        q = q + accepted + 1
        need_step = eos is not None and int(fix.item()) == eos

    return ids, stats
