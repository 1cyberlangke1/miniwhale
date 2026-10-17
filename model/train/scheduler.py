"""学习率调度。

两种模式：
- 三段式（技术报告 §4.2.2，预训练用）：warmup → 恒定 → 余弦衰减 → 恒定 min_lr。
- 两段式（SFT / ViLM / DSpark / SigLIP 用）：warmup → 余弦衰减到 min_lr。
"""

from __future__ import annotations

import math


def get_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float,
           decay_start: float | None = None, decay_end: float | None = None) -> float:
    """学习率调度（带线性 warmup）。

    输入：step 当前步；total_steps 总步数；warmup_steps 预热步数；base_lr 峰值 lr；
          min_lr 最低 lr；decay_start / decay_end 余弦衰减窗口（占全程比例）。
    输出：本步的 lr。
    预期行为：
      - decay_start/decay_end 都为 None → 两段式：前 warmup_steps 步从 base_lr/10 线性升到
        base_lr，之后按余弦从 base_lr 衰减到 min_lr（到 total_steps）。
      - 都给定 → 三段式：warmup 后**恒定** base_lr 直到 decay_start·total_steps，再按余弦降到
        min_lr（到 decay_end·total_steps），末尾恒定 min_lr。参考实现：2.6e-4 恒定到 62%，
        余弦降到 2.6e-5（89%），其后不变。
    """
    if step < warmup_steps:
        return base_lr * (0.1 + 0.9 * step / max(warmup_steps, 1))
    if decay_start is None or decay_end is None:
        if step >= total_steps:
            return min_lr
        ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    else:
        start, end = decay_start * total_steps, decay_end * total_steps
        if step <= start:
            return base_lr
        if step >= end:
            return min_lr
        ratio = (step - start) / max(end - start, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cos
