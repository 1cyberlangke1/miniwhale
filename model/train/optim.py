"""优化器（按参考实现报告 §2.5）。

| 参数组 | 优化器 |
|---|---|
| 线性层矩阵 | Muon（torch 内置），Q/K head-wise |
| RMSNorm / 非矩阵 | AdamW |
| Embedding + LM Head | Sinkhorn-balanced update |
| Engram | lr × 5 |

只有 Sinkhorn-balanced 需要自己写；Muon / AdamW 用 torch 现成。
"""

from __future__ import annotations

import torch
from torch import nn


def sinkhorn_balanced_update(grad: torch.Tensor, momentum: torch.Tensor, lr: float,
                             momentum_coef: float = 0.95, k: int = 11,
                             tau: float = 1e-3, eps: float = 1e-20) -> torch.Tensor:
    """Sinkhorn 均衡更新（用于 embedding / 输出头）。

    输入：grad [rows, cols]；momentum 同形状；lr；动量系数；K 迭代数；tau；eps。
    输出：更新后的参数张量。
    预期行为：动量累积 → 行/列交替归一 K 次（行和列和都趋向 1）→ 按 lr 更新。
    """
    momentum = momentum_coef * momentum + (1 - momentum_coef) * grad
    m = momentum.abs() + tau
    # 交替归一 K 次
    for _ in range(k):
        m = m / (m.sum(dim=1, keepdim=True) + eps)
        m = m / (m.sum(dim=0, keepdim=True) + eps)
    return lr * momentum.sign() * m


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.1,
                    engram_lr_mult: float = 5.0):
    """按参考实现分组构造优化器。

    输入：model；lr；weight_decay；Engram 学习率倍数。
    输出：dict，含 'muon'（矩阵参数）、'adamw'（其余），及 Engram 单独组。
    预期行为：2 维权重（Linear 矩阵）→ Muon；1 维（norm/bias）→ AdamW；
              Engram 参数 lr × engram_lr_mult；embedding/head 单独标注（供 Sinkhorn）。
    """
    matrix, other, engram, emb_head = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "engram" in name:
            engram.append(p)
        elif "embed" in name or name.startswith("head"):
            emb_head.append(p)
        elif p.dim() >= 2:
            matrix.append(p)
        else:
            other.append(p)

    groups = {
        "muon": [{"params": matrix, "lr": lr, "weight_decay": weight_decay}],
        "adamw": [{"params": other, "lr": lr, "weight_decay": weight_decay},
                  {"params": emb_head, "lr": lr, "weight_decay": weight_decay}],
        "engram": [{"params": engram, "lr": lr * engram_lr_mult, "weight_decay": weight_decay}],
    }
    return groups


def make_optimizers(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.1,
                    engram_lr_mult: float = 5.0):
    """构造实际的优化器对象（Muon + AdamW）。

    输入：同 build_optimizer。
    输出：(muon, adamw) 两个 torch 优化器。
    预期行为：Muon 管矩阵参数；AdamW 管其余 + Engram（Engram 用放大 lr）。
    """
    g = build_optimizer(model, lr, weight_decay, engram_lr_mult)
    muon = torch.optim.Muon(g["muon"], lr=lr, momentum=0.95, weight_decay=weight_decay)
    adamw_groups = g["adamw"] + g["engram"]
    adamw = torch.optim.AdamW(adamw_groups, betas=(0.9, 0.95), eps=1e-20, weight_decay=weight_decay)
    return muon, adamw
