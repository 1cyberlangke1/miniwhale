"""优化器（按参考实现报告 §2.5 + Algorithm 1）。

| 参数组 | 优化器 |
|---|---|
| 线性层矩阵 | Muon（torch 内置），Q/K head-wise |
| RMSNorm / 非矩阵（bias、scale） | AdamW |
| Engram / Embedding / Prediction Head | Sinkhorn-balanced update |
| Engram | lr × 5 |

参考实现 Algorithm 1（Momentum update with Sinkhorn balancing）：
  1. M ← βM + (1-β)G；Ĝ ← βM + (1-β)G（Nesterov）
  2. ρ_i = ‖Ĝ_i,:‖₂，ρ̄ = mean(ρ)；行 i 的 ρ_i ≤ τ·ρ̄ 则整行置 0（掩蔽近零行）
  3. 交替对行/列做 L2 归一（K 次，奇数步归行、偶数步归列）
  4. Δ = √n · U^(K)（把单位行 L2 范数转成单位行 RMS）
  5. 有效 lr：η̃ = γ·η（γ=0.18，对齐 Adam 的更新幅度）
  6. W ← W - η̃·Δ

只有 Sinkhorn-balanced 需要自己写；Muon / AdamW 用 torch 现成。
"""

from __future__ import annotations

import torch
from torch import nn


def sinkhorn_balanced_update(grad: torch.Tensor, momentum: torch.Tensor, lr: float,
                             momentum_coef: float = 0.95, k: int = 11,
                             tau: float = 1e-3, eps: float = 1e-20,
                             gamma: float = 0.18) -> torch.Tensor:
    """Sinkhorn 均衡更新（按参考实现 Algorithm 1，用于 Engram / Embedding / Head）。

    输入：grad [m,n]；momentum 同形状；lr；β；K（奇数）；τ 掩蔽阈值；eps；γ lr 修正。
    输出：参数更新量 Δ（调用方执行 W ← W - Δ）。
    预期行为：Nesterov 动量 → 掩蔽近零行 → 行/列 L2 归一交替 K 次 → √n 缩放 → γ·lr。
    """
    m, n = grad.shape
    momentum = momentum_coef * momentum + (1 - momentum_coef) * grad
    g_hat = momentum_coef * momentum + (1 - momentum_coef) * grad   # Nesterov
    # 掩蔽近零行
    row_norm = g_hat.norm(dim=1)
    rho_bar = row_norm.mean()
    mask = (row_norm > tau * rho_bar).to(g_hat.dtype).unsqueeze(1)
    u = g_hat * mask
    # 交替行/列 L2 归一（奇数步归行，偶数步归列）
    for step in range(1, k + 1):
        if step % 2 == 1:
            u = u / (u.norm(dim=1, keepdim=True) + eps)
        else:
            u = u / (u.norm(dim=0, keepdim=True) + eps)
    delta = (n ** 0.5) * u
    return (gamma * lr) * delta


def build_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.1,
                    engram_lr_mult: float = 5.0):
    """按参考实现分组构造参数组。

    输入：model；lr；weight_decay；Engram 学习率倍数。
    输出：dict，含 'muon'（线性矩阵）、'adamw'（norm/bias/scale）、
          'sinkhorn'（Engram 表 / Embedding / Head，走 Sinkhorn 均衡更新）、
          'engram'（Engram 投影层，走 Muon 但 lr ×5）。
    预期行为：技术报告 §2.5——Muon 用于线性权重矩阵（含 Engram 投影层）；
              Sinkhorn 用于 Engram 表 / Embedding / 预测头；Engram 学习率 ×5。
    """
    matrix, other, sinkhorn, engram_proj = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_engram = "engram" in name
        if is_engram and p.dim() >= 2 and "embed" not in name:
            # Engram 投影层（wkv 等矩阵）→ Muon（参考实现：Muon for Engram projection layers）
            engram_proj.append(p)
        elif is_engram or "embed" in name or name.startswith("head"):
            # Engram 表 / Embedding / 预测头 → Sinkhorn 均衡更新
            sinkhorn.append(p)
        elif p.dim() >= 2:
            matrix.append(p)
        else:
            other.append(p)

    return {
        "muon": [{"params": matrix, "lr": lr, "weight_decay": weight_decay}],
        "adamw": [{"params": other, "lr": lr, "weight_decay": weight_decay}],
        "sinkhorn": [{"params": sinkhorn, "lr": lr * engram_lr_mult}],
        "engram": [{"params": engram_proj, "lr": lr * engram_lr_mult, "weight_decay": weight_decay}],
    }


class SinkhornBalanced(torch.optim.Optimizer):
    """Sinkhorn-balanced 优化器（参考实现 Algorithm 1，用于 Engram / Embedding / Head）。

    输入：参数组（含 lr）；默认 β=0.95、K=11、τ=1e-3、γ=0.18。
    输出：就地更新参数。
    预期行为：维护每参数的 Nesterov 动量缓冲；每步调 sinkhorn_balanced_update 得 Δ 并做减法。
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, k: int = 11,
                 tau: float = 1e-3, gamma: float = 0.18):
        defaults = dict(lr=lr, momentum=momentum, k=k, tau=tau, gamma=gamma)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.dim() < 2:                       # 1 维参数不做 Sinkhorn（参考实现只对矩阵）
                    p.add_(g, alpha=-group["lr"])
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(g)
                buf = state["momentum"]
                delta = sinkhorn_balanced_update(
                    g, buf, group["lr"], momentum_coef=group["momentum"],
                    k=group["k"], tau=group["tau"], gamma=group["gamma"])
                p.add_(-delta)
                # 同步动量缓冲（Nesterov 用到的 M_t）
                buf.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])
        return loss


def make_optimizers(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.1,
                    engram_lr_mult: float = 5.0):
    """构造实际的优化器对象（Muon + AdamW + Sinkhorn）。

    输入：同 build_optimizer。
    输出：(muon, adamw, sinkhorn) 三个 torch 优化器。
    预期行为：Muon 管矩阵参数；AdamW 管 norm/bias/scale；Sinkhorn 管 Engram/Embedding/Head。
    """
    g = build_optimizer(model, lr, weight_decay, engram_lr_mult)
    # Engram 投影层用 Muon（参考实现），故与主干矩阵组同优化器
    muon = torch.optim.Muon(g["muon"] + g["engram"], lr=lr, momentum=0.95, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(g["adamw"], betas=(0.9, 0.95), eps=1e-20, weight_decay=weight_decay)
    sinkhorn = SinkhornBalanced(g["sinkhorn"], lr=lr * engram_lr_mult)
    return muon, adamw, sinkhorn
