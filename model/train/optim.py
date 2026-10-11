"""优化器（按参考实现报告 §2.5 + Algorithm 1）。

| 参数组 | 优化器 |
|---|---|
| 线性层矩阵（含 Engram 投影层） | Muon（**head-wise**：Q/K 权重按 head 拆开各自正交化） |
| RMSNorm 权重 | AdamW（带 weight decay） |
| bias / scaling factor 等其它非矩阵 | AdamW（**不带** weight decay） |
| Engram 表 / token embedding / 预测头 | Sinkhorn-balanced update |

参考实现超参（报告 §4.2.2）：
- AdamW：β=(0.9, 0.95)，eps=1e-20，weight_decay=0.1。
- Muon：momentum=0.95，weight_decay=0.1，Nesterov，**把每个更新矩阵的 RMS 缩放到 0.18**
  （以复用 AdamW 的学习率）。
- Sinkhorn：与 Muon 同 momentum 系数与 lr 修正因子；K=11（归一化步数，奇数），τ=1e-3，γ=0.18。
- Engram 学习率 ×5。

参考实现 Algorithm 1（Momentum update with Sinkhorn balancing）：
  1. M_t ← βM_{t-1} + (1-β)G；Ĝ_t ← βM_t + (1-β)G（Nesterov）
  2. ρ_i = ‖Ĝ_i,:‖₂，ρ̄ = mean(ρ)；行 i 的 ρ_i ≤ τ·ρ̄ 则整行置 0（掩蔽近零行）
  3. 交替对行/列做 L2 归一（K 次，奇数步归行、偶数步归列）
  4. Δ = √n · U^(K)（把单位行 L2 范数转成单位行 RMS）
  5. 有效 lr：η̃ = γ·η（γ=0.18，对齐 Adam 的更新幅度）
  6. W ← W - η̃·Δ

只有 head-wise Muon 与 Sinkhorn-balanced 需要自己写；AdamW 用 torch 现成。
"""

from __future__ import annotations

import math

import torch
from torch import nn

# 参考实现 Muon 的 Newton-Schulz 系数（Keller Jordan 五次迭代，最大化零点斜率）
_NS_A, _NS_B, _NS_C = 3.4445, -4.7750, 2.0315


def _newton_schulz(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz 正交化（把梯度矩阵的奇异值拉向 1）。

    输入：g [A,B]（2D）；steps 迭代步数；eps 数值下限。
    输出：与 g 同形状的正交化矩阵（近似 U·Vᵀ）。
    预期行为：转置成「行数 ≤ 列数」后按谱范数 ≤1 归一，跑五次迭代，再转回原朝向。
    """
    if g.size(0) > g.size(1):
        g = g.T
        transposed = True
    else:
        transposed = False
    x = g.bfloat16()
    x = x / x.norm().clamp_min(eps)
    for _ in range(steps):
        gram = x @ x.T
        gram = torch.addmm(gram, gram, gram, beta=_NS_B, alpha=_NS_C)
        x = torch.addmm(x, gram, x, beta=_NS_A)
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    """head-wise Muon（技术报告 §2.5）。

    输入：参数组（含 lr / weight_decay / lr_scale / headwise 映射）。
    输出：就地更新参数。
    预期行为：Nesterov 动量 → Newton-Schulz 正交化（head-wise 参数按 head 拆开各自正交化）
              → decoupled weight decay → 有效 lr = lr·lr_scale·√max(A,B)（把更新 RMS 缩放到 lr_scale）。
    """

    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, weight_decay: float = 0.1,
                 nesterov: bool = True, ns_steps: int = 5, lr_scale: float = 0.18,
                 headwise: dict[int, int] | None = None):
        """headwise：{id(param): n_heads}；n_heads>1 时把该参数按行（输出维）拆成 n_heads 块各自正交化。"""
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, ns_steps=ns_steps, lr_scale=lr_scale)
        super().__init__(params, defaults)
        self._headwise = headwise or {}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            lr_scale = group["lr_scale"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # 1) Nesterov 动量：buf ← β·buf + (1-β)·g；update = β·buf + (1-β)·g
                buf.lerp_(g, 1 - momentum)
                update = g.lerp(buf, momentum) if group["nesterov"] else buf.clone()
                # 2) 正交化：head-wise 参数按 head 拆开，各自 Newton-Schulz
                n_heads = self._headwise.get(id(p), 1)
                if n_heads > 1:
                    sub = update.size(0) // n_heads
                    blocks = update.split(sub, dim=0)
                    ortho = torch.cat([_newton_schulz(b, group["ns_steps"]) for b in blocks], dim=0)
                    a, b_ = sub, update.size(1)
                else:
                    ortho = _newton_schulz(update, group["ns_steps"])
                    a, b_ = update.shape[:2]
                # 3) 有效 lr：把更新 RMS 缩放到 lr_scale（参考实现 0.18），复用 AdamW 的学习率
                adjusted_lr = lr * lr_scale * math.sqrt(max(a, b_))
                # 4) decoupled weight decay，再做参数更新
                p.mul_(1 - lr * wd)
                p.add_(ortho.to(p.dtype), alpha=-adjusted_lr)
        return loss


def sinkhorn_balanced_update(grad: torch.Tensor, momentum: torch.Tensor, lr: float,
                             momentum_coef: float = 0.95, k: int = 11,
                             tau: float = 1e-3, eps: float = 1e-20,
                             gamma: float = 0.18) -> torch.Tensor:
    """Sinkhorn 均衡更新（按参考实现 Algorithm 1，用于 Engram / Embedding / Head）。

    输入：grad [m,n]；momentum 同形状（**不改动**）；lr；β；K（奇数）；τ 掩蔽阈值；eps；γ lr 修正。
    输出：参数更新量 Δ（调用方执行 W ← W - η̃·Δ）。
    预期行为：Nesterov 动量 → 掩蔽近零行 → 行/列 L2 归一交替 K 次 → √n 缩放 → γ·lr。
              momentum 缓冲的更新由调用方负责（本函数为纯函数，便于与参考实现 Algorithm 1 对拍）。
    """
    m, n = grad.shape
    M = momentum_coef * momentum + (1 - momentum_coef) * grad     # M_t = βM + (1-β)G
    g_hat = momentum_coef * M + (1 - momentum_coef) * grad        # Ĝ = βM_t + (1-β)G（Nesterov）
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


class SinkhornBalanced(torch.optim.Optimizer):
    """Sinkhorn-balanced 优化器（参考实现 Algorithm 1，用于 Engram / Embedding / Head）。

    输入：参数组（含 lr）；默认 β=0.95、K=11、τ=1e-3、γ=0.18。
    输出：就地更新参数。
    预期行为：维护每参数的 Nesterov 动量缓冲；每步调 sinkhorn_balanced_update 得 Δ 并做减法。
              参考实现：Sinkhorn-balanced update 用 Nesterov 动量但**不**施加 weight decay。
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
                # 同步动量缓冲（Alg 1 第 1 行：M_t = βM_{t-1} + (1-β)G）
                buf.mul_(group["momentum"]).add_(g, alpha=1 - group["momentum"])
        return loss


def _is_rmsnorm_weight(name: str) -> bool:
    """判断参数是否为 RMSNorm 的可学习权重（参考实现：带 weight decay）。"""
    return name.endswith("norm.weight")


def _is_embedding_or_head(name: str) -> bool:
    """判断参数是否为 embedding 表 / 预测头（参考实现：走 Sinkhorn，不带 weight decay）。"""
    return "embed" in name or name == "head.weight" or name.endswith(".head.weight")


def build_param_groups(model: nn.Module):
    """按参考实现 §2.5 把参数分成四组。

    输入：model。
    输出：dict，含 'muon'（线性矩阵 + Engram 投影）、'headwise'（{id: n_heads}，Q/K 权重按 head 拆）、
          'adamw_wd'（RMSNorm 权重，带 WD）、'adamw_nowd'（bias / scaling / 其它 1D）、
          'sinkhorn'（Engram 表 / embedding / 预测头）。
    预期行为：参考实现——Muon 用于语言主干与 Engram 投影层的线性权重矩阵（Q/K 用 head-wise）；
              AdamW 用于 RMSNorm 权重（带 WD）与 bias / scaling factor（不带 WD）；
              Sinkhorn 用于全部 embedding 与预测头。
    """
    muon, adamw_wd, adamw_nowd, sinkhorn = [], [], [], []
    headwise: dict[int, int] = {}
    n_heads = getattr(model.cfg.text, "num_attention_heads", 0) or 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_embedding_or_head(name):
            sinkhorn.append(p)
        elif p.dim() >= 2:
            muon.append(p)
            # 参考实现 head-wise Muon：Q 权重（wq_b，输出维 = n_heads·head_dim）按 head 拆开各自正交化
            if n_heads > 1 and name.endswith("wq_b.weight") and p.size(0) % n_heads == 0:
                headwise[id(p)] = n_heads
        elif _is_rmsnorm_weight(name):
            adamw_wd.append(p)          # norm 权重带 weight decay
        else:
            adamw_nowd.append(p)        # bias / scaling factor 等不带 weight decay
    return {"muon": muon, "headwise": headwise, "adamw_wd": adamw_wd,
            "adamw_nowd": adamw_nowd, "sinkhorn": sinkhorn}


def make_optimizers(model: nn.Module, lr: float = 1e-3, weight_decay: float = 0.1,
                    engram_lr_mult: float = 5.0):
    """构造实际的优化器对象（head-wise Muon + AdamW + Sinkhorn）。

    输入：model；lr；weight_decay；Engram 学习率倍数。
    输出：(muon, adamw, sinkhorn) 三个 torch 优化器。
    预期行为：Muon 管线性矩阵（Q/K head-wise）；AdamW 管 norm 权重（带 WD）与 bias/scaling（不带 WD）；
              Sinkhorn 管 Engram 表 / embedding / 预测头（Engram 相关 lr ×5）。
    """
    g = build_param_groups(model)
    # Engram 表 / embedding / 预测头：Engram 的 lr ×5（参考实现 §4.2.2）
    engram_ids = {id(p) for n, p in model.named_parameters() if "engram" in n}
    sinkhorn_engram = [p for p in g["sinkhorn"] if id(p) in engram_ids]
    sinkhorn_other = [p for p in g["sinkhorn"] if id(p) not in engram_ids]
    sinkhorn_groups = []
    if sinkhorn_other:
        sinkhorn_groups.append({"params": sinkhorn_other, "lr": lr})
    if sinkhorn_engram:
        sinkhorn_groups.append({"params": sinkhorn_engram, "lr": lr * engram_lr_mult})

    muon = Muon(
        [{"params": g["muon"], "lr": lr, "weight_decay": weight_decay, "lr_scale": 0.18}],
        lr=lr, momentum=0.95, weight_decay=weight_decay, lr_scale=0.18,
        headwise=g["headwise"])
    adamw = torch.optim.AdamW(
        [{"params": g["adamw_wd"], "lr": lr, "weight_decay": weight_decay},
         {"params": g["adamw_nowd"], "lr": lr, "weight_decay": 0.0}],
        betas=(0.9, 0.95), eps=1e-20)
    sinkhorn = SinkhornBalanced(sinkhorn_groups, lr=lr)
    return muon, adamw, sinkhorn
