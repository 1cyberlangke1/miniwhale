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

    输入：g [A,B]（2D）或 [N,A,B]（3D，N 个矩阵各自正交化）。
    输出：与 g 同形状的正交化矩阵（近似 U·Vᵀ）。
    预期行为：转置成「行数 ≤ 列数」后按谱范数 ≤1 归一，跑五次迭代，再转回原朝向。
              3D 时用 bmm/baddbmm 一次算完所有矩阵——与逐矩阵循环**逐位等价**
              （每个矩阵独立、无跨矩阵归约；已验证 head-wise 情形 max diff = 0.0），
              kernel 数从 N×15 降到 15，是 head-wise Muon 提速的关键。
    """
    transposed = g.size(-2) > g.size(-1)
    if transposed:
        g = g.transpose(-2, -1)
    x = g.bfloat16()
    if x.dim() == 2:
        # 2D 走 mm/addmm（原路径，逐位不变；bmm 对单矩阵会用不同 kernel）
        x = x / x.norm().clamp_min(eps)
        for _ in range(steps):
            gram = x @ x.T
            gram = torch.addmm(gram, gram, gram, beta=_NS_B, alpha=_NS_C)
            x = torch.addmm(x, gram, x, beta=_NS_A)
    else:
        # 3D 批量：bmm/baddbmm，与逐矩阵 mm 循环逐位等价（已验证）
        x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
        for _ in range(steps):
            gram = torch.bmm(x, x.transpose(-2, -1))
            gram = torch.baddbmm(gram, gram, gram, beta=_NS_B, alpha=_NS_C)
            x = torch.baddbmm(x, gram, x, beta=_NS_A)
    return x.transpose(-2, -1) if transposed else x


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

    # 批量正交化不逐位一致的 shape（实测）：(512,256) 的 bmm 与逐矩阵 mm 累加顺序不同
    # （diff 2.5e-1），必须逐个算；其余 shape 批量 bmm 与逐矩阵**逐位一致**（diff 0.0）。
    _NO_BATCH_SHAPES = {(512, 256)}

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
            ns_steps = group["ns_steps"]
            # 1) Nesterov 动量 + 按 shape 分桶（同 shape 的参数堆成一个 3D 张量批量正交化）。
            #    grouped Muon 是公认优化：同 shape 桶批量 bmm 与逐矩阵 mm 逐位一致
            #    （除 _NO_BATCH_SHAPES），kernel 数从 N×15 降到 桶数×15。
            buckets: dict[tuple, list] = {}
            singles = []  # head-wise（各自 [n_heads,sub,B] 批量）或不可批量 shape
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                update = g.lerp(buf, momentum) if group["nesterov"] else buf.clone()
                n_heads = self._headwise.get(id(p), 1)
                if n_heads > 1 or tuple(update.shape) in self._NO_BATCH_SHAPES:
                    singles.append((p, update, n_heads))
                else:
                    buckets.setdefault(tuple(update.shape), []).append((p, update))
            # 2) 同 shape 桶批量正交化（>1 个才批量；单元素桶退化为逐个）
            for key, items in buckets.items():
                if len(items) > 1:
                    stacked = torch.stack([u for _, u in items])
                    orthos = _newton_schulz(stacked, ns_steps)
                    for (p, _), o in zip(items, orthos):
                        a, b_ = p.shape[:2]
                        adj = lr * lr_scale * math.sqrt(max(a, b_))
                        p.mul_(1 - lr * wd)
                        p.add_(o.to(p.dtype), alpha=-adj)
                else:
                    p, u = items[0]
                    o = _newton_schulz(u, ns_steps)
                    a, b_ = p.shape[:2]
                    adj = lr * lr_scale * math.sqrt(max(a, b_))
                    p.mul_(1 - lr * wd)
                    p.add_(o.to(p.dtype), alpha=-adj)
            # 3) head-wise：按 head 拆开各自正交化（reshape [n_heads,sub,B] 一次 bmm 算完）
            for p, update, n_heads in singles:
                if n_heads > 1:
                    sub = update.size(0) // n_heads
                    batched = update.view(n_heads, sub, update.size(1))
                    ortho = _newton_schulz(batched, ns_steps).view_as(update)
                    a, b_ = sub, update.size(1)
                else:
                    ortho = _newton_schulz(update, ns_steps)
                    a, b_ = update.shape[:2]
                adj = lr * lr_scale * math.sqrt(max(a, b_))
                p.mul_(1 - lr * wd)
                p.add_(ortho.to(p.dtype), alpha=-adj)
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
    # lr_mult 记录每组相对「当前基准 lr」的倍数：调度器只更新基准 lr，按此倍数缩放，
    # 否则 engram 的 ×5 会被每步 `g["lr"] = lr` 抹平（等于 Engram 没按参考实现配方训）。
    sinkhorn_groups = []
    if sinkhorn_other:
        sinkhorn_groups.append({"params": sinkhorn_other, "lr": lr, "lr_mult": 1.0})
    if sinkhorn_engram:
        sinkhorn_groups.append({"params": sinkhorn_engram, "lr": lr * engram_lr_mult,
                                "lr_mult": engram_lr_mult})

    muon = Muon(
        [{"params": g["muon"], "lr": lr, "weight_decay": weight_decay, "lr_scale": 0.18,
          "lr_mult": 1.0}],
        lr=lr, momentum=0.95, weight_decay=weight_decay, lr_scale=0.18,
        headwise=g["headwise"])
    adamw = torch.optim.AdamW(
        [{"params": g["adamw_wd"], "lr": lr, "weight_decay": weight_decay, "lr_mult": 1.0},
         {"params": g["adamw_nowd"], "lr": lr, "weight_decay": 0.0, "lr_mult": 1.0}],
        betas=(0.9, 0.95), eps=1e-20)
    sinkhorn = SinkhornBalanced(sinkhorn_groups, lr=lr)
    return muon, adamw, sinkhorn


def set_base_lr(optimizers, lr: float):
    """按当前基准 lr 更新各优化器参数组的 lr（保留每组的 lr_mult 倍数）。

    输入：optimizers（Muon/AdamW/Sinkhorn 等）；lr 当前基准学习率。
    输出：就地更新各组 `lr`。
    预期行为：`g["lr"] = lr * g.get("lr_mult", 1.0)`——这样 Engram 组始终是基准的 ×5，
              不被调度器抹平；无 lr_mult 键的组退化为 lr（向后兼容）。
    """
    for opt in optimizers:
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
