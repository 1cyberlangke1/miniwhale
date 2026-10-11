"""V4.1 式 MoE。

与常见 MoE 的差异：
- 打分用 sqrt(softplus(x))，非 softmax（值域 [0,∞)，靠 top-sum 归一）
- 负载均衡用可学习纠正 bias（noaux_tc），**不用 aux loss**
- bias 只影响专家选择，**不缩放路由权重**（权重取未加 bias 的原始分）
- 每 token 额外过 1 个共享专家
- 专家 SwiGLU 带 clamp（up 两侧、gate 上界），为 fp8/fp4 激活保范围
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from model.nn.norm import RMSNorm  # noqa: F401


class Expert(nn.Module):
    """一个 SwiGLU FFN 专家。

    输入：x [*, dim]；可选 weights [*, 1] 路由权重。
    输出：同前导维度、末维 dim。
    预期行为：up 分支两侧 clamp、gate 分支仅上界 clamp；中间以 fp32 计算。
    """

    def __init__(self, dim: int, inter_dim: int, swiglu_limit: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = weights * h
        return self.w2(h.to(dtype))


class Gate(nn.Module):
    """MoE 门控：sqrtsoftplus 打分 + noaux_tc 纠正 bias。

    输入：x [n, dim]；可选 image_mask [n]（图像 token 用单独 bias）。
    输出：(weights [n, topk], indices [n, topk])。
    预期行为：bias 只参与选择；权重取原始分并做 top-sum 归一（+1e-20）；fp32 计算。
    """

    def __init__(self, text_cfg):
        super().__init__()
        self.dim = text_cfg.hidden_size
        self.topk = text_cfg.num_experts_per_tok
        self.n_routed = text_cfg.n_routed_experts
        self.scoring_func = text_cfg.scoring_func
        self.norm_topk_prob = text_cfg.norm_topk_prob
        self.route_scale = text_cfg.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(self.n_routed, self.dim))
        self.bias = nn.Parameter(torch.zeros(self.n_routed, dtype=torch.float32))
        # 负载均衡用的每专家 token 计数（非梯度更新，见 update_bias）
        self.register_buffer("tokens_per_expert", torch.zeros(self.n_routed, dtype=torch.float32),
                             persistent=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """门控权重正态初始化（std=0.02），纠正 bias 置零。

        输入：无。
        输出：无（原地）。
        预期行为：避免 torch.empty 的垃圾初值污染路由。
        """
        w = torch.randn(self.n_routed, self.dim, dtype=torch.float32) * 0.02
        self.weight.copy_(w.to(self.weight.dtype))
        self.bias.zero_()
        self.tokens_per_expert.zero_()

    @torch.no_grad()
    def update_bias(self, rate: float):
        """按负载均衡规则更新纠正 bias（非梯度）。

        输入：rate 更新步长。
        输出：无（原地）。
        预期行为：过载专家（token 数 > 均值）减 bias，欠载加 bias；
                  这是 noaux_tc 的核心，bias **不走梯度**。
        """
        if rate <= 0:
            return
        avg = self.tokens_per_expert.mean()
        offset = avg - self.tokens_per_expert
        self.bias.add_(torch.sign(offset) * rate)
        self.tokens_per_expert.zero_()

    def forward(self, x: torch.Tensor):
        scores = F.linear(x.float(), self.weight.float())
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        # bias 只挑专家，不缩放：权重取未加 bias 的原始分
        indices = (scores + self.bias).topk(self.topk, dim=-1)[1]
        if self.training:
            with torch.no_grad():  # 累计每专家被选次数（供 update_bias）
                self.tokens_per_expert += torch.bincount(
                    indices.flatten(), minlength=self.n_routed).float()
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices


class MoE(nn.Module):
    """Top-k 路由专家 + 1 个共享专家。

    输入：x [B, L, dim]。
    输出：同形状张量。
    预期行为：按路由权重加权求和选中专家的输出，再加共享专家输出。
    """

    def __init__(self, text_cfg):
        super().__init__()
        self.dim = text_cfg.hidden_size
        self.n_routed = text_cfg.n_routed_experts
        self.n_shared = text_cfg.n_shared_experts
        # 对照实验的 dense 基线：无路由专家也无共享专家时退化为单个 dense FFN
        self.dense = self.n_routed == 0
        inter = text_cfg.moe_intermediate_size
        lim = text_cfg.swiglu_limit
        if self.dense:
            self.dense_ffn = Expert(self.dim, inter, lim)
        else:
            self.gate = Gate(text_cfg)
            self.experts = nn.ModuleList([Expert(self.dim, inter, lim) for _ in range(self.n_routed)])
            self.shared_experts = nn.ModuleList([Expert(self.dim, inter, lim) for _ in range(self.n_shared)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dense:
            return self.dense_ffn(x)
        b, L, d = x.shape
        flat = x.reshape(-1, d)
        weights, indices = self.gate(flat)

        # 累加器用 fp32（一致），最后转回输入 dtype
        out = torch.zeros_like(flat, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            out[idx] += expert(flat[idx], weights[idx, top, None].float())

        for shared in self.shared_experts:
            out = out + shared(flat).float()
        return out.view(b, L, d).to(x.dtype)
