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


class _AuxLossScaler(torch.autograd.Function):
    """把 MoE 序列级平衡损失挂到模块输出上：前向透传，反向以**单位梯度**注入其梯度。

    预期行为：前向返回 output 本身；反向对 output 的梯度原样透传，对 aux_loss 的梯度置 1，
              使 aux_loss 在后续 backward 中按自身权重（α）参与梯度。与参考实现 AddAuxiliaryLoss 同构。
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (aux_loss,) = ctx.saved_tensors
        return grad_output, torch.ones_like(aux_loss)


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
        self.gate_temp = getattr(text_cfg, "gate_temp", 1.0)
        self.aux_loss_alpha = getattr(text_cfg, "seq_aux_loss_alpha", 1e-4)

        self.weight = nn.Parameter(torch.empty(self.n_routed, self.dim))
        self.bias = nn.Parameter(torch.zeros(self.n_routed, dtype=torch.float32))
        # 图像 span 用单独的路由 bias（参考实现 noaux_tc_for_vl）；仅 vision_enabled 时存在
        self.vision_enabled = getattr(text_cfg, "vision_enabled", False)
        self.bias_vl = (nn.Parameter(torch.zeros(self.n_routed, dtype=torch.float32))
                        if self.vision_enabled else None)
        # 负载均衡用的每专家 token 计数（非梯度更新，见 update_bias）。
        # 图像 token 用单独计数（参考实现 §2.1.1：两套 bias 按各自模态的负载独立更新）。
        self.register_buffer("tokens_per_expert", torch.zeros(self.n_routed, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("tokens_per_expert_vl", torch.zeros(self.n_routed, dtype=torch.float32),
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
        self.tokens_per_expert_vl.zero_()

    @torch.no_grad()
    def update_bias(self, rate: float):
        """按负载均衡规则更新纠正 bias（非梯度）。

        输入：rate 更新步长。
        输出：无（原地）。
        预期行为：过载专家（token 数 > 均值）减 bias，欠载加 bias；文本与图像两套 bias
                  按各自模态的负载**独立**更新（参考实现 §2.1.1）。bias **不走梯度**。
        """
        if rate <= 0:
            return
        avg = self.tokens_per_expert.mean()
        offset = avg - self.tokens_per_expert
        self.bias.add_(torch.sign(offset) * rate)
        self.tokens_per_expert.zero_()
        if self.bias_vl is not None:
            avg_vl = self.tokens_per_expert_vl.mean()
            offset_vl = avg_vl - self.tokens_per_expert_vl
            self.bias_vl.add_(torch.sign(offset_vl) * rate)
            self.tokens_per_expert_vl.zero_()

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None):
        """x [n, dim]；image_mask [n]（True=图像 span token，用单独路由 bias）。

        输出：(weights [n, topk], indices [n, topk], aux_loss 标量或 None)。
        预期行为：aux_loss 为序列级平衡损失（noaux_tc 的兜底项，防单序列内极端偏斜），
                  仅训练时计算；权重/选择路径不受其影响。
        """
        # 参考实现：scores = linear(x, weight) / gate_temp
        scores = F.linear(x.float(), self.weight.float()) / self.gate_temp
        if self.scoring_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        # 图像 token 用 bias_vl（参考实现 noaux_tc_for_vl）；bias 只挑专家，不缩放
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        # 序列级平衡损失（α 极小，兜底防极端偏斜，参考实现 V4.1 报告 §训练超参）。
        # 仅在训练时计算；权重/选择路径不受其影响。
        aux_loss = None
        if self.training and self.aux_loss_alpha > 0.0:
            n = indices.size(0)
            n_seq = self.topk * n
            # f_i：专家 i 被选中比例 × n_routed（Eq 18）
            ce = torch.zeros(n, self.n_routed, device=x.device, dtype=scores.dtype)
            ce.scatter_add_(1, indices, torch.ones_like(indices, dtype=scores.dtype))
            f_i = ce.sum(dim=0) * (self.n_routed / n_seq)
            # P_i：专家 i 的平均归一化亲和分（Eq 19-20）
            s_norm = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
            P_i = s_norm.mean(dim=0)
            aux_loss = self.aux_loss_alpha * (f_i * P_i).sum()
        if self.training:
            with torch.no_grad():  # 累计每专家被选次数（供 update_bias）
                if image_mask is not None:
                    # 图像/文本 token 分别计数（参考实现 §2.1.1：两套 bias 各自独立更新）
                    img = image_mask.bool()
                    self.tokens_per_expert += torch.bincount(
                        indices[~img].flatten(), minlength=self.n_routed).float()
                    self.tokens_per_expert_vl += torch.bincount(
                        indices[img].flatten(), minlength=self.n_routed).float()
                else:
                    self.tokens_per_expert += torch.bincount(
                        indices.flatten(), minlength=self.n_routed).float()
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices, aux_loss


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

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.dense:
            return self.dense_ffn(x)
        b, L, d = x.shape
        flat = x.reshape(-1, d)
        weights, indices, aux_loss = self.gate(flat, None if image_mask is None else image_mask.flatten())

        # 累加器用 fp32（一致），最后转回输入 dtype。
        # 全专家张量化：每个专家对**全部** token 算一遍，再按路由权重掩码累加。
        # 未选中的 token 权重为 0，`w2(0·h)=0`，加 0 在 fp32 下精确 → 与参考实现逐专家累加逐位一致。
        # 全静态形状、无 `Tensor.item()`、无数据依赖分支 → torch.compile 可完全融合（不再把
        # Block.forward 切碎）。代价是专家 FLOPs ×n_experts，但模型是 launch-bound（GPU 13%），
        # 少而大的 kernel 反而更快。
        out = torch.zeros_like(flat, dtype=torch.float32)
        for i, expert in enumerate(self.experts):
            sel = indices == i                                   # [n_tok, topk]
            w = torch.where(sel, weights, torch.zeros_like(weights)).sum(-1)  # [n_tok]
            out += expert(flat, w[:, None].float())

        for shared in self.shared_experts:
            out = out + shared(flat).float()
        out = out.view(b, L, d).to(x.dtype)
        # 序列级平衡损失挂到输出（前向值不变，反向注入其梯度）——自动汇入主 loss 的反向图
        if aux_loss is not None:
            out = _AuxLossScaler.apply(out, aux_loss)
        return out
