"""DSpark 训练 loss —— 逐行对按参考实现 DeepSpec 的 compute_dspark_loss。

参考实现（deepseek-ai/DeepSpec, deepspec/modeling/dspark/loss.py）总 loss：

    ce_alpha * CE + l1_alpha * L1 + confidence_alpha * BCE

各项定义（参考实现公式，本文件原样复刻）：
- CE：draft logits 对 target_ids 的交叉熵，按 loss_weight_mask 加权。
- L1：draft 与 target 的 softmax 分布 L1 距离（蒸馏），同样加权。
      l1_dist_per_token = |softmax(draft) - softmax(target)| 沿 vocab 求和。
- confidence：BCE(confidence_pred, accept_rate)，目标是 **接受率**（非命中率）：
      accept_rate = 1 - 0.5 * |softmax(draft) - softmax(target)|_1，clamp 到 [0,1]。
- loss_weight_mask = eval_mask * exp(-pos / loss_decay_gamma)（块内位置衰减）。

参考实现超参（config/dspark/dspark_qwen3_4b.py）：gamma=4.0, ce=0.1, l1=0.9, confidence=1.0。

本文件去掉参考实现的分布式归约与指标上报，只保留数值公式；单卡下 den 即本地求和。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# L1 蒸馏沿 anchor 维的分块大小。整块要物化两份 [B,N,K,V] float32 概率，
# 8GB 上会 OOM；沿 N 切**逐位等价**（softmax 每个 token 独立）。
_L1_CHUNK = 64


@dataclass
class DSparkForwardOutput:
    """一次 DSpark 训练前向的输出（按参考实现 DSparkForwardOutput 的字段）。

    形状符号：B 批；N anchor 块数；K block_size；V 词表。
    """

    draft_logits: torch.Tensor          # [B, N, K, V]  draft 各位置 logits
    target_ids: torch.Tensor            # [B, N, K]     各位置的真实下一 token
    eval_mask: torch.Tensor             # [B, N, K]     有效前缀掩码（bool/int）
    block_keep_mask: torch.Tensor       # [B, N]        有效 anchor 块掩码
    confidence_pred: torch.Tensor | None = None       # [B, N, K]  置信度 logit
    aligned_target_logits: torch.Tensor | None = None  # [B, N, K, V]  目标模型 logits


def _build_loss_weight_mask(eval_mask, block_size, device, loss_decay_gamma):
    """参考实现 _build_loss_weight_mask：eval_mask * exp(-pos / gamma)。

    输入：eval_mask [B,N,K]；block_size；device；loss_decay_gamma（None/<=0 则不衰减）。
    输出：[B,N,K] float32 权重。
    """
    w = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        pos = torch.arange(block_size, device=device).view(1, 1, -1)
        w = w * torch.exp(-pos.float() / float(loss_decay_gamma))
    return w


def _accept_rate_3d(outputs: DSparkForwardOutput, aligned_target_logits):
    """参考实现 _compute_accept_rate_3d：1 - 0.5*L1(draft, target)，clamp[0,1]。

    输入：outputs；aligned_target_logits [B,N,K,V] 或 None。
    输出：[B,N,K] 或 None。
    """
    if aligned_target_logits is None:
        return None
    # 只作为 BCE 的 **target**（调用处 detach）→ 不需要梯度；沿 anchor 维分块，
    # 避免同时物化两份 [B,N,K,V] float32（8GB 上会 OOM）。逐位等价（softmax 每 token 独立）。
    outs = []
    with torch.no_grad():
        for s in range(0, outputs.draft_logits.size(1), _L1_CHUNK):
            e = min(s + _L1_CHUNK, outputs.draft_logits.size(1))
            dp = torch.softmax(outputs.draft_logits[:, s:e].float(), dim=-1)
            tp = torch.softmax(aligned_target_logits[:, s:e].float(), dim=-1)
            outs.append(1.0 - 0.5 * (dp - tp).abs().sum(dim=-1))
    return torch.cat(outs, dim=1).clamp_(0.0, 1.0)


def compute_dspark_loss(
    outputs: DSparkForwardOutput,
    *,
    loss_decay_gamma: float | None = 4.0,
    ce_loss_alpha: float = 0.1,
    l1_loss_alpha: float = 0.9,
    confidence_head_alpha: float = 1.0,
):
    """参考实现 compute_dspark_loss 的单卡版（数值公式一致）。

    输入：outputs（见 DSparkForwardOutput）；三个 alpha；位置衰减 gamma。
    输出：(总 loss（可反传）, {"ce","l1","conf"} 三项**未加权**归一值 dict（detach）)。
    预期行为：CE 与 L1 与 confidence 各按 mask 加权后归一，再按 alpha 线性组合；
              分项单独返回供日志/曲线观测（参考实现 add_metric 同口径）。
    """
    draft_logits = outputs.draft_logits
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    wmask = _build_loss_weight_mask(
        eval_mask=outputs.eval_mask, block_size=block_size, device=device,
        loss_decay_gamma=loss_decay_gamma)
    flat_w = wmask.reshape(-1)

    # CE：draft 对真实下一 token 的加权交叉熵
    ce_per_token = F.cross_entropy(
        draft_logits.reshape(-1, vocab_size), outputs.target_ids.reshape(-1),
        reduction="none")
    ce_num = (ce_per_token * flat_w).sum()
    ce_den = flat_w.sum()

    # L1 蒸馏：与目标模型分布的距离（需要 aligned_target_logits）
    aligned = outputs.aligned_target_logits
    l1_num = draft_logits.new_zeros(())
    l1_den = draft_logits.new_zeros(())
    if l1_loss_alpha > 0 and aligned is not None:
        # **沿 anchor 维分块**：draft_probs/target_probs 是 [B,N,K,V] 的 float32
        # （8×512×4×6400×4B ≈ 419MB 各一份，反向再存中间量），8GB 上会 OOM。
        # softmax 对每个 token 独立 → 沿 N 切与整块**逐位等价**（已实测 torch.equal）。
        l1_num = draft_logits.new_zeros(())
        for s in range(0, draft_logits.size(1), _L1_CHUNK):
            e = min(s + _L1_CHUNK, draft_logits.size(1))
            dp = torch.softmax(draft_logits[:, s:e].float(), dim=-1)
            tp = torch.softmax(aligned[:, s:e].float(), dim=-1)
            l1_per_token = (dp - tp).abs().sum(dim=-1)
            l1_num = l1_num + (l1_per_token * wmask[:, s:e]).sum()
        l1_den = wmask.sum()

    # confidence：BCE(confidence_pred, accept_rate)
    conf_num = draft_logits.new_zeros(())
    conf_den = draft_logits.new_zeros(())
    has_conf = outputs.confidence_pred is not None
    if has_conf:
        accept = _accept_rate_3d(outputs, aligned)
        assert accept is not None, "confidence 头需要 aligned_target_logits"
        conf_err = F.binary_cross_entropy_with_logits(
            outputs.confidence_pred.float(), accept.detach(), reduction="none") * wmask
        conf_num = conf_err.sum()
        conf_den = wmask.sum()

    ce_loss = ce_num / (ce_den + 1e-6)
    l1_loss = l1_num / (l1_den + 1e-6) if l1_den.item() > 0 else draft_logits.new_zeros(())
    conf_loss = conf_num / (conf_den + 1e-6) if has_conf else draft_logits.new_zeros(())
    total = (ce_loss_alpha * ce_loss
             + l1_loss_alpha * l1_loss
             + confidence_head_alpha * conf_loss)
    # 分项（**未加权**的归一化值，参考实现 add_metric 同口径）：训练日志/曲线据此看清
    # CE / L1 / confidence 各自趋势。只返回合成标量时，loss 平台期无法归因是哪项卡住。
    parts = {"ce": ce_loss.detach(), "l1": l1_loss.detach(), "conf": conf_loss.detach()}
    return total, parts


__all__ = ["DSparkForwardOutput", "compute_dspark_loss"]
