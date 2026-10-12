"""DSpark 训练（参考实现二阶段：冻结 backbone，只训 draft 头）。

技术报告 §2.4.3：
  - backbone 预训练**不含** DSpark
  - 冻结 backbone，只训 DSpark（本脚本）
  - post-training 同训但梯度不回传 backbone

训练前向与 loss 逐行对按参考实现 DeepSpec（deepseek-ai/DeepSpec）：
  - 前向：teacher-forced 并行（Qwen3DSparkModel.forward）——采 anchor、噪声嵌入、并行出草稿，
    markov 偏置用**真实前驱 token**（非自回归采样）。
  - loss：compute_dspark_loss —— CE(0.1) + L1 蒸馏(0.9) + confidence BCE(1.0)，位置衰减 exp(-i/4)。
    confidence 目标是接受率 accept_rate，不是命中率。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# **必须在 import torch 之前**设（torch._inductor.config 首次 import 时读 env）。
# compile_threads=1：Windows 上多线程编译会走 SubprocPool 的 pass_fds → CPython 断言崩溃。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train.data import PretrainDataset
from model.train.dspark_loss import compute_dspark_loss
from model.model import Model
from model.train.compile import maybe_compile
from model.tokenizer import load_tokenizer


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """余弦学习率调度（带线性 warmup）。

    输入：step 当前步；total_steps 总步数；warmup_steps 预热步数；base_lr 峰值 lr；min_lr 最低 lr。
    输出：本步的 lr。
    预期行为：前 warmup_steps 步从 base_lr/10 线性升到 base_lr；之后按余弦从 base_lr 衰减到 min_lr。
    """
    if step < warmup_steps:
        return base_lr * (0.1 + 0.9 * step / max(warmup_steps, 1))
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cos


def freeze_backbone(model: Model):
    """冻结主干，只留 DSpark 相关参数可训练。

    输入：model。
    输出：可训练参数列表。
    预期行为：只有 `dspark.*`（含首块 main_proj/main_norm、末块 markov/confidence 头）
              requires_grad=True；主干（embed/head/backbone）全部冻结。
    """
    trainable = []
    for name, p in model.named_parameters():
        keep = name.startswith("dspark.")
        p.requires_grad_(keep)
        if keep:
            trainable.append(p)
    return trainable


def main():
    ap = argparse.ArgumentParser(description="miniwhale DSpark 训练")
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--ckpt", default=None, help="已训练的 backbone 权重（可选）")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4, help="参考实现 lr=6e-4")
    ap.add_argument("--min_lr", type=float, default=6e-5, help="余弦衰减的最低 lr")
    ap.add_argument("--warmup_steps", type=int, default=100, help="线性预热步数")
    ap.add_argument("--router_bias_rate", type=float, default=1e-3,
                    help="noaux_tc 纠正 bias 的更新步长（非梯度）")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=1000,
                    help="每多少步存一次 checkpoint（0=不存）")
    ap.add_argument("--keep_last_n", type=int, default=3,
                    help="只保留最近 N 个 checkpoint（0=全留）")
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    ap.add_argument("--compile_mode", type=str, default="default",
                    help='torch.compile 模式（固定 "default"）')
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)
    model = maybe_compile(model, args.compile, args.compile_mode)
    trainable = freeze_backbone(model)
    print(f"DSpark 可训练参数 {sum(p.numel() for p in trainable)/1e6:.2f}M")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    # 数据集是 IterableDataset（流式），自带缓冲打乱，DataLoader 不能再传 shuffle=True
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    model.train()

    tc = cfg.text
    it = iter(loader)
    for step in range(args.steps):
        lr = get_lr(step, args.steps, args.warmup_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        loss_mask = (labels != -100).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 主干前向：拿 target 层 hidden 与全序列 logits（后者供 L1 蒸馏 / accept_rate）
            logits, _ = model(ids)
            context_hidden = model._last_main_hidden
            out = model.dspark.forward_train(ids, context_hidden, loss_mask,
                                             aligned_target_logits=logits)
            loss = compute_dspark_loss(
                out,
                loss_decay_gamma=tc.dspark_loss_decay_gamma,
                ce_loss_alpha=tc.dspark_ce_loss_alpha,
                l1_loss_alpha=tc.dspark_l1_loss_alpha,
                confidence_head_alpha=tc.dspark_confidence_head_alpha,
            )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        # noaux_tc 纠正 bias 按负载均衡规则更新（非梯度；主干 MoE 冻结但仍需均衡）
        model.update_router_bias(args.router_bias_rate)
        if step % args.log_interval == 0:
            print(f"dspark step {step}/{args.steps} loss {loss.item():.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）
        if args.save_interval and (step + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"dspark_s{step+1}.pth")
            torch.save(model.state_dict(), path)
            if args.keep_last_n:
                cks = sorted(
                    (os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir)
                     if f.startswith("dspark_") and f.endswith(".pth")),
                    key=os.path.getmtime)
                for old in cks[:-args.keep_last_n]:
                    os.remove(old)
            print(f"checkpoint saved at step {step+1}", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "dspark.pth"))
    print("DSpark 训练完成")


if __name__ == "__main__":
    main()
