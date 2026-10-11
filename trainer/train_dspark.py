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
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train.data import PretrainDataset
from model.train.dspark_loss import compute_dspark_loss
from model.model import Model
from model.tokenizer import load_tokenizer


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
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)
    trainable = freeze_backbone(model)
    print(f"DSpark 可训练参数 {sum(p.numel() for p in trainable)/1e6:.2f}M")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    model.train()

    tc = cfg.text
    it = iter(loader)
    for step in range(args.steps):
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
        if step % args.log_interval == 0:
            print(f"dspark step {step}/{args.steps} loss {loss.item():.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "dspark.pth"))
    print("DSpark 训练完成")


if __name__ == "__main__":
    main()
