"""DSpark 训练（参考实现二阶段：冻结 backbone，只训 draft 头）。

技术报告 §2.4.3：
  - backbone 预训练**不含** DSpark
  - 冻结 backbone，只训 DSpark（本脚本）
  - post-training 同训但梯度不回传 backbone

loss：draft 每个位置的交叉熵（草稿 token vs 真实下一 token）+ 置信度 BCE。
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


def dspark_loss(out_ids, logits, confidence, target, block_size, ignore_index=-100):
    """DSpark 训练 loss：draft 交叉熵 + 置信度 BCE。

    输入：out_ids [B, block+1]；logits [B, block, V]；confidence [B, block]；
          target [B, block]（草稿各位置的真实下一 token）。
    输出：标量 loss。
    预期行为：draft 位置 i 预测 target[:, i]；置信度对命中与否做 BCE。
    """
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=ignore_index)
    with torch.no_grad():
        hit = (out_ids[:, 1:1 + block_size] == target).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        confidence.reshape(-1), hit.reshape(-1))
    return ce + bce


def main():
    ap = argparse.ArgumentParser(description="miniwhale DSpark 训练")
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--ckpt", default=None, help="已训练的 backbone 权重（可选）")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
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

    it = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 参考实现流程：先整段 prefill（填主干与 draft 的窗口缓存），再在末位置 decode 出草稿
            L = ids.size(1)
            model(ids)                                        # 主干前向填 _last_main_hidden
            model.forward_spec(ids[:, -1], start_pos=0)       # draft prefill：填窗口缓存
            out_ids, logits, conf = model.forward_spec(ids[:, -1], start_pos=L)
            # 草稿位置 i 的真实下一 token 是 labels[:, L+i]（末尾不足则忽略）
            bs = cfg.text.dspark_block_size
            tgt = labels[:, L - 1:L - 1 + bs]
            if tgt.size(1) < bs:                              # 末尾补 -100（忽略）
                pad = tgt.new_full((tgt.size(0), bs - tgt.size(1)), -100)
                tgt = torch.cat([tgt, pad], dim=1)
            loss = dspark_loss(out_ids, logits, conf, tgt, bs)
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
