"""多模态训练（ViT + Aligner + LLM 联合）。

照参考实现 的 VL 接法：图像 → ViT → Aligner → 覆盖图像 span 的 embedding
（IMAGE 槽填 aligner 行，start/end/newline 填学习嵌入）→ 主干前向。图像 span 全部携带
同一 image_token_id，靠 token_types 区分（参考实现机制）。

loss：与纯文本一致（只对非 -100 位置），但图像位置参与前向、其梯度回传 ViT/Aligner。
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
from model.train.data import MultimodalDataset
from model.model import Model
from model.train.compile import maybe_compile
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer


def collate_fn(batch):
    """把样本列表拼成 batch；图像 span 保持为 list（每样本的图像输入）。

    输入：样本 dict 列表。
    输出：{"input_ids":[B,L], "labels":[B,L], "token_types":[B,L], "images": list[list]}。
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    token_types = torch.stack([b["token_types"] for b in batch])
    images = [b["images"] for b in batch]
    return {"input_ids": input_ids, "labels": labels,
            "token_types": token_types, "images": images}


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0):
    """跑一个 epoch（图像注入 + 文本交叉熵）。

    输入：epoch；loader；model（含 ViT/Aligner）；三个主干优化器；args；start_step。
    输出：本 epoch 平均 loss。
    预期行为：每步把图像 span 注入主干，反向传播更新 ViT/Aligner/主干。
    """
    model.train()
    total_loss, steps = 0.0, 0
    for step, batch in enumerate(loader, start=start_step):
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        token_types = batch["token_types"].to(args.device)
        # images 是 list[list[ImageInput]]，仅在有图像时传
        images = batch["images"] if any(batch["images"]) else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, images=images, token_types=token_types)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
            model.update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if step % args.log_interval == 0:
            print(f"epoch {epoch} step {step} loss {total_loss / steps:.4f}", flush=True)
        if args.steps and step + 1 >= args.steps:
            break
    return total_loss / max(steps, 1)


def main():
    ap = argparse.ArgumentParser(description="miniwhale 多模态训练")
    ap.add_argument("--data_path", default="dataset/sft_i2t.parquet")
    ap.add_argument("--ckpt", default=None, help="已训练的主干权重（可选）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1000, help="最多训练步数（0=跑满 epoch）")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--router_bias_rate", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 加速训练（emulate_precision_casts 保数值）")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)
    model = maybe_compile(model, args.compile)
    assert model.vision is not None, "模型未启用视觉塔"

    ds = MultimodalDataset(args.data_path, tokenizer, max_length=args.seq_len,
                           image_token_id=cfg.text.image_token_id, vision_cfg=cfg.vision)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    # 主干 + ViT/Aligner 一起训（参考实现：视觉塔与 LLM 联合优化）
    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args)
        print(f"epoch {epoch} done, avg loss {loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "vlm.pth"))
    print("多模态训练完成，已保存 vlm.pth")


if __name__ == "__main__":
    main()
