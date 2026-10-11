"""多模态训练（ViT + Aligner + LLM 联合）。

照 minimind-v 的接法：图像 → ViT → Aligner → 覆盖 `<image>` 占位符位置的 embedding → 主干前向。
与旧版的关键区别：**图像真的进模型**（batch["image"] 被编码并注入），ViT/Aligner 不是死参数。

loss：与纯文本一致（只对非 -100 位置），但图像位置参与前向、其梯度回传到 ViT/Aligner。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train.data import MultimodalDataset
from model.model import Model
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer
from model.layers.vision import Aligner, ViT


def images_to_embeds(images, vit, aligner, patch_size, image_size, device, dtype):
    """把一批图像编码成视觉 token 序列。

    输入：images PIL 列表（可能含 None）；vit；aligner；patch_size；image_size；device；dtype。
    输出：list[Tensor]，每个是 [n_tokens, lang_dim] 或 None。
    预期行为：resize 到 image_size → 切 patch → ViT → Aligner（梯度保留，回传 ViT/Aligner）。
    """
    n_side = image_size // patch_size          # 256/14 = 18
    side = n_side * patch_size                 # 252（patch 整数倍）
    out = []
    for img in images:
        if img is None:
            out.append(None)
            continue
        img = img.resize((side, side), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0       # [side, side, 3]
        # 切成 n_side×n_side 个 patch，每个 [3, ps, ps]
        p = arr.reshape(n_side, patch_size, n_side, patch_size, 3)
        p = p.transpose(0, 2, 4, 1, 3).reshape(n_side * n_side, 3, patch_size, patch_size)
        patches = torch.from_numpy(p).to(device, dtype)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16):
            feats = vit(patches, n_side, n_side)
            embeds = aligner(feats, n_side, n_side)
        out.append(embeds)
    return out


def collate_fn(batch):
    """把样本列表拼成 batch；图像保持为 list（每张尺寸一致但允许 None）。

    输入：样本 dict 列表。
    输出：{"input_ids":[B,L], "labels":[B,L], "images": list[PIL|None]}。
    """
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    images = [b["image"] for b in batch]
    return {"input_ids": input_ids, "labels": labels, "images": images}


def train_epoch(epoch, loader, model, vit, aligner, muon, adamw, sinkhorn, vit_opt, args, start_step=0):
    """跑一个 epoch（图像注入 + 文本交叉熵）。

    输入：epoch；loader；model；vit；aligner；主干优化器 muon/adamw/sinkhorn；ViT/Aligner 优化器 vit_opt；args；start_step。
    输出：本 epoch 平均 loss。
    预期行为：每步把图像编码成 embeds 注入主干，反向传播更新 ViT/Aligner/主干。
    """
    model.train(); vit.train(); aligner.train()
    total_loss, steps = 0.0, 0
    for step, batch in enumerate(loader, start=start_step):
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        image_embeds = images_to_embeds(
            batch["images"], vit, aligner,
            args.patch_size, args.image_size, args.device, torch.bfloat16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, image_embeds=image_embeds)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
            loss = loss / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(vit.parameters()) + list(aligner.parameters()),
                args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step(); vit_opt.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad(); vit_opt.zero_grad()
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
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--router_bias_rate", type=float, default=1e-3)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--num_workers", type=int, default=0)
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
    vit = ViT(cfg.vision).to(args.device)
    aligner = Aligner(cfg.vision, cfg.text.hidden_size).to(args.device)

    image_token_len = (args.image_size // args.patch_size // cfg.vision.downsample_ratio) ** 2
    ds = MultimodalDataset(args.data_path, tokenizer, max_length=args.seq_len,
                           image_token_len=image_token_len, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    # 主干用参考实现优化器分组；ViT/Aligner 用 AdamW（新增模块）
    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    vit_opt = torch.optim.AdamW(list(vit.parameters()) + list(aligner.parameters()),
                                lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(epoch, loader, model, vit, aligner, muon, adamw, sinkhorn, vit_opt, args)
        print(f"epoch {epoch} done, avg loss {loss:.4f}")

    torch.save({"model": model.state_dict(), "vit": vit.state_dict(),
                "aligner": aligner.state_dict()}, os.path.join(args.save_dir, "vlm.pth"))
    print("多模态训练完成，已保存 vlm.pth")


if __name__ == "__main__":
    main()
