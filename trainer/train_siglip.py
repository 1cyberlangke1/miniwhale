"""SigLIP 对比预训练入口（报告 §4.2.2 第一阶段）。

按参考实现两阶段流程：**先**用 SigLIP sigmoid 对比损失把 ViT 与文本对齐（本脚本），
**再**接进语言骨干做自回归微调（`train_vlm.py`）。本脚本产出 `out/vit_siglip.pth`
（只存视觉编码器——参考实现「丢弃 LLM、只留优化后的视觉编码器」）。

风格照 minimind：argparse + train_epoch + __main__。
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
from model.train.data import ImageTextPairDataset
from model.train.siglip import SigLIP
from model.train.compile import maybe_compile
from model.train.optim import make_optimizers, set_base_lr
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


def collate_fn(batch):
    """过滤 None（无图/空 caption 的脏行），把有效样本拼成一批。

    输入：样本 dict 列表（可能含 None）。
    输出：{"patches": list, "grid": list, "text_ids": [B,L], "text_mask": [B,L]}；全空返回 None。
    预期行为：patches 每张图形状不同，保持为 list；文本 pad 到同长后 stack。
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return {
        "patches": [b["patches"] for b in batch],
        "grid": [b["grid"] for b in batch],
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "text_mask": torch.stack([b["text_mask"] for b in batch]),
    }


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0,
                max_steps=None):
    """跑一个 epoch（SigLIP 对比）。

    输入：epoch；loader；model（SigLIP 双塔）；三个优化器；args；start_step；max_steps（步数上限，None=跑满）。
    输出：本 epoch 平均 loss。
    预期行为：bf16 autocast 前向；pairwise sigmoid loss；梯度累积 + 裁剪；
              每步按 warmup+cosine 更新三个优化器的 lr。max_steps=0 时一步都不跑。
    """
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=start_step):
        if max_steps is not None and step - start_step >= max_steps:
            break
        # 对比损失需要 batch 内至少 2 个样本（1×1 的 logits 退化）
        if batch is None or batch["text_ids"].size(0) < 2:
            continue
        t_step0 = time.time()
        lr = get_lr(step, args.total_steps, args.warmup_steps, args.learning_rate, args.min_lr)
        set_base_lr((muon, adamw, sinkhorn), lr)
        text_ids = batch["text_ids"].to(args.device)
        text_mask = batch["text_mask"].to(args.device)
        patches = [p.to(args.device) for p in batch["patches"]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(patches, batch["grid"], text_ids, text_mask)
            loss = SigLIP.loss(logits) / args.accumulation_steps
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if step % args.log_interval == 0:
            # 打印**瞬时**每步耗时（非累计均值），才能看清稳态速度
            now = time.time()
            print(f"epoch {epoch} step {step} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        if args.save_interval and (step + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(model.image.vit.state_dict(),
                       os.path.join(args.save_dir, f"vit_siglip_s{step+1}.pth"))
            print(f"checkpoint saved at step {step+1}", flush=True)
    return total_loss / max(steps, 1)


def main():
    ap = argparse.ArgumentParser(description="miniwhale ViT SigLIP 对比预训练")
    ap.add_argument("--data_path", default="dataset/sft_i2t_mini.parquet")
    ap.add_argument("--source", default="conversations",
                    choices=["conversations", "flickr30k"],
                    help="数据格式：conversations（列 conversations+image_bytes）"
                         "或 flickr30k（列 image+caption 列表）")
    ap.add_argument("--image_size", type=int, default=224,
                    help="把图统一 resize 到 N×N（0=保持原尺寸）；参考实现 SigLIP 阶段为 224")
    ap.add_argument("--epochs", type=int, default=0, help="训练轮数（默认 0=不跑）")
    ap.add_argument("--steps", type=int, default=0,
                    help="按步数训练（默认 0）；与 --epochs 同时指定时 epoch 优先")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--caption_len", type=int, default=64, help="caption 最大 token 数")
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5, help="余弦衰减的最低 lr")
    ap.add_argument("--warmup_steps", type=int, default=100, help="线性预热步数")
    ap.add_argument("--total_steps", type=int, default=0,
                    help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--proj_dim", type=int, default=256, help="对比嵌入维")
    ap.add_argument("--text_dim", type=int, default=256, help="文本塔隐藏维")
    ap.add_argument("--text_layers", type=int, default=4, help="文本塔层数")
    ap.add_argument("--text_heads", type=int, default=4, help="文本塔注意力头数")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--save_interval", type=int, default=1000,
                    help="每多少步存一次 checkpoint（0=不存）")
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    ap.add_argument("--compile_mode", type=str, default="default",
                    help='torch.compile 模式（固定 "default"）')
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # 参数保持 fp32 主权重；计算走 bf16 autocast（同其它训练入口）。
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = SigLIP(cfg.text.vocab_size, cfg.vision, proj_dim=args.proj_dim,
                   text_dim=args.text_dim, text_layers=args.text_layers,
                   text_heads=args.text_heads).to(args.device)
    # 优化器分组（Muon/AdamW/Sinkhorn）按名字路由，需读 model.cfg.text.num_attention_heads
    model.cfg = cfg
    model = maybe_compile(model, args.compile, args.compile_mode)

    ds = ImageTextPairDataset(args.data_path, tokenizer, max_length=args.caption_len,
                              vision_cfg=cfg.vision, source=args.source,
                              image_size=args.image_size)
    # 训练模式：steps 与 epochs 默认都为 0。两者互斥——同时指定时 epoch 优先（打印提醒）。
    # 默认全 0 且不传 = 直接报错不跑（防止误触发整轮训练）。
    if not args.steps and not args.epochs:
        ap.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    if args.steps and args.epochs:
        print(f"提醒：--steps={args.steps} 与 --epochs={args.epochs} 同时指定，"
              f"按 epoch 优先（跑 {args.epochs} 轮，忽略 --steps）", flush=True)
        epochs_to_run, max_steps = args.epochs, None
    elif args.steps:
        epochs_to_run, max_steps = 1, args.steps
    else:
        epochs_to_run, max_steps = args.epochs, None
    if not args.total_steps:
        args.total_steps = (max_steps if max_steps
                            else max(len(ds) // args.batch_size, 1) * epochs_to_run)
    print(f"数据集 {len(ds)} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(epochs_to_run):
        ds.set_epoch(epoch)  # 流式数据集：每轮重排 shard 顺序 + shuffle buffer
        loss = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args,
                           max_steps=max_steps)
        print(f"epoch {epoch} done, avg loss {loss:.4f}")

    # 只留视觉编码器（参考实现：丢弃 LLM，保留优化后的 ViT）
    torch.save(model.image.vit.state_dict(), os.path.join(args.save_dir, "vit_siglip.pth"))
    # 另存**完整双塔**（ViT + 投影头 + 文本塔 + 可学习温度/偏置）：参考实现只留 ViT，但保留
    # 全模型才能对训好的编码器做**检索评测**（image→text 命中率），否则无从判断对比质量。
    torch.save(model.state_dict(), os.path.join(args.save_dir, "siglip_full.pth"))
    print("SigLIP 对比预训练完成，已保存 vit_siglip.pth（仅视觉编码器）"
          "+ siglip_full.pth（完整双塔，供检索评测）")


if __name__ == "__main__":
    main()
