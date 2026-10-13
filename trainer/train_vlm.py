"""多模态训练（ViT + Aligner + LLM 联合）。

照参考实现 的 VL 接法：图像 → ViT → Aligner → 覆盖图像 span 的 embedding
（IMAGE 槽填 aligner 行，start/end/newline 填学习嵌入）→ 主干前向。图像 span 全部携带
同一 image_token_id，靠 token_types 区分（参考实现机制）。

loss：与纯文本一致（只对非 -100 位置），但图像位置参与前向、其梯度回传 ViT/Aligner。
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
from model.train.data import MultimodalDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
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


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0,
                max_steps=None):
    """跑一个 epoch（图像注入 + 文本交叉熵）。

    输入：epoch；loader；model（含 ViT/Aligner）；三个主干优化器；args；start_step；max_steps（步数上限，None=跑满）。
    输出：本 epoch 平均 loss。
    预期行为：每步把图像 span 注入主干，反向传播更新 ViT/Aligner/主干；
              每步按 warmup+cosine 更新三个优化器的 lr。max_steps=0 时一步都不跑。
    """
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=start_step):
        if max_steps is not None and step - start_step >= max_steps:
            break
        t_step0 = time.time()
        # 只更新基准 lr，各组按 lr_mult 缩放（保留 Engram ×5，参考实现 §4.2.2）
        lr = get_lr(step, args.total_steps, args.warmup_steps, args.learning_rate, args.min_lr)
        set_base_lr((muon, adamw, sinkhorn), lr)
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
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        loss.backward()
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
            model.update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if step % args.log_interval == 0:
            # 打印**瞬时**每步耗时（非累计均值），才能看清稳态速度
            now = time.time()
            print(f"epoch {epoch} step {step} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）
        if args.save_interval and (step + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"vlm_e{epoch}_s{step+1}.pth")
            torch.save(model.state_dict(), path)
            if args.keep_last_n:
                cks = sorted(
                    (os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir)
                     if f.startswith("vlm_") and f.endswith(".pth")),
                    key=os.path.getmtime)
                for old in cks[:-args.keep_last_n]:
                    os.remove(old)
            print(f"checkpoint saved at step {step+1}", flush=True)
    return total_loss / max(steps, 1)


def main():
    ap = argparse.ArgumentParser(description="miniwhale 多模态训练")
    ap.add_argument("--data_path", default="dataset/sft_i2t_mini.parquet")
    ap.add_argument("--ckpt", default=None, help="已训练的主干权重（可选）")
    ap.add_argument("--vit_ckpt", default=None,
                    help="第一阶段（SigLIP 对比）产出的视觉编码器权重（可选；按参考实现两阶段）")
    ap.add_argument("--epochs", type=int, default=0, help="训练轮数（默认 0=不跑）")
    ap.add_argument("--steps", type=int, default=0,
                    help="按步数训练（默认 0）；与 --epochs 同时指定时 epoch 优先")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5, help="余弦衰减的最低 lr")
    ap.add_argument("--warmup_steps", type=int, default=100, help="线性预热步数")
    ap.add_argument("--total_steps", type=int, default=0,
                    help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--router_bias_rate", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--log_interval", type=int, default=200)
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

    # 参数保持 **fp32 主权重**（按技术报告 §3.4）；计算走 bf16 autocast。
    # 见 train_pretrain 同名注释：bf16 参数会让 RMSNorm 权重的更新被舍入抹掉。
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)
    # 第二阶段从第一阶段（SigLIP 对比）的视觉编码器起（按参考实现：先对比预训练再自回归微调）
    if args.vit_ckpt:
        vit_state = torch.load(args.vit_ckpt, map_location=args.device)
        model.vision.load_state_dict(vit_state)
        print(f"已载入第一阶段视觉编码器 {args.vit_ckpt}", flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)
    assert model.vision is not None, "模型未启用视觉塔"

    ds = MultimodalDataset(args.data_path, tokenizer, max_length=args.seq_len,
                           image_token_id=cfg.text.image_token_id, vision_cfg=cfg.vision)
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

    # 主干 + ViT/Aligner 一起训（参考实现：视觉塔与 LLM 联合优化）
    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(epochs_to_run):
        ds.set_epoch(epoch)  # 流式数据集：每轮重排 shard 顺序 + shuffle buffer
        loss = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args,
                           max_steps=max_steps)
        print(f"epoch {epoch} done, avg loss {loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.save_dir, "vlm.pth"))
    # ViT 自回归微调阶段（报告 §4.2.2 阶段二）：只保留优化后的视觉编码器，丢弃 LLM
    torch.save(model.vision.state_dict(), os.path.join(args.save_dir, "vit_ar.pth"))
    print("多模态训练完成，已保存 vlm.pth + vit_ar.pth（仅视觉编码器）")


if __name__ == "__main__":
    main()
