"""预训练入口。

照 minimind 风格：argparse + train_epoch + __main__。
用 miniwhale 模型 + Muon/AdamW + 局部 fp32 的 bf16 混合精度。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# **必须在 import torch 之前**设：`torch._inductor.config` 首次 import 时用
# `os.getenv(..., 默认)` 读取，之后设就无效。compile_threads=1：Windows 上多线程
# 编译会走 SubprocPool 的 pass_fds → CPython 断言崩溃。不调 autotune warmup/rep
# （调小等于拿 kernel 质量换编译时间）。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train.data import PretrainDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.optim import make_optimizers, set_base_lr
from model.tokenizer import load_tokenizer


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """余弦学习率调度（带线性 warmup），照 minimind 惯例。

    输入：step 当前步；total_steps 总步数；warmup_steps 预热步数；base_lr 峰值 lr；min_lr 最低 lr。
    输出：本步的 lr。
    预期行为：前 warmup_steps 步从 base_lr/10 线性升到 base_lr；
              之后按余弦从 base_lr 衰减到 min_lr。
    """
    if step < warmup_steps:
        return base_lr * (0.1 + 0.9 * step / max(warmup_steps, 1))
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cos = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + (base_lr - min_lr) * cos


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0):
    """跑一个 epoch。

    输入：epoch；loader；model；三个优化器（Muon/AdamW/Sinkhorn）；args；start_step（续训）。
    输出：本 epoch 的平均 loss。
    预期行为：bf16 autocast 前向；交叉熵只算非 -100 位置；梯度累积 + 裁剪；
              每步按 warmup+cosine 更新三个优化器的 lr。
    """
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=start_step):
        t_step0 = time.time()
        # 学习率调度：warmup + cosine（只更新基准 lr，各组按 lr_mult 缩放——
        # 这样 Engram 的 ×5 学习率不会被抹平，参考实现 §4.2.2）
        lr = get_lr(step, args.total_steps, args.warmup_steps,
                    args.learning_rate, args.min_lr)
        set_base_lr((muon, adamw, sinkhorn), lr)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = loss / args.accumulation_steps
        # indexer loss 的梯度按主 loss scale 缩放（参考实现 DSA 做法）
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        loss.backward()
        # 首步含 torch.compile 冷编译 + autotune（前向+反向），单独计时打印，避免黑盒等待
        if step == start_step:
            torch.cuda.synchronize()
            print(f"[首次前向+反向 含编译] {(time.time() - t_step0):.1f}s", flush=True)
        if (step + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
            # noaux_tc 纠正 bias 按负载均衡规则更新（非梯度）
            model.update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if step % args.log_interval == 0:
            # 打印**瞬时**每步耗时（不是累计均值），才能看清稳态速度
            now = time.time()
            print(f"epoch {epoch} step {step} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）
        if args.save_interval and (step + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"pretrain_e{epoch}_s{step+1}.pth")
            torch.save(model.state_dict(), path)
            # 只保留最近 N 个（按 mtime），避免塞满磁盘
            if args.keep_last_n:
                cks = sorted(
                    (os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir)
                     if f.startswith("pretrain_") and f.endswith(".pth")),
                    key=os.path.getmtime)
                for old in cks[:-args.keep_last_n]:
                    os.remove(old)
            print(f"checkpoint saved at step {step+1}", flush=True)
        if args.steps and step + 1 >= args.steps:
            break
    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser(description="miniwhale 预训练")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0, help="最多训练步数（0=跑满 epoch）")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5, help="余弦衰减的最低 lr")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="线性预热步数")
    parser.add_argument("--total_steps", type=int, default=0,
                        help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--router_bias_rate", type=float, default=1e-3,
                        help="noaux_tc 纠正 bias 的更新步长（非梯度）")
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="每多少步存一次 checkpoint（0=不存）")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="只保留最近 N 个 checkpoint（0=全留）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 续训（只恢复模型权重）")
    parser.add_argument("--compile", action="store_true",
                        help="用 torch.compile 加速前向/反向（训练用；推理对拍走 eager 保逐位一致）")
    parser.add_argument("--compile_mode", type=str, default="default",
                        help='torch.compile 模式（固定 "default"）。')
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)

    cfg = ModelConfig.from_json()
    tokenizer = load_tokenizer()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    model = maybe_compile(model, args.compile, args.compile_mode)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 余弦调度总步数：未显式指定则按数据集行数 // batch × epochs
    if not args.total_steps:
        args.total_steps = max(ds.num_rows // args.batch_size, 1) * args.epochs
    print(f"数据集 {ds.num_rows} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    # 数据集是 IterableDataset（流式），自带缓冲打乱，DataLoader 不能再传 shuffle=True
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        loss = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args)
        print(f"epoch {epoch} done, avg loss {loss:.4f}")
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"pretrain_e{epoch}.pth"))


if __name__ == "__main__":
    main()
