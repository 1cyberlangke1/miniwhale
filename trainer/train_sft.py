"""全参数 SFT 入口（对话微调，从预训练权重起）。

照 minimind `train_full_sft.py` 风格：argparse + train_epoch + __main__。
与预训练的区别：数据是多轮对话（`SFTDataset`，只对 assistant 段算 loss）、学习率更小。
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
from model.train.data import SFTDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.logfile import setup_log_file
from model.train.dspark_loss import compute_dspark_loss
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


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0,
                max_steps=None):
    """跑一个 epoch（或按步数模式跑到 max_steps）。

    输入：epoch；loader；model；三个优化器；args；start_step（续训）；max_steps（步数上限，None=跑满）。
    输出：本 epoch 的平均 loss。
    预期行为：bf16 autocast 前向；交叉熵只算 assistant 段（非 -100）；梯度累积 + 裁剪；
              每步按 warmup+cosine 更新三个优化器的 lr。max_steps=0 时一步都不跑。
    """
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        gstep = start_step + step          # **全局**步数：lr 调度与 checkpoint 都按它算
        t_step0 = time.time()
        # 只更新基准 lr，各组按 lr_mult 缩放（保留 Engram ×5，参考实现 §4.2.2）
        lr = get_lr(gstep, args.total_steps, args.warmup_steps, args.learning_rate, args.min_lr)
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
            # post-training 阶段同训 DSpark（照报告 §2.4.3）：主干由上面的 CE 更新；
            # DSpark 由 DSpark loss 更新。DSpark loss 的 target（主干 hidden/logits）
            # 天然不带梯度（主干前向在 no_grad 外但 target 走 detach），故其梯度
            # 不回传主干——正是参考实现「同训但梯度不回传 backbone」的语义。
            if args.dspark and model.dspark is not None:
                tc = model.cfg.text
                loss_mask = (labels != -100).float()
                out = model.dspark.forward_train(ids, model._last_main_hidden.detach(),
                                                 loss_mask, aligned_target_logits=logits.detach())
                dspark_loss = compute_dspark_loss(
                    out,
                    loss_decay_gamma=tc.dspark_loss_decay_gamma,
                    ce_loss_alpha=tc.dspark_ce_loss_alpha,
                    l1_loss_alpha=tc.dspark_l1_loss_alpha,
                    confidence_head_alpha=tc.dspark_confidence_head_alpha,
                )
                loss = loss + dspark_loss
            loss = loss / args.accumulation_steps
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        loss.backward()
        if (gstep + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
            model.update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if gstep % args.log_interval == 0:
            # 打印**瞬时**每步耗时（非累计均值），才能看清稳态速度
            now = time.time()
            print(f"epoch {epoch} step {gstep} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）
        if args.save_interval and (gstep + 1) % args.save_interval == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"sft_e{epoch}_s{gstep+1}.pth")
            torch.save(model.state_dict(), path)
            if args.keep_last_n:
                cks = sorted(
                    (os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir)
                     if f.startswith("sft_") and f.endswith(".pth")),
                    key=os.path.getmtime)
                for old in cks[:-args.keep_last_n]:
                    os.remove(old)
            print(f"checkpoint saved at step {gstep+1}", flush=True)
    return total_loss / max(steps, 1), steps


def main():
    parser = argparse.ArgumentParser(description="miniwhale 全参数 SFT")
    parser.add_argument("--data_path", type=str, default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--ckpt", type=str, default=None, help="预训练权重（可选）")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数（默认 0=不跑）")
    parser.add_argument("--steps", type=int, default=0,
                        help="按步数训练（默认 0）；与 --epochs 同时指定时 epoch 优先")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6, help="余弦衰减的最低 lr")
    parser.add_argument("--warmup_steps", type=int, default=200, help="线性预热步数")
    parser.add_argument("--total_steps", type=int, default=0,
                        help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--router_bias_rate", type=float, default=1e-3)
    parser.add_argument("--max_seq_len", type=int, default=768)
    # 单 jsonl = 1 个 shard；`datasets` 的分片规则是 worker 数 ≤ shard 数，
    # 故 >1 会被库自动压回 1（并打 warning）。默认 1 = 实际行为，不误导。
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="每多少步存一次 checkpoint（0=不存）")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="只保留最近 N 个 checkpoint（0=全留）")
    parser.add_argument("--dspark", action="store_true",
                        help="post-training 阶段同训 DSpark（照报告 §2.4.3；主干由 CE 训，"
                             "DSpark 由 DSpark loss 训，梯度不回传主干）")
    parser.add_argument("--compile", action="store_true",
                        help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    parser.add_argument("--compile_mode", type=str, default="default",
                        help='torch.compile 模式（固定 "default"）')
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None,
                        help="训练日志落盘路径（Python 内直写，不经 shell 管道；None=只打印控制台）")
    args = parser.parse_args()
    setup_log_file(args.log_file)

    torch.manual_seed(args.seed)
    # 参数保持 **fp32 主权重**（按技术报告 §3.4）；计算走 bf16 autocast。
    # 见 train_pretrain 同名注释：bf16 参数会让 RMSNorm 权重的更新被舍入抹掉。

    cfg = ModelConfig.from_json()
    tokenizer = load_tokenizer()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)
    model = maybe_compile(model, args.compile, args.compile_mode)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 训练模式：steps 与 epochs 默认都为 0。两者互斥——同时指定时 epoch 优先（打印提醒）。
    # 默认全 0 且不传 = 直接报错不跑（防止误触发整轮训练）。
    if not args.steps and not args.epochs:
        parser.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    if args.steps and args.epochs:
        print(f"提醒：--steps={args.steps} 与 --epochs={args.epochs} 同时指定，"
              f"按 epoch 优先（跑 {args.epochs} 轮，忽略 --steps）", flush=True)
        epochs_to_run, max_steps = args.epochs, None
    elif args.steps:
        epochs_to_run, max_steps = 1, args.steps
    else:
        epochs_to_run, max_steps = args.epochs, None
    # 数据集是 IterableDataset（流式），自带缓冲打乱，DataLoader 不能再传 shuffle=True
    if not args.total_steps:
        args.total_steps = (max_steps if max_steps
                            else max(ds.num_rows // args.batch_size, 1) * epochs_to_run)
    print(f"数据集 {ds.num_rows} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    for epoch in range(epochs_to_run):
        ds.set_epoch(epoch)  # 流式数据集：每轮换 seed 重排数据顺序
        loss, n = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args,
                              start_step=global_step, max_steps=max_steps)
        global_step += n   # 跨 epoch 累加：lr 余弦调度与 checkpoint 间隔都按全局步数
        print(f"epoch {epoch} done, avg loss {loss:.4f}（累计 {global_step} 步）")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "sft.pth"))
    print("SFT 完成，已保存 sft.pth")


if __name__ == "__main__":
    main()
