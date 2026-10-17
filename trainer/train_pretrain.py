"""预训练入口。

照 minimind 风格：argparse + train_epoch + __main__。
用 miniwhale 模型 + Muon/AdamW + 局部 fp32 的 bf16 混合精度。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# **必须在 import torch 之前**设：`torch._inductor.config` 首次 import 时用
# `os.getenv(..., 默认)` 读取，之后设就无效。compile_threads=1：**仅 Windows**——
# 多线程编译会走 SubprocPool 的 pass_fds → CPython 断言崩溃。Linux（含 Kaggle）放开
# 多线程，编译更快。不调 autotune warmup/rep（调小等于拿 kernel 质量换编译时间）。
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train import ddp
from model.train.data import PretrainDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.checkpoint import load_resume_ckpt, save_checkpoint
from model.train.logfile import setup_log_file
from model.train.scheduler import get_lr
from model.train.optim import make_optimizers, set_base_lr
from model.tokenizer import load_tokenizer


def train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args, start_step=0,
                max_steps=None):
    """跑一个 epoch（或按步数模式跑到 max_steps）。

    输入：epoch；loader；model；三个优化器（Muon/AdamW/Sinkhorn）；args；start_step（续训）；
          max_steps（按步数模式的上限；None=跑满整个 loader）。
    输出：本次的平均 loss。
    预期行为：bf16 autocast 前向；交叉熵只算非 -100 位置；梯度累积 + 裁剪；
              每步按 warmup+cosine 更新三个优化器的 lr。max_steps=0 时一步都不跑。
    """
    model.train()
    total_loss, steps = 0.0, 0
    t0 = time.time()
    # DDP 防死锁：各 rank 数据量可能不等，先耗尽者若退出循环会让其余 rank 卡在 all_reduce。
    # 数据耗尽时重启迭代器继续，由 max_steps 统一步数（非分布式行为不变）。
    it = iter(loader)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            if max_steps is None or steps >= max_steps:
                break
            it = iter(loader)
            batch = next(it)
        # 按步数模式：到量即停（max_steps=0 → 0 步，字面语义，不是哨兵）
        if max_steps is not None and steps >= max_steps:
            break
        step = steps
        gstep = start_step + step          # **全局**步数：lr 调度与 checkpoint 都按它算
        t_step0 = time.time()
        # 学习率调度：warmup + cosine（只更新基准 lr，各组按 lr_mult 缩放——
        # 这样 Engram 的 ×5 学习率不会被抹平，参考实现 §4.2.2）
        lr = get_lr(gstep, args.total_steps, args.warmup_steps,
                    args.learning_rate, args.min_lr,
                    decay_start=0.62, decay_end=0.89)
        set_base_lr((muon, adamw, sinkhorn), lr)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids)
            # **全局 token 归一化**（DDP 逐位等价的关键）：单卡时 = 批内 mean（数值不变）；
            # 多卡时各卡算 sum、按全局 valid token 数归一 → DDP 平均梯度后等于单卡全局 mean。
            tgt = labels[:, 1:].reshape(-1)
            ce_sum = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tgt,
                ignore_index=-100, reduction="sum")
            n_valid = (tgt != -100).sum()
            loss = ddp.scale_loss_for_ddp(ce_sum, n_valid) / args.accumulation_steps
        # indexer loss 的梯度按主 loss scale 缩放（参考实现 DSA 做法）
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        loss.backward()
        # 首步含 torch.compile 冷编译 + autotune（前向+反向），单独计时打印，避免黑盒等待
        if step == 0 and ddp.is_main():
            torch.cuda.synchronize()
            print(f"[首次前向+反向 含编译] {(time.time() - t_step0):.1f}s", flush=True)
        if (gstep + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            muon.step(); adamw.step(); sinkhorn.step()
            muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
            # noaux_tc 纠正 bias 按负载均衡规则更新（非梯度）
            ddp.unwrap(model).update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if gstep % args.log_interval == 0 and ddp.is_main():
            # 打印**瞬时**每步耗时（不是累计均值），才能看清稳态速度。仅 rank0 打印。
            now = time.time()
            print(f"epoch {epoch} step {gstep} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）。**仅 rank0 存**，unwarp 去前缀。
        if args.save_interval and (gstep + 1) % args.save_interval == 0:
            save_checkpoint(args.save_dir, "pretrain", epoch, gstep + 1,
                           model, muon, adamw, sinkhorn, args.keep_last_n)
            if ddp.is_main():
                print(f"checkpoint saved at step {gstep+1}", flush=True)
    return total_loss / max(steps, 1), steps


def main():
    parser = argparse.ArgumentParser(description="miniwhale 预训练")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--save_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=0, help="训练轮数（默认 0=不跑）")
    parser.add_argument("--steps", type=int, default=0,
                        help="按步数训练（默认 0=不跑）；与 --epochs 同时指定时 epoch 优先")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2.6e-4,
                        help="峰值 lr（按参考实现报告 §4.2.2：2.6e-4）")
    parser.add_argument("--min_lr", type=float, default=2.6e-5, help="余弦衰减的最低 lr")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="线性预热步数")
    parser.add_argument("--total_steps", type=int, default=0,
                        help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--router_bias_rate", type=float, default=1e-3,
                        help="noaux_tc 纠正 bias 的更新步长（非梯度）")
    parser.add_argument("--max_seq_len", type=int, default=512)
    # 单 jsonl = 1 个 shard；`datasets` 的分片规则是 worker 数 ≤ shard 数，
    # 故 >1 会被库自动压回 1（并打 warning）。默认 1 = 实际行为，不误导。
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=1000,
                        help="每多少步存一次 checkpoint（0=不存）")
    parser.add_argument("--keep_last_n", type=int, default=3,
                        help="只保留最近 N 个 checkpoint（0=全留）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 checkpoint 续训（恢复模型 + 三个优化器动量 + 步数）")
    parser.add_argument("--compile", action="store_true",
                        help="用 torch.compile 加速前向/反向（训练用；推理对拍走 eager 保逐位一致）")
    parser.add_argument("--compile_mode", type=str, default="default",
                        help='torch.compile 模式（固定 "default"）。')
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None,
                        help="训练日志落盘路径（Python 内直写，不经 shell 管道；None=只打印控制台）")
    args = parser.parse_args()
    # DDP 初始化（非分布式时为 no-op，返回 --device）。**必须在建模型之前**。
    args.device = ddp.setup(args.device)
    if ddp.is_main():
        setup_log_file(args.log_file, append=bool(args.resume))

    torch.manual_seed(args.seed)
    # 参数保持 **fp32 主权重**（按技术报告 §3.4："the FP32 master weights maintained
    # by the optimizer"）。计算用 bf16 autocast。若把默认 dtype 设成 bf16，参数直接是 bf16，
    # RMSNorm 权重（初值 1.0，bf16 ULP≈4e-3）的 AdamW 更新（≈lr≈2.6e-4）会被舍入抹掉 →
    # norm 权重不更新 → 收敛受阻。

    cfg = ModelConfig.from_json()
    tokenizer = load_tokenizer()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    model = maybe_compile(model, args.compile, args.compile_mode)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 训练模式：steps 与 epochs 默认都为 0。两者互斥——同时指定时 epoch 优先（打印提醒）。
    # 默认全 0 且不传 = 直接报错不跑（防止误触发整轮训练）。
    if not args.steps and not args.epochs:
        parser.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    if args.steps and args.epochs:
        print(f"提醒：--steps={args.steps} 与 --epochs={args.epochs} 同时指定，"
              f"按 epoch 优先（跑 {args.epochs} 轮，忽略 --steps）", flush=True)
        epochs_to_run, max_steps = args.epochs, None
    elif args.steps:
        epochs_to_run, max_steps = 1, args.steps   # 按步数：跑 args.steps 步
    else:
        epochs_to_run, max_steps = args.epochs, None  # 按轮次
    # 余弦调度总步数：按步数模式取 max_steps；按轮次模式按数据集行数 // batch × 轮数
    if not args.total_steps:
        # 与 ep_steps 同分母（都除 world_size）；否则 DDP 下 lr 调度步数翻倍、只衰减一半。
        args.total_steps = (max_steps if max_steps
                            else max(ds.num_rows // args.batch_size // ddp.world_size(), 1)
                            * epochs_to_run)
    print(f"数据集 {ds.num_rows} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    # 数据集是 IterableDataset（流式），自带缓冲打乱，DataLoader 不能再传 shuffle=True
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    start_step = 0
    if args.resume:
        _st = load_resume_ckpt(args.resume, model, args.device)
        start_step = _st["step"]
        resume_ck = _st
        print(f"从 {args.resume} 续训，起始 step {start_step}", flush=True)

