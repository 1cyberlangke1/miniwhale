"""训练主循环（五个 trainer 共用的骨架）。

各 trainer 的差异只有三处，全部用回调注入：
  - **前向 + loss**：`forward_loss(batch)` 返回 `(loss, extra)`；`loss` 已按
    `accumulation_steps` 缩放，`extra` 是可选的分项 dict（如 DSpark 的 ce/l1/conf）。
  - **更新**：`step_fn()`（优化器 step + 清梯度 + 路由 bias 更新）。
  - **存盘前缀**：`ckpt_prefix`（"sft"/"pretrain"/"vlm"/"siglip"）。

其余（lr 调度、梯度累积、日志、checkpoint、DDP 防死锁）完全一致，故集中在此。
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from model.train import ddp
from model.train.checkpoint import save_checkpoint
from model.train.scheduler import get_lr
from model.train.optim import set_base_lr


def train_epoch(epoch, loader, model, optimizers, args, forward_loss: Callable,
                step_fn: Callable, ckpt_prefix: str, start_step: int = 0,
                max_steps: int | None = None,
                lr_decay: tuple[float | None, float | None] = (None, None),
                skip_bad_batch: Callable | None = None):
    """跑一个 epoch（或按步数模式跑到 max_steps）。

    输入：epoch；loader；model（已 DDP 包装）；optimizers（三优化器元组，用于 lr 调度）；
          args（需 learning_rate/min_lr/warmup_steps/total_steps/accumulation_steps/
          grad_clip/log_interval/save_interval/save_dir/keep_last_n）；
          forward_loss(batch) → (loss, extra_dict|None)；step_fn() 执行一次参数更新；
          ckpt_prefix；start_step（本 epoch 起始**全局**步数）；max_steps（None=跑满）；
          lr_decay=(decay_start, decay_end)（都 None=两段式 warmup+cosine）；
          skip_bad_batch(batch) → True 时跳过该批（如对比损失需要 ≥2 样本）。
    输出：(本 epoch 平均 loss, 实际步数)。
    预期行为：每步按 warmup(+恒定段)+cosine 更新 lr；梯度累积；每步打印**瞬时**耗时；
              每 save_interval 步存完整训练状态（仅 rank0）。max_steps=0 时一步都不跑。
    """
    model.train()
    total_loss, steps = 0.0, 0
    epoch_t0 = time.time()          # 本 epoch 起点（ETA 用）
    start_step_of_epoch = start_step
    # 本 epoch 的结束步：max_steps 给定则用它，否则按 loader 长度估（仅影响 ETA 显示）
    end_step = (start_step + max_steps if max_steps is not None
                else start_step + max(len(loader), 1))
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
        if max_steps is not None and steps >= max_steps:
            break
        if skip_bad_batch is not None and skip_bad_batch(batch):
            continue
        gstep = start_step + steps      # **全局**步数：lr 调度与 checkpoint 都按它算
        t_step0 = time.time()
        # 只更新基准 lr，各组按 lr_mult 缩放（保留 Engram ×5，参考实现 §4.2.2）
        lr = get_lr(gstep, args.total_steps, args.warmup_steps, args.learning_rate,
                    args.min_lr, decay_start=lr_decay[0], decay_end=lr_decay[1])
        set_base_lr(optimizers, lr)
        loss, extra = forward_loss(batch)
        loss.backward()
        if (gstep + 1) % args.accumulation_steps == 0:
            step_fn()
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if gstep % args.log_interval == 0 and ddp.is_main():
            now = time.time()
            # ETA：按本 epoch 已用时间线性外推（仅打印，不参与调度）
            spent = now - epoch_t0
            done = steps - start_step_of_epoch
            eta_min = (spent / max(done, 1) * (end_step - steps) / 60
                       if end_step > steps else 0.0)
            print(f"epoch {epoch} step {gstep} loss {total_loss / steps:.4f} "
                  f"lr {optimizers[1].param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step{_fmt_extra(extra)} "
                  f"eta {eta_min:.0f}min", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）。**仅 rank0 存**。
        if args.save_interval and (gstep + 1) % args.save_interval == 0:
            save_checkpoint(args.save_dir, ckpt_prefix, epoch, gstep + 1,
                            model, *optimizers, args.keep_last_n)
            if ddp.is_main():
                print(f"checkpoint saved at step {gstep + 1}", flush=True)
    return total_loss / max(steps, 1), steps


def _fmt_extra(extra: dict | None) -> str:
    """把 loss 分项格式化成日志后缀。

    输入：{"ce":..., "l1":..., "conf":...} 或 None。
    输出："" 或 " ce 1.2345 l1 0.9876 conf 0.5432"。
    """
    if not extra:
        return ""
    return "".join(f" {k} {v:.4f}" for k, v in extra.items())


def update_weights(model, muon, adamw, sinkhorn, args):
    """标准参数更新（梯度裁剪 → 三优化器 step → 清梯度 → 路由 bias）。

    输入：model；三个优化器；args（需 grad_clip / router_bias_rate）。
    输出：无。
    预期行为：DDP 下 clip_grad_norm_ 作用于全部参数；noaux_tc 的 bias 用非梯度规则更新。
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    muon.step(); adamw.step(); sinkhorn.step()
    muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
    ddp.unwrap(model).update_router_bias(args.router_bias_rate)


def build_loaders_and_steps(ds, args, collate_fn=None):
    """建 DataLoader，并解析训练模式（epochs/steps）与总步数。

    输入：ds（IterableDataset）；args（含 steps/epochs/batch_size/total_steps 等）；collate_fn。
    输出：(loader, epochs_to_run, max_steps)。
    预期行为：steps 与 epochs 都未给时由调用方先 ap.error；两者同给时 epoch 优先并打印提醒；
              total_steps 未给时按「数据集行数 // batch_size // world_size × 轮数」推算
              （与下方 ep_steps 同分母，否则 DDP 下 lr 只衰减一半）。
    """
    from torch.utils.data import DataLoader

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
                            else max(len(ds) // args.batch_size // ddp.world_size(), 1)
                            * epochs_to_run)
    print(f"数据集 {len(ds)} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    kw = {"collate_fn": collate_fn} if collate_fn is not None else {}
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True, **kw)
    return loader, epochs_to_run, max_steps


def run_epochs(ds, loader, model, optimizers, args, forward_loss, step_fn, ckpt_prefix,
               start_step: int, epochs_to_run: int, max_steps: int | None,
               lr_decay=(None, None), skip_bad_batch=None):
    """按轮次跑训练（各 trainer 的 main 尾部完全一致的部分）。

    输入：ds（需有 set_epoch/__len__）；其余同 train_epoch；start_step 为续训起始全局步数。
    输出：训练结束后的全局步数。
    预期行为：每轮前调 ds.set_epoch 重排；DDP 下按固定步数跑（防 all_reduce 死锁）；
              跨 epoch 累加全局步数（lr 余弦调度与 checkpoint 间隔都按它算）。
    """
    global_step = start_step
    for epoch in range(epochs_to_run):
        ds.set_epoch(epoch)  # 流式数据集：每轮换 seed 重排数据顺序
        # DDP 下各 rank 数据量可能不等 → 固定步数（否则先耗尽者退出循环致 all_reduce 死锁）。
        ep_steps = max_steps
        if ddp.is_distributed() and ep_steps is None:
            ep_steps = max(len(ds) // args.batch_size // ddp.world_size(), 1)
        loss, n = train_epoch(epoch, loader, model, optimizers, args, forward_loss,
                              step_fn, ckpt_prefix, start_step=global_step,
                              max_steps=ep_steps, lr_decay=lr_decay,
                              skip_bad_batch=skip_bad_batch)
        global_step += n   # 跨 epoch 累加：lr 余弦调度与 checkpoint 间隔都按全局步数
        if ddp.is_main():
            print(f"epoch {epoch} done, avg loss {loss:.4f}（累计 {global_step} 步）")
    return global_step
