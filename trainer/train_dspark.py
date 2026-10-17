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
import time
from pathlib import Path

# **必须在 import torch 之前**设（torch._inductor.config 首次 import 时读 env）。
# compile_threads=1：**仅 Windows**——多线程编译会走 SubprocPool 的 pass_fds → CPython
# 断言崩溃。Linux（含 Kaggle）放开多线程，编译更快。
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train import ddp
from model.train.data import PretrainDataset
from model.train.dspark_loss import compute_dspark_loss
from model.model import Model
from model.train.compile import maybe_compile
from model.train.checkpoint import load_resume_ckpt, save_checkpoint
from model.train.logfile import setup_log_file
from model.train.scheduler import get_lr
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
    ap.add_argument("--resume", default=None,
                    help="从本阶段中间档续训（dspark_s{STEP}.pth；恢复完整训练状态）")
    ap.add_argument("--epochs", type=int, default=0, help="训练轮数（默认 0=不跑）")
    ap.add_argument("--steps", type=int, default=0,
                    help="按步数训练（默认 0=不跑）；与 --epochs 同时指定时 epoch 优先")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4, help="参考实现 lr=6e-4")
    ap.add_argument("--min_lr", type=float, default=6e-5, help="余弦衰减的最低 lr")
    ap.add_argument("--warmup_steps", type=int, default=100, help="线性预热步数")
    ap.add_argument("--router_bias_rate", type=float, default=1e-3,
                    help="noaux_tc 纠正 bias 的更新步长（非梯度）")
    ap.add_argument("--log_interval", type=int, default=200)
    ap.add_argument("--save_interval", type=int, default=1000,
                    help="每多少步存一次 checkpoint（0=不存）")
    ap.add_argument("--keep_last_n", type=int, default=3,
                    help="只保留最近 N 个 checkpoint（0=全留）")
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--log_file", default=None,
                    help="训练日志落盘路径（Python 内直写，不经 shell 管道；None=只打印控制台）")
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    ap.add_argument("--compile_mode", type=str, default="default",
                    help='torch.compile 模式（固定 "default"）')
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    # DDP 初始化（非分布式时为 no-op，返回 --device）。**必须在建模型之前**。
    args.device = ddp.setup(args.device)
    if ddp.is_main():
        setup_log_file(args.log_file, append=bool(args.resume))

    # 参数保持 **fp32 主权重**（按技术报告 §3.4）；计算走 bf16 autocast。
    # 见 train_pretrain 同名注释：bf16 参数会让 RMSNorm 权重的更新被舍入抹掉。
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        # strict=False：允许起点权重与当前模型有键差（主干 vs draft 头）。
        # 但**必须打印缺失/多余的键**——否则键名对不上时静默拿随机初始化训。
        _st = torch.load(args.ckpt, map_location=args.device)
        _miss, _unexp = model.load_state_dict(_st, strict=False)
        print(f"载入 {args.ckpt}：缺失 {len(_miss)} 键 / 多余 {len(_unexp)} 键", flush=True)
        if _miss:
            print(f"  缺失（用随机初始化）: {_miss[:8]}{' …' if len(_miss) > 8 else ''}",
                  flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)
    trainable = freeze_backbone(model)
    print(f"DSpark 可训练参数 {sum(p.numel() for p in trainable)/1e6:.2f}M")

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    # 数据集是 IterableDataset（流式），自带缓冲打乱，DataLoader 不能再传 shuffle=True
    loader = DataLoader(ds, batch_size=args.batch, num_workers=0)
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95))
    # 续训：恢复完整训练状态（模型 + 优化器动量 + 步数）。只存权重会让 AdamW 一/二阶矩归零。
    start_step, resume_ck = 0, None
    if args.resume:
        _st = load_resume_ckpt(args.resume, model, args.device)
        start_step = _st["step"]
        if _st["muon"] is not None:
            opt.load_state_dict(_st["muon"])
        print(f"从 {args.resume} 续训，起始 step {start_step}", flush=True)
    # DDP 包装：**必须在 freeze_backbone + opt 建立之后**（否则 trainable 列表与按名路由
    # 会带 `module.` 前缀而失效）。DSpark 冻结主干只训 draft 头，梯度同步只覆盖 draft 参数。
    model = ddp.wrap(model)
    model.train()

    # 训练模式：steps 与 epochs 默认都为 0。两者互斥——同时指定时 epoch 优先（打印提醒）。
    # 默认全 0 且不传 = 直接报错不跑（防止误触发整轮训练）。
    if not args.steps and not args.epochs:
        ap.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    if args.steps and args.epochs:
        print(f"提醒：--steps={args.steps} 与 --epochs={args.epochs} 同时指定，"
              f"按 epoch 优先（跑 {args.epochs} 轮，忽略 --steps）", flush=True)
        total_steps = max(len(ds) // args.batch // ddp.world_size(), 1) * args.epochs
    elif args.steps:
        total_steps = args.steps
    else:
        total_steps = max(len(ds) // args.batch, 1) * args.epochs

    tc = cfg.text
    epoch = 0
    ds.set_epoch(epoch)  # 流式数据集：每轮换 seed 重排数据顺序
    it = iter(loader)
    for step in range(start_step, total_steps):
        t_step0 = time.time()   # 每步起点：打印**瞬时**耗时（非累计均值），看清稳态速度
        lr = get_lr(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            epoch += 1
            ds.set_epoch(epoch)  # 新一轮：换 seed 重排，避免每轮顺序一样
            it = iter(loader); batch = next(it)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        loss_mask = (labels != -100).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 主干 + DSpark **一次 forward**：DSpark 前向必须在 model.forward 内，
            # 否则 DDP 遍历不到 dspark 参数（前向在 forward 之外）→ 判其 unused →
            # backward 收到梯度时报「Expected to mark a variable ready only once」。
            # 详见 model.forward 的 dspark_train 说明。
            logits, _, out = model(ids, dspark_train={"loss_mask": loss_mask})
            loss, loss_parts = compute_dspark_loss(
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
        # noaux_tc 纠正 bias 按负载均衡规则更新（非梯度；主干 MoE 冻结但仍需均衡）
        ddp.unwrap(model).update_router_bias(args.router_bias_rate)
        if step % args.log_interval == 0 and ddp.is_main():
            # 分项（未加权）便于看清哪一项在动/卡住：CE 学预测、L1 贴主干分布、conf 校准接受率。
            print(f"dspark step {step}/{total_steps} loss {loss.item():.4f} "
                  f"ce {loss_parts['ce'].item():.4f} l1 {loss_parts['l1'].item():.4f} "
                  f"conf {loss_parts['conf'].item():.4f} "
                  f"lr {opt.param_groups[0]['lr']:.2e} "
                  f"{(time.time() - t_step0) * 1000:.0f}ms/step", flush=True)
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）。**仅 rank0 存**，unwarp 去前缀。
        if args.save_interval and (step + 1) % args.save_interval == 0:
            save_checkpoint(args.save_dir, "dspark", epoch, step + 1,
                           model, muon, adamw, sinkhorn, args.keep_last_n)
            if ddp.is_main():
                print(f"checkpoint saved at step {step+1}", flush=True)
    if ddp.is_main():
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(ddp.unwrap(model).state_dict(), os.path.join(args.save_dir, "dspark.pth"))
        print("DSpark 训练完成")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
