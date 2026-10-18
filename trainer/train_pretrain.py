"""预训练入口。

照 minimind 风格：argparse + train_epoch + __main__。
用 miniwhale 模型 + Muon/AdamW + 局部 fp32 的 bf16 混合精度。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# **必须在 import torch 之前**设：`torch._inductor.config` 首次 import 时用
# `os.getenv(..., 默认)` 读取，之后设就无效。compile_threads=1：**仅 Windows**——
# 多线程编译会走 SubprocPool 的 pass_fds → CPython 断言崩溃。Linux（含 Kaggle）放开
# 多线程，编译更快。不调 autotune warmup/rep（调小等于拿 kernel 质量换编译时间）。
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train import ddp
from model.train.cli import add_common_args, apply_defaults
from model.train.data import PretrainDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.checkpoint import load_resume_ckpt
from model.train.logfile import setup_log_file
from model.train.loop import (build_loaders_and_steps, run_epochs, update_weights)
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="miniwhale 预训练")
    parser.add_argument("--data_path", type=str, default="dataset/pretrain_t2t_mini.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--router_bias_rate", type=float, default=1e-3,
                        help="noaux_tc 纠正 bias 的更新步长（非梯度）")
    add_common_args(parser)
    args = parser.parse_args()
    # 各阶段默认值（命令行显式给出的参数不受影响）
    apply_defaults(args, learning_rate=2.6e-4, min_lr=2.6e-5, warmup_steps=2000,
                   accumulation_steps=8, num_workers=1)
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
    # 训练模式：steps 与 epochs 默认都为 0，不传 = 直接报错不跑（防误触发整轮训练）。
    if not args.steps and not args.epochs:
        parser.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    loader, epochs_to_run, max_steps = build_loaders_and_steps(ds, args)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    os.makedirs(args.save_dir, exist_ok=True)
    start_step, resume_ck = 0, None
    if args.resume:
        _st = load_resume_ckpt(args.resume, model, args.device)
        start_step = _st["step"]
        resume_ck = _st
        print(f"从 {args.resume} 续训，起始 step {start_step}", flush=True)
    if resume_ck is not None:
        muon.load_state_dict(resume_ck["muon"])
        adamw.load_state_dict(resume_ck["adamw"])
        sinkhorn.load_state_dict(resume_ck["sinkhorn"])
        print("已恢复三优化器状态（Muon/AdamW/Sinkhorn 动量）", flush=True)
    # DDP 包装：**必须在 make_optimizers 之后**（否则 named_parameters 带 `module.` 前缀
    # → 按名路由的优化器分组失效）。
    model = ddp.wrap(model)

    optimizers = (muon, adamw, sinkhorn)

    def forward_loss(batch):
        """前向 + CE loss（全局 token 归一化，DDP 逐位等价）。

        输入：batch（input_ids/labels）。
        输出：(loss, None)。loss 已按 accumulation_steps 缩放。
        预期行为：bf16 autocast；只对非 -100 位置算 CE；indexer loss 的梯度按主 loss 缩放。
        """
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids)
            # **全局 token 归一化**：单卡时 = 批内 mean；多卡时各卡算 sum、按全局 valid
            # token 数归一 → DDP 平均梯度后等于单卡全局 mean。
            tgt = labels[:, 1:].reshape(-1)
            ce_sum = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tgt,
                ignore_index=-100, reduction="sum")
            n_valid = (tgt != -100).sum()
            loss = ddp.scale_loss_for_ddp(ce_sum, n_valid) / args.accumulation_steps
        # indexer loss 的梯度按主 loss scale 缩放（参考实现 DSA 做法）
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        return loss, None

    def step_fn():
        """标准更新（裁剪 → 三优化器 step → 清梯度 → 路由 bias）。"""
        update_weights(model, muon, adamw, sinkhorn, args)

    run_epochs(ds, loader, model, optimizers, args, forward_loss, step_fn, "pretrain",
               start_step, epochs_to_run, max_steps, lr_decay=(0.62, 0.89))

    if ddp.is_main():
        torch.save(ddp.unwrap(model).state_dict(), os.path.join(args.save_dir, "pretrain.pth"))
        print("预训练完成，已保存 pretrain.pth")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
