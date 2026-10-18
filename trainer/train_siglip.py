"""SigLIP 对比预训练入口（报告 §4.2.2 第一阶段）。

按参考实现两阶段流程：**先**用 SigLIP sigmoid 对比损失把 ViT 与文本对齐（本脚本），
**再**接进语言骨干做自回归微调（`train_vlm.py`）。本脚本产出 `out/vit_siglip.pth`
（只存视觉编码器——参考实现「丢弃 LLM、只留优化后的视觉编码器」）。

风格照 minimind：argparse + train_epoch + __main__。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# **必须在 import torch 之前**设（torch._inductor.config 首次 import 时读 env）。
# compile_threads=1：**仅 Windows**——多线程编译会走 SubprocPool 的 pass_fds → CPython
# 断言崩溃。Linux（含 Kaggle）放开多线程，编译更快。
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import ModelConfig
from model.train import ddp
from model.train.cli import add_common_args, apply_defaults, resolve_resume
from model.train.data import ImageTextPairDataset, PreprocessedPairDataset
from model.train.siglip import SigLIP
from model.train.compile import maybe_compile
from model.train.checkpoint import _atomic_save, load_resume_ckpt
from model.train.logfile import setup_log_file
from model.train.loop import (build_loaders_and_steps, run_epochs)
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer


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


def main():
    ap = argparse.ArgumentParser(description="miniwhale ViT SigLIP 对比预训练")
    ap.add_argument("--data_path", default="dataset/sft_i2t_mini.parquet")
    ap.add_argument("--source", default="conversations",
                    choices=["conversations", "flickr30k"],
                    help="数据格式：conversations（列 conversations+image_bytes）"
                         "或 flickr30k（列 image+caption 列表）")
    ap.add_argument("--image_size", type=int, default=224,
                    help="把图统一 resize 到 N×N（0=保持原尺寸）；参考实现 SigLIP 阶段为 224")
    ap.add_argument("--shuffle_buffer", type=int, default=0,
                    help="流式 shuffle 缓冲（0=顺序读）。图文对数据设 0：流式读 parquet 的 "
                         "Image 列时，shuffle 需先把 buffer_size 行**物化解码**进内存，"
                         "对 512 大 batch 是严重的 I/O/CPU 放大，收益却很小（每 epoch 已换图序+换 caption）。")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--caption_len", type=int, default=64, help="caption 最大 token 数")
    ap.add_argument("--proj_dim", type=int, default=256, help="对比嵌入维")
    ap.add_argument("--text_dim", type=int, default=256, help="文本塔隐藏维")
    ap.add_argument("--text_layers", type=int, default=4, help="文本塔层数")
    ap.add_argument("--text_heads", type=int, default=4, help="文本塔注意力头数")
    add_common_args(ap)
    args = ap.parse_args()
    # 各阶段默认值（命令行显式给出的参数不受影响）
    apply_defaults(args, learning_rate=1e-4, min_lr=1e-5, warmup_steps=100,
                   accumulation_steps=4, num_workers=4)
    # DDP 初始化（非分布式时为 no-op，返回 --device）。**必须在建模型之前**。
    args.device = ddp.setup(args.device)
    if ddp.is_main():
        setup_log_file(args.log_file, append=bool(args.resume or args.from_resume))

    torch.manual_seed(args.seed)
    # 参数保持 fp32 主权重；计算走 bf16 autocast（同其它训练入口）。
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = SigLIP(cfg.text.vocab_size, cfg.vision, proj_dim=args.proj_dim,
                   text_dim=args.text_dim, text_layers=args.text_layers,
                   text_heads=args.text_heads).to(args.device)
    # 优化器分组（Muon/AdamW/Sinkhorn）按名字路由，需读 model.cfg.text.num_attention_heads
    model.cfg = cfg
    # 续训：恢复完整训练状态（模型 + 三优化器动量 + 步数）。优化器在下方才创建，先暂存待回填。
    start_step, resume_ck = 0, None
    resume_path = resolve_resume(args, "siglip")
    if resume_path:
        _st = load_resume_ckpt(resume_path, model, args.device)
        start_step = _st["step"]
        resume_ck = _st
        print(f"从 {resume_path} 续训，起始 step {start_step}", flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)

    # 预处理目录（含 meta.json）：读离线张量，零现场解码；否则现场从 parquet 解码。
    if os.path.isfile(os.path.join(args.data_path, "meta.json")):
        ds = PreprocessedPairDataset(args.data_path)
    else:
        ds = ImageTextPairDataset(args.data_path, tokenizer, max_length=args.caption_len,
                                  vision_cfg=cfg.vision, source=args.source,
                                  image_size=args.image_size,
                                  shuffle_buffer=args.shuffle_buffer)
    # 训练模式：steps 与 epochs 默认都为 0，不传 = 直接报错不跑（防误触发整轮训练）。
    if not args.steps and not args.epochs:
        ap.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    loader, epochs_to_run, max_steps = build_loaders_and_steps(ds, args, collate_fn)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    if resume_ck is not None:
        muon.load_state_dict(resume_ck["muon"])
        adamw.load_state_dict(resume_ck["adamw"])
        sinkhorn.load_state_dict(resume_ck["sinkhorn"])
        print("已恢复三优化器状态（Muon/AdamW/Sinkhorn 动量）", flush=True)
    # DDP 包装：**必须在 make_optimizers 之后**（否则 named_parameters 带 `module.` 前缀
    # → 按名路由的优化器分组失效）。
    model = ddp.wrap(model)

    os.makedirs(args.save_dir, exist_ok=True)
    optimizers = (muon, adamw, sinkhorn)

    def forward_loss(batch):
        """SigLIP pairwise sigmoid 对比损失。

        输入：batch（patches/grid/text_ids/text_mask）。
        输出：(loss, None)。loss 已按 accumulation_steps 缩放。
        预期行为：bf16 autocast；对角为配对正样本。
        """
        text_ids = batch["text_ids"].to(args.device)
        text_mask = batch["text_mask"].to(args.device)
        patches = [p.to(args.device) for p in batch["patches"]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(patches, batch["grid"], text_ids, text_mask)
            # **DDP 注意**：SigLIP 对比损失是**卡内**对比（正负样本都在本卡 batch 内）。
            # 多卡下各卡独立算对比 → 与单卡大 batch 的对比**不等价**（负样本集不同）。
            # 要做到等价须 all_gather 各卡特征再算全局对比（另一处改动）。SigLIP 阶段
            # 已完成、不再重跑，故此处仅保证 DDP 能跑，数值语义差异**已知且未修**。
            loss = SigLIP.loss(logits) / args.accumulation_steps
        return loss, None

    def step_fn():
        """标准更新（裁剪 → 三优化器 step → 清梯度）。

        预期行为：SigLIP 双塔没有 MoE，故不做 noaux_tc 的 router bias 更新。
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        muon.step(); adamw.step(); sinkhorn.step()
        muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()

    def skip_bad_batch(batch):
        """对比损失需要 batch 内至少 2 个样本（1×1 的 logits 退化）。

        输入：collate 后的 batch（可能为 None）。
        输出：True=跳过。
        """
        return batch is None or batch["text_ids"].size(0) < 2

    run_epochs(ds, loader, model, optimizers, args, forward_loss, step_fn, "siglip",
               start_step, epochs_to_run, max_steps, skip_bad_batch=skip_bad_batch)

    # 存盘只用 rank0，且 unwrap 去 `module.` 前缀（与单卡兼容）。
    if ddp.is_main():
        raw = ddp.unwrap(model)
        # 只留视觉编码器（参考实现：丢弃 LLM，保留优化后的 ViT）
        _atomic_save(raw.image.vit.state_dict(), os.path.join(args.save_dir, "vit_siglip.pth"))
        # 另存**完整双塔**（ViT + 投影头 + 文本塔 + 可学习温度/偏置）：参考实现只留 ViT，但保留
        # 全模型才能对训好的编码器做**检索评测**（image→text 命中率），否则无从判断对比质量。
        _atomic_save(raw.state_dict(), os.path.join(args.save_dir, "siglip_full.pth"))
        print("SigLIP 对比预训练完成，已保存 vit_siglip.pth（仅视觉编码器）"
              "+ siglip_full.pth（完整双塔，供检索评测）")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
