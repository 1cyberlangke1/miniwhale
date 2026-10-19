"""多模态训练（ViT + Aligner + LLM 联合）。

按参考实现的 VL 接法：图像 → ViT → Aligner → 覆盖图像 span 的 embedding
（IMAGE 槽填 aligner 行，start/end/newline 填学习嵌入）→ 主干前向。图像 span 全部携带
同一 image_token_id，靠 token_types 区分（参考实现机制）。

loss：与纯文本一致（只对非 -100 位置），但图像位置参与前向、其梯度回传 ViT/Aligner。
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
from model.train.data import collate_fn, MultimodalDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.checkpoint import _atomic_save, load_resume_ckpt
from model.train.logfile import setup_log_file
from model.train.loop import (build_loaders_and_steps, run_epochs, update_weights)
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser(description="miniwhale 多模态训练")
    ap.add_argument("--data_path", default="dataset/sft_i2t_mini.parquet",
                    help="纯图文 QA parquet（由 scripts/extract_i2t_mini.py 抽取）")
    ap.add_argument("--ckpt", default=None, help="已训练的主干权重（可选）")
    ap.add_argument("--vit_ckpt", default=None,
                    help="第一阶段（SigLIP 对比）产出的视觉编码器权重（可选；按参考实现两阶段）")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--router_bias_rate", type=float, default=1e-3)
    add_common_args(ap)
    args = ap.parse_args()
    # 各阶段默认值（命令行显式给出的参数不受影响）
    apply_defaults(args, learning_rate=1e-4, min_lr=1e-5, warmup_steps=100,
                   accumulation_steps=4, num_workers=0)
    # DDP 初始化（非分布式时为 no-op，返回 --device）。**必须在建模型之前**。
    args.device = ddp.setup(args.device)
    # 日志落盘：**在 Python 里直写文件**（不经 shell 管道）——shell 的 `| grep` 会块缓冲，
    # 导致 `flush=True` 的 print 也到不了文件（攒满 4KB 才吐）。
    # 分布式下只有 rank0 写日志，避免多进程争抢同一文件。
    if ddp.is_main():
        setup_log_file(args.log_file, append=bool(args.resume or args.from_resume))

    torch.manual_seed(args.seed)
    # 参数保持 **fp32 主权重**（按技术报告 §3.4）；计算走 bf16 autocast。
    # 见 train_pretrain 同名注释：bf16 参数会让 RMSNorm 权重的更新被舍入抹掉。
    tokenizer = load_tokenizer()
    cfg = ModelConfig.from_json()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        # strict=False：主干权重里没有视觉塔的键是正常的（视觉塔另有 --vit_ckpt）。
        # 但**必须打印缺失/多余的键**——否则起点权重键名对不上时静默拿随机初始化训，
        # 表面在跑、实则白训（曾因静默吞键埋隐患）。
        _st = torch.load(args.ckpt, map_location=args.device)
        _miss, _unexp = model.load_state_dict(_st, strict=False)
        print(f"载入主干 {args.ckpt}：缺失 {len(_miss)} 键 / 多余 {len(_unexp)} 键", flush=True)
        if _miss:
            print(f"  缺失（用随机初始化）: {_miss[:8]}{' …' if len(_miss) > 8 else ''}",
                  flush=True)
        if _unexp:
            print(f"  多余（被忽略）: {_unexp[:8]}{' …' if len(_unexp) > 8 else ''}", flush=True)
    # 第二阶段从第一阶段（SigLIP 对比）的视觉编码器起（按参考实现：先对比预训练再自回归微调）
    if args.vit_ckpt:
        vit_state = torch.load(args.vit_ckpt, map_location=args.device)
        model.vision.load_state_dict(vit_state)
        print(f"已载入第一阶段视觉编码器 {args.vit_ckpt}", flush=True)
    # 续训：从本阶段中间档（vlm_e{E}_s{STEP}.pth）恢复**完整**训练状态
    # （模型权重 + 三个优化器动量 + 步数）。只存权重不存优化器 → 续训时 Muon/AdamW/
    # Sinkhorn 的动量全归零，等于优化器失忆，比不续训好不了多少。优化器在下面才创建，
    # 故此处先暂存 checkpoint，待 make_optimizers 后回填。
    start_step, resume_ck = 0, None
    resume_path = resolve_resume(args, "vlm")
    if resume_path:
        _st = load_resume_ckpt(resume_path, model, args.device)
        start_step = _st["step"]
        resume_ck = _st
        print(f"从 {resume_path} 续训，起始 step {start_step}", flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)
    assert model.vision is not None, "模型未启用视觉塔"
    # **DDP 包装不在此处**：必须在 make_optimizers **之后**。DDP 包装会让
    # named_parameters() 带 `module.` 前缀 → make_optimizers 的按名路由（Muon/AdamW/
    # Sinkhorn 分组）会失效。故先建优化器（此时 model 未包），再包 DDP（见下方）。

    ds = MultimodalDataset(args.data_path, tokenizer, max_length=args.seq_len,
                           image_token_id=cfg.text.image_token_id, vision_cfg=cfg.vision)
    # 训练模式：steps 与 epochs 默认都为 0，不传 = 直接报错不跑（防误触发整轮训练）。
    if not args.steps and not args.epochs:
        ap.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    loader, epochs_to_run, max_steps = build_loaders_and_steps(ds, args, collate_fn)

    # 主干 + ViT/Aligner 一起训（参考实现：视觉塔与 LLM 联合优化）
    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    if resume_ck is not None:
        # 回填三优化器动量（resume_ck 在 make_optimizers 之前载入，此处才能恢复）
        muon.load_state_dict(resume_ck["muon"])
        adamw.load_state_dict(resume_ck["adamw"])
        sinkhorn.load_state_dict(resume_ck["sinkhorn"])
        print("已恢复三优化器状态（Muon/AdamW/Sinkhorn 动量）", flush=True)

    # DDP 包装：**必须在 make_optimizers 之后**（否则 named_parameters 带 `module.`
    # 前缀 → 按名路由的优化器分组失效）。find_unused_parameters=True：MoE 每步有未激活
    # 专家，须允许部分参数不参与梯度。
    model = ddp.wrap(model)

    os.makedirs(args.save_dir, exist_ok=True)
    optimizers = (muon, adamw, sinkhorn)

    def forward_loss(batch):
        """图像注入 + 文本交叉熵。

        输入：batch（input_ids/labels/token_types/images）。
        输出：(loss, None)。loss 已按 accumulation_steps 缩放。
        预期行为：bf16 autocast；图像 span 覆盖为 ViT/Aligner 特征；梯度回传
                  ViT/Aligner/主干；全局 token 归一化保证 DDP 逐位等价。
        """
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        token_types = batch["token_types"].to(args.device)
        # images 是 list[list[ImageInput]]，仅在有图像时传
        images = batch["images"] if any(batch["images"]) else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids, images=images, token_types=token_types)
            # **全局 token 归一化**（DDP 逐位等价的关键）：单卡时 = 批内 mean（数值不变）；
            # 多卡时各卡算 sum、按全局 valid token 数归一 → DDP 平均梯度后恰好等于
            # 单卡 batch=world_size*B 的全局 mean。直接用批内 mean 会因各卡 token 数不同而不等价。
            tgt = labels[:, 1:].reshape(-1)
            ce_sum = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tgt,
                ignore_index=-100, reduction="sum")
            n_valid = (tgt != -100).sum()
            loss = ddp.scale_loss_for_ddp(ce_sum, n_valid) / args.accumulation_steps
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        return loss, None

    def step_fn():
        """标准更新（裁剪 → 三优化器 step → 清梯度 → 路由 bias）。"""
        update_weights(model, muon, adamw, sinkhorn, args)

    run_epochs(ds, loader, model, optimizers, args, forward_loss, step_fn, "vlm",
               start_step, epochs_to_run, max_steps)

    # 存盘只用 rank0；且必须 unwrap（否则 state_dict 键带 `module.` 前缀，与单卡不兼容）。
    if ddp.is_main():
        raw = ddp.unwrap(model)
        _atomic_save(raw.state_dict(), os.path.join(args.save_dir, "vlm.pth"))
        # ViT 自回归微调阶段（报告 §4.2.2 阶段二）：只保留优化后的视觉编码器，丢弃 LLM
        _atomic_save(raw.vision.state_dict(), os.path.join(args.save_dir, "vit_ar.pth"))
        print("多模态训练完成，已保存 vlm.pth + vit_ar.pth（仅视觉编码器）")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
