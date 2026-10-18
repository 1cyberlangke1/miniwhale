"""全参数 SFT 入口（对话微调，从预训练权重起）。

照 minimind `train_full_sft.py` 风格：argparse + train_epoch + __main__。
与预训练的区别：数据是多轮对话（`SFTDataset`，只对 assistant 段算 loss）、学习率更小。
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
from model.train.data import collate_fn, SFTDataset
from model.model import Model
from model.layers.indexer import IndexerLossAutoScaler
from model.train.compile import maybe_compile
from model.train.checkpoint import load_resume_ckpt, save_final
from model.train.logfile import setup_log_file
from model.train.loop import (build_loaders_and_steps, run_epochs, update_weights)
from model.train.dspark_loss import compute_dspark_loss
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="miniwhale 全参数 SFT")
    parser.add_argument("--data_path", type=str, default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="起点权重：**只能**是上一阶段（阶段 4）的产物 dspark.pth。"
                             "它是完整模型（主干 + vision.* + aligner.* + dspark.*，416 键），"
                             "SFT 直接在此基础上继续训。更早阶段的权重会跳过已训好的组件，不可使用")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=768)
    parser.add_argument("--router_bias_rate", type=float, default=1e-3)
    parser.add_argument("--dspark", action=argparse.BooleanOptionalAction, default=True,
                        help="post-training 阶段同训 DSpark（照报告 §2.4.3；主干由 CE 训，"
                             "DSpark 由 DSpark loss 训，梯度不回传主干）。**默认开启**，"
                             "与报告一致；确需关闭用 --no-dspark")
    add_common_args(parser)
    args = parser.parse_args()
    # 各阶段默认值（命令行显式给出的参数不受影响）
    apply_defaults(args, learning_rate=1e-5, min_lr=1e-6, warmup_steps=200,
                   accumulation_steps=8, num_workers=1)
    # DDP 初始化（非分布式时为 no-op，返回 --device）。**必须在建模型之前**。
    args.device = ddp.setup(args.device)
    if ddp.is_main():
        setup_log_file(args.log_file, append=bool(args.resume or args.from_resume))

    torch.manual_seed(args.seed)
    # 参数保持 **fp32 主权重**（按技术报告 §3.4）；计算走 bf16 autocast。
    # 见 train_pretrain 同名注释：bf16 参数会让 RMSNorm 权重的更新被舍入抹掉。

    cfg = ModelConfig.from_json()
    tokenizer = load_tokenizer()
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    if args.ckpt:
        # strict=False：允许起点权重与当前模型有键差（如视觉塔）。
        # 但**必须打印缺失/多余的键**——否则键名对不上时静默拿随机初始化训。
        # 训练中间档是 {"model","muon","adamw","sinkhorn","step"}（见 --resume 分支）；
        # 传这种档给 --ckpt（如上一桶产物）时必须解包 "model"，否则 416 键全对不上、
        # 静默从随机初始化训。纯 state_dict 则原样使用。
        _st = torch.load(args.ckpt, map_location=args.device)
        if isinstance(_st, dict) and "model" in _st:
            _st = _st["model"]
        _miss, _unexp = model.load_state_dict(_st, strict=False)
        print(f"载入 {args.ckpt}：缺失 {len(_miss)} 键 / 多余 {len(_unexp)} 键", flush=True)
        if _miss:
            print(f"  缺失（用随机初始化）: {_miss[:8]}{' …' if len(_miss) > 8 else ''}",
                  flush=True)
    # 续训：恢复完整训练状态（模型 + 三优化器动量 + 步数）。优化器在下方才创建，先暂存待回填。
    start_step, resume_ck = 0, None
    resume_path = resolve_resume(args, "sft")
    if resume_path:
        _st = load_resume_ckpt(resume_path, model, args.device)
        start_step = _st["step"]
        resume_ck = _st
        print(f"从 {resume_path} 续训，起始 step {start_step}", flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # **多模态 SFT**（报告 §1：post-training 覆盖 text + multimodal）：SFTDataset 已扩展
    # 支持 image_bytes——给 vision_cfg/image_token_id 时展开图像 span 并输出
    # token_types/images；数据无图时行为与纯文本 SFT 完全一致。
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                    vision_cfg=cfg.vision, image_token_id=cfg.text.image_token_id)
    # 训练模式：steps 与 epochs 默认都为 0，不传 = 直接报错不跑（防误触发整轮训练）。
    if not args.steps and not args.epochs:
        parser.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    # collate_fn 必须显式给：多模态样本的 `images` 是长度不定的嵌套 list，
    # 默认 collate 会按「等长」去 stack 而报错。
    loader, epochs_to_run, max_steps = build_loaders_and_steps(ds, args, collate_fn)

    muon, adamw, sinkhorn = make_optimizers(model, lr=args.learning_rate)
    if resume_ck is not None:
        muon.load_state_dict(resume_ck["muon"])
        adamw.load_state_dict(resume_ck["adamw"])
        sinkhorn.load_state_dict(resume_ck["sinkhorn"])
        print("已恢复三优化器状态（Muon/AdamW/Sinkhorn 动量）", flush=True)
    # DDP 包装：**必须在 make_optimizers 之后**（否则 named_parameters 带 `module.` 前缀
    # → 按名路由的优化器分组失效）。find_unused_parameters=True：MoE 每步有未激活专家。
    model = ddp.wrap(model)

    os.makedirs(args.save_dir, exist_ok=True)
    optimizers = (muon, adamw, sinkhorn)
    dspark_on = args.dspark and ddp.unwrap(model).dspark is not None

    def forward_loss(batch):
        """主干 CE + （可选）DSpark loss。

        输入：batch（input_ids/labels/token_types/images）。
        输出：(loss, extra)。extra 为 DSpark 分项（未开 DSpark 时为 None）。
        预期行为：bf16 autocast；主干 + DSpark **一次 forward**（DSpark 前向必须在
                  model.forward 内，否则 DDP 遍历不到 dspark 参数 → 判其 unused →
                  backward 报「Expected to mark a variable ready only once」）；
                  DSpark 的 target 走 detach，故其梯度不回传主干（参考实现 §2.4.3 语义）。
        """
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        # 多模态：图像 span 与 token_types（TEXT=-1，非负为图像位置类型）
        token_types = batch["token_types"].to(args.device)
        images = batch["images"] if any(batch["images"]) else None
        # 只在 DSpark 开启时才建 loss_mask（未开时这张量纯属浪费）。
        loss_mask = (labels != -100).float() if dspark_on else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 多模态：有图时传 images/token_types（与阶段 3 的 VL 前向同一接口）。
            # 无图样本 token_types 全 -1，前向行为与纯文本完全一致。
            if dspark_on:
                logits, _, out = model(ids, images=images, token_types=token_types,
                                       dspark_train={"loss_mask": loss_mask})
            else:
                logits, _ = model(ids, images=images, token_types=token_types)
            # **全局 token 归一化**（DDP 逐位等价的关键）：单卡时 = 批内 mean（数值不变）；
            # 多卡时各卡算 sum、按全局 valid token 数归一 → DDP 平均梯度后等于单卡全局 mean。
            tgt = labels[:, 1:].reshape(-1)
            ce_sum = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tgt,
                ignore_index=-100, reduction="sum")
            n_valid = (tgt != -100).sum()
            loss = ddp.scale_loss_for_ddp(ce_sum, n_valid)
            # post-training 阶段同训 DSpark（照报告 §2.4.3）：主干由上面的 CE 更新；
            # DSpark 由 DSpark loss 更新。DSpark loss 的 target（主干 hidden/logits）
            # 天然不带梯度（target 走 detach），故其梯度不回传主干——正是参考实现
            # 「同训但梯度不回传 backbone」的语义。
            # **DDP 注意**：compute_dspark_loss 内部是 per-rank masked mean，多卡下
            # 与全局 mean 不完全等价（各卡 mask 和 token 数不同）。SFT+dspark 同训属
            # 阶段 5，上 DDP 前须先把其归一化改为全局（或确认差异可接受）。
            extra = None
            if dspark_on:
                tc = ddp.unwrap(model).cfg.text
                dspark_loss, dparts = compute_dspark_loss(
                    out,
                    loss_decay_gamma=tc.dspark_loss_decay_gamma,
                    ce_loss_alpha=tc.dspark_ce_loss_alpha,
                    l1_loss_alpha=tc.dspark_l1_loss_alpha,
                    confidence_head_alpha=tc.dspark_confidence_head_alpha,
                )
                loss = loss + dspark_loss
                extra = dparts
            loss = loss / args.accumulation_steps
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        return loss, extra

    def step_fn():
        """标准更新（裁剪 → 三优化器 step → 清梯度 → 路由 bias）。"""
        update_weights(model, muon, adamw, sinkhorn, args)

    run_epochs(ds, loader, model, optimizers, args, forward_loss, step_fn, "sft",
               start_step, epochs_to_run, max_steps)

    if ddp.is_main():
        save_final(model, args.save_dir, "sft")
        print("SFT 完成，已保存 sft.pth")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
