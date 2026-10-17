"""全参数 SFT 入口（对话微调，从预训练权重起）。

照 minimind `train_full_sft.py` 风格：argparse + train_epoch + __main__。
与预训练的区别：数据是多轮对话（`SFTDataset`，只对 assistant 段算 loss）、学习率更小。
"""

from __future__ import annotations

import argparse
import math
import os
import re
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


def collate_fn(batch):
    """把多模态样本拼成 batch；图像 span 保持 list（每样本的图像输入）。

    输入：样本 dict 列表（MultimodalDataset 产物）。
    输出：{"input_ids":[B,L], "labels":[B,L], "token_types":[B,L], "images": list[list]}。
    预期行为：张量字段 stack，images 原样保留为嵌套 list（含 ImageInput 对象）。
    """
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "token_types": torch.stack([b["token_types"] for b in batch]),
        "images": [b["images"] for b in batch],
    }


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
        step = steps
        gstep = start_step + step          # **全局**步数：lr 调度与 checkpoint 都按它算
        t_step0 = time.time()
        # 只更新基准 lr，各组按 lr_mult 缩放（保留 Engram ×5，参考实现 §4.2.2）
        lr = get_lr(gstep, args.total_steps, args.warmup_steps, args.learning_rate, args.min_lr)
        set_base_lr((muon, adamw, sinkhorn), lr)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        # 多模态：图像 span 与 token_types（TEXT=-1，非负为图像位置类型）
        token_types = batch["token_types"].to(args.device)
        images = batch["images"] if any(batch["images"]) else None
        dspark_on = args.dspark and ddp.unwrap(model).dspark is not None
        # 只在 DSpark 开启时才建 loss_mask（未开时这张量纯属浪费）。
        loss_mask = (labels != -100).float() if dspark_on else None
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # 主干 + DSpark **一次 forward**：DSpark 前向必须在 model.forward 内，否则
            # DDP 遍历不到 dspark 参数（前向在 forward 之外）→ 判其 unused → backward
            # 收到梯度时报「Expected to mark a variable ready only once」。见 model.forward。
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
            if dspark_on:
                tc = ddp.unwrap(model).cfg.text
                dspark_loss, _ = compute_dspark_loss(
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
            ddp.unwrap(model).update_router_bias(args.router_bias_rate)
        total_loss += loss.item() * args.accumulation_steps
        steps += 1
        if gstep % args.log_interval == 0 and ddp.is_main():
            # 打印**瞬时**每步耗时（非累计均值），才能看清稳态速度。仅 rank0 打印。
            now = time.time()
            print(f"epoch {epoch} step {gstep} loss {total_loss / steps:.4f} "
                  f"lr {adamw.param_groups[0]['lr']:.2e} "
                  f"{(now - t_step0) * 1000:.0f}ms/step", flush=True)
            t_step0 = now
        # 长训练定期存 checkpoint（否则中途崩溃会丢整轮）。**仅 rank0 存**。
        if args.save_interval and (gstep + 1) % args.save_interval == 0 and ddp.is_main():
            os.makedirs(args.save_dir, exist_ok=True)
            path = os.path.join(args.save_dir, f"sft_e{epoch}_s{gstep+1}.pth")
            # 存**完整训练状态**（模型 + 三优化器动量 + 步数），续训才能无缝接上。
            # unwrap 去 `module.` 前缀，与单卡兼容。
            torch.save({
                "model": ddp.unwrap(model).state_dict(),
                "muon": muon.state_dict(),
                "adamw": adamw.state_dict(),
                "sinkhorn": sinkhorn.state_dict(),
                "step": gstep + 1,
            }, path)
            # 只清理**本次训练自己产生的中间档** `sft_e<epoch>_s<step>.pth`。
            # 曾用 `f.startswith("sft_")` → 会连桶快照 `sft_<bucket>.pth` 一起匹配删除。
            # 用正则精确匹配中间档命名，桶快照不匹配 → 永不误删。
            if args.keep_last_n:
                pat = re.compile(r"^sft_e\d+_s\d+\.pth$")
                cks = sorted(
                    (os.path.join(args.save_dir, f) for f in os.listdir(args.save_dir)
                     if pat.match(f)),
                    key=os.path.getmtime)
                for old in cks[:-args.keep_last_n]:
                    os.remove(old)
            print(f"checkpoint saved at step {gstep+1}", flush=True)
    return total_loss / max(steps, 1), steps


def main():
    parser = argparse.ArgumentParser(description="miniwhale 全参数 SFT")
    parser.add_argument("--data_path", type=str, default="dataset/sft_t2t_mini.jsonl")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="起点权重：**只能**是上一阶段（阶段 4）的产物 dspark.pth。"
                             "它是完整模型（主干 + vision.* + aligner.* + dspark.*，416 键），"
                             "SFT 直接在此基础上继续训。更早阶段的权重会跳过已训好的组件，不可使用")
    parser.add_argument("--resume", type=str, default=None,
                        help="从本阶段中间档续训（sft_e{E}_s{STEP}.pth；恢复完整训练状态）")
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
    parser.add_argument("--dspark", action=argparse.BooleanOptionalAction, default=True,
                        help="post-training 阶段同训 DSpark（照报告 §2.4.3；主干由 CE 训，"
                             "DSpark 由 DSpark loss 训，梯度不回传主干）。**默认开启**，"
                             "与报告一致；确需关闭用 --no-dspark")
    parser.add_argument("--compile", action="store_true",
                        help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    parser.add_argument("--compile_mode", type=str, default="default",
                        help='torch.compile 模式（固定 "default"）')
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
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device)
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"])
            start_step = int(ck.get("step", 0))
            resume_ck = ck
        else:
            model.load_state_dict(ck)
            m = re.search(r"_s(\d+)\.pth$", os.path.basename(args.resume))
            start_step = int(m.group(1)) if m else 0
        print(f"从 {args.resume} 续训，起始 step {start_step}", flush=True)
    model = maybe_compile(model, args.compile, args.compile_mode)
    print(f"参数量 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # **多模态 SFT**（报告 §1：post-training 覆盖 text + multimodal）：SFTDataset 已扩展
    # 支持 image_bytes——给 vision_cfg/image_token_id 时展开图像 span 并输出
    # token_types/images；数据无图时行为与纯文本 SFT 完全一致。
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len,
                    vision_cfg=cfg.vision, image_token_id=cfg.text.image_token_id)
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
        # 与 ep_steps 同分母（都除 world_size）；否则 DDP 下 lr 调度步数翻倍、只衰减一半。
        args.total_steps = (max_steps if max_steps
                            else max(ds.num_rows // args.batch_size // ddp.world_size(), 1)
                            * epochs_to_run)
    print(f"数据集 {ds.num_rows} 行；总步数 {args.total_steps}（warmup {args.warmup_steps}）",
          flush=True)
    # collate_fn 必须显式给：多模态样本的 `images` 是长度不定的嵌套 list，
    # 默认 collate 会按「等长」去 stack 而报错。
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=args.num_workers, pin_memory=True)

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
    global_step = start_step
    for epoch in range(epochs_to_run):
        ds.set_epoch(epoch)  # 流式数据集：每轮换 seed 重排数据顺序
        # DDP 下各 rank 数据量可能不等 → 固定步数（否则先耗尽者退出循环致 all_reduce 死锁）。
        ep_steps = max_steps
        if ddp.is_distributed() and ep_steps is None:
            ep_steps = max(ds.num_rows // args.batch_size // ddp.world_size(), 1)
        loss, n = train_epoch(epoch, loader, model, muon, adamw, sinkhorn, args,
                              start_step=global_step, max_steps=ep_steps)
        global_step += n   # 跨 epoch 累加：lr 余弦调度与 checkpoint 间隔都按全局步数
        if ddp.is_main():
            print(f"epoch {epoch} done, avg loss {loss:.4f}（累计 {global_step} 步）")
    if ddp.is_main():
        torch.save(ddp.unwrap(model).state_dict(), os.path.join(args.save_dir, "sft.pth"))
        print("SFT 完成，已保存 sft.pth")
    ddp.barrier()
    ddp.cleanup()


if __name__ == "__main__":
    # 多进程启动：单卡时直接跑；多卡时 spawn（notebook / 无 torchrun 环境也能用）。
    ddp.launch(main)
