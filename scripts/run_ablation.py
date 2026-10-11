"""对照实验：五个阶段各训一小段，记录 loss 曲线。

同分词器、同语料、同 token 预算，唯一变量是架构。
用法：python scripts/run_ablation.py --steps 200 --seq_len 256 --batch 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train.ablation import make_ablation
from model.train.data import PretrainDataset
from model.model import Model
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer

STAGES = ["dense", "+MoE", "+mHC", "+CSA2", "+Engram"]


def run_stage(stage, loader, tokenizer, args):
    """训练单个阶段，返回 loss 曲线与参数量。"""
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    cfg = make_ablation(stage)
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    muon, adamw = make_optimizers(model, lr=args.lr)
    model.train()

    losses = []
    it = iter(loader)
    t0 = time.time()
    for step in range(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # 梯度健康检查：非有限梯度直接报错，避免静默 NaN
        if not torch.isfinite(gn):
            print(f"  [{stage}] step {step} 梯度非有限，停止本阶段")
            break
        muon.step(); adamw.step(); muon.zero_grad(); adamw.zero_grad()
        losses.append(loss.item())
        if step % 20 == 0:
            print(f"  {stage} step {step}/{args.steps} loss {loss.item():.4f}")
    dt = time.time() - t0
    print(f"{stage}: {n_params/1e6:.1f}M params, final loss {losses[-1]:.4f}, {dt:.0f}s")
    return {"stage": stage, "params": n_params, "losses": losses, "time": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out/ablation_results.json")
    args = ap.parse_args()

    tokenizer = load_tokenizer()
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    results = []
    for stage in STAGES:
        print(f"=== {stage} ===")
        results.append(run_stage(stage, loader, tokenizer, args))
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()   # 释放上一阶段，避免显存/状态污染

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n结果写入 {args.out}")
    for r in results:
        print(f"  {r['stage']:12s} {r['params']/1e6:6.1f}M  final loss {r['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
