"""通宵训练：对照阶梯跑更长步数，出收敛曲线。

同分词器/语料/token 预算，唯一变量是架构。
每阶段独立 seed、独立保存 loss 曲线，异常自动记录不中断。

用法：python -u scripts/run_overnight.py --steps 2000 --seq_len 256 --batch 4
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
from model.layers.indexer import IndexerLossAutoScaler
from model.model import Model
from model.train.optim import make_optimizers
from model.tokenizer import load_tokenizer

STAGES = ["dense", "+MoE", "+mHC", "+CSA2", "+Engram"]


def run_stage(stage, loader, tokenizer, args, log):
    """训练单个阶段并返回曲线。"""
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    cfg = make_ablation(stage)
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    muon, adamw = make_optimizers(model, lr=args.lr)
    model.train()

    losses, it = [], iter(loader)
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
        IndexerLossAutoScaler.set_loss_scale(loss.detach())
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(gn):
            log(f"  [{stage}] step {step} 梯度非有限，停止")
            break
        muon.step(); adamw.step(); muon.zero_grad(); adamw.zero_grad()
        model.update_router_bias(args.router_bias_rate)
        losses.append(loss.item())
        if step % args.log_interval == 0:
            log(f"  {stage} step {step}/{args.steps} loss {loss.item():.4f} "
                f"{(time.time()-t0)/max(step+1,1)*1000:.0f}ms/step")
    dt = time.time() - t0
    log(f"{stage}: {n_params/1e6:.1f}M params, final loss {losses[-1]:.4f}, {dt:.0f}s")
    safe = stage.replace("+", "p").replace("/", "_")
    torch.save(model.state_dict(), Path(args.save_dir) / f"{safe}.pth")
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"stage": stage, "params": n_params, "losses": losses, "time": dt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--router_bias_rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", default="out")
    ap.add_argument("--out", default="out/overnight_results.json")
    args = ap.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    logf = Path(args.out).with_suffix(".log")
    def log(msg):
        print(msg, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    tokenizer = load_tokenizer()
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    results = []
    for stage in STAGES:
        log(f"=== {stage} ===")
        try:
            results.append(run_stage(stage, loader, tokenizer, args, log))
        except Exception as e:
            log(f"[{stage}] 异常：{type(e).__name__}: {e}")
            results.append({"stage": stage, "error": str(e)})
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    log("\n=== 汇总 ===")
    for r in results:
        if "losses" in r:
            log(f"  {r['stage']:12s} {r['params']/1e6:6.1f}M  final {r['losses'][-1]:.4f}")


if __name__ == "__main__":
    main()
