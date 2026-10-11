"""Engram 容量扫描：验证论文 §3.1 的最优比例（Engram 占非激活参数的 20~25%）。

固定骨干，只改 Engram 表大小（bucket 数），看 loss 曲线。
预期：U 型——太小欠容量、太大浪费，最优点在中间。
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

# bucket 数 → 对应 Engram 表规模
BUCKETS = [2432, 4864, 9728, 19456, 38912]


def run_one(bucket, loader, tokenizer, args, log):
    """固定骨干、只改 bucket 训练一段。"""
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.bfloat16)
    cfg = make_ablation("+Engram")
    cfg.text.engram_vocab_size = bucket
    # 行数需按素数总和重算（用 EngramHash 的规则）
    from model.layers.engram_hash import find_next_prime
    seen, rows_per_layer = set(), []
    for _ in cfg.text.engram_layer_ids:
        ps = []
        for _ in range(cfg.text.engram_max_ngram_size - 1):
            cur = bucket - 1
            for _ in range(cfg.text.engram_n_heads):
                cur = find_next_prime(cur, seen); seen.add(cur); ps.append(cur)
        rows_per_layer.append(sum(ps))
    cfg.text.engram_num_embeddings = tuple(rows_per_layer)

    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    n_engram = sum(p.numel() for n, p in model.named_parameters() if "engram" in n)
    n_total = sum(p.numel() for p in model.parameters())
    muon, adamw, sinkhorn = make_optimizers(model, lr=args.lr)
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
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(gn):
            log(f"  bucket {bucket} 梯度非有限，停止"); break
        muon.step(); adamw.step(); sinkhorn.step()
        muon.zero_grad(); adamw.zero_grad(); sinkhorn.zero_grad()
        losses.append(loss.item())
    dt = time.time() - t0
    log(f"bucket {bucket:6d}: engram {n_engram/1e6:5.1f}M / total {n_total/1e6:5.1f}M "
        f"final loss {losses[-1]:.4f} ({dt:.0f}s)")
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"bucket": bucket, "engram_params": n_engram, "total_params": n_total,
            "final_loss": losses[-1], "losses": losses}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out/engram_sweep.json")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    logf = Path(args.out).with_suffix(".log")
    def log(m):
        print(m, flush=True)
        with logf.open("a", encoding="utf-8") as f:
            f.write(m + "\n")

    tokenizer = load_tokenizer()
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    results = []
    for b in BUCKETS:
        log(f"=== bucket {b} ===")
        try:
            results.append(run_one(b, loader, tokenizer, args, log))
        except Exception as e:
            log(f"[bucket {b}] 异常：{type(e).__name__}: {e}")
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    log("\n=== 汇总 ===")
    for r in results:
        log(f"  bucket {r['bucket']:6d}  engram {r['engram_params']/1e6:5.1f}M  loss {r['final_loss']:.4f}")


if __name__ == "__main__":
    main()
