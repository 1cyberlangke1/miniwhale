"""长上下文基准：对比各阶段在不同序列长度下的 perplexity。

CSA2 压缩 KV 的价值在长序列上才显现（压缩 KV 让远处信息以更低成本保留）。
本脚本在 256 / 512 / 1024 / 2048 长度上测各阶段模型，看衰减曲线。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train.ablation import make_ablation
from model.train.data import PretrainDataset
from model.model import Model
from model.tokenizer import load_tokenizer

STAGES = ["dense", "+MoE", "+mHC", "+CSA2", "+Engram"]
LENGTHS = [256, 512, 1024, 2048]


@torch.no_grad()
def eval_at_length(model, tokenizer, data_path, seq_len, batch, device, max_batches=8):
    """在指定序列长度上算平均 loss。"""
    model.eval()
    ds = PretrainDataset(data_path, tokenizer, max_length=seq_len)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    total, n = 0.0, 0
    for i, batch_ in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch_["input_ids"].to(device)
        labels = batch_["labels"].to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16):
            logits, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        total += loss.item(); n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="out")
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="out/long_context.json")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = load_tokenizer()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    results = {}
    for stage in STAGES:
        safe = stage.replace("+", "p").replace("/", "_")
        ckpt = Path(args.ckpt_dir) / f"{safe}.pth"
        if not ckpt.exists():
            print(f"{stage}: 无权重，跳过")
            continue
        cfg = make_ablation(stage)
        model = Model(cfg, tokenizer=tokenizer).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device), strict=False)
        row = {}
        for L in LENGTHS:
            row[L] = eval_at_length(model, tokenizer, args.data_path, L, args.batch, args.device)
            print(f"  {stage} len={L}: loss {row[L]:.4f}", flush=True)
        results[stage] = row
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n写入 {args.out}")
    for stage, row in results.items():
        print(f"  {stage:12s} " + "  ".join(f"{L}:{v:.3f}" for L, v in row.items()))


if __name__ == "__main__":
    main()
