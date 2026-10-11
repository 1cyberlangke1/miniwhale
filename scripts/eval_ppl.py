"""评测：在保留集上算困惑度（perplexity）。

用于对照实验的收敛质量度量。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train.ablation import make_ablation
from model.train.data import PretrainDataset
from model.model import Model
from model.tokenizer import load_tokenizer


@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 20):
    """算平均交叉熵与困惑度。

    输入：model；loader；device；max_batches 上限。
    输出：(loss, ppl)。
    预期行为：只对非 -100 位置算 loss；困惑度 = exp(loss)。
    """
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu",
                            dtype=torch.bfloat16):
            logits, _ = model(ids)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)
        total += loss.item(); n += 1
    avg = total / max(n, 1)
    return avg, float(torch.exp(torch.tensor(avg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", default="+Engram")
    ap.add_argument("--data_path", default="dataset/pretrain_t2t_mini.jsonl")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    tokenizer = load_tokenizer()
    cfg = make_ablation(args.stage)
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    # strict=False：DSpark 参数不在对照阶段权重里
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device), strict=False)

    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    loss, ppl = evaluate(model, loader, args.device)
    print(f"{args.stage}  loss {loss:.4f}  ppl {ppl:.2f}")


if __name__ == "__main__":
    main()
