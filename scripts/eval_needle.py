"""长上下文 needle 检索评测。

在长文本中插入一个「针」，测模型能否在对应位置检索到它。
用于对照实验中度量长上下文能力（CSA2 压缩 KV 的主要卖点）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.train.ablation import make_ablation
from model.model import Model
from model.tokenizer import load_tokenizer

NEEDLE = "魔法数字是七四三。"
QUESTION = "魔法数字是多少？"


def build_prompt(filler: str, depth: float) -> str:
    """按深度把针插进填充文本。

    输入：filler 背景文本；depth ∈ [0,1] 插入位置比例。
    输出：拼好的 prompt 字符串。
    预期行为：针放在 depth 位置，问题放最后。
    """
    cut = int(len(filler) * depth)
    return filler[:cut] + NEEDLE + filler[cut:] + QUESTION


def eval_needle(model, tokenizer, filler, depths, max_len, device):
    """在多个深度上测检索命中率。

    输入：model；tokenizer；filler；depths 深度列表；max_len；device。
    输出：{depth: 是否命中}。
    预期行为：生成 8 个 token，看是否包含「七四三」。
    """
    model.eval()
    hits = {}
    for d in depths:
        prompt = build_prompt(filler, d)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=8) if hasattr(model, "generate") else None
        hits[d] = None  # 模型暂无 generate，占位
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", default="+Engram")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = load_tokenizer()
    cfg = make_ablation(args.stage)
    model = Model(cfg, tokenizer=tokenizer).to(args.device)
    model.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    print("needle 评测需要 generate 支持（待补）")


if __name__ == "__main__":
    main()
