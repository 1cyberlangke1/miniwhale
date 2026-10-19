"""命令行推理入口。

用法：
    python -m inference.chat --ckpt out/sft.pth --prompt "你好"
    python -m inference.chat --ckpt out/sft.pth --spec --temperature 0.6
"""

from __future__ import annotations

import argparse
import time

import torch

from inference.engine import Engine, GenerationConfig
from inference.spec import SpeculativeConfig, generate_speculative
from model.tokenizer import load_tokenizer


def main():
    ap = argparse.ArgumentParser(description="miniwhale 推理")
    ap.add_argument("--ckpt", default="out/sft.pth")
    ap.add_argument("--prompt", default="你好")
    ap.add_argument("--max_new", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--spec", action="store_true", help="启用 DSpark 投机解码")
    ap.add_argument("--confidence_threshold", type=float, default=0.0)
    ap.add_argument("--compile", action="store_true", help="torch.compile 解码路径")
    ap.add_argument("--cuda_graph", action="store_true",
                    help="CUDA Graph 重放单步解码（形状静态，换输入不重编译）")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = load_tokenizer()
    engine = Engine(args.ckpt, device=args.device, compile=args.compile, tokenizer=tok,
                    cuda_graph=args.cuda_graph)
    ids = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                  tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    x = torch.tensor([list(ids)], device=args.device)

    t0 = time.time()
    if args.compile or args.cuda_graph:
        # 用真实 prompt 形状预热：编译 / 图上传 / 首次 replay 的代价都要挪到计时之外
        engine.warmup(x)
        t0 = time.time()
    if args.spec:
        cfg = SpeculativeConfig(max_new_tokens=args.max_new, temperature=args.temperature,
                                eos_token_id=tok.eos_token_id,
                                confidence_threshold=args.confidence_threshold)
        out, stats = generate_speculative(engine, x, cfg)
        print(f"\n[dspark] drafted={stats.drafted} accepted={stats.accepted} "
              f"rounds={stats.rounds} accept_rate={stats.accept_rate:.3f} "
              f"tau={stats.tau:.2f}")
    else:
        cfg = GenerationConfig(max_new_tokens=args.max_new, temperature=args.temperature,
                               eos_token_id=tok.eos_token_id)
        out = engine.generate(x, cfg)
    dt = time.time() - t0
    n_new = out.size(1) - x.size(1)
    print(f"\n[{n_new} tokens, {dt:.2f}s, {n_new / max(dt, 1e-9):.1f} tok/s]")
    print(tok.decode(out[0, x.size(1):].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
