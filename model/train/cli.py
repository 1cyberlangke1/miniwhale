"""argparse 公共参数（五个 trainer 共用的那 18 项）。

各 trainer 只注册自己特有的参数；通用项（优化器/调度/日志/checkpoint/设备）在这里统一
注册，保证名字与默认值一致。**默认值一律为 None**，由 `apply_defaults` 按 trainer
给出的覆盖表回填——这样既统一了名字，又保留了各阶段不同的默认值（如 pretrain lr=2.6e-4、
sft lr=1e-5）。
"""

from __future__ import annotations

import argparse


# 通用参数名 → 各 trainer 的默认值由 apply_defaults 覆盖
_COMMON_DEFAULTS = {
    "epochs": 0,
    "steps": 0,
    "learning_rate": None,
    "min_lr": None,
    "warmup_steps": None,
    "total_steps": 0,
    "accumulation_steps": 1,
    "grad_clip": 1.0,
    "num_workers": 1,
    "log_interval": 200,
    "save_interval": 1000,
    "keep_last_n": 3,
    "save_dir": "out",
    "log_file": None,
    "compile": False,
    "compile_mode": "default",
    "device": None,
    "seed": 42,
    "resume": None,
}


def add_common_args(ap: argparse.ArgumentParser):
    """注册通用训练参数。

    输入：argparse 解析器。
    输出：无（原地添加）。
    预期行为：名字/帮助文本与历史一致（`--epochs`/`--steps`/`--learning_rate`/
              `--min_lr`/`--warmup_steps`/`--total_steps`/`--accumulation_steps`/
              `--grad_clip`/`--num_workers`/`--log_interval`/`--save_interval`/
              `--keep_last_n`/`--save_dir`/`--log_file`/`--compile`/`--compile_mode`/
              `--device`/`--seed`/`--resume`）；默认值先占位，由 apply_defaults 覆盖。
    """
    d = _COMMON_DEFAULTS
    ap.add_argument("--epochs", type=int, default=d["epochs"], help="训练轮数（默认 0=不跑）")
    ap.add_argument("--steps", type=int, default=d["steps"],
                    help="按步数训练（默认 0）；与 --epochs 同时指定时 epoch 优先")
    ap.add_argument("--learning_rate", type=float, default=d["learning_rate"])
    ap.add_argument("--min_lr", type=float, default=d["min_lr"], help="余弦衰减的最低 lr")
    ap.add_argument("--warmup_steps", type=int, default=d["warmup_steps"], help="线性预热步数")
    ap.add_argument("--total_steps", type=int, default=d["total_steps"],
                    help="余弦调度的总步数（0=按数据集行数 // batch_size 自动）")
    ap.add_argument("--accumulation_steps", type=int, default=d["accumulation_steps"])
    ap.add_argument("--grad_clip", type=float, default=d["grad_clip"])
    ap.add_argument("--num_workers", type=int, default=d["num_workers"])
    ap.add_argument("--log_interval", type=int, default=d["log_interval"])
    ap.add_argument("--save_interval", type=int, default=d["save_interval"],
                    help="每多少步存一次 checkpoint（0=不存）")
    ap.add_argument("--keep_last_n", type=int, default=d["keep_last_n"],
                    help="只保留最近 N 个 checkpoint（0=全留）")
    ap.add_argument("--save_dir", default=d["save_dir"])
    ap.add_argument("--log_file", default=d["log_file"],
                    help="训练日志落盘路径（Python 内直写，不经 shell 管道；None=只打印控制台）")
    ap.add_argument("--compile", action="store_true",
                    help="用 torch.compile 加速训练（保持 inductor 默认数值语义）")
    ap.add_argument("--compile_mode", type=str, default=d["compile_mode"],
                    help='torch.compile 模式（固定 "default"）')
    ap.add_argument("--device", default=d["device"])
    ap.add_argument("--seed", type=int, default=d["seed"])
    ap.add_argument("--resume", default=d["resume"])


def apply_defaults(args, **overrides):
    """把各 trainer 的默认值回填到未显式给出的通用参数上。

    输入：已 parse 的 args；overrides（如 learning_rate=2.6e-4, accumulation_steps=8）。
    输出：无（原地修改）。
    预期行为：命令行**显式给出**的参数保持不变；未给出的用 overrides 里的值，
              overrides 未覆盖的用 _COMMON_DEFAULTS 的值。
    """
    import sys

    argv = sys.argv[1:]
    for key, value in {**_COMMON_DEFAULTS, **overrides}.items():
        flag = "--" + key
        given = any(a == flag or a.startswith(flag + "=") for a in argv)
        if not given:
            setattr(args, key, value)
    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_run_mode(ap, args):
    """校验并解析训练模式（epochs / steps 互斥）。

    输入：解析器（用于报错）；args。
    输出：(epochs_to_run, max_steps)。
    预期行为：两者都未给 → ap.error 直接退出（防止误触发整轮训练）；
              两者同给 → epoch 优先并打印提醒；只给 steps → 1 轮跑 max_steps 步。
    """
    if not args.steps and not args.epochs:
        ap.error("必须显式指定 --epochs 或 --steps（默认都为 0，不跑）")
    if args.steps and args.epochs:
        print(f"提醒：--steps={args.steps} 与 --epochs={args.epochs} 同时指定，"
              f"按 epoch 优先（跑 {args.epochs} 轮，忽略 --steps）", flush=True)
        return args.epochs, None
    if args.steps:
        return 1, args.steps
    return args.epochs, None
