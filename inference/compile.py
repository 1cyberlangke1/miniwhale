"""推理侧 torch.compile 封装。

与训练侧（`model/train/compile.py`）分开：训练编 `Block.forward` / `forward_train`，
推理只编**单步解码**（形状固定 [1,1]，最适合编译）。

数值语义保持不变：只编整条 forward，不引入新的归约顺序。
"""

from __future__ import annotations

import os
import sys

import torch


def _set_windows_threads():
    """Windows 下 inductor 多线程编译会崩（SubprocPool pass_fds）。**必须在 import torch 前设**。"""
    if sys.platform == "win32":
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")


def compile_for_decode(model, mode: str = "max-autotune-no-cudagraphs"):
    """编译单步解码。

    输入：model（已 eval 的 Model）；mode（torch.compile 模式）。
    输出：无（原地替换 `forward_decode`）。
    预期行为：整条解码 forward（embed → 8 层 → norm → head）编成**一张图**，
              CPU 侧只付一次图调用开销；层内 200+ 个 kernel 由 inductor 生成的
              C++ launcher 连续发出，避免 eager 下逐算子 Python dispatch。
              **只编 `forward_decode`**：prefill 走 eager 的 `forward`。两者若共用同一
              编译入口，prefill 的动态长度会先建一张符号形状图，解码再用时形状对不上
              （实测 fake tensor 推断直接报错，或每步重编译掉到 0.9 tok/s）。
              decode 图要求全程静态：`past_ids` 只取最后 max_ngram-1 个 token，
              压缩/索引缓存一律读**整个定长缓存**。
    """
    _set_windows_threads()
    torch._dynamo.config.recompile_limit = 256
    # 整数参数（start_pos）不按值特化，否则每步一张新图
    torch._dynamo.config.specialize_int = False
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")

    model.forward_decode = torch.compile(model.forward_decode, mode=mode)
    # prefill 也编译：eager 一次 ~1100 个 kernel（实测 ~43ms，占 128-token 生成的四分之一）。
    # `dynamic=True` 让不同 prompt 长度共用一张符号形状图，不会每个长度重编译。
    model.forward = torch.compile(model.forward, mode=mode, dynamic=True)
    return model


def warmup_decode(model, device: str = "cuda", prompt_len: int = 8, steps: int = 8,
                  ids=None):
    """在启动时预热 decode 形状，把编译代价挪出生成过程。

    输入：model（已 compile）；device；prompt_len（预热用 prompt 长度）；
          steps；ids（可选，**直接用这段真实 token** 预热，保证形状与后续一致）。
    输出：无。
    预期行为：prefill 走 eager（不触发编译），随后跑若干 decode 让 dynamo 编好
              `[1,1]` 的静态图（含 `full_group` 两种取值的分支）。
              用假数据不影响权重（只前向、no_grad）；**必须带 autocast**——
              MoE 的 `_grouped_mm` 只接受 bf16，fp32 输入会直接报错。
    """
    vocab = model.cfg.text.vocab_size
    ac = "cuda" if device.startswith("cuda") else "cpu"
    ratios = [layer.attn.compress_ratio for layer in model.layers]
    if ids is None:
        torch.manual_seed(0)
        ids = torch.randint(0, vocab, (1, prompt_len), device=device)
    with torch.no_grad(), torch.autocast(device_type=ac, dtype=torch.bfloat16):
        model.reset_cache()
        logits, _ = model(ids, start_pos=0, full_logits=False)
        nxt = logits.argmax(dim=-1, keepdim=True)
        cur = torch.cat([ids, nxt], dim=1)
        for _ in range(steps):
            posv = cur.size(1) - 1
            pos = torch.tensor(posv, device=device, dtype=torch.long)
            emit = tuple(tuple(j for j in range(1)
                               if cr > 0 and (posv + j + 1) % cr == 0) for cr in ratios)
            model.forward_decode(nxt, pos, cur[:, :-1], emit_ats=emit)
            nxt = torch.zeros_like(nxt)
            cur = torch.cat([cur, nxt], dim=1)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    model.reset_cache()
