"""torch.compile 封装（训练用）。

训练用 compile 提速；**只设 Windows 必需的编译开关与跨进程缓存**，不调任何会改变
kernel 质量/数值的 inductor 配置（不拿 kernel 质量换编译时间）。

推理 / 对拍路径**不**走 compile（保持与参考实现逐位一致）。
"""

from __future__ import annotations

import os

# **必须在 import torch 之前**设这些 env：`torch._inductor.config` 在首次 import 时用
# `os.getenv(..., 默认)` 读取它们，之后再设就无效（setdefault 静默失效）。放在模块顶部
# 保证任何 import 本模块的路径都生效。
# compile_threads=1：Windows 上多线程编译会走 SubprocPool 的 pass_fds → CPython 断言崩溃。
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
from torch import nn


def maybe_compile(model: nn.Module, enabled: bool, mode: str = "default") -> nn.Module:
    """按需把模型包进 torch.compile。

    输入：model；enabled（True 则编译）；mode（torch.compile 模式，如 "default"）。
    输出：编译后的模型（enabled=False 时原样返回）。
    预期行为：开启跨进程编译缓存后逐层 compile。保持 inductor 默认数值语义——
              模型关键处的精度由代码里**显式** `.float()` 控制（MoE/attention/mHC），
              与编译开关无关。
    """
    if not enabled:
        return model
    # 跨进程编译缓存：FX 图编译产物 + autotune 结果落盘（%TEMP%/torchinductor_*/），
    # 第二次启动直接命中，参考实现称 warm 编译可省 90%+ 时间（冷启动才是几分钟的元凶）。
    # 缓存 key 含输入 shape —— 训练 batch/seq 固定即可稳定命中。
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    # AOTAutograd 层缓存（参考实现 Meta 实测占编译 13.5%）：缓存 autograd 图，
    # 跨进程复用。参考实现文档：`TORCHINDUCTOR_AUTOGRAD_CACHE` 需 `FX_GRAPH_CACHE` 才生效。
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    # SigLIP 对比阶段（双塔，非主干 Model）：文本塔逐层 + 图像塔的 ViT 逐块 compile。
    # 图像塔逐图调用（patch 数随分辨率变化），故按 block 编译而非整个 ViT。
    if not hasattr(model, "layers") and hasattr(model, "text") and hasattr(model, "image"):
        for blk in model.text.blocks:
            blk.forward = torch.compile(blk.forward, mode=mode)
        for blk in model.image.vit.blocks:
            blk.forward = torch.compile(blk.forward, mode=mode)
        return model
    # 逐层 compile：每层独立成图；只替换 forward（不包成 OptimizedModule），
    # 以免 state_dict 键被加 `_orig_mod.` 前缀。Block.forward 是同一个 code object，
    # dynamo 按 code object 缓存，结构相同的层自动复用同一张编译产物。
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, mode=mode)
    return model
