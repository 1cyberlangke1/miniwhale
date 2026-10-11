"""torch.compile 封装（训练用）。

训练用 compile 提速；开启 `emulate_precision_casts` 让 inductor 复现 eager 的低精度语义
（eager 对 bf16/fp16 会「上转 fp32 算完再下转」，inductor 默认省略中间下转、结果更接近
fp64，与 eager 数值不同）。开启后逐算子与 eager 位一致，训练 loss 与 eager 相对差 <1e-4。

推理 / 对拍路径**不**走 compile（保持与参考实现逐位一致）。
"""

from __future__ import annotations

import os

import torch
from torch import nn


def maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    """按需把模型包进 torch.compile。

    输入：model；enabled（True 则编译）。
    输出：编译后的模型（enabled=False 时原样返回）。
    预期行为：开启前先设 `torch._inductor.config.emulate_precision_casts = True`，
              使 inductor 与 eager 的低精度语义一致（参考实现 config 明确此选项用于此目的）。
    """
    if not enabled:
        return model
    # Windows 上 inductor 的编译线程数被硬编码为 1（torch/_inductor/config.py
    # `compile_threads = 1 for win32`）——单核编译，冷启动慢。环境变量
    # TORCHINDUCTOR_COMPILE_THREADS 在该 win32 判定**之前**读取，可覆盖为多线程。
    # 纯并行编译，不改数值、不改 kernel 质量。
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(os.cpu_count() or 8))
    import torch._inductor.config as _ind
    _ind.emulate_precision_casts = True
    # 区域编译：逐层编译 Block，而不是整模型。
    # - 整模型 compile 把 N 层内联成一张巨图，异构层反复重编译、冷启动爆炸。
    # - 逐层 compile 让每层独立成图；`isolate_recompiles=True`（参考实现 API）让每层的
    #   重编译预算独立，8 层异构结构互不挤上限。
    # 实测（batch4/seq256）：首步 23s，稳态 0.13s/步（约 eager 的 2 倍快）。
    # 只替换 forward（不包成 OptimizedModule），以免 state_dict 键被加 `_orig_mod.` 前缀。
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, isolate_recompiles=True)
    return model
