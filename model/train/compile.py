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


def maybe_compile(model: nn.Module, enabled: bool, mode: str = "default") -> nn.Module:
    """按需把模型包进 torch.compile。

    输入：model；enabled（True 则编译）；mode（torch.compile 模式，如 "default" /
          "reduce-overhead"（CUDA Graphs，压掉小 kernel 启动开销））。
    输出：编译后的模型（enabled=False 时原样返回）。
    预期行为：开启前先设 `torch._inductor.config.emulate_precision_casts = True`，
              使 inductor 与 eager 的低精度语义一致（参考实现 config 明确此选项用于此目的）。
    """
    if not enabled:
        return model
    # Windows 上 inductor 默认单线程编译（torch/_inductor/config.py `compile_threads = 1
    # for win32`）。曾尝试用 TORCHINDUCTOR_COMPILE_THREADS 覆盖为多线程，但 Windows 上
    # 多线程编译会走 inductor 的 SubprocPool，其 subprocess 调用带 pass_fds 参数，而
    # CPython 在 Windows 上直接断言 `pass_fds not supported on Windows` → 编译崩溃。
    # 故必须保持单线程（= 1），靠下面的跨进程 FX 图缓存把二次启动变暖。
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    # 跨进程编译缓存：FX 图编译产物 + autotune 结果落盘（%TEMP%/torchinductor_*/），
    # 第二次启动直接命中，参考实现称 warm 编译可省 90%+ 时间（冷启动才是几分钟的元凶）。
    # 缓存 key 含输入 shape —— 训练 batch/seq 固定即可稳定命中。
    os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")
    # AOTAutograd 层缓存（参考实现 Meta 实测占编译 13.5%）：缓存 autograd 图，
    # 跨进程复用。参考实现文档：`TORCHINDUCTOR_AUTOGRAD_CACHE` 需 `FX_GRAPH_CACHE` 才生效。
    os.environ.setdefault("TORCHINDUCTOR_AUTOGRAD_CACHE", "1")
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
    import torch._inductor.config as _ind
    # 让 inductor 复现 eager 的低精度语义（bf16/fp16 上转 fp32 算完再下转），
    # 否则 inductor 省略中间下转、数值更接近 fp64，与参考实现逐位对拍不一致。仅此一项。
    _ind.emulate_precision_casts = True
    # CUDA Graph（mode="reduce-overhead"）只对**静态 shape** 的图有意义。
    # MoE 前向用 `torch.where(indices == i)` 取各专家命中的 token（参考实现写法），
    # 每步命中数不同 → 动态 shape → 强行录图会**每步录一张新图**（警告
    # "observed 9 distinct sizes"），永远录不完、卡死。torch 参考实现为此提供开关：
    # 让含动态 shape 的图跳过 CUDA Graph，静态算子（attention GEMM / norm / mHC）
    # 仍走 CUDA Graph 提速。**只改执行调度，不改任何算子数值。**
    if mode == "reduce-overhead":
        _ind.triton.cudagraph_skip_dynamic_graphs = True
    # 逐层 compile：每层独立成图；只替换 forward（不包成 OptimizedModule），
    # 以免 state_dict 键被加 `_orig_mod.` 前缀。Block.forward 是同一个 code object，
    # dynamo 按 code object 缓存，结构相同的层自动复用同一张编译产物。
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, mode=mode)
    return model
