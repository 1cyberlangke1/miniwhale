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
    os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE", "1")
    import torch._inductor.config as _ind
    _ind.emulate_precision_casts = True
    # 小 batch 下模型是 launch-bound（实测 2.8 万 kernel/步，GPU 利用率 13%）。
    # inductor 默认把 reduction 拆成多个小 kernel，进一步放大 launch 开销。
    # 关掉拆分让 reduction 合成单 kernel（PyTorch issue #179697 的推荐修复），
    # 只改 kernel 划分、不改数值。
    _ind.split_reductions = False
    # CUDA Graph（mode="reduce-overhead"）只对**静态 shape** 的图有意义。
    # MoE 前向用 `torch.where(indices == i)` 取各专家命中的 token（参考实现写法），
    # 每步命中数不同 → 动态 shape → 强行录图会**每步录一张新图**（警告
    # "observed 9 distinct sizes"），永远录不完、卡死。torch 参考实现为此提供开关：
    # 让含动态 shape 的图跳过 CUDA Graph，静态算子（attention GEMM / norm / mHC）
    # 仍走 CUDA Graph 提速。**只改执行调度，不改任何算子数值。**
    if mode == "reduce-overhead":
        _ind.triton.cudagraph_skip_dynamic_graphs = True
    # 8 层 attention 结构不同（压缩比 / 是否 KV 源 / 是否 index 源 / 是否用候选块），
    # 但 Block.forward 是同一个 code object，dynamo 按 code object 缓存 —— **5 种结构
    # 只编译 5 次**（自动去重），比 isolate_recompiles（每层各编一次 = 8 次）更省编译时间。
    # 放宽单函数重编译上限，防止第 9 个变体回退 eager 后每步重试（掉速）。
    torch._dynamo.config.recompile_limit = 64
    torch._dynamo.config.accumulated_recompile_limit = 256
    # 逐层 compile：每层独立成图（含 CUDA Graph）；只替换 forward（不包成 OptimizedModule），
    # 以免 state_dict 键被加 `_orig_mod.` 前缀。
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, mode=mode)
    return model
