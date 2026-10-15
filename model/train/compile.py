"""torch.compile 封装（训练用）。

训练用 compile 提速；**只设 Windows 必需的编译开关与跨进程缓存**，不调任何会改变
kernel 质量/数值的 inductor 配置（不拿 kernel 质量换编译时间）。

推理 / 对拍路径**不**走 compile（保持与参考实现逐位一致）。
"""

from __future__ import annotations

import os
import sys

# **必须在 import torch 之前**设这些 env：`torch._inductor.config` 在首次 import 时用
# `os.getenv(..., 默认)` 读取它们，之后再设就无效（setdefault 静默失效）。放在模块顶部
# 保证任何 import 本模块的路径都生效。
# compile_threads=1：**仅 Windows 需要**——多线程编译会走 SubprocPool 的 pass_fds →
# CPython 断言崩溃。Linux（含 Kaggle）多线程编译更快，放开（默认 = CPU 核数）。
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
# 编译缓存目录：由环境变量 TORCHINDUCTOR_CACHE_DIR 指定（不在此硬编码）。
# 云端环境（如 Kaggle）把它指向**会话内可写且跨进程可见**的目录，即可让多个训练
# 进程复用同一份编译产物；要跨会话复用，需把该目录持久化（存成云端数据集再挂载）。
# 本机不设时用 torch 默认缓存目录，行为不变。

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
    # 每层 Block 的整数/布尔属性（compress_ratio、is_kv_source、is_index_source …）各不相同，
    # 8 层共享同一 forward code object 但按属性值特化 → 触发多次重编译；默认 recompile_limit=8
    # 会撞上限并回退 eager（稳态变慢）。提高上限让各特化都真编译（一次性成本），稳态走编译图。
    torch._dynamo.config.recompile_limit = 64
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
        # 文本塔形状固定（B, L 固定）→ 静态编译。
        for blk in model.text.blocks:
            blk.forward = torch.compile(blk.forward, mode=mode)
        # 图像塔 patch 数随分辨率变化（序列长不定）→ **动态形状**编译：一张图吃所有
        # patch 数，避免每遇新形状重编译（重编译比不编译还慢）。
        for blk in model.image.vit.blocks:
            blk.forward = torch.compile(blk.forward, mode=mode, dynamic=True)
        return model
    # 逐层 compile：每层独立成图；只替换 forward（不包成 OptimizedModule），
    # 以免 state_dict 键被加 `_orig_mod.` 前缀。Block.forward 是同一个 code object，
    # dynamo 按 code object 缓存，结构相同的层自动复用同一张编译产物。
    for layer in model.layers:
        layer.forward = torch.compile(layer.forward, mode=mode)
        # Engram 挂在 Block 上，但**在 Model.forward 里被调用**（不在 Block.forward 内），
        # 故编 Block.forward 覆盖不到它 → 单独编译（否则每层 Engram 全走 eager）。
        if getattr(layer, "engram", None) is not None:
            layer.engram.forward = torch.compile(layer.engram.forward, mode=mode)
    # 视觉塔（ViT 的 N 个 block）与 Aligner 单独 compile：它们形状规则可编，
    # 但 patch 数随图像分辨率变 → dynamic=True，避免每遇新形状重编译。
    # （主干 Model.forward 的 merge_image_embeddings 是按图像个数的 Python 循环，
    #   不在此编译——模块级逐块编译已覆盖其内部全部可编算子。）
    vision = getattr(model, "vision", None)
    if vision is not None:
        for blk in vision.blocks:
            blk.forward = torch.compile(blk.forward, mode=mode, dynamic=True)
    aligner = getattr(model, "aligner", None)
    if aligner is not None:
        aligner.forward = torch.compile(aligner.forward, mode=mode, dynamic=True)
    # DSpark drafter 的 block（`mtp` 是 nn.ModuleList of DSparkBlock，与主干 Block 同级）。
    # 训练走 forward_train（teacher-forced 并行路径），故编它而不是 forward。
    # 阶段 3 不调用 dspark，此处只是预先编好供阶段 4 用；编 Block 本身形状规则、无循环。
    dspark = getattr(model, "dspark", None)
    if dspark is not None:
        for blk in dspark.mtp:
            blk.forward_train = torch.compile(blk.forward_train, mode=mode)
    # EngramHash：每步一次的整数 n-gram 哈希（gather/where/stack/xor/%/cat），
    # 是一串小算子 → 编译融合成少量 kernel。形状 [B,L] 固定，静态编译。
    engram_hash = getattr(model, "engram_hash", None)
    if engram_hash is not None:
        engram_hash.forward = torch.compile(engram_hash.forward, mode=mode)
    # 主干顶层叶子模块：embed / norm / head。Model.forward 本身含**变长图像循环**
    # （每样本图像数不定）故不整体编译；但这三个叶子形状规则、无循环，且 head 是
    # 全序列 fp32 GEMM（[B,L,V]，V=6400）——逐模块编译可把其前后的 cast 一并交给
    # inductor 融合（profiler 里 eager 的 aten::to/copy_/mul 主要来自这些未编区域）。
    for name in ("embed", "norm", "head"):
        leaf = getattr(model, name, None)
        if leaf is not None:
            leaf.forward = torch.compile(leaf.forward, mode=mode)
    # 视觉塔的 patch_embed（线性投影）与末端 norm：blocks 已逐块编译，这两处仍是
    # eager。patch 数随分辨率变 → dynamic=True，避免每遇新网格重编译。
    vision = getattr(model, "vision", None)
    if vision is not None:
        vision.patch_embed.forward = torch.compile(vision.patch_embed.forward, mode=mode, dynamic=True)
        vision.norm.forward = torch.compile(vision.norm.forward, mode=mode, dynamic=True)
    return model
