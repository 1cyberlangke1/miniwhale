"""对照实验配置：按阶段逐机制打开。

对照链：dense → +MoE → +mHC → +CSA2 → +Engram
唯一变量是架构；分词器、语料、token 预算保持一致。

用法：
    cfg = make_ablation("dense")   # 或 "+MoE" / "+mHC" / "+CSA2" / "+Engram" / "full"
"""

from __future__ import annotations

from model.config import ModelConfig

# 阶段 → 打开的机制集合（累积）
_STAGES = ["dense", "+MoE", "+mHC", "+CSA2", "+Engram"]


def make_ablation(stage: str) -> ModelConfig:
    """按对照阶段构造配置。

    输入：stage，见 _STAGES（或 "full"）。
    输出：ModelConfig。
    预期行为：累积打开机制；未打开的机制退化为 dense 等价形式。
    """
    cfg = ModelConfig.from_json()
    tc = cfg.text
    # 基线：纯 dense（无 MoE / mHC / CSA2 / Engram）
    tc.n_routed_experts = 0
    tc.n_shared_experts = 0
    tc.num_experts_per_tok = 0
    tc.hc_mult = 1
    tc.compress_ratios = tuple(0 for _ in tc.compress_ratios)
    tc.kv_source_layer_ids = ()
    tc.index_source_layer_ids = ()
    tc.engram_layer_ids = ()

    if stage == "dense":
        return cfg
    idx = _STAGES.index(stage) if stage in _STAGES else len(_STAGES)
    # +MoE
    if idx >= 1:
        tc.n_routed_experts = 4
        tc.n_shared_experts = 1
        tc.num_experts_per_tok = 2
    # +mHC
    if idx >= 2:
        tc.hc_mult = 4
    # +CSA2
    if idx >= 3:
        tc.compress_ratios = (0, 0, 2, 2, 1, 1, 1, 1)
        tc.kv_source_layer_ids = (2, 4)
        tc.index_source_layer_ids = (2, 4, 6)
    # +Engram
    if idx >= 4:
        tc.engram_layer_ids = (1, 6)
        # 两层行数不同（素数集合随层递增），取 config.json 的真实值
        tc.engram_num_embeddings = (235730, 241278)
    return cfg
