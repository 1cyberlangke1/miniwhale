"""miniwhale 配置。

字段名与参考实现 config 保持一致，只改数值。
读取同目录的 config.json（缩水后的最小档）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_CONFIG_PATH = Path(__file__).with_name("config.json")


@dataclass
class TextConfig:
    """语言骨干配置。

    输入：无（由 config.json 的 text_config 段填充）。
    输出：一个持有全部结构超参的 dataclass，供各模块读取。
    预期行为：字段语义一致；数值为本项目的缩水档。
    """

    vocab_size: int = 6400
    hidden_size: int = 512
    moe_intermediate_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 64
    qk_rope_head_dim: int = 16
    q_lora_rank: int = 128
    o_lora_rank: int = 128
    o_groups: int = 2
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-20
    initializer_range: float = 0.02
    max_position_embeddings: int = 32768
    # KV/index 缓存的 batch 上限：__init__ 里按此预分配 buffer，避免 forward 中
    # `_ensure_cache` 重建 buffer（改 _buffers）→ dynamo guard 失败 → 反复重编译。
    # 训练 batch 固定，一开始就够大 → 全程不再重建。
    max_batch_size: int = 16
    rope_theta: float = 10000.0
    # HF 标准字段（参考实现 HF config 同款）
    model_type: str = "miniwhale_text"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    rope_scaling: object = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    # YaRN 外推：original_seq_len>0 时对压缩层（用 compress_rope_theta）启用
    original_seq_len: int = 0
    rope_factor: float = 40.0
    beta_fast: int = 32
    beta_slow: int = 1

    # MoE：V4.1 式（sqrtsoftplus 打分 + noaux_tc bias + 共享专家）
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    gate_temp: float = 1.0
    # 序列级平衡损失权重（noaux_tc 的兜底项，防单序列内极端偏斜）
    seq_aux_loss_alpha: float = 1e-4
    # 多模态：图像 span token 用单独的路由 bias（参考实现 noaux_tc_for_vl）
    vision_enabled: bool = True
    # 图像占位 token 的 id（参考实现 image_token_id）；minimind 词表里 <|image_pad|> = 12
    image_token_id: int = 12

    # CSA2
    sliding_window: int = 128
    compress_ratios: tuple = (0, 0, 2, 2, 1, 1, 1, 1)
    compress_rope_theta: float = 160000.0
    kv_source_layer_ids: tuple = (2,)
    index_source_layer_ids: tuple = (2, 4)
    index_n_heads: int = 8
    index_head_dim: int = 64
    index_topk: int = 32
    candidate_source_layer_id: int = 4
    candidate_topk_blocks: int = 64
    candidate_block_size: int = 8

    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # Engram
    engram_layer_ids: tuple = (1, 6)
    engram_num_embeddings: tuple = (235730, 235730)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 9728
    engram_n_heads: int = 8
    engram_head_dim: int = 32
    engram_pad_token_id: int = 2
    engram_compressed_vocab_size: Optional[int] = None

    # DSpark
    num_nextn_predict_layers: int = 2
    # draft 位数：满血档 = 5（报告 §2.4.3「five draft positions」；参考实现部署配置
    # num_speculative_tokens=5）。miniwhale 缩水为 4。
    dspark_block_size: int = 4
    dspark_noise_token_id: int = 6199
    dspark_target_layer_ids: tuple = (6, 7)
    dspark_markov_rank: int = 64
    dspark_n_routed_experts: int = 4
    dspark_num_experts_per_tok: int = 2
    # 参考实现训练超参（DeepSpec config/dspark/dspark_qwen3_4b.py）
    dspark_num_anchors: int = 512
    dspark_loss_decay_gamma: float = 4.0
    dspark_ce_loss_alpha: float = 0.1
    dspark_l1_loss_alpha: float = 0.9
    dspark_confidence_head_alpha: float = 1.0


@dataclass
class VisionConfig:
    """视觉编码器配置（按参考实现 vision_config 缩水）。"""

    model_type: str = "miniwhale_vision"
    num_hidden_layers: int = 4
    hidden_size: int = 128
    num_attention_heads: int = 4
    intermediate_size: int = 352
    patch_size: int = 14
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    max_image_tokens: int = 256
    # 参考实现 image_processor 的 resize 约束（num_hidden_layers==0 关闭视觉）
    min_pixels: int = 0
    max_wh_ratio: float | None = None


@dataclass
class ModelConfig:
    """顶层配置。

    输入：可选 path（config.json 路径），默认同目录。
    输出：持有 text / vision 两个子配置的 dataclass。
    预期行为：tuple 字段从 JSON 的 list 还原；缺字段用默认值。
    """

    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def from_json(cls, path: Path | str = _CONFIG_PATH) -> "ModelConfig":
        """从 JSON 加载，list → tuple。

        顶层字段 image_token_id（参考实现放在顶层但属 text 语义）注入 text_config；
        vision_n_layers 决定 vision_enabled（参考实现 vision_enabled = vision_n_layers > 0）。
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        text_raw = dict(raw.get("text_config", {}))
        if "image_token_id" in raw:
            text_raw["image_token_id"] = raw["image_token_id"]
        vision_raw = raw.get("vision_config", {})
        # 参考实现：vision_enabled 由视觉层数决定
        text_raw["vision_enabled"] = vision_raw.get("num_hidden_layers", 0) > 0
        text = TextConfig(**{
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in text_raw.items()
            if k in TextConfig.__dataclass_fields__
        })
        vision = VisionConfig(**{
            k: v for k, v in vision_raw.items()
            if k in VisionConfig.__dataclass_fields__
        })
        return cls(text=text, vision=vision)


if __name__ == "__main__":
    cfg = ModelConfig.from_json()
    print(cfg.text)
    print(cfg.vision)
