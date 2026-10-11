# miniwhale

在 **8GB 单卡**上从零训练 **DeepSeek-V4.1-Flash 架构**（机制不变，尺寸缩水）。

目标不是复现能力，而是**验证机制在不在、有没有用**：
CSA2、层级稀疏 indexer、Single-Pass mHC、Engram、V4.1 式 MoE、DSpark、ViT。

**与参考实现对齐**：各层公式逐行对按参考实现 发布的 V4.1-Flash 推理实现（低层算子如
mHC Sinkhorn、MoE 门控、fp8/fp4 量化、RoPE/YaRN、indexer 两级 Top-K 等按其 kernel
的纯 torch 公式复刻）。同一份权重、同一输入下，本仓库与参考实现 Transformer 的 logits
最大绝对差 **0.0000**；带 Engram(fp8 表)+DSpark target 重叠时为 5.6e-5（bf16 舍入量级）。
训练侧优化器按参考实现技术报告 §2.5 / Algorithm 1（Muon + AdamW + Sinkhorn 均衡更新）。

## 快速开始

```bash
# 环境（Python 3.11）
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt

# 数据（放到 dataset/）
#   pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl（文本）
#   sft_i2t.parquet（多模态，含 pretrain 子集）

# 预训练
.venv/Scripts/python trainer/train_pretrain.py --data_path dataset/pretrain_t2t_mini.jsonl

# 对照实验（5 阶段，唯一变量是架构）
.venv/Scripts/python scripts/run_overnight.py --steps 1500 --seq_len 256 --batch 4

# 困惑度评测
.venv/Scripts/python scripts/eval_ppl.py --ckpt out/dense.pth --stage dense

# DSpark 二阶段（冻结 backbone）
.venv/Scripts/python trainer/train_dspark.py --ckpt out/pEngram.pth

# 多模态
.venv/Scripts/python trainer/train_vlm.py --data_path dataset/sft_i2t.parquet
```

## 结构

```
model/              源码包
  config.py         配置（字段名与参考实现 HF config 一致）
  config.json       最小档配置
  tokenizer.py      6400 词表（随包分发）
  image_processor.py 图像 span 布局（按参考实现）
  model.py          组装（含视觉塔 + DSpark）
  nn/               linear / norm / rope（YaRN）/ quant（fp8/fp4）
  layers/           attention / compressor / indexer / mhc / moe /
                    engram / engram_hash / vision / dspark
  train/            optim / data / dspark_loss / ablation
trainer/            训练脚本（pretrain / sft / vlm / dspark）
scripts/            工具脚本
```

## 尺寸（最小档）

| | V4.1 | miniwhale |
|---|---|---|
| 层数 / hidden | 40 / 5120 | **8 / 512** |
| 词表 | 129280 | **6400** |
| 路由专家 / 激活 | 384 / 6 | **4 / 2** |
| hc_mult | 4 | **4** |
| compress_ratios | `[0,0,2×18,1×20]` | **`[0,0,2,2,1,1,1,1]`** |
| Engram | 2 层 / 196B | **2 层 / 19.0M** |

**参数量**：骨干 26.3M + Engram 19.0M = **45.3M**；激活/token 20.7M（prefill）。

## 优化器（技术报告 §2.5）

| 参数组 | 优化器 |
|---|---|
| 线性矩阵 | Muon（`torch.optim.Muon`，内置） |
| RMSNorm / 非矩阵 | AdamW |
| Embedding + Head | Sinkhorn-balanced update |
| Engram | lr × 5 |

## 对照实验

同分词器、同语料、同 token 预算，**唯一变量是架构**：

```
dense (12.6M) → +MoE (25.2M) → +mHC (26.0M) → +CSA2 (26.3M) → +Engram (45.5M)
```

```python
from miniwhale.ablation import make_ablation
cfg = make_ablation("+CSA2")
```

指标：loss 曲线、长上下文衰减、KV 字节/token、prefill/decode 耗时比。

## 实验结果（8GB 实测）

**对照实验保留集困惑度**（1500 步 / seq 256）：

| 阶段 | 参数量 | ppl |
|---|---|---|
| dense | 14.0M | 286 |
| **+MoE** | 26.6M | **245** |
| **+mHC** | 27.3M | **245** |
| +CSA2 | 27.6M | 260 |

**Engram 容量 U 型曲线**（复现论文 §3.1）：最优 Engram 占非激活参数 **30%**（11.7M / 39.3M），
太小欠容量、太大欠拟合。

**机制验证 6 项全过**：跨层共享（无投影）、CSA2 三模式、因果性、mHC 双随机、
Engram 哈希分布、**0 个意外死参数**。

