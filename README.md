# miniwhale

在 **8GB 单卡**上从零训练 **DeepSeek-V4.1-Flash 架构**（机制不变，尺寸缩水）。

目标不是复现能力，而是**验证机制在不在、有没有用**：
CSA2、层级稀疏 indexer、Single-Pass mHC、Engram、V4.1 式 MoE、DSpark、ViT。

## 快速开始

```bash
# 环境（Python 3.11）
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt

# 数据（放到 dataset/）
#   pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl（文本）
#   sft_i2t.parquet（多模态，含 pretrain 子集）

# 测试（需指向参考实现实现目录）
MINIWHALE_ORACLE=<path>/tmp .venv/Scripts/python -m pytest tests/

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
miniwhale/          源码包
  config.py         配置（字段名与参考实现 V4.1 config 一致）
  tokenizer.py      6400 词表（随包分发）
  linear.py         全局 dtype 机制（set_dtype）
  norm.py  rope.py
  engram_hash.py    n-gram 哈希（素数桶）
  engram.py         条件记忆（门控 + 表）
  moe.py            V4.1 式 MoE（sqrtsoftplus + noaux_tc）
  mhc.py            Single-Pass mHC（Sinkhorn）
  compressor.py     KV 池化
  indexer.py        两级 Top-K
  attention.py      MLA + CSA2 + 跨层共享
  quant.py          QAT 假量化（fp8/fp4）
  vision.py         ViT + Aligner
  dspark.py         draft head（Markov + confidence）
  model.py          组装
  optim.py          优化器分组（Muon/AdamW/Sinkhorn）
  data.py           数据加载
  ablation.py       对照实验开关
tests/              每模块 vs 参考实现 oracle 对拍
trainer/            训练脚本
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

