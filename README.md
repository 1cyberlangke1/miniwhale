# miniwhale

> **非参考实现个人项目**：在 8GB 单卡上从零训练 [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai) 的**模型架构**尝试。
> 机制不变、尺寸缩水，目标是验证「机制在不在、有没有用」，而不是复现模型能力。
> 本项目与 DeepSeek 无隶属关系，不代表参考实现实现。

## 致谢

感谢 DeepSeek 开源模型架构、推理实现与训练细节。没有这些公开材料，这个项目无法存在：

- DeepSeek 参考实现组织：[GitHub](https://github.com/deepseek-ai) / [HuggingFace](https://huggingface.co/deepseek-ai)
- [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai) 的推理实现、config 与技术报告 —— 本仓库的全部层公式、术语、字段名都照此对齐
- [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)（Apache-2.0）—— Engram 的 n-gram 哈希与素数分桶设计
- [deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec) —— DSpark 草稿头的训练 loss 与前向

**关于权重**：本仓库不上传 HuggingFace 权重。训练产物在本地 `out/` 下（`pretrain_e0.pth`、`sft.pth` 等），
尺寸与训练数据见下文；如需发布请自行上传并在此补充链接。

## 我实现了什么

### 1. 模型架构（逐机制复刻）

| 机制 | 实现位置 | 说明 |
|---|---|---|
| MLA 低秩 Q/O + YaRN RoPE | `model/layers/attention.py`、`model/nn/rope.py` | 与参考实现 `Attention` 逐字段一致；attn_sink 只进 softmax 分母 |
| Sliding Window + Compressor | `attention.py`、`compressor.py` | 滑窗原始 KV 与压缩位置拼接 |
| **CSA2**（跨层 KV 共享） | `attention.py` | `kv_source_layer_ids` 在 L/2，源层算一次压缩 KV，后续层**直接读同一份缓存**（无 W_KV 投影）；index 源层同理发布 top-k 索引 |
| 层级稀疏 Indexer | `indexer.py` | 两级 Top-K：候选块（块内最大分 + 最新块钉住）→ 块内精选 |
| Single-Pass mHC | `mhc.py`、`mhc_kernel.py` | hc_mult=4 残差流、20 轮 Sinkhorn 双随机归一、pre_mix 错位传递；融合 kernel 与 torch 实现逐位一致 |
| V4.1 MoE | `moe.py` | sqrtsoftplus 打分、noaux_tc 纠正 bias、1 个共享专家、SwiGLU clamp、稀疏 top-k（不张量化成全专家） |
| Engram | `engram.py`、`engram_hash.py` | 压缩词表 + 每 (层, n-gram 阶, head) 独立素数桶；n-gram 回看支持增量解码 |
| DSpark 草稿头 | `dspark.py` | block=4 并行草稿、markov 低秩 logits 偏置、confidence head、target 层 hidden 拼接 |
| ViT + Aligner | `vision.py`、`image_processor.py` | 图像 span 布局（start/newline/end/IMAGE）一致 |
| fp8/fp4 假量化 | `model/nn/quant.py` | 训练期 QAT：逐块 amax → pow2 scale → clamp → 反量化 |

**与参考实现对齐**：同一份权重、同一输入下，本仓库与 DeepSeek 发布的 V4.1-Flash 推理实现的
logits 最大绝对差 **0.0000**（带 Engram fp8 表 + DSpark target 重叠时为 5.6e-5，bf16 舍入量级）。
见 `tests/test_match_official.py`。

### 2. 训练

- **五阶段管线**：SigLIP 视觉塔 → 文本预训练 → ViT 自回归微调 → DSpark 二阶段（冻结 backbone）→ SFT
- **优化器**（照技术报告 §2.5 / Algorithm 1）：Muon（线性矩阵）+ AdamW（RMSNorm/非矩阵）+ Sinkhorn 均衡更新（Embedding/Head），Engram 学习率 ×5
- **8GB 显存工程**：序列长度 bucket（512/768/1024）、梯度累积、按 step 原子写 checkpoint + 自动续训、ETA 估计
- 训练统一 `--epochs`（不写死 `--steps`），`--steps` 仍保留用于限步调试

### 3. 推理（`inference/`）

- `engine.py`：权重加载（bf16）、prefill + 增量 decode、KV cache、Engram 历史回看
- `compile.py`：torch.compile + **CUDA Graph**。位置全程走张量、形状与控制流静态，按压缩组边界各捕获一张图，重放只改输入 buffer —— **换输入不会重编译**
- `chat.py`：命令行推理入口（`--cuda_graph` 即最快路径，隐含 `torch.compile`）
- `spec.py`：DSpark 投机解码（多 token 并行验证）

**实测（RTX 4060 Laptop 8GB）**：

| 配置 | 速度 |
|---|---|
| eager | 22.8 tok/s |
| torch.compile | 155 tok/s |
| CUDA Graph + compile | **1061 tok/s（decode 稳态 0.94 ms/token）** |
| prefill（编译后） | 12.4 ms（eager 约 43 ms） |

说明：2000 tok/s 需要约 0.5 ms/步，而当前步内约 350 个 kernel、GPU 侧已占 0.87 ms，
PyTorch 层面到顶；再快只能手工融合 MoE / attention 的自定义 kernel。

### 4. 验证

- `pytest tests/`：**101 passed**（含逐位对拍、组件数值、梯度覆盖、dtype 契约、无死参数等）
- 架构对照实验（同分词器、同语料、同 token 预算，唯一变量是架构）：

## 尺寸（最小档）

| | DeepSeek-V4.1-Flash | miniwhale |
|---|---|---|
| 层数 / hidden | 40 / 5120 | **8 / 512** |
| 词表 | 129280 | **6400** |
| 路由专家 / 激活 | 384 / 6 | **4 / 2** |
| hc_mult | 4 | **4** |
| compress_ratios | `[0,0,2×18,1×20]` | `[0,0,2,2,1,1,1,1]` |
| Engram | 2 层 / 196B | **2 层 / 19.0M** |

参数量：骨干 26.3M + Engram 19.0M = **45.3M**；激活/token 20.7M（prefill）。

## 快速开始

```bash
# 环境（Python 3.11）
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python -m pip install -r requirements.txt

# 数据（放到 dataset/）
#   pretrain_t2t_mini.jsonl / sft_t2t_mini.jsonl（文本）
#   sft_i2t_mini.parquet（纯图文，供 ViT 自回归微调）
#   从完整版 sft_i2t.parquet 抽纯图文子集：python scripts/extract_i2t_mini.py

# 训练
.venv/Scripts/python trainer/train_siglip.py
.venv/Scripts/python trainer/train_pretrain.py --data_path dataset/pretrain_t2t_mini.jsonl
.venv/Scripts/python trainer/train_vlm.py      --data_path dataset/sft_i2t_mini.parquet
.venv/Scripts/python trainer/train_dspark.py   --ckpt out/pEngram.pth
.venv/Scripts/python trainer/train_sft.py

# 对照实验（5 阶段，唯一变量是架构）
.venv/Scripts/python scripts/run_overnight.py --steps 1500 --seq_len 256 --batch 4

# 推理
.venv/Scripts/python -m inference.chat --ckpt out/sft.pth --prompt "你好" --cuda_graph

# 测试
.venv/Scripts/python -m pytest tests/ -q
```

## 目录结构

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
  train/            optim / data / dspark_loss / ablation / loop / cli / checkpoint
trainer/            训练脚本（pretrain / sft / vlm / dspark）
inference/          推理（engine / compile / chat / spec）
scripts/            工具脚本（数据抽取、对照实验、评测）
tests/              101 个测试（含参考实现逐位对拍）
```

## 优化器（技术报告 §2.5）

| 参数组 | 优化器 |
|---|---|
| 线性矩阵 | Muon（`torch.optim.Muon`） |
| RMSNorm / 非矩阵 | AdamW |
| Embedding + Head | Sinkhorn-balanced update |
| Engram | lr × 5 |

## 实验结果（8GB 实测）

**对照实验保留集困惑度**（1500 步 / seq 256）：

| 阶段 | 参数量 | ppl |
|---|---|---|
| dense | 14.0M | 286 |
| **+MoE** | 26.6M | **245** |
| **+mHC** | 27.3M | **245** |
| +CSA2 | 27.6M | 260 |

**Engram 容量 U 型曲线**（复现论文 §3.1）：最优 Engram 占非激活参数 **30%**（11.7M / 39.3M）。

**机制验证 6 项全过**：跨层共享（无投影）、CSA2 三模式、因果性、mHC 双随机、

## 许可

代码以 [MIT](LICENSE) 协议开源。引用到的 DeepSeek 开源仓库（Engram 为 Apache-2.0 等）
其版权与许可归 DeepSeek 所有；本项目对其机制的实现为独立重写，并在上文注明出处。
