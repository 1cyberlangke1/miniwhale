# miniwhale

> **由 DeepSeek-V4.1-Flash 自身编写，在 8GB 消费级显卡（RTX 4060 Laptop）上从零复现并验证 DeepSeek-V4.1-Flash 完整架构。**
>
> 核心原则：**机制严格对齐，规模等比缩减**。
> 核心目标是验证关键创新机制（MLA、CSA2、Single-Pass mHC、Engram、V4.1 MoE、DSpark 等）的设计正确性与工程有效性。
> 本项目代码由 DeepSeek-V4.1-Flash 编写，与 DeepSeek 官方无隶属关系。

## 核心机制

- **MLA + YaRN RoPE**：低秩 Q/O 投影与长上下文位置编码，对齐参考实现
- **CSA2**：跨层 KV 共享与层级稀疏 Indexer（无额外投影直接共享缓存）
- **Single-Pass mHC**：4 副本扩展残差流与 Sinkhorn 双随机归一化
- **V4.1 生产级 MoE**：Sqrt-Softplus 打分、`noaux_tc` 路由纠偏、共享专家与 SwiGLU 动态截断
- **Engram 条件记忆**：压缩词表空间与独立素数哈希桶
- **DSpark 投机草稿**：Block=4 并行预测与低秩 Markov 转移偏置
- **ViT**：图像 Span 布局与多模态联合对齐
- **FP8 / FP4 QAT**：训练期量化感知训练

## 快速开始

```bash
# 环境依赖（Python 3.11）
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# 训练流程（五阶段流水线）
python trainer/train_siglip.py   --epochs 1
python trainer/train_pretrain.py --epochs 1 --data_path <文本数据.jsonl>
python trainer/train_vlm.py      --epochs 1 --ckpt out/pretrain.pth --vit_ckpt out/vit_siglip.pth --data_path <图文数据.parquet>
python trainer/train_dspark.py   --epochs 1 --ckpt out/vlm.pth
python trainer/train_sft.py      --epochs 1 --ckpt out/dspark.pth

# 推理与投机解码
python -m inference.chat --ckpt out/sft.pth --prompt "你好" --cuda_graph
python -m inference.spec --ckpt out/sft.pth --prompt "请写一段快速排序算法"
```

## 目录结构

```
model/       核心模型架构、算子层与训练组件
trainer/     各阶段训练入口（Pretrain / SFT / VLM / DSpark / SigLIP）
inference/   推理加速系统（Engine / Compile / Chat / Spec）
scripts/     消融实验与评测脚本
```

## 致谢与许可

感谢 DeepSeek 开源模型架构与技术报告（[GitHub](https://github.com/deepseek-ai) / [HuggingFace](https://huggingface.co/deepseek-ai)）。
本项目代码采用 [MIT](LICENSE) 协议开源。
