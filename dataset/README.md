# dataset 目录说明

本目录存放全部训练数据，**不入库**（见 `.gitignore`）。下面记录每个文件的来源、格式、用途。

---

## 1. 纯文本语料（主干预训练 / SFT）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `pretrain_t2t.jsonl` / `pretrain_t2t_mini.jsonl` | 主干预训练 | 2 |
| `sft_t2t.jsonl` / `sft_t2t_mini.jsonl` | 文本 SFT | 5 |

- **来源**：MiniMind 项目
  - HuggingFace：<https://huggingface.co/datasets/jingyaogong/minimind_dataset>
  - ModelScope：<https://www.modelscope.cn/datasets/gongjy/minimind_dataset>
- **格式**：`jsonl`。预训练每行 `{"text": "..."}`；SFT 为多轮对话。
- **说明**：`*_mini` 是轻量版（推荐 `max_seq_len≈768`），本机 8GB 用 mini 版。

## 2. 图文对数据（多模态：ViT 自回归微调 / 后续混合 SFT）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `sft_i2t.parquet` | 完整版图文 QA | 3（可换） |
| `sft_i2t_mini.parquet` | **纯图文** mini 子集（本机用） | 3 |

- **来源**：MiniMind-V 项目；原始图文来自 **ALLAVA-4V** 系列（LAION + VFLAN 子集，中英双语）
  - HuggingFace：<https://huggingface.co/datasets/jingyaogong/minimind-v_dataset>
  - ModelScope：<https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset>
- **格式**：`parquet`，列 `conversations`(json 字符串) + `image_bytes`(binary)。
  图像统一 resize 到 256×256 后编码为 JPEG。
- **完整版构成**（2,904,511 行，实测）：**92.07% 图文**（2,674,248 行，含 `<image>` 占位符）
  + **7.93% 纯文本**（230,263 行，图像列填 8×8 黑图占位、**无** `<image>` 占位符，用于保留基础语言能力）。
- **`sft_i2t_mini.parquet`**：从完整版抽取的 **905,818 行纯图文子集**（按比例随机抽样、
  剔除全部纯文本占位行，含 `<image>` 占 100%）。抽取脚本见 `scripts/`。
- **为什么 mini 只留纯图文**：本项目的多模态阶段拆成「ViT 自回归微调（纯图文）→ 混合 SFT
  （图文 + 纯文本）」两步，各自负责视觉对齐 / 语言保留；故 ViT 阶段的数据必须无纯文本干扰。
- **备注**：这是 **VQA 指令格式**（user 段是问题、assistant 段是描述），适合指令微调，
  **不适合**直接做 SigLIP 对比学习（对比学习要「图 ↔ 描述」的纯图文对，见下）。

## 3. Flickr30k（ViT 对比预训练）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `flickr30k/data/test-0000*-of-00009.parquet` | ViT SigLIP 对比预训练 | 1 |

- **来源**：Flickr30k（Young et al., 2014, TACL）
  - 原始主页：<https://shannon.cs.illinois.edu/DenotationGraph/>
  - HuggingFace 镜像：<https://huggingface.co/datasets/nlphuji/flickr30k>
  - **实际下载源**：ModelScope `lmms-lab/flickr30k`（9 个 parquet 分片）
    <https://www.modelscope.cn/datasets/lmms-lab/flickr30k>
- **格式**：`parquet`，列
  `image`(struct<bytes, path>) + `caption`(list<5 个字符串>) + `sentids` + `img_id` + `filename`。
- **规模**：31,014 张图，每张图 5 条英文描述（共约 15.5 万条 caption）。
- **为什么不用上面那份 `sft_i2t.parquet`**：那是 VQA 问答格式，同一张图配的是「问题 + 回答」，
  不是纯描述；且一张图对应多条不同问答，对比学习里会把同图的其它样本当成负例，检索指标失真。
  Flickr30k 是标准「图 ↔ 5 条描述」图文对，正是 SigLIP 阶段需要的。
