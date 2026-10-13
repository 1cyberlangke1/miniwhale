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

## 2. 图文对数据（多模态 SFT）

| 文件 | 用途 | 阶段 |
|---|---|---|
| `sft_i2t.parquet` / `sft_i2t_mini.parquet` | 多模态 SFT | 3 / 5 |

- **来源**：MiniMind-V 项目；原始图文来自 **ALLaVA-4V** 系列（LAION + VFLAN 子集，中英双语）
  - HuggingFace：<https://huggingface.co/datasets/jingyaogong/minimind-v_dataset>
  - ModelScope：<https://www.modelscope.cn/datasets/gongjy/minimind-v_dataset>
- **格式**：`parquet`，列 `conversations`(json 字符串) + `image_bytes`(binary)。
  图像统一 resize 到 256×256 后编码为 JPEG。
- **规模**：完整版约 290 万条（约 65 万张唯一图像），Pretrain 已作为子集并入。
- **备注**：这是 **VQA 指令格式**（user 段是问题、assistant 段是描述），适合 SFT，
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
