"""数据加载。

复用 minimind 的数据格式：
- 预训练：jsonl，每行 {"text": "..."}
- 多模态 SFT：parquet，列 conversations（多轮对话）+ image_bytes

标签构造：只对非 padding 位置算 loss（pad 置 -100）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset


class PretrainDataset(IterableDataset):
    """预训练数据集（jsonl，每行 {"text": ...}）——**流式**读取，不整表载入。

    输入：data_path jsonl 路径；tokenizer；max_length 截断长度；streaming（默认 True=流式）；
          shuffle 缓冲区大小（0=顺序）；seed。
    输出（迭代）：{"input_ids": [L], "labels": [L]}，pad 位置 label=-100。
    预期行为：streaming=True 时 datasets 流式逐行读（1.2GB / 127 万行也不占内存）；
              streaming=False 时整表载入内存（map-style 语义，可 __len__）。
              tokenize 后加 bos/eos，右侧 padding 到 max_length；每个 epoch 打乱。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768,
                 streaming: bool = True, shuffle: int = 1000, seed: int = 42):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.path = str(data_path)
        self.streaming = streaming
        self.shuffle = shuffle
        self.seed = seed
        self.samples = load_dataset("json", data_files=self.path, split="train",
                                    streaming=streaming)
        # 行数：datasets 对本地 jsonl 已统计，流式/非流式都在 info.splits 里给，直接用。
        # 极小的临时文件 datasets 可能不给 splits（None）——回退 0，调用方按 0 处理。
        splits = getattr(self.samples.info, "splits", None)
        n_docs = splits["train"].num_examples if splits and "train" in splits else 0
        # 打包后的**片段数**（而非文档数）：采样若干文档估平均 token 数，估算总片段数。
        # 用于 lr 调度总步数；估偏一点无妨（调度只是曲线形状），但必须量级正确。
        self.num_rows = self._estimate_packed_chunks(n_docs) if n_docs else 0

    def _estimate_packed_chunks(self, n_docs: int, sample: int = 512) -> int:
        """采样估总 token，除以 max_length 得片段数。"""
        tot, n = 0, 0
        for row in self.samples.take(sample) if self.streaming else self.samples.select(range(min(sample, n_docs))):
            ids = self.tokenizer(str(row["text"]), add_special_tokens=False,
                                 max_length=self.max_length, truncation=True).input_ids
            tot += len(ids) + 2  # +bos +eos
            n += 1
        avg = tot / max(n, 1)
        return int(n_docs * avg / self.max_length)

    def _stream(self):
        """按配置返回文档流（流式打乱 / 顺序）。"""
        if self.streaming:
            return self.samples.shuffle(buffer_size=self.shuffle, seed=self.seed) \
                if self.shuffle else self.samples
        return (self.samples[i] for i in torch.randperm(len(self.samples)).tolist())

    def __iter__(self):
        # **序列打包**：把文档首尾相连成一条 token 流，再切成定长片段——零 padding。
        # 否则短文档（本语料平均 ~160 token）右侧 pad 到 max_length 会浪费 ~80% 的
        # GPU 算力（算在 pad 上）。打包是标准预训练做法，让 GPU 只算真实 token。
        buf: list[int] = []
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        for row in self._stream():
            ids = self.tokenizer(str(row["text"]), add_special_tokens=False,
                                 max_length=self.max_length, truncation=True).input_ids
            buf.append(bos)
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.max_length:
                chunk = buf[:self.max_length]
                del buf[:self.max_length]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}

    def __len__(self):
        return self.num_rows


class SFTDataset(IterableDataset):
    """SFT 数据集（jsonl，多轮对话）——**流式**读取，不整表载入。

    输入：jsonl 路径；tokenizer；max_length；streaming（默认 True=流式）；
          shuffle 缓冲区大小（0=顺序）；seed。
    输出（迭代）：{"input_ids": [L], "labels": [L]}，只对 assistant 段算 loss。
    预期行为：streaming=True 时流式逐行读；apply_chat_template 渲染；非 assistant 段与 pad 位置 label=-100。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768,
                 streaming: bool = True, shuffle: int = 1000, seed: int = 42):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.path = str(data_path)
        self.streaming = streaming
        self.shuffle = shuffle
        self.seed = seed
        self.samples = load_dataset("json", data_files=self.path, split="train",
                                    streaming=streaming)
        # 行数：datasets 对本地 jsonl 已统计，流式/非流式都在 info.splits 里给，直接用。
        # 极小的临时文件 datasets 可能不给 splits（None）——回退 0，调用方按 0 处理。
        splits = getattr(self.samples.info, "splits", None)
        self.num_rows = splits["train"].num_examples if splits and "train" in splits else 0
        # assistant 段起止标记（照 minimind：`<|im_start|>assistant\n` ... `<|im_end|>\n`）
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n",
                                add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n",
                                add_special_tokens=False).input_ids

    def generate_labels(self, input_ids):
        """只在 assistant 段算 loss（照 minimind）。

        输入：input_ids list[int]。
        输出：labels list[int]，非 assistant 段与 padding 置 -100。
        预期行为：扫描 `<|im_start|>assistant\n` … `<|im_end|>\n` 区间，区间内置原 id。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def _encode(self, row):
        convs = row.get("conversations", []) or []
        # 只保留模板认识的字段，避免多余键报错
        convs = [{"role": c["role"], "content": c["content"]} for c in convs]
        out = self.tokenizer.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=False,
            max_length=self.max_length, truncation=True,
        )
        ids = out["input_ids"] if hasattr(out, "keys") else out
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        labels = self.generate_labels(ids)
        pad = self.tokenizer.pad_token_id
        input_ids = list(ids) + [pad] * (self.max_length - len(ids))
        labels = labels + [-100] * (self.max_length - len(labels))
        return {"input_ids": torch.tensor(input_ids[:self.max_length]),
                "labels": torch.tensor(labels[:self.max_length])}

    def __iter__(self):
        if self.streaming:
            ds = self.samples.shuffle(buffer_size=self.shuffle, seed=self.seed) \
                if self.shuffle else self.samples
            for row in ds:
                yield self._encode(row)
        else:
            order = torch.randperm(len(self.samples)).tolist()
            for i in order:
                yield self._encode(self.samples[i])

    def __len__(self):
        return self.num_rows


class MultimodalDataset(Dataset):
    """多模态 SFT 数据集（parquet：conversations + image_bytes）。

    输入：parquet 路径；tokenizer；max_length；vision 配置（patch/下采样/尺寸约束）。
    输出：{"input_ids": [L], "labels": [L], "token_types": [L], "images": list[ImageInput]}。
    预期行为：按参考实现 image_processor —— `<image>` 占位符展开成
              [IMAGE_START] + ([IMAGE]*w + [IMAGE_NEW_LINE])*h + [IMAGE_END]，全用同一
              image_token_id，靠 token_types 区分；图像切成 ViT patch。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768,
                 image_placeholder: str = "<image>", image_token_id: int = 12,
                 vision_cfg=None, row_group_size: int = 64):
        import pyarrow.parquet as pq

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_placeholder = image_placeholder
        self.image_token_id = image_token_id
        self.vision_cfg = vision_cfg
        self.path = str(data_path)
        self.pf = pq.ParquetFile(self.path)
        self.row_group_size = row_group_size

    def __len__(self):
        return self.pf.metadata.num_rows

    def _image_inputs(self, prompt_ids, raw_images, max_length):
        """把 prompt 里的 image_token 占位展开成参考实现图像 span。

        输入：prompt_ids list[int]；raw_images list[bytes]；max_length 序列长度上限。
        输出：(tokens list[int], token_types list[int], images list[ImageInput])。
        预期行为：占位符处插入 IMAGE_START/IMAGE/NEW_LINE/END 布局，并记录每张图的 patch。
                  展开后若总长超过 max_length，停止展开（丢弃其后 token）——保证每个
                  ImageInput 的 span 完整落在 max_length 内（否则下游 merge 时 mask 与
                  span 形状不匹配）。
        """
        from model.image_processor import (ImageInput, TEXT, image_token_types,
                                           load_image_from_bytes)

        vc = self.vision_cfg
        tokens, token_types, images = [], [], []
        img_iter = iter(raw_images)
        for tok in prompt_ids:
            if len(tokens) >= max_length:
                break
            if tok != self.image_token_id:
                tokens.append(tok)
                token_types.append(TEXT)
                continue
            # 占位符多于实际图（脏数据：prompt 里写了多个 `<image>` 但只附一张图）时，
            # 多余的占位符跳过——不能 `next()` 抛 StopIteration，否则 DataLoader 会把
            # 它当作迭代结束、静默终止整个 epoch（整轮训练只跑几步就"完成"）。
            raw = next(img_iter, None)
            if raw is None:
                continue
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image_from_bytes(
                raw, vc.patch_size, vc.downsample_ratio, vc.max_image_tokens,
                vc.min_pixels, vc.max_wh_ratio)
            types = image_token_types(n_llm_h, n_llm_w)
            if len(tokens) + types.numel() > max_length:
                break  # 整张图 span 放不下 → 停止（不半展开，避免 span 越界）
            images.append(ImageInput(len(tokens), patches, n_vit_h, n_vit_w, types))
            tokens += [self.image_token_id] * types.numel()
            token_types += types.tolist()
        return tokens, token_types, images

    def __getitem__(self, index):
        import json

        # 不整表加载（数 GB）：按全局行号定位所属 row group，再读那一组
        rg, offset = 0, index
        while rg < self.pf.num_row_groups and offset >= self.pf.metadata.row_group(rg).num_rows:
            offset -= self.pf.metadata.row_group(rg).num_rows
            rg += 1
        row = self.pf.read_row_group(rg).slice(offset, 1).to_pylist()[0]
        convs = row["conversations"] or []
        if isinstance(convs, str):
            convs = json.loads(convs)
        raw = row.get("image_bytes")
        # 有图：把 `<image>` 换成 image_token 占位（一个），随后由 _image_inputs 展开成 span
        if raw:
            convs = [{"role": c["role"],
                      "content": str(c["content"]).replace(
                          self.image_placeholder, self.tokenizer.decode([self.image_token_id]))}
                     for c in convs]
        out = self.tokenizer.apply_chat_template(
            convs, tokenize=True, add_generation_prompt=False,
            max_length=self.max_length, truncation=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        ids = list(ids)

        token_types = [(-1)] * len(ids)
        images = []
        if raw:
            ids, token_types, images = self._image_inputs(ids, [raw], self.max_length)

        pad = self.tokenizer.pad_token_id
        input_ids = ids[:self.max_length] + [pad] * max(0, self.max_length - len(ids))
        token_types = token_types[:self.max_length] + [-1] * max(0, self.max_length - len(token_types))
        labels = [(-100 if v == pad else v) for v in input_ids]
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "token_types": torch.tensor(token_types),
                "images": images}
