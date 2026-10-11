"""数据加载。

复用 minimind 的数据格式：
- 预训练：jsonl，每行 {"text": "..."}
- 多模态 SFT：parquet，列 conversations（多轮对话）+ image_bytes

标签构造：只对非 padding 位置算 loss（pad 置 -100）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """预训练数据集（jsonl，每行 {"text": ...}）。

    输入：data_path jsonl 路径；tokenizer；max_length 截断长度。
    输出：{"input_ids": [L], "labels": [L]}，pad 位置 label=-100。
    预期行为：tokenize 后加 bos/eos，右侧 padding 到 max_length。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=str(data_path), split="train")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = str(self.samples[index]["text"])
        tokens = self.tokenizer(text, add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        pad = self.tokenizer.pad_token_id
        input_ids = tokens + [pad] * (self.max_length - len(tokens))
        labels = list(input_ids)
        for i, v in enumerate(labels):
            if v == pad:
                labels[i] = -100
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


class SFTDataset(Dataset):
    """SFT 数据集（jsonl，多轮对话）。

    输入：jsonl 路径；tokenizer；max_length。
    输出：{"input_ids": [L], "labels": [L]}，只对 assistant 段算 loss。
    预期行为：apply_chat_template 渲染；非 assistant 段与 pad 位置 label=-100。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset("json", data_files=str(data_path), split="train")
        # assistant 段起止标记（照 minimind：`<|im_start|>assistant\n` ... `<|im_end|>\n`）
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant\n",
                                add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}\n",
                                add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

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

    def __getitem__(self, index):
        convs = self.samples[index].get("conversations", [])
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

    def _image_inputs(self, prompt_ids, raw_images):
        """把 prompt 里的 image_token 占位展开成参考实现图像 span。

        输入：prompt_ids list[int]；raw_images list[bytes]。
        输出：(tokens list[int], token_types list[int], images list[ImageInput])。
        预期行为：占位符处插入 IMAGE_START/IMAGE/NEW_LINE/END 布局，并记录每张图的 patch。
        """
        from model.image_processor import (ImageInput, TEXT, image_token_types,
                                           load_image_from_bytes)

        vc = self.vision_cfg
        tokens, token_types, images = [], [], []
        img_iter = iter(raw_images)
        for tok in prompt_ids:
            if tok != self.image_token_id:
                tokens.append(tok)
                token_types.append(TEXT)
                continue
            raw = next(img_iter)
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = load_image_from_bytes(
                raw, vc.patch_size, vc.downsample_ratio, vc.max_image_tokens,
                vc.min_pixels, vc.max_wh_ratio)
            types = image_token_types(n_llm_h, n_llm_w)
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
            ids, token_types, images = self._image_inputs(ids, [raw])

        pad = self.tokenizer.pad_token_id
        input_ids = ids[:self.max_length] + [pad] * max(0, self.max_length - len(ids))
        token_types = token_types[:self.max_length] + [-1] * max(0, self.max_length - len(token_types))
        labels = [(-100 if v == pad else v) for v in input_ids]
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "token_types": torch.tensor(token_types),
                "images": images}
