"""数据加载。

复用 minimind 的数据格式：
- 预训练：jsonl，每行 {"text": "..."}
- 多模态 SFT：parquet，列 conversations（多轮对话）+ image_bytes

标签构造：只对非 padding 位置算 loss（pad 置 -100）。
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info


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
        # 每条文档 = 1 个训练样本（不打包），故样本数即文档数。用于 lr 调度总步数。
        self.num_rows = n_docs
        # 流式：把 shuffle 建在 __init__（**持久**对象），每 epoch 用 set_epoch 换 seed 重排；
        # 建在 __iter__ 里会因 seed 固定 → 每轮顺序一模一样（等于没打乱）。
        if streaming and shuffle:
            self.samples = self.samples.shuffle(buffer_size=shuffle, seed=seed)

    def set_epoch(self, epoch: int):
        """每 epoch 前调用：流式 shuffle 有效 seed = seed + epoch，重排数据顺序。"""
        if self.streaming and self.shuffle and hasattr(self.samples, "set_epoch"):
            self.samples.set_epoch(epoch)

    def _stream(self):
        """按配置返回文档流（流式打乱 / 顺序）。"""
        if self.streaming:
            return self.samples
        return (self.samples[i] for i in torch.randperm(len(self.samples)).tolist())

    def __iter__(self):
        # 照 minimind：每条文档右侧 pad 到 max_length（**固定形状** → compile 只编一次）。
        # 样本各自独立、右侧 pad，天然满足 sample-level masking（不做打包、不做分桶）。
        bos, eos = self.tokenizer.bos_token_id, self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        for row in self._stream():
            ids = self.tokenizer(str(row["text"]), add_special_tokens=False,
                                 max_length=self.max_length - 2, truncation=True).input_ids
            toks = ([bos] + ids + [eos])[:self.max_length]
            t = torch.full((self.max_length,), pad, dtype=torch.long)
            t[:len(toks)] = torch.tensor(toks, dtype=torch.long)
            labels = t.clone()
            labels[t == pad] = -100
            yield {"input_ids": t, "labels": labels}

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
        # 流式：shuffle 建在 __init__（持久对象），每 epoch 用 set_epoch 换 seed 重排。
        if streaming and shuffle:
            self.samples = self.samples.shuffle(buffer_size=shuffle, seed=seed)

    def set_epoch(self, epoch: int):
        """每 epoch 前调用：流式 shuffle 有效 seed = seed + epoch，重排数据顺序。"""
        if self.streaming and self.shuffle and hasattr(self.samples, "set_epoch"):
            self.samples.set_epoch(epoch)

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
            for row in self.samples:
                yield self._encode(row)
        else:
            order = torch.randperm(len(self.samples)).tolist()
            for i in order:
                yield self._encode(self.samples[i])

    def __len__(self):
        return self.num_rows


class MultimodalDataset(IterableDataset):
    """多模态 SFT 数据集（parquet：conversations + image_bytes）——**流式**读取。

    输入：parquet 路径；tokenizer；max_length；vision 配置（patch/下采样/尺寸约束）；
          shuffle_buffer（近似 shuffle 缓冲，0=顺序）。
    输出（迭代）：{"input_ids": [L], "labels": [L], "token_types": [L], "images": list[ImageInput]}。
    预期行为：用 `datasets.load_dataset(..., streaming=True)` 原地流式读原 parquet（不复制、
              不落缓存）；`reshard()` 按 row group 切 shard，DataLoader 的 num_workers 各占
              互不相交的 shard。按参考实现 image_processor —— `<image>` 占位符展开成
              [IMAGE_START] + ([IMAGE]*w + [IMAGE_NEW_LINE])*h + [IMAGE_END]，全用同一
              image_token_id，靠 token_types 区分；图像切成 ViT patch。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 768,
                 image_placeholder: str = "<image>", image_token_id: int = 12,
                 vision_cfg=None, shuffle_buffer: int = 10000, seed: int = 42):
        from datasets import load_dataset
        import pyarrow.parquet as pq

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_placeholder = image_placeholder
        self.image_token_id = image_token_id
        self.vision_cfg = vision_cfg
        self.path = str(data_path)
        # 行数：只读 parquet footer 元数据（几 KB），不加载数据。
        self._num_rows = pq.ParquetFile(self.path).metadata.num_rows
        # 流式读原文件（streaming=True：**不复制、不落缓存**，原地按 row group 读）。
        # reshard() 按 row group 切 shard，num_workers 各占互不相交的 shard（库保证不重读）。
        self.samples = load_dataset("parquet", data_files=self.path,
                                    split="train", streaming=True).reshard()
        if shuffle_buffer:
            # 近似 shuffle（shard 顺序打乱 + shuffle buffer），每 epoch 用 set_epoch 重排。
            self.samples = self.samples.shuffle(buffer_size=shuffle_buffer, seed=seed)

    def __len__(self):
        return self._num_rows

    def set_epoch(self, epoch: int):
        """每个 epoch 前调用：重排 shard 顺序与 shuffle buffer（有效 seed = seed + epoch）。"""
        if hasattr(self.samples, "set_epoch"):
            self.samples.set_epoch(epoch)

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

    def __iter__(self):
        for row in self.samples:
            yield self._encode_row(row)

    def _encode_row(self, row):
        """把一行 parquet 记录编码成训练样本。

        输入：row dict（conversations + image_bytes）。
        输出：{"input_ids": [L], "labels": [L], "token_types": [L], "images": list[ImageInput]}。
        预期行为：chat template 渲染 → 有图则把 image 占位展开成参考实现 span；pad 位 label=-100。
        """
        import json

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


class ImageTextPairDataset(IterableDataset):
    """图文对数据集（SigLIP 对比预训练用，报告 §4.2.2 第一阶段）——**流式**读取。

    输入：parquet 路径（可含通配符，多分片）；tokenizer；max_length；
          shuffle_buffer（近似 shuffle 缓冲，0=顺序）；
          source 数据格式："conversations"（列 conversations + image_bytes，多模态对话）
          或 "flickr30k"（列 image[PIL] + caption[list[str]]，标准图文对）；
          image_size 把图统一 resize 到 N×N（0=保持原尺寸）——参考实现 SigLIP 阶段
          固定 224×224，且固定形状才能让同批图一次过 ViT（否则按 patch 数分组，组碎）。
    输出（迭代）：{"patches": [n,3,p,p], "grid": (n_vit_h, n_vit_w), "text_ids": [L], "text_mask": [L]}。
    预期行为：用 `load_dataset(..., streaming=True)` 原地流式读（不复制、不落缓存）；
              conversations 取首轮 assistant 文本（图像描述）去占位符作 caption；
              flickr30k 每行 5 条描述，**随机取一条**（同图多描述若都当独立正例，
              会互成假负例，故每行只出一个样本）。图像切成 ViT patch；
              无图或空 caption 的行返回 None（由 collate 过滤）。
    """

    def __init__(self, data_path: str | Path, tokenizer, max_length: int = 64,
                 image_placeholder: str = "<image>", vision_cfg=None,
                 shuffle_buffer: int = 10000, seed: int = 42, source: str = "conversations",
                 image_size: int = 0):
        import glob as _glob

        import pyarrow.parquet as pq
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_placeholder = image_placeholder
        self.vision_cfg = vision_cfg
        self.path = str(data_path)
        self.source = source
        self.seed = seed
        self.image_size = image_size
        # 行数：只读各分片 parquet footer 元数据（几 KB），不加载数据。支持通配符多分片。
        files = sorted(_glob.glob(self.path)) if _glob.has_magic(self.path) else [self.path]
        self._num_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        # 流式读原文件（streaming=True：**不复制、不落缓存**）；reshard() 按 row group
        # 切 shard，num_workers 各占互不相交的 shard（库保证不重读）。
        self.samples = load_dataset("parquet", data_files=files,
                                    split="train", streaming=True).reshard()
        if shuffle_buffer:
            self.samples = self.samples.shuffle(buffer_size=shuffle_buffer, seed=seed)

    def __len__(self):
        return self._num_rows

    def set_epoch(self, epoch: int):
        """每个 epoch 前调用：重排 shard 顺序与 shuffle buffer（有效 seed = seed + epoch）。"""
        if hasattr(self.samples, "set_epoch"):
            self.samples.set_epoch(epoch)

    def __iter__(self):
        for row in self.samples:
            sample = self._encode_row(row)
            if sample is not None:
                yield sample

    def _encode_row(self, row):
        """把一行图文记录编码成 SigLIP 训练样本（无图/空 caption 返回 None）。

        输入：row dict（按 self.source 取图与 caption）。
        输出：{"patches", "grid", "text_ids", "text_mask"} 或 None。
        预期行为：conversations 格式取首轮 assistant 文本；flickr30k 格式随机取 5 条描述之一；
                  图像切 ViT patch；文本 pad 到 max_length。
        """
        import random

        from model.image_processor import load_image

        vc = self.vision_cfg
        if self.source == "flickr30k":
            image = row.get("image")
            if image is None:
                return None
            caps = row.get("caption") or []
            if not caps:
                return None
            # 每张图 5 条描述：随机取一条（同图其它描述会被当假负例，故每行只出一个样本）。
            caption = str(random.choice(caps)).strip()
            # 统一 resize 到 N×N：参考实现 SigLIP 阶段固定 224×224；固定形状才能让同批图
            # 一次过 ViT（否则按 patch 数分组，图尺寸各异 → 组碎、kernel 启动多、慢）。
            if self.image_size:
                image = image.convert("RGB").resize((self.image_size, self.image_size))
            patches, n_vit_h, n_vit_w, _, _ = load_image(
                image, vc.patch_size, vc.downsample_ratio, vc.max_image_tokens,
                vc.min_pixels, vc.max_wh_ratio)
        else:
            import json

            from model.image_processor import load_image_from_bytes

            raw = row.get("image_bytes")
            if not raw:
                return None
            convs = row.get("conversations") or []
            if isinstance(convs, str):
                convs = json.loads(convs)
            # 取首轮 assistant 文本当 caption（去掉图像占位符）。
            # 数据是 VQA 格式：user 段是**问题**（如"对应的内容是什么？"），assistant 段才是
            # **图像描述**。对比学习要的是「图 ↔ 描述」配对，取 user（问题）会得到无语义的
            # 负样本（图文对不上），检索指标会被拉垮。参考实现 SigLIP 阶段用 alt-text（描述）同理。
            caption = ""
            for c in convs:
                if c.get("role") == "assistant":
                    caption = str(c.get("content", "")).replace(
                        self.image_placeholder, "").strip()
                    break
            if not caption:
                return None
            patches, n_vit_h, n_vit_w, _, _ = load_image_from_bytes(
                raw, vc.patch_size, vc.downsample_ratio, vc.max_image_tokens,
                vc.min_pixels, vc.max_wh_ratio)
        if not caption:
            return None
        ids = self.tokenizer(caption, add_special_tokens=False,
                             max_length=self.max_length, truncation=True)["input_ids"]
        pad = self.tokenizer.pad_token_id
        text_ids = ids + [pad] * max(0, self.max_length - len(ids))
        text_mask = [1] * len(ids) + [0] * max(0, self.max_length - len(ids))
        return {"patches": patches, "grid": (n_vit_h, n_vit_w),
                "text_ids": torch.tensor(text_ids[:self.max_length]),
                "text_mask": torch.tensor(text_mask[:self.max_length], dtype=torch.bool)}


class PreprocessedPairDataset(IterableDataset):
    """图文对数据集（读**离线预处理**张量，零现场解码）——SigLIP 对比预训练用。

    输入：预处理目录（含 patches.npy / captions.npy / caption_mask.npy / meta.json）；
          shuffle（True=每 epoch 打乱图序，默认）。
    输出（迭代）：{"patches": [n,3,p,p](bf16), "grid": (n_h,n_w), "text_ids": [L], "text_mask": [L]}。
    预期行为：patches 以 uint16 位模式存 memmap，`.view(torch.bfloat16)` 零拷贝还原；
              每张图（=一行）随机取一条 caption 当正样本（同图多描述会互成假负例，
              故每行只出一个样本，多 caption 通过多 epoch 轮换覆盖）。
              num_workers>0 时各 worker 按下标取模切分，互不重叠。
    """

    def __init__(self, pre_dir: str | Path, shuffle: bool = True, seed: int = 42):
        import json

        self.dir = str(pre_dir)
        meta = json.loads(Path(self.dir, "meta.json").read_text(encoding="utf-8"))
        self.total = meta["total"]
        self.grid = (meta["grid"], meta["grid"])
        self.num_captions = meta["num_captions"]
        self.caption_len = meta["caption_len"]
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # **只存路径**，不在此打开 memmap：DataLoader 用 spawn 起 worker 要 pickle 整个
        # dataset，np.memmap 不可 pickle（会 OSError: Invalid argument）。memmap 在
        # __iter__ 里由**每个 worker 自己**惰性打开。

    def __len__(self):
        return self.total

    def set_epoch(self, epoch: int):
        """每 epoch 前调用：换随机种子，图序与 caption 选择都随之变化。"""
        self.epoch = epoch

    def __iter__(self):
        import random

        import numpy as np

        # 每个 worker 各自打开 memmap（不跨进程 pickle）；只在访问时读该行，不占常驻内存。
        patches = np.load(Path(self.dir, "patches.npy"), mmap_mode="r")
        caps = np.load(Path(self.dir, "captions.npy"), mmap_mode="r")
        cmask = np.load(Path(self.dir, "caption_mask.npy"), mmap_mode="r")
        order = list(range(self.total))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        wi = get_worker_info()
        if wi is not None:                      # 多 worker：按下标取模切分，互不重叠
            order = order[wi.id::wi.num_workers]
        for i in order:
            # 每张图（=一行）随机取一条 caption（同图多描述会互成假负例，故每行只出一个样本）
            k = random.Random(self.seed + self.epoch * 1000003 + i).randrange(self.num_captions)
            # uint16 memmap → bf16 张量：.copy() 保证可写连续（from_numpy 视图不能 view）
            yield {"patches": torch.from_numpy(patches[i].copy()).view(torch.bfloat16),
                   "grid": self.grid,
                   "text_ids": torch.from_numpy(caps[i, k].copy()).long(),
                   "text_mask": torch.from_numpy(cmask[i, k].copy()).bool()}
