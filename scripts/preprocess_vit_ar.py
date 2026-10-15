"""离线预处理 ViT-AR 分桶数据 → 训练直接读张量（消除现场 PIL 解码）。

动机（同 preprocess_flickr30k.py）：训练时 `num_workers=0` 会让 PIL 解码在主线程同步执行，
GPU 纯等 CPU（profiler 实测占主线程 23%）。而 HF 流式数据集在本机 parquet 上
`reshard()` 只切出 1 个 shard → `num_workers>0` 会被 HF 停掉。
改为**离线多进程解码 + memmap**：训练时只按 index 读张量，worker 用下标取模切分
（不依赖 HF shard），`num_workers>0` 真正生效。

**关键**：编码逻辑**直接复用** `MultimodalDataset._encode_row`（纯函数，不碰流式迭代），
不另写一份——保证预处理结果与现场编码**逐位一致**（重写必然产生偏差，如 break 条件）。

输入：`--data_glob` 指向的 parquet（列 conversations[large_string] + image_bytes[large_binary]）。
输出（out_dir/）：
  - tokens.npy    int32  [N, L]          对话 tokenize + 图像 span 展开，pad 到 L
  - labels.npy    int32  [N, L]          只对非 pad 位置算 loss
  - ttypes.npy    int8   [N, L]          token_types（TEXT=-1，图像 span 0..3）
  - img_start.npy int32  [N]             该样本图像 span 起始下标（-1 = 无图）
  - patches.npy   uint16 [N, P, 3, p, p] ViT patch（bf16 位模式；P = 全局最大 patch 数）
  - n_vit_hw.npy  int32  [N, 2]          每样本 ViT 网格 (n_vit_h, n_vit_w)
  - n_patch.npy   int32  [N]             每样本实际 patch 数（不足 P 的补零）
  - img_types.npy int8   [N, T]          图像 span 的 token_types（T = 全局最大 span 长）
  - n_types.npy   int32  [N]             每样本实际 span 长（< T 的补零）
  - meta.json                            {total, max_length, max_patch, patch_size, ...}

注：`img_types` **必须存**而非按网格重算——现场编码的 span 可能因序列长截断而
    短于 `image_token_types(n_h,n_w)` 的理论值，重算会不一致。

预期行为：多进程解码；逐条写 memmap（每 512 条 flush）；P 取扫描到的全局最大值，
          **不针对特定网格/长度特化**（任何数据都能装下）。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)

# 每个 worker 的全局状态（initializer 建，避免重复加载 tokenizer / 重开 memmap）
_G: dict = {}


def _init(out_dir, data_path, max_length, max_patch, max_types):
    """worker 初始化：建一个**只用于编码**的 MultimodalDataset + 打开 memmap。

    输入：out_dir；data_path（编码器只用它的 tokenizer/vision_cfg，不迭代）；
          max_length；max_patch；max_types（全局最大图像 span 长）。
    输出：无（写全局 _G）。
    预期行为：`MultimodalDataset.__init__` 会建 HF 流式对象（开销小、不读数据），
              我们只调它的 `_encode_row`——保证与训练路径**同一份编码代码**。
    """
    sys.path.insert(0, ROOT)
    from model.config import ModelConfig
    from model.tokenizer import load_tokenizer
    from model.train.data import MultimodalDataset

    cfg = ModelConfig.from_json()
    tok = load_tokenizer()
    _G["ds"] = MultimodalDataset(
        data_path, tok, max_length=max_length,
        image_token_id=cfg.text.image_token_id, vision_cfg=cfg.vision,
        shuffle_buffer=0)
    _G["L"] = max_length
    _G["P"] = max_patch
    _G["T"] = max_types
    for k, name in (("tokens", "tokens.npy"), ("labels", "labels.npy"),
                    ("ttypes", "ttypes.npy"), ("start", "img_start.npy"),
                    ("patch", "patches.npy"), ("hw", "n_vit_hw.npy"),
                    ("np_", "n_patch.npy"), ("itypes", "img_types.npy"),
                    ("ntypes", "n_types.npy")):
        _G[k] = np.load(os.path.join(out_dir, name), mmap_mode="r+")
    _G["n"] = 0


def _encode_write(task):
    """worker 主函数：用训练路径的 `_encode_row` 编码一条并写 memmap。

    输入：(idx, row_dict)，row_dict 含 conversations + image_bytes。
    输出：idx（供主进程计数）。
    预期行为：**调用 `MultimodalDataset._encode_row`**（与训练同一份代码）→ 逐位一致；
              图像 patch 从 ImageInput 取出（最多 1 张，与数据规格一致）。
    """
    import torch

    idx, row = task
    ds = _G["ds"]
    s = ds._encode_row(row)          # 复用训练路径的编码
    L, P = _G["L"], _G["P"]
    _G["tokens"][idx] = s["input_ids"].numpy()
    _G["labels"][idx] = s["labels"].numpy()
    _G["ttypes"][idx] = s["token_types"].numpy()
    imgs = s["images"]
    if imgs:
        img = imgs[0]
        _G["start"][idx] = img.start
        _G["hw"][idx] = (img.n_vit_h, img.n_vit_w)
        n_p = int(img.patches.size(0))
        _G["np_"][idx] = n_p
        _G["patch"][idx, :n_p] = img.patches.contiguous().view(torch.uint16).numpy()
        # types **必须存原样**（不能按网格重算：现场 span 可能因序列长截断而更短）
        tp = img.types.numpy().astype(np.int8)
        _G["itypes"][idx, :tp.size] = tp
        _G["ntypes"][idx] = tp.size
    else:
        _G["start"][idx] = -1
        _G["hw"][idx] = (0, 0)
        _G["np_"][idx] = 0
        _G["ntypes"][idx] = 0
    _G["n"] += 1
    if _G["n"] % 512 == 0:
        for k in ("tokens", "labels", "ttypes", "start", "patch", "hw", "np_",
                  "itypes", "ntypes"):
            _G[k].flush()
    return idx


def _scan_max_patch(files, pp):
    """先扫一遍求全局最大 patch 数（P）与最大图像 span 长（T）。

    输入：parquet 文件列表；图像处理参数 (patch_size, downsample_ratio, max_tokens,
          min_pixels, max_wh_ratio)。
    输出：(max_patch, max_types)。
    预期行为：只解码图像求 patch 数与 span 长（不 tokenize）；代价换固定形状 memmap。
    """
    import pyarrow.parquet as pq

    from model.image_processor import load_image_from_bytes, num_image_tokens

    mx, mt = 0, 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=64, columns=["image_bytes"]):
            for raw in batch.column("image_bytes"):
                b = raw.as_py()
                if not b:
                    continue
                patches, n_h, n_w, n_lh, n_lw = load_image_from_bytes(b, *pp)
                mx = max(mx, patches.size(0))
                mt = max(mt, num_image_tokens(n_lh, n_lw))
    return mx, mt


def main():
    ap = argparse.ArgumentParser(description="ViT-AR 分桶 → 预处理张量")
    ap.add_argument("--data_glob", required=True, help="输入 parquet（可通配）")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, required=True, help="桶上界（= 训练 seq_len）")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    import pyarrow.parquet as pq

    from model.config import ModelConfig

    cfg = ModelConfig.from_json()
    v = cfg.vision
    pp = (v.patch_size, v.downsample_ratio, v.max_image_tokens, v.min_pixels, v.max_wh_ratio)
    files = sorted(glob.glob(args.data_glob))
    if not files:
        ap.error(f"没有匹配的文件：{args.data_glob}")
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"{len(files)} 分片 / {total} 行；max_length={args.max_length}", flush=True)

    print("扫描全局最大 patch 数…", flush=True)
    max_patch, max_types = _scan_max_patch(files, pp)
    print(f"max_patch = {max_patch}  max_types = {max_types}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    L, P, T, p = args.max_length, max_patch, max_types, v.patch_size
    np.lib.format.open_memmap(f"{args.out_dir}/tokens.npy", mode="w+",
                              dtype=np.int32, shape=(total, L))
    np.lib.format.open_memmap(f"{args.out_dir}/labels.npy", mode="w+",
                              dtype=np.int32, shape=(total, L))
    np.lib.format.open_memmap(f"{args.out_dir}/ttypes.npy", mode="w+",
                              dtype=np.int8, shape=(total, L))
    np.lib.format.open_memmap(f"{args.out_dir}/img_start.npy", mode="w+",
                              dtype=np.int32, shape=(total,))
    np.lib.format.open_memmap(f"{args.out_dir}/patches.npy", mode="w+",
                              dtype=np.uint16, shape=(total, P, 3, p, p))
    np.lib.format.open_memmap(f"{args.out_dir}/n_vit_hw.npy", mode="w+",
                              dtype=np.int32, shape=(total, 2))
    np.lib.format.open_memmap(f"{args.out_dir}/n_patch.npy", mode="w+",
                              dtype=np.int32, shape=(total,))
    np.lib.format.open_memmap(f"{args.out_dir}/img_types.npy", mode="w+",
                              dtype=np.int8, shape=(total, T))
    np.lib.format.open_memmap(f"{args.out_dir}/n_types.npy", mode="w+",
                              dtype=np.int32, shape=(total,))
    with open(f"{args.out_dir}/meta.json", "w", encoding="utf-8") as fh:
        json.dump({"total": total, "max_length": L, "max_patch": P, "max_types": T,
                   "patch_size": p, "image_token_id": cfg.text.image_token_id,
                   "downsample_ratio": v.downsample_ratio}, fh)

    def tasks():
        """惰性产出 (idx, row_dict)：row_dict 与 `_encode_row` 期望的键一致。"""
        idx = 0
        for f in files:
            for b in pq.ParquetFile(f).iter_batches(
                    batch_size=256, columns=["conversations", "image_bytes"]):
                cc, ic = b.column("conversations"), b.column("image_bytes")
                for j in range(b.num_rows):
                    yield (idx, {"conversations": cc[j].as_py(),
                                 "image_bytes": ic[j].as_py()})
                    idx += 1

    initargs = (args.out_dir, files[0], L, P, T)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=initargs) as ex:
        for _ in ex.map(_encode_write, tasks(), chunksize=32):
            done += 1
            if done % 5000 == 0:
                print(f"  已处理 {done}/{total}", flush=True)
    print(f"完成：{done} 条 → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
