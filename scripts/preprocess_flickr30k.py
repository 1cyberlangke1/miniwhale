"""离线预处理 Flickr30k → SigLIP 训练用的紧凑张量。

把每张图 resize 到 N×N、切成 ViT patch，存成**固定形状**的 memmap（uint16 装 bf16 位模式），
caption 预 tokenize。训练时直接按 index 读张量，**不再现场 PIL 解码**——消除 GPU 等 CPU
解码的瓶颈（实测 num_workers=0 时主进程串行解码 512 张图/步，GPU 利用率 0%）。

输入：Flickr30k parquet（列 image[struct<bytes,path>] + caption[list[str]]）。
输出：out_dir/patches.npy (uint16, [N, n_patch, 3, p, p])、captions.npy (int32, [N, K, L])、
      caption_mask.npy (uint8, [N, K, L])、meta.json。
预期行为：多进程解码 + 切 patch，逐条写入 memmap（写够一批 flush 一次）；进度打印到 stdout。
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)

# 每个 worker 进程的全局状态（initializer 里建，避免重复加载 tokenizer/重开 memmap）
_G: dict = {}


def _init(out_dir, image_size, caption_len, patch_size, downsample_ratio,
          max_image_tokens, min_pixels, max_wh_ratio):
    """worker 初始化：加载 tokenizer + 打开 memmap（可写）。

    输入：out_dir；image_size；caption_len；图像处理参数（同 load_image 签名）。
    输出：无（写全局 _G）。
    预期行为：每个 worker 独立打开同一 memmap 的不同区域写（OS 页缓存保证一致）。
    """
    sys.path.insert(0, ROOT)
    from model.tokenizer import load_tokenizer

    _G["tok"] = load_tokenizer()
    _G["size"] = image_size
    _G["clen"] = caption_len
    _G["pp"] = (patch_size, downsample_ratio, max_image_tokens, min_pixels, max_wh_ratio)
    _G["arr"] = np.load(os.path.join(out_dir, "patches.npy"), mmap_mode="r+")
    _G["cap"] = np.load(os.path.join(out_dir, "captions.npy"), mmap_mode="r+")
    _G["cm"] = np.load(os.path.join(out_dir, "caption_mask.npy"), mmap_mode="r+")
    _G["n"] = 0


def _encode_write(task):
    """解码一张图 → patch，并写进 memmap 的第 idx 行；caption 预 tokenize。

    输入：(idx, 图像 bytes, caption 字符串列表)。
    输出：idx（供主进程计数）。
    预期行为：PIL 解码 → resize N×N → load_image 切 patch（bf16）→ 位模式存 uint16；
              每条 caption tokenize 到 caption_len，写 ids 与 mask。
    """
    import torch
    from PIL import Image

    from model.image_processor import load_image

    idx, raw, caps = task
    img = Image.open(io.BytesIO(raw)).convert("RGB").resize((_G["size"], _G["size"]))
    patches, *_ = load_image(img, *_G["pp"])
    # bf16 → uint16 位模式（无损存储：训练时 .view(torch.bfloat16) 原样还原）
    _G["arr"][idx] = patches.contiguous().view(torch.uint16).numpy()
    tok, clen, K = _G["tok"], _G["clen"], _G["cap"].shape[1]
    for k, c in enumerate(caps[:K]):
        ids = tok(str(c), add_special_tokens=False,
                  max_length=clen, truncation=True).input_ids
        _G["cap"][idx, k, :len(ids)] = ids
        _G["cm"][idx, k, :len(ids)] = 1
    _G["n"] += 1
    if _G["n"] % 512 == 0:      # 每 512 条落盘一次，避免进程被杀时丢太多
        _G["arr"].flush(); _G["cap"].flush(); _G["cm"].flush()
    return idx


def main():
    ap = argparse.ArgumentParser(description="Flickr30k → SigLIP 预处理张量")
    ap.add_argument("--data_glob", default="dataset/flickr30k/data/*.parquet")
    ap.add_argument("--out_dir", default="dataset/flickr30k_pre")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--caption_len", type=int, default=64)
    ap.add_argument("--num_captions", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2000, help="每批提交给进程池的任务数")
    args = ap.parse_args()

    sys.path.insert(0, ROOT)
    import pyarrow.parquet as pq

    from model.config import ModelConfig

    v = ModelConfig.from_json().vision
    p = v.patch_size
    grid = args.image_size // p
    n_patch = grid * grid
    files = sorted(glob.glob(args.data_glob))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(files)} 分片 / {total} 行；patch {n_patch} 个 ({grid}×{grid})，"
          f"每 patch (3,{p},{p})", flush=True)

    # 创建固定形状 memmap（.npy 格式，支持 mmap_mode='r+' 打开可写）
    np.lib.format.open_memmap(f"{args.out_dir}/patches.npy", mode="w+",
                              dtype=np.uint16, shape=(total, n_patch, 3, p, p))
    np.lib.format.open_memmap(f"{args.out_dir}/captions.npy", mode="w+",
                              dtype=np.int32,
                              shape=(total, args.num_captions, args.caption_len))
    np.lib.format.open_memmap(f"{args.out_dir}/caption_mask.npy", mode="w+",
                              dtype=np.uint8,
                              shape=(total, args.num_captions, args.caption_len))
    with open(f"{args.out_dir}/meta.json", "w", encoding="utf-8") as fh:
        json.dump({"total": total, "grid": grid, "patch_size": p,
                   "image_size": args.image_size, "caption_len": args.caption_len,
                   "num_captions": args.num_captions}, fh)

    def tasks():
        """惰性产出 (idx, image_bytes, captions)：不一次把全部图像读进内存。"""
        idx = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=256, columns=["image", "caption"]):
                ic, cc = batch.column("image"), batch.column("caption")
                for j in range(batch.num_rows):
                    yield (idx, ic[j].as_py()["bytes"], cc[j].as_py())
                    idx += 1

    initargs = (args.out_dir, args.image_size, args.caption_len, p,
                v.downsample_ratio, v.max_image_tokens, v.min_pixels, v.max_wh_ratio)
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=initargs) as ex:
        buf = []
        for t in tasks():
            buf.append(t)
            if len(buf) >= args.batch:
                done += sum(1 for _ in ex.map(_encode_write, buf, chunksize=16))
                print(f"  {done}/{total}", flush=True)
                buf = []
        if buf:
            done += sum(1 for _ in ex.map(_encode_write, buf, chunksize=16))
    print(f"完成：{done} 条 → {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
