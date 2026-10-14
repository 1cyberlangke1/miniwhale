"""混合 SFT 数据：文本 jsonl + 图文 parquet 按条数比例交错成统一 parquet。

输入：
  --text_data   文本 jsonl（每行 {"conversations": [...]}）
  --image_data  图文 parquet（列 conversations + image_bytes）
  --image_ratio 图文占最终条数的比例（0.25 → 文本:图文 = 3:1）
  --out         输出 parquet
  --seed        随机种子
  --chunk       写出时每批行数
输出：--out parquet，列 conversations + image_bytes（文本行 image_bytes = null）。
预期行为：两个源各自打散；图文按比例随机抽取；按「文本×(1/r-1) 条 + 图文 1 条」交错；
          图文量不足时按实际可用量交错。不做 tokenize、不按长度分桶（那是另一步）。
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_text(path: str) -> list[str]:
    """读文本 jsonl，返回 conversations 的 JSON 串列表（原样保留，不解析语义）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                convs = json.loads(line)["conversations"]
            except Exception:
                continue
            out.append(json.dumps(convs, ensure_ascii=False))
    return out


def sample_images(path: str, n_want: int, seed: int) -> tuple[list[str], list[bytes]]:
    """从图文 parquet 按比例随机抽 n_want 行，返回 (conversations 串, image_bytes) 两列表。

    输入：parquet 路径；目标条数；种子。
    输出：(convs 列表, imgs 列表)。
    预期行为：逐 row group 按 n_want/总数 的概率随机保留行，最后裁到精确条数；不整表进内存两次。
    """
    f = pq.ParquetFile(path)
    total = f.metadata.num_rows
    p = min(1.0, n_want / max(total, 1))
    rng = np.random.default_rng(seed)
    convs, imgs = [], []
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["conversations", "image_bytes"])
        n = len(t)
        keep = rng.random(n) < p
        if not keep.any():
            continue
        t = t.filter(pa.array(keep))
        convs += t.column("conversations").to_pylist()
        imgs += t.column("image_bytes").to_pylist()
    if len(convs) > n_want:                       # 裁到精确条数（随机裁）
        idx = rng.permutation(len(convs))[:n_want]
        convs = [convs[i] for i in idx]
        imgs = [imgs[i] for i in idx]
    return convs, imgs


def main():
    ap = argparse.ArgumentParser(description="混合 SFT 数据（文本 + 图文交错）")
    ap.add_argument("--text_data", required=True)
    ap.add_argument("--image_data", required=True)
    ap.add_argument("--image_ratio", type=float, default=0.25,
                    help="图文占最终条数的比例（0.25 → 3:1）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=50000)
    args = ap.parse_args()

    t0 = time.time()
    print("读文本 ...", flush=True)
    texts = load_text(args.text_data)
    print(f"  文本 {len(texts)} 条（{time.time()-t0:.0f}s）", flush=True)

    r = args.image_ratio
    n_img = int(round(len(texts) * r / (1 - r))) if 0 < r < 1 else 0
    print(f"目标图文 {n_img} 条（比例 {r}）", flush=True)
    convs_i, imgs_i = sample_images(args.image_data, n_img, args.seed)
    print(f"  抽到图文 {len(convs_i)} 条（{time.time()-t0:.0f}s）", flush=True)

    # 各自打散
    rng = np.random.default_rng(args.seed)
    pt = rng.permutation(len(texts))
    texts = [texts[i] for i in pt]
    pi = rng.permutation(len(convs_i))
    convs_i = [convs_i[i] for i in pi]
    imgs_i = [imgs_i[i] for i in pi]

    # 交错：每 k 条文本插 1 条图文（k = ceil(文本/图文)），余下的文本收尾
    n_i = len(convs_i)
    n_t = len(texts)
    k = max(1, round(n_t / n_i)) if n_i else n_t

    def rows():
        """惰性产出 (conversations, image_bytes)：先交错，再补尾部剩余文本。"""
        ti = ii = 0
        while ii < n_i and ti < n_t:
            for _ in range(k):
                if ti >= n_t:
                    break
                yield texts[ti], None
                ti += 1
            yield convs_i[ii], imgs_i[ii]
            ii += 1
        while ti < n_t:
            yield texts[ti], None
            ti += 1
        while ii < n_i:
            yield convs_i[ii], imgs_i[ii]
            ii += 1

    schema = pa.schema([("conversations", pa.large_string()),
                        ("image_bytes", pa.large_binary())])
    writer = pq.ParquetWriter(args.out, schema)
    buf_c, buf_b, n_out, n_img_out = [], [], 0, 0
    for c, b in rows():
        buf_c.append(c); buf_b.append(b); n_out += 1
        if b is not None:
            n_img_out += 1
        if len(buf_c) >= args.chunk:
            writer.write_table(pa.table({"conversations": pa.array(buf_c, type=pa.large_string()),
                                         "image_bytes": pa.array(buf_b, type=pa.large_binary())}))
            buf_c, buf_b = [], []
            print(f"  写出 {n_out} 条（{time.time()-t0:.0f}s）", flush=True)
    if buf_c:
        writer.write_table(pa.table({"conversations": pa.array(buf_c, type=pa.large_string()),
                                     "image_bytes": pa.array(buf_b, type=pa.large_binary())}))
    writer.close()
    print(f"完成：{n_out} 条（图文 {n_img_out}，占 {n_img_out/max(n_out,1)*100:.1f}%）→ {args.out}"
          f"（{time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
