"""从完整版图文 QA parquet 抽取纯图文 mini 子集（剔除纯文本占位行）。

输入：--src 完整版 parquet（列 conversations + image_bytes）；--target 目标行数。
输出：--dst 纯图文 mini（只保留 conversations 含 `<image>` 的行，按比例随机抽样到 --target 行）。
预期行为：按 row group 流式读，不整表载入内存；先写 `.tmp` 再原子替换，避免中途失败留下半个文件。
          完整版含约 8% 纯文本占位行（图像列填 8×8 黑图、无 `<image>` 占位符，用于保留基础语言
          能力）；本脚本把它们全部剔除，产出 100% 图文，供 ViT 自回归微调阶段使用。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def count_image_rows(f: pq.ParquetFile) -> int:
    """统计 parquet 里 conversations 含 '<image>' 的行数（纯图文）。

    输入：已打开的 ParquetFile。
    输出：含占位符的行数。
    预期行为：逐 row group 做子串匹配求和，不把 conversations 全量拉到 Python 层。
    """
    total = 0
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g, columns=["conversations"])
        m = pc.match_substring(t.column("conversations"), "<image>")
        total += pc.sum(pc.cast(m, "int64")).as_py()
    return total


def main():
    ap = argparse.ArgumentParser(description="抽取纯图文 mini 子集")
    ap.add_argument("--src", default="dataset/sft_i2t.parquet", help="完整版 parquet")
    ap.add_argument("--dst", default="dataset/sft_i2t_mini.parquet", help="输出 mini parquet")
    ap.add_argument("--target", type=int, default=905718, help="目标行数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    f = pq.ParquetFile(args.src)
    total = count_image_rows(f)
    print(f"完整版纯图文行数 = {total}（{time.time() - t0:.0f}s）", flush=True)
    frac = min(args.target / total, 1.0)
    print(f"抽样比例 = {frac:.4f}（目标 {args.target}）", flush=True)

    rng = np.random.default_rng(args.seed)
    tmp = args.dst + ".tmp"
    writer = None
    written = 0
    for g in range(f.metadata.num_row_groups):
        t = f.read_row_group(g)
        t = t.filter(pc.match_substring(t.column("conversations"), "<image>"))
        if len(t) == 0:
            continue
        t = t.filter(pa.array(rng.random(len(t)) < frac))
        if len(t) == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(tmp, f.schema_arrow)
        writer.write_table(t)
        written += len(t)
        if g % 100 == 0:
            print(f"  rg {g}/{f.metadata.num_row_groups}  written={written}  "
                  f"{time.time() - t0:.0f}s", flush=True)
    writer.close()
    print(f"写出 {written} 行（{time.time() - t0:.0f}s）", flush=True)
    os.replace(tmp, args.dst)
    print(f"已覆盖 {args.dst}", flush=True)


if __name__ == "__main__":
    main()
