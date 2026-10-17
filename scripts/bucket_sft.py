"""按序列长度分桶，输出可直接训练的 parquet。

为什么分桶：桶内 pad 到桶上界 → 形状固定 → torch.compile 每桶只编一次图；桶越小 pad 越少、
batch 可开越大。**桶按长度升序链式续训**（小桶 → 大桶，每桶在上一个桶的产物权重上继续训，
不是各自从统一起点独立训）。

**只做分桶，不做混合**（文本/图文的比例、交错由上游数据准备好，本脚本原样透传）。

输入：
  --data       输入 parquet（列 conversations + image_bytes；文本行 image_bytes 为 null）或 jsonl
  --buckets    桶上界（升序整数），默认 512 768 1024
  --out_dir    输出目录（默认本地数据目录下的 sft_buckets）
  --chunk / --limit / --workers / --seed
输出：
  out_dir/bucket_<N>.parquet   每桶一个（列 conversations + image_bytes，原样保留）
  out_dir/meta.json            每桶条数
预期行为：并行进程分块扫源、**批量** tokenize 求长度（不逐行 apply_chat_template）、
          每任务写独立 part 文件；扫完按桶合并、桶内 shuffle。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# 每张图在 LLM 侧占的视觉 span token：1(start) + (w+1)*h(newline) + 1(end)。
# 本仓库图 256×256 / patch 14 / downsample 3 → 6×6 → 1 + 7*6 + 1 = 44。
VISION_SPAN = 44

_TOK = None          # worker 进程内缓存（initializer 里建一次）


def _ids_of(out):
    """从 apply_chat_template 返回值取 input_ids（兼容 dict / list）。"""
    return out["input_ids"] if hasattr(out, "keys") else out


def _render(convs):
    """渲染单条对话，**保留工具调用**（照 minimind create_chat_prompt）。

    输入：conversations（list[dict]，可含 tools/tool_calls/reasoning_content）。
    输出：input_ids（list[int]）。
    预期行为：解析 system 的 `tools`（字符串→对象）作为模板参数、解析 `tool_calls`；
              保留全部字段。长度须与训练时 model/train/data.render_chat 一致。
    """
    tools = None
    messages = []
    for m in convs:
        m = dict(m)
        if m.get("role") == "system" and m.get("tools"):
            tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
        if m.get("tool_calls") and isinstance(m["tool_calls"], str):
            m["tool_calls"] = json.loads(m["tool_calls"])
        messages.append(m)
    out = _TOK.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                   tools=tools)
    ids = _ids_of(out)
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return ids


def _lengths(convs_list, n_img_list):
    """批量算长度：chat template token 数 + 每图视觉 span。

    输入：conversations 列表；每条的图数列表。
    输出：长度列表。
    预期行为：逐条渲染（与训练口径一致）；工具调用样本一并计入。
    """
    return [len(_render(c)) + n * VISION_SPAN
            for c, n in zip(convs_list, n_img_list)]


def _count_img(convs):
    """统计一条对话里 `<image>` 占位符个数。"""
    return sum(str(t.get("content", "")).count("<image>") for t in convs)


def _bucket_of(n, buckets):
    """n 落入的桶上界（超过最大桶归入最大桶）。"""
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


def _win_init():
    """worker 初始化：加载一次 tokenizer。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.tokenizer import load_tokenizer

    global _TOK
    _TOK = load_tokenizer()


def _write_part(parts_dir, tag, rows):
    """把 rows=[(convs_str, img_bytes), ...] 写成一个 part parquet。"""
    pq.write_table(pa.table({
        "conversations": pa.array([r[0] for r in rows], type=pa.large_string()),
        "image_bytes": pa.array([r[1] for r in rows], type=pa.large_binary()),
    }), os.path.join(parts_dir, f"{tag}.parquet"))


def _bucket_rows(cs, convs, nimg, imgs, buckets, by_bucket):
    """把一批行按桶归入 by_bucket（bucket -> [(convs_str, img_bytes), ...]）。"""
    for c, im, L in zip(cs, imgs, _lengths(convs, nimg)):
        by_bucket.setdefault(_bucket_of(L, buckets), []).append((c, im))


def _run_jsonl(task):
    """worker：处理 jsonl 的一段 byte 区间（文本行 image_bytes=None），按桶写 part。"""
    path, s, e, buckets, chunk, parts_dir, tag = task
    by_bucket = {}
    n = 0
    cs, convs, nimg, imgs = [], [], [], []
    with open(path, "rb") as f:
        f.seek(s)
        while f.tell() < e:
            raw = f.readline()
            if not raw:
                break
            try:
                c = json.loads(raw.decode("utf-8"))["conversations"]
            except Exception:
                continue
            cs.append(json.dumps(c, ensure_ascii=False))
            convs.append(c); nimg.append(0); imgs.append(None)
            if len(convs) >= chunk:
                _bucket_rows(cs, convs, nimg, imgs, buckets, by_bucket)
                n += len(convs)
                cs, convs, nimg, imgs = [], [], [], []
    if convs:
        _bucket_rows(cs, convs, nimg, imgs, buckets, by_bucket)
        n += len(convs)
    for b, rows in by_bucket.items():
        _write_part(parts_dir, f"b{b}_{tag}", rows)
    return n


def _run_parquet(task):
    """worker：处理 parquet 的一段 row group 区间，按桶写 part。"""
    path, g0, g1, buckets, chunk, parts_dir, tag = task
    by_bucket = {}
    f = pq.ParquetFile(path)
    n = 0
    cs, convs, nimg, imgs = [], [], [], []
    for g in range(g0, g1):
        t = f.read_row_group(g, columns=["conversations", "image_bytes"])
        for c, b in zip(t.column("conversations").to_pylist(),
                        t.column("image_bytes").to_pylist()):
            cc = json.loads(c)
            cs.append(c); convs.append(cc); nimg.append(_count_img(cc)); imgs.append(b)
            if len(convs) >= chunk:
                _bucket_rows(cs, convs, nimg, imgs, buckets, by_bucket)
                n += len(convs)
                cs, convs, nimg, imgs = [], [], [], []
    if convs:
        _bucket_rows(cs, convs, nimg, imgs, buckets, by_bucket)
        n += len(convs)
    for b, rows in by_bucket.items():
        _write_part(parts_dir, f"b{b}_{tag}", rows)
    return n


def _run_task(task):
    """并行调度入口（顶层函数，可 pickle）：分派到 jsonl / parquet worker。"""
    kind, payload = task
    return _run_jsonl(payload) if kind == "jsonl" else _run_parquet(payload)


def _byte_blocks(path, rows_per_block):
    """把 jsonl 按行数切成 byte 区间（只扫换行，不解析）。"""
    blocks, start, cnt = [], 0, 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            cnt += 1
            if cnt % rows_per_block == 0:
                blocks.append((start, f.tell()))
                start = f.tell()
    size = os.path.getsize(path)
    if start < size:
        blocks.append((start, size + 1))
    return blocks


def main():
    ap = argparse.ArgumentParser(description="按序列长度分桶（不做混合）")
    ap.add_argument("--data", required=True, help="输入 parquet 或 jsonl")
    ap.add_argument("--buckets", type=int, nargs="+", default=[512, 768, 1024])
    ap.add_argument("--out_dir", default="dataset/sft_buckets")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=2000, help="批量 tokenize 的块大小")
    ap.add_argument("--limit", type=int, default=0, help="最多处理行数（0=全部）")
    ap.add_argument("--workers", type=int, default=4,
                    help="并行进程数（默认 4；显式传 0 = os.cpu_count()）")
    args = ap.parse_args()

    from concurrent.futures import ProcessPoolExecutor

    is_jsonl = args.data.lower().endswith((".jsonl", ".json"))
    n_workers = args.workers or os.cpu_count() or 4
    t0 = time.time()
    parts = os.path.join(args.out_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    for f in glob.glob(os.path.join(parts, "*.parquet")):
        os.remove(f)

    tasks, tag = [], 0
    if is_jsonl:
        rows_per_block = max(args.chunk, 20000)
        blocks = _byte_blocks(args.data, rows_per_block)
        if args.limit:
            blocks = blocks[: max(1, args.limit // rows_per_block)]
        for s, e in blocks:
            tasks.append(("jsonl", (args.data, s, e, args.buckets, args.chunk, parts, tag)))
            tag += 1
    else:
        f = pq.ParquetFile(args.data)
        nrg = f.metadata.num_row_groups
        step = max(1, nrg // (n_workers * 4))
        for g in range(0, nrg, step):
            tasks.append(("parquet", (args.data, g, min(g + step, nrg),
                                      args.buckets, args.chunk, parts, tag)))
            tag += 1
    print(f"并行 {n_workers} 进程，任务块 {len(tasks)}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_win_init) as ex:
        done = 0
        for n in ex.map(_run_task, tasks):
            done += n
            print(f"  已处理 {done} 行（{time.time()-t0:.0f}s）", flush=True)
    print(f"扫描完成 {done} 行（{time.time()-t0:.0f}s）", flush=True)

    # 合并：每桶 concat part + shuffle
    rng = np.random.default_rng(args.seed)
    meta = {"buckets": args.buckets, "per_bucket": {}}
    for b in args.buckets:
        ps = sorted(glob.glob(os.path.join(parts, f"b{b}_*.parquet")))
        if not ps:
            meta["per_bucket"][str(b)] = 0
            print(f"  bucket_{b}: 0 条", flush=True)
            continue
        tbl = pa.concat_tables([pq.read_table(p) for p in ps])
        n = len(tbl)
        perm = rng.permutation(n)
        tbl = tbl.take(pa.array(perm.tolist()))
        pq.write_table(tbl, os.path.join(args.out_dir, f"bucket_{b}.parquet"))
        meta["per_bucket"][str(b)] = n
        print(f"  bucket_{b}: {n} 条", flush=True)

    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"完成 → {args.out_dir}（总 {time.time()-t0:.0f}s）", flush=True)


if __name__ == "__main__":
    main()
