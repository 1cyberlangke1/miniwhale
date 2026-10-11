"""分词器加载。

使用 6400 词表（随包分发，见 ./tokenizer/）。
"""

from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

# 词表随源码包分发，保证包自包含
_TOKENIZER_DIR = Path(__file__).with_name("tokenizer")


def load_tokenizer(path: Path | str | None = None) -> PreTrainedTokenizerFast:
    """加载 6400 词表分词器。

    输入：可选 path（含 tokenizer.json 的目录），默认包内词表目录。
    输出：PreTrainedTokenizerFast 实例。
    预期行为：编码/解码往返一致；pad/bos/eos 可用。
    """
    p = Path(path) if path is not None else _TOKENIZER_DIR
    return AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)


if __name__ == "__main__":
    tok = load_tokenizer()
    s = "白日依山尽，黄河入海流。"
    ids = tok(s).input_ids
    print("vocab", len(tok), "| ids", ids, "| roundtrip", tok.decode(ids) == s)
