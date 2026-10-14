"""训练日志落盘工具（Python 内直写文件，不经 shell 管道）。

为什么需要：shell 的 `python ... | grep ... | tee log` 里，**grep 会块缓冲**——
即使 python 端 `flush=True`，输出也卡在 grep 的 4KB 缓冲里，训练日志迟迟不落盘
（表现为日志文件长时间 0 字节）。正解是**在 Python 里直接写文件**，不依赖管道。
"""

from __future__ import annotations

import sys


class _Tee:
    """把写入转发到原 stdout 与日志文件（逐行 flush，实时可见）。

    输入：write(s) 字符串；flush() 无参。
    输出：无（副作用：同时写 sys.__stdout__ 与文件）。
    预期行为：绕过被替换的 sys.stdout，直接写原始 stdout，避免递归。
    """

    def __init__(self, fh):
        self._fh = fh

    def write(self, s):
        sys.__stdout__.write(s)
        self._fh.write(s)
        self._fh.flush()

    def flush(self):
        sys.__stdout__.flush()
        self._fh.flush()


def setup_log_file(path: str | None) -> None:
    """把 stdout 重定向为「控制台 + 文件」双写。

    输入：path 日志文件路径（None 或空 = 不启用，保持原样）。
    输出：无（副作用：替换 sys.stdout 为 _Tee）。
    预期行为：文件以**覆盖** + 行缓冲打开（每次训练调用 = 一条新曲线，不与旧 run 混在同一文件，
              否则前端会把两次 run 的点连成一条误导曲线）；每行立即落盘；不捕获 stderr。
    """
    if not path:
        return
    fh = open(path, "w", encoding="utf-8", buffering=1)  # 覆盖 + 行缓冲：换行即 flush
    sys.stdout = _Tee(fh)
