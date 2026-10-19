"""推理引擎（加载权重 + 增量解码 + DSpark 投机解码）。

与训练代码分离：本包只读模型，不改训练路径。默认 eager（逐位一致），
`--compile` 时按解码路径编译（数值语义不变，见 `inference/compile.py`）。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.engine import Engine, GenerationConfig
from inference.spec import SpeculativeConfig, generate_speculative

__all__ = ["Engine", "GenerationConfig", "SpeculativeConfig", "generate_speculative"]
