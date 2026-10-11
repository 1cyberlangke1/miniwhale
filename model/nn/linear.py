"""Linear 与全局 dtype 管理。

按参考实现做法：权重 dtype 由**全局 default_dtype** 决定（默认 bf16），
只有少数位置显式指定（如 compressor 的 fp32、wo_a 的 bf16）。
训练时通过 `set_dtype()` 切换全局精度，而非逐层硬编码。
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn

# 权重默认精度由 torch 全局 default dtype 决定（同样结构机制）。
# 训练基底 bf16；用 set_dtype() 临时覆盖（如 mHC 的 fp32）。
default_dtype = torch.bfloat16


@contextlib.contextmanager
def set_dtype(dtype):
    """临时覆盖全局默认 dtype，退出时恢复（即使抛异常）。

    输入：dtype，如 torch.float32 / torch.bfloat16。
    输出：上下文管理器。
    预期行为：覆盖 torch 默认 dtype；作用域内新建的 Linear 继承它。
    """
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def set_default_dtype(dtype):
    """设定全局权重默认精度（模型构造前调用）。"""
    global default_dtype
    default_dtype = dtype
    torch.set_default_dtype(dtype)


def _current_dtype():
    """当前权重默认精度：跟随 torch 全局 default dtype。"""
    return torch.get_default_dtype()


class Linear(nn.Module):
    """线性层：权重 dtype 由 dtype 参数或全局 default_dtype 决定。

    输入：x [..., in_features]。
    输出：[..., out_features]。
    预期行为：dtype=None 时用全局 default_dtype；权重按正态初始化（std=initializer_range）；
              不做 dtype 转换（由上层 autocast 负责）。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None,
                 initializer_range: float = 0.02):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initializer_range = initializer_range
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or _current_dtype()))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype or _current_dtype()))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """正态初始化权重（std=initializer_range），bias 置零。

        输入：无。
        输出：无（原地）。
        预期行为：fp32 下采样再转回权重 dtype，保证 bf16 权重也有正确分布。
        """
        w = torch.randn(self.out_features, self.in_features, dtype=torch.float32) * self.initializer_range
        self.weight.copy_(w.to(self.weight.dtype))
        if self.bias is not None:
            self.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)
