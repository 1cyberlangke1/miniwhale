"""图像预处理 —— 照参考实现 的 image_processor（纯 torch/PIL/numpy，无外部依赖）。

一张图变成：ViT 的 `n_vit_h × n_vit_w` patch 网格，以及 Aligner 3×3 降采样后的
`n_llm_h × n_llm_w` token 网格。后者在 LLM 里展开成

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

这些位置在 `input_ids` 里**都携带同一个 `image_token_id`**，仅靠 `token_types` 区分类型
（参考实现机制）。IMAGE 槽按阅读顺序填入 Aligner 的行。
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps

# token 类型（参考实现 image_processor）：TEXT=-1，其余为非负
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


@dataclass
class ImageInput:
    """一张图在序列中的位置与 patch 网格（按参考实现 ImageInput）。"""

    start: int              # 该图像 span 在 tokens 里的起始下标
    patches: torch.Tensor   # [n_vit_h*n_vit_w, 3, p, p]
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor     # [num_image_tokens] 各位置类型（IMAGE_START/IMAGE/...）


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """一个 token 网格占用的序列长度（含 start/end/newline）。"""
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """给定像素尺寸，Aligner 产出的 token 网格 (n_llm_h, n_llm_w)。"""
    return (math.ceil((best_height // patch_size) / downsample_ratio),
            math.ceil((best_width // patch_size) / downsample_ratio))


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """保持宽高比、使 token 网格仍 <= max_n_token 的最大像素尺寸。"""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:      # 极高的图：塌成一列
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:      # 极宽的图：塌成一行
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return (math.floor(height * beta / patch_size) * patch_size,
            math.floor(width * beta / patch_size) * patch_size)


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """缩小像素尺寸，直到图像最多占 max_n_token 个 LLM token。"""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, patch_size: int, downsample_ratio: int,
                    max_n_token: int, min_pixels: int = 0, max_wh_ratio: float | None = None):
    """给定原图尺寸的 resize 计划（纯函数，只依赖参数）。

    输入：原宽高；patch_size；downsample_ratio；max_n_token；min_pixels；max_wh_ratio。
    输出：(n_llm_h, n_llm_w, best_height, best_width)。
    预期行为：超宽裁剪、过小放大、对齐到 patch 整数倍、再按 token 上限收缩。
    """
    p = patch_size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    return safe_resize(height, width, best_height, best_width, p, downsample_ratio, max_n_token)


def load_image(image: Image.Image, patch_size: int, downsample_ratio: int, max_n_token: int,
               min_pixels: int = 0, max_wh_ratio: float | None = None):
    """把一张 PIL 图转成 ViT patches。

    输入：PIL Image；patch_size；downsample_ratio；max_n_token；min_pixels；max_wh_ratio。
    输出：(patches [n_vit_h*n_vit_w, 3, p, p], n_vit_h, n_vit_w, n_llm_h, n_llm_w)。
    预期行为：按 plan_image_grid 定尺寸 → pad/resize → 归一化到 [-1,1] → 切 patch。
    """
    p = patch_size
    image = image.convert("RGB")
    n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
        image.width, image.height, p, downsample_ratio, max_n_token, min_pixels, max_wh_ratio)
    n_vit_h, n_vit_w = best_height // p, best_width // p
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(
        n_vit_h * n_vit_w, 3, p, p)
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """默认布局：Aligner 网格按阅读顺序，每行末一个 IMAGE_NEW_LINE。"""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def load_image_from_bytes(raw: bytes, patch_size: int, downsample_ratio: int, max_n_token: int,
                          min_pixels: int = 0, max_wh_ratio: float | None = None):
    """从字节流加载并转 patch（按参考实现 load_image 的字节入口）。"""
    with Image.open(io.BytesIO(raw)) as source:
        return load_image(source, patch_size, downsample_ratio, max_n_token, min_pixels, max_wh_ratio)


__all__ = [
    "TEXT", "IMAGE_START", "IMAGE", "IMAGE_NEW_LINE", "IMAGE_END", "ImageInput",
    "num_image_tokens", "llm_grid", "solve_resize_ratio", "safe_resize",
    "plan_image_grid", "load_image", "load_image_from_bytes", "image_token_types",
]
