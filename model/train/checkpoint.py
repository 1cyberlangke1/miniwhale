"""checkpoint 保存 / 续训加载（五个 trainer 共用）。

文件名约定：`<prefix>_e<epoch>_s<step>.pth`（中间档），最终产物 `<prefix>.pth`。
中间档存**完整训练状态**（模型 + 三优化器动量 + 步数），续训才能无缝接上。
"""

from __future__ import annotations

import os
import re

import torch

from model.train import ddp


def save_checkpoint(save_dir: str, prefix: str, epoch: int, step: int, model, muon, adamw,
                    sinkhorn, keep_last_n: int = 3) -> str:
    """保存完整训练状态并清理旧的中间档。

    输入：save_dir；prefix（如 "sft"）；epoch；step（已 +1）；三个优化器；keep_last_n（0=全留）。
    输出：保存路径。
    预期行为：仅 rank0 落盘（其余 rank 直接返回空串）；文件存 model/muon/adamw/sinkhorn/step。
              清理时用正则精确匹配 `<prefix>_e\\d+_s\\d+\\.pth`，**不会**误删最终产物
              `<prefix>.pth`（曾用 startswith 匹配导致产物被删）。
    """
    if not ddp.is_main():
        return ""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_e{epoch}_s{step}.pth")
    torch.save({
        "model": ddp.unwrap(model).state_dict(),
        "muon": muon.state_dict(),
        "adamw": adamw.state_dict(),
        "sinkhorn": sinkhorn.state_dict(),
        "step": step,
    }, path)
    if keep_last_n:
        pat = re.compile(rf"^{re.escape(prefix)}_e\d+_s\d+\.pth$")
        cks = sorted(
            (os.path.join(save_dir, f) for f in os.listdir(save_dir) if pat.match(f)),
            key=os.path.getmtime)
        for old in cks[:-keep_last_n]:
            os.remove(old)
    return path


def load_resume_ckpt(path: str, model, device) -> dict | None:
    """加载续训 checkpoint 到模型，返回待回填的优化器状态。

    输入：path；model；device。
    输出：{"muon","adamw","sinkhorn","step"} 或 None（旧格式纯 state_dict 时）。
    预期行为：新格式（含 "model" 键）载入 model 并返回优化器状态；
              旧格式（纯 state_dict）只载入权重，步数从文件名 `_s<step>.pth` 解析，
              优化器状态返回 None（调用方按「无状态」处理）。
    """
    ck = torch.load(path, map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        model.load_state_dict(ck["model"])
        return {"muon": ck.get("muon"), "adamw": ck.get("adamw"),
                "sinkhorn": ck.get("sinkhorn"), "step": int(ck.get("step", 0))}
    model.load_state_dict(ck)
    m = re.search(r"_s(\d+)\.pth$", os.path.basename(path))
    return {"muon": None, "adamw": None, "sinkhorn": None,
            "step": int(m.group(1)) if m else 0}
