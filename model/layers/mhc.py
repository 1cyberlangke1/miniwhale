"""Single-Pass mHC（流形约束超连接）。

残差流以 hc_mult 份并行副本承载。
- `hc_pre`：把 hc 份折叠成 1 份，喂给子层
- `hc_post`：把子层输出展开回 hc 份，并用 comb 把残差混进来
- comb 经 Sinkhorn 归一成双随机矩阵

Single-Pass 的错位：某个子层用到的 pre_mix 由**上一层**产出（本模块只负责产出供下一层用）。
全程 fp32 —— mHC 系数在 bf16 下会 NaN。
"""

from __future__ import annotations

import torch
from torch import nn


def sinkhorn_normalize(x: torch.Tensor, repeat: int = 20, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn 双随机归一化。

    输入：x [..., n, n]，任意浮点。
    输出：同形状，行列交替归一后的双随机矩阵。
    预期行为：先 softmax(-1)+eps，再 (列归一, 行归一) 交替 repeat 次，每步分母 +eps。
              注：迭代**本质串行**（每步依赖上一步），无法向量化；算子数靠 torch.compile 融合。
    """
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def split_mixes(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                hc_mult: int, sinkhorn_iters: int, eps: float):
    """把一次投影拆成 pre / post / comb 三组系数。

    输入：mixes [..., mix_hc]，mix_hc=(2+hc_mult)*hc_mult；hc_scale [3]；hc_base [mix_hc]。
    输出：(pre [..., hc_mult], post [..., hc_mult], comb [..., hc_mult, hc_mult])。
    预期行为：scale 按 [hc, hc, hc*hc] 展开；pre=sigmoid+eps；post=2*sigmoid；
              comb=Sinkhorn(softmax(-1)+eps)。
    """
    *lead, mix_hc = mixes.shape
    hc = hc_mult
    scale = torch.cat([
        hc_scale[0].expand(hc), hc_scale[1].expand(hc), hc_scale[2].expand(hc * hc),
    ])
    m = mixes * scale + hc_base
    pre = m[..., :hc].sigmoid() + eps
    post = 2.0 * m[..., hc:2 * hc].sigmoid()
    comb = m[..., 2 * hc:].view(*lead, hc, hc)
    comb = sinkhorn_normalize(comb, repeat=sinkhorn_iters, eps=eps)
    return pre, post, comb


class HyperConnection(nn.Module):
    """一层 mHC 的系数投影与折叠/展开。

    输入：x [B, L, hc_mult, dim]；pre_mix [B, L, hc_mult]（由上一层产出，折叠用）。
    输出：hc_mixes 返回 (pre, post, comb)；hc_pre 返回 [B, L, dim]；
          hc_post 返回 [B, L, hc_mult, dim]。
    预期行为：系数投影以 fp32 计算；参数存 fp32；残差混合全程 fp32。
    """

    def __init__(self, text_cfg):
        super().__init__()
        self.hc_mult = text_cfg.hc_mult
        self.hc_eps = text_cfg.hc_eps
        # 参考实现 hc_mixes 的 rsqrt 用 norm_eps（不是 hc_eps）——两处 eps 不同，须分开存
        self.norm_eps = text_cfg.rms_norm_eps
        self.sinkhorn_iters = text_cfg.hc_sinkhorn_iters
        dim = text_cfg.hidden_size
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        self.mix_hc = mix_hc

        # 系数一律存 fp32（参考实现 with set_dtype(torch.float32)）——bf16 会 NaN
        for name in ("attn", "ffn"):
            setattr(self, f"hc_{name}_fn",
                    nn.Parameter(torch.empty(mix_hc, self.hc_mult * dim, dtype=torch.float32)))
            setattr(self, f"hc_{name}_scale", nn.Parameter(torch.empty(3, dtype=torch.float32)))
            setattr(self, f"hc_{name}_base", nn.Parameter(torch.empty(mix_hc, dtype=torch.float32)))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """初始化系数投影（正态 std=0.02）、base（零）、scale（1）。

        输入：无。
        输出：无（原地）。
        预期行为：scale 必须非零——否则 `mixes*scale` 对 fn 的梯度为 0（死参数）。
        """
        for name in ("attn", "ffn"):
            fn = getattr(self, f"hc_{name}_fn")
            fn.copy_(torch.randn_like(fn) * 0.02)
            getattr(self, f"hc_{name}_base").zero_()
            getattr(self, f"hc_{name}_scale").fill_(1.0)

    def hc_mixes(self, x: torch.Tensor, which: str):
        """从残差流投影出 pre/post/comb 系数。

        输入：x [B, L, hc_mult, dim]；which ∈ {"attn","ffn"}。
        输出：(pre, post, comb)。
        预期行为：flatten 后做 RMS 归一（整条 hc*dim 流一个统计量），投影，再拆分。
        """
        hc_fn = getattr(self, f"hc_{which}_fn")
        hc_scale = getattr(self, f"hc_{which}_scale")
        hc_base = getattr(self, f"hc_{which}_base")
        flat = x.flatten(2).float()
        # 参考实现 hc_mixes：rsqrt 用 norm_eps（不是 hc_eps），且无 nan_to_num/clamp——保持一致
        rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = torch.nn.functional.linear(flat, hc_fn.float()) * rsqrt
        return split_mixes(mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """折叠 hc 副本成一份。x [B,L,hc,dim] × pre_mix [B,L,hc] → [B,L,dim]。

        用 einsum 做沿 hc 的加权求和：与参考实现 `(pre_mix.unsqueeze(-1)*x).sum(2)` 数学等价，
        但不会先造 [B,L,hc,dim] 的中间张量（同一求和顺序，逐位一致）。
        """
        return torch.einsum("blh,blhd->bld", pre_mix, x.float()).to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """展开子层输出并混入残差。x [B,L,dim] → [B,L,hc,dim]。

        `comb` 混入项是沿 hc 的收缩（sum over k），用 einsum 直接算：
        与参考实现 `torch.sum(comb.unsqueeze(-1)*residual.unsqueeze(-2), dim=2)` 数学等价，
        但避免 [B,L,hc,hc,dim] 的大中间张量与 permute（训练提速的主杠杆）。
        """
        mixed = torch.einsum("blij,bljd->blid", comb, residual.float())
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2) + mixed
        return y.type_as(x)
