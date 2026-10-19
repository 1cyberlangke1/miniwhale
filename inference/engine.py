"""推理引擎：权重加载 + prefill + 增量 decode（KV 缓存复用）。

设计要点：
- **prefill 一次，decode 每步只喂 1 个 token**（走 `Model.forward(start_pos=...)` 的增量路径）。
- 每段对话开始前调 `reset_cache()`，否则新序列会读到上一段的 KV。
- Engram 的 n-gram 哈希在 decode 时要用**真实历史**（`past_ids`），否则哈希退化。
- 默认 eager；`compile=True` 时把 decode 相关子模块编译（见 `inference/compile.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from model.config import ModelConfig
from model.model import Model


@dataclass
class GenerationConfig:
    """生成超参（默认值与参考实现一致）。"""

    max_new_tokens: int = 128
    temperature: float = 0.6          # 0=贪心；参考实现 generate.py 默认 0.6
    eos_token_id: int | None = None
    use_dspark: bool = False          # 是否启用 DSpark 投机解码
    confidence_threshold: float = 0.0  # DSpark 置信度截断阈值（0=不截断）


def _disable_fake_quant():
    """把假量化换成恒等（推理专用）。

    输入：无。
    输出：无。
    预期行为：替换 `model.nn.quant` 与各层模块**已导入**的同名符号为恒等函数。
              假量化是训练期 QAT（前向量化-反量化模拟部署精度、反向直通）：
              推理若保留，每一处都要跑「逐块 amax → scale → clamp → cast → 反量化」，
              在 [1,1,64] 的小张量上是十几次纯 launch 开销（实测 ~7ms/步），
              且对已经训好的 bf16 权重没有任何数值收益。
              **逐层模块持有自己的引用**（`from model.nn.quant import ...`），
              故必须同时改模块内的符号，只改 quant 模块自身无效。
    """
    import model.nn.quant as _q

    def _id8(x, block_size: int = 32, pow2: bool = True):
        return x

    def _id4(x, block_size: int = 32):
        return x

    _q.fake_quant_fp8 = _id8
    _q.fake_quant_fp4 = _id4
    _q.fake_quant_compressed_kv = _id4
    for name in ("model.layers.attention", "model.layers.indexer",
                 "model.layers.compressor", "model.layers.dspark"):
        try:
            mod = __import__(name, fromlist=["_"])
        except ImportError:
            continue
        if hasattr(mod, "fake_quant_fp8"):
            mod.fake_quant_fp8 = _id8
        if hasattr(mod, "fake_quant_fp4"):
            mod.fake_quant_fp4 = _id4
        if hasattr(mod, "fake_quant_compressed_kv"):
            mod.fake_quant_compressed_kv = _id4


class _DecodeGraphs:
    """按 `emit_ats` 组合捕获的 CUDA Graph（每步只重放，Python 不参与）。

    输入：model（eval 后的 Model）；device；need（Engram 回看的 token 数）；
          keys（(示例位置, 每层 emit 下标元组) 的列表）。
    输出：`step()` 返回 [1,V] logits（固定地址，重放后覆盖写）。
    预期行为：捕获前先用真实形状在旁路 stream 上热身，再 `torch.cuda.graph` 录制。
              重放时只把新 token / 位置 / 历史写进**静态输入 buffer**，形状与控制流
              全部不变 → 不会重编译，也不会有逐算子 Python 开销。
    """

    def __init__(self, model, device: str, need: int, keys, steps: int = 1):
        self.model = model
        self.device = device
        self.need = need
        self.steps = steps
        self.ids = torch.zeros(1, steps, dtype=torch.long, device=device)
        self.pos = torch.zeros((), dtype=torch.long, device=device)
        self.past = torch.zeros(1, need, dtype=torch.long, device=device)
        self.graphs: dict[tuple, tuple] = {}
        for pos_value, key in keys:
            self.graphs[key] = self._capture(pos_value, key)

    def _capture(self, pos_value: int, emit_ats):
        """录制一张图。

        输入：pos_value（与该分支一致的示例位置，捕获时写进 pos buffer）；
              emit_ats（每层本步凑满压缩组的 token 下标元组）。
        输出：(graph, logits)。
        预期行为：旁路 stream 跑两次热身（让一切惰性初始化完成），再录制成图。
                  **pos 必须是该分支真实会出现的值**：图内的 `pos+1-ratio` 等位置
                  运算会在录制时求值，用不相符的值会越界（如 ratio=2 的压缩步用 pos=0
                  会得到 -1）。
        """
        ac = "cuda" if self.device.startswith("cuda") else "cpu"
        self.pos.fill_(pos_value)
        with torch.no_grad(), torch.autocast(device_type=ac, dtype=torch.bfloat16):
            # 热身与捕获必须用**同一条 stream**：torch.compile 的守卫会记录当前 stream，
            # 换流会在捕获期触发一次新编译（dynamo 明确禁止在捕获中编译）。
            side = torch.cuda.Stream(device=self.device)
            side.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.model.forward_decode(self.ids, self.pos, self.past, emit_ats)
            side.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=side):
                logits, _ = self.model.forward_decode(
                    self.ids, self.pos, self.past, emit_ats)
            torch.cuda.current_stream(self.device).wait_stream(side)
            # 每张图有自己的 `_last_main_hidden` buffer；必须在**本图捕获后**取引用，
            # 否则多张图互相覆盖模块属性，读到的会是最晚捕获那张图的（陈旧）buffer。
            hidden = self.model._last_main_hidden
        return g, logits, hidden

    def step(self, token, start_pos: int, past, emit_ats):
        """写输入 → 重放 → 返回 logits 与 target 层 hidden。

        输入：token [1,steps]；start_pos（int）；past [1,need]；emit_ats（同捕获时）。
        输出：(logits [1,steps,V], hidden [1,steps,2*dim])，均为图内固定 buffer。
        """
        g, logits, hidden = self.graphs[emit_ats]
        self.ids.copy_(token)
        self.pos.fill_(start_pos)
        self.past.copy_(past)
        g.replay()
        return logits, hidden


class Engine:
    """模型推理封装（单条序列 / 单 batch）。"""

    def __init__(self, ckpt: str, device: str = "cuda:0", compile: bool = False,
                 dtype: torch.dtype = torch.bfloat16, quantize: bool = False,
                 tokenizer=None, cuda_graph: bool = False):
        """加载权重并准备推理。

        输入：ckpt（权重路径）；device；compile（是否 torch.compile 解码路径）；
              dtype（autocast 精度，默认 bf16，与训练一致）；
              quantize（推理时是否保留假量化。**默认 False**：假量化是训练期 QAT 手段，
              推理做「量化→反量化」往返只会多花 kernel 时间，不改变数值精度层级）；
              tokenizer（HF tokenizer；**Engram 哈希依赖它**，不传则 Engram 整条路径关闭，
              与训练时的模型不是同一个函数）；
              cuda_graph（是否用 CUDA Graph 重放单步解码。形状与控制流全程静态，
              只有输入 buffer 的值变化 → 不会因换输入重编译）。
        输出：Engine 实例。
        预期行为：权重按 fp32 主权重载入（与训练存盘一致）；前向走 autocast。
                  缺失/多余键会打印（静默吞键会导致推理结果无意义）。
        """
        self.cfg = ModelConfig.from_json()
        self.model = Model(self.cfg, tokenizer=tokenizer).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"载入 {ckpt}：缺失 {len(missing)} 键 / 多余 {len(unexpected)} 键")
        self.model.eval()
        self.device = device
        self.dtype = dtype
        # **推理权重转 bf16**：训练存的是 fp32 主权重（优化器精度需要），但推理只用前向。
        # fp32 权重会让每次 `F.linear(bf16_x, fp32_w)` 先做一次 cast（每层多一个 kernel），
        # 且显存带宽翻倍——在 launch-bound 的 decode 上这是纯浪费。53M 全转 bf16 省 ~105MB。
        # RMSNorm/亲和参数保持 fp32 会破坏 autocast 语义，故整模统一转（与训练的计算精度一致）。
        self.model.to(dtype)
        # 小 batch MoE 走 index_select+bmm，免掉 `_grouped_mm` 的 offs 同步
        # （decode 单 token 时每层 3 次 DtoH，整步 24 次，是主要 launch 瓶颈之一）。
        from model.layers.moe import enable_infer_gather
        enable_infer_gather(True)
        if not quantize:
            _disable_fake_quant()
        # CUDA Graph 必须配合 torch.compile：不融合的话图里有 ~1100 个 eager kernel
        # （replay 3.09ms）；融合后 ~350 个（replay 0.89ms）。两者叠加才是最快路径。
        if compile or cuda_graph:
            from inference.compile import compile_for_decode
            compile_for_decode(self.model)

        # CUDA Graph：按 `emit_ats` 的取值组合各捕获一张。捕获只依赖形状与控制流，
        # 重放时只改输入 buffer 的值 → 换输入不会重编译。
        self._ratios = [layer.attn.compress_ratio for layer in self.model.layers]
        self._last_hidden = None
        self._need = (self.model.engram_hash.max_ngram_size - 1
                      if self.model.engram_hash is not None else 0)
        self._graphs = None
        if cuda_graph:
            self._graphs = {}
            self._graphs_for(1)

    def _graphs_for(self, steps: int):
        """取（必要时捕获）本步 token 数对应的那组 CUDA Graph。

        输入：steps（一次前向喂几个 token）。
        输出：`_DecodeGraphs` 或 None（未启用图）。
        预期行为：每个 steps 值有独立的静态输入 buffer 与编译产物；首次使用才捕获
                  （捕获要重新编译，约几十秒，属于一次性启动成本）。
        """
        if self._graphs is None:
            return None
        if steps not in self._graphs:
            period = max(max(self._ratios), 1)
            keys = [(p, self._emit_ats(p, steps)) for p in range(period)]
            self._graphs[steps] = _DecodeGraphs(self.model, self.device, self._need,
                                                keys, steps=steps)
        return self._graphs[steps]

    def _emit_ats(self, start_pos: int, steps: int = 1) -> tuple:
        """每层本步内「凑满压缩组」的 token 下标（图捕获的控制流键）。

        输入：start_pos（本步首 token 的绝对位置）；steps（本步 token 数）。
        输出：长度 = 层数的 tuple[tuple[int]]。
        预期行为：ratio<=0 的层无压缩（空元组）；ratio==1 每 token 都成组（全部下标）；
                  ratio>1 按 `(start_pos + j + 1) % ratio == 0` 判定。
        """
        return tuple(
            tuple(j for j in range(steps) if cr > 0 and (start_pos + j + 1) % cr == 0)
            for cr in self._ratios)

    def warmup(self, input_ids: torch.Tensor, steps: int = 8):
        """用**真实输入形状**预热（把编译代价挪到计时之外）。

        输入：input_ids [1, L]（实际要生成的 prompt）；steps（预热 decode 步数，
              **至少 4**：要覆盖 `full_group` 的两种奇偶，并让 dynamo 把动态维
              （past_ids 长度）泛化掉；少于这个数，编译会发生在正式生成期间，
              表现为「tok/s 极低」但其实只是把编译时间算进了计时）。
        输出：无。
        预期行为：CUDA Graph 模式下图已在 `__init__` 捕获完毕，这里跑真实 prompt 的
                  prefill + 若干次 replay 预热（首次 replay 有图上传开销），再清空缓存；
                  否则跑一次 prefill + 若干 decode，让 dynamo 按这个 prompt 长度编好图。
        """
        if self._graphs is not None:
            # 图捕获只录了 kernel 序列；首次 replay 还有图上传/惰性初始化开销
            # （实测冷启 2.84ms/token，热身后 1.31ms/token，其中还含 prefill）。
            # 这里用真实 prompt 跑一次 prefill + 若干 replay，把它挪出计时。
            self.prefill(input_ids)
            g = self._graphs_for(1)
            dummy = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            past = torch.zeros(1, self._need, dtype=torch.long, device=self.device)
            for p in range(max(steps, 4)):
                pos = p + 2
                g.step(dummy, pos, past, self._emit_ats(pos))
            torch.cuda.synchronize()
            self.model.reset_cache()
            return
        from inference.compile import warmup_decode
        warmup_decode(self.model, self.device,
                      prompt_len=input_ids.size(1), steps=steps,
                      ids=input_ids)

    # ---- 底层：单步前向 -------------------------------------------------------

    def prefill(self, input_ids: torch.Tensor, past_ids: torch.Tensor | None = None):
        """整段前向，填充 KV 缓存。

        输入：input_ids [1, L]；past_ids（prefill 时为 None）。
        输出：末位置 logits [1, V]。
        预期行为：start_pos=0；只算末位置（full_logits=False）。
        """
        with torch.no_grad(), torch.autocast(device_type=self._ac_type(), dtype=self.dtype):
            logits, _ = self.model(input_ids, start_pos=0, full_logits=False,
                                   past_ids=past_ids)
        self._last_hidden = self.model._last_main_hidden
        return logits.float()

    def decode_step(self, token: torch.Tensor, start_pos: int, past_ids: torch.Tensor):
        """单 token 增量前向（取最后一个位置的 logits）。

        输入：token [1, 1]；start_pos（该 token 的位置）；past_ids [1, P]
              （完整历史，供 Engram n-gram 哈希回看；只取末尾 need 个）。
        输出：logits [1, V]。
        """
        return self.decode_step_multi(token, start_pos, past_ids)[:, -1]

    def decode_step_multi(self, tokens: torch.Tensor, start_pos: int,
                          past_ids: torch.Tensor):
        """多 token 增量前向（一次吃 T 个 token，返回每个位置的 logits）。

        输入：tokens [1, T]；start_pos（**首** token 的绝对位置）；past_ids [1, P]
              （T 个 token **之前**的历史）。
        输出：logits [1, T, V]。
        预期行为：CUDA Graph 模式下按 T 选（必要时捕获）对应的一组图并重放；
                  否则把位置转成 0 维张量走普通前向。
                  T 个 token 顺序追加进 KV 缓存，语义等价于 T 次单 token 前向。
        """
        if self._graphs is not None:
            if self._need:
                past = torch.narrow(past_ids, 1, past_ids.size(1) - self._need, self._need)
            else:
                past = past_ids
            g = self._graphs_for(tokens.size(1))
            logits, hidden = g.step(tokens, start_pos, past,
                                    self._emit_ats(start_pos, tokens.size(1)))
            self._last_hidden = hidden
            return logits.float()
        pos = torch.tensor(start_pos, device=tokens.device, dtype=torch.long)
        with torch.no_grad(), torch.autocast(device_type=self._ac_type(), dtype=self.dtype):
            logits, _ = self.model.forward_decode(
                tokens, pos, past_ids, emit_ats=self._emit_ats(start_pos, tokens.size(1)))
        self._last_hidden = self.model._last_main_hidden
        return logits.float()

    def _ac_type(self) -> str:
        return "cuda" if self.device.startswith("cuda") else "cpu"

    # ---- 采样 ---------------------------------------------------------------

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Gumbel-max 采样（与参考实现 `generate.py` 同一算法）。

        输入：logits [1, V]；temperature（0=贪心）。
        输出：[1, 1]。
        预期行为：temperature>0 时 `probs / Exp(1)` 的 argmax，fp32 softmax。
        """
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1, dtype=torch.float32)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)

    # ---- 生成 ---------------------------------------------------------------

    def generate(self, input_ids: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
        """普通增量解码（非投机）。

        输入：input_ids [1, L0]；cfg。
        输出：[1, L0 + 实际生成数]（末尾多余的 token 已在 eos 处截断）。
        预期行为：reset_cache → prefill → 循环 decode_step。
                  **每 4 步才检查一次 eos**：`int(tensor.item())` 会强制 GPU→CPU 同步，
                  实测每次约 8ms（占单步耗时 20%）；隔步检查把同步开销摊薄到 1/4，
                  代价是最多多生成 3 个 token（返回前按 eos 截断，语义不变）。
        """
        self.model.reset_cache()
        ids = input_ids
        logits = self.prefill(ids)
        nxt = self.sample(logits, cfg.temperature)
        ids = torch.cat([ids, nxt], dim=1)
        # eos 状态留在 GPU 上累计（`(nxt == eos).any()` 不产生同步），每 8 步才拉一次
        eos = cfg.eos_token_id
        done = torch.zeros((), dtype=torch.bool, device=ids.device)
        for step in range(cfg.max_new_tokens - 1):
            logits = self.decode_step(nxt, start_pos=ids.size(1) - 1, past_ids=ids[:, :-1])
            nxt = self.sample(logits, cfg.temperature)
            ids = torch.cat([ids, nxt], dim=1)
            if eos is not None:
                done = done | (nxt == eos).any()
                if step % 8 == 7 and bool(done.item()):
                    break
        if eos is not None:
            # 兜底截断：隔步检查期间可能已生成 eos，这里按首次出现位置切掉
            gen = ids[0, input_ids.size(1):]
            pos = (gen == eos).nonzero(as_tuple=False)
            if pos.numel():
                ids = ids[:, :input_ids.size(1) + int(pos[0].item()) + 1]
        return ids
