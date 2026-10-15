"""分布式数据并行（DDP）封装——供五个 trainer 复用。

设计原则：**非分布式环境下全部是 no-op**，单卡训练路径与本模块引入前逐位一致
（RANK 未设时，init 不建进程组、wrap 原样返回、分片退化为原样）。

两种启动方式，任选其一：
1. torchrun（本地 / 有 shell 的环境）：
       torchrun --nproc_per_node=2 trainer/train_vlm.py --data_path ... --epochs 1
   torchrun 设好 RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT。
2. `launch()`（notebook / 无 torchrun 的环境，如 Kaggle）：
       from model.train import ddp
       ddp.launch(trainer_main, nproc_per_node=2)
   用 `torch.multiprocessing.spawn` 起多进程；每个子进程内设好上述环境变量再调
   trainer_main。Jupyter 里无法直接 fork 多进程跑训练循环，故必须走 spawn。
   注意：spawn 要求被调函数可 pickle，故 trainer 的 `main()` 需是模块级函数。

**梯度等价（逐位对拍的关键）**：
DDP 把各卡梯度**平均**（sum / world_size）。若各卡 loss 仍是「批内 mean」，
则变长序列下各卡 valid token 数不同 → 与单卡「全局 mean」**不等价**。
故本模块提供 `scale_loss_for_ddp(local_sum, local_count)`：
返回 `world_size * local_sum / global_count`，使 DDP 平均后 = 单卡全局 mean。
world_size==1 时退化为 `local_sum / local_count` = 普通 mean，单卡数值不变。
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def is_distributed() -> bool:
    """当前是否处于分布式环境。

    输入：无。
    输出：True/False。
    预期行为：**照 minimind 的判定**——torchrun 会设 RANK（0,1,2...），普通 python 启动
              不设（默认 -1）→ 未设即非 DDP。比 `WORLD_SIZE>1` 更可靠：torchrun 单进程
              (nproc_per_node=1) 也设 RANK=0/WORLD_SIZE=1，此时也应按分布式初始化
              （否则 dist 相关调用会报「未初始化」）。
    """
    return int(os.environ.get("RANK", "-1")) != -1


def get_rank() -> int:
    """当前进程的全局 rank（非分布式时为 0）。"""
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """当前进程在**本机**的 rank（决定用哪张 GPU；非分布式时为 0）。"""
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    """是否主进程（rank 0）——只有它存 checkpoint / 写日志 / 打印。

    输入：无。
    输出：True/False。
    预期行为：分布式时仅 rank0 为 True；单卡时恒 True（保持原行为）。
    """
    return get_rank() == 0


def setup(device_arg: str | None = None) -> torch.device:
    """初始化进程组并返回本进程该用的 device。

    输入：device_arg（命令行 --device；分布式时忽略，改由 LOCAL_RANK 决定）。
    输出：torch.device。
    预期行为：分布式时 init_process_group(nccl) + set_device(local_rank)；
              非分布式时原样返回 device_arg（默认 cuda:0）。
              必须在建模型**之前**调用。
    """
    if not is_distributed():
        return torch.device(device_arg or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    # MASTER_ADDR/PORT 由 torchrun 注入，不在此补缺省（硬塞 127.0.0.1 会破坏多机场景）。
    dist.init_process_group(backend="nccl")
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")


def wrap(model: torch.nn.Module, find_unused_parameters: bool = True) -> torch.nn.Module:
    """把模型包进 DDP（非分布式时原样返回）。

    输入：model；find_unused_parameters（MoE 有条件分支 → 每步可能有未用专家，
          默认 True 保证正确；关掉更快但要求所有参数每步都参与，MoE 下不成立）。
    输出：DDP 包装后的模型，或原模型（非分布式）。
    预期行为：device_ids 用 LOCAL_RANK 对应的卡；DDP 只同步梯度，不改前向语义。
    """
    if not is_distributed():
        return model
    return DDP(model, device_ids=[get_local_rank()], output_device=get_local_rank(),
               find_unused_parameters=find_unused_parameters)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """取回 DDP 内部的原始模型（用于访问自定义方法/属性，如 update_router_bias、vision）。

    输入：可能是 DDP 包装的模型。
    输出：原始模型。
    预期行为：DDP 会把被包模型注册为子模块 `module`；非 DDP 则原样返回。
              DDP 不代理任意属性访问（`ddp.vision` 会 AttributeError），故访问自定义
              属性前必须先 unwrap。
    """
    return model.module if isinstance(model, DDP) else model


def scale_loss_for_ddp(local_sum: torch.Tensor, local_count: torch.Tensor,
                       world_size: int | None = None) -> torch.Tensor:
    """把「本卡 loss 之和」缩放成 DDP 平均后等于单卡全局 mean 的形式。

    输入：local_sum（本卡 CE 的 sum 归约）；local_count（本卡有效 token 数，标量张量）；
          world_size（None=从环境读）。
    输出：标量 loss 张量。
    预期行为：返回 `world_size * local_sum / global_count`，其中 global_count 由
              all_reduce(local_count) 得到。DDP 对梯度取平均后恰好等于
              单卡 batch=world_size*B 的全局 mean（逐位等价，浮点结合律差异除外）。
              world_size==1 时退化为 local_sum / local_count（普通 mean，单卡数值不变）。
    """
    ws = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", "1"))
    if ws == 1:
        return local_sum / local_count.clamp(min=1)
    global_count = local_count.clone()
    dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    return ws * local_sum / global_count.clamp(min=1)


def barrier() -> None:
    """进程组同步屏障（非分布式时为 no-op）。用于存盘后等待，避免 rank 间错位。"""
    if is_distributed():
        dist.barrier()


def shard_index_info() -> tuple[int, int]:
    """返回本进程的 (shard_index, shard_count)，用于 IterableDataset 的 DDP 分片。

    输入：无。
    输出：(idx, n)。idx ∈ [0, n)，本进程应取的切片下标；n = world_size（非分布式时 n=1）。
    预期行为：**非分布式时返回 (0, 1)** → 数据集切分退化为原样（单卡逐位不变）。
              分布式时 idx = rank，各 rank 取不同切片、互不重叠。
              调用方负责再叠加 worker 维度（见 shard_worker_index）。
    """
    return get_rank(), (int(os.environ.get("WORLD_SIZE", "1")) if is_distributed() else 1)


def world_size() -> int:
    """当前进程组大小（非分布式时为 1）。"""
    return int(os.environ.get("WORLD_SIZE", "1")) if is_distributed() else 1


def shard_worker_index() -> tuple[int, int]:
    """返回 (idx, n)：把 rank 与 DataLoader worker 合并成一个扁平分片维度。

    输入：无（读 RANK / WORLD_SIZE 环境变量与当前 DataLoader worker 信息）。
    输出：(idx, n)。n = world_size * num_workers；idx = rank * num_workers + worker_id。
    预期行为：使所有 (rank, worker) 组合覆盖全部数据且互不重叠。
              **单进程单 worker（非分布式 + num_workers=0）时返回 (0, 1)** → 原样，
              单卡路径逐位不变。
    """
    from torch.utils.data import get_worker_info

    ws = world_size()
    rank = get_rank() if ws > 1 else 0
    wi = get_worker_info()
    nw = wi.num_workers if wi is not None else 1
    wid = wi.id if wi is not None else 0
    if ws == 1 and nw == 1:
        return 0, 1
    return rank * nw + wid, ws * nw


def cleanup() -> None:
    """销毁进程组（非分布式时为 no-op）。训练结束时调用。"""
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def launch(trainer_main, nproc_per_node: int | None = None, argv=None) -> None:
    """多进程启动器（spawn）——供 notebook / 无 torchrun 的环境使用。

    输入：trainer_main（模块级函数，如 train_vlm.main）；nproc_per_node（每机进程数，
          None = GPU 数）；argv（子进程命令行参数，None = 当前 sys.argv[1:]）。
    输出：无。
    预期行为：**仅当 GPU 数 > 1 时**才走 spawn（单卡/无卡时直接单进程调 trainer_main，
              与原来完全一致）。spawn 的每个子进程内设好 RANK/WORLD_SIZE/LOCAL_RANK/
              MASTER_ADDR/MASTER_PORT 环境变量，再调 trainer_main —— 之后 ddp.setup()
              就能正常初始化进程组。
              必须在 `if __name__ == "__main__":` 保护下调用（spawn 要求）。
    """
    import sys
    from torch.multiprocessing import spawn

    # 已由 torchrun 启动（RANK 已设）→ 直接跑，**不再 spawn**（否则双重启动）。
    if is_distributed():
        trainer_main()
        return
    n = nproc_per_node if nproc_per_node is not None else torch.cuda.device_count()
    if n <= 1:
        # 单卡：不 spawn，直接跑（与原行为一致）
        trainer_main()
        return
    args = argv if argv is not None else sys.argv[1:]
    spawn(_worker, args=(trainer_main, n, args), nprocs=n, join=True)


def _worker(local_rank: int, trainer_main, nproc: int, argv) -> None:
    """spawn 的子进程入口：设环境变量 → 调 trainer_main。内部使用，勿直接调。"""
    import sys

    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    # 子进程的 sys.argv 需还原（trainer_main 里 argparse 读它）
    sys.argv = [sys.argv[0]] + list(argv)
    trainer_main()
