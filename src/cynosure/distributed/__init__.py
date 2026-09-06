"""分布式执行层（spec #15 模块划分「distributed」；ADR-0003）。

- 进程组装配（process）：torchrun env → init_process_group 的单点
  Facade，单进程 = world-1 恒等退化；
- policy 分片（shard）：FSDP full-shard + 梯度检查点封装与 full state
  dict 出入口（spec：同构 FSDP rank、无角色划分）；
- 判别器副本（replica）：每 rank 完整副本 + 标准 DDP（不分片、梯度
  allreduce，ADR-0003）；
- pool 切片（pool）：Real sample pool manifest 的 rank 条带视图
  （判别器 real 侧「本 rank 切片更新」语义）；
- 指标归并（merge）：iter 事件的 rank 0 归并写出（无重复/丢失、顺序
  稳定）。
"""

from cynosure.distributed.merge import EventMerger
from cynosure.distributed.pool import RankSlicedPool
from cynosure.distributed.process import DistributedContext
from cynosure.distributed.replica import ReplicatedDiscriminator
from cynosure.distributed.shard import PolicySharding

__all__ = [
    "DistributedContext",
    "EventMerger",
    "PolicySharding",
    "RankSlicedPool",
    "ReplicatedDiscriminator",
]
