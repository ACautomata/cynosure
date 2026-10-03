"""分布式执行层残余（spec #15 模块划分「distributed」；ADR-0003）。

**第三批删除面收口后（#238 / ADR-0018）**：判别器副本（Replicated
Discriminator）与指标归并器（EventMerger）本体已随 pretrain driver
async 化退役——新执行序的跨卡一致性由「同初始化 + 主控单点
allreduce + 同步 step」结构性承载（``train/discriminator``），事件流
单进程单写者直写。本包残余面只服务**旧执行序 trainer 的衰减窗口**
（切换期第二步整删）：

- 进程组装配（process）：torchrun env → init_process_group 的单点
  Facade，单进程 = world-1 恒等退化；
- policy 分片（shard）：FSDP full-shard 封装与 full state dict 出入口
  （旧执行序 torchrun 路径）；
- pool 切片（pool）：Real sample pool manifest 的 rank 条带视图
  （旧执行序判别器 real 侧「本 rank 切片更新」语义）。
"""

from cynosure.distributed.pool import RankSlicedPool
from cynosure.distributed.process import DistributedContext
from cynosure.distributed.shard import PolicySharding

__all__ = [
    "DistributedContext",
    "PolicySharding",
    "RankSlicedPool",
]
