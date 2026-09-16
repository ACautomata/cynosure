"""Real sample pool 的 rank 切片（orchestration 章「Real sample pool 切片
更新」）。

按条件分层的条带切片：每条件条目内部做 entries[rank::world]——各 rank
切片覆盖全部条件（全局条带会与病例×条件的交错排列周期冲突，world
不整除条件数时某条件在片内归零）。条件条目数不足 world_size = 该条件
real 侧在某 rank 断供（per-target-sequence 健康监控与 real 采样的前提
被破坏），装配期显式拒绝而非静默空采；**校验消费切片前的 full
manifest**（每 rank 对同一全量判定同一结果——按切片后本地视图校验时
rank 间可见性不同，失败方单方面退出装配、其余 rank 进入集合操作互等）。
held-out real 不切片（out-of-sample 监控信号保持全量）。

分层轴 = **活动条件集**（#129 装配期注入：BraTS = 四序列常量语义、
MR-RATE = 词汇表条件集）：本类不设代码内条件副本（口径同
``LatentManifest.assert_condition_capacity`` 的注入面——副本会让换域线
的每条目条件在切片校验里计数为零、全 rank 一致地拒绝，多 rank 换域
运行直接不可达）。
"""

from cynosure.distributed.process import DistributedContext
from cynosure.reward.artifacts import LatentManifest


class RankSlicedPool:
    """判别器 real 侧 pool 的本 rank 视图（分层条带切片的单点语义）。"""

    def __init__(
        self, manifest: LatentManifest, context: DistributedContext,
        conditions: list[str] | tuple[str, ...],
    ) -> None:
        self._manifest = manifest
        self._context = context
        self._conditions = tuple(conditions)

    def view(self) -> LatentManifest:
        """本 rank 的 pool 条带视图（单进程恒等返回原 manifest）。"""
        if not self._context.distributed:
            return self._manifest
        self._assert_pool_sufficient()
        banded: list = []
        for condition in self._conditions:
            layer = [
                entry for entry in self._manifest.entries
                if entry.modality == condition
            ]
            banded.extend(layer[self._context.rank::self._context.world_size])
        return self._manifest.with_entries(banded)

    def _assert_pool_sufficient(self) -> None:
        """切片前的全量充分性校验（失败路径全 rank 一致）：条带切片是
        entries[rank::world]，条件条目数 ≥ world_size 时每 rank 的条带
        才非空。条件集 = 注入的活动条件集（切片分层轴同源）。"""
        starved: list[str] = []
        for condition in self._conditions:
            count = sum(
                1 for entry in self._manifest.entries
                if entry.modality == condition
            )
            if count < self._context.world_size:
                starved.append(f"{condition}×{count}")
        if starved:
            rank, world = self._context.rank, self._context.world_size
            raise ValueError(
                f"Real sample pool 不足以支撑 {world}-路 rank 切片"
                f"（每条件每 rank 至少 1 条，不足条件: "
                f"{', '.join(starved)}；pool 共 "
                f"{len(self._manifest.entries)} 条）: 判别器 real 侧断供"
            )
