"""Real sample pool 的 rank 切片（orchestration 章「Real sample pool 切片
更新」）。

按序列分层的条带切片：每序列条目内部做 entries[rank::world]——各 rank
切片覆盖全部四序列（全局条带会与病例×序列的交错排列周期冲突，world
不整除序列数时某序列在片内归零）。序列条目数不足 world_size = 该序列
real 侧在某 rank 断供（per-target-sequence 健康监控与 real 采样的前提
被破坏），装配期显式拒绝而非静默空采；**校验消费切片前的 full
manifest**（每 rank 对同一全量判定同一结果——按切片后本地视图校验时
rank 间可见性不同，失败方单方面退出装配、其余 rank 进入集合操作互等）。
held-out real 不切片（out-of-sample 监控信号保持全量）。
"""

from cynosure.config import MODALITIES
from cynosure.distributed.process import DistributedContext
from cynosure.reward.artifacts import LatentManifest


class RankSlicedPool:
    """判别器 real 侧 pool 的本 rank 视图（分层条带切片的单点语义）。"""

    def __init__(
        self, manifest: LatentManifest, context: DistributedContext,
    ) -> None:
        self._manifest = manifest
        self._context = context

    def view(self) -> LatentManifest:
        """本 rank 的 pool 条带视图（单进程恒等返回原 manifest）。"""
        if not self._context.distributed:
            return self._manifest
        self._assert_pool_sufficient()
        banded: list = []
        for modality in MODALITIES:
            layer = [
                entry for entry in self._manifest.entries
                if entry.modality == modality
            ]
            banded.extend(layer[self._context.rank::self._context.world_size])
        return self._manifest.with_entries(banded)

    def _assert_pool_sufficient(self) -> None:
        """切片前的全量充分性校验（失败路径全 rank 一致）：条带切片是
        entries[rank::world]，序列条目数 ≥ world_size 时每 rank 的条带
        才非空。"""
        starved: list[str] = []
        for modality in MODALITIES:
            count = sum(
                1 for entry in self._manifest.entries
                if entry.modality == modality
            )
            if count < self._context.world_size:
                starved.append(f"{modality}×{count}")
        if starved:
            rank, world = self._context.rank, self._context.world_size
            raise ValueError(
                f"Real sample pool 不足以支撑 {world}-路 rank 切片"
                f"（每序列每 rank 至少 1 条，不足序列: "
                f"{', '.join(starved)}；pool 共 "
                f"{len(self._manifest.entries)} 条）: 判别器 real 侧断供"
            )
