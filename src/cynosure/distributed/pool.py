"""Real sample pool 的 rank 切片（orchestration 章「Real sample pool 切片
更新」）。

按序列分层的条带切片：每序列条目内部做 entries[rank::world]——各 rank
切片覆盖全部四序列（全局条带会与病例×序列的交错排列周期冲突，world
不整除序列数时某序列在片内归零）。切片后每序列条目数不足 1 = 该序列
real 侧断供（per-target-sequence 健康监控与 real 采样的前提被破坏），
装配期显式拒绝而非静默空采。held-out real 不切片（out-of-sample 监控
信号保持全量）。
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
        banded: list = []
        for modality in MODALITIES:
            layer = [
                entry for entry in self._manifest.entries
                if entry.modality == modality
            ]
            banded.extend(layer[self._context.rank::self._context.world_size])
        occupied = {entry.modality for entry in banded}
        empty = [
            modality for modality in MODALITIES if modality not in occupied
        ]
        if empty:
            rank, world = self._context.rank, self._context.world_size
            raise ValueError(
                f"Real sample pool 的 rank {rank}/{world}"
                f" 切片缺少序列 {empty} 条目（判别器 real 侧断供）："
                f"pool 共 {len(self._manifest.entries)} 条，"
                "需每 rank 每序列至少 1 条"
            )
        return self._manifest.with_entries(banded)
