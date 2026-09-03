"""Real sample 批采样（判别器「真」侧的数据访问）。

Real sample pool manifest 的消费端：均匀无放回采 K 条 latent 成批，
供判别器 online update 的 real 侧。确定性经 torch.Generator 注入
（fixture 与生产共用同一采样语义；固定 seed + 同一 generator 状态
序列 → 采样序列可复现）。pool / held-out 两侧共用本采样器，
kind 语义由 LatentManifest 装载层守卫。
"""

from typing import Protocol

import torch

from cynosure.reward.artifacts import LatentManifest


class RealSampling(Protocol):
    """「真」侧批采样的策略接口（glossary「Real sample」的采样语义）。

    Online update（train real）与 HeldOutAuc（held-out real）依赖本接口；
    pool / held-out 两侧共用 RealPoolSampler 实现，kind 语义由
    LatentManifest 装载层守卫。
    """

    @property
    def size(self) -> int:
        """条目总数（held-out AUC 对称采样的参照）。"""
        ...

    def sample(self, count: int) -> torch.Tensor:
        """无放回均匀采 count 条 latent。"""
        ...


class RealPoolSampler:
    """Real sample pool 的批采样器（无放回均匀采样）。"""

    def __init__(self, manifest: LatentManifest, generator: torch.Generator) -> None:
        self._manifest = manifest
        self._generator = generator

    @property
    def size(self) -> int:
        """pool 条目总数（held-out AUC 对称采样的参照）。"""
        return len(self._manifest.entries)

    def sample(self, count: int) -> torch.Tensor:
        """无放回均匀采 count 条 latent；超出 pool 条目数显式拒绝。"""
        if count < 1 or count > self.size:
            raise ValueError(
                f"采样数 {count} 超出 pool 条目 {self.size}（无放回采样）"
            )
        indices = torch.randperm(self.size, generator=self._generator)[:count]
        return torch.stack([
            self._manifest.load_latent(self._manifest.entries[index])
            for index in indices.tolist()
        ])
