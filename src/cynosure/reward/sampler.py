"""Real sample 批采样（判别器「真」侧的数据访问）。

Real sample pool manifest 的消费端：均匀无放回采 K 条 latent 成批，
供判别器 online update 的 real 侧。确定性经 torch.Generator 注入
（fixture 与生产共用同一采样语义；固定 seed + 同一 generator 状态
序列 → 采样序列可复现）。pool / held-out 两侧共用本采样器，
kind 语义由 LatentManifest 装载层守卫。
"""

from typing import Protocol

import torch

from cynosure.config import Modality
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

    def sample(
        self, count: int, *, modality: Modality | None = None,
    ) -> torch.Tensor:
        """无放回均匀采 count 条 latent（``modality`` 给定时仅在该序列
        的条目内采样）。"""
        ...


class RealPoolSampler:
    """Real sample pool 的批采样器（无放回均匀采样）。

    manifest latents 持久化在 CPU，采样结果经 ``device`` 迁移到消费方
    （判别器/打分）所在设备——CPU generator 的采样决策与设备解耦。"""

    def __init__(
        self,
        manifest: LatentManifest,
        generator: torch.Generator,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._manifest = manifest
        self._generator = generator
        self._device = device

    @property
    def size(self) -> int:
        """pool 条目总数（held-out AUC 对称采样的参照）。"""
        return len(self._manifest.entries)

    def sample(
        self, count: int, *, modality: Modality | None = None,
    ) -> torch.Tensor:
        """无放回均匀采 count 条 latent；超出候选条目数显式拒绝。

        ``modality`` 给定时候选收窄为该序列条目（held-out AUC 按本
        iteration 采样的目标序列归因）；缺省 None 为全池（Online update
        的 real 侧语义）。"""
        candidates = self._manifest.entries
        if modality is not None:
            candidates = [
                entry for entry in candidates if entry.modality == modality
            ]
        if count < 1 or count > len(candidates):
            raise ValueError(
                f"采样数 {count} 超出 pool 条目 {len(candidates)}（无放回采样）"
            )
        indices = torch.randperm(len(candidates), generator=self._generator)[:count]
        return torch.stack([
            self._manifest.load_latent(candidates[index])
            for index in indices.tolist()
        ]).to(self._device)
