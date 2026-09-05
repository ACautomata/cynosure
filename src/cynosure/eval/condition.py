"""manifest 条目 → rollout 条件的解析（Baseline/重采/里程碑评测共用）。

三侧消费同一 manifest 条目（同 seed 同条件）：组1 条目条件 = 目标序列名；
组2 条目条件 = [源序列, 目标序列] + 锁定的源病例（``source_case`` 由
baseline 采样期确定性派生并写回 manifest——重采与里程碑读回同一病例，
配对数据集里该病例的目标序列即 ground-truth 参照）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from cynosure.config import Modality
from cynosure.policy.condition import (
    CONDITION_SPACING_X1E2,
    ModalityMapping,
    RolloutCondition,
)
from cynosure.reward.artifacts import LatentManifest, PoolEntry

if TYPE_CHECKING:
    from cynosure.train.artifacts import ManifestEntry


class EntryConditionResolver:
    """manifest 条目的 rollout 条件解析器（组2 的源病例锁定点）。"""

    def __init__(
        self,
        mapping: ModalityMapping,
        device: torch.device,
        pool: LatentManifest | None = None,
    ) -> None:
        self._mapping = mapping
        self._device = device
        self._pool = pool

    def resolve(self, entry: ManifestEntry) -> tuple[RolloutCondition, str]:
        """条目 → (rollout 条件, 目标序列名)。组2 的源影像条件按锁定
        病例整条装载——latent 与 per-case spacing 同条目同源（issue #46
        侧车；与训练侧 CrossModalConditionSampler 同一原则），缩放发生
        在采样场。"""
        if isinstance(entry.condition, str):
            return self._label_condition(entry.condition), entry.condition
        source_modality, target_modality = entry.condition
        source_entry = self._source_entry(self.source_case(entry), source_modality)
        return (
            RolloutCondition(
                label=torch.tensor(
                    [self._mapping.label(target_modality)], device=self._device,
                ),
                spacing=torch.tensor(
                    [source_entry.spacing], device=self._device,
                ),
                source_latent=self._pool.load_latent(source_entry).unsqueeze(0).to(
                    self._device,
                ),
            ),
            target_modality,
        )

    def source_case(self, entry: ManifestEntry) -> str:
        """条目锁定的源病例（组2）：已锁定直接返回；未锁定（baseline 前
        首采）则按池清单确定性派生并写回条目——派生是 (noise_seed, 池内容)
        的纯函数，重采必得同病例。"""
        if entry.source_case is not None:
            return entry.source_case
        source_modality = entry.condition[0]
        entry.source_case = self._derive_source_case(entry, source_modality)
        return entry.source_case

    def _derive_source_case(
        self, entry: ManifestEntry, source_modality: Modality,
    ) -> str:
        if self._pool is None:
            raise ValueError(
                "组2 条目需要 Real sample pool manifest（源影像条件的病例池）"
            )
        case_ids = sorted(
            pool_entry.case_id
            for pool_entry in self._pool.entries
            if pool_entry.modality == source_modality
        )
        if not case_ids:
            raise ValueError(
                f"Real sample pool 缺序列 {source_modality} 的病例"
                f"（组2 源影像条件不可用）"
            )
        return case_ids[entry.noise_seed % len(case_ids)]

    def _source_entry(
        self, case_id: str, source_modality: Modality,
    ) -> PoolEntry:
        """锁定病例某序列的 pool 条目（latent 与 spacing 侧车的共同取数点）。"""
        if self._pool is None:
            raise ValueError(
                "组2 条目需要 Real sample pool manifest（源影像条件的病例池）"
            )
        for pool_entry in self._pool.entries:
            if (
                pool_entry.case_id == case_id
                and pool_entry.modality == source_modality
            ):
                return pool_entry
        raise ValueError(
            f"Real sample pool 无病例 {case_id!r} 的 {source_modality} 条目"
        )

    def _label_condition(self, modality: str) -> RolloutCondition:
        return RolloutCondition(
            label=torch.tensor([self._mapping.label(modality)], device=self._device),
            spacing=torch.tensor([CONDITION_SPACING_X1E2], device=self._device),
        )
