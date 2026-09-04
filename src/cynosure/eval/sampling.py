"""Baseline 采样与 RL 后重采（experiment-design「Baseline 协议」）。

两相位共用同一 manifest 条目（同 seed 同条件、冻结只采一次）：

- **baseline**（训练启动期）：冻结初始 policy（更新开始前的当前 policy）
  对全部条目采 Anchor 终点 → 解码像素体落盘 → 路径写回 manifest；
- **resample**（训练结束后）：RL 后 policy 对**同一条目**同噪声种子重采
  → 落盘 → 路径写回 manifest——两侧差异唯一归因于 RL。

组2 条目在 baseline 期锁定源病例（``source_case`` 写回 manifest），
重采与里程碑评测读回同一病例。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cynosure.config import CynosureConfig
from cynosure.eval.condition import EntryConditionResolver
from cynosure.eval.decode import VolumeDecoder
from cynosure.policy.numerics import AmpContext
from cynosure.policy.sampler import RolloutSampler

if TYPE_CHECKING:
    from cynosure.train.artifacts import BaselineManifest, ManifestEntry, RunPaths

PHASE_BASELINE = "baseline"
PHASE_RESAMPLE = "resample"


@dataclass(frozen=True)
class EntrySample:
    """一个 manifest 条目的 policy 采样产出（条目 + 目标序列 + Anchor 终点）。"""

    entry: ManifestEntry
    target: str
    """采样的目标序列（组2 = 锁定源病例的目标端）。"""
    source_case: str | None
    """组2 锁定的源病例（组1 为 None）——配对参照的依据。"""
    terminal: torch.Tensor
    """该条目的 Anchor 终点 latent [1, C, D, H, W]。"""


class ManifestLatentSampler:
    """manifest 条目的 policy latent 采样（条目噪声种子确定性）——里程碑
    评测与 Baseline/重采的单一共享实现（同条目必得同 latent）。"""

    def __init__(
        self,
        config: CynosureConfig,
        sampler: RolloutSampler,
        resolver: EntryConditionResolver,
        numerics: AmpContext,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._resolver = resolver
        self._numerics = numerics

    def sample(self, entries: list[ManifestEntry]) -> list[EntrySample]:
        """逐条目解析条件、按噪声种子采样 Anchor 终点（eval + no_grad +
        autocast，与训练 rollout 同口径）。噪声经 CPU generator 生成
        （跨设备可复现）后随数值口径迁移到 policy 设备。"""
        samples: list[EntrySample] = []
        with torch.no_grad(), torch.autocast(
            self._numerics.device_type, dtype=self._numerics.dtype,
        ):
            for entry in entries:
                condition, target = self._resolver.resolve(entry)
                noise = torch.randn(
                    (1, *self._config.latent_shape),
                    generator=torch.Generator().manual_seed(entry.noise_seed),
                ).to(self._numerics.device)
                terminal = self._sampler.anchor_trajectory(noise, condition)[-1]
                source_case = (
                    self._resolver.source_case(entry)
                    if not isinstance(entry.condition, str) else None
                )
                samples.append(EntrySample(entry, target, source_case, terminal))
        return samples


class ManifestVolumeSampler:
    """按 manifest 条目采样解码像素体并回写样本路径（两相位同一编排）。"""

    def __init__(
        self,
        stage: int,
        manifest: "BaselineManifest",
        latent_sampler: ManifestLatentSampler,
        decoder: VolumeDecoder,
        paths: "RunPaths",
    ) -> None:
        self._stage = stage
        self._manifest = manifest
        self._latent_sampler = latent_sampler
        self._decoder = decoder
        self._paths = paths

    def sample_baseline(self) -> None:
        """冻结初始 policy 的 Baseline 采样（冻结只采一次；须在首个
        policy 梯度步之前调用——条目采样用的是当时的当前权重）。"""
        self._sample(PHASE_BASELINE)

    def sample_resample(self) -> None:
        """RL 后 policy 的同 manifest 重采（训练结束后调用）。"""
        self._sample(PHASE_RESAMPLE)

    def _sample(self, phase: str) -> None:
        entries = self._manifest.entries_for_stage(self._stage)
        volumes = self._volumes(entries)
        for entry, volume in zip(entries, volumes[:, 0]):
            relative = f"samples/stage{self._stage}/{phase}/{entry.index:04d}.pt"
            target = self._paths.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(volume, target)
            if phase == PHASE_BASELINE:
                entry.baseline_sample = relative
            else:
                entry.resample_sample = relative
        self._manifest.write(self._paths.manifest)

    def _volumes(self, entries: list[ManifestEntry]) -> torch.Tensor:
        """全部条目的解码像素体批 [K, 1, X, Y, Z]（整批一次解码，与
        里程碑评测同口径）。"""
        samples = self._latent_sampler.sample(entries)
        terminals = torch.cat([sample.terminal for sample in samples])
        return self._decoder.decode(terminals)
