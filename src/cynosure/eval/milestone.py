"""里程碑评测（ADR-0004 定量评测的里程碑路径执行）。

到达里程碑间隔时由 train 循环触发：当前 policy 按 Baseline manifest
条目（前缀 K 条，同 seed 同条件 → 跨里程碑可比）采出 Anchor 终点
latent → VAE 解码到像素域（``decode_batch_size`` 分块，峰值显存以块
为界）→ 三正交面逐面提取特征 → 2.5D FID + KID（重采样 95% CI）；
跨模态组另加 3D SSIM/MAE/PSNR——合成 target 影像与**同一病例
ground-truth 的 target 序列影像**逐例配对（配对数据集，非 source 与
target 直接比较）。

设备口径：合成侧（解码输出）与参照侧（CPU NIfTI 装载）统一归一到
构造注入的 ``device``；float64 度量核（Frechet/MMD）消费 CPU 特征，
与加速器解耦。影像空间口径：两侧体栈形状必须一致（生产参照须先经
prepare 预处理到模型影像空间）。

本类只在里程碑路径被构造调用；逐 iteration 训练循环不经解码
（结构断言见测试面）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cynosure.config import CynosureConfig
from cynosure.eval.decode import VolumeDecoder
from cynosure.eval.features import SliceFeatureExtractor
from cynosure.eval.frechet import BootstrapKernelMmd, FrechetDistance
from cynosure.eval.sampling import EntrySample, ManifestLatentSampler
from cynosure.eval.volumes import OrthoPlane, RealVolumeStore, VolumePairFidelity

if TYPE_CHECKING:
    from cynosure.train.artifacts import BaselineManifest


@dataclass(frozen=True)
class PlaneMetrics:
    """三正交面逐面度量的汇总（FID 均值 + KID 均值与重采样 CI 界）。"""

    fid: float
    kid: float
    plane_fid: dict[str, float]
    plane_kid: dict[str, float]
    kid_ci_low: float
    kid_ci_high: float


@dataclass(frozen=True)
class MilestoneMetrics:
    """一次里程碑评测的度量结果（MilestoneEvent 的取数面）。"""

    fid: float
    kid: float
    kid_ci_low: float
    kid_ci_high: float
    plane_fid: dict[str, float]
    plane_kid: dict[str, float]
    ssim: float | None = None
    """跨模态组另加：合成 target vs 同病例 ground-truth target 的 3D SSIM。"""
    mae: float | None = None
    """跨模态组另加：同上配对的 MAE。"""
    psnr: float | None = None
    """跨模态组另加：同上配对的 PSNR（dB，封顶 100）。"""

    def summary(self) -> dict[str, float]:
        """criteria_summary 的度量侧条目（逐面 FID/KID + CI 界）。"""
        summary = {
            **{f"fid_{name.lower()}": value for name, value in self.plane_fid.items()},
            **{f"kid_{name.lower()}": value for name, value in self.plane_kid.items()},
            "kid_ci_low": self.kid_ci_low,
            "kid_ci_high": self.kid_ci_high,
        }
        return summary


class MilestoneEvaluator:
    """里程碑解码评测编排（decode 的唯一调用方之一，均在里程碑路径）。"""

    def __init__(
        self,
        config: CynosureConfig,
        stage: int,
        latent_sampler: ManifestLatentSampler,
        decoder: VolumeDecoder,
        extractor: SliceFeatureExtractor,
        reals: RealVolumeStore,
        manifest: BaselineManifest,
        device: torch.device,
    ) -> None:
        self._config = config
        self._stage = stage
        self._latent_sampler = latent_sampler
        self._decoder = decoder
        self._extractor = extractor
        self._reals = reals
        self._manifest = manifest
        self._device = device

    def evaluate(self) -> MilestoneMetrics:
        """当前 policy 的里程碑度量（条目前缀 K 条，与 Baseline 同 seed 同条件）。"""
        entries = self._manifest.entries_for_stage(self._stage)[
            : self._config.schedule.milestone_eval_samples
        ]
        samples = self._latent_sampler.sample(entries)
        synthetic = self._decode(samples)  # [K, 1, X, Y, Z]
        reference = torch.stack([
            self._reals.volume(self._reference_case(sample), sample.target)
            for sample in samples
        ]).to(self._device).unsqueeze(1)  # [K, 1, X, Y, Z]（与合成侧同形，逐例对齐）
        if reference.shape != synthetic.shape:
            raise ValueError(
                f"参照体栈 {tuple(reference.shape)} 与合成体栈 "
                f"{tuple(synthetic.shape)} 形状不一致——两侧必须在同一影像"
                f"空间（生产参照须经 prepare 预处理到模型影像空间后入参照"
                f"库；dataset_root 原生 NIfTI 直读不构成对齐参照）"
            )

        planes = self._plane_metrics(synthetic[:, 0], reference[:, 0])
        ssim = mae = psnr = None
        if self._is_cross_modal:
            ssim, mae, psnr = VolumePairFidelity().score(synthetic, reference)
        return MilestoneMetrics(
            fid=planes.fid, kid=planes.kid,
            kid_ci_low=planes.kid_ci_low, kid_ci_high=planes.kid_ci_high,
            plane_fid=planes.plane_fid, plane_kid=planes.plane_kid,
            ssim=ssim, mae=mae, psnr=psnr,
        )

    def _decode(self, samples: list[EntrySample]) -> torch.Tensor:
        """Anchor 终点 latent 的分块解码（峰值显存以块为界），输出落
        ``device`` 的 [K, 1, X, Y, Z]——解码设备即度量归一设备（注入
        替身可能产出别的设备）。"""
        batch = self._config.schedule.decode_batch_size
        return torch.cat([
            self._decoder.decode(
                torch.cat([
                    sample.terminal
                    for sample in samples[start:start + batch]
                ]),
            ).to(self._device)
            for start in range(0, len(samples), batch)
        ])

    @property
    def _is_cross_modal(self) -> bool:
        """跨模态组另加 SSIM/MAE/PSNR（组3 stage-2 同语义，随阶段 config 判定）。"""
        return self._config.experiment.group == "cross-modal"

    def _reference_case(self, sample: EntrySample) -> str:
        """参照病例：组2 = entry 锁定的源病例（其目标序列 = ground-truth
        target，配对数据集）；组1 = 按条目序号确定性轮转病例。"""
        if sample.source_case is not None:
            return sample.source_case
        case_ids = self._reals.case_ids()
        return case_ids[sample.entry.index % len(case_ids)]

    def _plane_metrics(
        self, synthetic: torch.Tensor, reference: torch.Tensor,
    ) -> PlaneMetrics:
        """逐面 FID/KID 点估计 + 汇总 KID 的重采样 CI。

        CI 的自助分布对**汇总统计量本身**构造：每次重复取各面的无放回
        半样本 MMD²、跨面取均值，分位取自该重复值分布——CI 与点估计
        （三面全量 MMD² 的均值）同口径，而非逐面 CI 界的算术平均。"""
        frechet = FrechetDistance()
        kid = BootstrapKernelMmd(
            replicates=self._config.schedule.kid_bootstrap_replicates,
            generator=torch.Generator().manual_seed(self._config.schedule.seed),
        )
        plane_fid: dict[str, float] = {}
        plane_kid: dict[str, float] = {}
        replicate_means: list[torch.Tensor] = []
        for plane in OrthoPlane.all_planes():
            # 度量核（float64）与设备解耦：特征归一 CPU——MPS/CUDA 无
            # float64（或代价高），而 Frechet/MMD 的计算量毫不足道
            features_synth = self._extractor.extract(
                plane.slice(synthetic),
            ).cpu()
            features_real = self._extractor.extract(
                plane.slice(reference),
            ).cpu()
            name = plane.name
            plane_fid[name] = frechet.score(features_synth, features_real)
            point, replicates = kid.score_and_replicates(features_synth, features_real)
            plane_kid[name] = point
            replicate_means.append(replicates)
        pooled = torch.stack(replicate_means).mean(dim=0)
        return PlaneMetrics(
            fid=sum(plane_fid.values()) / len(plane_fid),
            kid=sum(plane_kid.values()) / len(plane_kid),
            plane_fid=plane_fid,
            plane_kid=plane_kid,
            kid_ci_low=float(pooled.quantile(kid.LOWER_QUANTILE)),
            kid_ci_high=float(pooled.quantile(kid.UPPER_QUANTILE)),
        )
