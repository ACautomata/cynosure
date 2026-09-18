"""里程碑评测（ADR-0004 定量评测的里程碑路径执行）。

到达里程碑间隔时由 train 循环触发：当前 policy 按 Baseline manifest
条目（前缀 K 条，同 seed 同条件 → 跨里程碑可比）采出 Anchor 终点
latent → VAE 解码到像素域（``decode_batch_size`` 分块，峰值显存以块
为界）→ 三正交面逐面提取特征 → 2.5D FID + KID（重采样 95% CI）。
距离**按目标序列分层计算再宏平均**（fid/kid = 各 target 距离均值）：
条件坍缩（忽略/交换模态标签、保持总体混合）在全池聚合下不可见，分层
宏平均使其在主判据上直接显形；分层值随 ``criteria_summary`` 落盘。
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

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from cynosure.config import CynosureConfig
from cynosure.eval.decode import VolumeDecoder
from cynosure.eval.features import SliceFeatureExtractor
from cynosure.eval.frechet import BootstrapKernelMmd, FrechetDistance
from cynosure.eval.sampling import EntrySample, ManifestLatentSampler
from cynosure.eval.volumes import (
    OrthoPlane,
    ReferenceVolumes,
    VolumePairFidelity,
)

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
    replicates: torch.Tensor
    """KID 重采样重复值 [R]（跨面均值后的自助分布原料——上层宏平均
    构造汇总 CI 的消费面）。"""


@dataclass(frozen=True)
class MilestoneMetrics:
    """一次里程碑评测的度量结果（MilestoneEvent 的取数面）。"""

    fid: float
    kid: float
    kid_ci_low: float
    kid_ci_high: float
    plane_fid: dict[str, float]
    plane_kid: dict[str, float]
    target_fid: dict[str, float] = field(default_factory=dict)
    """按目标序列分层的 FID（宏平均的组分——条件坍缩的观测面）。"""
    target_kid: dict[str, float] = field(default_factory=dict)
    """按目标序列分层的 KID（同上）。"""
    ssim: float | None = None
    """跨模态组另加：合成 target vs 同病例 ground-truth target 的 3D SSIM。"""
    mae: float | None = None
    """跨模态组另加：同上配对的 MAE。"""
    psnr: float | None = None
    """跨模态组另加：同上配对的 PSNR（dB，封顶 100）。"""
    phase_seconds: dict[str, float] = field(default_factory=dict)
    """监控成本读数的相位卡时（#124，#111 监控账的成本行取数面）：
    ``decode`` = 合成侧 VAE 分块解码；``fid`` = 参照装载 + 特征提取 +
    距离核 + 配对保真。空 dict = 调用方未做相位计量（替身直调场景）。"""

    def summary(self) -> dict[str, float]:
        """criteria_summary 的度量侧条目（逐面 FID/KID + CI 界 + 按目标
        序列分层的距离——分层是宏平均的组分，随事件落盘供坍缩归因）。"""
        summary = {
            **{f"fid_{name.lower()}": value for name, value in self.plane_fid.items()},
            **{f"kid_{name.lower()}": value for name, value in self.plane_kid.items()},
            "kid_ci_low": self.kid_ci_low,
            "kid_ci_high": self.kid_ci_high,
            **{
                f"fid_target_{name.lower()}": value
                for name, value in self.target_fid.items()
            },
            **{
                f"kid_target_{name.lower()}": value
                for name, value in self.target_kid.items()
            },
        }
        return summary


class MilestoneEvaluator:
    """里程碑解码评测编排（decode 的唯一调用方之一，均在里程碑路径）。

    度量定位：本类产出的 FID/KID 是**训练期小样本相对信号**——条目前缀
    K（同 seed 同条件）保证跨里程碑可比，服务于早停的 plateau 比较；
    K 小则高维特征空间协方差秩亏、绝对值噪声大，验收判据以 N_baseline
    全量样本的对照评估为准（experiment-design「对照基线」）。
    """

    def __init__(
        self,
        config: CynosureConfig,
        stage: int,
        latent_sampler: ManifestLatentSampler,
        decoder: VolumeDecoder,
        extractor: SliceFeatureExtractor,
        reals: ReferenceVolumes,
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
        """当前 policy 的里程碑度量（条目前缀 K 条，与 Baseline 同 seed 同条件）。

        距离按**目标条件分层**计算再宏平均（fid/kid = 各 target 距离的
        算术均值）：条件模型若忽略/交换模态标签而保持总体混合比例，
        全池聚合距离下 FID/KID 可以依旧好看——分层后每个 target 的合成
        分布对该 **target 自己**的参照分布计距，坍缩在主判据上直接
        可见；分层值本身随 ``criteria_summary`` 落盘（``fid_target_``
        / ``kid_target_`` 前缀）。

        异形状批组织（#129）：条目按目标条件分组——组内同条件即同
        形状，合成侧分组解码（cat 仅同条件批内发生）、参照侧分组
        stack、读数逐条件产出；「里程碑评测按条件产出读数」在异形状
        下是批组织的结构前提而非仅是统计口径。

        监控成本读数（#124）：评测分解为 decode（合成侧 VAE 分块解码）
        与 fid（参照装载 + 特征提取 + 距离核 + 配对保真）两相打点，随
        度量透传（``phase_seconds``）——#111 监控账的 decode/FID 卡时
        行在里程碑事件上的取数面。"""
        entries = self._manifest.entries_for_stage(self._stage)[
            : self._config.schedule.milestone_eval_samples
        ]
        samples = self._latent_sampler.sample(entries)
        groups: dict[str, list[EntrySample]] = {}
        for sample in samples:
            groups.setdefault(sample.target, []).append(sample)
        decoded_groups: dict[str, torch.Tensor] = {}
        decode_started = time.monotonic()
        for target, group in sorted(groups.items()):
            decoded_groups[target] = self._decode(group)  # [k, 1, X, Y, Z]（组内同形）
        fid_started = time.monotonic()
        per_target: dict[str, PlaneMetrics] = {}
        for target, group in sorted(groups.items()):
            synthetic = decoded_groups[target]
            reference = torch.stack([
                self._reals.reference_volume(
                    sample.target, sample.entry.index, sample.source_case,
                )
                for sample in group
            ]).to(self._device).unsqueeze(1)  # 与合成侧同形、逐例对齐
            if reference.shape != synthetic.shape:
                raise ValueError(
                    f"目标条件 {target}: 参照体栈 {tuple(reference.shape)} 与"
                    f"合成体栈 {tuple(synthetic.shape)} 形状不一致——两侧必须"
                    "在同一影像空间（生产参照须经 prepare 预处理到模型影像"
                    "空间后入参照库；dataset_root 原生 NIfTI 直读不构成对齐"
                    "参照）"
                )
            per_target[target] = self._plane_metrics(
                synthetic[:, 0], reference[:, 0],
            )
        macro = self._macro_average(per_target)
        ssim = mae = psnr = None
        if self._is_cross_modal:
            # 组2 仅 BraTS 单域（各条件同形）：分组解码结果按**条目原序**
            # 回排后整体配对——按条件名序 cat 会把 synthetic 行序重排
            # （manifest 条目目标交错、轮转序非名字典序），与按 manifest
            # 序 stack 的参照栈逐位错位（SSIM/MAE/PSNR 配错对）
            row_of: dict[str, int] = {}
            synthetic_rows: list[torch.Tensor] = []
            for sample in samples:
                row = row_of.get(sample.target, 0)
                row_of[sample.target] = row + 1
                synthetic_rows.append(decoded_groups[sample.target][row])
            synthetic = torch.stack(synthetic_rows)
            reference = torch.stack([
                self._reals.reference_volume(
                    sample.target, sample.entry.index, sample.source_case,
                )
                for sample in samples
            ]).to(self._device).unsqueeze(1)
            ssim, mae, psnr = VolumePairFidelity().score(synthetic, reference)
        return MilestoneMetrics(
            fid=macro.fid, kid=macro.kid,
            kid_ci_low=macro.kid_ci_low, kid_ci_high=macro.kid_ci_high,
            plane_fid=macro.plane_fid, plane_kid=macro.plane_kid,
            target_fid={
                target: metrics.fid for target, metrics in per_target.items()
            },
            target_kid={
                target: metrics.kid for target, metrics in per_target.items()
            },
            ssim=ssim, mae=mae, psnr=psnr,
            phase_seconds={
                "decode": fid_started - decode_started,
                "fid": time.monotonic() - fid_started,
            },
        )

    @staticmethod
    def _macro_average(per_target: dict[str, PlaneMetrics]) -> PlaneMetrics:
        """跨目标序列的宏平均：标量与逐面字段取组分均值；汇总 KID 的
        重采样 CI 对**宏平均统计量本身**构造（组分重复值逐 replicate
        跨 target 均值后取分位，与点估计同口径）。单一 target 时恒等于
        该组自身（宏平均退化为原口径）。"""
        components = list(per_target.values())
        replicate_pool = torch.stack(
            [metrics.replicates for metrics in components],
        ).mean(dim=0)
        plane_names = list(components[0].plane_fid)
        return PlaneMetrics(
            fid=sum(metrics.fid for metrics in components) / len(components),
            kid=sum(metrics.kid for metrics in components) / len(components),
            plane_fid={
                name: sum(
                    metrics.plane_fid[name] for metrics in components
                ) / len(components)
                for name in plane_names
            },
            plane_kid={
                name: sum(
                    metrics.plane_kid[name] for metrics in components
                ) / len(components)
                for name in plane_names
            },
            kid_ci_low=float(
                replicate_pool.quantile(BootstrapKernelMmd.LOWER_QUANTILE)
            ),
            kid_ci_high=float(
                replicate_pool.quantile(BootstrapKernelMmd.UPPER_QUANTILE)
            ),
            replicates=replicate_pool,
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
            replicates=pooled,
        )
