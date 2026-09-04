"""eval 模块：VAE 解码 + 像素域 2.5D FID/KID（RadImageNet-ResNet50、
三正交面）+ 3D SSIM/MAE + Baseline 采样清单 + 里程碑评测路径。

关键接口（spec #15 模块划分）：从 checkpoint + Real sample pool 产出指标
与评测材料。``EvaluationPhase`` 是训练循环面对的评测相编排：Baseline
采样（训练启动期、冻结初始 policy）、里程碑解码评测（``milestone``
事件入训练指标流）与 RL 后重采（同 manifest 条目）三路径的唯一入口——
**解码只发生在这些评测路径，不进逐 iteration 训练循环**（ADR-0004）。
验收阶梯的 nnUNet 对齐接口与盲审导出由后续 ticket 交付。

依赖方向：本包不 import ``cynosure.train``（运行时）——manifest 等 train
侧契约经调用方注入/TYPE_CHECKING 引用，两包互不构成导入环
（train → eval 单向）。
"""

from typing import TYPE_CHECKING

import torch

from cynosure.config import CynosureConfig
from cynosure.eval.condition import EntryConditionResolver
from cynosure.eval.decode import LatentDecoder, VolumeDecoder
from cynosure.eval.features import (
    RadImageNetFeatureExtractor,
    SliceFeatureExtractor,
    StubSliceFeatureExtractor,
)
from cynosure.eval.milestone import MilestoneEvaluator, MilestoneMetrics
from cynosure.eval.sampling import ManifestLatentSampler, ManifestVolumeSampler
from cynosure.eval.volumes import RealVolumeStore
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import ModalityMapping
from cynosure.policy.numerics import AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import LatentManifest

if TYPE_CHECKING:
    from cynosure.train.artifacts import BaselineManifest, RunArtifacts

__all__ = [
    "EntryConditionResolver",
    "EvaluationPhase",
    "LatentDecoder",
    "ManifestLatentSampler",
    "ManifestVolumeSampler",
    "MilestoneEvaluator",
    "MilestoneMetrics",
    "RadImageNetFeatureExtractor",
    "StubSliceFeatureExtractor",
    "VolumeDecoder",
]


class EvaluationPhase:
    """一次训练运行的评测相编排：Baseline 采样 / 里程碑评测 / RL 后重采
    的单点持有（trainer 只面对本类的三个动作）。"""

    def __init__(
        self,
        evaluator: MilestoneEvaluator,
        volume_sampler: ManifestVolumeSampler,
    ) -> None:
        self._evaluator = evaluator
        self._volume_sampler = volume_sampler

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        artifacts: "RunArtifacts",
        sampler: RolloutSampler,
        stage: int,
        manifest: "BaselineManifest",
        amp: AmpContext,
        decoder: VolumeDecoder | None = None,
        extractor: SliceFeatureExtractor | None = None,
    ) -> "EvaluationPhase":
        """按 config 装配评测相（manifest 由调用方从 run 目录装载注入；
        数值口径随训练循环的 AmpContext 单点传入；decoder/extractor 可
        注入替身：fixture stub、测试计数解码器；缺省按 fixture/生产分派）。"""
        pool = cls._load_pool(config)
        resolver = EntryConditionResolver(
            ModalityMapping.load(config.artifacts.modality_mapping_json),
            amp.device,
            pool=pool,
        )
        latent_sampler = ManifestLatentSampler(config, sampler, resolver, amp)
        resolved_decoder = decoder if decoder is not None else cls._build_decoder(
            config, amp.device,
        )
        resolved_extractor = (
            extractor if extractor is not None else cls._build_extractor(config)
        )
        evaluator = MilestoneEvaluator(
            config,
            stage,
            latent_sampler,
            resolved_decoder,
            resolved_extractor,
            RealVolumeStore(config.artifacts.dataset_root),
            manifest,
        )
        volume_sampler = ManifestVolumeSampler(
            stage,
            manifest,
            latent_sampler,
            resolved_decoder,
            artifacts.paths,
        )
        return cls(evaluator, volume_sampler)

    def sample_baseline(self) -> None:
        """训练启动期的 Baseline 采样（冻结初始 policy、冻结只采一次）。"""
        self._volume_sampler.sample_baseline()

    def resample(self) -> None:
        """RL 后的同 manifest 重采（训练结束后、最终 policy）。"""
        self._volume_sampler.sample_resample()

    def milestone_metrics(self) -> MilestoneMetrics:
        """当前 policy 的里程碑度量（``milestone`` 事件的取数面）。"""
        return self._evaluator.evaluate()

    @staticmethod
    def _load_pool(config: CynosureConfig) -> LatentManifest | None:
        """组2 源影像病例池（组1 无需装载——condition 条目无源病例语义）。"""
        if config.experiment.group == "modal-label":
            return None
        return LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )

    @staticmethod
    def _build_decoder(config: CynosureConfig, device: torch.device) -> VolumeDecoder:
        if config.artifacts.vae_config_json is None:
            raise ValueError(
                "评测路径需要 VAE 网络工件（artifacts.vae_config_json + "
                "vae_ckpt）：里程碑解码评测与 Baseline/重采的解码装配源"
            )
        return LatentDecoder(
            NetworkArtifact(
                config=NetworkAssembler.load_json(config.artifacts.vae_config_json),
                checkpoint=config.artifacts.vae_ckpt,
            ),
            device,
        )

    @staticmethod
    def _build_extractor(config: CynosureConfig) -> SliceFeatureExtractor:
        if config.fixture_mode:
            return StubSliceFeatureExtractor()
        if config.artifacts.radimagenet_weights is None:
            raise ValueError(
                "生产里程碑评测需要 RadImageNet-ResNet50 权重"
                "（config artifacts.radimagenet_weights；公开发布权重的下载"
                "属施工），fixture 经 fixture_mode=true 走 stub 注入"
            )
        return RadImageNetFeatureExtractor(config.artifacts.radimagenet_weights)
