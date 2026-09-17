"""eval 模块：VAE 解码 + 像素域 2.5D FID/KID（RadImageNet-ResNet50、
三正交面）+ 3D SSIM/MAE + Baseline 采样清单 + 里程碑评测路径。

裁决性 MR FID 读数仪器（``mr_fid``，fork 口径原尺寸切片）与
real-vs-real 地板工具（``real_real_floor``，病例级 seed 半分）是
#73 双轨的独立轨：经 ``cynosure fid`` / ``cynosure fid-floor`` 子命令
驱动，**训练循环不经它们**——里程碑维持本包 224×224 设施口径只看
趋势，两侧数字不可互比。

关键接口（spec #15 模块划分）：从 checkpoint + Real sample pool 产出指标
与评测材料。``EvaluationPhase`` 是训练循环依赖的评测相接口（三个动作），
``ManifestEvaluation`` 是其 manifest 驱动实现与装配入口：Baseline 采样
（训练启动期、冻结初始 policy）、里程碑解码评测（``milestone`` 事件入
训练指标流）与 RL 后重采（同 manifest 条目）三路径的唯一入口——**解码
只发生在这些评测路径，不进逐 iteration 训练循环**（ADR-0004）。
验收阶梯的 nnUNet 对齐接口与盲审导出由后续 ticket 交付。

依赖方向：本包不 import ``cynosure.train``（运行时）——manifest 等 train
侧契约经调用方注入/TYPE_CHECKING 引用，两包互不构成导入环
（train → eval 单向）。
"""

from typing import Protocol, TYPE_CHECKING

import torch

from cynosure.conditions import ConditionVocabulary
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
from cynosure.policy.numerics import AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.preprocessing import UpstreamPreprocessChain

if TYPE_CHECKING:
    from cynosure.train.artifacts import BaselineManifest, RunArtifacts

__all__ = [
    "EntryConditionResolver",
    "EvaluationPhase",
    "LatentDecoder",
    "ManifestEvaluation",
    "ManifestLatentSampler",
    "ManifestVolumeSampler",
    "MilestoneEvaluator",
    "MilestoneMetrics",
    "RadImageNetFeatureExtractor",
    "StubSliceFeatureExtractor",
    "VolumeDecoder",
]


class EvaluationPhase(Protocol):
    """训练循环面对的评测相接口（trainer 依赖本接口而非具体装配，
    测试替身显式实现同一契约）：Baseline 采样 / 里程碑解码评测 /
    RL 后重采——解码只发生在这三条评测路径（ADR-0004）。"""

    def sample_baseline(self) -> None:
        """训练启动期的 Baseline 采样（冻结初始 policy、冻结只采一次）。"""
        ...

    def resample(self) -> None:
        """RL 后的同 manifest 重采（训练结束后、最终 policy）。"""
        ...

    def milestone_metrics(self) -> MilestoneMetrics:
        """当前 policy 的里程碑度量（``milestone`` 事件的取数面）。"""
        ...


class ManifestEvaluation:
    """``EvaluationPhase`` 的 manifest 驱动实现：Baseline 采样 / 里程碑
    评测 / RL 后重采的单点持有——两相位与里程碑共用同一 manifest 条目
    （同 seed 同条件，差异唯一归因于 RL）。

    监控相（里程碑评测）协作者可为空：无里程碑触发点的 run（``schedule``
    判定，见 ``_monitoring_reachable``）不装配参照影像库与特征提取器，
    只保留不依赖二者的 Baseline / 重采两相位。"""

    def __init__(
        self,
        evaluator: MilestoneEvaluator | None,
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
        write_enabled: bool = True,
    ) -> "ManifestEvaluation":
        """按 config 装配评测相（manifest 由调用方从 run 目录装载注入；
        数值口径随训练循环的 AmpContext 单点传入；decoder/extractor 可
        注入替身：fixture stub、测试计数解码器；缺省按 fixture/生产分派）。"""
        pool = cls._load_pool(config)
        # 条件词汇表自装载（eval 不 import train——train.trainer 依赖
        # 本包，经 train.runtime 装配会成环；与 GroupPolicy 同口径的
        # 两域内联分派）：rollout 条件解析与逐条目 latent 形状的共同
        # 取数面（#129 形状按条件贯通）
        vocabulary = cls._assemble_vocabulary(config)
        # 监控相（里程碑解码评测）的两道装配守卫与同一前提绑定：本 run
        # 是否存在里程碑触发点——不触发里程碑的 run 不消费监控相，其
        # 样本面与参照库的装配要求随之不适用（#123：把监控相的资源/
        # 配置前置强加给不消费它的 run，会让主循环 tracer 被下游监控票
        # 的交付物阻塞）
        monitoring_reachable = cls._monitoring_reachable(config)
        # 里程碑样本面守卫（schema 校验的 MR-RATE 承接面，#129）：
        # 条目按条件轮转，K < 词汇表条件数即永久漏尾部条件——生产 config
        # 在评测装配期显式拒绝（schema 不读词表工件，BraTS 的下界校验
        # 仍在 schema 层）；fixture 豁免（评测面以盘上条目为准）。
        if (
            monitoring_reachable
            and not config.fixture_mode
            and config.schedule.milestone_eval_samples < len(vocabulary.names())
        ):
            raise ValueError(
                f"schedule.milestone_eval_samples="
                f"{config.schedule.milestone_eval_samples} 未覆盖本域条件"
                f"词汇表（{len(vocabulary.names())} 个条件；manifest 条件"
                "轮转下 K 不足即永久漏尾部条件，早停判据对其失明——"
                "增大评测样本面或显式声明 fixture_mode"
            )
        # MR-RATE 参照影像库尚未交付（RealVolumeStore = BraTS 病例布局
        # 的参照库，dataset_root 扫描与序列键都是 BraTS 语义）：两域条件
        # 的采样与分层度量面已按词汇表贯通（#129），参照侧像素库随 MR
        # 数据管线后续 ticket（#124 监控链路）交付后在装配处同点分派
        # ——显式拒绝而非让 BraTS 布局扫描在 MR dataset_root 上炸出
        # 布局错误
        if config.experiment.dataset == "MR-RATE" and monitoring_reachable:
            raise ValueError(
                "MR-RATE 线的里程碑参照影像库尚未交付（RealVolumeStore "
                "是 BraTS 病例布局的参照库）：本 run 会走到里程碑、评测"
                "装配在此显式拒绝，MR 参照库随 MR 数据管线后续 ticket 交付"
                "后在同一装配点分派（不触发里程碑的 run 不装配监控相）"
            )
        resolver = EntryConditionResolver(vocabulary, amp.device, pool=pool)
        latent_sampler = ManifestLatentSampler(sampler, resolver, amp, vocabulary)
        resolved_decoder = decoder if decoder is not None else cls._build_decoder(
            config, amp.device,
        )
        # 监控相协作者按运行可达性装配：无里程碑触发点的 run 不构造特征
        # 提取器与参照影像库（生产上二者各自需要 RadImageNet 权重与
        # 参照库工件——把监控相的资源需求强加给不消费它的 run，会让
        # 主循环 tracer 被下游监控票的交付物阻塞）
        evaluator = (
            MilestoneEvaluator(
                config,
                stage,
                latent_sampler,
                resolved_decoder,
                extractor if extractor is not None
                else cls._build_extractor(config, amp.device),
                cls._build_reals(config, pool),
                manifest,
                amp.device,
            )
            if monitoring_reachable else None
        )
        volume_sampler = ManifestVolumeSampler(
            stage,
            manifest,
            latent_sampler,
            resolved_decoder,
            artifacts.paths,
            decode_batch_size=config.schedule.decode_batch_size,
            write_enabled=write_enabled,
        )
        return cls(evaluator, volume_sampler)

    def sample_baseline(self) -> None:
        """训练启动期的 Baseline 采样（冻结初始 policy、冻结只采一次）。"""
        self._volume_sampler.sample_baseline()

    def resample(self) -> None:
        """RL 后的同 manifest 重采（训练结束后、最终 policy）。"""
        self._volume_sampler.sample_resample()

    def milestone_metrics(self) -> MilestoneMetrics:
        """当前 policy 的里程碑度量（``milestone`` 事件的取数面）。

        监控相缺席（本 run 无里程碑触发点）时显式报错而非返回空读数：
        缺席是装配前提不满足，不是「无操作」——静默的空读数会让早停判据
        在无量测输入下做出判断。"""
        if self._evaluator is None:
            raise ValueError(
                "本 run 未装配监控相（schedule.max_iterations < "
                "schedule.milestone_interval：训练循环无里程碑触发点，"
                "里程碑读数不存在消费时机）——里程碑消费面不可达"
            )
        return self._evaluator.evaluate()

    @staticmethod
    def _monitoring_reachable(config: CynosureConfig) -> bool:
        """本 run 是否存在里程碑触发点（监控相可达性的装配期判定）。

        训练循环的里程碑触发条件 = 完成数整除 ``milestone_interval``
        （iteration 从 1 起计数），故「存在 k ∈ [1, max_iterations] 使
        k % interval == 0」等价于 ``max_iterations ≥ milestone_interval``
        ——续训同理：起点之后的剩余区间的可达性由同一对 (max_iterations,
        interval) 决定，起点本身不进入装配期判据（装配早于恢复，起点
        尚不可知；用全区间判定是保守方向——判定为可达而实际没走到，
        至多多装配一个不消费的监控相，反向漏判则会让里程碑在运行中途
        才炸）。"""
        return (
            config.schedule.max_iterations >= config.schedule.milestone_interval
        )

    @staticmethod
    def _assemble_vocabulary(config: CynosureConfig) -> ConditionVocabulary:
        """条件词汇表装配（eval 不经 train 装配面——依赖方向 train →
        eval 单向）；两域装载分派本体在 ``ConditionVocabulary.assemble``
        （与 runtime 同口径的消费侧单一来源）。"""
        return ConditionVocabulary.assemble(config)

    @staticmethod
    def _load_pool(config: CynosureConfig) -> LatentManifest:
        """Real sample pool（病例级 70% train split 的 latent 索引）：组2
        条目的源病例与**全部组**的里程碑参照病例库都取自它——参照分布
        不越过病例级分割（experiment-design「real 样本库」）。"""
        return LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )

    @staticmethod
    def _build_reals(
        config: CynosureConfig, pool: LatentManifest,
    ) -> RealVolumeStore:
        """参照影像库（BraTS 单域语义——MR-RATE 在 build 期已显式拒绝，
        MR 参照库交付后本构造面随装配处分派扩展）：病例白名单 = pool
        train split 的病例集；预处理链与 prepare 预编码同口径（resize
        基数随 config——生产钉上游基数，fixture 注入小基数保持夹具尺寸）。"""
        return RealVolumeStore(
            config.artifacts.dataset_root,
            case_ids={entry.case_id for entry in pool.entries},
            preprocess=UpstreamPreprocessChain(config.preprocessing.resize_base),
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
            config.policy.latent_scale_factor,
            tuple(config.schedule.decode_roi_size),
            config.schedule.decode_overlap,
        )

    @staticmethod
    def _build_extractor(
        config: CynosureConfig, device: torch.device,
    ) -> SliceFeatureExtractor:
        """特征提取器装配（fixture stub / 生产 RadImageNet）。生产骨干落
        ``device``——里程碑度量的归一设备与训练数值口径一致。"""
        if config.fixture_mode:
            return StubSliceFeatureExtractor()
        if config.artifacts.radimagenet_weights is None:
            raise ValueError(
                "生产里程碑评测需要 RadImageNet-ResNet50 权重"
                "（config artifacts.radimagenet_weights；公开发布权重的下载"
                "属施工），fixture 经 fixture_mode=true 走 stub 注入"
            )
        return RadImageNetFeatureExtractor(
            config.artifacts.radimagenet_weights, device,
        )
