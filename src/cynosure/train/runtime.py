"""公开装配缝的命名空间（train / pretrain driver / async 门面共用的
单一装配来源）。

config 驱动的装配单点收敛：条件词汇表（``assemble_vocabulary``）、
采样封装与 σ 日程（``assemble_sampler`` / ``assemble_schedules``）、
前向激活预算（``forward_activation_budget``）、real pool 守卫装载
（``assemble_real_pool``）、判别器更新批装配原语
（``assemble_pair_assembler``）、判别器 scorer 权重来源分派
（``assemble_scorer``）与数值口径（``amp_context``——定义在
policy/numerics，此处 re-export 保持既有消费面）。「无第二套装配逻辑」
由本类承载：新执行序门面（``AsyncTrainingExecutor``）与预训练 driver
（``PretrainDriver.build``）都只面对这些缝编排，装配细节不进循环
代码路径。

旧执行序（torchrun 多进程 trainer）的装配入口（``build``）与其分布式
装配面（seed rank 派生 / FSDP 分片 / pool 切片）已随 #226 切换期第二步
（#242 全删除面）删除——执行模型 = async 单进程多卡（每卡完整副本 +
主控单点 allreduce + 同步 step），跨卡一致性由执行序自身编排。
"""

import torch

from cynosure.conditions import ConditionVocabulary
from cynosure.config import CynosureConfig
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.field import VelocityField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import (
    RolloutSampler,
    auto_forward_activation_budget,
    cuda_total_memory,
)
from cynosure.policy.schedules import (
    ConditionSchedules,
    PerConditionSchedules,
    SingleConditionSchedules,
)
from cynosure.pretrain.artifacts import PretrainReport
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.assembly import ReconstructionAssembler
from cynosure.reward.scorer import RewardScorer
from cynosure.train.rollout import ConditionSampler

__all__ = ["AMP_DTYPES", "AmpContext", "TrainingRuntime"]


class TrainingRuntime:
    """公开装配缝的命名空间（train / pretrain driver / async 门面共用）。"""

    @classmethod
    def assemble_vocabulary(cls, config: CynosureConfig) -> ConditionVocabulary:
        """条件词汇表装配（公开装配缝：预训练 driver 等不经本运行时的
        消费方同口径消费）——两域装载分派本体在
        ``ConditionVocabulary.assemble``（消费侧单一来源）。"""
        return ConditionVocabulary.assemble(config)

    @classmethod
    def assemble_sampler(
        cls,
        config: CynosureConfig,
        field: VelocityField,
        device: "torch.device | None" = None,
    ) -> RolloutSampler:
        """policy 采样封装装配（日程表 + 本组采样场 + SDE 核）。

        公开装配缝：async 门面（在线 train）与预训练 driver 共用同一份
        装配代码——采样日程/核参数的调整单点生效。

        ``device`` 参与前向激活预算解析（``forward_activation_budget``）：
        config 显式值优先、缺省按设备总显存自动探测；无 CUDA 设备
        （CPU fixture 口径）回落默认常量。"""
        policy = config.policy
        kernel = SdeKernel(eta=policy.sde_eta, s_max=policy.sde_s_max)
        return RolloutSampler(
            field, kernel, cls.assemble_schedules(config),
            forward_activation_budget=cls.forward_activation_budget(
                config, device,
            ),
        )

    @classmethod
    def forward_activation_budget(
        cls,
        config: CynosureConfig,
        device: "torch.device | None" = None,
    ) -> int:
        """rollout 前向激活预算的单一解析点（字节）：
        ``policy.forward_activation_budget_gib`` 显式值优先——**设备总显存
        可探测时**（CUDA）超过即拒绝（永远装不下，装配期早失败优于运行中
        OOM）；无探测面（CPU fixture / 未传设备）只保正值约束，不做该上界
        校验。缺省按设备总显存自动探测（``AUTO_FORWARD_ACTIVATION_FRACTION``
        ——按总显存而非空闲，共享实例上须显式钉值）。"""
        pinned = config.policy.forward_activation_budget_gib
        if pinned is None:
            return auto_forward_activation_budget(device)
        budget = int(pinned * 2**30)
        total = cuda_total_memory(device)
        if total is not None and budget > total:
            raise ValueError(
                f"policy.forward_activation_budget_gib={pinned} 超过本"
                f"设备总显存 {total / 2**30:.1f} GiB——单次前向的激活"
                "预算不可能装下（常驻权重/优化器态/缓冲与碎片余量"
                "另占）：调低该值或改用缺省自动探测"
            )
        return budget

    @classmethod
    def assemble_schedules(cls, config: CynosureConfig) -> ConditionSchedules:
        """sigma 日程装配（#129 逐条件锚，ADR-0002 语义逐条件化）：
        MR-RATE = 逐条件日程表（锚 = 条件词汇表该条件的空间 numel，
        与 rollout 噪声形状结构性同源）；BraTS = 单条件日程表（单域
        语义 = 单条件词汇特例，全局锚日程的等价形态——数值零漂移）。"""
        if config.experiment.dataset == "MR-RATE":
            return PerConditionSchedules(
                num_inference_steps=config.policy.num_inference_steps,
                vocabulary=cls.assemble_vocabulary(config),
            )
        return SingleConditionSchedules(
            num_inference_steps=config.policy.num_inference_steps,
            input_img_size_numel=config.policy.input_img_size_numel,
        )

    @classmethod
    def assemble_real_pool(
        cls,
        config: CynosureConfig,
        vocabulary: ConditionVocabulary,
        card_count: int,
    ) -> LatentManifest:
        """real sample pool manifest 的装载与守卫单点（async 门面与预训练
        driver 共用的装配缝）：装载 → 逐条件容量守卫（全量 ≥ K ×
        ``card_count``）→ 逐条件形状契约对照（#129 消费侧守卫）。

        ``card_count`` = 守卫的需量倍数 = 本执行序的计算卡数：real 侧
        窗口起点全局无放回抽取（#234 判别器链期），窗口单条件最大需求
        上界 = K × 卡数（#220 决议 9/10）。判定放全量保失败路径全卡一致
        （ADR-0008 决策 4 / ADR-0008-03）。无放回采样语义不动，不引入
        有放回采样补洞。"""
        real_pool = LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )
        real_pool.assert_condition_capacity(
            config.reward.disc_batch_size_k, card_count,
            vocabulary.names(),
        )
        real_pool.assert_condition_shapes(vocabulary)
        return real_pool

    @classmethod
    def assemble_pair_assembler(
        cls,
        config: CynosureConfig,
        sampler: RolloutSampler | None,
        conditions: ConditionSampler | None,
        generator: torch.Generator,
        amp: AmpContext,
    ) -> ReconstructionAssembler | None:
        """判别器更新批装配原语的组装缝（ADR-0012 唯一新缝）。

        ``sampler``/``conditions`` 须成对提供（缺对即拒绝——半依赖的
        装配不可运行、静默 None 会让首 iter 才炸）。条件构造与重构前向
        按组自然分派（组1 CFG 组合场 / 组2 裸条件单前向，ADR-0012 决策
        7——condition 经 ``ConditionSampler.sample_target`` 产出、续跑经
        按组装配的采样场）。跨模态阶段（组2）同源重构的**语义**未裁决：
        真正的「同源」需要（源影像, 目标标签）配对条件化、真实样本库现
        无源影像（ADR-0012 非目标，stage-2 到来时另行设计）——机制缝本
        票照常对组2 开放（机械链路可行），判别任务语义留待专项票。

        real 侧 = 窗口起点全局无放回抽取（``WindowRealDraw``）经任务
        注入（#234 判别器链期），装配原语只承担 fake 构造
        （``reconstruct_assigned``，fake 侧 = 槽 recon 流，#218 五处
        消费面注入面零改动口径、#232 rollout 相重构任务）。"""
        if (sampler is None) != (conditions is None):
            raise ValueError(
                "配对批装配原语的 policy 侧依赖须成对提供：sampler 与 "
                "conditions 同时给定或同时缺省（单边缺省 = 装配期拒绝，"
                "不让缺装配原语的首个判别器更新步才暴露）"
            )
        if sampler is None:
            return None
        return ReconstructionAssembler(
            sampler=sampler,
            schedules=cls.assemble_schedules(config),
            conditions=conditions,
            train_step_indices=sorted(config.policy.train_step_indices_m),
            batch_size_k=config.reward.disc_batch_size_k,
            latent_scale_factor=config.policy.latent_scale_factor,
            generator=generator,
            amp=amp,
        )

    @staticmethod
    def assemble_scorer(
        config: CynosureConfig,
        report: PretrainReport | None,
        *,
        resume: bool = False,
    ) -> RewardScorer:
        """判别器 scorer 的权重来源分派（公开装配缝）：train 新 run =
        warm-start 守卫重载（``report`` 给定）；预训练 driver 冷启动 =
        ``report=None`` + ``resume=False``（``discriminator_ckpt`` 工件
        装载或随机初始化）；async 门面续训恢复 = ``resume=True`` 占位
        （不消费任何 checkpoint 工件，随机初始化被恢复整体覆写）。"""
        if report is not None:
            # warm-start 守卫链：数据口径指纹 → 形态指纹 → 严格装载
            # （任一不符在装配期拒绝，CLI 层回滚 run 目录）
            report.assert_data_provenance(config)
            return report.load_discriminator(config)
        # 冷启动/占位：网络构建（checkpoint 工件装载或随机初始化）在
        # schedule.seed 的派生流下进行，并 fork 隔离全局 RNG——同 config
        # 的两次冷启动判别器权重逐位可复现，且跨 rank 逐位一致（seed
        # 不含 rank 偏移：DDP 装配要求各 rank 初始副本一致）。不扰动
        # 进程全局 RNG 状态（sampling generators 独立对象本就不受影响）。
        # resume 占位不消费任何 checkpoint 工件（权重随恢复整体覆写）
        with torch.random.fork_rng():
            torch.manual_seed(config.schedule.seed + 6)
            return RewardScorer(
                NetworkArtifact(
                    config=NetworkAssembler.load_json(
                        config.artifacts.discriminator_config_json,
                    ),
                    checkpoint=(
                        None if resume
                        else config.artifacts.discriminator_ckpt
                    ),
                ),
                config.reward,
                ChannelStats.load(config.reward.channel_stats_json),
            )

    @staticmethod
    def amp_context(
        config: CynosureConfig, device: torch.device,
    ) -> AmpContext:
        """数值口径的装配单点（amp_dtype → ``AmpContext`` 的同形构造
        单点：async 门面的副本/槽装配共用）。"""
        return AmpContext(
            device=device, dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
