"""一次装配的训练运行时（spec #15「一次装配 = 一个组的一个阶段」的
组件组 Facade）。

config 驱动的装配产物收敛：policy 侧（GroupPolicy）、判别器侧
（RewardCoordinator）、逐 k 更新（StepwisePolicyUpdate）、rollout 相
（RolloutPhase）、七条命名 RNG 流（TrainingRngStreams 注册表）、数值
口径（AmpContext，定义在 policy/numerics——train 与 eval 共用的 import
环安全位，此处 re-export 保持既有消费面）与分布式运行时
（DistributedContext + EventMerger）。trainer 只面对本 Facade
编排 iteration 循环，装配细节（含分布式包装）不进循环代码路径。

分布式装配点（ADR-0003，仅 torchrun 多进程下生效、单进程恒等）：
- seed 的 rank 派生（各 rank 数据流独立；rank 0 恒等 = 单进程等价前提）；
- 可训练网络 FSDP full-shard + 梯度检查点（PolicySharding，optimizer
  构建于分片后参数之上）；
- 判别器 DDP 副本（ReplicatedDiscriminator）+ Real sample pool 切片
  （RankSlicedPool；held-out 不切）；
- 指标归并器（EventMerger，rank 0 顺序写出）。
"""

import torch

from cynosure.conditions import ConditionVocabulary
from cynosure.config import CynosureConfig
from cynosure.distributed import (
    DistributedContext,
    EventMerger,
    PolicySharding,
    RankSlicedPool,
    ReplicatedDiscriminator,
)
from cynosure.grpo import ClippedPolicyLoss, StepwisePolicyUpdate
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.field import VelocityField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.policy.schedules import (
    ConditionSchedules,
    PerConditionSchedules,
    SingleConditionSchedules,
)
from cynosure.pretrain.artifacts import PretrainReport
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.overfit import OverfitMonitor
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.gating import DynamicWhitelist
from cynosure.train.policy import GroupPolicy
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import RolloutPhase
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.whitelist import ConditionWhitelist

__all__ = ["AMP_DTYPES", "AmpContext", "TrainingRuntime"]


class TrainingRuntime:
    """一次装配的运行时组件组（trainer 循环的协作者容器 + 分布式装配点）。"""

    def __init__(
        self,
        config: CynosureConfig,
        policy: GroupPolicy,
        rewards: RewardCoordinator,
        updater: StepwisePolicyUpdate,
        rollout: RolloutPhase,
        rng: TrainingRngStreams,
        generators: dict[str, torch.Generator],
        amp: AmpContext,
        dist: DistributedContext,
        merger: EventMerger,
    ) -> None:
        self.config = config
        self.policy = policy
        self.rewards = rewards
        self.updater = updater
        self.rollout = rollout
        self.rng = rng
        self.generators = generators
        self.amp = amp
        self.dist = dist
        self.merger = merger

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        device: torch.device | None = None,
        dist_context: DistributedContext | None = None,
        resume: bool = False,
    ) -> "TrainingRuntime":
        """config 驱动装配：单进程与分布式同一条装配序，分布式包装点
        （FSDP/DDP/切片/归并/seed 派生）在 world-1 下恒等。

        ``resume=True`` 时跳过 warm-start 报告装载：续训状态分片已含
        判别器全量状态（权重与 optimizer 恢复时整体覆写），报告在
        resume 路径无消费价值——强制装载会让预训练产物被清理的中断
        run 永不可恢复；判别器占位装配走冷启动随机初始化路径，恢复即
        覆写（resume 模块「装配期随机性被整体覆写」的既有语义）。"""
        dist = dist_context if dist_context is not None else DistributedContext.bootstrap()
        # seed 的 rank 派生：七条流的演化各 rank 独立（rollout 数据多样性
        # 来源）；rank 0 恒等偏移 = world-1 与单进程逐位一致的等价性前提。
        # 判别器冷启动初始化不经派生（跨 rank 一致初始权重，装配内 fork_rng）。
        # 流注册表（TrainingRngStreams）按名保存/恢复续训状态；named() 的
        # dict 视图是装配期按名取流的消费面。
        streams = TrainingRngStreams(dist.derive_seed(config.schedule.seed))
        generators = streams.named()
        # 设备默认 = 本 rank 计算设备（cuda:LOCAL_RANK）：未索引 "cuda"
        # 会让各 rank 都把网络建到 GPU 0，与 FSDP/DDP 包装的 device_id
        # （cuda:LOCAL_RANK）错位；CPU fixture 下即 cpu
        amp = AmpContext(
            device=(
                device if device is not None else dist.local_device()
            ),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        sharding = PolicySharding.from_config(dist, config)
        # FSDP 包装在 optimizer 构建之前（优化器状态活在分片后参数上）
        policy = GroupPolicy.build(
            config, generators["rollout"], amp.device, sharding=sharding,
        )
        rewards = cls.assemble_rewards(
            config, amp, generators, dist,
            report=(
                None if resume
                else PretrainReport.load(config.reward.pretrain_report_json)
            ),
            resume=resume,
        )
        sampler = cls.assemble_sampler(config, policy.field)
        updater = StepwisePolicyUpdate(
            sampler=sampler,
            optimizer=policy.optimizer,
            loss=ClippedPolicyLoss(clip_range=config.policy.ratio_clip),
            device_type=amp.device_type,
            amp_dtype=amp.dtype,
        )
        rollout = RolloutPhase(
            config,
            sampler,
            rewards.update.scorer,
            generators["rollout"],
            condition_sampler=policy.conditions,
            vocabulary=cls.assemble_vocabulary(config),
            device_type=amp.device_type,
            autocast_dtype=amp.dtype,
            device=amp.device,
            # base 分区种子生成的独立派生流（seed+5）：其抽取数随 buffer
            # 容量变化，不占训练 rollout 的抽样流（容量实验不漂移样本流）
            base_generator=generators["base_partition"],
        )
        return cls(
            config=config,
            policy=policy,
            rewards=rewards,
            updater=updater,
            rollout=rollout,
            rng=streams,
            generators=generators,
            amp=amp,
            dist=dist,
            merger=EventMerger(dist, run_artifacts),
        )

    @classmethod
    def assemble_vocabulary(cls, config: CynosureConfig) -> ConditionVocabulary:
        """条件词汇表装配（公开装配缝：预训练 driver 等不经本运行时的
        消费方同口径消费）——两域装载分派本体在
        ``ConditionVocabulary.assemble``（消费侧单一来源）。"""
        return ConditionVocabulary.assemble(config)

    @classmethod
    def assemble_sampler(cls, config: CynosureConfig, field: VelocityField) -> RolloutSampler:
        """policy 采样封装装配（日程表 + 本组采样场 + SDE 核）。

        公开装配缝：train 运行时与预训练 driver（单进程 world-1 语境）
        共用同一份装配代码——采样日程/核参数的调整单点生效。"""
        policy = config.policy
        kernel = SdeKernel(eta=policy.sde_eta, s_max=policy.sde_s_max)
        return RolloutSampler(
            field, kernel, cls.assemble_schedules(config),
        )

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
    def assemble_rewards(
        cls,
        config: CynosureConfig,
        amp: AmpContext,
        generators: dict[str, torch.Generator],
        dist: DistributedContext,
        report: PretrainReport | None = None,
        *,
        resume: bool = False,
    ) -> RewardCoordinator:
        """判别器侧装配：网络构建 → DDP 副本升级（分布式）→ pool 切片
        （real 侧；held-out 不切）→ Online update / AUC 协作者。

        公开装配缝：train 运行时与预训练 driver（world-1 退化语境——
        RankSlicedPool / ReplicatedDiscriminator 在单进程下恒等）共用
        同一份装配代码——「无第二套判别器训练逻辑」在装配层同样成立。

        权重来源按语境三分：``report`` 给定（train 新 run 语境）=
        warm-start 守卫重载（数据口径指纹对照 → 形态指纹对照 → 报告
        checkpoint 严格装载，ADR-0007——RL 不带预训练产物在装配层就
        无法启动）；``report=None`` + ``resume=False`` = 冷启动路径
        （``discriminator_ckpt`` 工件装载或随机初始化，消费方 = 预训练
        driver——预训练本身即产物的生产方）；``resume=True`` = 占位
        装配（不消费任何 checkpoint 工件，纯随机初始化——续训分片已含
        判别器全量状态，占位权重被 ``ResumeStore.restore`` 整体覆写；
        ``discriminator_ckpt`` 即便指向预训练目录，产物清理也不再阻断
        续训）。"""
        if report is not None and resume:
            # 矛盾组合（带报告装配的恢复）显式拒绝：两来源权重同时
            # 声明时以谁为准的歧义不许静默消解（build 层恒传
            # report=None + resume=True，本守卫是 API 层的组合态收口）
            raise ValueError(
                "assemble_rewards 的 report 与 resume 互斥：warm-start 守卫"
                "重载（新 run）与占位装配（续训恢复）是互斥语境"
            )
        if config.artifacts.discriminator_config_json is None:
            raise ValueError(
                "判别器网络配置缺失（discriminator_config_json）：判别器"
                "网络构建的统一前提——warm-start 守卫重载（train 新 run）、"
                "预训练 driver 冷启动、resume 占位装配均经它构建网络"
                "（ADR-0007 后 train 侧不再有冷启动装配源，"
                "discriminator_ckpt 仅预训练 driver 冷启动消费）"
            )
        scorer = cls._assemble_scorer(config, report, resume=resume)
        scorer.to(amp.device)  # 单点递归迁移：判别器参数 + 统计量 buffer
        ReplicatedDiscriminator.replicate(scorer, dist)
        # real 池装配期守卫（ADR-0008 决策 4 / ADR-0008-03）：逐条件容量
        # ≥ K×world_size——条带切片后每 rank 视图 ≥ K 的等价条件，判定放
        # 全量保失败路径全 rank 一致（RankSlicedPool 切片前校验同款理由）；
        # 条件集 = 本域条件名清单（#129 经词汇表装配注入）；切片视图供
        # RealPoolSampler 消费（分布式 = 本 rank 条带切片，单进程 = 全池恒等）
        vocabulary = cls.assemble_vocabulary(config)
        real_pool = LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )
        real_pool.assert_condition_capacity(
            config.reward.disc_batch_size_k, dist.world_size,
            vocabulary.names(),
        )
        heldout_real = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        # 逐条件形状契约与活动词汇表的装配期对照（#129 消费侧守卫）：
        # fake 侧形状经 vocabulary.latent_shape(name) 解析、real 侧经
        # manifest 的 condition_latent_shapes 装载，两来源同名异形（词表
        # 工件改动而 manifest 未重建）此前只在首次判别器拼接时才炸——
        # 装配期显式拒绝（held-out 侧同款：分簇/AUC 的 real 侧同源）
        real_pool.assert_condition_shapes(vocabulary)
        heldout_real.assert_condition_shapes(vocabulary)
        real_view = RankSlicedPool(real_pool, dist, vocabulary.names()).view()
        update = OnlineUpdate(
            scorer=scorer,
            buffer=ReplayBuffer(config.reward.replay_buffer_capacity),
            real_sampler=RealPoolSampler(
                real_view,
                generators["real_pool"],
                amp.device,
            ),
            config=config.reward,
            generator=generators["disc_update"],
            # 训练期噪声注入的专属随机流（ADR-0009-α）：与回放抽样等
            # 训练采样流不交叉——σ_max 取值不漂移其余流的序列；σ_max = 0
            # 时该流零消耗（回归锚）。预训练 driver 经本装配缝自动同口径
            noise_generator=generators["disc_noise"],
        )
        auc = HeldOutAuc(
            heldout_manifest=heldout_real,
            scorer=scorer,
            generator=generators["heldout_auc"],
            device=amp.device,
        )
        # 条件白名单的动态运行时对象（ADR-0008 决策 5/8）：train 新 run =
        # 报告白名单起步（gate 产物）+ 实测快照；resume/预训练冷启动 =
        # 全条件放行占位（恢复点不重查白名单，恢复应用时分片的门控状态
        # 整体覆写；driver 自产 per-condition 判定不消费本名单）。条件闸
        # 关闭（``condition_gate_enabled=false``，维护者裁决）= 白名单退化为
        # 「不设条件闸」的全条件放行占位——报告仍装载（warm-start 权重
        # 是 ADR-0007 的另一件事），但其白名单不作上岗判据也不作更新开关。
        # EMA 动态恢复（决策 8）的名单变更在训练循环内经 observe 驱动
        gate_active = config.reward.condition_gate_enabled
        gating = DynamicWhitelist(
            initial=(
                ConditionWhitelist.from_report(report)
                if report is not None and gate_active
                else ConditionWhitelist.unrestricted(vocabulary.names())
            ),
            config=config.reward,
            dist=dist,
            conditions=vocabulary.names(),
        )
        return RewardCoordinator(
            update, auc, generators["fake_shuffle"], gating,
            overfit=OverfitMonitor(
                config.reward, conditions=vocabulary.names(),
            ),
        )

    @staticmethod
    def _assemble_scorer(
        config: CynosureConfig,
        report: PretrainReport | None,
        *,
        resume: bool = False,
    ) -> RewardScorer:
        """判别器 scorer 的权重来源分派（见 ``assemble_rewards``）。"""
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
