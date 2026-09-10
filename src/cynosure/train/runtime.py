"""一次装配的训练运行时（spec #15「一次装配 = 一个组的一个阶段」的
组件组 Facade）。

config 驱动的装配产物收敛：policy 侧（GroupPolicy）、判别器侧
（RewardCoordinator）、逐 k 更新（StepwisePolicyUpdate）、rollout 相
（RolloutPhase）、六条命名 RNG 流（TrainingRngStreams 注册表）、数值
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
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import VelocityField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.pretrain.artifacts import PretrainReport
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.policy import GroupPolicy
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import RolloutPhase
from cynosure.train.rng import TrainingRngStreams

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
    ) -> "TrainingRuntime":
        """config 驱动装配：单进程与分布式同一条装配序，分布式包装点
        （FSDP/DDP/切片/归并/seed 派生）在 world-1 下恒等。"""
        dist = dist_context if dist_context is not None else DistributedContext.bootstrap()
        # seed 的 rank 派生：六条流的演化各 rank 独立（rollout 数据多样性
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
            report=PretrainReport.load(config.reward.pretrain_report_json),
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
    def assemble_sampler(cls, config: CynosureConfig, field: VelocityField) -> RolloutSampler:
        """policy 采样封装装配（netbuild 日程 + 本组采样场 + SDE 核）。

        公开装配缝：train 运行时与预训练 driver（单进程 world-1 语境）
        共用同一份装配代码——采样日程/核参数的调整单点生效。"""
        policy = config.policy
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=policy.num_inference_steps,
            input_img_size_numel=policy.input_img_size_numel,
        )
        kernel = SdeKernel(eta=policy.sde_eta, s_max=policy.sde_s_max)
        return RolloutSampler(field, kernel, TrajectoryCursor(scheduler))

    @classmethod
    def assemble_rewards(
        cls,
        config: CynosureConfig,
        amp: AmpContext,
        generators: dict[str, torch.Generator],
        dist: DistributedContext,
        report: PretrainReport | None = None,
    ) -> RewardCoordinator:
        """判别器侧装配：网络构建 → DDP 副本升级（分布式）→ pool 切片
        （real 侧；held-out 不切）→ Online update / AUC 协作者。

        公开装配缝：train 运行时与预训练 driver（world-1 退化语境——
        RankSlicedPool / ReplicatedDiscriminator 在单进程下恒等）共用
        同一份装配代码——「无第二套判别器训练逻辑」在装配层同样成立。

        权重来源按语境二分：``report`` 给定（train 语境）= warm-start
        守卫重载（数据口径指纹对照 → 形态指纹对照 → 报告 checkpoint
        严格装载，ADR-0007——RL 不带预训练产物在装配层就无法启动）；
        ``None``（pretrain driver 语境）= 冷启动路径（checkpoint 工件
        或随机初始化，预训练本身即产物的生产方）。"""
        if config.artifacts.discriminator_config_json is None:
            raise ValueError(
                "训练循环需要判别器网络配置（discriminator_config_json）："
                "在线 reward model 的装配源（discriminator_ckpt 缺省 = "
                "随机初始化起步的在线训练，冷启动工作流）"
            )
        scorer = cls._assemble_scorer(config, report)
        scorer.to(amp.device)  # 单点递归迁移：判别器参数 + 统计量 buffer
        ReplicatedDiscriminator.replicate(scorer, dist)
        update = OnlineUpdate(
            scorer=scorer,
            buffer=ReplayBuffer(config.reward.replay_buffer_capacity),
            real_sampler=RealPoolSampler(
                RankSlicedPool(
                    LatentManifest.load(
                        config.reward.real_pool_manifest, kind="real_pool",
                    ),
                    dist,
                ).view(),
                generators["real_pool"],
                amp.device,
            ),
            config=config.reward,
            generator=generators["disc_update"],
        )
        auc = HeldOutAuc(
            heldout_manifest=LatentManifest.load(
                config.reward.heldout_real_manifest, kind="heldout_real",
            ),
            scorer=scorer,
            generator=generators["heldout_auc"],
            device=amp.device,
        )
        return RewardCoordinator(update, auc, generators["fake_shuffle"])

    @staticmethod
    def _assemble_scorer(
        config: CynosureConfig, report: PretrainReport | None,
    ) -> RewardScorer:
        """判别器 scorer 的权重来源分派（见 ``assemble_rewards``）。"""
        if report is not None:
            # warm-start 守卫链：数据口径指纹 → 形态指纹 → 严格装载
            # （任一不符在装配期拒绝，CLI 层回滚 run 目录）
            report.assert_data_provenance(config)
            return report.load_discriminator(config)
        # 冷启动：网络构建（随机初始化或 checkpoint 工件装载）在
        # schedule.seed 的派生流下进行，并 fork 隔离全局 RNG——同 config
        # 的两次冷启动判别器权重逐位可复现，且跨 rank 逐位一致（seed
        # 不含 rank 偏移：DDP 装配要求各 rank 初始副本一致）。不扰动
        # 进程全局 RNG 状态（sampling generators 独立对象本就不受影响）
        with torch.random.fork_rng():
            torch.manual_seed(config.schedule.seed + 6)
            return RewardScorer(
                NetworkArtifact(
                    config=NetworkAssembler.load_json(
                        config.artifacts.discriminator_config_json,
                    ),
                    checkpoint=config.artifacts.discriminator_ckpt,
                ),
                config.reward,
                ChannelStats.load(config.reward.channel_stats_json),
            )
