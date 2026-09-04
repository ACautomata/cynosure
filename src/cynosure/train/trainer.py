"""单进程 Granular-GRPO 训练循环（ticket #21 tracer bullet）。

spec #15 执行序的单进程版（分布式 allreduce/barrier 留 T09）：

    每 iteration：
      1. eval() + no_grad —— Rollout 与打分（RolloutPhase）
      2. train() —— 逐 k 独立 forward→backward→optimizer.step（|M| 次）；
         判别器 Online update（50% 当前 fake + 50% 回放，单进程直通）
      3. iter 事件落盘（Anchor eval reward / 组内 reward std / held-out
         AUC / loss 组件 / buffer 占比 / lr / 耗时）
    train 启动时：用冻结初始 policy 自动生成 buffer base 分区。

（dist.barrier() 与断点续训状态机分属 T09 / T07。）
"""

import time

import torch
from pydantic import BaseModel, ConfigDict

from cynosure.config import CynosureConfig
from cynosure.grpo import ClippedPolicyLoss, MgaiAdvantage, StepwisePolicyUpdate
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer, ReplayStore
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate, UpdateReport
from cynosure.train.artifacts import IterEvent, RunArtifacts
from cynosure.train.rollout import IterationRollout, RolloutPhase

AMP_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16}
"""config amp_dtype（Literal["bf16"] 定死）→ torch autocast dtype。"""


class TrainingLogProbPair(BaseModel):
    """训练侧 log-prob 对：rollout 记录的 π_old vs 更新时（同权重、
    更新循环开始前）重算值——测试面 #3 的诊断载体。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    iteration: int
    step_index: int
    direction: int
    recorded: float
    recomputed: float


class TrainingDiagnostic(BaseModel):
    """训练诊断工件（run 目录 ``training.json`` 的契约；--dump-trajectory
    时产出。字段为最小集，施工可扩不可改名）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    logprob_pairs: list[TrainingLogProbPair]


class RewardCoordinator:
    """判别器侧协作者组（Facade）：聚合两区缓冲、Online update 与
    held-out AUC——trainer 只面对「种植/更新/AUC」三个动作与判别器引用。

    fake 供给与判别器相位约定：更新批的当前半区从全批 fake 随机抽取
    （rollout 产出按 (k, λ) 有序堆叠，确定性取头部会使 K=4 的当前半区
    永远只见最小 step、λ=1 的头部方向）；判别器默认保持 eval 相（打分
    与监控前向不得推进 spectral norm power iteration），仅更新一步
    期间短暂 train。"""

    def __init__(
        self, update: OnlineUpdate, auc: HeldOutAuc,
        generator: torch.Generator,
    ) -> None:
        self.update = update
        self.auc = auc
        self._generator = generator

    @property
    def buffer(self) -> ReplayStore:
        """两区回放缓冲（Online update 的混采源，单点持有）。"""
        return self.update.buffer

    @property
    def discriminator(self) -> torch.nn.Module:
        """底层判别器（checkpoint 落盘用）。"""
        return self.update.scorer.discriminator

    def seed_base(self, samples: torch.Tensor) -> None:
        """冻结初始 policy 的产出填充 base 分区（train 启动期一次）。"""
        self.buffer.fill_base(samples)

    def update_step(self, current_fakes: torch.Tensor) -> UpdateReport:
        """判别器 Online update 一步：全批 fake 随机置换后交更新
        （50% 当前 / 50% 回放的混采由 update 消费置换批的头部），更新
        期间判别器 train 相、结束后恢复 eval 相。置换过的整批照常入
        近期分区（近期分布记录是集合语义，次序无关）。"""
        order = torch.randperm(current_fakes.shape[0], generator=self._generator)
        shuffled = current_fakes[order]
        self.discriminator.train()
        try:
            return self.update.step(shuffled)
        finally:
            self.discriminator.eval()

    def heldout_auc(self, current_fakes: torch.Tensor) -> float:
        """held-out real vs 当前 fake 的判别器 AUC（hacking 监控信号）。"""
        return self.auc.compute(current_fakes)


class GranularGrpoTrainer:
    """单进程 Granular-GRPO 训练循环编排（config 驱动装配）。"""

    def __init__(
        self,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        dump_trajectory: bool = False,
        device: torch.device | None = None,
    ) -> None:
        if config.experiment.group != "modal-label":
            raise ValueError(
                f"实验组 {config.experiment.group} 的采样场（ControlNet / "
                "序贯衔接）由后续 ticket 交付，当前仅支持 modal-label"
            )
        if config.policy.sde_eta <= 0.0:
            raise ValueError(
                "η=0 是确定性步、无高斯密度可求（log-prob 仅在扰动步有意义）:"
                "训练循环需要 η>0 才存在 policy gradient；η=0 对照属纯诊断路径"
                "（trajectory.json 仍随 --dump-trajectory 产出）",
            )
        self.config = config
        self.artifacts = run_artifacts
        self.advantage = MgaiAdvantage(clamp=config.grpo.advantage_clamp)
        self._dump = dump_trajectory
        # 装配期单点选设备（local rank 0 的加速器）；分布式 sharding 属 T09。
        # 所有模型与 rollout/打分张量随该设备放置——autocast(device_type)
        # 只影响前向 dtype，不移动任何张量
        self._device = (
            device if device is not None
            else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self._amp_dtype = AMP_DTYPES[config.policy.amp_dtype]

        self.unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=config.artifacts.unet_ckpt,
        )).to(self._device)
        sampler = self._assemble_sampler()
        self.rewards = self._assemble_rewards()
        self.updater = StepwisePolicyUpdate(
            sampler=sampler,
            optimizer=torch.optim.AdamW(
                self.unet.parameters(),
                lr=config.policy.policy_lr,
                weight_decay=config.policy.policy_weight_decay,
            ),
            loss=ClippedPolicyLoss(clip_range=config.policy.ratio_clip),
            device_type=self._device.type,
            amp_dtype=self._amp_dtype,
        )
        self.rollout = RolloutPhase(
            config,
            sampler,
            self.rewards.update.scorer,
            torch.Generator().manual_seed(config.schedule.seed),
            device_type=self._device.type,
            autocast_dtype=self._amp_dtype,
            device=self._device,
        )

    def seed_base_partition(self) -> None:
        """train 启动期的 buffer base 分区自动生成：用冻结初始 policy
        （未参与任何梯度步）rollout 产出填满 base 分区（spec 补钉）。"""
        self.rewards.seed_base(
            self.rollout.base_partition_samples(
                self.rewards.buffer.base_capacity,
            ),
        )

    def run(self) -> int:
        """训练主循环：base 分区自动生成 → 逐 iteration 执行序 → checkpoint。
        返回完成的 iteration 数。"""
        self.unet.eval()  # base 分区生成与 rollout 同为 eval 相（执行序第 1 相口径）
        self.rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        self.seed_base_partition()
        pairs: list[TrainingLogProbPair] = []
        last_checkpoint = 0
        update_interval = self.config.reward.disc_update_interval_n_d
        for iteration in range(self.config.schedule.max_iterations):
            started = time.monotonic()
            self.unet.eval()  # 执行序第 1 相：eval() + no_grad 的 Rollout
            record = self.rollout.run_iteration()
            if self._dump:
                pairs.extend(self._consistency_pairs(record, iteration))
            self.unet.train()  # 执行序第 2 相：train() 逐 k 更新
            loss_terms = self._update_policy(record)
            # 判别器 Online update 按 N_d 节奏（每 N_d 个 iteration 一步，
            # D:G 更新比 ≈ 1:1 由 N_d=1 默认落实；跳过的 iteration 不动判别器）
            report = (
                self.rewards.update_step(record.new_fakes)
                if iteration % update_interval == 0 else None
            )
            if report is not None:
                loss_terms["discriminator"] = report.loss_discriminator
            batch_size_k = self.config.reward.disc_batch_size_k
            zone_sizes = self.rewards.buffer.zone_sizes()
            self.artifacts.append_event(IterEvent(
                iteration=iteration,
                modality=record.modality,
                anchor_eval_reward=record.anchor_eval_reward,
                intra_group_reward_std=record.intra_group_reward_std,
                heldout_auc=self.rewards.heldout_auc(record.new_fakes),
                loss=loss_terms,
                buffer_current_fraction=(
                    report.num_current / batch_size_k if report else 0.0
                ),
                buffer_replay_fraction=(
                    report.num_replay / batch_size_k if report else 0.0
                ),
                buffer_base_occupied=zone_sizes.base,
                buffer_recent_occupied=zone_sizes.recent,
                lr=self.config.policy.policy_lr,
                elapsed_s=time.monotonic() - started,
            ))
            if (
                iteration + 1
            ) % self.config.schedule.checkpoint_interval == 0:
                self._write_checkpoint(iteration + 1)
                last_checkpoint = iteration + 1
        if last_checkpoint != self.config.schedule.max_iterations:
            self._write_checkpoint(self.config.schedule.max_iterations)
        if pairs:
            self.artifacts.paths.training_diagnostic.write_text(
                TrainingDiagnostic(logprob_pairs=pairs).model_dump_json(indent=2),
                encoding="utf-8",
            )
        return self.config.schedule.max_iterations

    def _assemble_sampler(self) -> RolloutSampler:
        """policy 采样封装装配（netbuild 日程 + 组合场 + SDE 核）。"""
        policy = self.config.policy
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=policy.num_inference_steps,
            input_img_size_numel=policy.input_img_size_numel,
        )
        kernel = SdeKernel(eta=policy.sde_eta, s_max=policy.sde_s_max)
        return RolloutSampler(
            CfgCombinedField(self.unet), kernel, TrajectoryCursor(scheduler),
        )

    def _assemble_rewards(self) -> RewardCoordinator:
        """判别器侧装配（T05 管线的同一组合方式）。"""
        config = self.config
        if (
            config.artifacts.discriminator_config_json is None
            or config.artifacts.discriminator_ckpt is None
        ):
            raise ValueError(
                "训练循环需要判别器网络工件（discriminator_config_json / "
                "discriminator_ckpt）：在线 reward model 的装配源"
            )
        scorer = RewardScorer(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.discriminator_config_json,
                ),
                checkpoint=config.artifacts.discriminator_ckpt,
            ),
            config.reward,
            ChannelStats.load(config.reward.channel_stats_json),
        )
        scorer.discriminator.to(self._device)
        seed = config.schedule.seed
        update = OnlineUpdate(
            scorer=scorer,
            buffer=ReplayBuffer(config.reward.replay_buffer_capacity),
            real_sampler=RealPoolSampler(
                LatentManifest.load(
                    config.reward.real_pool_manifest, kind="real_pool",
                ),
                torch.Generator().manual_seed(seed + 1),
                self._device,
            ),
            config=config.reward,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        auc = HeldOutAuc(
            heldout_manifest=LatentManifest.load(
                config.reward.heldout_real_manifest, kind="heldout_real",
            ),
            scorer=scorer,
            generator=torch.Generator().manual_seed(seed + 3),
            device=self._device,
        )
        return RewardCoordinator(
            update, auc, torch.Generator().manual_seed(seed + 4),
        )

    def _update_policy(self, record: IterationRollout) -> dict[str, float]:
        """逐 k 独立梯度步（执行序第 2 相）：每 k 一次
        forward→backward→optimizer.step，返回逐 k loss 组件。"""
        loss_terms: dict[str, float] = {}
        for step in record.steps:
            advantages = self.advantage.compute(step.rewards)
            value = self.updater.step(
                step_index=step.step_index,
                x_k=step.anchor_latent,
                condition=record.condition,
                directions=step.directions,
                old_log_probs=step.old_log_probs,
                advantages=advantages,
            )
            loss_terms[f"policy_step_{step.step_index}"] = value
        return loss_terms

    def _consistency_pairs(
        self, record: IterationRollout, iteration: int,
    ) -> list[TrainingLogProbPair]:
        """更新循环开始前（权重未变）逐 k 重算 log-prob，与 π_old 记录
        成对——同权重同口径，测试面 #3 断言两侧逐位一致。"""
        pairs: list[TrainingLogProbPair] = []
        with torch.no_grad(), torch.autocast(
            self._device.type, dtype=self._amp_dtype,
        ):
            for step in record.steps:
                recomputed = self.updater.sampler.evaluate_log_prob(
                    step.anchor_latent, step.step_index,
                    record.condition, step.directions,
                )
                for direction in range(self.config.policy.group_size_g):
                    pairs.append(TrainingLogProbPair(
                        iteration=iteration,
                        step_index=step.step_index,
                        direction=direction,
                        recorded=float(step.old_log_probs[direction]),
                        recomputed=float(recomputed[direction]),
                    ))
        return pairs

    def _write_checkpoint(self, iteration: int) -> None:
        """policy 与判别器权重落盘（可装载 state_dict；断点续训全状态
        （optimizer/RNG/buffer）属 T07）。"""
        torch.save(
            self.unet.state_dict(),
            self.artifacts.paths.checkpoints / f"policy_iter{iteration}.pt",
        )
        torch.save(
            self.rewards.discriminator.state_dict(),
            self.artifacts.paths.checkpoints / f"discriminator_iter{iteration}.pt",
        )
