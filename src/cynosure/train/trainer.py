"""单进程 Granular-GRPO 训练循环（ticket #21 tracer bullet，#23 扩展到
组2；组3 的两阶段序贯由 SequentialTrainer 编排本类）。

spec #15 执行序的单进程版（分布式 allreduce/barrier 留 T09）：

    每 iteration：
      1. eval() + no_grad —— Rollout 与打分（RolloutPhase）
      2. train() —— 逐 k 独立 forward→backward→optimizer.step（|M| 次）；
         判别器 Online update（50% 当前 fake + 50% 回放，单进程直通）
      3. iter 事件落盘（Anchor eval reward / 组内 reward std / held-out
         AUC / loss 组件 / buffer 占比 / lr / 耗时；组3 stage-2 带 stage=2）
    定期：续训状态全清单落盘（resume 模块，T07）——与产物 checkpoint 同
    节奏（checkpoint_interval 周期 + 每里程碑强制 + 收尾兜底）
    train 启动时：用冻结初始 policy 自动生成 buffer base 分区（续训恢复
    时跳过——buffer 两区内容随状态整体回归），并对 Baseline manifest 条目
    采样落盘（冻结只采一次，续训恢复时同样跳过——恢复点 policy 已非初始
    权重）；到达里程碑间隔时触发评测相的解码评测 → ``milestone`` 事件
    写入同一指标流 → train 进程内早停判定消费该流（plateau / hacking
    签名命中即停）；训练结束后对同 manifest 条目 RL 后重采。**解码只在
    Baseline/里程碑/重采三条评测路径**（ADR-0004，结构断言见测试面）。

可训练对象按组装配（GroupPolicy）：组1 = UNet 全参、组2 = ControlNet
（base UNet 冻结经装配期断言验证）。每组一次装配 → 判别器与 Replay
buffer 随训练实例天然隔离（跨组/跨阶段不复用）。

（dist.barrier() 属 T09。）
"""

import time
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict

from cynosure.config import CynosureConfig, Modality
from cynosure.eval import EvaluationPhase, MilestoneMetrics
from cynosure.grpo import ClippedPolicyLoss, MgaiAdvantage, StepwisePolicyUpdate
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import VelocityField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer, ReplayStore
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate, UpdateReport
from cynosure.train.artifacts import (
    BaselineManifest,
    IterEvent,
    MilestoneEvent,
    POLICY_CHECKPOINT_TEMPLATE,
    RunArtifacts,
)
from cynosure.train.earlystop import EarlyStopJudge
from cynosure.train.policy import GroupPolicy
from cynosure.train.rollout import IterationRollout, RolloutPhase
from cynosure.train.resume import resume_latest, save_resume_state


@dataclass(frozen=True)
class StageTag:
    """本阶段在 run 目录内的产物归属（组3 两阶段隔离的载体）：指标事件
    的 stage 号 + checkpoint 文件名前缀。单阶段组（组1/组2）用默认值
    （stage 1、无前缀），产物名与历史布局逐字一致。"""

    stage: int = 1
    checkpoint_prefix: str = ""


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

    def heldout_auc(
        self, current_fakes: torch.Tensor, modality: Modality,
    ) -> float:
        """held-out real vs 当前 fake 的判别器 AUC（hacking 监控信号）。

        real 侧按本 iteration 采样的目标序列过滤——iter 事件按序列归因
        reward/loss/AUC，混采会让其他序列的判别器分数偏移伪装成本序列
        realism 变化（per-target-sequence 健康监控）。"""
        return self.auc.compute(current_fakes, modality=modality)


class GranularGrpoTrainer:
    """单进程 Granular-GRPO 训练循环编排（config 驱动装配；一次装配 =
    一个组的一个阶段）。"""

    def __init__(
        self,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        dump_trajectory: bool = False,
        device: torch.device | None = None,
        stage: StageTag | None = None,
        evaluation: EvaluationPhase | None = None,
    ) -> None:
        if config.experiment.group == "sequential":
            raise ValueError(
                "组3（sequential）的两阶段序贯由 SequentialTrainer 编排："
                "stage-1/阶段-2 各自是一次本类的单阶段运行"
            )
        if config.policy.sde_eta <= 0.0:
            raise ValueError(
                "η=0 是确定性步、无高斯密度可求（log-prob 仅在扰动步有意义）:"
                "训练循环需要 η>0 才存在 policy gradient；η=0 对照属纯诊断路径"
                "（trajectory.json 仍随 --dump-trajectory 产出）",
            )
        if config.grpo.ema_anchor_enabled:
            # EMA 锚是升级项（ADR-0001：hacking 签名出现时启用）且实现未
            # 交付：静默忽略会让续训状态清单的 EMA 条件项失真
            raise ValueError(
                "参数 EMA 锚为升级项（ADR-0001），实现未交付："
                "ema_anchor_enabled=true 显式拒绝"
            )
        # 回放供给跨字段守卫：首次判别器更新时近期分区为空，回放半区
        # （floor(K/2) 条，50/50 混采定死）全由 base 分区（capacity//2）
        # 承担；K=1 则回放半区为 0 条、回放采样 API 直接拒绝。无效组合
        # 在装配期显式拒绝，而非让昂贵 rollout 先行、更新时才缺样本
        replay_count = config.reward.disc_batch_size_k // 2
        if replay_count < 1:
            raise ValueError(
                f"回放供给不足：disc_batch_size_k={config.reward.disc_batch_size_k}"
                " 的回放半区为 0 条（K 须 ≥2）"
            )
        if config.reward.replay_buffer_capacity // 2 < replay_count:
            raise ValueError(
                "回放供给不足：replay_buffer_capacity="
                f"{config.reward.replay_buffer_capacity} 的 base 分区仅 "
                f"{config.reward.replay_buffer_capacity // 2} 条，不足以承担"
                f"首次判别器更新的回放半区 {replay_count} 条"
                f"（disc_batch_size_k={config.reward.disc_batch_size_k}）"
            )
        self.config = config
        self.artifacts = run_artifacts
        self._dump = dump_trajectory
        self.stage_tag = stage if stage is not None else StageTag()
        # 装配期单点选设备（local rank 0 的加速器）；分布式 sharding 属 T09。
        # 所有模型与 rollout/打分张量随该设备放置——autocast(device_type)
        # 只影响前向 dtype，不移动任何张量
        self._amp = AmpContext(
            device=(
                device if device is not None
                else torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu",
                )
            ),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        # 全循环的六条命名 RNG 流（续训状态机按名保存/恢复的注册表）：
        # rollout 相与条件分布共享主流；real 采样 / 判别器更新 / AUC /
        # fake 置换 / base 分区生成各自独立派生流（互不漂移）
        seed = config.schedule.seed
        self.generators: dict[str, torch.Generator] = {
            "rollout": torch.Generator().manual_seed(seed),
            "real_pool": torch.Generator().manual_seed(seed + 1),
            "disc_update": torch.Generator().manual_seed(seed + 2),
            "heldout_auc": torch.Generator().manual_seed(seed + 3),
            "fake_shuffle": torch.Generator().manual_seed(seed + 4),
            "base_partition": torch.Generator().manual_seed(seed + 5),
        }
        self.policy = GroupPolicy.build(
            config, self.generators["rollout"], self._amp.device,
        )
        sampler = self._assemble_sampler(self.policy.field)
        self.rewards = self._assemble_rewards()
        self.updater = StepwisePolicyUpdate(
            sampler=sampler,
            optimizer=self.policy.optimizer,
            loss=ClippedPolicyLoss(clip_range=config.policy.ratio_clip),
            device_type=self._amp.device_type,
            amp_dtype=self._amp.dtype,
        )
        self.rollout = RolloutPhase(
            config,
            sampler,
            self.rewards.update.scorer,
            self.generators["rollout"],
            condition_sampler=self.policy.conditions,
            device_type=self._amp.device_type,
            autocast_dtype=self._amp.dtype,
            device=self._amp.device,
            # base 分区种子生成的独立派生流（seed+5）：其抽取数随 buffer
            # 容量变化，不占训练 rollout 的抽样流（容量实验不漂移样本流）
            base_generator=self.generators["base_partition"],
        )
        # 评测相（Baseline 采样 / 里程碑解码评测 / RL 后重采）；测试可注入替身。
        # manifest 由本侧从 run 目录装载注入（eval 不反向依赖 train 契约模块）：
        # 组3 stage-2 各自重读盘上 manifest，天然含 stage-1 已回写的样本路径
        self.evaluation = evaluation if evaluation is not None else (
            EvaluationPhase.build(
                config,
                run_artifacts,
                sampler,
                self.stage_tag.stage,
                BaselineManifest.load(run_artifacts.paths.manifest),
                amp=self._amp,
            )
        )

    @property
    def unet(self) -> torch.nn.Module:
        """base UNet（组1 即可训练 policy 本体；组2/组3-stage2 为冻结 base，
        经 GroupPolicy 装配期断言验证 requires_grad 全关）。"""
        return self.policy.unet

    @property
    def device(self) -> torch.device:
        """装配期单点选定的训练设备（续训状态恢复的迁入目标）。"""
        return self._amp.device

    def seed_base_partition(self) -> None:
        """train 启动期的 buffer base 分区自动生成：用冻结初始 policy
        （未参与任何梯度步）rollout 产出填满 base 分区（spec 补钉）。"""
        self.rewards.seed_base(
            self.rollout.base_partition_samples(
                self.rewards.buffer.base_capacity,
            ),
        )

    def run(self, *, resume: bool = False) -> int:
        """训练主循环：base 分区自动生成 → Baseline 采样（冻结初始
        policy）→ 逐 iteration 执行序（里程碑触发解码评测 + 早停判定）
        → RL 后重采 → checkpoint 与续训状态落盘。``resume=True`` 时从
        run 目录最新续训状态恢复（全清单覆写，resume 模块），回退指标流
        中恢复点之后的半截事件后从恢复点继续——base 分区种子生成与
        Baseline 采样随之跳过（buffer 随状态整体回归；Baseline 冻结只采
        一次，恢复点 policy 已非初始权重）。返回完成的 iteration 数
        （config 口径的累计完成数；早停时小于 max_iterations）。"""
        start_iteration = 0
        if resume:
            start_iteration = resume_latest(self)
            self.artifacts.rewind_events(start_iteration, self.stage_tag.stage)
        self.policy.eval_phase()  # base 分区生成与 rollout 同为 eval 相（执行序第 1 相口径）
        self.rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        if not resume:
            self.seed_base_partition()
            self.evaluation.sample_baseline()  # 冻结只采一次：更新开始前的当前权重即初始 policy
        pairs: list[TrainingLogProbPair] = []
        last_checkpoint = start_iteration
        completed = 0
        update_interval = self.config.reward.disc_update_interval_n_d
        for iteration in range(start_iteration, self.config.schedule.max_iterations):
            started = time.monotonic()
            self.policy.eval_phase()  # 执行序第 1 相：eval() + no_grad 的 Rollout
            record = self.rollout.run_iteration()
            if self._dump:
                pairs.extend(self._consistency_pairs(record, iteration))
            self.policy.train_phase()  # 执行序第 2 相：train() 逐 k 更新（冻结 base 恒 eval）
            loss_terms = self._update_policy(record)
            # held-out AUC 在判别器更新之前测得：与 anchor_eval_reward 同一
            # 判别器快照（更新后测同一 fake 批会把 in-sample 拟合计入 AUC，
            # 联合 hacking 签名失真）；real 侧按本 iteration 采样的目标
            # 序列过滤（per-target-sequence 归因，#40）
            heldout_auc = self.rewards.heldout_auc(
                record.new_fakes, record.modality,
            )
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
                stage=self.stage_tag.stage,
                modality=record.modality,
                anchor_eval_reward=record.anchor_eval_reward,
                intra_group_reward_std=record.intra_group_reward_std,
                heldout_auc=heldout_auc,
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
            completed = iteration + 1
            milestone_due = completed % self.config.schedule.milestone_interval == 0
            if (
                milestone_due
                or completed % self.config.schedule.checkpoint_interval == 0
            ):
                # checkpoint 周期之外，每个里程碑也强制落盘（config 契约：
                # milestone 评测器与恢复路径的取数点，周期不覆盖时仍须产出）
                self._write_checkpoint(completed)
                save_resume_state(self, completed)
                last_checkpoint = completed
            if milestone_due:
                # 里程碑评测相与 rollout/baseline 同为 eval（相位敏感层的
                # 口径一致——baseline/里程碑/重采三条评测路径同场采样）
                self.policy.eval_phase()
                if self._run_milestone(completed):
                    break  # 早停：早停签名命中，最终 policy 状态已随里程碑 checkpoint 落盘
        if last_checkpoint < completed:
            # 收尾兜底只允许前向推进：恢复点已在目标之后（收缩 max_iterations
            # 的续训 = 无操作）时不得把更后的训练态改写成更小的 iteration 标签
            self._write_checkpoint(completed)
            save_resume_state(self, completed)
        self.policy.eval_phase()  # 重采与 baseline 同为 eval 相（差异唯一归因于 RL）
        self.evaluation.resample()  # RL 后同 seed 同条件重采（共用同一 manifest）
        if pairs:
            self.artifacts.paths.training_diagnostic.write_text(
                TrainingDiagnostic(logprob_pairs=pairs).model_dump_json(indent=2),
                encoding="utf-8",
            )
        return completed

    def _run_milestone(self, iteration: int) -> bool:
        """里程碑解码评测 → ``milestone`` 事件写入训练指标流 → 早停判定
        消费该流。返回是否早停（解码评测只发生在本路径，不进逐 iteration
        循环；本阶段的早停判定只消费本阶段的流前缀——组3 两阶段事件同流
        存放，按 stage 过滤防跨阶段串扰）。"""
        metrics: MilestoneMetrics = self.evaluation.milestone_metrics()
        stage_events = [
            event for event in self.artifacts.read_events()
            if event.get("stage", 1) == self.stage_tag.stage
        ]
        verdict = EarlyStopJudge(self.config).judge(
            stage_events, current_fid=metrics.fid,
        )
        criteria = dict(metrics.summary())
        criteria["plateau_stalled"] = float(verdict.plateau_stalled)
        criteria["hacking_signature"] = float(verdict.hacking_signature)
        self.artifacts.append_event(MilestoneEvent(
            iteration=iteration,
            stage=self.stage_tag.stage,
            fid=metrics.fid,
            kid=metrics.kid,
            ssim=metrics.ssim,
            mae=metrics.mae,
            psnr=metrics.psnr,
            criteria_summary=criteria,
            early_stop=verdict.stop,
            early_stop_reason=verdict.reason,
        ))
        return verdict.stop

    def _assemble_sampler(self, field: VelocityField) -> RolloutSampler:
        """policy 采样封装装配（netbuild 日程 + 本组采样场 + SDE 核）。"""
        policy = self.config.policy
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=policy.num_inference_steps,
            input_img_size_numel=policy.input_img_size_numel,
        )
        kernel = SdeKernel(eta=policy.sde_eta, s_max=policy.sde_s_max)
        return RolloutSampler(field, kernel, TrajectoryCursor(scheduler))

    def _assemble_rewards(self) -> RewardCoordinator:
        """判别器侧装配（T05 管线的同一组合方式）。"""
        config = self.config
        if config.artifacts.discriminator_config_json is None:
            raise ValueError(
                "训练循环需要判别器网络配置（discriminator_config_json）："
                "在线 reward model 的装配源（discriminator_ckpt 缺省 = "
                "随机初始化起步的在线训练，冷启动工作流）"
            )
        # 网络构建（含冷启动随机初始化）在 schedule.seed 的派生流下进行，
        # 并 fork 隔离全局 RNG——同 config 的两次冷启动判别器权重逐位
        # 可复现，且不扰动进程全局 RNG 状态（sampling generators 独立
        # 对象本就不受影响）
        with torch.random.fork_rng():
            torch.manual_seed(config.schedule.seed + 6)
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
        scorer.to(self._amp.device)  # 单点递归迁移：判别器参数 + 统计量 buffer
        update = OnlineUpdate(
            scorer=scorer,
            buffer=ReplayBuffer(config.reward.replay_buffer_capacity),
            real_sampler=RealPoolSampler(
                LatentManifest.load(
                    config.reward.real_pool_manifest, kind="real_pool",
                ),
                self.generators["real_pool"],
                self._amp.device,
            ),
            config=config.reward,
            generator=self.generators["disc_update"],
        )
        auc = HeldOutAuc(
            heldout_manifest=LatentManifest.load(
                config.reward.heldout_real_manifest, kind="heldout_real",
            ),
            scorer=scorer,
            generator=self.generators["heldout_auc"],
            device=self._amp.device,
        )
        return RewardCoordinator(update, auc, self.generators["fake_shuffle"])

    def _update_policy(self, record: IterationRollout) -> dict[str, float]:
        """逐 k 独立梯度步（执行序第 2 相）：每 k 一次
        forward→backward→optimizer.step，返回逐 k loss 组件。"""
        advantage = MgaiAdvantage(clamp=self.config.grpo.advantage_clamp)
        loss_terms: dict[str, float] = {}
        for step in record.steps:
            advantages = advantage.compute(step.rewards)
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
            self._amp.device_type, dtype=self._amp.dtype,
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
        """policy（本组可训练网络）与判别器权重落盘（可装载 state_dict）。
        判别器经 loadable_state_dict 固化有效权重（spectral norm 启用时
        仍可严格重载）；stage 前缀隔离组3 两阶段的同名产物（stage-1 无
        前缀 = 历史布局逐字一致）。续训全状态由 resume.save_resume_state
        同节奏落盘。"""
        prefix = self.stage_tag.checkpoint_prefix
        torch.save(
            self.policy.network.state_dict(),
            self.artifacts.paths.checkpoints / (
                f"{prefix}{POLICY_CHECKPOINT_TEMPLATE.format(iteration=iteration)}"
            ),
        )
        torch.save(
            NetworkAssembler.loadable_state_dict(self.rewards.discriminator),
            self.artifacts.paths.checkpoints
            / f"{prefix}discriminator_iter{iteration}.pt",
        )
