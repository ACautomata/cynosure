"""Granular-GRPO 训练循环（ticket #21 tracer bullet 起；#24 分布式化）。

spec #15 执行序（单进程与 torchrun 多进程同一条路径，world-1 恒等）：

    每 iteration（每 rank，同卡交替）：
      1. eval() + no_grad —— Rollout 与打分（RolloutPhase）
      2. train() —— 逐 k 独立 forward→backward→optimizer.step（|M| 次，
         FSDP 梯度 allreduce）；判别器 Online update（本 rank fake +
         pool 切片 + 回放混采，DDP 梯度 allreduce）
      3. iter 事件归并（EventMerger：rank 0 顺序写出）→ dist.barrier()
    定期：续训状态全清单落盘（per-rank 文件）+ 产物 checkpoint
    （rank 0 独写，契约文件名不变）
    train 启动时：RM readiness gate 硬检查（per-rank base 分区由冻结
    初始 policy 生成后、按当前 run 数据口径重算 held-out AUC，不过线
    拒绝开跑并回滚——ADR-0007；续训恢复时跳过，resume 状态已含判别器
    全量状态）；Baseline manifest
    条目采样落盘只在 rank 0（冻结只采一次，续训恢复时同样跳过——恢复点
    policy 已非初始权重）；到达里程碑间隔时触发评测相的解码评测 →
    ``milestone`` 事件写入同一指标流（rank 0 独写）→ train 进程内早停
    判定消费该流（plateau / hacking 签名命中即停，verdict 广播各 rank
    一致 break）；训练结束后对同 manifest 条目 RL 后重采（rank 0）。
    **解码只在 Baseline/里程碑/重采三条评测路径**（ADR-0004，结构断言
    见测试面）。

装配（含分布式包装：FSDP full-shard + 梯度检查点、判别器 DDP、pool
切片、seed 的 rank 派生、指标归并器）收敛在 TrainingRuntime——循环
代码对部署形态无分支。可训练对象按组装配（GroupPolicy）；每组一次
装配 → 判别器与 Replay buffer 随训练实例天然隔离（per-rank buffer，
跨组/跨阶段不复用）。
"""

import time
from dataclasses import dataclass
from typing import Mapping

import torch
from pydantic import BaseModel, ConfigDict

from cynosure.config import CynosureConfig, Modality
from cynosure.distributed import DistributedContext
from cynosure.eval import EvaluationPhase, ManifestEvaluation, MilestoneMetrics
from cynosure.grpo import MgaiAdvantage, StepwisePolicyUpdate
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.numerics import AmpContext
from cynosure.reward.buffer import assert_replay_supply, base_condition_quota
from cynosure.train.artifacts import (
    BaselineManifest,
    IterEvent,
    MilestoneEvent,
    POLICY_CHECKPOINT_TEMPLATE,
    RunArtifacts,
)
from cynosure.train.earlystop import EarlyStopJudge
from cynosure.train.gate import ReadinessGate
from cynosure.train.policy import GroupPolicy
from cynosure.train.resume import ResumeStore
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import IterationRollout, RolloutPhase
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime


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


class IterationLoop:
    """单 iteration 执行序的策略侧两相（判别器侧见 RewardCoordinator）：
    eval 相 rollout（Anchor → 单步 SDE 扰动 → 各 λ ODE 续跑 → 打分）→
    train 相逐 k 独立梯度步（分布式下 FSDP 梯度 allreduce 同步权重）——
    两相共享采样场，由 trainer 逐 iteration 驱动；log-prob 一致性诊断
    （π_old 记录 vs 同权重重算）同属本循环面。"""

    def __init__(
        self,
        config: CynosureConfig,
        rollout: RolloutPhase,
        updater: StepwisePolicyUpdate,
        amp: AmpContext,
    ) -> None:
        self._config = config
        self.rollout = rollout
        self.updater = updater
        self._amp = amp

    def base_partition_samples(
        self, quota: Mapping[Modality, int],
    ) -> tuple[torch.Tensor, list[Modality]]:
        """冻结初始 policy 的 base 分区供给（train 启动期一次，按每条件
        配额量产；返回样本批 + 逐样本目标模态标签）。"""
        return self.rollout.base_partition_samples(quota)

    def run_iteration(self) -> IterationRollout:
        """执行序第 1 相：eval 相 rollout（调用方负责相位于前就绪）。"""
        return self.rollout.run_iteration()

    def update_policy(self, record: IterationRollout) -> dict[str, float]:
        """执行序第 2 相：逐 k 独立梯度步——每 k 一次
        forward→backward→optimizer.step（MGAI advantage 融合后交
        updater），返回逐 k loss 组件。"""
        advantage = MgaiAdvantage(clamp=self._config.grpo.advantage_clamp)
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

    def consistency_pairs(
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
                for direction in range(self._config.policy.group_size_g):
                    pairs.append(TrainingLogProbPair(
                        iteration=iteration,
                        step_index=step.step_index,
                        direction=direction,
                        recorded=float(step.old_log_probs[direction]),
                        recomputed=float(recomputed[direction]),
                    ))
        return pairs


class GranularGrpoTrainer:
    """Granular-GRPO 训练循环编排（config 驱动装配；一次装配 = 一个组
    的一个阶段）。分布式拓扑（ADR-0003）下各同构 rank 执行同一循环，
    权重同步经 FSDP/DDP 梯度 allreduce 自然生效，指标由 rank 0 归并。"""

    def __init__(
        self,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        dump_trajectory: bool = False,
        device: torch.device | None = None,
        stage: StageTag | None = None,
        dist_context: DistributedContext | None = None,
        evaluation: EvaluationPhase | None = None,
        resume: bool = False,
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
        # 回放供给装配期守卫（ADR-0008 决策 4，预训练 driver 同口径）：
        # 回放半区非零 + base 分区每条件配额 ≥ 回放半区需求（首次判别器
        # 更新时近期分区为空，按条件过滤的回放全量由 base 承担；无效
        # 组合在装配期显式拒绝，而非让昂贵 rollout 先行、更新时才缺样本）
        assert_replay_supply(config.reward)
        self.config = config
        self.artifacts = run_artifacts
        self._dump = dump_trajectory
        self.stage_tag = stage if stage is not None else StageTag()
        # resume 单点声明（装配与 run 执行共用同一开关）：resume 装配
        # 跳过 warm-start 报告装载、判别器占位冷启动（不消费任何
        # checkpoint 工件），权重被 ResumeStore.restore 整体覆写
        self._resume_assembled = resume
        self.runtime = TrainingRuntime.build(
            config, run_artifacts, device=device, dist_context=dist_context,
            resume=resume,
        )
        # 策略侧执行序两相（eval 相 rollout → train 相逐 k 更新）合成
        # 单 iteration 循环体（判别器侧协作者见 self.rewards）
        self.loop = IterationLoop(
            config=config,
            rollout=self.runtime.rollout,
            updater=self.runtime.updater,
            amp=self.runtime.amp,
        )
        # 续训状态分片存取（per-rank 落盘/恢复的单点，checkpoint 同节奏）
        self.resume_store = ResumeStore(
            run_artifacts.paths.checkpoints,
            self.stage_tag.checkpoint_prefix,
            self.runtime.dist,
        )
        # RM readiness gate（ADR-0007）：启动期的上岗硬检查——重算口径
        # 与分布式裁决原语在此装配（resume 跳过检查，见 run）
        self.readiness = ReadinessGate(
            config, self.runtime.rewards.auc, self.runtime.dist,
        )
        # 评测相（Baseline 采样 / 里程碑解码评测 / RL 后重采）；测试可注入替身。
        # manifest 由本侧从 run 目录装载注入（eval 不反向依赖 train 契约模块）：
        # 组3 stage-2 各自重读盘上 manifest，天然含 stage-1 已回写的样本路径。
        # 评测执行全 rank 参与（FSDP 集合前向），样本写盘闸门只开在 rank 0
        self.evaluation = evaluation if evaluation is not None else (
            ManifestEvaluation.build(
                config,
                run_artifacts,
                self.runtime.updater.sampler,
                self.stage_tag.stage,
                BaselineManifest.load(run_artifacts.paths.manifest),
                amp=self.runtime.amp,
                write_enabled=self.runtime.dist.rank == 0,
            )
        )

    # —— 既有公开访问面（tests 与 resume 模块消费；组件归 runtime 持有）——

    @property
    def unet(self) -> torch.nn.Module:
        """base UNet（组1 即可训练 policy 本体；组2/组3-stage2 为冻结 base，
        经 GroupPolicy 装配期断言验证 requires_grad 全关）。"""
        return self.policy.unet

    @property
    def policy(self) -> GroupPolicy:
        """本组 policy 侧装配（可训练网络 + 采样场 + 条件分布 + 优化器）。"""
        return self.runtime.policy

    @property
    def rewards(self) -> RewardCoordinator:
        """判别器侧协作者组（种植/更新/AUC）。"""
        return self.runtime.rewards

    @property
    def updater(self) -> StepwisePolicyUpdate:
        """逐 k 更新编排（log π 重算 → clipped loss → 优化器步）。"""
        return self.runtime.updater

    @property
    def rollout(self) -> RolloutPhase:
        """rollout 相编排（eval + no_grad 的执行序第 1 相）。"""
        return self.runtime.rollout

    @property
    def rng(self) -> TrainingRngStreams:
        """六条命名 RNG 流注册表（续训状态机按名保存/恢复的枚举面）。"""
        return self.runtime.rng

    @property
    def device(self) -> torch.device:
        """装配期单点选定的训练设备（续训状态恢复的迁入目标）。"""
        return self.runtime.amp.device

    @property
    def amp(self) -> AmpContext:
        """数值口径（device + autocast dtype）。"""
        return self.runtime.amp

    def seed_base_partition(self) -> torch.Tensor:
        """train 启动期的 buffer base 分区自动生成：用冻结初始 policy
        （未参与任何梯度步）rollout 产出按每条件配额填满 per-rank base
        分区（spec 补钉 + ADR-0008-01：每条件配额量产、条目带目标模态
        标签；per-rank buffer 覆盖各自区域，rollout 走本 rank 独立流）。
        返回量产的样本批（RM readiness gate 重算的 fake 侧输入——同
        一批量产既是 buffer 种子又是门槛重算口径，不二次消耗 RNG 流）。"""
        quota = base_condition_quota(
            self.config.reward.replay_buffer_capacity,
        )
        base_fakes, modalities = self.loop.base_partition_samples(quota)
        self.rewards.seed_base(base_fakes, modalities)
        return base_fakes

    def run(self) -> int:
        """训练主循环：base 分区自动生成 → RM readiness gate（ADR-0007，
        resume 跳过）→ Baseline 采样（rank 0，冻结
        初始 policy）→ 逐 iteration 执行序（里程碑触发解码评测 + 早停
        判定）→ RL 后重采（rank 0）→ checkpoint 与续训状态落盘。
        恢复语义由构造的 ``resume`` 单点声明（装配与执行共用同一开关，
        无双点声明可错位）：resume 构造时从 run 目录各 rank 的最新续训
        状态恢复（全清单覆写，resume 模块），rank 0 回退指标流中恢复点
        之后的半截事件后从恢复点继续——base 分区种子生成与 Baseline
        采样随之跳过（buffer 随状态整体回归；Baseline 冻结只采一次，
        恢复点 policy 已非初始权重）。恢复点已达标（无训练迭代）的续训
        是完整无操作：不重执行收官重采、不改写任何工件。返回完成的
        iteration 数（config 口径的累计完成数；早停时小于
        max_iterations；零训练迭代时 = 恢复点）。"""
        dist = self.runtime.dist
        resume = self._resume_assembled
        start_iteration = 0
        if resume:
            start_iteration = self.resume_store.restore(self)
            if dist.rank == 0:
                # 指标流回退只在 rank 0（RunArtifacts 的 rank 0 写盘契约）；
                # barrier 保证回退先于任何 rank 的下一事件追加
                self.artifacts.rewind_events(
                    start_iteration, self.stage_tag.stage,
                )
            dist.barrier()
        self.policy.eval_phase()  # base 分区生成与 rollout 同为 eval 相（执行序第 1 相口径）
        self.rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        if not resume:
            base_fakes = self.seed_base_partition()
            # RM readiness gate（ADR-0007）：warm-start 产物按当前 run
            # 数据口径（本 rank base fake 批 + held-out real）重算 AUC，
            # 不过线拒绝开跑——在 Baseline 采样等昂贵启动动作之前。
            # resume 跳过：续训状态已含判别器全量状态（恢复点判别器
            # 已在岗），门槛只把守「从预训练产物开跑」的新 run
            self.readiness.check(base_fakes)
            # Baseline 采样（冻结只采一次：更新开始前的当前权重即初始
            # policy）。policy 采样前向是 FSDP 集合操作，全 rank 对称
            # 参与；样本落盘与 manifest 回写是 rank 0 独写产物契约
            # （eval 侧 write_enabled 闸门）
            self.evaluation.sample_baseline()
        pairs: list[TrainingLogProbPair] = []
        last_checkpoint = start_iteration
        completed = start_iteration  # 零训练迭代的续训（恢复点已达标）报告恢复点本身
        update_interval = self.config.reward.disc_update_interval_n_d
        for iteration in range(start_iteration, self.config.schedule.max_iterations):
            started = time.monotonic()
            self.policy.eval_phase()  # 执行序第 1 相：eval() + no_grad 的 Rollout
            record = self.loop.run_iteration()
            if self._dump:
                pairs.extend(self.loop.consistency_pairs(record, iteration))
            self.policy.train_phase()  # 执行序第 2 相：train() 逐 k 更新（冻结 base 恒 eval）
            loss_terms = self.loop.update_policy(record)
            # held-out AUC 在判别器更新之前测得：与 anchor_eval_reward 同一
            # 判别器快照（更新后测同一 fake 批会把 in-sample 拟合计入 AUC，
            # 联合 hacking 签名失真）；real 侧按本 iteration 采样的目标
            # 序列过滤（per-target-sequence 归因，#40）
            heldout_auc = self.rewards.heldout_auc(
                record.new_fakes, record.modality,
            )
            # 判别器 Online update 按 N_d 节奏（每 N_d 个 iteration 一步，
            # D:G 更新比 ≈ 1:1 由 N_d=1 默认落实；跳过的 iteration 不动判别器）；
            # 本 iteration 的目标模态随 fake 批穿入——回放按条件过滤
            # （ADR-0008 决策 2），real 侧条件匹配归 ADR-0008-03
            report = (
                self.rewards.update_step(record.new_fakes, record.modality)
                if iteration % update_interval == 0 else None
            )
            if report is not None:
                loss_terms["discriminator"] = report.loss_discriminator
            batch_size_k = self.config.reward.disc_batch_size_k
            zone_sizes = self.rewards.buffer.zone_sizes()
            self.runtime.merger.emit(IterEvent(
                iteration=iteration,
                stage=self.stage_tag.stage,
                rank=dist.rank,
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
                self._checkpoint_at(completed)
                last_checkpoint = completed
            if milestone_due:
                # 里程碑评测相与 rollout/baseline 同为 eval（相位敏感层的
                # 口径一致——baseline/里程碑/重采三条评测路径同场采样）
                self.policy.eval_phase()
                if self._run_milestone(completed):
                    break  # 早停：早停签名命中，最终 policy 状态已随里程碑 checkpoint 落盘
            dist.barrier()  # 执行序第 3 步：iteration 节奏的集合点
        if last_checkpoint < completed:
            # 收尾兜底只允许前向推进：恢复点已在目标之后（收缩 max_iterations
            # 的续训 = 无操作）时不得把更后的训练态改写成更小的 iteration
            # 标签；早停时 completed < max_iterations，收尾落实际完成数
            self._checkpoint_at(completed)
        self.policy.eval_phase()  # 重采与 baseline 同为 eval 相（差异唯一归因于 RL）
        if completed > start_iteration:
            # 本次调用执行了训练才收官重采：恢复点已达标的无操作续训不重采
            # （policy 未变，重采只因 RNG 流位置不同而静默改写产物）。
            # 同 Baseline：采样全 rank 集合参与、落盘 rank 0 独写（eval 侧
            # write_enabled 闸门）
            self.evaluation.resample()
        self._write_diagnostic(pairs)
        return completed

    def _run_milestone(self, iteration: int) -> bool:
        """里程碑解码评测 → ``milestone`` 事件写入训练指标流 → 早停判定
        消费该流。返回是否早停（解码评测只发生在本路径，不进逐 iteration
        循环；本阶段的早停判定只消费本阶段的流前缀——组3 两阶段事件同流
        存放，按 stage 过滤防跨阶段串扰）。评测（policy 采样是 FSDP 集合
        前向）全 rank 对称参与；事件写出与早停 verdict 判定只在 rank 0
        （RunArtifacts 的 rank 0 写盘契约），verdict 经广播同步——早停是
        全局决定，各 rank 必须一致 break（分歧会让 iteration 节奏的
        barrier 互等死锁）。"""
        dist = self.runtime.dist
        metrics: MilestoneMetrics = self.evaluation.milestone_metrics()
        if dist.rank == 0:
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
            return dist.broadcast_flag(verdict.stop)
        return dist.broadcast_flag(False)  # 返回值被 rank 0 的广播覆盖

    def _write_diagnostic(self, pairs: list[TrainingLogProbPair]) -> None:
        """训练侧 log-prob 对落盘（--dump-trajectory）：多 rank 下归并到
        rank 0（按 rank 升序拼接，per-rank 记录全量保留）；单进程直写。
        本次运行未产出新对（零训练迭代的续训 = 恢复点已达标）不写——
        空清单会覆盖既有诊断工件，把文档化的 no-op resume 变成对一致性
        证据的破坏。"""
        if not self._dump or not pairs:
            return
        dist = self.runtime.dist
        gathered = dist.gather(pairs)
        if dist.rank != 0:
            return
        merged = [pair for source in gathered for pair in source]
        self.artifacts.paths.training_diagnostic.write_text(
            TrainingDiagnostic(logprob_pairs=merged).model_dump_json(indent=2),
            encoding="utf-8",
        )

    def _checkpoint_at(self, iteration: int) -> None:
        """指定 iteration 的 checkpoint 节奏（全 rank 的单一入口）：full
        state 导出是 FSDP 集合操作（FULL_STATE_DICT），必须全 rank 调用本
        方法，且**只导出一次**——产物 checkpoint 写盘（rank 0）与续训状态
        分片共享同一份（二次导出会让每 rank 同时驻留两份完整 CPU 权重，
        checkpoint 期 host 内存翻倍）；产物 checkpoint 只 rank 0 独写
        （契约文件名不变），续训状态每 rank 写自己的分片文件。"""
        full_state = self.policy.full_state()
        if self.runtime.dist.rank == 0:
            self._write_checkpoint(iteration, full_state)
        self.resume_store.save(self, iteration, full_state)

    def _write_checkpoint(
        self, iteration: int, full_state: dict,
    ) -> None:
        """policy（本组可训练网络）与判别器权重落盘（可装载 state_dict，
        rank 0 独写——多 rank 下仅 rank 0 产出契约工件）。
        policy 经 GroupPolicy.full_state 固化 full state（FSDP 装配下由
        PolicySharding 导出、键形与裸网络一致；导出本身是集合操作，
        ``full_state`` 必传——由调用方全 rank 完成导出后传入，本方法内部
        不按需导出（rank0-only 导出会互等死锁））；判别器经
        loadable_state_dict 固化可装载形态（spectral norm 启用时携带
        参数化状态，消费面按形态分派逐位还原）；stage 前缀隔离组3
        两阶段的同名产物（stage-1 无前缀 = 历史布局逐字一致）。续训
        全状态由 ResumeStore.save 同节奏落盘（per-rank 分片文件）。"""
        prefix = self.stage_tag.checkpoint_prefix
        torch.save(
            full_state,
            self.artifacts.paths.checkpoints / (
                f"{prefix}{POLICY_CHECKPOINT_TEMPLATE.format(iteration=iteration)}"
            ),
        )
        torch.save(
            NetworkAssembler.loadable_state_dict(self.rewards.discriminator),
            self.artifacts.paths.checkpoints
            / f"{prefix}discriminator_iter{iteration}.pt",
        )
