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
    train 启动时：per-rank buffer 的 base 分区由冻结初始 policy 生成
    （续训恢复时跳过——buffer 两区内容随状态整体回归）。

装配（含分布式包装：FSDP full-shard + 梯度检查点、判别器 DDP、pool
切片、seed 的 rank 派生、指标归并器）收敛在 TrainingRuntime——循环
代码对部署形态无分支。可训练对象按组装配（GroupPolicy）；每组一次
装配 → 判别器与 Replay buffer 随训练实例天然隔离（per-rank buffer，
跨组/跨阶段不复用）。
"""

import time
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict

from cynosure.config import CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.grpo import MgaiAdvantage, StepwisePolicyUpdate
from cynosure.netbuild import NetworkAssembler
from cynosure.train.artifacts import (
    POLICY_CHECKPOINT_TEMPLATE,
    IterEvent,
    RunArtifacts,
)
from cynosure.train.policy import GroupPolicy
from cynosure.train.resume import ResumeStore
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import IterationRollout, RolloutPhase
from cynosure.train.runtime import AmpContext, TrainingRuntime


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
        self.advantage = MgaiAdvantage(clamp=config.grpo.advantage_clamp)
        self.runtime = TrainingRuntime.build(
            config, run_artifacts, device=device, dist_context=dist_context,
        )
        self.resume_store = ResumeStore(
            run_artifacts.paths.checkpoints,
            self.stage_tag.checkpoint_prefix,
            self.runtime.dist,
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
    def generators(self) -> dict[str, torch.Generator]:
        """六条命名 RNG 流（续训状态机按名保存/恢复的注册表）。"""
        return self.runtime.generators

    @property
    def device(self) -> torch.device:
        """装配期单点选定的训练设备（续训状态恢复的迁入目标）。"""
        return self.runtime.amp.device

    @property
    def amp(self) -> AmpContext:
        """数值口径（device + autocast dtype）。"""
        return self.runtime.amp

    def seed_base_partition(self) -> None:
        """train 启动期的 buffer base 分区自动生成：用冻结初始 policy
        （未参与任何梯度步）rollout 产出填满 per-rank base 分区（spec
        补钉；per-rank buffer 覆盖各自区域，rollout 走本 rank 独立流）。"""
        self.rewards.seed_base(
            self.rollout.base_partition_samples(
                self.rewards.buffer.base_capacity,
            ),
        )

    def run(self, *, resume: bool = False) -> int:
        """训练主循环：base 分区自动生成 → 逐 iteration 执行序 → checkpoint
        与续训状态落盘。``resume=True`` 时从 run 目录各 rank 的最新续训
        状态恢复（全清单覆写，resume 模块），rank 0 回退指标流中恢复点
        之后的半截事件后从恢复点继续——base 分区种子生成随之跳过（buffer
        随状态整体回归）。返回完成的 iteration 数（config 口径的累计完
        成数）。"""
        dist = self.runtime.dist
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
            self.seed_base_partition()
        pairs: list[TrainingLogProbPair] = []
        last_checkpoint = start_iteration
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
            if (
                (iteration + 1) % self.config.schedule.checkpoint_interval == 0
                or (iteration + 1) % self.config.schedule.milestone_interval == 0
            ):
                # checkpoint 周期之外，每个里程碑也强制落盘（config 契约：
                # milestone 评测器与恢复路径的取数点，周期不覆盖时仍须产出）
                self._checkpoint_at(iteration + 1)
                last_checkpoint = iteration + 1
            dist.barrier()  # 执行序第 3 步：iteration 节奏的集合点
        if last_checkpoint < self.config.schedule.max_iterations:
            # 收尾兜底只允许前向推进：恢复点已在目标之后（收缩 max_iterations
            # 的续训 = 无操作）时不得把更后的训练态改写成更小的 iteration 标签
            self._checkpoint_at(self.config.schedule.max_iterations)
        self._write_diagnostic(pairs)
        return self.config.schedule.max_iterations

    def _update_policy(self, record: IterationRollout) -> dict[str, float]:
        """逐 k 独立梯度步（执行序第 2 相）：每 k 一次
        forward→backward→optimizer.step（MGAI advantage 融合后交 updater），
        返回逐 k loss 组件。"""
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
            self.amp.device_type, dtype=self.amp.dtype,
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

    def _write_diagnostic(self, pairs: list[TrainingLogProbPair]) -> None:
        """训练侧 log-prob 对落盘（--dump-trajectory）：多 rank 下归并到
        rank 0（按 rank 升序拼接，per-rank 记录全量保留）；单进程直写。"""
        if not self._dump:
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
        方法；产物 checkpoint 只 rank 0 独写（契约文件名不变），续训状态
        每 rank 写自己的分片文件（per-rank RNG/buffer/optimizer）。"""
        full_state = self.policy.full_state()
        if self.runtime.dist.rank == 0:
            self._write_checkpoint(iteration, full_state)
        self.resume_store.save(self, iteration)

    def _write_checkpoint(
        self, iteration: int, full_state: dict,
    ) -> None:
        """policy（本组可训练网络）与判别器权重落盘（可装载 state_dict，
        rank 0 独写——多 rank 下仅 rank 0 产出契约工件）。
        policy 经 GroupPolicy.full_state 固化 full state（FSDP 装配下由
        PolicySharding 导出、键形与裸网络一致；导出本身是集合操作，
        ``full_state`` 必传——由调用方全 rank 完成导出后传入，本方法内部
        不按需导出（rank0-only 导出会互等死锁））；判别器经
        loadable_state_dict 固化有效权重（spectral norm 启用时仍可严格
        重载）；stage 前缀隔离组3 两阶段的同名产物（stage-1 无前缀 =
        历史布局逐字一致）。续训全状态由 ResumeStore.save 同节奏
        落盘（per-rank 分片文件）。"""
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
