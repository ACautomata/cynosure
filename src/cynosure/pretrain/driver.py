"""判别器 warm-start 预训练 driver（ADR-0007：预训练 + 继续在线更新；
ADR-0008 决策 3：per-condition 步进与终止）。

密集步进循环：每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态
均匀轮转，确定性不耗 RNG）→ 以 base policy 冻结 rollout 量产**该条件**
fake 批（复用回放缓冲 base 分区采样入口——批量分块、独立随机流、输出
归一到 pool 存储域）→ 以更新前快照测该条件 held-out AUC（held-out 侧
按同条件过滤、全量卷池化点估计；更新后测同一 fake 批会把 in-sample
拟合计入 AUC）→ 支撑度规则判定过线（``SupportRule.passes``：该条件
held-out 卷数 < 界用 bootstrap CI 下界、≥ 界用点估计——ADR-0008 决策
6 / #85）→ 首测过线换新批复测确认：两次独立测量都过线该条件入白名单
（单批贴线越过被非确定性拒绝），报告值取两次较小者，确认步不更新
（无更新即无事件）→ 未确认则以在线期同款 ``OnlineUpdate.step`` 原语
更新一步（预训练期无「当前 policy」，混采语义退化为 base fake 库内
采样；real 侧同条件匹配）。每个更新步同时消费与在线**同一**过拟合
分叉监控组件、同一 config knobs（ADR-0009-γ：共享装配缝挂进
``RewardCoordinator`` 的 ``OverfitMonitor``——train 侧干净域复算准确率
与本步更新前 held-out AUC 合成分叉观测，per-condition EMA 自下而上
越线落预训练相 ``overfit_alert`` 事件（``phase="pretrain"``，EXEMPT
记账——预训练执行史全量保留）；只报警不动作，确认步不更新不观测）——
per-condition 分叉监控在 RM readiness gate 之前的预训练相即暴露稀疏
模态（MRA）记忆化。已入白名单的条件不再复测（棘轮：复测确认
已拦住单批噪声，后续掉线由在线期白名单动态恢复机制兜底）。终止 =
全部条件最近一次确认过线即停；``pretrain_max_steps`` 耗尽 → 白名单 =
已确认者，未确认条件逐个对落盘权重补测（报告值与 checkpoint 同快照）。
白名单为空不拒跑——报告与 checkpoint 照常落盘供诊断（拒跑由 train
gate 把守，不丢诊断产物）。

单进程执行（World-1 退化路径）：``DistributedContext.bootstrap()`` 在
无 torchrun 环境下不初始化进程组、集合通信恒等，产物全局唯一——多 rank
各自预训练会分叉判别器（CLI 层另有 RANK env 显式拒绝守卫）。判别器侧
装配经 ``TrainingRuntime.assemble_rewards``、采样封装经
``TrainingRuntime.assemble_sampler``、policy 侧经 ``GroupPolicy.build``
（组1/组2 的采样场与条件分布按 config 分派）——与在线期同一份装配与
同一条执行路径，仅 config 不同。
"""

import math
import time

import torch

from cynosure.config import CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.pretrain.artifacts import (
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.reward.buffer import assert_replay_supply, base_condition_quota
from cynosure.reward.support import SupportRule
from cynosure.train.artifacts import OverfitAlertEvent, PretrainEvent
from cynosure.train.policy import GroupPolicy
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import RolloutPhase
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime


class PretrainDriver:
    """判别器 warm-start 预训练编排：装配（单进程 world-1）→ 密集步进 →
    产物落盘（判别器 checkpoint + 预训练报告）。"""

    def __init__(
        self,
        config: CynosureConfig,
        run: PretrainRun,
        device: torch.device | None = None,
    ) -> None:
        self._config = config
        self._run = run
        reward = config.reward
        # 装配守卫（train 装配同口径，ADR-0008 决策 4：assert_replay_supply
        # 管回放半区非零 + base 分区每条件配额 ≥ 回放半区需求；real 侧的
        # 逐 (全池, 模态) 容量 ≥ K 守卫在共享装配缝 assemble_rewards 内，
        # ADR-0008-03）+ 预训练特有守卫：每步量产的 fake 须覆盖判别器
        # 更新批的当前半区——无效组合在装配期显式拒绝，而非让昂贵 rollout
        # 先行、更新时才缺样本
        assert_replay_supply(
            reward, TrainingRuntime.assemble_vocabulary(config).names(),
        )
        current_count = math.ceil(
            reward.disc_batch_size_k * reward.replay_current_fraction,
        )
        if reward.pretrain_fake_batch < current_count:
            raise ValueError(
                f"预训练 fake 批量 {reward.pretrain_fake_batch} 不足判别器"
                f"更新批的当前半区 {current_count} 条"
                f"（disc_batch_size_k={reward.disc_batch_size_k}）"
            )
        # 单进程执行：无 torchrun 环境下 bootstrap 为 world-1 恒等
        # （不初始化进程组），集合通信原语退化——与 train 同一条装配序
        dist = DistributedContext.bootstrap()
        amp = AmpContext(
            device=device if device is not None else dist.local_device(),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        generators = TrainingRngStreams(
            dist.derive_seed(config.schedule.seed),
        ).named()
        self._policy = GroupPolicy.build(
            config, generators["rollout"], amp.device,
        )
        self._rewards = TrainingRuntime.assemble_rewards(
            config, amp, generators, dist,
        )
        self._rollout = RolloutPhase(
            config,
            TrainingRuntime.assemble_sampler(
                config, self._policy.field, device=amp.device,
            ),
            self._rewards.update.scorer,
            generators["rollout"],
            condition_sampler=self._policy.conditions,
            vocabulary=TrainingRuntime.assemble_vocabulary(config),
            device_type=amp.device_type,
            autocast_dtype=amp.dtype,
            device=amp.device,
            # base 分区与每步 fake 量产的独立派生流（seed+5，与 train 同源）：
            # 抽取数随缓冲容量/批量配置变化，不漂移其余抽样流
            base_generator=generators["base_partition"],
        )
        # 过线判定原语（ADR-0008-04 消费 ADR-0008-02/#85 的支撑度规则）：
        # bootstrap 的随机性独立派生（seed+7——六流注册表之外，预训练不
        # 参与续训、判定可复现性由 seed 纯函数保证；进注册表反而令续训
        # 状态清单失配）
        self._support = SupportRule(
            threshold=reward.pretrain_gate_auc,
            support_bound=reward.gate_support_min_volumes,
            generator=torch.Generator().manual_seed(
                dist.derive_seed(config.schedule.seed + 7),
            ),
        )
        # 轮转条件集守卫：每条件 held-out 非空（per-condition AUC 归因的
        # 前提——缺条目的条件在装配期显式拒绝，而非首步测量时才炸）
        starved = [
            target for target in self._policy.conditions.targets()
            if self._rewards.auc.condition_volume_count(target) < 1
        ]
        if starved:
            raise ValueError(
                f"held-out real 缺条件 {starved} 的条目（per-condition 步进"
                "要求轮转条件集每条件 held-out 非空——AUC 测量按条件归因"
                f"无米下锅；heldout_real_manifest={reward.heldout_real_manifest}）"
            )

    @property
    def policy(self) -> GroupPolicy:
        """本组 policy 侧装配（冻结 base；组1/组2 的采样场与条件分布）。"""
        return self._policy

    @property
    def rewards(self) -> RewardCoordinator:
        """判别器侧协作者组（Online update 原语 / held-out AUC / buffer）。"""
        return self._rewards

    @property
    def rollout(self) -> RolloutPhase:
        """rollout 封装（base fake 量产的公开面——报告值与同 seed 重演
        测量的可复现性验证消费它，ADR-0008-04）。"""
        return self._rollout

    def run(self) -> PretrainReport:
        """密集步进至全部轮转条件过线（ADR-0008 决策 3 的 per-condition
        终止语义）或步数上限，产出判别器 checkpoint 与预训练报告（产物
        全局唯一：单进程唯一写者）。

        每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态均匀轮转），
        量产该条件 fake 批、real 同条件匹配、AUC 归因该条件；首测过线
        （``SupportRule.passes``）换新批复测确认——两次独立测量都过线才
        入白名单（producer 侧成功判据对单批测量噪声鲁棒，train 侧按独立
        采样的重算不再与非确定性拒绝耦合），报告值取两次较小者。已入
        白名单的条件不再复测（棘轮：复测确认已拦住单批噪声，在线期的
        掉线由白名单动态恢复机制兜底）；``pretrain_max_steps`` 耗尽 →
        白名单 = 已确认者、未确认条件对落盘权重补测。白名单为空仍落盘
        全部产物（拒跑由 train gate 把守）。"""
        reward = self._config.reward
        targets = self._policy.conditions.targets()
        self._policy.eval_phase()  # 冻结 base 的 rollout（执行序第 1 相口径）
        self._rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        # buffer base 分区由冻结初始 policy 产出按每条件配额填充（与在线期
        # 同源：base 分区采样入口；条目带目标模态标签——ADR-0008-01）
        quota = base_condition_quota(
            reward.replay_buffer_capacity,
            self._policy.conditions.targets(),
        )
        base_fakes, base_modalities = self._rollout.base_partition_samples(quota)
        self._rewards.seed_base(base_fakes, base_modalities)
        confirmed: dict[str, float] = {}
        steps_completed = 0
        gate_passed = False
        for step in range(reward.pretrain_max_steps):
            started = time.monotonic()
            modality = targets[step % len(targets)]
            fakes = self._measurement_batch(modality)
            clusters = self._rewards.auc.compute_volume_clusters(fakes, modality)
            auc = clusters.pooled_auc()  # 更新前快照（在线期口径）
            if modality not in confirmed and self._support.passes(auc, clusters):
                confirm = self._rewards.auc.compute_volume_clusters(
                    self._measurement_batch(modality), modality,
                )
                confirm_auc = confirm.pooled_auc()
                if self._support.passes(confirm_auc, confirm):
                    confirmed[modality] = min(auc, confirm_auc)  # 保守口径：两次取小
                    if len(confirmed) == len(targets):
                        gate_passed = True  # 全部条件过线：终止
                        break
                    continue  # 本条件已确认：本步不更新（无更新即无事件）
            update = self._rewards.update_step(fakes, modality)
            # 过拟合分叉观测（ADR-0009-γ）：与在线同一监控组件、同一
            # knobs（共享装配缝挂进 RewardCoordinator 的 OverfitMonitor，
            # 阈值/跨度同源于 config.reward.overfit_*）——train 侧干净域
            # 复算准确率（随更新报告上行）与本步更新前 held-out AUC 合成
            # 分叉观测，per-condition EMA 越线即落预训练相告警（确认步
            # 不更新不观测；报警不动作，人工裁决——口径同在线）
            reading = self._rewards.overfit.observe(
                modality,
                train_pairwise_acc=update.train_pairwise_acc,
                heldout_auc=auc,
            )
            zones = self._rewards.buffer.zone_sizes()
            self._run.append_event(PretrainEvent(
                step=step,
                modality=modality,
                loss_discriminator=update.loss_discriminator,
                heldout_auc=auc,
                buffer_base_occupied=zones.base,
                buffer_recent_occupied=zones.recent,
                lr=reward.disc_lr,
                elapsed_s=time.monotonic() - started,
            ))
            if reading.alerted:
                # 预训练相告警排本步 pretrain 事件之后（与在线侧「iter
                # 后随告警」同构的写出序）；``phase="pretrain"`` 是回退
                # 记账的 EXEMPT 分轨轴——预训练执行史全量保留（预训练相
                # 的 ``iteration`` 记本步步号）
                self._run.append_event(OverfitAlertEvent(
                    iteration=step,
                    phase="pretrain",
                    modality=modality,
                    divergence_ema=reading.divergence,
                    train_pairwise_acc=update.train_pairwise_acc,
                    heldout_auc=auc,
                ))
            steps_completed += 1  # 更新步计数（确认步占步号但不更新不事件）
        reported = dict(confirmed)
        if not gate_passed:
            # 步数上限耗尽：未确认条件逐个对落盘权重补测（循环内最后一次
            # 测得值属于更新前的上一份权重，与 checkpoint 不同快照；
            # 已确认条件的报告值 = 确认时的两次较小者，保留不覆盖）
            for target in targets:
                if target not in reported:
                    reported[target] = self._rewards.auc.compute_volume_clusters(
                        self._measurement_batch(target), target,
                    ).pooled_auc()
        return self._finalize(
            steps_completed, reported, list(confirmed), gate_passed,
        )

    def _measurement_batch(self, modality: str) -> torch.Tensor:
        """单条件量产一批 fake（gate 测量/复测/补测共用入口）：update_step
        的回放按本步条件过滤，测量批与更新批同条件——混采量产批没有诚实
        标签可穿（ADR-0008-03 的最小诚实形态）。同条件批量量产（逐条清单
        语义，#129——同条件同形状，单条件内 stack 成批）。"""
        latents, _ = self._rollout.base_partition_samples(
            {modality: self._config.reward.pretrain_fake_batch},
        )
        return torch.stack(latents)

    def _finalize(
        self,
        steps_completed: int,
        condition_auc: dict[str, float],
        whitelist: list[str],
        gate_passed: bool,
    ) -> PretrainReport:
        """产物落盘：判别器 checkpoint（可装载 state_dict，与训练期产物
        checkpoint 同构）+ 预训练报告（kind 标识 + per-condition held-out
        AUC + 条件白名单 + 数据口径指纹，含 checkpoint 内容指纹——报告的
        白名单与实测值只对落盘这份权重负责，装载面按指纹对照，
        ``load_discriminator``）。白名单为空同样落盘——报告与 checkpoint
        是失败预训练的诊断产物，不丢。"""
        torch.save(
            NetworkAssembler.loadable_state_dict(self._rewards.discriminator),
            self._run.paths.discriminator_ckpt,
        )
        discriminator_relative = (
            self._run.paths.discriminator_ckpt.relative_to(
                self._run.paths.root,
            ).as_posix()
        )
        discriminator_config = self._config.artifacts.discriminator_config_json
        if discriminator_config is None:
            raise ValueError(
                "判别器网络配置缺失（artifacts.discriminator_config_json）"
            )
        reward = self._config.reward
        # 词表工件绑定 = 多条件线的判据（schema 面 MR-RATE 必填 / BraTS
        # 携带即拒），不复制 dataset 字符串
        vocabulary_path = self._config.artifacts.condition_vocabulary_json
        report = PretrainReport(
            group=self._config.experiment.group,
            # 形状口径两态（#129）：多条件线的形状逐条件派生自词表工件、
            # 报告不落派生副本（口径由 provenance 指纹承载）；单域线记
            # 全局形状（单条件词汇特例）
            latent_shape=(
                None if vocabulary_path is not None
                else self._config.latent_shape
            ),
            condition_auc=condition_auc,
            gate_whitelist=whitelist,
            steps_completed=steps_completed,
            gate_auc=reward.pretrain_gate_auc,
            gate_passed=gate_passed,
            discriminator_ckpt=discriminator_relative,
            provenance=PretrainProvenance(
                real_pool_manifest=str(reward.real_pool_manifest),
                real_pool_manifest_sha256=PretrainProvenance.digest(
                    reward.real_pool_manifest,
                ),
                heldout_manifest=str(reward.heldout_real_manifest),
                heldout_manifest_sha256=PretrainProvenance.digest(
                    reward.heldout_real_manifest,
                ),
                channel_stats=str(reward.channel_stats_json),
                channel_stats_sha256=PretrainProvenance.digest(
                    reward.channel_stats_json,
                ),
                discriminator_config=str(discriminator_config),
                discriminator_config_sha256=PretrainProvenance.digest(
                    discriminator_config,
                ),
                discriminator_ckpt=discriminator_relative,
                discriminator_ckpt_sha256=PretrainProvenance.digest(
                    self._run.paths.discriminator_ckpt,
                ),
                # 词表工件口径指纹（多条件线；单域线无工件，两侧同为 None）
                condition_vocabulary=(
                    None if vocabulary_path is None else str(vocabulary_path)
                ),
                condition_vocabulary_sha256=(
                    None if vocabulary_path is None
                    else PretrainProvenance.digest(vocabulary_path)
                ),
            ),
        )
        self._run.paths.report.write_text(
            report.model_dump_json(indent=2), encoding="utf-8",
        )
        return report
