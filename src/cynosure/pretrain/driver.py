"""判别器 warm-start 预训练 driver（ADR-0007：预训练 + 继续在线更新）。

密集步进循环：每步以 base policy 冻结 rollout 量产一批 fake（复用回放
缓冲 base 分区采样入口——批量分块、独立随机流、输出归一到 pool 存储
域）→ 以更新前快照测 held-out AUC（与在线期 iter 事件同口径：更新后测
同一 fake 批会把 in-sample 拟合计入 AUC）→ 达 RM readiness gate 即终止，
否则以在线期同款 ``OnlineUpdate.step`` 原语更新一步（预训练期无「当前
policy」，混采语义退化为 base fake 库内采样；real 侧口径与在线期一致）
→ 预训练事件落盘。步数上限耗尽仍未达标时补测一次落盘权重的 AUC
（报告 ``final_heldout_auc`` 与 checkpoint 同快照）。

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
from cynosure.train.artifacts import PretrainEvent
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
        # 装配守卫（train 循环同款先例：无效组合在装配期显式拒绝，而非
        # 让昂贵 rollout 先行、更新时才缺样本）：每步量产的 fake 须覆盖
        # 判别器更新批的当前半区；回放半区须非空且 base 分区足以承担
        # 首次更新的回放需求（首次更新时近期分区为空）
        current_count = math.ceil(
            reward.disc_batch_size_k * reward.replay_current_fraction,
        )
        if reward.pretrain_fake_batch < current_count:
            raise ValueError(
                f"预训练 fake 批量 {reward.pretrain_fake_batch} 不足判别器"
                f"更新批的当前半区 {current_count} 条"
                f"（disc_batch_size_k={reward.disc_batch_size_k}）"
            )
        replay_count = reward.disc_batch_size_k - current_count
        if replay_count < 1:
            raise ValueError(
                f"回放供给不足：disc_batch_size_k={reward.disc_batch_size_k}"
                " 的回放半区为 0 条（K 须 ≥2）"
            )
        if reward.replay_buffer_capacity // 2 < replay_count:
            raise ValueError(
                "回放供给不足：replay_buffer_capacity="
                f"{reward.replay_buffer_capacity} 的 base 分区仅 "
                f"{reward.replay_buffer_capacity // 2} 条，不足以承担"
                f"首次判别器更新的回放半区 {replay_count} 条"
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
            TrainingRuntime.assemble_sampler(config, self._policy.field),
            self._rewards.update.scorer,
            generators["rollout"],
            condition_sampler=self._policy.conditions,
            device_type=amp.device_type,
            autocast_dtype=amp.dtype,
            device=amp.device,
            # base 分区与每步 fake 量产的独立派生流（seed+5，与 train 同源）：
            # 抽取数随缓冲容量/批量配置变化，不漂移其余抽样流
            base_generator=generators["base_partition"],
        )

    @property
    def policy(self) -> GroupPolicy:
        """本组 policy 侧装配（冻结 base；组1/组2 的采样场与条件分布）。"""
        return self._policy

    @property
    def rewards(self) -> RewardCoordinator:
        """判别器侧协作者组（Online update 原语 / held-out AUC / buffer）。"""
        return self._rewards

    def run(self) -> PretrainReport:
        """密集步进至 RM readiness gate 达标或步数上限，产出判别器
        checkpoint 与预训练报告（产物全局唯一：单进程唯一写者）。"""
        reward = self._config.reward
        batch = reward.pretrain_fake_batch
        gate = reward.pretrain_gate_auc
        self._policy.eval_phase()  # 冻结 base 的 rollout（执行序第 1 相口径）
        self._rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        # buffer base 分区由冻结初始 policy 产出填充（与在线期同源：base
        # 分区采样入口；预训练期无「当前 policy」，混采语义退化为 base
        # fake 库内采样）
        self._rewards.seed_base(
            self._rollout.base_partition_samples(
                self._rewards.buffer.base_capacity,
            ),
        )
        steps_completed = 0
        gate_passed = False
        final_auc = 0.0
        for step in range(reward.pretrain_max_steps):
            started = time.monotonic()
            fakes = self._rollout.base_partition_samples(batch)
            auc = self._rewards.auc.compute(fakes)  # 更新前快照（在线期口径）
            if auc >= gate:
                # 达标即停：判别器不再更新，checkpoint 与判定同快照
                final_auc = auc
                gate_passed = True
                break
            update = self._rewards.update_step(fakes)
            zones = self._rewards.buffer.zone_sizes()
            self._run.append_event(PretrainEvent(
                step=step,
                loss_discriminator=update.loss_discriminator,
                heldout_auc=auc,
                buffer_base_occupied=zones.base,
                buffer_recent_occupied=zones.recent,
                lr=reward.disc_lr,
                elapsed_s=time.monotonic() - started,
            ))
            steps_completed = step + 1
        if not gate_passed:
            # 步数上限耗尽：补测落盘权重的 held-out AUC（循环内最后一次
            # 测得值属于更新前的上一份权重，与 checkpoint 不同快照）
            final_auc = self._rewards.auc.compute(
                self._rollout.base_partition_samples(batch),
            )
        return self._finalize(steps_completed, final_auc, gate_passed)

    def _finalize(
        self, steps_completed: int, final_auc: float, gate_passed: bool,
    ) -> PretrainReport:
        """产物落盘：判别器 checkpoint（可装载 state_dict，与训练期产物
        checkpoint 同构）+ 预训练报告（kind 标识 + 最终 held-out AUC +
        数据口径指纹）。"""
        torch.save(
            NetworkAssembler.loadable_state_dict(self._rewards.discriminator),
            self._run.paths.discriminator_ckpt,
        )
        discriminator_config = self._config.artifacts.discriminator_config_json
        if discriminator_config is None:
            raise ValueError(
                "判别器网络配置缺失（artifacts.discriminator_config_json）"
            )
        reward = self._config.reward
        report = PretrainReport(
            group=self._config.experiment.group,
            latent_shape=self._config.latent_shape,
            final_heldout_auc=final_auc,
            steps_completed=steps_completed,
            gate_auc=reward.pretrain_gate_auc,
            gate_passed=gate_passed,
            discriminator_ckpt=self._run.paths.discriminator_ckpt.relative_to(
                self._run.paths.root,
            ).as_posix(),
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
            ),
        )
        self._run.paths.report.write_text(
            report.model_dump_json(indent=2), encoding="utf-8",
        )
        return report
