"""判别器 warm-start 预训练 driver（ADR-0007：预训练 + 继续在线更新；
ADR-0008 决策 3：per-condition 步进与终止）。

密集步进循环：每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态
均匀轮转，确定性不耗 RNG）→ 该条件的 gate 测量批 = 装配原语对
**全量 held-out 卷**的冻结基座同源重构（``ReconstructionAssembler.
measure_condition``：定序轮转 σ + 复位测量流 ⇒ 同输入同输出、可复算）
→ 以更新前快照测该条件 recon-AUC（real = held-out real 原始、fake = 其
重构体，逐样本配对；更新后测同一测量批会把 in-sample 拟合计入 AUC）
→ 支撑度规则判定过线（``SupportRule.passes``：该条件 held-out 卷数
< 界用 bootstrap CI 下界、≥ 界用点估计——ADR-0008 决策 6 / #85）→
首测过线换新批复测确认：两次独立测量都过线该条件入白名单（单批贴线
越过被非确定性拒绝），报告值取两次较小者，确认步不更新（无更新即无
事件）→ 未确认则以**同一装配原语**产出的配对批（real + 冻结基座同源
重构 fake，ADR-0012）走在线期同款 ``OnlineUpdate.step`` 更新一步
（两阶段构造同构、warm-start 权重不面临分布跳变；测量批与更新批同一
原语的两条入口：前者定序、后者抽样，见 ``reward.assembly`` 模块
docstring）。每个更新步同时消费与在线**同一**过拟合
分叉监控组件、同一 config knobs（ADR-0009-γ：共享装配缝挂进
``RewardCoordinator`` 的 ``OverfitMonitor``——train 侧干净域复算准确率
与本步更新前 recon-AUC 合成分叉观测，per-condition EMA 自下而上
越线落预训练相 ``overfit_alert`` 事件（``phase="pretrain"``，EXEMPT
记账——预训练执行史全量保留）；只报警不动作，确认步不更新不观测）——
per-condition 分叉监控在 RM readiness gate 之前的预训练相即暴露稀疏
模态（MRA）记忆化。已入白名单的条件不再复测（棘轮：复测确认
已拦住单批噪声，后续掉线由在线期白名单动态恢复机制兜底）。终止 =
全部条件最近一次确认过线即停；``pretrain_max_steps`` 耗尽 → 白名单 =
已确认者，未确认条件逐个对落盘权重补测（报告值与 checkpoint 同快照）。
白名单为空不拒跑——报告与 checkpoint 照常落盘供诊断（拒跑由 train
gate 把守，不丢诊断产物）。

**预训练相不产 rollout**（ADR-0012 决策 6）：量产 rollout（num_steps 步
全 ODE）整体退出本执行路径——fake 侧只剩「全量 held-out 卷的重构」
（测量批，σ 定序轮转）与「更新批的重构」（σ 逐样本抽自被优化步）两条，
均经装配原语；事件流的 ``reconstruction_forwards`` / ``measurement_volumes``
是本口径的读数面。判据口径的两阶段差异记录在案（ADR-0012 决策 5）：
预训练判据是 recon-AUC（判别器训练任务的 out-of-sample 泛化力）、在线
运行口径是 rollout-AUC（对打分对象的分辨力），**不可跨阶段比较绝对值**
——准入体检 vs 在岗考核。

单进程执行（World-1 退化路径）：``DistributedContext.bootstrap()`` 在
无 torchrun 环境下不初始化进程组、集合通信恒等，产物全局唯一——多 rank
各自预训练会分叉判别器（CLI 层另有 RANK env 显式拒绝守卫）。判别器侧
装配经 ``TrainingRuntime.assemble_rewards``（配对批装配原语同缝组装）、
采样封装经 ``TrainingRuntime.assemble_sampler``、policy 侧经
``GroupPolicy.build``（组1/组2 的采样场与条件分布按 config 分派）——
与在线期同一份装配与同一条执行路径，仅 config 不同。base 分区量产与
``RolloutPhase`` 装配同随量产退役（回放缓冲无消费者、预训练不需
rollout 相）；``reward.buffer`` 的组件与事件占位字段按契约保留，物理
删除归退役票（#173）。
"""

import time
from typing import TYPE_CHECKING

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
from cynosure.reward.assembly import PairBatch
from cynosure.reward.support import SupportRule
from cynosure.train.artifacts import OverfitAlertEvent, PretrainEvent
from cynosure.train.policy import GroupPolicy
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime

if TYPE_CHECKING:
    # 卷级聚类观测面仅作返回类型标注（运行时由 auc 组件产出）
    from cynosure.reward.auc import VolumeScoreClusters


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
        # 回放供给装配守卫（ADR-0008 决策 4）的调用点已随 ADR-0012 退役：
        # gate 测量批的量产口径（pretrain_fake_batch）与更新批（配对批 = K，
        # 装配原语）解耦后，回放半区不再有消费者——守卫留着会误拒合法
        # 配对批配置（如 disc_batch_size_k=1）。真实约束是 real 侧容量守卫
        # （逐 (全池, 模态) 容量 ≥ K，位于共享装配缝 assemble_rewards 内，
        # ADR-0008-03）。守卫函数与 ReplayBuffer 组件的物理删除归退役票
        # （#173）。
        # 单进程执行：无 torchrun 环境下 bootstrap 为 world-1 恒等
        # （不初始化进程组），集合通信原语退化——与 train 同一条装配序
        dist = DistributedContext.bootstrap()
        amp = AmpContext(
            device=device if device is not None else dist.local_device(),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        generators = TrainingRngStreams(
            dist.derive_seed(config.schedule.seed),
            # 与 train runtime 同一 seeding 规则（数据侧逐 rank 派生、
            # recon 用 shared）——单进程下 derive_seed 恒等，两参同值
            shared_seed=config.schedule.seed,
        ).named()
        self._policy = GroupPolicy.build(
            config, generators["rollout"], amp.device,
        )
        # 采样封装先行装配（判别器侧配对批装配原语与其共享同一实例——
        # 确定性 ODE 续跑 kernel、日程表与分块调度单点）
        sampler = TrainingRuntime.assemble_sampler(
            config, self._policy.field, device=amp.device,
        )
        self._rewards = TrainingRuntime.assemble_rewards(
            config, amp, generators, dist,
            sampler=sampler, conditions=self._policy.conditions,
        )
        # 量产 rollout（``RolloutPhase``）不装配：ADR-0012 决策 6 后预训练
        # 相 fake 全由装配原语重构产出（测量批 / 更新批两条入口），无
        # rollout 相的消费者——装配它只会让「fake 是否走了量产」留一条
        # 静默可用的旧路（``base_partition`` 流随之下岗，注册表结构不动）。
        #
        # 过线判定原语（ADR-0008-04 消费 ADR-0008-02/#85 的支撑度规则）：
        # bootstrap 的随机性独立派生（seed+7——命名流注册表之外，预训练
        # 不参与续训、判定可复现性由 seed 纯函数保证；进注册表反而令续训
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

    def run(self) -> PretrainReport:
        """密集步进至全部轮转条件过线（ADR-0008 决策 3 的 per-condition
        终止语义）或步数上限，产出判别器 checkpoint 与预训练报告（产物
        全局唯一：单进程唯一写者）。

        每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态均匀轮转），
        以该条件**全量 held-out 卷**的冻结基座同源重构作测量批（recon-AUC
        的 fake 侧）、real 同条件同批配对、AUC 归因该条件；首测过线
        （``SupportRule.passes``）换新批复测确认——两次独立测量都过线才
        入白名单（producer 侧成功判据对单批测量噪声鲁棒，train 侧按独立
        采样的重算不再与非确定性拒绝耦合），报告值取两次较小者。已入
        白名单的条件不再复测（棘轮：复测确认已拦住单批噪声，在线期的
        掉线由白名单动态恢复机制兜底）；``pretrain_max_steps`` 耗尽 →
        白名单 = 已确认者、未确认条件对落盘权重补测。白名单为空仍落盘
        全部产物（拒跑由 train gate 把守）。"""
        reward = self._config.reward
        targets = self._policy.conditions.targets()
        self._policy.eval_phase()  # 冻结 base 的推理相（重构是 policy 前向）
        self._rewards.discriminator.eval()  # 打分/监控前向恒 eval（见 RewardCoordinator）
        confirmed: dict[str, float] = {}
        volumes: dict[str, int] = {}
        steps_completed = 0
        gate_passed = False
        for step in range(reward.pretrain_max_steps):
            started = time.monotonic()
            modality = targets[step % len(targets)]
            batch, clusters, forwards = self._measurement(modality)
            volumes[modality] = clusters.volume_count  # 支撑度判定的卷数留痕
            auc = clusters.pooled_auc()  # 更新前快照（本步判别器权重）
            if modality not in confirmed and self._support.passes(auc, clusters):
                # 复测（同条件独立测量）：判别器权重同刻，变化的是随机
                # 面——held-out 卷的抽取（heldout_auc 流）与重构 ε 的
                # 抽取（heldout_auc 流推进后、复位测量流从同一起手点再
                # 走一遍）都拿到新一批随机数，两次读数不是同一个样本
                _confirm_batch, confirm, _confirm_forwards = (
                    self._measurement(modality)
                )
                confirm_auc = confirm.pooled_auc()
                if self._support.passes(confirm_auc, confirm):
                    confirmed[modality] = min(auc, confirm_auc)  # 保守口径：两次取小
                    if len(confirmed) == len(targets):
                        gate_passed = True  # 全部条件过线：终止
                        break
                    continue  # 本条件已确认：本步不更新（无更新即无事件）
            update = self._rewards.update_step(
                # 更新批 = 装配原语的配对批（ADR-0012）：fake = 冻结基座
                # 对同批 real 的同源重构（专属 recon 流、先抽 s 后抽 ε、
                # η=0 确定性 ODE 续跑）——与在线更新同一原语供批、判别器
                # 任务两阶段同构（warm-start 权重不面临分布跳变）。与本步
                # 测量批同一原语的另一条入口（定序轮转 σ + 复位测量流）。
                self._rewards.assembler.assemble(modality),
            )
            # 过拟合分叉观测（ADR-0009-γ）：与在线同一监控组件、同一
            # knobs（共享装配缝挂进 RewardCoordinator 的 OverfitMonitor，
            # 阈值/跨度同源于 config.reward.overfit_*）——train 侧干净域
            # 复算准确率（随更新报告上行）与本步更新前 recon-AUC 合成
            # 分叉观测，per-condition EMA 越线即落预训练相告警（确认步
            # 不更新不观测；报警不动作，人工裁决——口径同在线。两侧估计
            # 量同为 Mann-Whitney pairwise 占比、同为干净域，只差 in/out
            # of sample 平面与 fake 来源）
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
                # 重构成本读数（#171 AC5 的成本口径落点）：测量批重构的
                # 前向次数（逐卷定序 σ 的续跑步数之和）与测量批规模——
                # 30 步全 ODE 量产路径已不在本执行路径，这两项让「没有
                # 量产」在事件流上可核对（口径见模块 docstring）
                reconstruction_forwards=forwards,
                measurement_volumes=batch.fakes.shape[0],
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
                    _, clusters, _ = self._measurement(target)
                    reported[target] = clusters.pooled_auc()
                    volumes[target] = clusters.volume_count
        return self._finalize(
            steps_completed, reported, list(confirmed), gate_passed, volumes,
        )

    def _measurement(
        self, modality: str,
    ) -> tuple[PairBatch, "VolumeScoreClusters", int]:
        """单条件测量批 → 卷级分数聚类（gate 测量/复测/补测共用入口）。

        测量批 = 该条件**全量 held-out 卷**的冻结基座同源重构（装配原语
        ``measure_condition``：定序轮转 σ + 复位测量流 ⇒ 逐次测量逐位同
        输出）；real 侧**就是这批重构的源**（同一次
        ``condition_latents`` 抽取的逐样本配对）——recon-AUC 的判别目标
        因此只剩重构伪影（ADR-0012 决策 5）。与更新批同条件归因
        （ADR-0008-03 条件归因口径），条件轴与本步更新的条件轴一致。

        随机流：held-out 全量卷的抽取消耗 ``heldout_auc`` 命名流（卷内
        顺序不影响读数——AUC 是集合级秩统计），重构的 ε 走装配原语的
        复位测量流（不碰 recon 流）。预训练单进程、不参与续训，两处
        消耗都由 seed 纯函数确定 ⇒ 同 seed 重跑逐位可复算。
        """
        assembler = self._rewards.assembler
        if assembler is None:
            raise ValueError(
                "配对批装配原语未装配（RewardCoordinator.assembler=None）："
                "预训练测量批与更新批同源于它（ADR-0012）"
            )
        # real 侧先抽一次（该条件全量 held-out 卷）：同一批张量既作
        # AUC 的 real 侧、又作重构的源——逐样本配对由构造保证
        reals = self._rewards.auc.condition_latents(modality)
        batch = assembler.measure_condition(reals, modality)
        clusters = self._rewards.auc.compute_volume_clusters(
            batch.reals, batch.fakes, modality,
        )
        return batch, clusters, assembler.measurement_forward_count(
            reals, modality,
        )

    def _finalize(
        self,
        steps_completed: int,
        condition_auc: dict[str, float],
        whitelist: list[str],
        gate_passed: bool,
        condition_volumes: dict[str, int],
    ) -> PretrainReport:
        """产物落盘：判别器 checkpoint（可装载 state_dict，与训练期产物
        checkpoint 同构）+ 预训练报告（kind 标识 + per-condition
        recon-AUC + 条件白名单 + 支撑度卷数 + 数据口径指纹，含 checkpoint
        内容指纹——报告的白名单与实测值只对落盘这份权重负责，装载面按
        指纹对照，``load_discriminator``）。白名单为空同样落盘——报告与
        checkpoint 是失败预训练的诊断产物，不丢。"""
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
            # 判据口径标识（ADR-0012 决策 5 的审计面）：本报告的 AUC 是
            # held-out real 原始 vs 冻结基座同源重构体的 recon-AUC——
            # 与在线 iter 事件的 rollout-AUC 不可横向比较
            gate_criterion="recon_auc",
            condition_volumes=condition_volumes,
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
