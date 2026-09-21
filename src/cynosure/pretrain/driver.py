"""判别器 warm-start 预训练 driver（ADR-0007：预训练 + 继续在线更新；
ADR-0008 决策 3：per-condition 步进与终止；ADR-0016：torchrun 测量批
分片 + rank0 gate）。

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

单进程与 torchrun 同一条代码路径（ADR-0016 决策 3）：``DistributedContext``
无 torchrun 环境时 bootstrap 为 world-1 恒等（不初始化进程组、集合通信
与广播原样返回传入值），driver 的 rank0 判定/分发退化为本地判定——
World-1 行为与分布式化前**逐位一致**（同代码路径、同 RNG 消耗序）。
分布式下（``torchrun --nproc_per_node=N``）：

- **测量批按卷切片**（决策 1/4）：全量索引排列照旧经 ``heldout_auc``
  命名流抽出（各 rank 本地流同步推进——复测换批的消耗序逐 rank 对齐；
  rank0 的派生 seed = seed 恒等，其排列与单进程逐位同序），rank0 的
  排列经 ``broadcast_object`` 镜像到全 rank；每 rank 只 ``load_order``
  本地连续段（测量批不整批驻留，#174 OOM 修复原则延伸到加载侧），
  切片的 σ 轮转按全量位次偏移、ε 走测量流前缀消耗（
  ``measure_condition(volume_offset=...)``——切片与整批逐位同位，
  gather 拼回的全量测量与单卡 rank0 全量测量数值等价）；
- **gate 控制流 rank0 单点**（决策 5）：每 rank 本地切片打分 → 逐卷
  分数 object-gather 到 rank0（分数先落 CPU：对象集合的 pickle 不携带
  跨 rank 设备语义，且 AUC 秩统计是 CPU 工作负载）→ rank0 重组全量
  ``VolumeScoreClusters``（卷级归属保留、连续段按 rank 序拼接还原全量
  排列序）重算全局 recon-AUC（Mann-Whitney estimand 不变）→
  ``SupportRule`` 判定、复测确认、白名单棘轮、``steps_completed`` 计数
  rank0 单点 → ``broadcast_object`` 分发步进四态（更新/复测/确认/终止，
  world-1 下广播恒等 = 本地判定）。控制流单点 = 集合序列单点：gather
  与广播的调用次数由 rank0 判定驱动，全 rank 同一条执行序，复测确认步
  全 rank 一致跳过更新（确认/终止路径的集合对齐由此成立，#165 式挂死
  面不外露）；
- **判别器更新 = 在线期同款 DDP**（决策 2）：经 ``TrainingRuntime.
  assemble_rewards`` 装配缝自动生效——``ReplicatedDiscriminator.replicate``
  梯度 allreduce（更新后各 rank 权重逐位一致）、``RankSlicedPool`` real
  池条带（容量守卫 K×world_size 把门，装配期全 rank 一致拒绝）；有效
  batch = K×N，config 字面不变，world_size 是唯一变量；
- **冻结基座不 FSDP**（决策 6）：每 rank 完整副本，不注入 ``chunk_sync``
  （各 rank 同条件同尺寸，分块 cap 天然一致，无集合可绑定）；
- **产物写者 rank0 门**（决策 8）：事件流（``metrics.jsonl``）、判别器
  checkpoint、预训练报告全部 rank0 落盘（``PretrainRun`` 单进程唯一
  写者契约由调用方 rank 门满足）；``OverfitMonitor`` 保持 rank 本地状态
  （ADR-0009 决策 4/5：per-rank 离散本身是诊断信号，不跨 rank 平均），
  各 rank 以广播下发的全局 AUC 合成观测，越线告警的归并落 rank0 事件
  流（最小版：rank0 写本 rank 读数）。

判别器侧装配经 ``TrainingRuntime.assemble_rewards``（配对批装配原语同缝
组装）、采样封装经 ``TrainingRuntime.assemble_sampler``、policy 侧经
``GroupPolicy.build``（组1/组2 的采样场与条件分布按 config 分派）——
与在线期同一份装配与同一条执行路径，仅 config 不同。
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
from cynosure.reward.auc import VolumeScoreClusters
from cynosure.reward.support import SupportRule
from cynosure.train.artifacts import OverfitAlertEvent, PretrainEvent
from cynosure.train.policy import GroupPolicy
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime

if TYPE_CHECKING:
    from collections.abc import Sequence


_DECISION_UPDATE = "update"
_DECISION_REMEASURE = "remeasure"
_DECISION_CONFIRM = "confirm"
_DECISION_HALT = "halt"
"""gate 步进四态（ADR-0016 决策 5 的广播协议值）：rank0 判定的分发面，
world-1 下 ``broadcast_object`` 恒等 = 本地判定。``update`` = 本步更新；
``remeasure`` = 首测过线、换批复测；``confirm`` = 复测确认入白名单、
跳过更新；``halt`` = 末个条件确认、全体终止。消息体 = (态, AUC)：更新/
复测态携带首测值（更新步事件与分叉观测的全局 recon-AUC 记账面），
确认/终止态携带 ``min(首测, 复测)`` 的保守报告值。"""


class PretrainDriver:
    """判别器 warm-start 预训练编排：装配（world-1 恒等退化 ↔ torchrun
    多 rank 同一路径）→ 密集步进（rank0 gate 单点 + 四态分发）→ 产物
    落盘（判别器 checkpoint + 预训练报告，rank0 唯一写者）。"""

    def __init__(
        self,
        config: CynosureConfig,
        run: PretrainRun,
        device: torch.device | None = None,
        dist_context: DistributedContext | None = None,
    ) -> None:
        self._config = config
        self._run = run
        reward = config.reward
        # 单进程与 torchrun 同一条装配序（ADR-0016 决策 3）：无 torchrun
        # 环境下 bootstrap 为 world-1 恒等（不初始化进程组），集合通信与
        # 广播原语退化——与 train 同一条装配序；CLI 层装配进程组后经
        # ``dist_context`` 注入（run 目录 init 的广播裁决先于 driver 构造）
        self._dist = (
            dist_context
            if dist_context is not None
            else DistributedContext.bootstrap()
        )
        amp = AmpContext(
            device=device if device is not None else self._dist.local_device(),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        generators = TrainingRngStreams(
            self._dist.derive_seed(config.schedule.seed),
            # 与 train runtime 同一 seeding 规则（数据侧逐 rank 派生、
            # recon 用 shared）——rank 0 恒等偏移 = 分布式下 rank0 的全部
            # 数据流与单进程逐位同序（重放锚）
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
            config, amp, generators, self._dist,
            sampler=sampler, conditions=self._policy.conditions,
        )
        # 量产 rollout（``RolloutPhase``）不装配：ADR-0012 决策 6 后预训练
        # 相 fake 全由装配原语重构产出（测量批 / 更新批两条入口），无
        # rollout 相的消费者——装配它只会让「fake 是否走了量产」留一条
        # 静默可用的旧路（base_partition 流已随 ADR-0012 退役，#173）。
        #
        # 过线判定原语（ADR-0008-04 消费 ADR-0008-02/#85 的支撑度规则）：
        # bootstrap 的随机性独立派生（seed+7——命名流注册表之外，预训练
        # 不参与续训、判定可复现性由 seed 纯函数保证；进注册表反而令续训
        # 状态清单失配）。判定只在 rank0 发生（分布式下非 0 rank 的流
        # 不消费——广播镜像 rank0 的决定，ADR-0016 决策 5）
        self._support = SupportRule(
            threshold=reward.pretrain_gate_auc,
            support_bound=reward.gate_support_min_volumes,
            generator=torch.Generator().manual_seed(
                self._dist.derive_seed(config.schedule.seed + 7),
            ),
        )
        # 轮转条件集守卫：每条件 held-out 非空（per-condition AUC 归因的
        # 前提——缺条目的条件在装配期显式拒绝，而非首步测量时才炸）；
        # 分布式下收紧为 ≥ world_size（测量批按卷切片到各 rank，每 rank
        # 至少 1 卷——与 RankSlicedPool 的切片前全量校验同理由：失败路径
        # 全 rank 一致，manifest 查询无通信、各 rank 同判定）
        minimum_volumes = self._dist.world_size if self._dist.distributed else 1
        starved = [
            target
            for target in self._policy.conditions.targets()
            if self._rewards.auc.condition_volume_count(target) < minimum_volumes
        ]
        if starved:
            if not self._dist.distributed:
                raise ValueError(
                    f"held-out real 缺条件 {starved} 的条目（per-condition 步进"
                    "要求轮转条件集每条件 held-out 非空——AUC 测量按条件归因"
                    f"无米下锅；heldout_real_manifest={reward.heldout_real_manifest}）"
                )
            raise ValueError(
                f"held-out real 不足以支撑 {self._dist.world_size}-路测量批"
                f"切片（每条件每 rank 至少 1 卷，不足条件: {starved}；"
                f"heldout_real_manifest={reward.heldout_real_manifest}）："
                "切片后某 rank 测量批为空——gather 的全局 AUC 缺该 rank "
                "分数即不完整"
            )

    @property
    def policy(self) -> GroupPolicy:
        """本组 policy 侧装配（冻结 base；组1/组2 的采样场与条件分布）。"""
        return self._policy

    @property
    def rewards(self) -> RewardCoordinator:
        """判别器侧协作者组（Online update 原语 / held-out AUC / 过拟合分叉监控）。"""
        return self._rewards

    def run(self) -> PretrainReport | None:
        """密集步进至全部轮转条件过线（ADR-0008 决策 3 的 per-condition
        终止语义）或步数上限，产出判别器 checkpoint 与预训练报告（产物
        rank0 唯一写者；非 0 rank 返回 None——报告与 checkpoint 无多写者
        消费者，CLI 层只 rank0 打印）。

        每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态均匀轮转），
        以该条件**全量 held-out 卷**的冻结基座同源重构作测量批（分布式
        下按卷切片到各 rank、rank0 重组全局分数，见 ``_measurement``；
        recon-AUC 的 fake 侧）、real 同条件同批配对、AUC 归因该条件；首测
        过线（``SupportRule.passes``，rank0 单点判定 + 四态广播）换新批复
        测确认——两次独立测量都过线才入白名单（producer 侧成功判据对单批
        测量噪声鲁棒，train 侧按独立采样的重算不再与非确定性拒绝耦合），
        报告值取两次较小者。已入白名单的条件不再复测（棘轮：复测确认
        已拦住单批噪声，在线期的掉线由白名单动态恢复机制兜底）；
        ``pretrain_max_steps`` 耗尽 → 白名单 = 已确认者、未确认条件对
        落盘权重补测（补测循环全 rank 走同一目标序列——gather 的集合
        对齐由「reported 全 rank 一致」保证）。白名单为空仍落盘全部产物
        （拒跑由 train gate 把守）。"""
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
            if self._dist.rank == 0:
                # 支撑度判定的卷数留痕（rank0 的全局口径：分布式下 =
                # 全量卷数，聚合于 ``_measurement`` 的 gather 重组）
                assert clusters is not None  # rank0 恒拿到全局聚类
                volumes[modality] = clusters.volume_count
            # gate 首测判定：rank0 单点（分布式）/ 本地（world-1 恒等），
            # broadcast_object 在 world-1 下原样返回传入值——控制流全
            # rank 同一条，集合序列（gather/广播的调用次数）随之对齐
            decision = self._first_decision(modality, confirmed, clusters)
            decision = self._dist.broadcast_object(decision)
            if decision[0] == _DECISION_REMEASURE:
                # 复测（同条件换批测量）：判别器权重同刻，变化的随机面
                # 只有一处——held-out 全量卷的抽取序（各 rank 本地
                # heldout_auc 流同步推进 → 新排列；广播镜像保证全 rank
                # 同批）。重构 ε 经批次起手复位 + 前缀消耗逐位同位（
                # ``measure_condition(volume_offset=...)``）：重排列把每卷
                # 配到的 (σ, ε) 槽位换掉，≥2 卷条件下两次读数是不同样本；
                # 单卷条件排列平凡、复测与首测同读数（确认退化——小池由
                # 数据侧池规模与支撑度界兜底，不以本相为抗噪防线）。复测
                # 只读 AUC——配对批不留存（与首测批同一释放口径，见下）
                first_auc = decision[1]
                _, confirm_clusters, _ = self._measurement(modality)
                decision = self._confirm_decision(
                    modality, confirmed, targets, confirm_clusters, first_auc,
                )
                decision = self._dist.broadcast_object(decision)
                if decision[0] in (_DECISION_CONFIRM, _DECISION_HALT):
                    confirmed[modality] = decision[1]  # 保守口径：两次取小
                    del batch  # 确认/终止步不更新：测量批到此释放
                    if decision[0] == _DECISION_HALT:
                        gate_passed = True  # 全部条件过线：终止
                        break
                    continue  # 本条件已确认：本步不更新（无更新即无事件）
            # 测量批到此消费完毕（AUC 已归因、卷数已留痕 volumes）——
            # 配对张量不进更新步（#174 生产重跑 OOM 修复：全量 held-out
            # 测量批在大尺寸条件下数十 GiB 驻留，与更新批装配叠加是
            # 64 GiB 卡的 OOM 峰值；分布式下每 rank 只驻留本地切片、释放
            # 更早；事件面的 measurement_volumes 走 volumes 留痕，同值）
            del batch
            auc = decision[1]  # 更新步的 heldout_auc（全局口径）
            update = self._rewards.update_step(
                # 更新批 = 装配原语的配对批（ADR-0012）：fake = 冻结基座
                # 对同批 real 的同源重构（专属 recon 流、先抽 s 后抽 ε、
                # η=0 确定性 ODE 续跑）——与在线更新同一原语供批、判别器
                # 任务两阶段同构（warm-start 权重不面临分布跳变）。分布式
                # 下 DDP 自动装配（assemble_rewards 缝）：本 rank 的 K 条
                # 配对批、梯度 allreduce，有效 batch = K×world_size。与本步
                # 测量批同一原语的另一条入口（定序轮转 σ + 复位测量流）
                self._rewards.assembler.assemble(modality),
            )
            # 过拟合分叉观测（ADR-0009-γ）：与在线同一监控组件、同一
            # knobs（共享装配缝挂进 RewardCoordinator 的 OverfitMonitor，
            # 阈值/跨度同源于 config.reward.overfit_*）——train 侧干净域
            # 复算准确率（随更新报告上行）与本步更新前 recon-AUC 合成
            # 分叉观测，per-condition EMA 越线即落预训练相告警（确认步
            # 不更新不观测；报警不动作，人工裁决——口径同在线。两侧估计
            # 量同为 Mann-Whitney pairwise 占比、同为干净域，只差 in/out
            # of sample 平面与 fake 来源）。OverfitMonitor 保持 rank 本地
            # 状态（ADR-0016 决策 8：per-rank 离散本身是诊断信号，不跨
            # rank 平均）——全 rank 以广播下发的全局 AUC 合成观测，越线
            # 告警由 rank0 写本 rank 读数落事件流（最小版归并）
            reading = self._rewards.overfit.observe(
                modality,
                train_pairwise_acc=update.train_pairwise_acc,
                heldout_auc=auc,
            )
            if self._dist.rank == 0:
                self._run.append_event(PretrainEvent(
                    step=step,
                    modality=modality,
                    loss_discriminator=update.loss_discriminator,
                    heldout_auc=auc,
                    # 重构成本读数（#171 AC5 的成本口径落点）：测量批重构的
                    # 前向次数（逐卷定序 σ 的续跑步数之和）与测量批规模——
                    # 30 步全 ODE 量产路径已不在本执行路径，这两项让「没有
                    # 量产」在事件流上可核对（口径见模块 docstring）。分布式
                    # 下两者都是全局口径（各 rank 本地读数经 gather 求和/
                    # 卷数聚合），语义与单进程逐字一致
                    reconstruction_forwards=forwards,
                    measurement_volumes=volumes[modality],
                    lr=reward.disc_lr,
                    elapsed_s=time.monotonic() - started,
                ))
                if reading.alerted:
                    # 预训练相告警排本步 pretrain 事件之后（与在线侧「iter
                    # 后随告警」同构的写出序）；``phase="pretrain"`` 是回退
                    # 记账的 EXEMPT 分轨轴——预训练执行史全量保留（预训练
                    # 相的 ``iteration`` 记本步步号）
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
            # 已确认条件的报告值 = 确认时的两次较小者，保留不覆盖）。补测
            # 的测量/gather 全 rank 一致参与（reported 由广播驱动的
            # confirmed 全 rank 一致 → 目标序列全 rank 相同 → 集合序列
            # 对齐）；只有 rank0 的读数进报告
            for target in targets:
                if target not in reported:
                    _, clusters, _ = self._measurement(target)
                    if self._dist.rank == 0:
                        assert clusters is not None
                        reported[target] = clusters.pooled_auc()
                        volumes[target] = clusters.volume_count
        if self._dist.rank != 0:
            return None  # 产物 rank0 唯一写者：非 0 rank 无报告消费者
        return self._finalize(
            steps_completed, reported, list(confirmed), gate_passed, volumes,
        )

    def _first_decision(
        self,
        modality: str,
        confirmed: dict[str, float],
        clusters: VolumeScoreClusters | None,
    ) -> tuple[str, float]:
        """首测判定（rank0 单点）：未确认且过线 → 复测；否则 → 更新。
        非 0 rank 的 ``clusters`` 为 None（本地切片分数已在 gather 后由
        rank0 消费），返回占位（``broadcast_object`` 覆盖，值无关紧要）。
        world-1 下广播恒等，本方法即全部判定。"""
        if self._dist.rank != 0:
            return (_DECISION_UPDATE, 0.0)
        assert clusters is not None
        auc = clusters.pooled_auc()  # 更新前快照（本步判别器权重）
        if modality not in confirmed and self._support.passes(auc, clusters):
            return (_DECISION_REMEASURE, auc)
        return (_DECISION_UPDATE, auc)

    def _confirm_decision(
        self,
        modality: str,
        confirmed: dict[str, float],
        targets: "Sequence[str]",
        confirm_clusters: VolumeScoreClusters | None,
        first_auc: float,
    ) -> tuple[str, float]:
        """复测判定（rank0 单点）：复测亦过线 → 确认（报告值 = 两次较小）
        或终止（最后一个条件确认完毕）；未过线 → 更新（首测值记账）。
        确认/终止态全 rank 一致跳过更新——复测确认步的集合对齐由四态
        广播的单点语义保证。"""
        if self._dist.rank != 0:
            return (_DECISION_UPDATE, first_auc)
        assert confirm_clusters is not None
        confirm_auc = confirm_clusters.pooled_auc()
        if self._support.passes(confirm_auc, confirm_clusters):
            value = min(first_auc, confirm_auc)
            if len(confirmed) + 1 == len(targets):
                return (_DECISION_HALT, value)
            return (_DECISION_CONFIRM, value)
        return (_DECISION_UPDATE, first_auc)

    def _measurement(
        self, modality: str,
    ) -> tuple[PairBatch, VolumeScoreClusters | None, int]:
        """单条件测量批 → 卷级分数聚类（gate 测量/复测/补测共用入口）。

        测量批 = 该条件**全量 held-out 卷**的冻结基座同源重构（装配原语
        ``measure_condition``：定序轮转 σ + 复位测量流 ⇒ 逐次测量逐位同
        输出）；real 侧**就是这批重构的源**（同一次索引排列抽取的逐样本
        配对）——recon-AUC 的判别目标因此只剩重构伪影（ADR-0012 决策 5）。
        与更新批同条件归因（ADR-0008-03 条件归因口径），条件轴与本步
        更新的条件轴一致。

        随机流：held-out 全量卷的抽取消耗 ``heldout_auc`` 命名流（卷内
        顺序不影响读数——AUC 是集合级秩统计），重构的 ε 走装配原语的
        复位测量流（不碰 recon 流）。预训练不参与续训，两处消耗都由
        seed 纯函数确定 ⇒ 同 seed 同 world_size 重跑逐位可复算。

        World-1（``distributed=False``）：全量路径——排列抽出即全量加载，
        行为与分布式化前逐位一致。分布式：各 rank 本地 heldout_auc 流
        同步推进抽排列（消耗序对齐，复测换批 = 流推进的语义逐 rank 同
        构），rank0 的排列经 ``broadcast_object`` 镜像（rank0 派生 seed
        恒等 ⇒ 与单进程逐位同序）；每 rank 只 load 本地连续段切片（
        ``volume_offset`` 把 σ 轮转与 ε 前缀消耗对齐到全量位次——切片
        行与整批对应行逐位一致）；本地切片打分后逐卷分数 object-gather
        到 rank0（分数先落 CPU：对象集合的 pickle 不携带跨 rank 设备
        语义，且秩统计是 CPU 工作负载），rank0 按 rank 序重组全量聚类
        （卷级归属保留、连续段拼接还原全量排列序），全局前向次数 =
        各 rank 本地读数之和。返回：rank0 = (批, 全局聚类, 全局前向数)，
        非 0 rank = (本地批, None, 0)——聚类已被 rank0 消费，占位仅为
        类型对齐。
        """
        assembler = self._rewards.assembler
        if assembler is None:
            raise ValueError(
                "配对批装配原语未装配（RewardCoordinator.assembler=None）："
                "预训练测量批与更新批同源于它（ADR-0012）"
            )
        auc = self._rewards.auc
        if not self._dist.distributed:
            # real 侧先抽一次（该条件全量 held-out 卷）：同一批张量既作
            # AUC 的 real 侧、又作重构的源——逐样本配对由构造保证
            reals = auc.condition_latents(modality)
            batch = assembler.measure_condition(reals, modality)
            clusters = auc.compute_volume_clusters(
                batch.reals, batch.fakes,
            )
            return batch, clusters, assembler.measurement_forward_count(
                reals, modality,
            )
        # 分布式：同排列（rank0 抽取 + 广播镜像）、切加载（每 rank 本地段）
        order = auc.condition_order(modality)
        order = self._dist.broadcast_object(
            order if self._dist.rank == 0 else None,
        )
        start, stop = self._shard_bounds(len(order))
        reals = auc.load_order(order[start:stop])  # 只 load/驻留本 rank 段
        batch = assembler.measure_condition(
            reals, modality, volume_offset=start,
        )
        clusters = auc.compute_volume_clusters(batch.reals, batch.fakes)
        forwards = assembler.measurement_forward_count(
            reals, modality, volume_offset=start,
        )
        # 分数落 CPU 再 gather：对象集合经 pickle 传输，CUDA 设备号不随
        # 对象迁移（生产 NCCL 下跨 rank 反序列化会指向不存在的源设备）；
        # AUC/bootstrap 的秩统计与重采样本来就工作在 CPU 负载面上
        gathered = self._dist.gather([
            VolumeScoreClusters(
                real_volume_scores=tuple(
                    volume.cpu() for volume in clusters.real_volume_scores
                ),
                fake_scores=clusters.fake_scores.cpu(),
            ),
            forwards,
        ])
        if self._dist.rank != 0:
            return batch, None, 0
        rank_clusters = [submission[0] for submission in gathered]
        global_clusters = VolumeScoreClusters(
            real_volume_scores=tuple(
                volume
                for rank in range(self._dist.world_size)
                for volume in rank_clusters[rank].real_volume_scores
            ),
            fake_scores=torch.cat([
                rank_clusters[rank].fake_scores
                for rank in range(self._dist.world_size)
            ]),
        )
        return (
            batch,
            global_clusters,
            sum(submission[1] for submission in gathered),
        )

    def _shard_bounds(self, total: int) -> tuple[int, int]:
        """全量测量批按 rank 均分的连续段边界（前余均分：排位 < 余数的
        rank 多领一卷）。连续段（非条带）使 gather 后的 rank 序拼接
        **还原全量排列序**——「切片合并还原全量」是全局 AUC 的卷级
        归属前提；末段 rank 的切片可能为空（total < world_size 的防御
        已由装配期每条件 ≥ world_size 守卫把住，此处恒有 start < stop）。"""
        world = self._dist.world_size
        base, remainder = divmod(total, world)
        start = self._dist.rank * base + min(self._dist.rank, remainder)
        stop = start + base + (1 if self._dist.rank < remainder else 0)
        return start, stop

    def _finalize(
        self,
        steps_completed: int,
        condition_auc: dict[str, float],
        whitelist: list[str],
        gate_passed: bool,
        condition_volumes: dict[str, int],
    ) -> PretrainReport:
        """产物落盘（rank0 唯一写者）：判别器 checkpoint（可装载
        state_dict，与训练期产物 checkpoint 同构）+ 预训练报告（kind
        标识 + per-condition recon-AUC + 条件白名单 + 支撑度卷数 + 数据
        口径指纹，含 checkpoint 内容指纹——报告的白名单与实测值只对
        落盘这份权重负责，装载面按指纹对照，
        ``load_discriminator``）。白名单为空同样落盘——报告与
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
