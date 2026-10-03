"""判别器 warm-start 预训练 driver（ADR-0007：预训练 + 继续在线更新；
ADR-0008 决策 3：per-condition 步进与终止；**ADR-0018：单进程多卡
async 化**，#221 结票全口径——本模块 docstring 即该口径的落位说明）。

密集步进循环：每步条件 = 轮转条件集的 ``targets[step % n]``（目标模态
均匀轮转，确定性不耗 RNG）→ 该条件的过线测量批 = 装配原语对全量
held-out 卷的冻结基座同源重构（定序轮转 σ + 复位测量模板 ⇒ 同输入同
输出、可复算）→ 以更新前快照测该条件 recon-AUC（real = held-out real
原始、fake = 其重构体，逐样本配对；更新后测同一测量批会把 in-sample
拟合计入 AUC）→ 支撑度规则判定过线（``SupportRule.passes``：该条件
held-out 卷数 < 界用 bootstrap CI 下界、≥ 界用点估计——ADR-0008
决策 6 / #85）→ 首测过线换新批复测确认：两次独立测量都过线该条件
确认过线（单批贴线越过被非确定性拒绝），报告值取两次较小者，确认步
不更新（无更新即无事件）→ 未确认则以**同一装配原语**产出的配对批
（real + 冻结基座同源重构 fake，ADR-0012）走在线期同构的判别器单步
更新（两阶段构造同构、warm-start 权重不面临分布跳变）。每个更新步
同时消费与在线**同一**过拟合分叉监控组件、同一 config knobs
（ADR-0009-γ：``OverfitMonitor`` 每卡实例——本地 train acc 与全局
recon-AUC 合成分叉观测，per-condition EMA 自下而上越线落预训练相
``overfit_alert`` 事件（``phase="pretrain"``，EXEMPT 记账——预训练
执行史全量保留）；只报警不动作，确认步不更新不观测）——per-condition
分叉监控在预训练棘轮终止之前即暴露稀疏模态（MRA）记忆化。已确认过线
的条件不再复测（棘轮：复测确认已拦住单批噪声）。终止 = 全部条件最近
一次确认过线即停；``pretrain_max_steps`` 耗尽 → 过线条件 = 已确认者，
未确认条件逐个对落盘权重补测（报告值与 checkpoint 同快照）。过线条件
为空同样落盘全部产物——报告与 checkpoint 是诊断产物，不丢（warm-start
装载无门槛判定，ADR-0017）。

**预训练相不产 rollout**（ADR-0012 决策 6）：量产 rollout 整体退出本
执行路径——fake 侧只剩「全量 held-out 卷的重构」（测量批，σ 定序
轮转）与「更新批的重构」（σ 逐样本抽自被优化步）两条，均经装配原语；
事件流的 ``reconstruction_forwards`` / ``measurement_volumes`` 是本
口径的读数面。判据口径的两阶段差异记录在案（ADR-0012 决策 5）：预训练
判据是 recon-AUC（判别器训练任务的 out-of-sample 泛化力）、在线运行
口径是 rollout-AUC（对打分对象的分辨力），**不可跨阶段比较绝对值**
——准入体检 vs 在岗考核。

**执行形态（ADR-0018，取代 ADR-0016 的 torchrun 多进程拓扑）**：单进程
多卡，步进循环 resident **主控（编排层）**，每卡一条静态绑卡执行线程
（``PretrainCardWorker``，#217 门面形态——任务粒度 = 每卡每相一个单元
任务直交绑卡线程；预训练无分配表、无 k 循环、无协程语义面，任务载体
不承载任何决策）。单步序：**排列抽取（主控单点）** → 测量扇出（每卡
一片）→ join → 卡序 plain 拼接卷级分数聚类 → pooled AUC（CPU）→
SupportRule / 复测确认 / 棘轮 / steps_completed 判定 **inline 主控**
（单进程无集合可对齐，``broadcast_object``/``broadcast_flag``/
object-gather 退役——四态语义保留为编排控制值）→ 更新扇出（每卡
K 对装配 + 本地前向反向）→ join → 判别器步（逐对等权 loss 的跨卡
梯度 SUM allreduce（M1 形态，主控单线程驱动多卡）→ 各卡 AdamW →
步末 u/v broadcast（卡 0 权威，与在线协议统一））→ 事件主控直写 +
告警卡序追加。

**随机面（#221 决议 13 消费清单）**：held-out 排列 = ``seed+3`` 主控
单点直派（``RealPoolSampler.permutation`` 一次 randperm，每步单点抽
全量排列 → ``ShardPlan`` 切片分派——各 rank 本地流同步抽 +
``broadcast_object`` 镜像退役）；测量模板 = ``seed+19`` 主控显式直锚
（复位一次 → 条件构造一次 → 按卡序逐段抽 ε；``volume_offset`` 前缀
消耗退役——分段 ≡ 全量的顺序流等价性由行宽 ≡ 0 (mod 16) 的装配期
断言与 #198 平移锚守护）；SupportRule bootstrap = ``seed+7`` 单点；
冷启动判别器 = ``seed+6``（``assemble_scorer`` 既有口径）；更新批
recon 流 = **卡轴**派生 ``seed + 卡×10⁶ + 9``（卡 0 恒等 = 现行 rank0
recon 数值；每卡 σ 位与 ε 独立、「先 s 后 ε」次序契约保持——accepted
drift：卡 ≥1 的 σ/ε 进迁移轨迹对照验收）；real 侧 = 主控对全池全局
无放回抽 K×卡数再切片（``seed+1`` 单点直派——``RankSlicedPool`` 条带
切片退役，抽取空间条带→全池、抽取者各卡→主控，同为 accepted drift）；
rollout 流恒零消费死条目（``TrainingRngStreams`` per-卡实例的四流中
仅 recon 有消费者）。恒等面（排列 +3 / 模板 +19 / SupportRule +7 /
冷启动 +6 / 卡 0 recon +9）与 drift 面（real 抽取空间与抽取者、
卡 ≥1 σ/ε）的对照表见 ADR-0018。

**产物契约零改动（决议 17/18）**：checkpoint = 判别器步末**卡 0 副本
直写**（确定性 allreduce + u/v broadcast 保证全卡逐位一致，含
``_u``/``_v`` buffer；``loadable_state_dict`` 键集/格式不变；不设
checkpoint 周期 bitwise 守卫 = 显式裁决——run 短风险低）；``Pretrain
Report`` 字段集与 provenance 指纹面不变；``PretrainEvent`` 字段不动
（``reconstruction_forwards`` / ``measurement_volumes`` = 全量 σ 列表
推算与全量排列长度——分段求和的加法结合恒等）；train 侧消费面
（``load_discriminator`` + 组别绑定守卫 + 报告白名单）零改动。
告警轴 (步, rank) → (步, 卡) 的迁移对象仅 ``OverfitAlertEvent``
（``rank`` 字段归因观测卡；``OverfitMonitor`` 每卡实例——本地
train acc + 全局 AUC 的卡轴诊断保持，per-卡离散本身是诊断信号）；
EventMerger 归并与写者门退役——主控是唯一写者，步内 pretrain 事件
先直写、告警后按卡序追加（写出序 = 步序 + 步内 pretrain 先于告警）。

复现承诺（决议 19）：#218 决议 0 双口径（生产统计等价 / 测试进程
逐位）+ **同 config 同卡数重放逐位**；跨卡数重跑 = 新 run（预训练无
payload、天然无拒绝面——与 RL 跨拓扑显式拒绝的差异见 ADR-0018）。

判别器侧装配复用 ``TrainingRuntime`` 公开装配缝（scorer 冷启动 / 配对
批装配原语 / real pool 守卫装载 / schedules / 数值口径）——「无第二套
判别器训练逻辑」在装配层同样成立；判别器步相位编排复用 #234 的
``DiscriminatorPhase``（先全桶 eval 后全桶 train、逐对等权加权
backward、跨卡 SUM allreduce、步末 u/v broadcast）——预训练每步单条件
单桶，逐对等权 ≡ 现行 patch 级 mean（每对 patch 数恒等，决议 3）。
"""

import copy
import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass

import torch

from cynosure.config import CynosureConfig
from cynosure.conditions import ConditionVocabulary
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.condition import RolloutCondition
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.schedules import ConditionSchedules
from cynosure.pretrain.artifacts import (
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.pretrain.measurement import MeasurementTemplate
from cynosure.pretrain.sharding import ShardPlan
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.assembly import ReconstructionAssembler
from cynosure.reward.auc import HeldOutAuc, VolumeScoreClusters
from cynosure.reward.overfit import OverfitMonitor
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import LatentScorer
from cynosure.reward.support import SupportRule
from cynosure.reward.update import OnlineUpdate, UpdateReport
from cynosure.train.artifacts import OverfitAlertEvent, PretrainEvent
from cynosure.train.discriminator import DiscriminatorBucket, DiscriminatorPhase
from cynosure.train.policy import GroupPolicy
from cynosure.train.rng import SLOT_SEED_STRIDE, TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime


_WORKER_JOIN_TIMEOUT_S = 10.0
"""绑卡线程收尾的 join 上限（秒）：卡死线程放行给进程退出兜底
（daemon 线程——预训练卡任务无进程组集合操作，挂死面只剩 CUDA op
本身的极端故障，口径与 async 门面一致）。"""


@dataclass(frozen=True)
class PretrainCardRig:
    """单卡的预训练组件组（绑卡线程装配、单线程消费）：冻结基座
    policy 副本 + 判别器 scorer 副本 + 配对批装配原语（卡轴 recon 流）
    + 判别器步相位编排 + 数值口径 + 两侧 manifest 的加载面。"""

    policy: GroupPolicy
    scorer: LatentScorer
    assembler: ReconstructionAssembler
    disc_phase: DiscriminatorPhase
    amp: AmpContext
    real_pool: LatentManifest
    heldout: LatentManifest


class PretrainCardWorker:
    """每卡预训练执行线程（#217 门面的直交提交形态）：静态绑卡
    （current device 线程局部语义），组件装配在本线程完成（卡轴
    recon 流的 owner = 绑卡线程），任务经 ``submit`` 直交、``Future``
    回传（异常自动传播——主控 join 即 fail-fast 面）。"""

    def __init__(
        self,
        index: int,
        device: torch.device,
        build_rig: Callable[[], PretrainCardRig],
    ) -> None:
        self.index = index
        self.device = device
        self._rig: PretrainCardRig | None = None
        self._started = False
        self._build_rig = build_rig
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name=f"cynosure-pretrain-card-{index}",
        )

    def _serve(self) -> None:
        """线程本体：任务队列常驻（(动作, Future) 序对逐个执行——
        单线程串行即绑卡语义）；``None`` 哨兵 = 收尾。"""
        while True:
            task = self._queue.get()
            if task is None:
                return
            action, future = task
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(action())
            except BaseException as error:
                future.set_exception(error)

    def start(self) -> None:
        """启动线程并等待组件装配就绪（装配失败在此原样抛出，不留
        半装配线程；幂等——先行探针式 start（rig 取数面）后接 ``run``
        的正式 start 不重复启动）。"""
        if self._started:
            return
        self._started = True
        self._thread.start()
        self.submit(self._assemble).result()

    def _assemble(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        rig = self._build_rig()
        rig.policy.eval_phase()  # 冻结基座的推理相（重构是 policy 前向）
        # 打分/监控前向恒 eval（判别器相位由 disc_phase 编排）
        rig.scorer.eval()
        self._rig = rig

    def submit(self, action: Callable, /, *args) -> Future:
        """单元任务直交绑卡线程（``Future`` 回传：``result()`` 即主控
        的 join 与 fail-fast 面）。"""
        future: Future = Future()
        self._queue.put((lambda: action(*args), future))
        return future

    def stop(self) -> None:
        """线程收尾：哨兵停转 + 限时 join（卡死线程放行给进程退出
        兜底，见 ``_WORKER_JOIN_TIMEOUT_S`` 口径）。"""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(_WORKER_JOIN_TIMEOUT_S)

    @property
    def conditions(self):
        """本卡条件分布（主控测量模板的条件构造消费面——穿模板流，
        抽取消耗传入的 generator、不触碰实例自有流）。"""
        return self._require_rig().policy.conditions

    @property
    def disc_phase(self) -> DiscriminatorPhase:
        """本卡判别器步相位编排（主控跨卡集合段的取数面）。"""
        return self._require_rig().disc_phase

    def measure_segment(
        self,
        entries: Sequence[PoolEntry],
        sigmas: Sequence[float],
        noise: torch.Tensor,
        condition: RolloutCondition,
    ) -> VolumeScoreClusters:
        """测量相的本卡段：real 连续段上卡 → 冻结基座同源重构
        （主控给定的定序 σ 与模板流 ε——本段零随机消耗）→ 本卡判别器
        副本打分的卷级分数聚类（CPU 形态回传——跨线程传递不带设备
        语义，秩统计是 CPU 工作负载）。"""
        rig = self._require_rig()
        reals = self._load(rig.heldout, entries)
        with torch.no_grad(), torch.autocast(
            rig.amp.device_type, dtype=rig.amp.dtype,
        ):
            fakes = rig.assembler.reconstruct(
                reals, condition.to(self.device), list(sigmas),
                noise.to(self.device),
            )
        clusters = HeldOutAuc.volume_clusters(reals, fakes, rig.scorer)
        return VolumeScoreClusters(
            real_volume_scores=tuple(
                volume.cpu() for volume in clusters.real_volume_scores
            ),
            fake_scores=clusters.fake_scores.cpu(),
        )

    def update_segment(
        self, entries: Sequence[PoolEntry], modality: str,
    ) -> UpdateReport:
        """更新相的本卡段：主控全局无放回抽取的本卡 K 对 real 上卡 →
        配对批装配原语（卡轴 recon 流：先 s 后 ε、同源重构）→ 单条件
        单桶的判别器步相位段（先 eval 复算 train acc 后 train 加权
        backward——梯度落本卡 .grad，跨卡 SUM 由主控编排）。"""
        rig = self._require_rig()
        reals = self._load(rig.real_pool, entries)
        pair = rig.assembler.reconstruct_assigned(reals, modality)
        bucket = DiscriminatorBucket(
            condition=modality, reals=pair.reals, fakes=pair.fakes,
        )
        return rig.disc_phase.accumulate([bucket])

    def disc_step(self) -> None:
        """判别器步收尾（主控跨卡梯度 SUM 之后）：本卡 optimizer.step
        + eval 相恢复。"""
        self._require_rig().disc_phase.step()

    def export_discriminator(self) -> dict:
        """本卡判别器的可装载 state_dict（checkpoint 直写的取数面——
        卡 0 副本直写；确定性 allreduce + u/v broadcast 后全卡逐位
        一致，取卡 0 即全局口径）。"""
        return NetworkAssembler.loadable_state_dict(
            self._require_rig().scorer.discriminator,
        )

    def _load(
        self, manifest: LatentManifest, entries: Sequence[PoolEntry],
    ) -> torch.Tensor:
        """按条目序列加载 latent 批上本卡（逐条目懒加载 + stack + 设备
        迁移，零随机性——切片加载与整批对应切片逐位一致，#198 锚）。"""
        return torch.stack([
            manifest.load_latent(entry) for entry in entries
        ]).to(self.device)

    def _require_rig(self) -> PretrainCardRig:
        if self._rig is None:
            raise RuntimeError(
                f"卡 {self.index} 的组件未装配（start 前提交任务）"
            )
        return self._rig


class MeasurementSources:
    """预训练测量批与 real 侧的数据来源面（主控单点）：held-out
    全量排列（``seed+3`` 单点直派）、real 侧全局无放回抽取（``seed+1``
    单点直派、全池直读）、逐条件形状与卷数查询、装配期守卫。

    两条流都是注册表外的**主控单实例**（预训练不参与续训——流不进
    ``TrainingRngStreams`` 注册表；排列流消耗序 = 每次测量一次
    randperm，与现行 rank0 的 ``HeldOutAuc.condition_order`` 逐位恒等）。
    """

    def __init__(
        self,
        seed: int,
        heldout: LatentManifest,
        real_pool: LatentManifest,
    ) -> None:
        self._heldout = heldout
        self._order_sampler = RealPoolSampler(
            heldout, torch.Generator().manual_seed(seed + 3),
        )
        self._real_sampler = RealPoolSampler(
            real_pool, torch.Generator().manual_seed(seed + 1),
        )

    def assert_conditions_ready(
        self, targets: Sequence[str], cards: int,
    ) -> None:
        """轮转条件集守卫：每条件 held-out ≥ 卡数（#221 决议 8——现行
        world-1/distributed 两态守卫统一换名，单卡即 ≥ 1 的既有非空
        守卫）：测量批按卷切片到各卡，每卡至少 1 卷——缺卡的条件在
        装配期显式拒绝，而非首步测量时才炸。"""
        starved = [
            target for target in targets
            if self.volume_count(target) < cards
        ]
        if starved:
            raise ValueError(
                f"held-out real 不足以支撑 {cards}-路测量批切片"
                f"（每条件每卡至少 1 卷，不足条件: {starved}）——"
                "切片后某卡测量段为空，全局 AUC 缺该卡分数即不完整"
            )

    def draw_order(self, modality: str) -> tuple[PoolEntry, ...]:
        """该条件 held-out 全量卷的索引排列（测量批来源两步分解的第一
        步，#198 缝）：主控单点一次 randperm——复测换批 = 本流推进，
        新排列语义与数值保持。"""
        return self._order_sampler.permutation(modality=modality)

    def draw_reals(self, modality: str, count: int) -> tuple[PoolEntry, ...]:
        """该条件 real 侧的全局无放回抽取（#221 决议 12：主控对全池
        抽 K×卡数 → ``ShardPlan`` 切片到卡——跨卡无重复的条带互斥语义
        保持，抽取空间条带→全池为 accepted drift）。"""
        return self._real_sampler.permutation(modality=modality)[:count]

    def shape_of(self, modality: str) -> tuple[int, ...]:
        """该条件单卷 latent 形状（``LatentManifest.shape_of`` 的两态
        解析——``assert_condition_shapes`` 已保证多条件域表覆盖全条件）。"""
        return self._heldout.shape_of(modality)

    def volume_count(self, modality: str) -> int:
        """该条件 held-out 卷数（装配守卫与报告留痕的查询面）。"""
        return self._heldout.modalities.get(modality, 0)


class PretrainDriver:
    """判别器 warm-start 预训练编排（单进程多卡主控）：装配（每卡副本
    + 绑卡线程）→ 密集步进（主控单点测量/gate inline/扇出-join）→
    产物落盘（判别器 checkpoint + 预训练报告，主控唯一写者）。"""

    def __init__(
        self,
        config: CynosureConfig,
        run: PretrainRun,
        sources: MeasurementSources,
        support: SupportRule,
        schedules: ConditionSchedules,
        vocabulary: ConditionVocabulary,
        cards: list[PretrainCardWorker],
    ) -> None:
        self._config = config
        self._run = run
        self._sources = sources
        self._support = support
        self._schedules = schedules
        self._cards = cards
        self._template: MeasurementTemplate | None = None
        # 过拟合分叉监控（ADR-0009-γ）每卡实例：本地 train acc + 全局
        # AUC 的合成分叉观测、per-condition EMA 账轴 = 词汇表条件集
        self._overfit: list[OverfitMonitor] = [
            OverfitMonitor(config.reward, conditions=vocabulary.names())
            for _ in cards
        ]

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        run: PretrainRun,
        *,
        devices: list[torch.device] | None = None,
    ) -> "PretrainDriver":
        """config 驱动装配：设备发现（卡数）→ 数据面守卫装载（real 池
        容量 ≥ K×卡数、held-out 逐条件形状契约、行宽 ≡ 0 (mod 16)、
        每条件 held-out ≥ 卡数）→ 判别器冷启动原型（``seed+6``，逐位
        复制到各卡）→ 每卡绑卡线程。"""
        devices = (
            list(devices) if devices is not None else cls.execution_devices()
        )
        if not devices:
            raise ValueError("设备集不得为空（缺省 = 本进程可见计算设备）")
        vocabulary = TrainingRuntime.assemble_vocabulary(config)
        # real 侧装载与守卫（共用装配缝）：全池直读、主控全局抽取——
        # 需量倍数 = 卡数（每步每卡 K 对，最大需求上界 = K×卡数，
        # #221 决议 8「容量守卫 ≥ K×卡数零改动」）
        real_pool = TrainingRuntime.assemble_real_pool(
            config, vocabulary, world_size=len(devices),
        )
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        heldout.assert_condition_shapes(vocabulary)
        # 顺序流等价性的装配期不变式（#221 决议 6）：行宽非 16 倍数 =
        # 词表/manifest 工件异常，fail-fast
        heldout.assert_measurement_row_width()
        sources = MeasurementSources(config.schedule.seed, heldout, real_pool)
        targets = vocabulary.names()
        sources.assert_conditions_ready(targets, len(devices))
        # 冷启动原型（seed+6 fork_rng，跨卡 deepcopy 逐位一致）；判别器
        # optimizer 每卡独立（DiscriminatorPhase 编排其 step）
        scorer_prototype = TrainingRuntime.assemble_scorer(config, None)
        total_pairs = config.reward.disc_batch_size_k * len(devices)
        amp_dtype = AMP_DTYPES[config.policy.amp_dtype]
        cards = [
            PretrainCardWorker(
                index,
                device,
                cls._card_rig_factory(
                    config, index, device, scorer_prototype,
                    real_pool, heldout, total_pairs, amp_dtype,
                ),
            )
            for index, device in enumerate(devices)
        ]
        support = SupportRule(
            threshold=config.reward.pretrain_pass_threshold,
            support_bound=config.reward.gate_support_min_volumes,
            # bootstrap 的随机性独立派生（seed+7——命名流注册表之外，
            # 预训练不参与续训、判定可复现性由 seed 纯函数保证）
            generator=torch.Generator().manual_seed(
                config.schedule.seed + 7,
            ),
        )
        return cls(
            config, run, sources, support,
            TrainingRuntime.assemble_schedules(config), vocabulary, cards,
        )

    @staticmethod
    def execution_devices() -> list[torch.device]:
        """本进程的计算设备发现（卡轴）：CUDA 栈可用 = 逐卡设备清单
        （``CUDA_VISIBLE_DEVICES`` 天然承担卡集裁剪）；否则 CPU 单卡
        （fixture 口径——单设备上分片/扇出/重放全可测）。"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [
                torch.device("cuda", index)
                for index in range(torch.cuda.device_count())
            ]
        return [torch.device("cpu")]

    @staticmethod
    def _card_rig_factory(
        config: CynosureConfig,
        index: int,
        device: torch.device,
        scorer_prototype: LatentScorer,
        real_pool: LatentManifest,
        heldout: LatentManifest,
        total_pairs: int,
        amp_dtype,
    ) -> Callable[[], PretrainCardRig]:
        """本卡组件装配的闭包工厂（绑卡线程执行）：冻结基座 policy +
        判别器副本（原型 deepcopy 迁卡——单点装载、逐位复制）+ 卡轴
        recon 流的装配原语（``seed + 卡×10⁶ + 9``——卡 0 恒等 = 现行
        rank0 recon 数值；real_sampler=None 供给语义，real 由主控全局
        抽取经任务注入）+ 判别器步相位编排。"""
        def build_rig() -> PretrainCardRig:
            amp = AmpContext(device=device, dtype=amp_dtype)
            # 条件分布主流在新执行序无消费（测量条件穿模板流、更新批
            # 条件穿卡轴 recon 流）——注入独立一次性 generator 仅满足
            # 装配签名（async 门面同款口径）
            policy = GroupPolicy.build(
                config,
                torch.Generator().manual_seed(config.schedule.seed),
                device,
            )
            scorer = copy.deepcopy(scorer_prototype).to(device)
            # per-卡注册表实例：仅 recon 流有消费者（rollout/real_pool/
            # heldout 三流是预训练消费清单里的恒零消费死条目，#221
            # 决议 13——实例在场、消耗为零）
            streams = TrainingRngStreams(
                config.schedule.seed + index * SLOT_SEED_STRIDE,
            )
            assembler = TrainingRuntime.assemble_pair_assembler(
                config,
                real_sampler=None,
                sampler=TrainingRuntime.assemble_sampler(
                    config, policy.field, device=device,
                ),
                conditions=policy.conditions,
                generator=streams.recon,
                amp=amp,
            )
            disc_phase = DiscriminatorPhase(
                scorer,
                OnlineUpdate.assemble_optimizer(scorer, config.reward),
                total_pairs,
            )
            return PretrainCardRig(
                policy=policy,
                scorer=scorer,
                assembler=assembler,
                disc_phase=disc_phase,
                amp=amp,
                real_pool=real_pool,
                heldout=heldout,
            )

        return build_rig

    def run(self) -> PretrainReport:
        """密集步进至全部轮转条件过线（ADR-0008 决策 3 的 per-condition
        终止语义）或步数上限，产出判别器 checkpoint 与预训练报告（主控
        唯一写者）。每步条件 = 轮转条件集的 ``targets[step % n]``，以
        该条件全量 held-out 卷的冻结基座同源重构作测量批（按卷切片到
        各卡、主控卡序拼接全局分数，见 ``_measurement``）。"""
        for card in self._cards:
            if card.device.type == "cuda":
                # CUDA 上下文主控单线程逐卡预热（多线程首触并行 lazy
                # init 的竞争面消除，async 门面同款口径）
                torch.zeros(1, device=card.device)
                torch.cuda.synchronize(card.device)
        try:
            for card in self._cards:
                card.start()
            targets = self._cards[0].conditions.targets()
            # 测量模板（seed+19 主控直锚）在此装配：条件构造消费卡 0 的
            # 条件分布（穿模板流——抽取消耗传入 generator，不触碰实例
            # 自有流；构造必须等绑卡线程装配完成）
            self._template = MeasurementTemplate(
                self._config.schedule.seed,
                self._schedules,
                self._cards[0].conditions,
                sorted(self._config.policy.train_step_indices_m),
            )
            return self._step_loop(targets)
        finally:
            for card in self._cards:
                card.stop()

    def _step_loop(self, targets: Sequence[str]) -> PretrainReport:
        """步进循环体（四态 inline 主控）：更新 / 复测 / 确认 / 终止
        的密集步进 + 耗尽补测 + 产物落盘。"""
        reward = self._config.reward
        confirmed: dict[str, float] = {}
        volumes: dict[str, int] = {}
        steps_completed = 0
        all_conditions_passed = False
        for step in range(reward.pretrain_max_steps):
            started = time.monotonic()
            modality = targets[step % len(targets)]
            clusters, forwards = self._measurement(modality)
            volumes[modality] = clusters.volume_count
            # 首测判定（主控 inline 四态：更新/复测/确认/终止，即下方
            # if/continue/break 控制流——单进程无分发面）：更新前快照
            # （本步判别器权重）的 recon-AUC
            auc = clusters.pooled_auc()
            if modality not in confirmed and self._support.passes(
                auc, clusters,
            ):
                # 复测（同条件换批测量）：判别器权重同刻，变化的随机面
                # 只有一处——排列流单点推进 → 新排列；测量模板同起手
                # 复位（ε 逐位同输出现行语义不变）。重构 ε 把每卷配到
                # 的 (σ, ε) 槽位换掉，≥2 卷条件下两次读数是不同样本；
                # 单卷条件排列平凡、复测与首测同读数（确认退化——小池
                # 由数据侧池规模与支撑度界兜底，不以本相为抗噪防线）。
                # 复测只读 AUC——配对批不留存（与首测批同一释放口径）
                first_auc = auc
                confirm_clusters, _ = self._measurement(modality)
                confirm_auc = confirm_clusters.pooled_auc()
                if self._support.passes(confirm_auc, confirm_clusters):
                    confirmed[modality] = min(first_auc, confirm_auc)
                    if len(confirmed) == len(targets):
                        all_conditions_passed = True  # 末个条件：终止
                        break
                    continue  # 本条件已确认：本步不更新（无更新即无事件）
                # 复测掉线：本步事件/告警的 AUC 记账仍为首测值（``auc``
                # 未被改写）——复测读数只作确认判定，不入账
            # 测量批到此消费完毕（AUC 已归因、卷数已留痕 volumes）——
            # 测量段张量在各卡瞬态驻留、join 后即释放，不进更新步
            # （#174 OOM 修复：全量 held-out 测量批在大尺寸条件下数十
            # GiB 驻留的峰值面，在分段扇出形态下结构性消失）
            per_card_reports = self._update(modality)
            # 过拟合分叉观测（ADR-0009-γ）：每卡实例、本地 train acc +
            # 全局 AUC（卡轴诊断保持——per-卡离散本身是诊断信号）；
            # 越线告警主控按卡序追加（确认步不更新不观测）
            alerts = []
            for card_index, monitor in enumerate(self._overfit):
                train_acc = per_card_reports[
                    card_index
                ].conditions[0].train_pairwise_acc
                reading = monitor.observe(
                    modality,
                    train_pairwise_acc=train_acc,
                    heldout_auc=auc,
                )
                if reading.alerted:
                    # ``phase="pretrain"`` 是回退记账的 EXEMPT 分轨轴
                    # （预训练执行史全量保留；``iteration`` 记本步步号）；
                    # ``rank`` 归因观测卡（分叉按卡独立计算落盘的归因轴，
                    # ADR-0009 决策 4 的卡轴迁移）
                    alerts.append(OverfitAlertEvent(
                        iteration=step,
                        phase="pretrain",
                        rank=card_index,
                        modality=modality,
                        divergence_ema=reading.divergence,
                        train_pairwise_acc=train_acc,
                        heldout_auc=auc,
                    ))
            global_loss = sum(
                report.loss_discriminator for report in per_card_reports
            )
            self._run.append_event(PretrainEvent(
                step=step,
                modality=modality,
                loss_discriminator=global_loss,
                heldout_auc=auc,
                # 重构成本读数（#171 AC5 的成本口径落点）：测量批重构的
                # 前向次数（全量 σ 列表推算——分段求和与全量求和的加法
                # 结合恒等）与测量批规模——「没有量产」在事件流上可核对
                reconstruction_forwards=forwards,
                measurement_volumes=volumes[modality],
                lr=reward.disc_lr,
                elapsed_s=time.monotonic() - started,
            ))
            for alert in alerts:
                # 卡序追加（步内 pretrain 事件先于告警的写出序口径不变）
                self._run.append_event(alert)
            steps_completed += 1  # 更新步计数（确认步占步号但不更新不事件）
        reported = dict(confirmed)
        if not all_conditions_passed:
            # 步数上限耗尽：未确认条件逐个对落盘权重补测（循环内最后
            # 一次测得值属于更新前的上一份权重，与 checkpoint 不同快照；
            # 已确认条件的报告值 = 确认时的两次较小者，保留不覆盖）
            for target in targets:
                if target not in reported:
                    clusters, _ = self._measurement(target)
                    reported[target] = clusters.pooled_auc()
                    volumes[target] = clusters.volume_count
        return self._finalize(
            steps_completed, reported, list(confirmed),
            all_conditions_passed, volumes,
        )

    def _measurement(
        self, modality: str,
    ) -> tuple[VolumeScoreClusters, int]:
        """单条件测量批 → 卷级分数聚类（过线测量/复测/补测共用入口）。

        主控单点：排列抽取（seed+3 流一次 randperm）→ ``ShardPlan``
        卡轴连续段切片 → 测量模板复位抽取（seed+19 直锚：条件构造
        一次 → 全量 σ 定序轮转列表 → 按卡序逐段 ε）→ 测量扇出（每卡
        一段：real 切片加载 + 冻结基座同源重构 + 本卡打分）→ join →
        卡序 plain 拼接聚类（卷级归属保留、连续段拼接还原全量排列序）。
        返回 (全局聚类, 全局前向数)。随机流：held-out 全量卷的抽取消耗
        主控排列流（卷内顺序不影响读数——AUC 是集合级秩统计）、重构
        的 ε 走复位测量模板（不碰 recon 流）——预训练不参与续训，两处
        消耗都由 seed 纯函数确定 ⇒ 同 seed 同卡数重跑逐位可复算。
        """
        assert self._template is not None
        order = self._sources.draw_order(modality)
        plan = ShardPlan.split(len(order), len(self._cards))
        draw = self._template.draw(
            modality, self._sources.shape_of(modality), plan,
        )
        futures = [
            card.submit(
                card.measure_segment,
                order[start:stop],
                draw.sigmas_of(card_index, plan),
                draw.noises[card_index],
                draw.condition,
            )
            for card_index, (card, (start, stop)) in enumerate(
                zip(self._cards, plan.bounds),
            )
        ]
        per_card = [future.result() for future in futures]
        global_clusters = VolumeScoreClusters(
            real_volume_scores=tuple(
                volume
                for segment in per_card
                for volume in segment.real_volume_scores
            ),
            fake_scores=torch.cat(
                [segment.fake_scores for segment in per_card],
            ),
        )
        return global_clusters, self._template.forward_count(
            modality, draw.sigmas,
        )

    def _update(self, modality: str) -> list[UpdateReport]:
        """更新步：主控全局无放回抽 K×卡数 → ``ShardPlan`` 切片（K×D
        切 D 段 = base K rem 0，每卡恰 K）→ 更新扇出（每卡 K 对装配 +
        本地前向反向）→ join → 跨卡梯度 SUM allreduce（M1 形态，与
        在线判别器步同 communicator 语义）→ 各卡 AdamW → 步末 u/v
        broadcast（卡 0 权威）。返回逐卡报告（事件的全局 loss =
        逐对等权加权值的跨卡求和 ≡ 全局逐对 patch mean——每对 patch
        数恒等，决议 3；观测面取各卡本地 train acc）。"""
        card_count = len(self._cards)
        total = self._config.reward.disc_batch_size_k * card_count
        entries = self._sources.draw_reals(modality, total)
        plan = ShardPlan.split(total, card_count)
        futures = [
            card.submit(
                card.update_segment, entries[start:stop], modality,
            )
            for card, (start, stop) in zip(self._cards, plan.bounds)
        ]
        reports = [future.result() for future in futures]
        DiscriminatorPhase.reduce_gradients(
            [card.disc_phase for card in self._cards]
        )
        for card in self._cards:
            card.submit(card.disc_step).result()
        DiscriminatorPhase.synchronize_spectral(
            [card.disc_phase for card in self._cards]
        )
        return reports

    def _finalize(
        self,
        steps_completed: int,
        condition_auc: dict[str, float],
        conditions_passed: list[str],
        all_conditions_passed: bool,
        condition_volumes: dict[str, int],
    ) -> PretrainReport:
        """产物落盘（主控唯一写者）：判别器 checkpoint（卡 0 副本直写，
        可装载 state_dict 与训练期产物同构）+ 预训练报告（字段集与
        provenance 指纹面不变——决议 17/18）。过线条件为空同样落盘
        ——报告与 checkpoint 是失败预训练的诊断产物，不丢。"""
        discriminator_state = self._cards[0].submit(
            self._cards[0].export_discriminator,
        ).result()
        torch.save(discriminator_state, self._run.paths.discriminator_ckpt)
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
            conditions_passed=conditions_passed,
            # 判据口径标识（ADR-0012 决策 5 的审计面）：本报告的 AUC 是
            # held-out real 原始 vs 冻结基座同源重构体的 recon-AUC——
            # 与在线 iter 事件的 rollout-AUC 不可横向比较
            auc_criterion="recon_auc",
            condition_volumes=condition_volumes,
            steps_completed=steps_completed,
            pass_threshold=reward.pretrain_pass_threshold,
            all_conditions_passed=all_conditions_passed,
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
