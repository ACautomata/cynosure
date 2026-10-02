"""async 执行模型的单进程门面（骨架期 fixture 全宽薄切片，#231）。

执行形态（#217 结票门面 + #229 spike **go** 裁决的实证形态）：

- **主线程 = asyncio 门面**：任务枚举（静态分配表）/ barrier 状态机 /
  收集-同步单点 / 事件发射，不承载任何 torch 计算闭包（research P1：
  事件循环线程零阻塞 torch 调用）。
- **每卡 = 一条专用执行线程**（静态绑卡，current device 线程局部语义，
  research P2/P6）：线程宿主自己的事件循环，槽（协程）经
  ``run_coroutine_threadsafe`` 多路复用到绑卡线程。
- **每卡 = 完整本地副本**（#217 §4：FSDP/DDP 退役面）——policy 与判别器
  scorer 每卡一份逐位相同的副本；rollout 相零集合通信全并发；更新相
  逐 k barrier：各卡并发 fwd+bwd 本卡例子的第 k 任务（**loss 侧
  ×1/N**、梯度本卡累积）→ 主线程单点串行 allreduce（M1 形态：逐
  tensor、SUM、固定全卡组，#215 research；启动预热一次）→ 各卡一次
  optimizer.step；k 间严格串行。
- **RNG**：per-槽流注册表（``SlotRngRegistry``，#218 全口径——线性步长
  派生、四流全顺序推进、在线零复位、get_stream 出口 owner-thread
  断言）；槽装配在绑卡线程执行，流 owner 即绑卡线程。
- **异常 fail-fast 门面**（#217 §4）：任一槽线程异常（经 concurrent
  Future 回传）→ 训练中止异常（``TrainingAborted``）+ 线程收尾；NCCL
  in-flight collective 无法真取消，如实以「停止派发 + loop 停转 +
  join（限时）」表达，卡死线程由进程退出兜底（daemon 线程）。
  #231 字面「abort communicator + 进程级退出」的落位记档：torch 无
  communicator abort API 面，abort 的等效语义 = 本卡线程收尾 + 跨卡
  互等由 barrier 硬超时（``BarrierTimeoutPolicy``）有界化；进程级
  退出由入口层承载（``TrainingAborted`` 抛给调用方，骨架期无 CLI
  装配、生产入口票面明确不含，#226 决策 1）。
- **barrier 软/硬超时**（#217 §4）：软超时 → ``barrier_timeout_alert``
  告警事件（#222 两档 flush 告警族即写）后继续等待；硬超时 → fail-fast。
  ``CYNOSURE_PG_TIMEOUT_MIN`` 变量沿用、**语义换绑** per-k barrier 硬
  超时（分钟；sugon 已设 40，零部署变更；未设置 = torch 默认 10 分钟
  口径沿用）。
- **事件发射**：iter 族 (iteration, slot) 排序写——``IterEvent.rank``
  字段名不动、语义重定义为调度槽号（#217 事件契约「可扩不可改名」）；
  单进程单写者，slot 升序即 (iteration, slot) 归并序。

骨架期包含面（#226 决策 3）：静态分配表轮（轮内置换）+ per-槽协程骨架
+ 逐 k barrier 收集-同步 + 事件发射 + RNG 注册表 per-槽实例化 + 异常
fail-fast + barrier 软/硬超时；两域（MR + BraTS）贯穿。

rollout 期增量（#232，加厚 1/6）：**同源重构任务进 rollout 相**——
per-槽 ``ReconstructionAssembler``（重构流 = 槽 recon 流，#218 五处
消费面注入面零改动），消耗节奏钉 N_d（判别器窗口的逐 iter 摊派形态随
#234 定型）；配对批 fake = rollout 相当前权重 θ_t（drift #1 时点前移的
机制落地——旧执行序在 update_policy 之后的 θ_{t+1}）。打分归位消费点
= ``RolloutPhase._to_pool_domain`` 单点除 scale factor——域换算缝按
#226 接缝备忘保持开放（ADR-0015 DomainLatent 载体另行择期，实施时不
动现有域换算点形态）。

policy 更新期增量（#233，加厚 2/6）：**梯度检查点解耦**（#217 §4 最重
配套项随首个有梯度前向的真实消费点落地）——应用缝移 ``GroupPolicy.build``
（FSDP wrap 之前，``GradientCheckpointing``，`config.policy.
gradient_checkpointing` 默认开、fixture 可关），装配期 bitwise 探针
（plain/wrapped 双腿真实梯度前向逐位比较）fail-fast；更新相逐 k 收集-
同步（loss×(1/N) + 逐 tensor SUM allreduce，骨架期已落的机器面）的
验收锚升 multi-k 日程（重放锚 num_steps=5、M={1,2,3}，单卡与多卡档）。

判别器链期增量（#234，加厚 3/6，#220 结票全口径）：**判别器步在 k 循
环后落地**——混合条件配对批经判别器桶（``DiscriminatorBucket``，单
条件同形对集合、构造期断言）承载，相位序列先全桶 eval 后全桶 train
（``DiscriminatorPhase.accumulate``）→ 跨卡梯度 allreduce SUM →
optimizer.step → 步末 u/v broadcast（卡 0 权威）；**逐对等权全局
mean**（loss×n_b/N_total）显式新裁落地。窗口语义（``Discriminator
Window``）：每窗口每卡恰 K 任务、逐 iter 发射 floor(K/L)（余数补窗
口首 iter）、任务创建序 = 桶序×对序；real 侧窗口起点全局无放回抽取
（``WindowRealDraw``：splitmix64 一次性派生 generator、不进 RNG 注册
表），fake 产出即 CPU 暂存、步前 join 后按桶序回迁；**AUC per-condition
池化**（``PooledHeldOutAuc``：本卡 real 采样打分按条件名排序、分数
卡号升序收集、单点 float64 midrank、每活跃条件恰一次读数）进 iter
事件 heldout_auc 与分叉原料——per-rank 轴退役：分叉 = EMA(逐条件
train acc − 池化 AUC)、每条件一条、窗口内最后一次有效测量配窗口末
train acc（#220 决议 13/14）；UpdateReport/IterEvent 升格 per-condition
明细（可扩不可改名）。ReplicatedDiscriminator 的 RL 构造点随本票
解耦退役（本体留存至 pretrain driver 期删除，#226 用户故事 8）：
RL 装配缝（``TrainingRuntime.assemble_rewards``）不再 DDP 化判别器，
预训练 driver 自行装配副本语义。

续训与事件契约期增量（#236，加厚 4/6，#222/#218 结票全口径）：
**v12 单文件续训分片**——``AsyncResumeStore``（独立 nominal-v12
常量，旧 v11 分片被新 store 拒、新 v12 分片被旧 resume 拒 = 版本
对账承载跨执行器拒绝）、marker/集合化拒绝整体退役（单进程
fail-fast）、拓扑守卫只对账 slots（卡数不进对账）、generators
per-(槽×流) 嵌套、seeds 记录性字段、全局 RNG/world_size/ema 预留槽
删除；checkpoint 节奏（周期 + 收尾兜底）**同写者顺序定死**——跨卡
权重 bitwise 校验（#217，失配 = weight_divergence_alert 即写 +
abort）→ 续训分片 → 产物 checkpoint（policy_iter*.pt /
discriminator_iter*.pt）；resume 单点声明（占位装配 + 恢复 = 契约
校验 → 下发各卡 → 流状态回填 → 分叉 adopt → 指标流回退）。
**事件两档 flush**——iter 族缓冲到 iteration 边界、按 (iteration,
slot) 排序后单点写（``append_events``）；告警族（barrier_timeout_
alert / weight_divergence_alert / overfit_alert）产生即写。软超时
事件随本票按 #222 收口定名（骨架期 barrier_soft_timeout 临时类型
退役：barrier_timeout_alert、字段 iteration/k/elapsed_s）。

评测顺迁期增量（#237，加厚 5/6，#217 评测三路径结票口径）：**评测
三路径（Baseline 采样 / 里程碑解码评测 / RL 后重采）进本门面**——
采样前向走新执行器（``SlotDispatchLatentSampler``：manifest 条目按
槽分派到绑卡线程、per-entry noise_seed 独立 generator 按槽分派逐位
安全）；**汇聚缝 = 异形 latent 逐条传卡 0 + 按 entry index 字典保序
重组**（异形不可 cat）；解码 / FID / 特征提取不动（``ManifestEvaluation``
三路径编排经 ``EntryLatentSampler`` 缝注入新实现后零改动，KID
bootstrap 独立 generator 的主线程单点执行随之保留）；里程碑评测顺迁
——decode / fid 相位面保持既有口径（``MilestoneEvent.phase_seconds``
契约不收缩），早停判定喂入前缀化过滤（``EarlyStopJudge.prefix_events``，
#222 既裁口径；单进程单写者无 rank 广播面）。

**不含**（各进加厚期，#226）：pretrain driver、
生产入口（本门面仅被 fixture 测试驱动，#226 决策 1 生产入口单口径）。
"""

import asyncio
import copy
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch

from cynosure.conditions import ConditionVocabulary
from cynosure.config import CynosureConfig
from cynosure.distributed.process import (
    DistributedContext,
    PG_TIMEOUT_MINUTES_ENV,
)
from cynosure.eval import EvaluationPhase, ManifestEvaluation
from cynosure.grpo import ClippedPolicyLoss, MgaiAdvantage, StepwisePolicyUpdate
from cynosure.policy.sampler import RolloutSampler
from cynosure.pretrain.artifacts import PretrainReport
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.assembly import ReconstructionAssembler
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.overfit import DivergenceReading, OverfitMonitor
from cynosure.reward.scorer import ChunkedScorer, RewardScorer
from cynosure.reward.update import ConditionUpdateDetail, OnlineUpdate, UpdateReport
from cynosure.train.allocation import AllocationTable
from cynosure.train.artifacts import (
    POLICY_CHECKPOINT_TEMPLATE,
    BarrierTimeoutAlertEvent,
    BaselineManifest,
    DiscConditionReading,
    DiscUpdateDetail,
    IterEvent,
    MilestoneEvent,
    OverfitAlertEvent,
    RunArtifacts,
    WeightDivergenceAlertEvent,
)
from cynosure.train.async_eval import SlotDispatchLatentSampler
from cynosure.train.async_resume import (
    ASYNC_RESUME_FORMAT_VERSION,
    AsyncResumeStore,
)
from cynosure.train.discriminator import (
    DiscriminatorBucket,
    DiscriminatorPhase,
    DiscriminatorWindow,
    DiscriminatorWindowRun,
    PooledHeldOutAuc,
    WindowPairRecord,
    WindowRealDraw,
    WindowTask,
)
from cynosure.train.earlystop import EarlyStopJudge
from cynosure.train.policy import GroupPolicy
from cynosure.train.rollout import IterationRollout, RolloutPhase, StepRollout
from cynosure.train.rng import (
    DropoutGuard,
    SlotRngRegistry,
    TrainingRngStreams,
)
from cynosure.train.runtime import TrainingRuntime

_T = TypeVar("_T")

_STAGE_TAG = 1
"""骨架期事件的阶段号（单阶段组缺省；组3 StageTag 机制属 trainer 面，
本门面 fixture 薄切片不承载序贯）。"""

_STREAM_NAMES: tuple[str, ...] = (
    TrainingRngStreams.ROLLOUT,
    TrainingRngStreams.REAL_POOL,
    TrainingRngStreams.HELDOUT_AUC,
    TrainingRngStreams.RECON,
)
"""v12 分片 generators 键的流名轴：从 TrainingRngStreams 流常量派生
（async_resume 校验面同源派生——嵌套校验/状态导出/状态回填共用，
无字面量副本；tuple 保序、那边 frozenset 比集合）。"""

_WORKER_JOIN_TIMEOUT_S = 10.0
"""线程收尾的 join 上限（秒）：正常停转远快于此；卡死在 in-flight CUDA
op 的线程 join 超时后放行（daemon 线程由进程退出兜底——NCCL in-flight
collective 无法真取消，#217 门面「如实写」口径）。"""

_WORKER_READY_TIMEOUT_S = 120.0
"""绑卡线程就绪（设备绑定 + 槽装配完成）的等待上限（秒）。"""


class TrainingAborted(Exception):
    """fail-fast 门面的训练中止异常：槽线程异常或 barrier 硬超时后由
    门面抛出（原异常挂 ``__cause__``；进程级退出由入口层承载——门面
    仅被 fixture 测试驱动，骨架期无 CLI 装配，#226 决策 1）。"""


@dataclass
class SlotExample:
    """一个例子的门面记账载体：rollout 相产出 + 逐 k 相位累计（事件
    组装与逐 k 任务的共用面；``update_seconds`` 由本槽协程在更新相
    逐 k 累加——单槽单协程写者，无跨线程竞争）。"""

    slot: int
    record: IterationRollout
    rollout_seconds: float
    steps: dict[int, StepRollout] = field(default_factory=dict)
    """按被优化训练步 k 索引的 rollout 记录（更新相任务的取数面）。"""
    update_seconds: float = 0.0
    fake_scores: torch.Tensor | None = None
    """本例 new_fakes 的判别器展平分数（CPU，#234 池化 AUC 的原料：
    rollout 相内打分——打分在 fakes 所在卡，判别器副本同值；每活跃条
    件的池化 midrank 由门面在相末单点合成）。"""
    window_pairs: tuple[tuple[WindowTask, WindowPairRecord], ...] = ()
    """本例承载的窗口重构任务产出（(任务, 单对记录) 序，#234）：
    判别器步 iteration 所在窗口内各 iter 摊派的任务——fake = 本
    iteration rollout 相当前权重 θ_t（drift #1 时点前移保持），产出
    即 CPU 暂存；判别器步前 join（rollout 相 gather 即 join 形态），
    门面按任务累计进窗口记录、步前按桶序回迁装配。"""


class CardReplica:
    """每卡完整副本（#217 §4）：本卡的 policy 装配 + 判别器 scorer 副本
    + 采样封装 + 逐 k 更新编排 + 判别器步相位编排（#234）——副本间无
    共享可变张量状态，跨卡一致性由「同初始化 + 确定性 allreduce + 同
    步 step 序列」结构性保证。"""

    def __init__(
        self,
        index: int,
        device: torch.device,
        policy: GroupPolicy,
        scorer: RewardScorer,
        sampler: RolloutSampler,
        updater: StepwisePolicyUpdate,
        disc_phase: DiscriminatorPhase,
    ) -> None:
        self.index = index
        self.device = device
        self.policy = policy
        self.scorer = scorer
        self.sampler = sampler
        self.updater = updater
        self.disc_phase = disc_phase

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        index: int,
        device: torch.device,
        scorer_prototype: RewardScorer,
        total_pairs: int,
    ) -> "CardReplica":
        """本卡副本装配：policy 网络构建（checkpoint 装载，单进程无
        FSDP 包装——分片随执行模型退役）+ 判别器副本（原型 deepcopy 后
        迁卡——单点装载、逐位复制）+ 采样封装（前向激活预算本地解析，
        ``chunk_sync`` 恒 None：无 FSDP 即无「前向调用次数绑定集合序列」，
        #165 挂死类结构性消失）+ 逐 k 更新编排 + 判别器步相位编排
        （``DiscriminatorPhase``，判别器链期 #234——判别器更新步的
        train/eval 相位切换自此组件自持，打分/AUC 前向的 eval 钉相由
        步编排的 finally 语义保证）。"""
        amp = TrainingRuntime.amp_context(config, device)
        # GroupPolicy 的条件分布主流在新执行序无消费（目标条件来自分配
        # 表、组2 端内自由度显式传槽流，见 SlotRunner/RolloutPhase）——
        # 此处注入独立一次性 generator 仅满足装配签名
        policy = GroupPolicy.build(
            config,
            torch.Generator().manual_seed(config.schedule.seed),
            device,
        )
        sampler = TrainingRuntime.assemble_sampler(
            config, policy.field, device=device,
        )
        updater = StepwisePolicyUpdate(
            sampler=sampler,
            optimizer=policy.optimizer,
            loss=ClippedPolicyLoss(clip_range=config.policy.ratio_clip),
            device_type=amp.device_type,
            amp_dtype=amp.dtype,
        )
        scorer = copy.deepcopy(scorer_prototype).to(device)
        scorer.eval()
        DropoutGuard.assert_clean(scorer, origin="判别器 scorer 副本")
        disc_phase = DiscriminatorPhase(
            scorer,
            OnlineUpdate.assemble_optimizer(scorer, config.reward),
            total_pairs,
        )
        return cls(index, device, policy, scorer, sampler, updater, disc_phase)

    def gradient_tensors(self) -> list[torch.Tensor | None]:
        """可训练网络的逐参梯度出口（逐 k 归约的取数面——归约器不摸
        副本内部结构）；无梯度参数以 None 占位（跨副本结构性一致，
        归约器恒跳过）。"""
        return [
            parameter.grad for parameter in self.policy.network.parameters()
        ]


class SlotRunner:
    """per-槽协程骨架：一个例子的 rollout 相协程与逐 k 更新任务协程的
    载体（#217 调度单元契约——每协程承载一个「例子」；协程经门面多路
    复用到绑卡线程，torch 调用全部落在绑卡线程）。

    rollout 相本体的编排序（#217 §3 + #234 判别器链期）：rollout
    （anchor → 扰动 → λ 续跑 → 打分）→ 池化 AUC 原料打分（本例
    fakes 的判别器分数）→ 窗口重构任务（fake 构造即 CPU 暂存）——
    三段同属 rollout 相，``rollout_seconds`` 计时窗口整体涵盖
    （phase_seconds 无独立条目：AUC 池化与判别器步的计时归位见门面
    _run_iteration）。
    """

    def __init__(
        self,
        slot: int,
        updater: StepwisePolicyUpdate,
        rollout: RolloutPhase,
        scorer: RewardScorer,
        assembler: ReconstructionAssembler,
        real_pool: LatentManifest,
        device: torch.device,
        advantage_clamp: float,
    ) -> None:
        self.slot = slot
        self._updater = updater
        self._rollout = rollout
        self._chunked = ChunkedScorer(scorer)
        self._assembler = assembler
        self._real_pool = real_pool
        self._device = device
        self._advantage = MgaiAdvantage(clamp=advantage_clamp)

    async def run_example(
        self,
        condition_name: str,
        *,
        window_tasks: tuple[tuple[WindowTask, PoolEntry], ...] = (),
        score_fakes: bool = True,
    ) -> SlotExample:
        """一个例子的 rollout 相（分配表条件 → 初始噪声/扰动/续跑 →
        池化原料打分（门控）→ 窗口重构任务）。

        ``score_fakes`` = 本例 fakes 的池化原料打分门控：判别器步由
        门面经 ``run_examples`` 显式供给 True（#220 决议 12「每活跃
        条件恰一次读数」的 fake 原料，即产即用）；非判别器步 False——
        打分前不进静默丢弃面白付。缺省 True（直调/单测路径完整产出）。

        ``window_tasks`` = 本 (iteration, 槽) 摊派的重构任务及其
        real 池条目（窗口起点全局无放回抽取的切片，#220 决议 10）：
        逐任务加载 real（CPU）→ 供给入口 fake 构造（recon 流消耗
        「先 s 后 ε」、批维 1）→ 产出即 CPU 暂存（WindowPairRecord）。
        无任务（非窗口 iter 或本槽未摊派）为空 tuple。"""
        started = time.monotonic()
        record = self._rollout.run_iteration(condition_name)
        fake_scores = (
            self._score_fakes(record.new_fakes) if score_fakes else None
        )
        pairs = tuple(
            (task, self._reconstruct(task, entry))
            for task, entry in window_tasks
        )
        return SlotExample(
            slot=self.slot,
            record=record,
            rollout_seconds=time.monotonic() - started,
            steps={step.step_index: step for step in record.steps},
            fake_scores=fake_scores,
            window_pairs=pairs,
        )

    def _score_fakes(self, new_fakes: torch.Tensor) -> torch.Tensor:
        """本例 fakes 的判别器展平分数（no_grad + 定块打分，池化 AUC
        原料；CPU 形态跨线程传递——打分在 fakes 所在卡，判别器副本
        同值、步末 u/v 同步保证，#220 决议 12）。"""
        with torch.no_grad():
            return self._chunked.scores(new_fakes).cpu()

    def _reconstruct(
        self, task: WindowTask, entry: PoolEntry,
    ) -> WindowPairRecord:
        """单窗口任务：real 加载（CPU 池懒加载）→ 供给入口 fake 构造
        （recon 流、先 s 后 ε、no_grad + autocast 与 rollout 同数值
        口径）→ 单对 CPU 暂存（两侧同形断言在 WindowPairRecord）。"""
        real_cpu = self._real_pool.load_latent(entry).unsqueeze(0)
        pair = self._assembler.reconstruct_assigned(
            real_cpu.to(self._device), task.condition,
        )
        return WindowPairRecord(
            condition=task.condition,
            real=real_cpu,
            fake=pair.fakes.cpu(),
        )

    async def run_k(
        self, example: SlotExample, step_index: int, example_count: int,
    ) -> float:
        """本例第 k 任务的逐 k 累积（loss×(1/N)、梯度本卡累积；返回
        未缩放 reported loss，相位秒数累计进本例记账）。"""
        started = time.monotonic()
        step = example.steps[step_index]
        advantages = self._advantage.compute(step.rewards)
        reported = self._updater.accumulate(
            step_index,
            step.anchor_latent,
            example.record.condition,
            step.directions,
            step.old_log_probs,
            advantages,
            example_count,
        )
        example.update_seconds += time.monotonic() - started
        return reported


class CardWorker:
    """每卡专用执行线程（research P2/P6 静态绑卡）：线程宿主本卡事件
    循环，槽协程经 ``submit`` 多路复用；线程启动即完成设备绑定与槽装配
    （流 owner-thread 断言的记录点 = 绑卡线程，#218 出口断言口径）。

    判别器链期（#234）追加的卡级消费面：``pooled_auc``（池化 AUC 本卡
    测量段，绑卡线程装配——heldout 流 owner 即绑卡线程）、
    ``discriminate``（窗口记录 → 桶装配 → 全桶 eval/train/加权
    backward）、``disc_step``（optimizer.step + eval 归位）——跨卡
    集合段（梯度 allreduce、u/v broadcast）由门面主线程单点编排
    （``DiscriminatorPhase.reduce_gradients`` / ``synchronize_spectral``）。"""

    def __init__(
        self,
        replica: CardReplica,
        slot_ids: list[int],
        build_components: Callable[
            [], "tuple[list[SlotRunner], PooledHeldOutAuc]",
        ],
    ) -> None:
        self.replica = replica
        self.slot_ids = slot_ids
        self.slots: list[SlotRunner] = []
        self.pooled_auc: PooledHeldOutAuc | None = None
        self._build_components = build_components
        self._failure: BaseException | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name=f"cynosure-card-{replica.index}",
        )

    @property
    def index(self) -> int:
        """卡号（副本装配面的委托出口——静态绑卡身份的唯一来源）。"""
        return self.replica.index

    @property
    def device(self) -> torch.device:
        """本卡设备（副本装配面的委托出口）。"""
        return self.replica.device

    def _serve(self) -> None:
        """线程本体：静态绑卡（current device 线程局部）→ 槽装配 + 卡
        级组件装配（本线程取流，owner 记录即绑卡线程）→ 事件循环常驻。"""
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            self.slots, self.pooled_auc = self._build_components()
        except BaseException as error:  # 线程边界必须接住：装配失败经
            self._failure = error      # _ready 唤醒主线程显式失败
        finally:
            self._ready.set()
        if self._failure is not None:
            return
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self) -> None:
        """启动线程并等待就绪（装配失败在此显式抛出，不留半装配线程）。"""
        self._thread.start()
        if not self._ready.wait(_WORKER_READY_TIMEOUT_S):
            raise TimeoutError(
                f"卡 {self.index} 的绑卡线程未在 {_WORKER_READY_TIMEOUT_S}s "
                "内就绪（设备绑定/槽装配卡死）"
            )
        if self._failure is not None:
            raise TrainingAborted(
                f"卡 {self.index} 的槽装配失败"
            ) from self._failure

    def submit(self, coroutine: Coroutine[Any, Any, Any]) -> Future:
        """协程派发到本卡线程（concurrent Future 回传：异常自动传播，
        research T10 的通道形态）。"""
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def run_examples(
        self,
        conditions: dict[int, str],
        *,
        window_tasks: dict[int, tuple[tuple[WindowTask, PoolEntry], ...]],
        score_fakes: bool = True,
    ) -> dict[int, SlotExample]:
        """本卡全部例子的 rollout 相（槽间并发 gather——协程多路复用
        形态；torch 调用在本线程内串行落卡）。``window_tasks`` = 每槽
        本 iteration 摊派的窗口任务（real 条目切片随任务下发）；
        ``score_fakes`` = 池化原料打分门控（透传 ``SlotRunner.run_example``，
        门面按判别器步显式供给）。"""
        examples = await asyncio.gather(*(
            slot.run_example(
                conditions[slot.slot],
                window_tasks=window_tasks.get(slot.slot, ()),
                score_fakes=score_fakes,
            )
            for slot in self.slots
        ))
        return {example.slot: example for example in examples}

    async def score_heldout(
        self, conditions: tuple[str, ...], counts: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        """池化 AUC 本卡测量段：活跃条件（条件名排序）的 held-out real
        采样 + 打分（消耗本卡 heldout 流；#220 决议 12/15）。
        ``counts`` = 本卡本条件 fake 数（逐卡对称采样平面，见
        ``PooledHeldOutAuc.score_conditions``）。"""
        assert self.pooled_auc is not None
        return self.pooled_auc.score_conditions(conditions, counts)

    async def discriminate(
        self,
        window_records: list[tuple[WindowTask, WindowPairRecord]],
        expected: int,
    ) -> UpdateReport:
        """判别器步的卡内段：窗口记录按桶装配（桶序恒条件名排序、桶内
        对序 = 任务创建序）→ ``DiscriminatorPhase.accumulate``（先全桶
        eval 后全桶 train + 逐对等权加权 backward）。

        ``expected`` = 窗口计划的本卡任务数——join 后逐位对账，缺失 /
        多余即拒绝（#220 决议 6 的「不静默降批」机器面：任务在 rollout
        相内同步完成，异常早已 fail-fast 中止，正常路径恒对账相等）。"""
        if len(window_records) != expected:
            raise ValueError(
                f"判别器步窗口记录数 {len(window_records)} ≠ 计划任务数 "
                f"{expected}——任务缺失/多余不静默降批（#220 决议 6；"
                "fail-fast 门面之外的双重对账）"
            )
        buckets: list[DiscriminatorBucket] = []
        by_condition: dict[str, list[tuple[WindowTask, WindowPairRecord]]] = {}
        for item in window_records:
            by_condition.setdefault(item[1].condition, []).append(item)
        for condition in sorted(by_condition):
            members = sorted(
                by_condition[condition],
                key=lambda member: (member[0].iteration, member[0].index),
            )
            buckets.append(DiscriminatorBucket.assemble(
                condition,
                [record for _, record in members],
                self.device,
            ))
        return self.replica.disc_phase.accumulate(buckets)

    async def disc_step(self) -> None:
        """判别器步收尾：optimizer.step + eval 相恢复（门面跨卡梯度
        allreduce 之后；步末 u/v broadcast 由门面主线程单点编排）。"""
        self.replica.disc_phase.step()

    async def export_states(self) -> tuple[dict, dict]:
        """本卡权重快照（checkpoint 校验/续训存取的取数面）：policy
        网络 + 判别器网络的 CPU clone state_dict——主线程的跨卡逐位
        比较与 ``torch.save`` 序列化不受卡上存储别名影响；checkpoint
        点各卡 idle（iteration 已完成），读取无并发写者。"""
        replica = self.replica
        return (
            AsyncResumeStore.to_cpu_snapshot(
                replica.policy.network.state_dict(),
            ),
            AsyncResumeStore.to_cpu_snapshot(
                replica.scorer.discriminator.state_dict(),
            ),
        )

    async def load_restored(self, payload: dict) -> None:
        """恢复应用（本卡）：policy 权重 + optimizer、判别器权重 +
        optimizer 整体覆写——装配期随机性被整体覆写的既有语义
        （resume 占位装配不消费任何 checkpoint 工件，覆写即落盘时刻
        的训练机状态）。分片张量 CPU 形态装载：模块 ``load_state_dict``
        跨设备 copy_、optimizer ``load_state_dict`` cast 到参数设备，
        均不依赖分片保存时的设备环境。"""
        replica = self.replica
        replica.policy.load_full_state(payload["policy_network"])
        replica.policy.optimizer.load_state_dict(payload["policy_optimizer"])
        replica.scorer.discriminator.load_state_dict(
            payload["discriminator_network"], strict=True,
        )
        replica.disc_phase.optimizer.load_state_dict(
            payload["discriminator_optimizer"],
        )

    async def accumulate_k(
        self,
        examples: dict[int, SlotExample],
        step_index: int,
        example_count: int,
    ) -> list[float]:
        """k 相位的本卡段：零梯度 → 本卡各例第 k 任务并发累积（梯度
        落本卡副本 .grad；1/N 缩放在任务内，#217 SUM 化口径）。"""
        self.replica.updater.optimizer.zero_grad()
        return list(await asyncio.gather(*(
            slot.run_k(examples[slot.slot], step_index, example_count)
            for slot in self.slots
        )))

    async def step_optimizer(self) -> None:
        """k 相位的收尾：本卡一次 optimizer.step（barrier 的 reduce 段
        之后——默认流序保证 allreduce 先于 step 消费梯度，#215 M1）。"""
        self.replica.updater.optimizer.step()

    def stop(self) -> None:
        """线程收尾：loop 停转 + 限时 join（卡死线程放行给进程退出，
        见模块 docstring fail-fast 口径）；loop 仅在线程确已退出时关闭。"""
        if self._thread.is_alive():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(_WORKER_JOIN_TIMEOUT_S)
        if not self._thread.is_alive():
            self._loop.close()

    def stopped_within(self, grace_seconds: float) -> bool:
        """线程收尾的宽限观测面（fail-fast/超时中止路径的消费口）：宽限
        等待后线程确已退出则 True——退绕（槽内串行任务 + 运行时间歇
        停顿）超出宽限仍卡死则 False，由进程退出兜底。"""
        self._thread.join(grace_seconds)
        return not self._thread.is_alive()


class PerKCollectReduce:
    """逐 k 收集-同步（glossary「逐 k 收集-同步」）：主线程单点串行
    allreduce——M1 形态（``torch.cuda.nccl.all_reduce`` 逐 tensor、SUM、
    固定全卡组；单次 Python 调用 = 一个原子 NCCL group，单线程驱动多卡
    不死锁，#215 research）。单卡（CPU fixture / 单卡拓扑）无跨卡归约
    面，结构退化为本卡累积即全局平均（loss×(1/N) 已在任务内）。"""

    def warmup(self, replicas: list[CardReplica]) -> None:
        """communicator 预热一次（首个 allreduce 吃 comm 初始化——sugon
        实测 ~4.3s，移出正式迭代；缓存按设备列表键控，归约组固定全卡、
        全程不变，#215 风险清单 2）。"""
        tensors = self._across_cards(replicas, lambda device: torch.zeros(
            1, device=device,
        ))
        if tensors is not None:
            torch.cuda.nccl.all_reduce(
                tensors, op=torch.cuda.nccl.SUM,
            )

    def reduce(self, replicas: list[CardReplica]) -> None:
        """单 k 的跨卡梯度归约：逐 tensor 串行发射（~30 调用/barrier 的
        实证形态；分桶 ~25MB 属性能缝，骨架期取最简逐 tensor）。梯度经
        副本出口（``gradient_tensors``）取数，归约器不触碰副本内部
        结构。"""
        if not self._validate_multicard(replicas):
            return
        for grads in zip(*(
            replica.gradient_tensors() for replica in replicas
        )):
            if grads[0] is None:
                # 无梯度参数跨副本结构性一致（同一前向图）；全零参与只会
                # 白耗带宽——恒跳过
                continue
            torch.cuda.nccl.all_reduce(
                list(grads), op=torch.cuda.nccl.SUM,
            )

    @staticmethod
    def _validate_multicard(replicas: list[CardReplica]) -> bool:
        """多卡归约面的判定单点：单副本 = False（无跨卡归约面，结构性
        退化为本卡累积即全局平均）；多卡副本设备须全为 CUDA（M1 前提
        ——NCCL 只收 CUDA/HIP 张量；CPU fixture 恒单卡）。"""
        if len(replicas) <= 1:
            return False
        devices = {replica.device for replica in replicas}
        if any(device.type != "cuda" for device in devices):
            raise ValueError(
                f"多副本归约要求全部副本落 CUDA 设备，得到 {sorted(map(str, devices))}"
            )
        return True

    @staticmethod
    def _across_cards(
        replicas: list[CardReplica], make: Callable[[torch.device], _T],
    ) -> list[_T] | None:
        """多卡逐副本产值（预热占位张量等）：``_validate_multicard``
        判定通过后逐副本调 ``make``；单副本返回 None。"""
        if not PerKCollectReduce._validate_multicard(replicas):
            return None
        return [make(replica.device) for replica in replicas]


class BarrierTimeoutPolicy:
    """per-k barrier 软/硬超时口径（#217 §4；``CYNOSURE_PG_TIMEOUT_MIN``
    语义换绑）：硬超时 = fail-fast 中止；软超时 = 硬阈值 ×
    ``SOFT_FRACTION`` 处发 ``barrier_timeout_alert`` 告警事件后继续等待。
    生产读环境变量（分钟）；测试按构造注入小值快速触发。"""

    SOFT_FRACTION = 0.5
    """软阈值占硬阈值比例（门面常量；部署侧整体调 ``CYNOSURE_PG_TIMEOUT_MIN``
    即同时缩放两档）。"""

    DEFAULT_HARD_MINUTES = 10
    """环境变量未设置时的硬超时缺省（分钟）——沿用 torch 进程组 watchdog
    的默认 10 分钟口径，换绑不改缺省。"""

    ENV = PG_TIMEOUT_MINUTES_ENV
    """沿用变量的语义换绑声明：旧语义 = 进程组 watchdog 超时（分钟），
    新语义 = per-k barrier 硬超时（分钟）——sugon 已设 40，零部署变更
    （#217 §4 门面形态）。字面单点在 ``distributed.process``
    （``PG_TIMEOUT_MINUTES_ENV``），两语义共享变量名、各按自己的执行序
    解读。"""

    def __init__(
        self, hard_seconds: float, *, soft_fraction: float | None = None,
    ) -> None:
        if hard_seconds <= 0:
            raise ValueError(
                f"barrier 硬超时须为正秒数，得到 {hard_seconds}"
            )
        self.hard_seconds = float(hard_seconds)
        self.soft_seconds = float(hard_seconds) * (
            self.SOFT_FRACTION if soft_fraction is None else soft_fraction
        )
        if not 0 < self.soft_seconds <= self.hard_seconds:
            raise ValueError(
                f"软阈值须落在 (0, 硬阈值] 内，得到 {self.soft_seconds}"
                f"（硬阈值 {self.hard_seconds}s × 比率 {soft_fraction}）"
            )

    @classmethod
    def from_env(cls) -> "BarrierTimeoutPolicy":
        """生产口径：环境变量（分钟）→ 硬超时秒数；未设置 = 缺省 10
        分钟；非正整数显式拒绝（读取/校验单点 =
        ``DistributedContext.parse_timeout_minutes``）。"""
        minutes = DistributedContext.parse_timeout_minutes()
        if minutes is None:
            minutes = cls.DEFAULT_HARD_MINUTES
        return cls(minutes * 60)


class RunContinuation:
    """续训与 checkpoint 节奏的编排协作者（#236 续训与事件契约期）：

    - 持有 v12 ``AsyncResumeStore``（分片存取的校验单点）；
    - 持有槽 RNG 注册表（快照/恢复的流状态源——``generators`` 段与
      ``seeds`` 段的取数面、恢复的流回填目标，#237 评审归并：流状态
      的持有者随续训协作者走）；
    - **resume 单点声明**（装配与 run 执行共用同一开关，旧 trainer
      口径平移——无双点声明可错位）：build(resume=True) 驱动占位装配
      （判别器不消费任何 checkpoint 工件），run() 经本类声明从分片
      恢复；
    - checkpoint 周期判定（收尾兜底由门面按 last_checkpoint 前向
      推进口径裁决，旧 trainer 同口径）。

    分片的具体攒装（收集各卡权威状态）与恢复应用（下发各卡）由门面
    承担——它持有 cards；本类不触碰门面内部结构（「ResumeStore 挂
    运行时聚合层」裁决在新执行序的落点：#231 骨架期后新执行序的聚合
    层 = 本门面，store 与其节奏编排收为门面单一协作者，#222 §1 /
    ADR-0014 平移口径）。
    """

    def __init__(
        self,
        artifacts: RunArtifacts,
        rng: SlotRngRegistry,
        *,
        resume: bool,
    ) -> None:
        self._store = AsyncResumeStore(artifacts)
        self._rng = rng
        self._resume = resume

    @property
    def rng(self) -> SlotRngRegistry:
        """槽 RNG 注册表（流状态快照/回填的取数面）。"""
        return self._rng

    @property
    def resume_assembled(self) -> bool:
        """resume 单点声明的读面（装配分派与 run 恢复共用）。"""
        return self._resume

    @property
    def store(self) -> AsyncResumeStore:
        """v12 续训分片存取（门面 save/restore 的经手面）。"""
        return self._store

    def checkpoint_due(self, completed: int, interval: int) -> bool:
        """周期判定：completed % interval == 0（completed > 0 由调用
        循环保证——iteration 从 0 起、checkpoint 点在完成后）。"""
        return completed % interval == 0


class EvaluationRounds:
    """评测三路径的回合编排（#237 评测顺迁期）：Baseline 采样 / 里程碑
    解码评测回合 / RL 后重采的相位与簿记单点——评测采样前各卡 eval 相
    位（三条评测路径与 rollout 同为 eval 相，旧 trainer 评测前显式
    ``eval_phase`` 口径平移）、里程碑回合的解码评测 → ``milestone``
    事件写入训练指标流 → 早停判定。主循环只按节奏点调用，评测编排不
    进训练主循环体（#237 评审：门面属性数收拢 + 编排收出主循环）。"""

    def __init__(
        self,
        evaluation: EvaluationPhase,
        artifacts: RunArtifacts,
        cards: "list[CardWorker]",
        judge: EarlyStopJudge,
    ) -> None:
        self.evaluation = evaluation
        self._artifacts = artifacts
        self._cards = cards
        self._judge = judge

    def baseline(self) -> None:
        """训练启动期的 Baseline 采样（eval 相位 → 冻结初始 policy
        冻结只采一次）。"""
        self._set_policy_eval()
        self.evaluation.sample_baseline()

    def resample(self) -> None:
        """RL 后的同 manifest 重采（eval 相位 → 最终 policy）。"""
        self._set_policy_eval()
        self.evaluation.resample()

    def milestone(self, iteration: int) -> bool:
        """里程碑解码评测回合 → ``milestone`` 事件入训练指标流 → 早停
        判定。返回是否早停。

        评测采样前向走新执行器（``SlotDispatchLatentSampler`` 条目按槽
        分派），解码 / FID / 特征提取在主线程单点执行不动——decode/fid
        相位打点随 ``MilestoneMetrics.phase_seconds`` 透传，事件契约
        （``MilestoneEvent.phase_seconds`` 面）不收缩。早停判定喂入前缀
        化过滤（#222：只消费 iteration ≤ 当前里程碑的本 stage 事件，判定
        纯函数化）；单进程单写者，无旧执行序的 rank 0 判定 + 广播面。
        ``elapsed_s`` 是本方法侧的全区间口径（覆盖采样 + 解码 + FID +
        簿记），与旧执行序同口径。"""
        self._set_policy_eval()
        started = time.monotonic()
        metrics = self.evaluation.milestone_metrics()
        stage_events = EarlyStopJudge.prefix_events(
            self._artifacts.read_events(), iteration, _STAGE_TAG,
        )
        verdict = self._judge.judge(stage_events, current_fid=metrics.fid)
        criteria = dict(metrics.summary())
        criteria["plateau_stalled"] = float(verdict.plateau_stalled)
        criteria["hacking_signature"] = float(verdict.hacking_signature)
        self._artifacts.append_event(MilestoneEvent(
            iteration=iteration,
            stage=_STAGE_TAG,
            fid=metrics.fid,
            kid=metrics.kid,
            ssim=metrics.ssim,
            mae=metrics.mae,
            psnr=metrics.psnr,
            criteria_summary=criteria,
            early_stop=verdict.stop,
            early_stop_reason=verdict.reason,
            elapsed_s=time.monotonic() - started,
            phase_seconds=metrics.phase_seconds,
        ))
        return verdict.stop

    def _set_policy_eval(self) -> None:
        """评测采样前的 eval 相位（各卡；主线程直接调用纯 Python 相位
        状态，先例 = 逐 iteration 的 eval_phase/train_phase 编排）。"""
        for card in self._cards:
            card.replica.policy.eval_phase()


class AsyncTrainingExecutor:
    """async 执行序门面：静态分配表轮 + per-槽协程骨架 + 逐 k barrier
    收集-同步 + 事件发射 + fail-fast/超时口径的单点编排（#231 骨架期
    全宽薄切片；两域 MR + BraTS 贯穿，条件轴经条件分布 ``targets()``）。"""

    def __init__(
        self,
        config: CynosureConfig,
        artifacts: RunArtifacts,
        allocation: AllocationTable,
        cards: list[CardWorker],
        collect_reduce: PerKCollectReduce,
        timeout: BarrierTimeoutPolicy,
        window_run: DiscriminatorWindowRun,
        overfit: OverfitMonitor,
        continuation: RunContinuation,
        evaluation_rounds: EvaluationRounds,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.allocation = allocation
        self.cards = cards
        self._collect_reduce = collect_reduce
        self._timeout = timeout
        self._window_run = window_run
        self._overfit = overfit
        self._continuation = continuation
        self.evaluation_rounds = evaluation_rounds

    @property
    def rng(self) -> SlotRngRegistry:
        """槽 RNG 注册表（续训协作者持有；测试锚与恢复面的取数别名）。"""
        return self._continuation.rng

    @property
    def window(self) -> DiscriminatorWindow:
        """判别器窗口计划纯函数（事件面/测试锚的取数口）。"""
        return self._window_run.window

    @property
    def window_run(self) -> DiscriminatorWindowRun:
        """判别器窗口运行期账簿（测试锚的观测面：窗口内最后一次池化
        读数的对拍取数口）。"""
        return self._window_run

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        coroutines: int | None = None,
        devices: list[torch.device] | None = None,
        owner_check: bool = False,
        timeout: BarrierTimeoutPolicy | None = None,
        resume: bool = False,
        evaluation: EvaluationPhase | None = None,
    ) -> "AsyncTrainingExecutor":
        """config 驱动装配：设备发现（卡数）→ 条件轴（条件分布
        ``targets()``，集合知识归条件分布自身）→ 分配表 + 槽注册表 →
        判别器窗口计划 + real 抽取 + 分叉监控（#234）→ 每卡副本与
        绑卡线程（槽静态 round-robin 绑卡）。``coroutines`` 缺省 = 卡数
        （每卡一例的默认拓扑）；``devices`` 显式设备集（CPU fixture 档
        钉单 CPU 设备——生产口径缺省 = 本进程可见全卡，卡集裁剪经
        ``CUDA_VISIBLE_DEVICES``）；``owner_check`` 透传注册表（测试档
        开启，#218 debug-only 口径）。"""
        if DistributedContext.env_rank() is not None:
            raise ValueError(
                "async 门面是单进程执行序：检测到 torchrun 注入的 RANK "
                "环境——进程内多卡由设备发现承担、不经 torchrun（多进程"
                "入口属旧执行序）"
            )
        devices = (
            list(devices) if devices is not None else cls.execution_devices()
        )
        if not devices:
            raise ValueError("设备集不得为空（缺省 = 本进程可见计算设备）")
        slot_count = (
            coroutines if coroutines is not None
            else config.execution.coroutines if config.execution.coroutines is not None
            else len(devices)
        )
        if slot_count < 1:
            raise ValueError(f"调度槽数须 ≥ 1，得到 {slot_count}")
        # resume 装配分派（单点声明经 RunContinuation 贯穿装配与执行）：
        # 新 run = warm-start 装载（ADR-0007 守卫链）；resume = 占位装配
        # ——判别器不消费任何 checkpoint 工件（预训练产物被清理的中断
        # run 永不可恢复的问题在此结构性消失），随机初始化被 restore
        # 整体覆写（旧 trainer「装配期随机性被整体覆写」口径平移）。
        scorer_prototype = (
            TrainingRuntime.assemble_scorer(config, None, resume=True)
            if resume else cls.assemble_discriminator(config)
        )
        vocabulary = TrainingRuntime.assemble_vocabulary(config)
        total_pairs = config.reward.disc_batch_size_k * len(devices)
        replica = CardReplica.build(
            config, 0, devices[0], scorer_prototype, total_pairs,
        )
        conditions = replica.policy.conditions.targets()
        if config.experiment.group == "cross-modal":
            AllocationTable.assert_pair_symmetry(
                [tuple(pair) for pair in config.experiment.cross_modal_pairs],
                conditions,
            )
        allocation = AllocationTable(conditions, slot_count, config.schedule.seed)
        rng = SlotRngRegistry(
            config.schedule.seed, slot_count, owner_check=owner_check,
        )
        # 卡槽绑定（静态 round-robin 的纯函数镜像）：判别器窗口计划的
        # 任务→槽分派与门面的槽绑卡同源。空卡（槽数 < 卡数）显式拒绝
        # ——「每窗口每卡恰 K 对」的窗口语义前提每卡至少一槽，半绑定
        # 拓扑不静默可用（卡内槽装配与池化测量段都以非空槽为前提）
        card_slots = {
            index: tuple(
                slot for slot in range(slot_count)
                if slot % len(devices) == index
            )
            for index in range(len(devices))
        }
        if any(not slots for slots in card_slots.values()):
            raise ValueError(
                f"卡槽绑定存在空卡 {sorted(k for k, v in card_slots.items() if not v)}"
                f"（槽数 {slot_count} < 卡数 {len(devices)}）——每窗口每卡"
                "恰 K 对的窗口语义前提每卡至少一槽：减卡数或加协程数"
            )
        window = DiscriminatorWindow(
            allocation,
            n_d=config.reward.disc_update_interval_n_d,
            batch_size_k=config.reward.disc_batch_size_k,
            card_slots=card_slots,
        )
        # real 侧装载与守卫（两执行序共用装配缝）：全池直读、窗口起点
        # 全局无放回抽取——需量倍数传卡数（窗口单条件最大需求上界 =
        # K×卡数，#220 决议 9/10；装配期容量守卫把门不变）
        real_pool = TrainingRuntime.assemble_real_pool(
            config, vocabulary, world_size=len(devices),
        )
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        # 逐条件形状契约的装配期对照（#129 消费侧守卫沿袭——词表工件与
        # held-out manifest 的同名异形在首例测量时才炸属失败后移；
        # real 侧的对照在 assemble_real_pool 内）
        heldout.assert_condition_shapes(vocabulary)
        replicas = [replica] + [
            CardReplica.build(
                config, index, device, scorer_prototype, total_pairs,
            )
            for index, device in enumerate(devices[1:], 1)
        ]
        overfit = OverfitMonitor(config.reward, conditions=vocabulary.names())
        cards = []
        # 卡 0 副本引用先行捕获（评测缺省采样核的取数面）——for 循环
        # 泄漏的循环变量在循环后指向末卡（评审：注入移除时缺省 sampler
        # 会静默落在末卡设备而 amp 落卡 0，设备口径错位）
        card0_replica = replicas[0]
        for replica in replicas:
            bound_slots = list(card_slots[replica.index])
            cards.append(CardWorker(
                replica=replica,
                slot_ids=bound_slots,
                build_components=cls._card_components(
                    config, replica, rng, vocabulary, real_pool, heldout,
                    bound_slots,
                ),
            ))
        # 评测相装配（#237 评测顺迁）：采样前向走新执行器（条目按槽分派
        # + 异形 latent 逐条传卡 0 汇聚），解码 / FID / 特征提取与
        # MilestoneEvent.phase_seconds 打点在 ManifestEvaluation 三路径
        # 编排内零改动；测试可注入替身（EvaluationPhase 同契约）。
        # vocabulary / real pool / AmpContext 由上方装配单点装载、注入
        # 两个评测消费面（评审：不双读 pool manifest、不重复装配词表
        # 与数值口径；real pool 复用 assemble_real_pool 的守卫装载）。
        # 注入替身路径短路全部评测面装载（评审：死装载收进分支，与
        # ManifestEvaluation 缺省分支同口径）。
        if evaluation is not None:
            assembled_evaluation = evaluation
        else:
            amps = [
                TrainingRuntime.amp_context(config, device)
                for device in devices
            ]
            manifest = BaselineManifest.load(run_artifacts.paths.manifest)
            assembled_evaluation = ManifestEvaluation.build(
                config,
                run_artifacts,
                card0_replica.sampler,
                _STAGE_TAG,
                manifest,
                amp=amps[0],
                write_enabled=True,
                vocabulary=vocabulary,
                pool=real_pool,
                latent_sampler=SlotDispatchLatentSampler.assemble(
                    cards, slot_count, vocabulary, real_pool, amps,
                ),
            )
        return cls(
            config=config,
            artifacts=run_artifacts,
            allocation=allocation,
            cards=cards,
            collect_reduce=PerKCollectReduce(),
            timeout=(
                timeout if timeout is not None
                else BarrierTimeoutPolicy.from_env()
            ),
            window_run=DiscriminatorWindowRun(
                window, WindowRealDraw(real_pool), config.schedule.seed,
            ),
            overfit=overfit,
            continuation=RunContinuation(
                run_artifacts, rng, resume=resume,
            ),
            evaluation_rounds=EvaluationRounds(
                assembled_evaluation, run_artifacts, cards,
                EarlyStopJudge(config),
            ),
        )

    @staticmethod
    def _card_components(
        config: CynosureConfig,
        replica: CardReplica,
        rng: SlotRngRegistry,
        vocabulary: ConditionVocabulary,
        real_pool: LatentManifest,
        heldout: LatentManifest,
        slot_ids: list[int],
    ) -> Callable[[], "tuple[list[SlotRunner], PooledHeldOutAuc]"]:
        """本卡组件装配的闭包工厂（绑卡线程执行；流 owner-thread 断言的
        记录点 = 绑卡线程）：per-槽 RolloutPhase（槽 rollout 流注入，
        #218 五处消费面注入面零改动口径）+ ReconstructionAssembler
        （供给语义装配，#234：real_sampler=None——real 由窗口起点全局
        无放回抽取经任务注入，per-槽 real_pool 流随 #232 过渡形态退役、
        fake 侧随机性 = 槽 recon 流；assembler 构造只读流种子、不消耗
        流位置）+ 池化 AUC 测量段（heldout 流 = 卡首槽——per-卡测量取代
        per-例测量后的卡内单一消耗面）。"""

        def build_components() -> "tuple[list[SlotRunner], PooledHeldOutAuc]":
            amp = TrainingRuntime.amp_context(config, replica.device)
            runners = []
            for slot in slot_ids:
                rollout = RolloutPhase(
                    config,
                    replica.sampler,
                    replica.scorer,
                    rng.get_stream(slot, TrainingRngStreams.ROLLOUT),
                    condition_sampler=replica.policy.conditions,
                    vocabulary=vocabulary,
                    device_type=amp.device_type,
                    autocast_dtype=amp.dtype,
                    device=replica.device,
                )
                assembler = TrainingRuntime.assemble_pair_assembler(
                    config,
                    real_sampler=None,
                    sampler=replica.sampler,
                    conditions=replica.policy.conditions,
                    generator=rng.get_stream(slot, TrainingRngStreams.RECON),
                    amp=amp,
                )
                runners.append(SlotRunner(
                    slot,
                    replica.updater,
                    rollout,
                    replica.scorer,
                    assembler,
                    real_pool,
                    replica.device,
                    config.grpo.advantage_clamp,
                ))
            pooled = PooledHeldOutAuc(
                heldout_manifest=heldout,
                scorer=replica.scorer,
                generator=rng.get_stream(
                    slot_ids[0], TrainingRngStreams.HELDOUT_AUC,
                ),
                device=replica.device,
            )
            return runners, pooled

        return build_components

    @staticmethod
    def execution_devices() -> list[torch.device]:
        """本进程的计算设备发现（卡轴）：CUDA 栈可用 = 逐卡设备清单
        （``CUDA_VISIBLE_DEVICES`` 天然承担卡集裁剪）；否则 CPU 单卡
        （fixture 口径——单设备上调度/分配表/barrier/RNG 轴全可测）。"""
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [
                torch.device("cuda", index)
                for index in range(torch.cuda.device_count())
            ]
        return [torch.device("cpu")]

    @staticmethod
    def assemble_discriminator(config: CynosureConfig) -> RewardScorer:
        """判别器 scorer 的 warm-start 装载单点（新 run 语境，ADR-0007
        守卫链）：复用 ``TrainingRuntime.assemble_scorer`` 的权重来源
        分派——装配单一来源，不设第二份守卫链副本。"""
        return TrainingRuntime.assemble_scorer(
            config,
            PretrainReport.load(config.reward.pretrain_report_json),
        )

    def run(self) -> int:
        """训练主循环（#237 评测顺迁期口径）：预热 → 绑卡线程启动 →
        逐 iteration（rollout 相 → 逐 k barrier → 事件发射）+ Baseline
        采样 / 里程碑评测（早停判定）+ RL 后重采 + 线程收尾。返回完成
        的 iteration 数。续训（resume 单点声明经 ``RunContinuation``：
        恢复 + checkpoint 周期/收尾兜底 + v12 分片与产物落盘）、判别器
        链（#234）与评测三路径（#237：采样前向走新执行器）已进本门面。
        启动序在收尾兜底的 ``try`` 内：第 2..N 卡装配失败（``start`` 抛
        ``TrainingAborted``）时已启动的前序卡同样经 ``finally`` 收尾。"""
        self._warmup()
        try:
            for card in self.cards:
                card.start()
            return asyncio.run(self._run_iterations())
        finally:
            for card in self.cards:
                card.stop()

    def _warmup(self) -> None:
        """启动预热（主线程单点，research P5/T11 + #215 预热前提）：
        CUDA 上下文逐卡初始化（多线程首触并行 lazy init 的竞争面消除）
        + M1 communicator 一次 dummy 归约。"""
        for card in self.cards:
            if card.device.type == "cuda":
                torch.zeros(1, device=card.device)
                torch.cuda.synchronize(card.device)
        self._collect_reduce.warmup([card.replica for card in self.cards])

    async def _run_iterations(self) -> int:
        """训练主循环（#236 续训与事件契约期口径 + #237 评测三路径顺迁）：
        resume 装配时先恢复（契约校验 → 下发各卡 → 流回填 → 指标流回退）
        → Baseline 采样（非 resume；冻结只采一次）→ 逐 iteration → 周期
        /里程碑强制 checkpoint（分片先、产物后）→ 里程碑解码评测 + 早停
        判定（命中即停）→ RL 后重采（执行了训练才收官）。返回完成的
        iteration 数（config 口径累计完成数；早停时小于 max_iterations；
        恢复点已达标 = 无操作续训报告恢复点本身）。

        评测路径（baseline / 里程碑 / 重采）的相位与簿记收在评测回合
        编排协作者（``EvaluationRounds``），采样前向走新执行器（条目
        按槽分派到绑卡线程，评测采样零消耗训练 RNG 流——逐位重放锚不受
        milestone_interval 影响），解码 / FID / 特征提取与事件打点在主
        线程单点（单进程单写者，无旧执行序的 rank 0 闸门与广播面）。"""
        start_iteration = 0
        if self._continuation.resume_assembled:
            start_iteration = await self._restore()
        completed = start_iteration
        last_checkpoint = start_iteration
        interval = self.config.schedule.checkpoint_interval
        if not self._continuation.resume_assembled:
            self.evaluation_rounds.baseline()
        for iteration in range(start_iteration, self.config.schedule.max_iterations):
            await self._run_iteration(iteration)
            completed = iteration + 1
            milestone_due = (
                completed % self.config.schedule.milestone_interval == 0
            )
            if milestone_due or self._continuation.checkpoint_due(
                completed, interval,
            ):
                # checkpoint 周期之外，每个里程碑也强制落盘（config 契约：
                # milestone 评测器与恢复路径的取数点，周期不覆盖时仍须
                # 产出——旧 trainer 口径平移）
                await self._checkpoint_at(completed)
                last_checkpoint = completed
            if milestone_due:
                if self.evaluation_rounds.milestone(completed):
                    # 早停：最终 policy 状态已随上面的里程碑 checkpoint
                    # 落盘（评测不改权重，checkpoint 态即停时态）
                    break
        if last_checkpoint < completed:
            # 收尾兜底只允许前向推进：恢复点已在目标之后（收缩
            # max_iterations 的续训 = 无操作）时不得把更后的训练态
            # 改写成更小的 iteration 标签（旧 trainer 口径平移）
            await self._checkpoint_at(completed)
        if completed > start_iteration:
            # RL 后重采（同 manifest 条目，差异唯一归因于 RL）：恢复点
            # 已达标的无操作续训不重采（policy 未变，重采只因 RNG 流
            # 位置不同而静默改写产物——旧 trainer 口径平移）
            self.evaluation_rounds.resample()
        return completed

    async def _restore(self) -> int:
        """续训恢复（#222 恢复序直线）：分片输入契约校验（store：
        版本/键清单/slots 拓扑/generators 嵌套/config 漂移守卫）→
        权重与 optimizer 下发各卡（全卡覆写，非只卡 0——恢复的是
        整机的落盘时刻状态）→ lr 槽位实测对账 → 命名流状态回填
        （不重派生对账，#222 记录性字段口径）→ 分叉监控 adopt →
        指标流回退（删除恢复点之后的半截事件，重执行重写）。
        返回恢复点 iteration。"""
        payload = self._continuation.store.restore(
            self.config, slot_count=self._continuation.rng.slot_count,
        )
        iteration = payload["iteration"]
        await self._collect(
            [card.submit(card.load_restored(payload)) for card in self.cards],
            iteration,
            step_index=None,
            enforce_timeout=False,
        )
        self._assert_restored_lr(payload["lr"])
        saved_streams = payload["generators"]
        for slot in range(self._continuation.rng.slot_count):
            for stream in _STREAM_NAMES:
                self._continuation.rng.restore_stream(
                    slot, stream, saved_streams[f"slot{slot}"][stream],
                )
        self._overfit.adopt(payload["overfit"])
        self.artifacts.rewind_events(iteration, _STAGE_TAG)
        return iteration

    def _assert_restored_lr(self, lr_slots: dict) -> None:
        """LR 槽位对账（应用后实测）：常数 LR 实现的 scheduler 状态 =
        两 optimizer ``param_groups`` 的 lr（load_state_dict 已随
        param_groups 回归）——槽位与其实测值显式对账，不一致 = 分片
        损坏/篡改；全卡一致由结构保证，卡 0 实测（旧执行序 resume
        模块同口径平移）。"""
        card0 = self.cards[0].replica
        optimizers = {
            "policy": card0.policy.optimizer,
            "discriminator": card0.disc_phase.optimizer,
        }
        if set(lr_slots) != set(optimizers):
            raise ValueError(f"续训状态 lr 槽位字段不符: {sorted(lr_slots)}")
        for name, optimizer in optimizers.items():
            saved_lr = float(lr_slots[name])
            for group in optimizer.param_groups:
                if group["lr"] != saved_lr:
                    raise ValueError(
                        f"续训状态 lr 槽位与 optimizer state 不一致"
                        f"（{name}: {saved_lr} vs {group['lr']}）"
                    )

    async def _checkpoint_at(self, iteration: int) -> None:
        """checkpoint 节奏点（#222 同写者顺序定死）：跨卡权重逐位
        校验（#217 checkpoint 周期 bitwise 校验，多卡面）→ v12 续训
        分片（单文件原子写）→ 产物 checkpoint（契约工件，外部消费）。
        校验失败 = ``weight_divergence_alert`` 告警即写 + abort——分叉
        是吸收态（allreduce 只作用梯度、不能纠正权重分叉，#217 §4），
        不产出分叉产物。"""
        started = time.monotonic()
        per_card = await self._collect(
            [card.submit(card.export_states()) for card in self.cards],
            iteration,
            step_index=None,
            enforce_timeout=False,
        )
        divergence = self.detect_card_divergence(per_card)
        if divergence is not None:
            self.artifacts.append_event(WeightDivergenceAlertEvent(
                iteration=iteration,
                stage=_STAGE_TAG,
                elapsed_s=time.monotonic() - started,
            ))
            raise TrainingAborted(
                f"checkpoint 周期跨卡权重分叉（iteration {iteration}）："
                f"{divergence}——#217 bitwise 校验 fail-fast（分叉是吸收态，"
                "告警 + abort + 从最近一致 checkpoint 重启）"
            )
        self._continuation.store.save(
            self._capture_resume_payload(iteration, per_card[0]),
        )
        self._write_product_checkpoints(iteration, per_card[0])

    @staticmethod
    def detect_card_divergence(per_card: list[tuple[dict, dict]]) -> str | None:
        """跨卡权重逐位比较（policy + 判别器网络 state_dict 的每个
        张量，含 spectral norm ``_u``/``_v`` buffer）：全部卡与卡 0
        快照逐位相等 = None；失配 = 「网络:参数名（卡 i 与卡 0 逐位
        失配）」的定位描述（告警事件的归因文案与 abort 消息共用）。
        单卡拓扑无跨卡面，结构性恒一致（返回 None，无跳过日志——
        单卡的「一致」是结构事实不是测量结论）。"""
        if len(per_card) <= 1:
            return None
        reference_policy, reference_disc = per_card[0]
        for card_index, (policy_state, disc_state) in enumerate(
            per_card[1:], 1,
        ):
            for name, reference in reference_policy.items():
                if not torch.equal(reference, policy_state[name]):
                    return f"policy:{name}（卡 {card_index} 与卡 0 逐位失配）"
            for name, reference in reference_disc.items():
                if not torch.equal(reference, disc_state[name]):
                    return (
                        f"discriminator:{name}"
                        f"（卡 {card_index} 与卡 0 逐位失配）"
                    )
        return None

    def _capture_resume_payload(
        self, iteration: int, card0_states: tuple[dict, dict],
    ) -> dict:
        """v12 全清单快照的攒装（#222 终稿键清单）：卡 0 权威取数
        （各卡副本经「同初始化 + 确定性 allreduce + 同步 step」结构
        保证逐位一致，checkpoint 点的跨卡 bitwise 校验刚把守过）；
        optimizer 与 lr 卡 0 单份。``seeds`` = per-槽 seed 派生值的
        记录性落痕（恢复不对账，#222 §1）。"""
        policy_state, disc_state = card0_states
        card0 = self.cards[0].replica
        streams = _STREAM_NAMES
        return {
            "version": ASYNC_RESUME_FORMAT_VERSION,
            "iteration": int(iteration),
            "slots": self._continuation.rng.slot_count,
            "policy_network": policy_state,
            "policy_optimizer": AsyncResumeStore.to_cpu_snapshot(
                card0.policy.optimizer.state_dict(),
            ),
            "discriminator_network": disc_state,
            "discriminator_optimizer": AsyncResumeStore.to_cpu_snapshot(
                card0.disc_phase.optimizer.state_dict(),
            ),
            "lr": {
                "policy": card0.policy.optimizer.param_groups[0]["lr"],
                "discriminator": (
                    card0.disc_phase.optimizer.param_groups[0]["lr"]
                ),
            },
            "generators": {
                f"slot{slot}": {
                    stream: self._continuation.rng.stream_state(slot, stream)
                    for stream in streams
                }
                for slot in range(self._continuation.rng.slot_count)
            },
            "overfit": self._overfit.state(),
            "seeds": {
                "base": self.config.schedule.seed,
                "per_slot": [
                    SlotRngRegistry.slot_seed(
                        self.config.schedule.seed, slot,
                    )
                    for slot in range(self._continuation.rng.slot_count)
                ],
            },
        }

    def _write_product_checkpoints(
        self, iteration: int, card0_states: tuple[dict, dict],
    ) -> None:
        """产物 checkpoint（契约工件：policy_iter*.pt /
        discriminator_iter*.pt，外部消费/评测装载形态）——续训分片
        先行落盘、产物后写（#222 同写者顺序）：恢复以分片为准、产物
        只为外部消费；崩溃窗口内二者独立原子，最坏缺产物不缺分片。
        权重快照与分片共享同一份卡 0 导出（checkpoint 期 host 内存
        不双份，旧执行序「一次导出、两处共享」纪律平移）。判别器与
        分片**同形**：``loadable_state_dict`` 即 ``state_dict()`` 直
        通（``_u``/``_v`` 幂迭代 buffer 与参数化状态两处都在），此处
        直接写卡 0 已导出的 CPU 快照——不二次取数模块，CUDA 拓扑下
        产物也落 CPU 张量（评测装载面 ``map_location="cpu"`` 语义下
        与 policy 产物形态统一）。"""
        policy_state, disc_state = card0_states
        checkpoints = self.artifacts.paths.checkpoints
        torch.save(
            policy_state,
            checkpoints / POLICY_CHECKPOINT_TEMPLATE.format(
                iteration=iteration,
            ),
        )
        torch.save(
            disc_state,
            checkpoints / f"discriminator_iter{iteration}.pt",
        )

    async def _run_iteration(self, iteration: int) -> None:
        """单 iteration 执行序（#217 §3 相位结构 + #234 判别器链期）：

        窗口起点 real 抽取（``step_for_launch`` 判定）→ rollout 相
        （全并发零梯度耦合：rollout → 池化原料打分 → 窗口重构任务）→
        池化 AUC（本卡 real 采样打分 → 卡号升序收集 → 单点 midrank，
        每活跃条件恰一次读数）→ train 相逐 k barrier（loss×(1/N)+SUM
        → 各卡一次 step）→ 判别器步（is_step：桶装配 → 全桶 eval/train
        + 逐对等权 backward → 跨卡梯度 allreduce SUM → optimizer.step
        → 步末 u/v broadcast）→ 分叉观测与事件发射（(iteration, slot)
        排序写）。

        phase_seconds 发 rollout / policy_update / discriminator 三相
        （#217 §3 全集含 trajectory：现行仅 --dump 诊断打点消费、本
        门面无诊断路径——trajectory 相随诊断路径加厚）；AUC 池化原料
        打分计入 rollout 相（fake 打分在 fakes 所在卡、随例执行），
        判别器步计时（池化 real 打分 + 桶装配 + 相位两段 + 集合段）
        记入 discriminator 相。"""
        started = time.monotonic()
        # —— 判别器窗口：窗口首 iter 的起点抽取（任务 → real 条目）——
        self._window_run.launch(iteration)
        conditions = {
            slot: self.allocation.condition_for(iteration, slot)
            for slot in range(self.allocation.slot_count)
        }
        slot_tasks = {
            slot: self._window_run.tasks_for(iteration, slot)
            for slot in conditions
        }
        examples: dict[int, SlotExample] = {}
        # —— rollout 相：eval() + no_grad（执行序第 1 相口径）——
        for card in self.cards:
            card.replica.policy.eval_phase()
        # 判别器步判定先行：fake 池化原料打分（score_fakes）只在判别
        # 器步产——非判别器步 iter 的打分前向不进静默丢弃面白付
        # （#220 决议 12「每活跃条件恰一次读数」消耗面的两侧同门控）
        disc_step_now = self.window.is_step(iteration)
        per_card = await self._collect(
            [
                card.submit(card.run_examples(
                    {slot: conditions[slot] for slot in card.slot_ids},
                    window_tasks={
                        slot: slot_tasks[slot] for slot in card.slot_ids
                    },
                    score_fakes=disc_step_now,
                ))
                for card in self.cards
            ],
            iteration,
            step_index=None,
            enforce_timeout=False,
        )
        for mapping in per_card:
            examples.update(mapping)
        # rollout 相即 join 形态（协程 gather 完成 = 任务产出全部落账）：
        # 逐卡累计窗口记录，判别器步装配消费
        for example in examples.values():
            self._window_run.collect(example)
        # —— train 相：逐 k barrier（k 间严格串行）——
        for card in self.cards:
            card.replica.policy.train_phase()
        example_count = len(conditions)
        losses: dict[int, dict[int, float]] = {
            slot: {} for slot in conditions
        }
        for step_index in sorted(self.config.policy.train_step_indices_m):
            reported = await self._collect(
                [
                    card.submit(card.accumulate_k(
                        examples, step_index, example_count,
                    ))
                    for card in self.cards
                ],
                iteration,
                step_index,
                enforce_timeout=True,
            )
            for card, values in zip(self.cards, reported):
                for slot, value in zip(card.slot_ids, values):
                    losses[slot][step_index] = value
            self._collect_reduce.reduce([card.replica for card in self.cards])
            await self._collect(
                [card.submit(card.step_optimizer()) for card in self.cards],
                iteration,
                step_index,
                enforce_timeout=True,
            )
        # —— 判别器步（k 循环全部完成后，#217 §3 排程）——
        disc_started = time.monotonic()
        reports: dict[int, UpdateReport] = {}
        divergences: dict[str, DivergenceReading] = {}
        pooled_auc: dict[str, float] = {}
        if disc_step_now:
            # —— 池化 AUC：每活跃条件恰一次读数（#220 决议 12）——
            # 窗口末（判别器更新前）快照、覆盖式入持久账（本窗口活跃
            # 条件刷新读数）；非判别器步 iter 不打分（「恰一次」的消耗
            # 面），iter 事件的 heldout_auc 桥接持久账的逐条件末次读数
            # 或 None（下方事件段）。
            active = tuple(sorted(
                {example.record.modality for example in examples.values()}
            ))
            # 采样计数 = 本卡本条件 fake 数（非卡级 fake 总量）——池化
            # 后 real ≈ fake 的逐卡对称平面：旧卡级总量驱动使 n:m 不
            # 对称随卡数 × 条件数放大（少 fake 条件吃多 fake 条件的
            # 采样量），评审定裁改逐卡配平（#220 决议 12 未规定计数）
            per_card_fake_counts = [
                {
                    condition: sum(
                        example.record.new_fakes.shape[0]
                        for example in mapping.values()
                        if example.record.modality == condition
                    )
                    for condition in active
                }
                for mapping in per_card
            ]
            per_card_real = await self._collect(
                [
                    card.submit(
                        card.score_heldout(active, per_card_fake_counts[i])
                    )
                    for i, card in enumerate(self.cards)
                ],
                iteration,
                step_index=None,  # 池化段不套超时；k=None 的非 k 口径
                enforce_timeout=False,
            )
            pooled_auc = self._pool_auc(examples, per_card_real)
            self._window_run.record_auc(pooled_auc)
            planned = {
                card.index: self._window_run.planned_count(card.index)
                for card in self.cards
            }
            per_card_reports = await self._collect(
                [
                    card.submit(card.discriminate(
                        self._window_run.records(card.index),
                        planned[card.index],
                    ))
                    for card in self.cards
                ],
                iteration,
                step_index=None,  # 同上：判别器步 barrier 的非 k 口径
                enforce_timeout=True,
            )
            reports = {
                card.index: report
                for card, report in zip(self.cards, per_card_reports)
            }
            DiscriminatorPhase.reduce_gradients(
                [card.replica.disc_phase for card in self.cards]
            )
            await self._collect(
                [card.submit(card.disc_step()) for card in self.cards],
                iteration,
                step_index=None,  # 同上：判别器步 barrier 的非 k 口径
                enforce_timeout=True,
            )
            DiscriminatorPhase.synchronize_spectral(
                [card.replica.disc_phase for card in self.cards]
            )
            # 分叉观测（#220 决议 13/14 + 多卡聚合裁决补记）：**每条件
            # 每判别器步恰一次 observe**（每条件一个 EMA 的决议字面——
            # 同条件跨卡多桶时先按桶对数加权聚合成条件级单值 train
            # acc（逐对等权估计量的同构聚合，与 loss×n_b/N_total 同
            # 口径），再喂一次观测；多卡同条件多桶的聚合形态为 #220
            # 决议 14 未明文的补记，确定性纯函数、随本票 docstring
            # 立此存照）
            condition_pair_accs: dict[str, list[tuple[int, float]]] = {}
            for card in self.cards:
                for detail in reports[card.index].conditions:
                    condition_pair_accs.setdefault(
                        detail.condition, [],
                    ).append((detail.pair_count, detail.train_pairwise_acc))
            condition_train_accs: dict[str, float] = {}
            for condition, reads in sorted(condition_pair_accs.items()):
                heldout = self._window_run.window_auc(condition)
                if heldout is None:
                    raise ValueError(
                        f"条件 {condition!r} 有判别器桶但本 run 无任何池化 "
                        "AUC 读数——「窗口内最后一次有效测量」的跨窗口回退"
                        "点失守。生产几何（槽数 ≥ 条件数，轮长 1）每 iter "
                        "全条件活跃、桶条件必有本步读数；fixture 几何（槽数 "
                        "< 条件数）下非退化窗口可摊到从未被测的条件（轮长 "
                        "对齐 N_d 时逐轮条件置换不相交）——fail-fast 不"
                        "静默跳测，几何扩面随用例另行裁决（#220 决议 14 "
                        "回退点 = 该条件最后被分配的判别器步 iter）"
                    )
                total = sum(count for count, _ in reads)
                train_acc = sum(
                    acc * count for count, acc in reads
                ) / total
                condition_train_accs[condition] = train_acc
                divergences[condition] = self._overfit.observe(
                    condition,
                    train_pairwise_acc=train_acc,
                    heldout_auc=heldout,
                )
            # 越线告警原料在清窗前收集（heldout_auc 读窗口记账）——
            # 事件构造推迟到事件发射段（elapsed 计时完整覆盖）
            alert_data = [
                (
                    condition,
                    reading,
                    condition_train_accs[condition],
                    self._window_run.window_auc(condition),
                )
                for condition, reading in sorted(divergences.items())
                if reading.alerted
            ]
            self._window_run.clear()
        else:
            alert_data = []
        disc_seconds = time.monotonic() - disc_started
        # —— 事件发射：两档 flush（#222）——
        # iter 族：缓冲到 iteration 边界、按 (iteration, slot) 排序后
        # 单点写（单进程单写者，slot 升序即归并序；rank 字段语义 =
        # 槽号，#217 契约口径）；告警族：产出即写、不进 iter 族缓冲
        # （barrier 软超时在 _collect 内已即写；此处 overfit_alert
        # 产出即写、排在 iter 族块之后——同 iteration 告警随本槽
        # iter 事件之后的归并序口径不变）。
        elapsed = time.monotonic() - started
        alerts = [
            OverfitAlertEvent(
                iteration=iteration,
                stage=_STAGE_TAG,
                rank=0,  # 主线程 = 卡 0 写出口径（barrier_timeout_alert 同款）
                phase="rl",
                modality=condition,
                divergence_ema=reading.divergence,
                train_pairwise_acc=train_acc,
                heldout_auc=heldout,
            )
            for condition, reading, train_acc, heldout in alert_data
        ]
        # 槽 → 卡映射一次建立（事件段逐槽查卡的线性扫描收敛为 O(1)）
        slot_card_index = {
            slot: card.index
            for card in self.cards
            for slot in card.slot_ids
        }
        iter_events: list[IterEvent] = []
        for slot in sorted(conditions):
            example = examples[slot]
            card_index = slot_card_index[slot]
            report = reports.get(card_index)
            detail = (
                self._disc_detail(report, example.record.modality)
                if report is not None else None
            )
            condition = example.record.modality
            if condition in pooled_auc:
                heldout_auc: float | None = pooled_auc[condition]
            else:
                # 非判别器步 iter：桥接该条件最后一次有效池化读数
                # （#220 决议 14「窗口内最后一次有效测量」的窗口延拓——
                # AUC 账跨窗口持久，本窗口未活跃条件回退到上次读数）；
                # 本 run 尚无读数 = None（首现于非判别器步 iter 的条件，
                # 其所在窗口的测量尚未发生）——「每活跃条件恰一次读数」
                # 的消耗面：非判别器步 iter 不采样不打分，与
                # train_pairwise_acc/disc_update 的 N_d 跳过 None
                # 惯用法同构，不静默写零。
                heldout_auc = self._window_run.window_auc(condition)
            event_loss = {
                f"policy_step_{step_index}": value
                for step_index, value in losses[slot].items()
            }
            phase_seconds = {
                "rollout": example.rollout_seconds,
                "policy_update": example.update_seconds,
            }
            if report is not None:
                event_loss["discriminator"] = report.loss_discriminator
                phase_seconds["discriminator"] = disc_seconds
            divergence = divergences.get(condition)
            iter_events.append(IterEvent(
                iteration=iteration,
                stage=_STAGE_TAG,
                rank=slot,
                modality=condition,
                anchor_eval_reward=example.record.anchor_eval_reward,
                intra_group_reward_std=example.record.intra_group_reward_std,
                heldout_auc=heldout_auc,
                loss=event_loss,
                train_pairwise_acc=(
                    detail.train_pairwise_acc if detail is not None else None
                ),
                overfit_divergence_ema=(
                    divergence.divergence if divergence is not None else None
                ),
                disc_update=self._disc_update_event(
                    report, divergences,
                ) if report is not None else None,
                lr=self.config.policy.policy_lr,
                elapsed_s=elapsed,
                phase_seconds=phase_seconds,
            ))
        self.artifacts.append_events(iter_events)
        for alert in alerts:
            self.artifacts.append_event(alert)

    @staticmethod
    def _pool_auc(
        examples: dict[int, SlotExample],
        per_card_real: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """每活跃条件恰一次池化读数（#220 决议 12）：fake 分数按例记账
        （打分在 fakes 所在卡）+ real 分数按卡号升序（future 回传序）
        收集 → 单点 float64 midrank Mann-Whitney。"""
        fake_scores: dict[str, list[torch.Tensor]] = {}
        for example in examples.values():
            fake_scores.setdefault(
                example.record.modality, [],
            ).append(example.fake_scores)
        pooled: dict[str, float] = {}
        for condition in sorted(fake_scores):
            real = torch.cat([
                scores[condition] for scores in per_card_real
            ])
            fake = torch.cat(fake_scores[condition])
            pooled[condition] = HeldOutAuc.auc_from_scores(real, fake)
        return pooled

    @staticmethod
    def _disc_detail(
        report: UpdateReport, modality: str,
    ) -> ConditionUpdateDetail | None:
        """本例条件的报告明细（判别器步时；本例条件无桶 = None——该卡
        窗口未摊派本条件的任务）。"""
        for detail in report.conditions:
            if detail.condition == modality:
                return detail
        return None

    @staticmethod
    def _disc_update_event(
        report: UpdateReport, divergences: dict[str, DivergenceReading],
    ) -> DiscUpdateDetail:
        """判别器步的逐条件事件明细（#220 决议 11：升格面在 iter 事件
        的映射；同卡各例一致读数）。"""
        return DiscUpdateDetail(
            global_batch_size=report.global_batch_size,
            weighted_loss=report.loss_discriminator,
            conditions={
                detail.condition: DiscConditionReading(
                    loss=detail.loss_discriminator,
                    loss_real_term=detail.loss_real_term,
                    loss_fake_term=detail.loss_fake_term,
                    pair_count=detail.pair_count,
                    train_pairwise_acc=detail.train_pairwise_acc,
                    divergence_ema=(
                        divergences[detail.condition].divergence
                        if detail.condition in divergences else None
                    ),
                )
                for detail in report.conditions
            },
        )

    async def _collect(
        self,
        futures: list[Future],
        iteration: int,
        step_index: int | None,
        enforce_timeout: bool,
    ) -> list[Any]:
        """barrier 等待与 fail-fast 门面：任一槽线程异常 →
        ``TrainingAborted``（原异常挂 cause）；软超时 → 告警事件（#222
        告警族即写：产生即写、先于 iter 族块）后继续等待；硬超时 →
        fail-fast。rollout 相派发不套超时（超时口径属 per-k barrier，
        #217 §4），异常 fail-fast 恒生效。``step_index`` = None 的非 k
        口径（判别器步集合段、restore 下发）经 ``k_label`` 显式标注。"""
        started = time.monotonic()
        k_label = f"k{step_index}" if step_index is not None else "非k段"
        wrapped = [asyncio.wrap_future(future) for future in futures]
        soft_warned = False
        while True:
            waited = time.monotonic() - started
            if not enforce_timeout:
                budget = None
            elif not soft_warned:
                budget = max(0.0, self._timeout.soft_seconds - waited)
            else:
                budget = max(0.0, self._timeout.hard_seconds - waited)
            done, pending = await asyncio.wait(
                wrapped, timeout=budget,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for future in done:
                error = future.exception()
                if error is not None:
                    raise TrainingAborted(
                        f"槽线程异常（iteration {iteration}、{k_label}）："
                        f"{error}——fail-fast 门面中止训练"
                    ) from error
            if not pending:
                return [future.result() for future in wrapped]
            waited = time.monotonic() - started
            if not enforce_timeout:
                continue
            if not soft_warned:
                soft_warned = True
                self.artifacts.append_event(BarrierTimeoutAlertEvent(
                    iteration=iteration,
                    stage=_STAGE_TAG,
                    k=step_index,  # None = 非 k barrier（判别器步集合段）
                    elapsed_s=waited,
                ))
            if waited >= self._timeout.hard_seconds:
                raise TrainingAborted(
                    f"per-k barrier 硬超时（iteration {iteration}、"
                    f"{k_label}）：等待 {waited:.1f}s ≥ 阈值 "
                    f"{self._timeout.hard_seconds:.1f}s——"
                    "CYNOSURE_PG_TIMEOUT_MIN 语义换绑口径的 fail-fast"
                )
