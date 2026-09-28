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
- **barrier 软/硬超时**（#217 §4）：软超时 → ``barrier_soft_timeout``
  事件（主线程 = 卡 0 写出口径）后继续等待；硬超时 → fail-fast。
  ``CYNOSURE_PG_TIMEOUT_MIN`` 变量沿用、**语义换绑** per-k barrier 硬
  超时（分钟；sugon 已设 40，零部署变更；未设置 = torch 默认 10 分钟
  口径沿用）。
- **事件发射**：iter 族 (iteration, slot) 排序写——``IterEvent.rank``
  字段名不动、语义重定义为调度槽号（#217 事件契约「可扩不可改名」）；
  单进程单写者，slot 升序即 (iteration, slot) 归并序。

骨架期包含面（#226 决策 3）：静态分配表轮（轮内置换）+ per-槽协程骨架
+ 逐 k barrier 收集-同步 + 事件发射 + RNG 注册表 per-槽实例化 + 异常
fail-fast + barrier 软/硬超时；两域（MR + BraTS）贯穿。**不含**（各进
加厚期，#226）：续训分片、评测路径、判别器链（判别器更新步/混合条件
配对批/u-v broadcast——骨架期判别器仅承载打分与 AUC 的 rollout 消费，
恒 eval 相）、pretrain driver、生产入口（本门面仅被 fixture 测试驱动，
#226 决策 1 生产入口单口径）。
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

from cynosure.config import CynosureConfig
from cynosure.distributed.process import (
    DistributedContext,
    PG_TIMEOUT_MINUTES_ENV,
)
from cynosure.grpo import ClippedPolicyLoss, MgaiAdvantage, StepwisePolicyUpdate
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.pretrain.artifacts import PretrainReport
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.scorer import RewardScorer
from cynosure.train.allocation import AllocationTable
from cynosure.train.artifacts import (
    BarrierSoftTimeoutEvent,
    IterEvent,
    RunArtifacts,
)
from cynosure.train.policy import GroupPolicy
from cynosure.train.rollout import IterationRollout, RolloutPhase, StepRollout
from cynosure.train.rng import DropoutGuard, SlotRngRegistry, TrainingRngStreams
from cynosure.train.runtime import TrainingRuntime

_T = TypeVar("_T")

_STAGE_TAG = 1
"""骨架期事件的阶段号（单阶段组缺省；组3 StageTag 机制属 trainer 面，
本门面 fixture 薄切片不承载序贯）。"""

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
    heldout_auc: float
    rollout_seconds: float
    steps: dict[int, StepRollout] = field(default_factory=dict)
    """按被优化训练步 k 索引的 rollout 记录（更新相任务的取数面）。"""
    update_seconds: float = 0.0


class CardReplica:
    """每卡完整副本（#217 §4）：本卡的 policy 装配 + 判别器 scorer 副本
    + 采样封装 + 逐 k 更新编排——副本间无共享可变张量状态，跨卡一致性
    由「同初始化 + 确定性 allreduce + 同步 step 序列」结构性保证。"""

    def __init__(
        self,
        index: int,
        device: torch.device,
        policy: GroupPolicy,
        scorer: RewardScorer,
        sampler: RolloutSampler,
        updater: StepwisePolicyUpdate,
    ) -> None:
        self.index = index
        self.device = device
        self.policy = policy
        self.scorer = scorer
        self.sampler = sampler
        self.updater = updater

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        index: int,
        device: torch.device,
        scorer_prototype: RewardScorer,
    ) -> "CardReplica":
        """本卡副本装配：policy 网络构建（checkpoint 装载，单进程无
        FSDP 包装——分片随执行模型退役）+ 判别器副本（原型 deepcopy 后
        迁卡——单点装载、逐位复制）+ 采样封装（前向激活预算本地解析，
        ``chunk_sync`` 恒 None：无 FSDP 即无「前向调用次数绑定集合序列」，
        #165 挂死类结构性消失）+ 逐 k 更新编排。判别器副本钉 eval 相
        （骨架期无判别器更新步，打分/AUC 前向不得推进 spectral norm
        幂迭代）。"""
        amp = AmpContext(
            device=device, dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
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
        return cls(index, device, policy, scorer, sampler, updater)

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
    复用到绑卡线程，torch 调用全部落在绑卡线程）。"""

    def __init__(
        self,
        slot: int,
        updater: StepwisePolicyUpdate,
        rollout: RolloutPhase,
        auc: HeldOutAuc,
        advantage_clamp: float,
    ) -> None:
        self.slot = slot
        self._updater = updater
        self._rollout = rollout
        self._auc = auc
        self._advantage = MgaiAdvantage(clamp=advantage_clamp)

    async def run_example(self, condition_name: str) -> SlotExample:
        """一个例子的 rollout 相（分配表条件 → 初始噪声/扰动/续跑 →
        打分 → held-out AUC——AUC 属 rollout 相口径，#217 §3）。"""
        started = time.monotonic()
        record = self._rollout.run_iteration(condition_name)
        heldout = self._auc.compute(record.new_fakes, record.modality)
        return SlotExample(
            slot=self.slot,
            record=record,
            heldout_auc=heldout,
            rollout_seconds=time.monotonic() - started,
            steps={step.step_index: step for step in record.steps},
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
    （流 owner-thread 断言的记录点 = 绑卡线程，#218 出口断言口径）。"""

    def __init__(
        self,
        index: int,
        device: torch.device,
        replica: CardReplica,
        slot_ids: list[int],
        build_slots: Callable[[], list[SlotRunner]],
    ) -> None:
        self.index = index
        self.device = device
        self.replica = replica
        self.slot_ids = slot_ids
        self.slots: list[SlotRunner] = []
        self._build_slots = build_slots
        self._failure: BaseException | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._serve,
            daemon=True,
            name=f"cynosure-card-{index}",
        )

    def _serve(self) -> None:
        """线程本体：静态绑卡（current device 线程局部）→ 槽装配（本
        线程取流，owner 记录即绑卡线程）→ 事件循环常驻。"""
        try:
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            self.slots = self._build_slots()
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
        self, conditions: dict[int, str],
    ) -> dict[int, SlotExample]:
        """本卡全部例子的 rollout 相（槽间并发 gather——协程多路复用
        形态；torch 调用在本线程内串行落卡）。"""
        examples = await asyncio.gather(*(
            slot.run_example(conditions[slot.slot]) for slot in self.slots
        ))
        return {example.slot: example for example in examples}

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
    ``SOFT_FRACTION`` 处发 ``barrier_soft_timeout`` 告警事件后继续等待。
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


class AsyncTrainingExecutor:
    """async 执行序门面：静态分配表轮 + per-槽协程骨架 + 逐 k barrier
    收集-同步 + 事件发射 + fail-fast/超时口径的单点编排（#231 骨架期
    全宽薄切片；两域 MR + BraTS 贯穿，条件轴经条件分布 ``targets()``）。"""

    def __init__(
        self,
        config: CynosureConfig,
        artifacts: RunArtifacts,
        allocation: AllocationTable,
        rng: SlotRngRegistry,
        cards: list[CardWorker],
        collect_reduce: PerKCollectReduce,
        timeout: BarrierTimeoutPolicy,
    ) -> None:
        self.config = config
        self.artifacts = artifacts
        self.allocation = allocation
        self.rng = rng
        self.cards = cards
        self._collect_reduce = collect_reduce
        self._timeout = timeout

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
    ) -> "AsyncTrainingExecutor":
        """config 驱动装配：设备发现（卡数）→ 条件轴（条件分布
        ``targets()``，集合知识归条件分布自身）→ 分配表 + 槽注册表 →
        每卡副本与绑卡线程（槽静态 round-robin 绑卡）。``coroutines``
        缺省 = 卡数（每卡一例的默认拓扑）；``devices`` 显式设备集（
        CPU fixture 档钉单 CPU 设备——生产口径缺省 = 本进程可见全卡，
        卡集裁剪经 ``CUDA_VISIBLE_DEVICES``）；``owner_check`` 透传注册
        表（测试档开启，#218 debug-only 口径）。"""
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
        scorer_prototype = cls.assemble_discriminator(config)
        vocabulary = TrainingRuntime.assemble_vocabulary(config)
        replica = CardReplica.build(config, 0, devices[0], scorer_prototype)
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
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        # 逐条件形状契约的装配期对照（#129 消费侧守卫沿袭——词表工件与
        # held-out manifest 的同名异形在首例测量时才炸属失败后移）
        heldout.assert_condition_shapes(vocabulary)
        replicas = [replica] + [
            CardReplica.build(config, index, device, scorer_prototype)
            for index, device in enumerate(devices[1:], 1)
        ]
        cards = []
        for replica in replicas:
            bound_slots = [
                slot for slot in range(slot_count)
                if slot % len(devices) == replica.index
            ]
            cards.append(CardWorker(
                index=replica.index,
                device=replica.device,
                replica=replica,
                slot_ids=bound_slots,
                build_slots=cls._slot_assembly(
                    config, replica, rng, vocabulary, heldout, bound_slots,
                ),
            ))
        return cls(
            config=config,
            artifacts=run_artifacts,
            allocation=allocation,
            rng=rng,
            cards=cards,
            collect_reduce=PerKCollectReduce(),
            timeout=(
                timeout if timeout is not None
                else BarrierTimeoutPolicy.from_env()
            ),
        )

    @staticmethod
    def _slot_assembly(
        config: CynosureConfig,
        replica: CardReplica,
        rng: SlotRngRegistry,
        vocabulary,
        heldout: LatentManifest,
        slot_ids: list[int],
    ) -> Callable[[], list[SlotRunner]]:
        """本卡槽装配的闭包工厂：在绑卡线程执行（流 owner-thread 断言
        的记录点 = 绑卡线程）——per-槽 RolloutPhase（槽 rollout 流注入，
        #218 五处消费面注入面零改动口径）与 HeldOutAuc（槽 heldout 流）。
        """
        amp = AmpContext(
            device=replica.device, dtype=AMP_DTYPES[config.policy.amp_dtype],
        )

        def build_slots() -> list[SlotRunner]:
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
                auc = HeldOutAuc(
                    heldout_manifest=heldout,
                    scorer=replica.scorer,
                    generator=rng.get_stream(
                        slot, TrainingRngStreams.HELDOUT_AUC,
                    ),
                    device=replica.device,
                )
                runners.append(SlotRunner(
                    slot,
                    replica.updater,
                    rollout,
                    auc,
                    config.grpo.advantage_clamp,
                ))
            return runners

        return build_slots

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
        """训练主循环（骨架期口径）：预热 → 绑卡线程启动 → 逐 iteration
        （rollout 相 → 逐 k barrier → 事件发射）→ 线程收尾。返回完成的
        iteration 数。无续训/评测/判别器链/checkpoint——各进加厚期
        （#226 决策 3 骨架期包含面）。"""
        self._warmup()
        for card in self.cards:
            card.start()
        try:
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
        completed = 0
        for iteration in range(self.config.schedule.max_iterations):
            await self._run_iteration(iteration)
            completed = iteration + 1
        return completed

    async def _run_iteration(self, iteration: int) -> None:
        """单 iteration 执行序（#217 §3 相位结构）：rollout 相（全并发
        零梯度耦合）→ train 相逐 k barrier（loss×(1/N)+SUM → 各卡一次
        step）→ 事件发射（(iteration, slot) 排序写）。判别器更新步缺位
        属骨架期口径（判别器链期落地，#226）。phase_seconds 只发
        rollout / policy_update 两相（#217 §3 全集含 trajectory /
        discriminator 的记档偏差）：trajectory 相现行仅 --dump 诊断
        打点消费、骨架期无诊断路径，discriminator 相属判别器链期——
        两相随各自加厚期补入。"""
        started = time.monotonic()
        conditions = {
            slot: self.allocation.condition_for(iteration, slot)
            for slot in range(self.allocation.slot_count)
        }
        examples: dict[int, SlotExample] = {}
        # —— rollout 相：eval() + no_grad（执行序第 1 相口径）——
        for card in self.cards:
            card.replica.policy.eval_phase()
        per_card = await self._collect(
            [
                card.submit(card.run_examples({
                    slot: conditions[slot] for slot in card.slot_ids
                }))
                for card in self.cards
            ],
            iteration,
            step_index=None,
            enforce_timeout=False,
        )
        for mapping in per_card:
            examples.update(mapping)
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
        # —— 事件发射：iter 族 (iteration, slot) 排序写（单进程单写者，
        # slot 升序即归并序；rank 字段语义 = 槽号，#217 契约口径）——
        elapsed = time.monotonic() - started
        for slot in sorted(conditions):
            example = examples[slot]
            self.artifacts.append_event(IterEvent(
                iteration=iteration,
                stage=_STAGE_TAG,
                rank=slot,
                modality=example.record.modality,
                anchor_eval_reward=example.record.anchor_eval_reward,
                intra_group_reward_std=example.record.intra_group_reward_std,
                heldout_auc=example.heldout_auc,
                loss={
                    f"policy_step_{step_index}": value
                    for step_index, value in losses[slot].items()
                },
                lr=self.config.policy.policy_lr,
                elapsed_s=elapsed,
                phase_seconds={
                    "rollout": example.rollout_seconds,
                    "policy_update": example.update_seconds,
                },
            ))

    async def _collect(
        self,
        futures: list[Future],
        iteration: int,
        step_index: int | None,
        enforce_timeout: bool,
    ) -> list:
        """barrier 等待与 fail-fast 门面：任一槽线程异常 →
        ``TrainingAborted``（原异常挂 cause）；软超时 → 告警事件（主
        线程 = 卡 0 写出口径）后继续等待；硬超时 → fail-fast。rollout
        相派发不套超时（超时口径属 per-k barrier，#217 §4），异常
        fail-fast 恒生效。"""
        started = time.monotonic()
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
                        f"槽线程异常（iteration {iteration}、step "
                        f"{step_index}）：{error}——fail-fast 门面中止训练"
                    ) from error
            if not pending:
                return [future.result() for future in wrapped]
            waited = time.monotonic() - started
            if not enforce_timeout:
                continue
            if not soft_warned:
                soft_warned = True
                self.artifacts.append_event(BarrierSoftTimeoutEvent(
                    iteration=iteration,
                    stage=_STAGE_TAG,
                    step_index=step_index,
                    waited_s=waited,
                    threshold_s=self._timeout.soft_seconds,
                ))
            if waited >= self._timeout.hard_seconds:
                raise TrainingAborted(
                    f"per-k barrier 硬超时（iteration {iteration}、step "
                    f"{step_index}）：等待 {waited:.1f}s ≥ 阈值 "
                    f"{self._timeout.hard_seconds:.1f}s——"
                    "CYNOSURE_PG_TIMEOUT_MIN 语义换绑口径的 fail-fast"
                )
