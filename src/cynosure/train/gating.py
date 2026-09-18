"""条件白名单的动态恢复（ADR-0008 决策 8）与逐 iteration 门控的决定面。

「条件白名单」（CONTEXT.md 词条）在运行时的完整形态：名单不只是预训练
报告的静态产物——被门控条件的判别器持续受训（门控只跳过 policy 更新），
其在线判别力出带后名单应自动恢复。恢复信号 = 在线 per-condition held-out
AUC 流（iter 事件已按目标模态归因）经 EMA 平滑后的滞回判定：

- gated 条件的 EMA 越过 enter 阈值 → 恢复该条件的 policy 更新；
- 名单内条件的 EMA 跌破 exit 阈值 → 重新门控；
- enter/exit 之间的滞回带维持现状（防测量噪声下的名单抖动）。

决定为**全 rank 集体口径**（ADR-0008 决策 7/8 的分布式前提），两层
含义：

- 名单状态集体一致：各 rank 的 rollout 条件独立采样、AUC 测量各自
  演化，观测经 all_gather 归并、rank 0 单点更新 EMA 与滞回判定、门控
  状态快照 broadcast 镜像到全体（各 rank 状态字面一致，逐 iteration
  查询天然同源）；单进程（world-1）集合原语恒等，同一条执行序。
- 跳过决定集体一致：policy 更新的 FSDP 梯度 allreduce 是全 rank 集合
  操作——任一 rank 的条件被门控 → **全体**跳过本 iteration 的 policy
  更新（无效样本不进入任何 rank 的梯度贡献）；部分 rank 跳过会让
  allreduce 互等死锁，任 rank 不得私自跳过。

阈值与跨度是暂定 knob（config ``reward.gating_*``，标注「MR-RATE 预
训练曲线校准后定版」）；动态恢复可经 ``gating_dynamic_recovery=false``
关闭——静态白名单为降级路径（恢复评估停步、名单恒为启动名单，跳过
对账仍走集体回合——名单外条件的跳过决定不因降级而私有化）。

门控状态（名单成员 + per-condition EMA）随续训分片落盘（resume v4），
恢复逐位复原——同口径续训 roundtrip 的逐位一致不变式覆盖门控决定。
"""

from typing import TYPE_CHECKING, NamedTuple, cast

from cynosure.config import RewardConfig
from cynosure.train.whitelist import ConditionWhitelist

if TYPE_CHECKING:
    from cynosure.distributed import DistributedContext

_EMA_ALPHA_NUMERATOR = 2.0
"""指数移动平均的常用跨度换算（pandas ewm span 语义）：α = 2/(span+1)
——span=8 即 α=2/9，_recent 观测权重 ≈ 22%。"""


class Observation(NamedTuple):
    """逐 rank 提交的门控观测载荷（all_gather 的提交物）：目标条件、
    held-out AUC 测量、本条件是否被门控——集体跳过决定的 OR 归约输入。"""

    modality: str
    auc: float
    gated: bool


class ConditionAucEma:
    """单条件在线 AUC 的指数移动平均（ADR-0008 决策 8 的平滑观测器）。

    首个观测直接置值（递推无初值偏置）；此后
    ``ema ← (1−α)·ema + α·sample``，α 由 EMA 跨度换算（span=8 → 2/9）。
    观测流是 per-condition 稀疏序列（条件被采样到才有一条观测）——
    指数 EMA 的跨度语义 = **该条件观测条数**的平滑尺度（非 wall-clock
    iteration 数：条件轮转下 8 条观测 ≈ 4 条件的 32 iteration）；校准
    口径（config ``gating_ema_span`` 注释）同此。
    """

    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError(f"EMA 跨度须为正整数，得到 {span}")
        self._span = span
        self._value: float | None = None
        self._count = 0

    @property
    def value(self) -> float | None:
        """当前 EMA 值（尚无观测 = None——名单判定不消费未观测条件）。"""
        return self._value

    @property
    def count(self) -> int:
        """累计观测条数（诊断面；续训状态随之落盘）。"""
        return self._count

    def observe(self, sample: float) -> float:
        """喂入一条 AUC 观测，返回更新后的 EMA。"""
        alpha = _EMA_ALPHA_NUMERATOR / (self._span + 1.0)
        self._value = (
            sample if self._value is None
            else (1.0 - alpha) * self._value + alpha * sample
        )
        self._count += 1
        return self._value

    def restore(self, value: float, count: int) -> None:
        """落盘状态的逐位回填（续训恢复与广播镜像的入口）：EMA 的递推
        链不可从终值重放（初值与中间观测未落盘），状态机的回填语义
        即「value/count 直接置位」——与 observe 构成状态演化的一对
        写入口。"""
        if count < 1:
            raise ValueError(f"EMA 观测计数须 ≥ 1，得到 {count}")
        self._value = value
        self._count = count


class DynamicWhitelist:
    """条件白名单的动态恢复运行时对象（ADR-0008 决策 8）：持有当前
    ``ConditionWhitelist`` 快照并以显式替换驱动名单变更（不可变值对象
    的变更协议，见 whitelist 模块）。

    ``observe(modality, auc)`` 是逐 iteration 的集体观测入口：全 rank
    各自提交本 iteration 的 (目标模态, held-out AUC, 本条件是否被门
    控)——rank 0 把同 iteration 同条件的各 rank AUC 观测均值合并为一
    条（多 rank 同条件观测的集体口径；单 rank 即原值），逐条更新该条
    件 EMA 并做滞回判定，变更后的门控状态快照 broadcast 镜像到全体；
    返回值 = 全 rank 的 OR 归约门控决定（见模块 docstring 的集体口径）。
    任何 rank 不得私自跳过/恢复。

    名单查询走 ``whitelist``（当前快照，``modality in whitelist``）；
    续训状态经 ``state``/``adopt`` 与 ResumeStore 对接（v4 分片的
    ``gating`` 键，恢复逐位复原）。
    """

    def __init__(
        self,
        initial: ConditionWhitelist,
        config: RewardConfig,
        dist: "DistributedContext",
        conditions: tuple[str, ...],
    ) -> None:
        self._config = config
        self._dist = dist
        self._conditions = tuple(conditions)
        self._current = ConditionWhitelist(
            self._ordered(initial.members), initial.measured,
        )
        self._ema: dict[str, ConditionAucEma] = {}

    def _ordered(self, members) -> tuple[str, ...]:
        """名单成员归一到本域条件集固定序（轮转序的子序列，#129 经
        ConditionVocabulary.names() 注入）：动态增删不破坏名单的确定性
        顺序——诊断显示序与恢复 roundtrip 的逐位一致都依赖它。"""
        wanted = set(members)
        return tuple(m for m in self._conditions if m in wanted)

    @property
    def whitelist(self) -> ConditionWhitelist:
        """当前生效名单快照（逐 iteration 门控查询的单点）。"""
        return self._current

    def observe(self, modality: str, auc: float) -> bool:
        """集体观测一步 + 集体门控决定（本 iteration 是否全体跳过
        policy 更新）。

        各 rank 的条件独立采样、``modality`` 是**本 rank** 的目标条件——
        但 policy 更新的 FSDP 梯度 allreduce 是全 rank 集合操作，「部分
        rank 跳过、其余执行」会让 allreduce 互等死锁。门控因此是集体
        口径：任一 rank 的条件被门控 → 全体跳过本 iteration 的 policy
        更新（无效样本不进入任何 rank 的梯度贡献，GRPO 语义不受
        跨 rank 稀释）；全 rank 同一条执行序。

        恢复评估（rank 0 更新 EMA 并滞回判定、快照广播镜像）只在动态
        恢复开启时发生；关闭（静态白名单降级路径）时名单恒为启动名单，
        跳过对账仍走集体回合（名单外条件的跳过决定依旧全 rank 一致）。

        条件闸关闭（``condition_gate_enabled=false``）时本入口整体退化为
        「无条件放行 + 集体回合」：名单恒为全条件、EMA 不更新、判定不
        发生，返回值恒 False。AUC 观测本身不在此处消费——它经 iter 事件
        与分叉监控照常落盘，关闸关的是「AUC 驱动更新决定」。
        """
        local_gated = modality not in self._current
        if not self._config.condition_gate_enabled:
            # 条件闸总开关关闭（维护者裁决，见 config RewardConfig 同名字段）：
            # 无条件放行——白名单不参与决定（其装配形态即「不设条件闸」的
            # 全条件放行占位），本入口不读名单、不做 EMA 递推与滞回判定。
            # 集体回合照走（policy 更新的 allreduce 要全 rank 同一条执行序），
            # 返回值恒 False（无 iteration 被跳过）。AUC 不在此消费——它经
            # iter 事件与分叉监控照常落盘
            self._dist.all_gather([False])
            return False
        if not self._config.gating_dynamic_recovery:
            # 静态白名单降级路径：名单恒为启动名单、无进出，跳过对账仍走
            # 集体回合
            flags = [entry[0] for entry in self._dist.all_gather([local_gated])]
            return any(flags)
        submissions = self._dist.all_gather([Observation(modality, auc, local_gated)])
        if self._dist.rank == 0:
            # 全 rank 观测的跨 rank 合并（同条件均值在 _ingest 内按
            # modality 分组完成）；每 rank 提交单条观测，取各自 [0]
            self._ingest([peer[0] for peer in submissions])
        snapshot = self._dist.broadcast_object(
            self.state() if self._dist.rank == 0 else None,
        )
        # 广播语义：返回值恒为 rank 0 的门控快照（dict）；None 只是非 0
        # rank 的占位入参，被覆盖
        self.adopt(cast("dict", snapshot))
        return any(entry[0].gated for entry in submissions)

    def state(self) -> dict:
        """门控状态的落盘形态（weights_only 兼容原语；resume v4 的
        ``gating`` 键）：名单成员 + per-condition EMA（未观测条件无
        条目）。``measured``（报告实测快照）不属门控状态，不随它落盘
        ——resume 不重查白名单（ADR-0008 决策 5），该快照无恢复语义。"""
        return {
            "members": list(self._current.members),
            "ema": {
                modality: {"value": ema.value, "count": ema.count}
                for modality, ema in sorted(self._ema.items())
            },
        }

    def adopt(self, state: dict) -> None:
        """门控状态回填（恢复与广播镜像共用入口）：名单与 EMA 整体重建
        为快照时刻的逐位状态。损坏形态显式拒绝（分片损坏/篡改不在
        恢复面静默漂移）。"""
        if not isinstance(state, dict) or set(state) != {"members", "ema"}:
            raise ValueError(
                f"门控状态形态非法（须为 {{members, ema}}）: "
                f"{sorted(state) if isinstance(state, dict) else type(state)}"
            )
        members = state["members"]
        unknown = [m for m in members if m not in self._conditions]
        if unknown:
            raise ValueError(f"门控状态含非法名单成员: {sorted(set(unknown))}")
        ema_raw = state["ema"]
        if not isinstance(ema_raw, dict):
            raise ValueError(f"门控状态 EMA 清单形态非法: {type(ema_raw)}")
        span = self._config.gating_ema_span
        restored: dict[str, ConditionAucEma] = {}
        for modality, entry in ema_raw.items():
            if modality not in self._conditions:
                raise ValueError(f"门控状态 EMA 含非法条件: {modality!r}")
            if (
                not isinstance(entry, dict)
                or set(entry) != {"value", "count"}
            ):
                raise ValueError(
                    f"门控状态 EMA[{modality}] 条目形态非法（须为 "
                    f"{{value, count}}）: {entry!r}"
                )
            tracker = ConditionAucEma(span)
            value = entry["value"]
            count = entry["count"]
            if (
                not isinstance(value, float | int)
                or isinstance(value, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
            ):
                raise ValueError(
                    f"门控状态 EMA[{modality}] 字段类型非法（value 须为 "
                    f"数值、count 须为 int）: {entry!r}"
                )
            tracker.restore(float(value), count)
            restored[modality] = tracker
        self._current = ConditionWhitelist(
            self._ordered(members), self._current.measured,
        )
        self._ema = restored

    def _ingest(self, submissions: list[Observation]) -> None:
        """rank 0 的判定段：同条件观测均值合并 → EMA 递推 → 滞回进出。

        ``submissions`` 已按 rank 序排列（all_gather 的稳定序），均值
        与判定逐条确定——同输入下各次执行逐位一致（同 seed 双 run 的
        门控决定一致不变式）。"""
        grouped: dict[str, list[float]] = {}
        for observation in submissions:
            grouped.setdefault(observation.modality, []).append(
                float(observation.auc),
            )
        means = {
            modality: sum(values) / len(values)
            for modality, values in grouped.items()
        }
        members = list(self._current.members)
        changed = False
        for modality in self._ordered(means):
            value = self._tracker(modality).observe(means[modality])
            if modality in members:
                # 跌破 exit → 重新门控（判别器继续受训，名单出口）
                if value < self._config.gating_exit_auc:
                    members.remove(modality)
                    changed = True
            elif value >= self._config.gating_enter_auc:
                # 越过 enter → 恢复更新（判别力出带的自动上岗）
                members.append(modality)
                changed = True
        if changed:
            self._current = ConditionWhitelist(
                self._ordered(members), self._current.measured,
            )

    def _tracker(self, modality: str) -> ConditionAucEma:
        """该条件的 EMA 观测器（懒建：未观测条件不预置条目——落盘形态
        与「在线流从 run 起步」语义一致）。"""
        if modality not in self._ema:
            self._ema[modality] = ConditionAucEma(
                self._config.gating_ema_span,
            )
        return self._ema[modality]
