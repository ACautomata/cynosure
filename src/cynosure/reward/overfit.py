"""过拟合分叉监控（ADR-0009 决策 4/5 的在线检测面）。

判别器内收敛健康度的观测面：分叉 = EMA(train pairwise acc − held-out
AUC)，两侧统一干净域。train 侧每判别器步用干净域输入 no_grad 复算一次
pairwise 准确率（随单步更新报告上行，更新原语 ``OnlineUpdate.step`` 的
复算接线）——不复用 loss 伴生量，因其带噪输入使训练批任务天然更难、
系统性低估分叉；held-out 侧消费现成 per-condition AUC 流（iter 事件的
更新前快照口径）。健康判别器的两侧同估计量（Mann-Whitney pairwise
占比）近似相等、分叉贴 0；判别器记住训练批共性而非真假分界时 train
侧被 in-sample 拟合抬高、分叉上行——hacking 后果出现前的病因信号。

监控器是**薄的有状态组件**（EMA 跨 iteration，无法塞进单步更新原语）：

- per-condition 独立记账：条件间 EMA 与越线判定互不可见（观测流是
  per-condition 稀疏序列——该条件的判别器步才有一条观测，EMA 跨度
  语义 = 该条件观测条数的平滑尺度，与 gating 的 EMA(AUC) 同口径）；
- **按 rank 独立**：监控器是 rank 本地状态、无任何集合通信——判别器
  是 DDP 完整副本、权重各 rank 同步，分叉的 rank 间离散反映数据切片
  异质性，本身是诊断信号（不跨 rank 平均，随 iter 事件同归并序落盘）；
- 报警触发边界 = 分叉 EMA **自下而上**达到阈值（线上滞留不重发、
  回落后再越线重发；阈值点本身算越线，与门控 enter 判定的 ``>=`` 同
  语义；首观测即越线 = 出生即分叉，同样报警）；
- **报警不动作**（ADR-0009 决策 5）：越线只产 ``alerted=True`` 读数、
  由编排方落 ``overfit_alert`` 事件——监控器不持白名单与 σ 的任何
  引用，不自动移出白名单、不自动调 σ（人工裁决）。

per-condition EMA 状态经 ``state``/``adopt`` 与 ResumeStore 对接
（续训分片的 ``overfit`` 键，恢复逐位复原——iter 事件的分叉 EMA 字段
进逐位轨迹比对，状态不落盘即续训 roundtrip 失真）。
"""

import math
from dataclasses import dataclass

from cynosure.config import RewardConfig


@dataclass(frozen=True)
class DivergenceReading:
    """单次分叉观测的可观测结果（指标流与测试断言的数据）。"""

    divergence: float
    """本条件分叉的当前 EMA 值（train pairwise acc − held-out AUC 的
    平滑观测）。"""
    alerted: bool
    """本次观测是否触发越线报警（上升沿语义见模块 docstring）。"""


class DivergenceEma:
    """单条件分叉的指数移动平均（ADR-0009 决策 4 的平滑观测器）。

    首个观测直接置值（递推无初值偏置）；此后
    ``ema ← (1−α)·ema + α·sample``，α 由 EMA 跨度换算（span=8 → 2/9）
    ——与 gating 的 EMA(AUC)（``ConditionAucEma``）同一换算口径（本类
    与其互为镜像的独立实现：分叉监控不依赖门控模块，两处 docstring
    各自锚定自己的观测流语义）。

    观测流是 per-condition 稀疏序列（条件被采样到且判别器步发生才有一条
    观测）——指数 EMA 的跨度语义 = **该条件观测条数**的平滑尺度（非
    wall-clock iteration 数）；校准口径（config ``overfit_ema_span``
    注释）同此。
    """

    def __init__(self, span: int) -> None:
        if span < 1:
            raise ValueError(f"EMA 跨度须为正整数，得到 {span}")
        self._span = span
        self._value: float | None = None
        self._count = 0

    @property
    def value(self) -> float | None:
        """当前 EMA 值（尚无观测 = None——越线判定不消费未观测条件）。"""
        return self._value

    @property
    def count(self) -> int:
        """累计观测条数（诊断面；续训状态随之落盘）。"""
        return self._count

    def observe(self, sample: float) -> float:
        """喂入一条分叉观测，返回更新后的 EMA。"""
        alpha = 2.0 / (self._span + 1.0)
        self._value = (
            sample if self._value is None
            else (1.0 - alpha) * self._value + alpha * sample
        )
        self._count += 1
        return self._value

    def restore(self, value: float, count: int) -> None:
        """落盘状态的逐位回填（续训恢复的入口）：EMA 的递推链不可从
        终值重放（初值与中间观测未落盘），状态机的回填语义即
        「value/count 直接置位」——与 observe 构成状态演化的一对写入口
        （gating 的 EMA(AUC) 同款语义）。"""
        if count < 1:
            raise ValueError(f"EMA 观测计数须 ≥ 1，得到 {count}")
        self._value = value
        self._count = count


class OverfitMonitor:
    """per-condition 过拟合分叉监控器（ADR-0009-β）：分叉 EMA 记账与
    越线判定的 rank 本地单点。

    ``observe`` 由编排方（train 循环）在**每个判别器步**调用一次：
    train 侧干净域复算准确率（更新报告上行）与本 iteration 的 held-out
    AUC 合成分叉观测、递推该条件 EMA、做上升沿越线判定。阈值与跨度是
    暂定 knob（config ``reward.overfit_*``，标注「MR-RATE 预训练曲线
    校准后定版」）。

    监控器无集合通信（按 rank 独立，见模块 docstring）；续训状态经
    ``state``/``adopt`` 与 ResumeStore 对接（resume v6 的 ``overfit``
    键，恢复逐位复原）。
    """

    def __init__(
        self, config: RewardConfig, conditions: tuple[str, ...],
    ) -> None:
        self._threshold = config.overfit_alert_divergence
        self._ema: dict[str, DivergenceEma] = {}
        self._span = config.overfit_ema_span
        self._conditions = tuple(conditions)

    def observe(
        self,
        modality: str,
        *,
        train_pairwise_acc: float,
        heldout_auc: float,
    ) -> DivergenceReading:
        """喂入一次分叉观测（本步判别器的两侧干净域读数），返回读数。

        两侧量非有限即拒绝（判别器数值发散在测量层 fail-fast，不以
        「NaN 排尾、Inf 排头」的排序语义伪装成分叉值——与 AUC 分数的
        有限性闸口同口径，「非有限浮点全流拒绝」的源头）；拒绝发生在
        EMA 递推之前，状态不动。
        """
        if not (math.isfinite(train_pairwise_acc) and math.isfinite(heldout_auc)):
            raise ValueError(
                f"分叉观测须为有限值（判别器数值发散或测量损坏）："
                f"train pairwise acc={train_pairwise_acc!r}、"
                f"held-out AUC={heldout_auc!r}——非有限浮点进 EMA 会以"
                "毒值污染后续全部读数（全流拒绝口径的源头闸口）"
            )
        tracker = self._tracker(modality)
        previous = tracker.value
        divergence = tracker.observe(train_pairwise_acc - heldout_auc)
        crossed = previous is None or previous < self._threshold
        return DivergenceReading(
            divergence=divergence,
            alerted=bool(crossed and divergence >= self._threshold),
        )

    def state(self) -> dict:
        """监控状态的落盘形态（weights_only 兼容原语；resume v6 的
        ``overfit`` 键）：per-condition EMA（未观测条件无条目）。"""
        return {
            "ema": {
                modality: {"value": ema.value, "count": ema.count}
                for modality, ema in sorted(self._ema.items())
            },
        }

    def adopt(self, state: dict) -> None:
        """监控状态回填（续训恢复入口）：per-condition EMA 整体重建为
        快照时刻的逐位状态。损坏形态显式拒绝（分片损坏/篡改不在恢复
        面静默漂移）。"""
        if not isinstance(state, dict) or set(state) != {"ema"}:
            raise ValueError(
                f"分叉监控状态形态非法（须为 {{ema}}）: "
                f"{sorted(state) if isinstance(state, dict) else type(state)}"
            )
        ema_raw = state["ema"]
        if not isinstance(ema_raw, dict):
            raise ValueError(f"分叉监控状态 EMA 清单形态非法: {type(ema_raw)}")
        restored: dict[str, DivergenceEma] = {}
        for modality, entry in ema_raw.items():
            if self._conditions and modality not in self._conditions:
                raise ValueError(f"分叉监控状态 EMA 含非法条件: {modality!r}")
            if not isinstance(entry, dict) or set(entry) != {"value", "count"}:
                raise ValueError(
                    f"分叉监控状态 EMA[{modality}] 条目形态非法（须为 "
                    f"{{value, count}}）: {entry!r}"
                )
            value = entry["value"]
            count = entry["count"]
            if (
                not isinstance(value, float | int)
                or isinstance(value, bool)
                or not isinstance(count, int)
                or isinstance(count, bool)
            ):
                raise ValueError(
                    f"分叉监控状态 EMA[{modality}] 字段类型非法（value 须"
                    f"为数值、count 须为 int）: {entry!r}"
                )
            tracker = DivergenceEma(self._span)
            tracker.restore(float(value), count)
            restored[modality] = tracker
        self._ema = restored

    def _tracker(self, modality: str) -> DivergenceEma:
        """该条件的 EMA 观测器（懒建：未观测条件不预置条目——落盘形态
        与「在线流从 run 起步」语义一致）。"""
        if modality not in self._ema:
            self._ema[modality] = DivergenceEma(self._span)
        return self._ema[modality]
