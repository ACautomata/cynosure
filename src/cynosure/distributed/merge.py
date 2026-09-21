"""iter 事件的 rank 0 归并写出（orchestration 章「指标由 rank 0 归并写
出」）。

每个 RL iteration，各 rank 提交自己的 iter 事件（rollout 条件/噪声各
rank 独立 → 事件数值各 rank 不同），gather 到 rank 0 后按源 rank 升序
追加——同一 iteration 的 N 条事件连续排列且 rank 升序（无重复、无丢
失、顺序稳定：gather 的接收列表按源 rank 排列，归并序天然稳定）。
写盘只发生在 rank 0（RunArtifacts 的 rank 0 写盘契约）；单进程路径
= world-1 退化（gather 恒等、直写）。预训练相告警的归并（ADR-0016
决策 8）消费同一归并器：``PretrainRun`` 与 ``RunArtifacts`` 各自满足
``EventSink`` 写出协议，归并协议一处实现、两条指标流共用。
"""

from typing import TYPE_CHECKING, Protocol, Sequence

from cynosure.distributed.process import DistributedContext

if TYPE_CHECKING:
    # 运行时不依赖 train 包（cynosure.train 的初始化会拉起本包，反向
    # import 造成包级循环）：事件契约的类型仅作注解（train 侧
    # RunArtifacts 对本模块的依赖方向不变——它构造 EventMerger）
    from cynosure.train.artifacts import (
        IterEvent,
        OverfitAlertEvent,
    )


class EventSink(Protocol):
    """指标流的写出协议（归并器的落盘面）：``append_event(event)``
    单事件追加、事件类型混存同一 metrics.jsonl。train 的
    ``RunArtifacts`` 与预训练的 ``PretrainRun`` 各自满足——归并器对
    写者面无 train/pretrain 之分。"""

    def append_event(self, event: "IterEvent | OverfitAlertEvent") -> None:
        """向本 rank 的指标流追加一行 JSON 事件（仅 rank 0 被归并器
        触发；协议面只约束方法形态，混存事件类型集随写者而异）。"""
        ...


class EventMerger:
    """训练指标流的 rank 0 归并器（emit = 提交 + 归并追加）。"""

    def __init__(
        self, context: DistributedContext, artifacts: "EventSink",
    ) -> None:
        self._context = context
        self._artifacts = artifacts

    def emit(self, events: Sequence["IterEvent | OverfitAlertEvent"]) -> None:
        """本 rank 的事件清单提交归并（rank 0 按 rank 序追加写出）。

        事件清单内序保持：train 侧同 iteration 的 ``overfit_alert`` 排
        在本 rank iter 事件之后，预训练侧每更新步至多一条告警——归并
        序 = (iteration/步, rank, 清单内序)，分叉告警与其数值来源相邻
        可读。调用方负责清单外的直写序（预训练相 pretrain 事件先于本
        归并段落盘，写出序 = 步序 + 步内 pretrain 先于告警）。"""
        gathered = self._context.gather(list(events))
        if self._context.rank == 0:
            for source in gathered:
                for item in source:
                    self._artifacts.append_event(item)
