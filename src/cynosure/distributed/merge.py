"""iter 事件的 rank 0 归并写出（orchestration 章「指标由 rank 0 归并写
出」）。

每个 RL iteration，各 rank 提交自己的 iter 事件（rollout 条件/噪声各
rank 独立 → 事件数值各 rank 不同），gather 到 rank 0 后按源 rank 升序
追加——同一 iteration 的 N 条事件连续排列且 rank 升序（无重复、无丢
失、顺序稳定：gather 的接收列表按源 rank 排列，归并序天然稳定）。
写盘只发生在 rank 0（RunArtifacts 的 rank 0 写盘契约）；单进程路径
= world-1 退化（gather 恒等、直写）。
"""

from typing import TYPE_CHECKING

from cynosure.distributed.process import DistributedContext

if TYPE_CHECKING:
    # 运行时不依赖 train 包（cynosure.train 的初始化会拉起本包，反向
    # import 造成包级循环）：事件与产物契约的类型仅作注解
    from cynosure.train.artifacts import IterEvent, RunArtifacts


class EventMerger:
    """训练指标流的 rank 0 归并器（emit = 提交 + 归并追加）。"""

    def __init__(
        self, context: DistributedContext, artifacts: "RunArtifacts",
    ) -> None:
        self._context = context
        self._artifacts = artifacts

    def emit(self, event: "IterEvent") -> None:
        """本 rank 的 iteration 事件提交归并（rank 0 追加写出）。"""
        gathered = self._context.gather([event])
        if self._context.rank == 0:
            for source in gathered:
                for item in source:
                    self._artifacts.append_event(item)
