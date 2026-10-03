"""卡轴连续段分片（#221 决议 8：现行 ``_shard_bounds`` 的纯函数化）。

测量批与 real 侧共用的切片计划：全量 V 项按卡轴切 D 段连续区间，
前余均分（排位 < V % D 的卡多领一项）——连续段（非条带）使主控按
卡序拼接各卡产出**还原全量排列序**（全局 AUC 的卷级归属前提）；
σ 轮转按全量位次即「全量 σ 列表切片，零偏移算术」。K×D 项切 D 段
= base K rem 0（每卡恰 K——real 侧每卡配额的等分特例）。

计划纯函数（AllocationTable / DiscriminatorWindow 先例）：无状态、
可重导出，单进程与多卡同一条公式——单卡（D=1）退化为全量一段。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShardPlan:
    """一次分片的计划值：逐卡的 ``(start, stop)`` 连续段（半开区间）。"""

    bounds: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError("分片计划须至少一段（卡数 ≥ 1）")
        previous_stop = 0
        for index, (start, stop) in enumerate(self.bounds):
            if start != previous_stop or stop < start:
                raise ValueError(
                    f"分片段 {index} = ({start}, {stop}) 不与前段连续"
                    f"（前段终点 {previous_stop}）——卡轴连续段计划损坏"
                )
            previous_stop = stop

    @property
    def total(self) -> int:
        """全量项数（各段长度之和）。"""
        return self.bounds[-1][1]

    def slice_of(self, card: int) -> tuple[int, int]:
        """本卡的连续段边界。"""
        return self.bounds[card]

    @classmethod
    def split(cls, total: int, cards: int) -> "ShardPlan":
        """前余均分的卡轴连续段计划：base = V // D、余数 r = V % D，
        前 r 卡各领 base+1 项、其余各领 base 项。``total`` 须 ≥ 0；
        末段为空（total < cards）合法——空段的消费面（测量批切片）
        由装配期「每条件 held-out ≥ 卡数」守卫把住，正常路径不达。
        """
        if total < 0:
            raise ValueError(f"全量项数须 ≥ 0，得到 {total}")
        if cards < 1:
            raise ValueError(f"分片卡数须 ≥ 1，得到 {cards}")
        base, remainder = divmod(total, cards)
        bounds = []
        start = 0
        for card in range(cards):
            stop = start + base + (1 if card < remainder else 0)
            bounds.append((start, stop))
            start = stop
        return cls(bounds=tuple(bounds))
