"""条件白名单的运行时对象（ADR-0008 决策 5/7：RM readiness gate 产物
在 train 侧的载体）。

预训练报告的 ``gate_whitelist`` 是名单的生产侧（ADR-0008-04）；本对象
是 train 侧的运行时消费面——readiness gate 的上岗判定与 train 循环的
逐 iteration 查询（门控消费票）读同一实例，保证「上岗名单」与「更新
开关」永不分叉。名单内条件正常更新，名单外条件由门控票跳过 policy
更新（rollout 与判别器更新照常）；动态恢复（ADR-0008 决策 8，EMA
驱动的名单变更）由门控票在本对象上扩展，本票交付静态接线。

resume 语境（``unrestricted``）：恢复点不重查白名单（续训状态已含
判别器全量状态），白名单占位为全条件放行；名单的跨 run 持久化随
门控消费票交付。
"""

from types import MappingProxyType
from typing import Mapping

from cynosure.config import MODALITIES, Modality
from cynosure.pretrain.artifacts import PretrainReport


class ConditionWhitelist:
    """条件白名单（CONTEXT.md「条件白名单」词条的运行时对象）：名单
    成员查询的单点来源。不可变值对象——名单变更（动态恢复）由门控票
    以显式的替换操作交付，不做原地突变。"""

    def __init__(
        self,
        members: tuple[Modality, ...],
        measured: Mapping[Modality, float],
    ) -> None:
        self._members = members
        self._measured = MappingProxyType(dict(measured))

    @classmethod
    def from_report(cls, report: PretrainReport) -> "ConditionWhitelist":
        """train 新 run 语境：名单 = 报告白名单（gate 产物，轮转序），
        并携带 per-condition 实测快照（白名单空时的拒绝报错与诊断
        消费）。"""
        return cls(tuple(report.gate_whitelist), report.condition_auc)

    @classmethod
    def unrestricted(cls) -> "ConditionWhitelist":
        """resume 占位：全条件放行（恢复点不重查白名单——issue #88；
        Modality 全集即「不设条件闸」的表达）。预训练 driver 冷启动
        装配同此占位（driver 自产 per-condition 判定，不消费 train 侧
        名单）。无实测快照（报告未装载）。"""
        return cls(tuple(MODALITIES), {})

    def __contains__(self, modality: Modality) -> bool:
        """逐 iteration 查询面：该条件的 policy 更新当前是否放行。"""
        return modality in self._members

    def __len__(self) -> int:
        return len(self._members)

    @property
    def members(self) -> tuple[Modality, ...]:
        """名单成员只读视图（顺序 = 报告产出的轮转序）。"""
        return self._members

    @property
    def measured(self) -> Mapping[Modality, float]:
        """per-condition 实测快照（报告 ``condition_auc``——白名单空时
        拒绝报错的实测值来源；``unrestricted`` 占位为空）。"""
        return self._measured
