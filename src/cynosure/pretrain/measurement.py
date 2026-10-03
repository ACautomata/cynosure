"""主控单点测量模板（#221 决议 5/9：预训练测量批的随机面原语）。

每次测量：主控把模板复位到同一状态（``schedule.seed + 19`` **显式
直锚**——不再经 ``initial_seed()+10`` 的实现缝派生，卡轴化 recon 流
下该缝会静默随卡漂移）→ 条件构造一次 → 按卡序逐卡抽本地 ε 行。
``volume_offset`` 前缀消耗机制整体退役：分段抽 ≡ 全量对应行由
float32 ``randn`` 的 16 元素块 Box-Muller 顺序流性质承载（行宽
≡ 0 (mod 16) 的装配期断言是它的机器前提，见
``LatentManifest.assert_measurement_row_width``），等价性由锚守护
（#198 三组锚平移宿主）而非结构性事实。

测量模板不回 RNG 注册表（测量不参与续训——现行口径）；与 recon 流
解耦（装配原语不再持测量面）：同一 ``ReconstructionAssembler`` 的
卡轴 recon 流只服务更新批 σ/ε。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from cynosure.policy.condition import RolloutCondition
from cynosure.policy.schedules import ConditionSchedules

if TYPE_CHECKING:
    from cynosure.train.rollout import ConditionSampler

    from cynosure.pretrain.sharding import ShardPlan


MEASUREMENT_TEMPLATE_OFFSET = 19
"""测量模板相对 ``schedule.seed`` 的显式直锚偏移（#221 决议 9：数值
与现行 ``shared_seed+19`` 恒等；偏移登记见 ``train/rng`` 模块的
偏移布局权威——+19 的宿主自本类起为主控测量面单点）。"""


@dataclass(frozen=True)
class MeasurementDraw:
    """一次测量的主控产出：条件 + 全量 σ 列表 + 按卡序的 ε 段。

    ``noises[card]`` = 卡 ``card`` 连续段的 ε（CPU 张量——主控
    generator 的生成域；任务侧迁移到本卡设备）。``sigmas`` 是全量位
    次列表（σ 切片 = ``sigmas[start:stop]``，零偏移算术）。
    """

    condition: RolloutCondition
    sigmas: tuple[float, ...]
    noises: tuple[torch.Tensor, ...]

    def sigmas_of(
        self, card: int, plan: "ShardPlan",
    ) -> tuple[float, ...]:
        """本卡连续段的 σ 切片（``ShardPlan.slice_of`` 的半开区间切片）。"""
        start, stop = plan.slice_of(card)
        return self.sigmas[start:stop]


class MeasurementTemplate:
    """预训练测量批的主控随机面（复位模板 + 条件构造 + 逐卡 ε 序列）。

    消费序（与现行 ``measure_condition`` 同构）：复位 → 条件构造（穿
    模板流——组2 的源对/源条目抽取消耗本流，次序即重放锚）→ 全量 σ
    定序轮转列表 → 按卡序逐卡 ``randn``。逐卡顺序抽的拼接 ≡ 全量
    一次 ``randn`` 的对应行（行宽 mod 16 的顺序流等价性，锚守护）。
    """

    def __init__(
        self,
        seed: int,
        schedules: ConditionSchedules,
        conditions: "ConditionSampler",
        step_indices: Sequence[int],
    ) -> None:
        if not step_indices:
            raise ValueError("被优化步集合 M 不得为空（重构的 s 候选集）")
        if 0 in step_indices:
            raise ValueError(
                "被优化步集合 M 不得含日程下标 0（s≈1 最噪端是奇异点，"
                "天然排除于重构候选——ADR-0012 决策 2）"
            )
        self._schedules = schedules
        self._conditions = conditions
        self._step_indices: tuple[int, ...] = tuple(sorted(step_indices))
        # 模板只取状态、永不被推进——逐次测量复位到同一状态，测量输入
        # 与「本 run 此前测量过几次」无关（上岗判据可复算，ADR-0012
        # 决策 5）；seed+19 显式直锚（决议 9）
        self._template_state = (
            torch.Generator()
            .manual_seed(seed + MEASUREMENT_TEMPLATE_OFFSET)
            .get_state()
        )

    def draw(
        self,
        modality: str,
        shape: tuple[int, ...],
        plan: "ShardPlan",
    ) -> MeasurementDraw:
        """一次测量的主控抽取：``shape`` = 该条件单卷 latent 形状（含
        条件维；行宽 = numel，mod 16 断言在 manifest 装配面）、
        ``plan`` = 卡轴分片计划（ε 按其卡序逐段生成）。
        """
        cursor = self._schedules.cursor(modality)
        cursor.assert_reconstruction_candidates(self._step_indices, modality)
        candidates = [
            cursor.sigma_level(step) for step in self._step_indices
        ]
        measurement = torch.Generator()
        measurement.set_state(self._template_state)
        condition = self._conditions.sample_target(
            modality, generator=measurement,
        )
        sigmas = tuple(
            candidates[index % len(candidates)]
            for index in range(plan.total)
        )
        noises = tuple(
            torch.randn(
                (stop - start, *shape), generator=measurement,
            )
            for start, stop in plan.bounds
        )
        return MeasurementDraw(
            condition=condition, sigmas=sigmas, noises=noises,
        )

    def forward_count(self, modality: str, sigmas: Sequence[float]) -> int:
        """σ 序列的重构前向总数（定序轮转下的确定值——#171 AC5 成本
        口径的读数面：逐卷步数 = 日程步数 − 起点下标，恒严格小于
        ``num_steps``）。主控对全量 σ 列表调用一次 = 全局读数（分段
        求和与全量求和的加法结合恒等）。"""
        cursor = self._schedules.cursor(modality)
        return sum(
            cursor.num_steps - 1 - cursor.continue_start_index(sigma)
            for sigma in sigmas
        )

