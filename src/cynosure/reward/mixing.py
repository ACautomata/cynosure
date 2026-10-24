"""判别器 fake 双源的确定性交错规则（ADR-0020 的策略件单一实现）。

ρ（``reward.disc_rollout_fake_ratio``）→ 交错周期 interval 的解析与
位置判定收拢于此：在线窗口任务指派（``DiscriminatorWindow``，#277）
与预训练 fake 供给（#280）两消费方共用同一份解析——双源并存语义的
数值口径（snapping、取顶、域防御）单点承载，消费方不复制。

指派 = 创建序位置上的确定性交错（每 interval 个位置一个 rollout 型，
零随机流）；ρ=0（默认）恒 recon——消费方据此跳过 rollout 供给面的
装配（rollout 流保持零消耗，与现状逐位一致）。
"""

import math
from fractions import Fraction


class FakeSourceInterleave:
    """fake 双源（recon 同源重构 | rollout 终点）的位置交错规则。

    interval = ⌈1/ρ̂⌉，ρ̂ = 配置浮点 snapping 到最近简单有理
    （``limit_denominator`` 1e6 界）——裸浮点路径在此邻域全部失守：
    精确有理算术对 double(1/3)（< 1/3）取顶得 4，1.0/ρ 浮点除法对
    double(1/49) 取顶得 50，均非用户所写的 1/n 语义。ρ < snapping
    分辨率（≈5e-7）时退回浮点除法取顶（用户写的就是小量本身）；小到
    倒数溢出 = 无可行窗口尺度，可读拒绝。
    """

    def __init__(self, rollout_fake_ratio: float) -> None:
        if not math.isfinite(rollout_fake_ratio) or not (
            0.0 <= rollout_fake_ratio < 1.0
        ):
            raise ValueError(
                f"判别器 fake 双源的 rollout 型混合比 ρ 须 ∈ [0, 1)，得到 "
                f"{rollout_fake_ratio}（ρ=1 为纯 rollout 终点的退化面，"
                "见 ADR-0020）"
            )
        self._interval: int | None = None
        if rollout_fake_ratio != 0.0:
            snapped = Fraction(rollout_fake_ratio).limit_denominator(10**6)
            if snapped > 0:
                self._interval = math.ceil(1 / snapped)
            else:
                reciprocal = 1.0 / rollout_fake_ratio
                if not math.isfinite(reciprocal):
                    raise ValueError(
                        f"判别器 fake 双源的 rollout 型混合比 ρ 过小，交错"
                        f"周期 ⌈1/ρ⌉ 不可行：ρ = {rollout_fake_ratio}"
                    )
                self._interval = math.ceil(reciprocal)

    @property
    def interval(self) -> int | None:
        """交错周期（⌈1/ρ⌉）；ρ=0 为 None——消费方 rollout 供给面的
        装配开关（None = 恒 recon，rollout 流零消耗）。"""
        return self._interval

    def is_rollout(self, position: int) -> bool:
        """创建序位置（0 基）是否 rollout 型：每 interval 个位置一个、
        落在周期末位（(position+1) % interval == 0）；ρ=0 恒 False。
        """
        interval = self._interval
        return interval is not None and (position + 1) % interval == 0
