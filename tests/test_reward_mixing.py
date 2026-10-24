"""FakeSourceInterleave 的行为锚（ADR-0020 双源交错规则的单一实现）。

数值锚与在线 DiscriminatorWindow 的既有指派族同源（1/3、1/49 的单位
分数邻域、snapping 分辨率下界、极小 ρ 回退）——规则从窗口构造搬出为
独立策略件后，pretrain 侧（#280）与在线侧消费同一份解析，本文件钉
解析行为本身；消费侧行为由各自文件既有测试背书（指派序列不变）。
"""

import math
from fractions import Fraction

import pytest

from cynosure.reward.mixing import FakeSourceInterleave


class TestFakeSourceInterleave:
    def test_zero_ratio_means_never_rollout(self) -> None:
        """ρ=0：interval None、任意位置恒 recon（默认口径的显式形态——
        消费方据此跳过 rollout 供给面的装配）。"""
        rule = FakeSourceInterleave(0.0)
        assert rule.interval is None
        assert all(not rule.is_rollout(position) for position in range(50))

    @pytest.mark.parametrize(("ratio", "interval"), [
        (0.5, 2),
        (0.25, 4),
        (0.3, 4),   # ⌈1/0.3⌉ = ⌈1/0.25⌉（非整除倒数取顶）
        (0.6, 2),
        (1 / 3, 3),   # double 略低于 1/3：snapping 保语义
        (1 / 49, 49),  # 浮点除法 1.0/ρ=49.00…01 取顶得 50，snapping 得 49
    ])
    def test_interval_resolution_matches_window_anchors(
        self, ratio: float, interval: int,
    ) -> None:
        """interval 解析 = 窗口构造的既有数值锚逐项平移（snapping 后
        倒数取顶）。"""
        assert FakeSourceInterleave(ratio).interval == interval

    def test_is_rollout_hits_every_interval_th_position(self) -> None:
        """交错判定：位置序 i（0 基）满足 (i+1) % interval == 0 者为
        rollout——每 interval 个位置恰一个、落在周期末位。"""
        rule = FakeSourceInterleave(0.25)
        assert [rule.is_rollout(i) for i in range(8)] == [
            False, False, False, True,
        ] * 2

    def test_tiny_ratio_below_snap_resolution_falls_back(self) -> None:
        """ρ=1e-7 < snapping 分辨率：退回浮点除法取顶不崩（interval
        极大 → 实际窗口内零 rollout）；小到倒数溢出则可读拒绝。"""
        assert FakeSourceInterleave(1e-7).interval == math.ceil(1.0 / 1e-7)
        with pytest.raises(ValueError, match="ρ"):
            FakeSourceInterleave(1e-320)

    @pytest.mark.parametrize("bad", [1.0, -0.1, float("nan"), float("inf")])
    def test_rejects_ratio_out_of_domain(self, bad: float) -> None:
        """ρ 物理域 [0,1)（ρ=1 纯 rollout 退化面为开区间）的构造期
        防御——与在线 DiscriminatorWindow 直调防御同款拒绝。"""
        with pytest.raises(ValueError, match="ρ"):
            FakeSourceInterleave(bad)

    def test_interval_property_matches_unit_fraction_semantics(self) -> None:
        """语义钉：用户写 1/n 即每 n 个位置一个 rollout——interval 的
        Fraction 精确路径与取顶口径一致（snapping 文档锚）。"""
        for n in (2, 3, 4, 5, 7, 49, 97):
            assert FakeSourceInterleave(1 / n).interval == n
        assert Fraction(1) / Fraction(1 / 49) > 49  # 精确算术失守的证据
