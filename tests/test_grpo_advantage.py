"""GRPO 核心：MGAI advantage（组内标准化 → 跨 λ 求和 → clamp ±5）。

参考实现顺序（spec #15 / research/granular-grpo.md §4）：对每 (k, λ) 的
组内 reward 各自标准化 (r − mean)/(std + 1e-8)，再对 λ 求和、求和后统一
clamp ±5——顺序是「先标准化、再求和、最后截断」，不是逐 λ 截断后再求和。
"""

import pytest
import torch

from cynosure.grpo import MgaiAdvantage

GROUP_SIZE = 12  # G 保持 12（fixture 策略：G=2 会使标准化退化为恒 ±1）


class TestMgaiAdvantage:
    """MGAI：各 Granularity λ 的 advantage 组内标准化后求和、clamp ±5。"""

    @pytest.fixture
    def advantage(self) -> MgaiAdvantage:
        return MgaiAdvantage(clamp=5.0)

    def test_single_lambda_is_group_normalized(
        self, advantage: MgaiAdvantage,
    ) -> None:
        """单 λ：advantage = (r − mean)/(std + 1e-8)（组内标准化）。"""
        torch.manual_seed(0)
        rewards = torch.randn(GROUP_SIZE)
        outcome = advantage.compute({1: rewards})
        expected = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert outcome.shape == (GROUP_SIZE,)
        assert torch.allclose(outcome, expected, atol=1e-7)

    def test_lambda_sums_after_individual_normalization(
        self, advantage: MgaiAdvantage,
    ) -> None:
        """多 λ：各 λ 先组内标准化、再跨 λ 求和（对 advantage 求和，
        缓解不同 λ 的 latent 分数校准差异——不对原始 reward 求和）。"""
        torch.manual_seed(1)
        first = torch.randn(GROUP_SIZE)
        second = torch.randn(GROUP_SIZE) * 100 + 50  # 跨 λ 校准差刻意放大
        outcome = advantage.compute({1: first, 2: second})
        normalized_first = (first - first.mean()) / (first.std() + 1e-8)
        normalized_second = (second - second.mean()) / (second.std() + 1e-8)
        assert torch.allclose(
            outcome, normalized_first + normalized_second, atol=1e-6,
        )

    def test_clamp_applies_after_summation(
        self, advantage: MgaiAdvantage,
    ) -> None:
        """clamp 在跨 λ 求和之后统一截断（参考实现顺序）：单 λ 各自不超界、
        同向 λ 求和后超界 → 求和结果被 ±5 截断而非逐 λ 截断。"""
        # 重尾分布（11 个同值 + 1 个偏移）：单 λ 标准化后 |z| ≈ 3.18 < 5 不触界
        same_direction = torch.cat((torch.zeros(11), torch.tensor([1.0])))
        rewards = {lam: same_direction for lam in (1, 2, 3)}
        outcome = advantage.compute(rewards)
        single = (same_direction - same_direction.mean()) / (
            same_direction.std() + 1e-8
        )
        assert single.abs().max() < 5.0  # 单 λ 标准化后不触界
        assert outcome[-1] == 5.0  # 求和后超界的方向被统一截断（3 × 3.18 → +5）
        assert outcome[:-1].abs().max() < 5.0  # 未超界方向保持求和值

    def test_clamp_leaves_typical_advantage_untouched(
        self, advantage: MgaiAdvantage,
    ) -> None:
        torch.manual_seed(2)
        rewards = {lam: torch.randn(GROUP_SIZE) for lam in (1, 2)}
        outcome = advantage.compute(rewards)
        unclamped = sum(
            (r - r.mean()) / (r.std() + 1e-8) for r in rewards.values()
        )
        touched = (unclamped.abs() >= 5.0).any()
        assert torch.isfinite(outcome).all()
        if not touched:
            assert torch.allclose(outcome, unclamped, atol=1e-6)

    def test_g12_normalization_is_not_degenerate(
        self, advantage: MgaiAdvantage,
    ) -> None:
        """G=12 下组内标准化非退化：advantage 有真实的方向差异、非恒 ±1
        （fixture 策略：G=2 会使 (r−mean)/std 退化为恒 ±1）。"""
        torch.manual_seed(3)
        rewards = {1: torch.randn(GROUP_SIZE), 2: torch.randn(GROUP_SIZE)}
        outcome = advantage.compute(rewards)
        assert outcome.unique().numel() > 2  # 远超 ±1 两值
        assert outcome.std() > 0.5

    def test_near_constant_rewards_survive_via_epsilon(
        self, advantage: MgaiAdvantage,
    ) -> None:
        """std→0（组合场压缩方向差异）时 1e-8 保护兜底：不产生 Inf/NaN。"""
        rewards = {1: torch.full((GROUP_SIZE,), 3.0)}
        outcome = advantage.compute(rewards)
        assert torch.isfinite(outcome).all()
        assert torch.equal(outcome, torch.zeros(GROUP_SIZE))

    def test_empty_lambda_set_rejected(self, advantage: MgaiAdvantage) -> None:
        with pytest.raises(ValueError, match="Granularity"):
            advantage.compute({})

    def test_group_size_mismatch_rejected(self, advantage: MgaiAdvantage) -> None:
        with pytest.raises(ValueError, match="组内"):
            advantage.compute({1: torch.randn(3), 2: torch.randn(4)})

    def test_advantage_follows_reward_device(self, advantage: MgaiAdvantage) -> None:
        """累加器随 reward 的 device/dtype 初始化（new_zeros 语义）：加速器
        resident 的判别器 reward 进 compute 时，CPU 零向量与之相加会跨设备
        报错（cuda/HIP 训练路径在首个 policy update 前即中断）——输出须
        与输入同 device。fixture 测试面 CPU-only，meta device 锁同一装配
        契约（跨设备算术被 torch 拒绝的语义与 cuda 一致）。"""
        rewards = {1: torch.randn(GROUP_SIZE, device="meta")}
        outcome = advantage.compute(rewards)
        assert outcome.device.type == "meta"
        assert outcome.dtype == rewards[1].dtype
