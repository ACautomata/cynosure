"""GRPO policy loss：ratio clip 1e-4（定死）的 clipped surrogate。

ratio = exp(log π_new − log π_old)；loss = −mean(min(ratio·A,
clamp(ratio, 1−ε, 1+ε)·A))。无 KL、无参考模型（spec #15：极窄 trust
region 是无 KL 时的主要稳定器）。
"""

import pytest
import torch

from cynosure.grpo import ClippedPolicyLoss

CLIP_RANGE = 1e-4


class TestClippedPolicyLoss:
    @pytest.fixture
    def loss_fn(self) -> ClippedPolicyLoss:
        return ClippedPolicyLoss(clip_range=CLIP_RANGE)

    def test_identical_distributions_zero_loss(self, loss_fn: ClippedPolicyLoss) -> None:
        """同分布（ratio=1）：surrogate = −A 的均值（策略无位移、无 clip）。"""
        old = torch.tensor([-1.2, 0.4, 2.0])
        advantage = torch.tensor([1.0, -0.5, 2.0])
        loss = loss_fn.loss(old.clone(), old, advantage)
        assert loss.item() == pytest.approx(-advantage.mean().item(), rel=1e-6)

    def test_gradient_flows_through_new_log_probs_only(
        self, loss_fn: ClippedPolicyLoss,
    ) -> None:
        """π_old 是常数（rollout 记录）：梯度只流经 π_new 侧。"""
        old = torch.tensor([-1.0], requires_grad=False)
        new = torch.tensor([-1.0], requires_grad=True)
        advantage = torch.tensor([1.0])
        loss_fn.loss(new, old, advantage).backward()
        assert new.grad is not None
        assert new.grad.abs().item() > 0.0

    def test_positive_advantage_pulls_ratio_up_before_clip(
        self, loss_fn: ClippedPolicyLoss,
    ) -> None:
        """正 advantage、ratio 微升（未出 clip 窗）：loss = −ratio·A（未截断）。"""
        delta = torch.tensor(5e-5)  # ratio ≈ 1 + 5e-5，窗内（1e-4）
        old = torch.tensor([0.0])
        new = old + delta
        advantage = torch.tensor([1.0])
        loss = loss_fn.loss(new, old, advantage)
        assert loss.item() == pytest.approx(-(1.0 + 5e-5), rel=1e-6)

    def test_clip_caps_positive_advantage_branch(
        self, loss_fn: ClippedPolicyLoss,
    ) -> None:
        """正 advantage、ratio 超窗：surrogate 被钳在 (1+ε)·A（clip 生效）。"""
        old = torch.tensor([0.0])
        new = torch.tensor([0.01])  # ratio ≈ 1.01 ≫ 1+1e-4
        advantage = torch.tensor([1.0])
        loss = loss_fn.loss(new, old, advantage)
        assert loss.item() == pytest.approx(-(1.0 + CLIP_RANGE), rel=1e-6)

    def test_clip_caps_negative_advantage_branch(
        self, loss_fn: ClippedPolicyLoss,
    ) -> None:
        """负 advantage、ratio 超窗下探：min 取被钳的 (1−ε)·A（更小者），
        loss 为正——阻止策略沿负 advantage 方向继续下探。"""
        old = torch.tensor([0.0])
        new = torch.tensor([-0.01])  # ratio ≈ 0.99 ≪ 1−1e-4
        advantage = torch.tensor([-1.0])
        loss = loss_fn.loss(new, old, advantage)
        assert loss.item() == pytest.approx(1.0 - CLIP_RANGE, rel=1e-6)

    def test_mean_over_group_directions(self, loss_fn: ClippedPolicyLoss) -> None:
        """G 个方向逐方向 min 后取组内均值（每训练步一个标量 loss）。"""
        old = torch.zeros(12)
        new = torch.zeros(12)
        advantage = torch.linspace(-1.0, 1.0, 12)
        loss = loss_fn.loss(new, old, advantage)
        assert loss.ndim == 0
        assert loss.item() == pytest.approx(-advantage.mean().item(), rel=1e-6)

    def test_shape_mismatch_rejected(self, loss_fn: ClippedPolicyLoss) -> None:
        with pytest.raises(ValueError, match="形状"):
            loss_fn.loss(torch.zeros(3), torch.zeros(4), torch.zeros(3))
