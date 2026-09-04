"""GRPO policy loss：ratio clip 1e-4 极窄 trust region（spec #15）。

PPO 式 clipped surrogate（无 KL、无参考模型——clip 是无 KL 时的主要
稳定器）：

    ratio = exp(log π_new − log π_old)
    loss  = −mean( min(ratio·A, clamp(ratio, 1−ε, 1+ε)·A) )

ε = 1e-4（config 定死，schema 等值 validator 拒绝改动）。
"""

import torch


class ClippedPolicyLoss:
    """ratio clip 的 GRPO policy loss（PPO clipped surrogate，负号取最小项）。"""

    def __init__(self, clip_range: float) -> None:
        self._clip_range = clip_range

    def loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """单训练步 k 的标量 loss（G 个方向的组内均值）。"""
        if new_log_probs.shape != old_log_probs.shape:
            raise ValueError(
                f"π_old/π_new 形状不符: {tuple(old_log_probs.shape)} vs "
                f"{tuple(new_log_probs.shape)}",
            )
        ratio = (new_log_probs - old_log_probs).exp()
        surrogate = ratio * advantages
        clipped = ratio.clamp(
            min=1.0 - self._clip_range, max=1.0 + self._clip_range,
        ) * advantages
        return -torch.min(surrogate, clipped).mean()
