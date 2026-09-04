"""MGAI advantage（Multi-Granularity Advantage Integration，glossary）。

参考实现顺序（research/granular-grpo.md §4，spec #15 GRPO 核心节）：

    A^{(i)} = clamp( Σ_λ normalize(r^{(·,λ)}) , −5, +5 )

各 Granularity λ 的组内 reward **各自**标准化 (r − mean)/(std + 1e-8) 后
跨 λ 求和，clamp 在求和之后统一截断（先标准化、再求和、最后截断——顺序
即「参考实现顺序」，不是逐 λ 截断后再求和）。对 advantage 而非原始 reward
求和，天然缓解不同 λ 的 latent 分数校准差异；std 的 1e-8 保护兜底组合场
压缩方向差异时的 std→0。
"""

import torch

_EPSILON = 1e-8
"""组内标准化的 std 保护项（spec：std 加 1e-8）。"""


class MgaiAdvantage:
    """MGAI：各 λ 组内标准化 → 跨 λ 求和 → clamp ±5（G²RPO Eq.11 + adv clip）。"""

    def __init__(self, clamp: float) -> None:
        self._clamp = clamp

    def compute(self, rewards_by_lambda: dict[int, torch.Tensor]) -> torch.Tensor:
        """per-λ 组内 reward（λ → [G]）→ 组内方向 advantage [G]。

        各 λ 的组大小须一致（同一组 G 个方向跨 λ 打分）；Λ 为空显式拒绝。
        """
        if not rewards_by_lambda:
            raise ValueError("MGAI 需要非空的 Granularity λ 集合")
        group_size = next(iter(rewards_by_lambda.values())).shape[0]
        if any(rewards.shape[0] != group_size for rewards in rewards_by_lambda.values()):
            raise ValueError("各 λ 的组内 reward 数须一致（同一组 G 个方向）")
        total = torch.zeros(group_size)
        for rewards in rewards_by_lambda.values():
            total = total + self._normalize(rewards)
        return total.clamp(min=-self._clamp, max=self._clamp)

    @staticmethod
    def _normalize(rewards: torch.Tensor) -> torch.Tensor:
        """组内标准化：(r − mean)/(std + 1e-8)（std 取无偏估计，G≥2）。"""
        return (rewards - rewards.mean()) / (rewards.std() + _EPSILON)
