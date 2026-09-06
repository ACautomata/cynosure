"""数据工件契约的装载层校验测试（issue #51 review P2）。

工件由 prepare 落盘、train/eval 装载消费：非有限值（NaN/±inf）在装载层
显式拒绝，而不是穿透到 rollout 条件张量污染模型激活（与 ChannelStats
的 allow_inf_nan=False 同一防线）。"""

import pytest
from pydantic import ValidationError

from cynosure.reward.artifacts import PoolEntry


def pool_entry(spacing: tuple[float, float, float]) -> PoolEntry:
    return PoolEntry(
        case_id="BraTS-GLI-00000-000",
        modality="t1n",
        latent="real_pool_latents/entry.pt",
        spacing=spacing,
    )


class TestPoolEntrySpacingContract:
    def test_rejects_non_finite_spacing(self) -> None:
        """NaN 与 ±inf 均显式拒绝：NaN 与 +inf 不满足 ≤0 判断、会穿透
        正数校验，须由 allow_inf_nan=False 在装载层守住。"""
        for component in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError, match="spacing"):
                pool_entry((component, 100.0, 100.0))

    def test_accepts_finite_positive_spacing(self) -> None:
        pool_entry((100.0, 100.0, 100.0))
