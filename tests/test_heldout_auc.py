"""held-out AUC 信号测试（ticket #20 AC）：rank 统计数值口径（tie-aware）、
held-out 与训练 real 不相交语义、kind 守卫。

AUC = Mann-Whitney U 口径：real 分数高于 fake 分数的配对占比（并列 0.5）。
held-out real 永不参与判别器更新（prepare 工件 + kind 守卫保证），
AUC 因此是 out-of-sample 的 hacking 监控信号。
"""

import time

import pytest
import torch

from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.artifacts import LatentManifest

from tests.test_online_update import UpdateScenario


@pytest.fixture
def scenario(tmp_path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class TestAucNumericContract:
    def test_all_real_scores_above_fake(self) -> None:
        """real 全胜 → AUC 1。"""
        auc = HeldOutAuc.auc_from_scores(
            torch.tensor([1.0, 2.0]), torch.tensor([-1.0, 0.0]),
        )
        assert auc == pytest.approx(1.0)

    def test_all_real_scores_below_fake(self) -> None:
        """real 全负 → AUC 0（判别器倒挂信号）。"""
        auc = HeldOutAuc.auc_from_scores(
            torch.tensor([-1.0]), torch.tensor([1.0]),
        )
        assert auc == pytest.approx(0.0)

    def test_mixed_scores(self) -> None:
        """配对口径：1.0 胜 2 平 0；0.0 胜 0 平 0 → (2 + 0) / 4 = 0.5。"""
        auc = HeldOutAuc.auc_from_scores(
            torch.tensor([1.0, 0.0]), torch.tensor([0.5, 0.5]),
        )
        assert auc == pytest.approx(0.5)

    def test_ties_count_half(self) -> None:
        """并列各计 0.5：1.0 胜 2；0.5 平 2 → (2 + 1) / 4 = 0.75。"""
        auc = HeldOutAuc.auc_from_scores(
            torch.tensor([0.5, 1.0]), torch.tensor([0.5, 0.5]),
        )
        assert auc == pytest.approx(0.75)

    def test_result_is_scalar_float(self) -> None:
        auc = HeldOutAuc.auc_from_scores(
            torch.tensor([1.0]), torch.tensor([0.0]),
        )
        assert isinstance(auc, float)


class PairwiseAucReference:
    """配对枚举口径的 AUC 参考实现（O(n·m) 朴素循环，等价性锚定用）。"""

    @staticmethod
    def compute(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> float:
        wins = 0.0
        pairs = 0
        for real in real_scores.tolist():
            wins += sum(
                1.0 if real > fake else 0.5 if real == fake else 0.0
                for fake in fake_scores.tolist()
            )
            pairs += fake_scores.numel()
        return wins / pairs


class TestRankStatisticEquivalence:
    """秩统计实现（排序 midrank）与配对枚举口径严格等价。

    配对枚举在生产 patch 规模（数百 latent × 2048 patch/latent → ~5e11
    配对）下工作量达 TB 级内存或百秒级 CPU——监控不得支配训练；秩实现
    O((n+m)·log(n+m))，tie 取平均秩与「并列计 0.5」严格等价。"""

    def test_random_scores_match_pairwise_reference(self) -> None:
        generator = torch.Generator().manual_seed(0)
        for real_size, fake_size in ((1, 1), (7, 13), (37, 53), (128, 91)):
            real = torch.randn(real_size, generator=generator)
            fake = torch.randn(fake_size, generator=generator)
            assert HeldOutAuc.auc_from_scores(real, fake) == pytest.approx(
                PairwiseAucReference.compute(real, fake), abs=1e-12,
            )

    def test_cross_group_tie_blocks_match_pairwise_reference(self) -> None:
        """并列块跨 real/fake 边界（同一值同时出现在两侧）时 tie 各计
        0.5 的口径不变——midrank 平均秩的等价性锚定。"""
        real = torch.tensor([1.0, 2.0, 2.0, 3.0])
        fake = torch.tensor([2.0, 2.0, 4.0])
        assert HeldOutAuc.auc_from_scores(real, fake) == pytest.approx(
            PairwiseAucReference.compute(real, fake), abs=1e-12,
        )

    def test_all_identical_scores_are_chance(self) -> None:
        """全并列 → 每对 0.5 → AUC 0.5（chance）。"""
        scores = torch.full((6,), 0.5)
        auc = HeldOutAuc.auc_from_scores(scores[:3], scores[3:])
        assert auc == pytest.approx(0.5)

    def test_patch_scale_input_completes_via_ranking(self) -> None:
        """生产 patch 规模（2e5 × 2e5 = 4e10 配对）在秒级完成——配对
        枚举实现同输入需 ~4e10 元素差矩阵计算（实测 ~25s），监控不得
        支配训练；排序秩实现与枚举实现相差三个数量级。"""
        generator = torch.Generator().manual_seed(1)
        real = torch.randn(200_000, generator=generator)
        fake = torch.randn(200_000, generator=generator)
        started = time.monotonic()
        auc = HeldOutAuc.auc_from_scores(real, fake)
        elapsed = time.monotonic() - started
        assert 0.0 < auc < 1.0
        assert elapsed < 5.0  # 配对枚举实测 ~25s（多核），排序 <10ms


class TestHeldOutSemantics:
    def test_heldout_manifest_required_kind(self, scenario: UpdateScenario) -> None:
        """kind 守卫：real_pool manifest 不得当 held-out 使用——
        否则 AUC 失去 out-of-sample 语义（装配层守住）。"""
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        with pytest.raises(ValueError, match="heldout"):
            HeldOutAuc(
                heldout_manifest=pool, scorer=scenario.scorer(),
                generator=scenario.generator(1),
            )
