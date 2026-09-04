"""held-out AUC 信号测试（ticket #20 AC）：rank 统计数值口径（tie-aware）、
held-out 与训练 real 不相交语义、kind 守卫。

AUC = Mann-Whitney U 口径：real 分数高于 fake 分数的配对占比（并列 0.5）。
held-out real 永不参与判别器更新（prepare 工件 + kind 守卫保证），
AUC 因此是 out-of-sample 的 hacking 监控信号。
"""

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

    def test_chunked_matches_unchunked_reference(self) -> None:
        """分块配对计算与不分块参考逐位一致（生产 337 fake × 2048
        patch/latent 的 n×m 配对矩阵若整块分配约需 TB 级内存——第一次
        iteration 即 OOM；配对统计数学上可分块精确累加）。tie 跨块边界
        语义不变（并列计 0.5）。"""
        generator = torch.Generator().manual_seed(0)
        real = torch.randn(37, generator=generator)
        fake = torch.randn(53, generator=generator)
        reference = HeldOutAuc.auc_from_scores(real, fake, chunk_elements=10**9)
        for chunk in (1, 3, 37, 38, 10**9):
            assert HeldOutAuc.auc_from_scores(
                real, fake, chunk_elements=chunk,
            ) == reference

    def test_chunked_large_input_stays_bounded(self) -> None:
        """大输入（1e5 real × 1e4 fake = 1e9 配对）以固定块内存跑完——
        不分块实现此输入需 ~4 GB 差异矩阵（生产规模则会更大）。"""
        generator = torch.Generator().manual_seed(1)
        real = torch.randn(100_000, generator=generator)
        fake = torch.randn(10_000, generator=generator)
        auc = HeldOutAuc.auc_from_scores(real, fake, chunk_elements=2**20)
        assert 0.0 <= auc <= 1.0


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
