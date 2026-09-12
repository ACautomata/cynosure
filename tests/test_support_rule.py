"""支撑度判定原语 + bootstrap CI 下界规则测试（issue #85 AC）。

ADR-0008 决策 6 的统计形态：条件 held-out 卷数 < 支撑度界（暂定 20）时，
过线判据从池化点估计改为 bootstrap CI 下界 ≥ 门槛；≥ 界维持点估计口径。
重采样单元是卷级聚类（每卷一组 patch 分数整卷进出）——patch 级打散把
同卷强相关 patch 当独立观测、低估 CI 宽度。

边界用例按真实数据形状构造：MRA ≈ 16 卷命中规则（bootstrap 判定）、
T2w ≈ 67 卷不命中规则（点估计判定）。
"""

import pytest
import torch

from cynosure.reward.auc import HeldOutAuc, VolumeScoreClusters
from cynosure.reward.support import (
    DEFAULT_REPLICATES,
    LOWER_QUANTILE,
    SupportRule,
    bootstrap_ci_lower_bound,
    bootstrap_replicates,
)

THRESHOLD = 0.65
SUPPORT_BOUND = 20


def perfect_chance_clusters(
    perfect: int, chance: int, patches: int = 4,
) -> VolumeScoreClusters:
    """「完美卷 + chance 卷」两档聚类：完美卷 patch 分数全 1.0、chance 卷
    全 0.0、fake 侧全 0.0——chance 卷与 fake 全并列（每对 0.5）。池化点
    估计 = 0.5 + 完美占比/2；bootstrap 分布解析可推（每次重复的完美卷
    抽中数 K ~ Binomial(卷数, 完美占比)，重复 AUC = 0.5 + K/(2·卷数)），
    单测的期望锚由此算出。"""
    volumes = tuple(
        [torch.full((patches,), 1.0)] * perfect
        + [torch.full((patches,), 0.0)] * chance
    )
    return VolumeScoreClusters(
        real_volume_scores=volumes,
        fake_scores=torch.zeros(patches),
    )


class TestDecideDispatch:
    """判定原语分派（SupportRule.decide，纯函数无 RNG）：卷数与支撑度界
    的比较选择判据——< 界看 CI 下界、≥ 界看点估计，两条口径互不吃对方的
    值（传入另一口径的值证明被忽略）。"""

    def test_sixteen_volumes_hit_bootstrap_rule(self) -> None:
        """16 卷 < 20 界 → CI 口径：点估计 0.90 达标也枉然，CI 下界
        0.50 不过线即不过线（MRA ≈ 16 卷命中规则的判定面）。"""
        assert SupportRule.decide(
            0.90, 16, THRESHOLD, SUPPORT_BOUND, 0.50,
        ) is False

    def test_sixteen_volumes_ci_passes_despite_point_below(self) -> None:
        """16 卷 < 20 界 → CI 口径：点估计 0.50 不达标也无所谓，CI 下界
        0.70 过线即过线——证明 < 界路径完全不看 点估计。"""
        assert SupportRule.decide(
            0.50, 16, THRESHOLD, SUPPORT_BOUND, 0.70,
        ) is True

    def test_sixty_seven_volumes_use_point_estimate(self) -> None:
        """67 卷 ≥ 20 界 → 点估计口径：CI 下界 0.10 惨不过线也无关，
        点估计 0.66 过线即过线（T2w ≈ 67 卷不命中规则的判定面）。"""
        assert SupportRule.decide(
            0.66, 67, THRESHOLD, SUPPORT_BOUND, 0.10,
        ) is True

    def test_sixty_seven_volumes_point_fail_ignores_ci(self) -> None:
        """67 卷 ≥ 20 界 → 点估计口径：CI 下界 0.99 过线也无关，点估计
        0.64 不过线即不过线——证明 ≥ 界路径完全不看 CI。"""
        assert SupportRule.decide(
            0.64, 67, THRESHOLD, SUPPORT_BOUND, 0.99,
        ) is False

    def test_boundary_at_bound_is_point_path(self) -> None:
        """卷数恰 = 支撑度界（20 = 20）→ ≥ 界 → 点估计口径（ADR-0008
        决策 6 原文「< 20 时……」的边界闭合方向）。"""
        assert SupportRule.decide(
            0.66, 20, THRESHOLD, SUPPORT_BOUND, 0.10,
        ) is True
        assert SupportRule.decide(
            0.64, 20, THRESHOLD, SUPPORT_BOUND, 0.99,
        ) is False

    def test_zero_volumes_below_any_bound(self) -> None:
        """0 卷 < 界 → CI 口径（CI 下界 0.40 不过线 → False）。"""
        assert SupportRule.decide(
            0.90, 0, THRESHOLD, SUPPORT_BOUND, 0.40,
        ) is False


class TestVolumeScoreClusters:
    """卷级聚类值对象：观测面的结构守卫（支撑度规则的可复现原料）。"""

    def test_volume_count_is_cluster_cardinality(self) -> None:
        clusters = perfect_chance_clusters(10, 6)
        assert clusters.volume_count == 16

    def test_pooled_auc_matches_auc_from_scores(self) -> None:
        """聚类上的池化点估计（每卷自然重数一次）与 auc_from_scores
        直接口径逐位一致——同一 MW 秩实现，不经第二份统计代码。"""
        clusters = perfect_chance_clusters(10, 6)
        expected = HeldOutAuc.auc_from_scores(
            torch.cat([torch.full((4,), 1.0)] * 10 + [torch.full((4,), 0.0)] * 6),
            torch.zeros(4),
        )
        assert clusters.pooled_auc() == pytest.approx(expected, rel=0.0, abs=0.0)

    def test_two_level_pooled_auc_is_perfect_share_plus_half(self) -> None:
        """两档构造的解析锚：池化 AUC = 0.5 + 完美占比/2 = 0.8125。"""
        clusters = perfect_chance_clusters(10, 6)
        assert clusters.pooled_auc() == pytest.approx(0.8125, rel=0.0, abs=1e-12)

    def test_empty_clusters_rejected(self) -> None:
        with pytest.raises(ValueError, match="至少一卷"):
            VolumeScoreClusters(real_volume_scores=(), fake_scores=torch.zeros(2))

    def test_empty_volume_group_rejected(self) -> None:
        with pytest.raises(ValueError, match="非空"):
            VolumeScoreClusters(
                real_volume_scores=(torch.zeros(0),), fake_scores=torch.zeros(2),
            )

    def test_non_1d_scores_rejected(self) -> None:
        """聚类按展平 patch 分数组织：非一维即结构错位，构造期显式拒绝。"""
        with pytest.raises(ValueError, match="一维"):
            VolumeScoreClusters(
                real_volume_scores=(torch.zeros(2, 2),), fake_scores=torch.zeros(2),
            )

    def test_empty_fake_side_rejected(self) -> None:
        with pytest.raises(ValueError, match="fake"):
            VolumeScoreClusters(
                real_volume_scores=(torch.ones(2),), fake_scores=torch.zeros(0),
            )

    def test_non_finite_scores_rejected(self) -> None:
        """有限性闸口在聚类观测面前置（与 auc_from_scores 同口径同语义）：
        重采样中途抛 NaN 难定位，构造即拒。"""
        for label, volumes, fake in (
            ("real 含 NaN", (torch.tensor([torch.nan, 0.1]),), torch.zeros(2)),
            ("fake 含 NaN", (torch.ones(2),), torch.tensor([torch.nan, 0.1])),
            ("real 含 +Inf", (torch.tensor([torch.inf, 0.1]),), torch.zeros(2)),
            ("fake 含 -Inf", (torch.ones(2),), torch.tensor([-torch.inf, 0.1])),
        ):
            with pytest.raises(ValueError, match="有限"):
                VolumeScoreClusters(
                    real_volume_scores=volumes, fake_scores=fake,
                )


class TestBootstrapReplicates:
    """bootstrap 重复分布的统计形态：重采样单元 = 卷级聚类。"""

    def test_resampling_unit_is_volume_cluster(self) -> None:
        """2 卷（1 完美 + 1 chance）的解析锚：有放回抽 2 卷 → 完美卷抽中
        数 K ~ Binomial(2, 0.5)，重复 AUC ∈ {0.5 (K=0), 0.75 (K=1),
        1.0 (K=2)}，频率 ≈ {1/4, 1/2, 1/4}。重复值严格不落在这三点之间
        是「卷级进出」的判别性断言：patch 级打散必然产生 0.625/0.6875 等
        中间值（见 test 内的反例参考实现）。"""
        clusters = perfect_chance_clusters(1, 1)
        replicates = bootstrap_replicates(
            clusters, replicates=4000,
            generator=torch.Generator().manual_seed(0),
        )
        assert replicates.shape == (4000,)
        unique = set(replicates.tolist())
        assert unique <= {0.5, 0.75, 1.0}, (
            f"卷级重采样的重复值落在解析支撑之外: {sorted(unique)}"
        )
        frequencies = {
            value: (replicates == value).double().mean().item()
            for value in unique
        }
        assert frequencies[0.5] == pytest.approx(0.25, abs=0.02)
        assert frequencies[0.75] == pytest.approx(0.50, abs=0.02)
        assert frequencies[1.0] == pytest.approx(0.25, abs=0.02)

    def test_patch_level_scattering_would_produce_intermediate_values(self) -> None:
        """反例参考实现（patch 级打散，故意与实现对峙）：同一份聚类按
        patch 有放回重采样 → 重复 AUC 出现严格介于 0.5 与 0.75 之间的
        值——与上一测试的卷级支撑互斥，锁死「非 patch 级打散」的口径。
        （k=8 patch/卷：重复 AUC = (kK + 0.5·k(2−K))/(2k) = 0.5 + K/4，
        K ~ Binomial(2, 0.5) → K=1 → 0.75？不——patch 池 8 个 1.0 + 8 个
        0.0，抽 16 个，K ~ Binomial(16, 0.5) → AUC = 0.5 + K/32 ∈
        [0.5, 1.0] 步进 1/32，几乎必然命中 (0.5, 0.75) 开区间。）"""

        def patch_level_replicates(
            clusters: VolumeScoreClusters, replicates: int, seed: int,
        ) -> torch.Tensor:
            pool = torch.cat(clusters.real_volume_scores)
            count = pool.numel()
            generator = torch.Generator().manual_seed(seed)
            return torch.tensor([
                HeldOutAuc.auc_from_scores(
                    pool[torch.randint(count, (count,), generator=generator)],
                    clusters.fake_scores,
                )
                for _ in range(replicates)
            ])

        clusters = perfect_chance_clusters(1, 1, patches=8)
        replicates = patch_level_replicates(clusters, 400, seed=0)
        intermediate = {
            value for value in set(replicates.tolist()) if 0.5 < value < 0.75
        }
        assert intermediate, "patch 级打散竟无中间值——反例构造失效"

    def test_deterministic_under_fixed_seed(self) -> None:
        """RNG 注入的确定性契约：同 seed 重复分布逐位一致；异 seed 不同。"""
        clusters = perfect_chance_clusters(5, 11)
        first = bootstrap_replicates(
            clusters, replicates=200,
            generator=torch.Generator().manual_seed(7),
        )
        second = bootstrap_replicates(
            clusters, replicates=200,
            generator=torch.Generator().manual_seed(7),
        )
        other = bootstrap_replicates(
            clusters, replicates=200,
            generator=torch.Generator().manual_seed(8),
        )
        assert torch.equal(first, second)
        assert not torch.equal(first, other)

    def test_replicates_validation(self) -> None:
        clusters = perfect_chance_clusters(1, 1)
        with pytest.raises(ValueError, match="重复数"):
            bootstrap_replicates(
                clusters, replicates=0,
                generator=torch.Generator().manual_seed(0),
            )


class TestCiLowerBound:
    """CI 下界口径固化：分位约定 + 常量钉死。"""

    def test_quantile_convention_on_binary_support(self) -> None:
        """2 卷聚类（支撑 {0.5, 0.75, 1.0}，质量 {1/4, 1/2, 1/4}）：
        2.5% 分位 → 0.5（下界压在 chance 档）、97.5% 分位 → 1.0、
        中位 → 0.75——torch.quantile 的插值口径随大重复数锁死。"""
        clusters = perfect_chance_clusters(1, 1)
        lower = bootstrap_ci_lower_bound(
            clusters, replicates=4000, quantile=0.025,
            generator=torch.Generator().manual_seed(0),
        )
        upper = bootstrap_ci_lower_bound(
            clusters, replicates=4000, quantile=0.975,
            generator=torch.Generator().manual_seed(0),
        )
        median = bootstrap_ci_lower_bound(
            clusters, replicates=4000, quantile=0.5,
            generator=torch.Generator().manual_seed(0),
        )
        assert lower == 0.5
        assert upper == 1.0
        assert median == 0.75

    def test_quantile_validation(self) -> None:
        clusters = perfect_chance_clusters(1, 1)
        for bad_quantile in (0.0, 1.0, -0.1, 1.1):
            with pytest.raises(ValueError, match="分位"):
                bootstrap_ci_lower_bound(
                    clusters, replicates=10, quantile=bad_quantile,
                    generator=torch.Generator().manual_seed(0),
                )

    def test_defaults_pinned(self) -> None:
        """重复数与百分位由实现者定、以单测固化（issue #85）：1000 次重复、
        2.5% 下界分位（双侧 95% CI，与 KID BootstrapKernelMmd 同口径）。"""
        assert DEFAULT_REPLICATES == 1000
        assert LOWER_QUANTILE == 0.025


class TestSupportRulePass:
    """SupportRule.passes 组合入口：卷数 → 判据分派 + CI 计算（< 界才
    计算——≥ 界路径省下 1000 次重采样）。"""

    def _rule(self, seed: int = 0) -> SupportRule:
        return SupportRule(
            threshold=THRESHOLD,
            support_bound=SUPPORT_BOUND,
            generator=torch.Generator().manual_seed(seed),
        )

    def test_sixteen_volumes_marginal_point_fails_bootstrap(self) -> None:
        """16 卷命中规则的实质场景（MRA 形状）：5 完美 + 11 chance——
        池化点估计 0.65625 恰好压线过（≥ 0.65），但 16 卷支撑下
        bootstrap 2.5% 下界 ≈ 0.54（解析：K ~ Binomial(16, 5/16)，
        P(K ≤ 4) ≈ 0.39 → 下界远在门槛下）→ 不过线。低支撑条件的
        「运气抽样」被 CI 口径拦在门槛外——这正是规则存在的意义。"""
        clusters = perfect_chance_clusters(5, 11)
        assert clusters.pooled_auc() >= THRESHOLD  # 点估计口径本会放行
        assert self._rule().passes(clusters.pooled_auc(), clusters) is False

    def test_sixteen_volumes_strong_signal_passes(self) -> None:
        """16 卷命中规则的对照面：10 完美 + 6 chance——点估计 0.8125、
        bootstrap 下界 ≈ 0.69（K 的 2.5% 分位 ≈ 6 → 0.5 + 6/32）≥ 0.65
        → 过线（真信号不被 CI 口径误杀）。"""
        clusters = perfect_chance_clusters(10, 6)
        assert clusters.pooled_auc() == pytest.approx(0.8125, rel=0.0, abs=1e-12)
        assert self._rule().passes(clusters.pooled_auc(), clusters) is True

    def test_sixty_seven_volumes_point_path_decides_despite_wide_ci(self) -> None:
        """67 卷不命中规则的实质场景（T2w 形状）：21 完美 + 46 chance——
        CI 下界 ≈ 0.60 明明不过线，但 67 ≥ 20 走点估计口径（0.65625 ≥
        0.65）→ 过线。先单独算出 CI 下界 < 门槛，再断言 passes 为真，
        端到端证明 ≥ 界路径无视 CI。"""
        clusters = perfect_chance_clusters(21, 46)
        assert clusters.pooled_auc() == pytest.approx(
            0.5 + 21 / 134, rel=0.0, abs=1e-12,
        )
        assert self._rule().ci_lower_bound(clusters) < THRESHOLD
        assert self._rule().passes(clusters.pooled_auc(), clusters) is True

    def test_sixty_seven_volumes_marginal_point_fails(self) -> None:
        """67 卷不命中规则的对照面：20 完美 + 47 chance——点估计
        0.64925 < 0.65 → 不过线（≥ 界口径下 CI 宽窄不参与判定）。"""
        clusters = perfect_chance_clusters(20, 47)
        assert clusters.pooled_auc() == pytest.approx(
            0.5 + 20 / 134, rel=0.0, abs=1e-12,
        )
        assert clusters.pooled_auc() < THRESHOLD
        assert self._rule().passes(clusters.pooled_auc(), clusters) is False

    def test_constructor_validation(self) -> None:
        generator = torch.Generator().manual_seed(0)
        with pytest.raises(ValueError, match="门槛"):
            SupportRule(threshold=1.5, support_bound=20, generator=generator)
        with pytest.raises(ValueError, match="支撑度界"):
            SupportRule(threshold=0.65, support_bound=0, generator=generator)
