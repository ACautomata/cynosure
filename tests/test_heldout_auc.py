"""held-out AUC 信号测试（ticket #20 AC）：rank 统计数值口径（tie-aware）、
held-out 与训练 real 不相交语义、kind 守卫、推理相位与序列归因
（review #5109004720）、卷级分数聚类观测面（ADR-0008-02，issue #85）。

AUC = Mann-Whitney U 口径：real 分数高于 fake 分数的配对占比（并列 0.5）。
held-out real 永不参与判别器更新（prepare 工件 + kind 守卫保证），
AUC 因此是 out-of-sample 的 hacking 监控信号。
"""

import time
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES
from cynosure.reward.auc import HeldOutAuc, VolumeScoreClusters
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.support import SupportRule

from tests.test_online_update import SHAPE, UpdateScenario, WrittenPool


@pytest.fixture
def scenario(tmp_path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class HeldOutPoolWriter:
    """直写最小 heldout_real 工件（manifest + latent 文件）：latent 按序列
    填常数（t1n → 1.0、其余序列 → 2.0），real 侧条目的序列身份可从输入
    张量取值直接反查（modality 归因断言的观测面）。

    ``distinct_fill=True`` 时每条目填互不相同的常数（1.0 + 下标×0.05，
    跨序列同样互异）——逐卷分数组的成员断言（卷级分组正确性）的观测
    载体：同序列各卷同值时，分组错位在组值上不可见。"""

    FILL: dict[str, float] = {"t1n": 1.0, "t1c": 2.0, "t2w": 2.0, "t2f": 2.0}

    def __init__(
        self, root: Path, per_modality: dict[str, int],
        distinct_fill: bool = False,
    ) -> None:
        self._root = root
        self._per_modality = per_modality
        self._distinct_fill = distinct_fill
        self.manifest_path = root / "heldout_real.json"

    def write(self) -> Path:
        latent_dir = self._root / "heldout_latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        entries: list[PoolEntry] = []
        index = 0
        for modality, count in self._per_modality.items():
            for _ in range(count):
                fill = 1.0 + index * 0.05 if self._distinct_fill else self.FILL[modality]
                latent_path = latent_dir / f"{index}.pt"
                torch.save(torch.full(SHAPE, fill), latent_path)
                entries.append(PoolEntry(
                    case_id=f"case-{index:03d}",
                    modality=modality,
                    latent=f"heldout_latents/{index}.pt",
                    spacing=(100.0, 100.0, 100.0),
                ))
                index += 1
        manifest = LatentManifest(
            kind="heldout_real",
            encoder="synthetic",
            latent_shape=SHAPE,
            split_seed=0,
            split_sizes={"train": 20, "val": index, "test": 4},
            entries=entries,
        )
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )
        return self.manifest_path


class GradProbeScorer:
    """测试仪器：以组合注入记录打分前向的 grad 开关与输入批
    （LatentScorer Protocol 的观测载体，Recording* 先例）。"""

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.grad_enabled_at_call: list[bool] = []
        self.received_batches: list[torch.Tensor] = []

    @property
    def discriminator(self) -> torch.nn.Module:
        return self._inner.discriminator  # type: ignore[attr-defined]

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        self.grad_enabled_at_call.append(torch.is_grad_enabled())
        self.received_batches.append(latents)
        return self._inner.patch_logits(latents)  # type: ignore[attr-defined]


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

    def test_non_finite_scores_rejected(self) -> None:
        """NaN/Inf 分数显式拒绝：排序把非有限值当普通值排（NaN 排尾、
        +Inf 排头），发散或半损坏的 checkpoint 因此能伪装出高分（实测
        全 NaN → AUC 1.5、real 单侧 NaN → 0.875）——「判别器失明」反被
        认证为高判别力。有限性校验是所有消费点（在线监控 / 预训练 gate
        / RM readiness gate）共用的单一闸口。"""
        for label, real, fake in (
            ("real 含 NaN", torch.tensor([torch.nan, 0.1]), torch.tensor([0.0, 0.5])),
            ("fake 含 NaN", torch.tensor([0.9, 0.1]), torch.tensor([torch.nan, 0.5])),
            ("real 含 +Inf", torch.tensor([torch.inf, 0.1]), torch.tensor([0.0, 0.5])),
            ("fake 含 -Inf", torch.tensor([0.9, 0.1]), torch.tensor([-torch.inf, 0.5])),
            ("两侧全 NaN", torch.tensor([torch.nan]), torch.tensor([torch.nan])),
        ):
            with pytest.raises(ValueError, match="有限"):
                HeldOutAuc.auc_from_scores(real, fake)


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


class TestScoringPhase:
    """监控前向的推理相位与序列归因（review #5109004720）。"""

    def test_compute_runs_scorer_forward_without_grad(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """held-out AUC 是非可微监控：打分前向须在 no-grad 下进行。

        判别器参数 requires_grad（Online update 训练它），grad-enabled
        前向保留全部卷积激活图直到输出释放——AUC 永不 backward，加速器
        生产规模（数百 fake × 2048 patch）下白耗显存可至 OOM。与 rollout
        打分相同款约定（打分是 inference，无图）。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        probe = GradProbeScorer(scenario.scorer())
        auc = HeldOutAuc(
            heldout_manifest=manifest, scorer=probe,  # type: ignore[arg-type]
            generator=scenario.generator(1),
        )
        auc.compute(scenario.fakes(4))
        assert probe.grad_enabled_at_call == [False, False]  # real + fake 两前向

    def test_real_side_draws_only_sampled_modality(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """held-out real 侧按本 iteration 采样的序列过滤（per-target-
        sequence 健康监控的归因轴）：iter 事件已按 record.modality 归因
        reward/loss，AUC real 侧若从全池混采，其他序列的判别器分数偏移会
        伪装成本序列 realism 变化。manifest 中 t1n 仅 2 条、其余序列 14
        条：全池混采 count=min(8, 16)=8 必然混入非 t1n；过滤后
        count=min(8, 2)=2 且全部来自 t1n（latent 按序列填常数，身份从
        输入直接反查）。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout",
            {"t1n": 2, "t1c": 5, "t2w": 5, "t2f": 4},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        probe = GradProbeScorer(scenario.scorer())
        auc = HeldOutAuc(
            heldout_manifest=manifest, scorer=probe,  # type: ignore[arg-type]
            generator=scenario.generator(2),
        )
        auc.compute(scenario.fakes(8), modality="t1n")
        real_batch = probe.received_batches[0]  # compute 先 real 后 fake
        assert torch.all(real_batch == HeldOutPoolWriter.FILL["t1n"])

    def test_forward_runs_in_bounded_chunks_and_matches_full_batch(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """打分前向按定块分块累积（显存上界不随评估批量增长）且与全批
        单次前向逐位等价（判别器归一化定死 GroupNorm：前向对 batch 维
        逐样本独立，分块不改变分数；AUC 是分数上的 rank 统计，拼接
        等价）。RM readiness gate 的 fake 侧 = 本 rank base 分区
        （capacity//2 个 3D 体，量产侧限 ``_BASE_BATCH`` 分块生成），
        一次性全量前向会让评估显存随 buffer 容量无界增长——训练开始前
        就可能耗尽加速器。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        probe = GradProbeScorer(scenario.scorer())
        auc = HeldOutAuc(
            heldout_manifest=manifest, scorer=probe,  # type: ignore[arg-type]
            generator=scenario.generator(1),
        )
        chunked = auc.compute(scenario.fakes(20))  # > 单块上界 → 多次前向
        assert len(probe.received_batches) > 2  # real 1 次 + fake 多块
        assert all(
            batch.shape[0] <= HeldOutAuc.SCORE_CHUNK
            for batch in probe.received_batches
        )
        # 等价性参考：同流 real 采样 + 全批单次前向的 AUC（分数拼接在
        # rank 统计下与分块前向逐位一致——rel=0.0 + abs=0.0 真逐位断言）
        reals = RealPoolSampler(manifest, scenario.generator(1)).sample(
            min(20, len(manifest.entries)), modality=None,
        )
        scorer = scenario.scorer()
        expected = HeldOutAuc.auc_from_scores(
            scorer.patch_logits(reals).flatten(),
            scorer.patch_logits(scenario.fakes(20)).flatten(),
        )
        assert chunked == pytest.approx(expected, rel=0.0, abs=0.0)


class NanScorer:
    """测试仪器：打分前向输出全 NaN（LatentScorer Protocol 的替身）——
    判别器数值发散面经聚类观测面的有限性闸口的确定性载体。"""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @property
    def discriminator(self) -> torch.nn.Module:
        return self._inner.discriminator  # type: ignore[attr-defined]

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.full_like(self._inner.patch_logits(latents), torch.nan)  # type: ignore[attr-defined]


class TestVolumeScoreClusters:
    """卷级分数聚类观测面（ADR-0008-02，issue #85）：held-out AUC 暴露
    每卷一组 patch 分数 + fake 侧全量分数，支撑支撑度规则的卷级聚类
    bootstrap 重采样；现有全池/按条件单标量消费路径（compute）不变。"""

    def _auc(self, scenario: UpdateScenario, manifest: LatentManifest,
             seed: int = 1) -> HeldOutAuc:
        return HeldOutAuc(
            heldout_manifest=manifest, scorer=scenario.scorer(),
            generator=scenario.generator(seed),
        )

    def test_all_condition_volumes_exposed_regardless_of_fake_batch(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """聚类观测面与 compute() 的单标量口径的本质差：real 侧不做
        min(fake 批量, 池) 的对称下采样——该条件 held-out **全量卷**
        整卷暴露（卷数是支撑度判定的输入）。池 17 卷、fake 批仅 2：
        compute 只采 2 条 real，聚类仍 17 组。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout",
            {"t1n": 2, "t1c": 5, "t2w": 5, "t2f": 5},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        clusters = auc.compute_volume_clusters(
            scenario.fakes(2), modality="t2w",
        )
        assert clusters.volume_count == 5  # t2w 全量 5 卷，非 min(2, 5)=2
        assert len(clusters.real_volume_scores) == 5

    def test_pool_wide_clusters_cover_every_volume(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """缺省 modality=None（全池口径，诊断/预训练 gate 侧）：聚类覆盖
        全部 8 卷，fake 侧全量参与。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        clusters = auc.compute_volume_clusters(scenario.fakes(4))
        assert clusters.volume_count == 8
        patches_per_volume = clusters.real_volume_scores[0].numel()
        assert all(
            volume.numel() == patches_per_volume
            for volume in clusters.real_volume_scores
        )
        assert clusters.fake_scores.numel() == 4 * patches_per_volume

    def test_volume_groups_align_with_per_volume_forwards(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """卷级分组正确性：聚类的每组 patch 分数与「某一条目逐卷单次
        前向」的展平分数逐位一致，且全体构成双射（同形 latent 每卷等
        patch 数，展平分按卷 reshape 还原；逐卷常数填充互异，分组错位/
        跨界在组值上可见）。分组语义是支撑度规则「整卷进出」的前提。
        注意采样器的无放回 randperm 使组的**顺序**是随机的——按成员
        等价断言，不按 manifest 顺序断言。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 3 for modality in MODALITIES},
            distinct_fill=True,
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        clusters = auc.compute_volume_clusters(scenario.fakes(2))
        scorer = scenario.scorer()
        expected_groups = {
            tuple(scorer.patch_logits(
                manifest.load_latent(entry).unsqueeze(0),
            ).flatten().tolist())
            for entry in manifest.entries
        }
        actual_groups = {
            tuple(volume.tolist()) for volume in clusters.real_volume_scores
        }
        assert actual_groups == expected_groups

    def test_cluster_pooled_auc_matches_compute_when_pool_fully_sampled(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """fake 批量 ≥ 池时 compute() 的 real 侧采样 = 全量池（无放回
        采 n 条于 n 条池即全取）：聚类 pooled_auc 与 compute 单标量
        逐位一致——两个观测面在同一采样面上同源。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout",
            {"t1n": 2, "t1c": 3, "t2w": 5, "t2f": 4},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        clusters = auc.compute_volume_clusters(scenario.fakes(14))
        scalar = auc.compute(scenario.fakes(14))
        assert clusters.pooled_auc() == pytest.approx(scalar, rel=0.0, abs=0.0)

    def test_clusters_feed_support_rule_bootstrap(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """观测面 → 判定原语的端到端：聚类直接喂 SupportRule.passes
        （< 界路径算 CI、≥ 界路径看点估计），支撑度判定不需要第二份
        打分前向。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        clusters = auc.compute_volume_clusters(scenario.fakes(4))
        rule = SupportRule(
            threshold=0.65, support_bound=20,
            generator=scenario.generator(0),
        )
        assert isinstance(rule.passes(clusters.pooled_auc(), clusters), bool)

    def test_clusters_scored_without_grad(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """聚类打分与单标量口径同款推理相位约定：AUC 非可微，前向须在
        no_grad 下进行（生产 patch 规模下 grad-enabled 前向白耗激活
        显存至 OOM）。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        probe = GradProbeScorer(scenario.scorer())
        auc = HeldOutAuc(
            heldout_manifest=manifest, scorer=probe,  # type: ignore[arg-type]
            generator=scenario.generator(1),
        )
        auc.compute_volume_clusters(scenario.fakes(4))
        assert probe.grad_enabled_at_call
        assert all(enabled is False for enabled in probe.grad_enabled_at_call)

    def test_cluster_scoring_runs_in_bounded_chunks(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """聚类观测面与单标量口径共用 SCORE_CHUNK 定块上界：real 侧
        全量卷（可能远超 fake 批量）打分显存不随池长无界增长。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 3 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        probe = GradProbeScorer(scenario.scorer())
        auc = HeldOutAuc(
            heldout_manifest=manifest, scorer=probe,  # type: ignore[arg-type]
            generator=scenario.generator(1),
        )
        auc.compute_volume_clusters(scenario.fakes(20))
        assert all(
            batch.shape[0] <= HeldOutAuc.SCORE_CHUNK
            for batch in probe.received_batches
        )

    def test_empty_modality_pool_rejected(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """该条件 held-out 为空（卷数 = 0）显式拒绝：卷数是支撑度
        判定的输入，0 卷无判定面（与 compute 的口径守卫同语义）。"""
        writer = HeldOutPoolWriter(tmp_path / "heldout", {"t1n": 2})
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        with pytest.raises(ValueError, match="held-out"):
            auc.compute_volume_clusters(scenario.fakes(2), modality="t2w")

    def test_empty_fake_batch_rejected(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        with pytest.raises(ValueError, match="非空"):
            auc.compute_volume_clusters(scenario.fakes(2)[:0])

    def test_non_finite_scores_rejected_at_cluster_gate(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """发散判别器在聚类构造期即拒（有限性闸口前置）：bootstrap 在
        重采样簇上重算池化 AUC，NaN 若流入重采样中途才爆将难以定位。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout", {modality: 2 for modality in MODALITIES},
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = HeldOutAuc(
            heldout_manifest=manifest,
            scorer=NanScorer(scenario.scorer()),  # type: ignore[arg-type]
            generator=scenario.generator(1),
        )
        with pytest.raises(ValueError, match="有限"):
            auc.compute_volume_clusters(scenario.fakes(2))

    def test_existing_scalar_consumption_paths_unchanged(
        self, scenario: UpdateScenario, tmp_path: Path,
    ) -> None:
        """ADR-0008-02 的观测面是**新增**面：聚类方法存在且被调用后，
        compute() 的单标量语义（全池口径、按条件口径）不受影响。逐卷
        异值填充 + fake 批量 ≥ 池：real 侧无放回采样取全池（randperm
        只换顺序不换成员），重算值对采样器状态漂移免疫——逐位不等即
        语义真回归。"""
        writer = HeldOutPoolWriter(
            tmp_path / "heldout",
            {modality: 2 for modality in MODALITIES},
            distinct_fill=True,
        )
        manifest = LatentManifest.load(writer.write(), kind="heldout_real")
        auc = self._auc(scenario, manifest)
        before_pool = auc.compute(scenario.fakes(8))
        before_cond = auc.compute(scenario.fakes(8), modality="t1n")
        auc.compute_volume_clusters(scenario.fakes(2), modality="t2w")
        assert auc.compute(scenario.fakes(8)) == pytest.approx(
            before_pool, rel=0.0, abs=0.0,
        )
        assert auc.compute(scenario.fakes(8), modality="t1n") == pytest.approx(
            before_cond, rel=0.0, abs=0.0,
        )
