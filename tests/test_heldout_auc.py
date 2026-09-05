"""held-out AUC 信号测试（ticket #20 AC）：rank 统计数值口径（tie-aware）、
held-out 与训练 real 不相交语义、kind 守卫、推理相位与序列归因
（review #5109004720）。

AUC = Mann-Whitney U 口径：real 分数高于 fake 分数的配对占比（并列 0.5）。
held-out real 永不参与判别器更新（prepare 工件 + kind 守卫保证），
AUC 因此是 out-of-sample 的 hacking 监控信号。
"""

import time
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.artifacts import LatentManifest, PoolEntry

from tests.test_online_update import SHAPE, UpdateScenario, WrittenPool


@pytest.fixture
def scenario(tmp_path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class HeldOutPoolWriter:
    """直写最小 heldout_real 工件（manifest + latent 文件）：latent 按序列
    填常数（t1n → 1.0、其余序列 → 2.0），real 侧条目的序列身份可从输入
    张量取值直接反查（modality 归因断言的观测面）。"""

    FILL: dict[str, float] = {"t1n": 1.0, "t1c": 2.0, "t2w": 2.0, "t2f": 2.0}

    def __init__(self, root: Path, per_modality: dict[str, int]) -> None:
        self._root = root
        self._per_modality = per_modality
        self.manifest_path = root / "heldout_real.json"

    def write(self) -> Path:
        latent_dir = self._root / "heldout_latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        entries: list[PoolEntry] = []
        index = 0
        for modality, count in self._per_modality.items():
            for _ in range(count):
                latent_path = latent_dir / f"{index}.pt"
                torch.save(torch.full(SHAPE, self.FILL[modality]), latent_path)
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
