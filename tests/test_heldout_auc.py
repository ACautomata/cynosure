"""held-out AUC 信号测试（ticket #20 AC）：rank 统计数值口径（tie-aware）、
held-out 与训练 real 不相交语义、kind 守卫、推理相位与序列归因
（review #5109004720）。

AUC = Mann-Whitney U 口径：real 分数高于 fake 分数的配对占比（并列 0.5）。
held-out real 永不参与判别器更新（prepare 工件 + kind 守卫保证），
AUC 因此是 out-of-sample 的 hacking 监控信号。
"""

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
