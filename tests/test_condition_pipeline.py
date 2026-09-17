"""fixture 异形状多条件的全链贯通测试（#129，spec #125 实现决策 2）。

验收锚（issue #129）：
- fixture 下多条件异形状 config 走通 rollout 与 eval/baseline 采样，
  产出 latent 形状与各条件统一网格一致（词汇表 ``latent_shape(name)``
  为断言真值）；
- 回放缓冲与判别器输入在异形状条件下的批组织可验证（批内同形——
  同条件过滤使 stack/cat 恒安全）；
- resume 状态分片 v7（逐条目张量清单）在异形状条目上的 capture /
  对账 roundtrip。

fixture 异形小词汇表（2 条件）：t1w/axial → latent (4,16,16,8)、
flair/axial → latent (4,8,8,16)——空间 numel 2048 ≠ 1024，逐条件
sigma 日程真实分化（异形 + 异锚双重输入面）。真值锚 = MONAI 库本身
（Anchor 轨迹经 RFlowScheduler 实际 timesteps），零依赖。
"""

import json
from pathlib import Path

import pytest
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from cynosure.conditions import MrConditionVocabulary
from cynosure.eval import ManifestEvaluation, ManifestVolumeSampler
from cynosure.eval.condition import EntryConditionResolver
from cynosure.eval.milestone import MilestoneEvaluator
from cynosure.eval.sampling import EntrySample, ManifestLatentSampler
from cynosure.eval.features import StubSliceFeatureExtractor
from cynosure.eval.volumes import VolumePairFidelity
from cynosure.config import CynosureConfig, RewardConfig
from cynosure.distributed import DistributedContext
from cynosure.fixtures import Fixture
from cynosure.policy.condition import RolloutCondition
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AMP_DTYPES, AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.policy.schedules import PerConditionSchedules
from cynosure.pretrain.artifacts import PretrainProvenance, PretrainReport
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.scorer import LatentScorer, LsganTerms
from cynosure.reward.update import OnlineUpdate
from cynosure.train import TrainingRuntime
from cynosure.train.artifacts import (
    BaselineManifest,
    ManifestEntry,
    RunArtifacts,
)
from cynosure.train.resume import ResumeStore
from cynosure.train.rollout import MrConditionSampler, RolloutPhase
from cynosure.train.rng import TrainingRngStreams
from tests.test_condition_vocabulary import (
    PRODUCTION_CENSUS_PATH,
    PRODUCTION_VOCAB_PATH,
)


@pytest.fixture
def fixture_env(tmp_path: Path):
    """fixture MR-RATE 场景：异形小词汇表 + 迷你 UNet 采样场。"""
    fixture = Fixture()
    artifacts = fixture.write_artifacts(tmp_path / "fixtures")
    vocab = MrConditionVocabulary.load(
        artifacts.condition_vocabulary_json, fixture_mode=True,
    )
    torch.manual_seed(7)
    unet = fixture.unet().eval()
    sampler = RolloutSampler(
        CfgCombinedField(unet),
        SdeKernel(eta=0.7, s_max=0.999),
        PerConditionSchedules(
            num_inference_steps=fixture.NUM_INFERENCE_STEPS,
            vocabulary=vocab,
        ),
    )
    config = fixture.config(tmp_path / "fixtures", dataset="MR-RATE")
    return fixture, artifacts, vocab, sampler, config


class ZeroScorer:
    """打分替身：恒零 reward（rollout 编排的形状贯通不受打分数值影响）。"""

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0])

    def reward(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0])


class TestRolloutShapesFollowConditions:
    """rollout 初始噪声/扰动噪声形状逐条件贯通：产出 latent 形状与
    各条件统一网格一致（验收 1 前半）。"""

    def test_both_conditions_produce_vocabulary_shapes(
        self, fixture_env, tmp_path: Path,
    ) -> None:
        fixture, artifacts, vocab, sampler, config = fixture_env
        generator = torch.Generator().manual_seed(0)
        condition_sampler = MrConditionSampler(
            vocab, generator, torch.device("cpu"),
        )
        rollout = RolloutPhase(
            config,
            sampler,
            ZeroScorer(),  # type: ignore[arg-type]  # 替身：形状贯通不受打分影响
            generator,
            condition_sampler,
            vocabulary=vocab,
        )
        expected = {
            "t1w/axial": (4, 16, 16, 8),
            "flair/axial": (4, 8, 8, 16),
        }
        seen: set[tuple[int, ...]] = set()
        for _ in range(12):  # 均匀轮转下 12 次采样两条件覆盖概率 > 1−2⁻¹³
            record = rollout.run_iteration()
            latent_shape = tuple(record.new_fakes.shape[1:])
            assert latent_shape in expected.values()
            assert latent_shape == expected[record.modality]
            seen.add(latent_shape)
        assert seen == set(expected.values())

    def test_base_partition_shapes_follow_quota(
        self, fixture_env, tmp_path: Path,
    ) -> None:
        """base 分区量产：逐条件噪声形状随配额条件（验收 1 的量产路径）。"""
        fixture, artifacts, vocab, sampler, config = fixture_env
        generator = torch.Generator().manual_seed(0)
        condition_sampler = MrConditionSampler(
            vocab, generator, torch.device("cpu"),
        )
        rollout = RolloutPhase(
            config,
            sampler,
            ZeroScorer(),  # type: ignore[arg-type]
            generator,
            condition_sampler,
            vocabulary=vocab,
            base_generator=torch.Generator().manual_seed(5),
        )
        quota = {"t1w/axial": 2, "flair/axial": 3}
        fakes, names = rollout.base_partition_samples(quota)
        assert len(fakes) == 5
        assert names.count("t1w/axial") == 2
        assert names.count("flair/axial") == 3
        for latent, name in zip(fakes, names):
            assert tuple(latent.shape) == vocab.latent_shape(name)

    def test_base_batch_scales_with_condition_volume(self) -> None:
        """量产批量按条件空间体积缩放（#122 首跑 OOM 修复）：基准 =
        64×64×32 空间（BraTS 单域锚）× 8 批；大网格条件缩批防前向激活
        OOM（t1w/coronal 空间 [128,64,128] = 基准体积 8 倍，单域批量
        常数在 MR-RATE 多网格域把量产前向推向 OOM——T12 集群实录）；
        小网格截到基准批量、不放大。

        私有算术的定点测试（与「只测外部行为」的偏离及其理由）：量产
        批量不进任何输出（条数/形状与分批无关），OOM 行为无法在小规模
        单测里复现——缩放算术只能在此档直接钉住；批量本身的正误由
        fixture 全循环（test_base_partition_shapes_follow_quota）以
        输出面覆盖。"""
        assert RolloutPhase._base_batch_for((4, 64, 64, 32)) == 8
        assert RolloutPhase._base_batch_for((4, 128, 64, 128)) == 1
        assert RolloutPhase._base_batch_for((4, 128, 128, 32)) == 2
        assert RolloutPhase._base_batch_for((4, 32, 96, 96)) == 3
        assert RolloutPhase._base_batch_for((4, 32, 32, 16)) == 8


class TestBaselineSamplingShapes:
    """eval/baseline 采样：逐条目噪声形状从条目条件解析（验收 1 后半）。"""

    def test_entries_sample_their_condition_shapes(
        self, fixture_env, tmp_path: Path,
    ) -> None:
        fixture, artifacts, vocab, sampler, config = fixture_env
        resolver = EntryConditionResolver(vocab, torch.device("cpu"))
        latent_sampler = ManifestLatentSampler(
            sampler,
            resolver,
            AmpContext(device=torch.device("cpu"), dtype=torch.float32),
            vocab,
        )
        entries = [
            ManifestEntry(index=0, condition="t1w/axial", noise_seed=1),
            ManifestEntry(index=1, condition="flair/axial", noise_seed=2),
            ManifestEntry(index=2, condition="t1w/axial", noise_seed=3),
        ]
        samples = latent_sampler.sample(entries)
        for sample in samples:
            assert tuple(sample.terminal.shape) == (
                (1, *vocab.latent_shape(sample.target))
            )

    def test_baseline_manifest_builds_vocabulary_conditions(
        self, fixture_env, tmp_path: Path,
    ) -> None:
        """Baseline manifest 条目条件 = 词汇表条件集轮转（MR 线的条件
        取数域是词汇表工件，不是 BraTS 四序列）。"""
        fixture, artifacts, vocab, sampler, config = fixture_env
        manifest = BaselineManifest.build(config)
        assert manifest.conditions == list(vocab.names())
        condition_names = {entry.condition for entry in manifest.entries}
        assert condition_names == set(vocab.names())


class TestReplayBatchOrganization:
    """回放缓冲的异形状批组织（验收 4）：同条件过滤使 stack 恒同形。"""

    def test_heterogeneous_entries_sample_per_condition_shape(self) -> None:
        buffer = ReplayBuffer(8)  # base 4：每条件 2
        t1w_shape = (4, 16, 16, 8)
        flair_shape = (4, 8, 8, 16)
        buffer.fill_base(
            [*torch.randn(2, *t1w_shape), *torch.randn(2, *flair_shape)],
            ["t1w/axial", "t1w/axial", "flair/axial", "flair/axial"],
        )
        buffer.push(torch.randn(1, *t1w_shape), "t1w/axial")
        buffer.push(torch.randn(1, *flair_shape), "flair/axial")
        draw_t1w = buffer.sample_replay(
            2, torch.Generator().manual_seed(0), "t1w/axial",
        )
        assert tuple(draw_t1w.samples.shape) == (2, *t1w_shape)
        draw_flair = buffer.sample_replay(
            2, torch.Generator().manual_seed(0), "flair/axial",
        )
        assert tuple(draw_flair.samples.shape) == (2, *flair_shape)
        # 全池混采（None）在异形缓冲上显式失败——stack 无从同形（绝不
        # 静默混形，批组织在形状守卫处 fail-fast）
        with pytest.raises(RuntimeError):
            buffer.sample_replay(4, torch.Generator().manual_seed(0))


class TinyDiscriminator(torch.nn.Module):
    """最小可训练判别器（1×1×1 单卷积）：AdamW/反向传播面的协议最小
    替身；全卷积——异形批均可前向（批组织断言不受打分数值影响）。"""

    def __init__(self) -> None:
        super().__init__()
        self._conv = torch.nn.Conv3d(4, 1, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self._conv(latents).flatten(1).mean(dim=1)


class RecordingUpdateScorer:
    """更新批观测替身：记录训练带噪入口（``training_patch_logits``）
    与干净域入口（``patch_logits``）的输入批——判别器输入批组织
    （验收 4 判别器侧）的观测缝。"""

    def __init__(self) -> None:
        self._discriminator = TinyDiscriminator()
        self.training_batches: list[torch.Tensor] = []
        self.clean_batches: list[torch.Tensor] = []

    @property
    def discriminator(self) -> torch.nn.Module:
        return self._discriminator

    def training_patch_logits(
        self, latents: torch.Tensor, generator: torch.Generator,
    ) -> torch.Tensor:
        self.training_batches.append(latents)
        return self._discriminator(latents)

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        self.clean_batches.append(latents)
        return self._discriminator(latents)

    def discriminator_terms(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor,
    ) -> LsganTerms:
        real_term = (1.0 - logits_real).pow(2).mean()
        fake_term = (1.0 + logits_fake).pow(2).mean()
        return LsganTerms(
            total=real_term + fake_term,
            real_term=real_term,
            fake_term=fake_term,
        )


class RecordingRealDraw:
    """real 采样替身：按条件回对应形状的同形确定性批并记录调用
    （同条件匹配的 real 侧形状观测缝）。"""

    def __init__(self, shapes: dict[str, tuple[int, ...]]) -> None:
        self._shapes = shapes
        self.calls: list[tuple[int, str | None]] = []
        self.batches: list[torch.Tensor] = []

    @property
    def size(self) -> int:
        return 1024

    def sample(
        self, count: int, *, modality: str | None = None,
    ) -> torch.Tensor:
        self.calls.append((count, modality))
        batch = torch.zeros(count, *self._shapes[modality])
        self.batches.append(batch)
        return batch


class TestDiscriminatorInputBatchOrganization:
    """判别器输入批在异形状条件下的组织（验收 4 的判别器侧）：
    ``OnlineUpdate.step`` 的组合批（current 半区 + 同条件回放过滤）
    与 real 批（同条件匹配）在逐条件调用下批内恒同形、形状随条件切换。"""

    def test_step_batches_are_same_shape_per_condition(self) -> None:
        t1w_shape = (4, 16, 16, 8)
        flair_shape = (4, 8, 8, 16)
        buffer = ReplayBuffer(8)  # base 4：每条件 2（回放半区供给足额）
        buffer.fill_base(
            [*torch.randn(2, *t1w_shape), *torch.randn(2, *flair_shape)],
            ["t1w/axial", "t1w/axial", "flair/axial", "flair/axial"],
        )
        scorer = RecordingUpdateScorer()
        real_draw = RecordingRealDraw({
            "t1w/axial": t1w_shape, "flair/axial": flair_shape,
        })
        update = OnlineUpdate(
            scorer=scorer,
            buffer=buffer,
            real_sampler=real_draw,
            config=RewardConfig(
                disc_batch_size_k=4,
                replay_buffer_capacity=64,
                real_pool_manifest="pool.json",
                heldout_real_manifest="heldout.json",
                channel_stats_json="stats.json",
                pretrain_report_json="report.json",
            ),
            generator=torch.Generator().manual_seed(0),
            noise_generator=torch.Generator().manual_seed(1),
        )
        update.step(torch.randn(2, *t1w_shape), "t1w/axial")
        update.step(torch.randn(2, *flair_shape), "flair/axial")
        # 判别器训练前向批（每步 fake 批 + real 批两次带噪前向）与干净域
        # 复算批（每步 real + fake 两次干净前向）：各自批内同形、形状
        # 逐条件随词汇表（绝不跨形 cat/混批）
        doubled = [(4, *t1w_shape), (4, *t1w_shape),
                   (4, *flair_shape), (4, *flair_shape)]
        assert [tuple(batch.shape) for batch in scorer.training_batches] == doubled
        assert [tuple(batch.shape) for batch in scorer.clean_batches] == doubled
        assert [tuple(batch.shape) for batch in real_draw.batches] == [
            (4, *t1w_shape), (4, *flair_shape),
        ]
        assert real_draw.calls == [(4, "t1w/axial"), (4, "flair/axial")]


class TestRealPoolHeterogeneousContract:
    """real pool manifest 的逐条件形状契约（验收 4 的 real 侧）：MR
    多条件工件携带 ``condition_latent_shapes``、装载期逐条目对账；
    BraTS 单域不带（全局形状对账 = 单条件词汇特例）。"""

    def _manifest(
        self, tmp_path: Path, vocab: MrConditionVocabulary,
        corrupt: str | None = None,
    ) -> Path:
        shapes = {
            name: vocab.latent_shape(name) for name in vocab.names()
        }
        if corrupt is not None:
            shapes[corrupt] = (4, 16, 16, 8)  # flair 条目登记 t1w 形状
        latents_dir = tmp_path / "latents"
        latents_dir.mkdir(exist_ok=True)
        entries = []
        for index, name in enumerate(vocab.names()):
            path = latents_dir / f"{index}.pt"
            torch.save(torch.randn(*shapes[name]), path)
            entries.append({
                "case_id": f"case-{index}",
                "modality": name,
                "latent": f"latents/{index}.pt",
                "spacing": [100.0, 100.0, 100.0],
            })
        manifest_path = tmp_path / "real_pool.json"
        manifest_path.write_text(json.dumps({
            "kind": "real_pool",
            "encoder": "fixture-mr-test",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"train": 2, "val": 0, "test": 0},
            "entries": entries,
            "condition_latent_shapes": {
                name: list(vocab.latent_shape(name))
                for name in vocab.names()
            },
        }), encoding="utf-8")
        return manifest_path

    def test_per_condition_manifest_loads_and_reconciles(
        self, tmp_path: Path, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        manifest = LatentManifest.load(
            self._manifest(tmp_path, vocab), kind="real_pool",
        )
        assert manifest.condition_latent_shapes is not None
        for entry in manifest.entries:
            latent = manifest.load_latent(entry)
            assert tuple(latent.shape) == vocab.latent_shape(entry.modality)

    def test_corrupted_condition_shape_rejected(
        self, tmp_path: Path, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        # 工件登记 flair = t1w 形状 → 装载期契约自相矛盾（清单里 flair
        # 登记 16×16×8 而条目实存 8×8×16 → load_latent 对账拒绝）
        manifest = LatentManifest.load(
            self._manifest(tmp_path, vocab, corrupt="flair/axial"),
            kind="real_pool",
        )
        flair_entry = next(
            entry for entry in manifest.entries
            if entry.modality == "flair/axial"
        )
        with pytest.raises(ValueError, match="与契约"):
            manifest.load_latent(flair_entry)

    def test_condition_missing_from_contract_rejected(
        self, tmp_path: Path, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        path = self._manifest(tmp_path, vocab)
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["condition_latent_shapes"]["flair/axial"]
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="缺条目条件"):
            LatentManifest.load(path, kind="real_pool")


class TestResumeV7HeterogeneousZones:
    """resume v7 逐条目分片形态（验收 4 的持久化侧）：异形条目
    capture / 对账 roundtrip、条件形状错位拒绝。"""

    def test_capture_zone_is_per_entry_list(self) -> None:
        entries = [
            ReplayEntryStub(torch.randn(4, 16, 16, 8), "t1w/axial"),
            ReplayEntryStub(torch.randn(4, 8, 8, 16), "flair/axial"),
        ]
        zone = ResumeStore._capture_zone(entries)  # type: ignore[arg-type]
        assert zone is not None
        assert isinstance(zone["latents"], list)
        assert len(zone["latents"]) == 2
        assert tuple(zone["latents"][0].shape) == (4, 16, 16, 8)
        assert tuple(zone["latents"][1].shape) == (4, 8, 8, 16)
        assert zone["modalities"] == ["t1w/axial", "flair/axial"]

    def test_validate_zone_checks_per_condition_shapes(
        self, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        zone = {
            "latents": [
                torch.randn(*vocab.latent_shape("t1w/axial")),
                torch.randn(*vocab.latent_shape("flair/axial")),
            ],
            "modalities": ["t1w/axial", "flair/axial"],
        }
        latents, modalities = ResumeStore._validate_zone(
            "recent", zone, vocab,
        )
        assert modalities == ["t1w/axial", "flair/axial"]

    def test_validate_zone_rejects_shape_mismatch(
        self, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        zone = {
            "latents": [
                torch.randn(*vocab.latent_shape("flair/axial")),
            ],
            "modalities": ["t1w/axial"],  # t1w 条目带 flair 形状 → 拒绝
        }
        with pytest.raises(ValueError, match="词汇表形状"):
            ResumeStore._validate_zone("recent", zone, vocab)

    def test_validate_zone_rejects_unknown_condition(
        self, fixture_env,
    ) -> None:
        _, _, vocab, _, _ = fixture_env
        zone = {
            "latents": [torch.randn(4, 16, 16, 8)],
            "modalities": ["not-in-vocabulary"],
        }
        with pytest.raises(ValueError, match="非法目标条件"):
            ResumeStore._validate_zone("recent", zone, vocab)


# ---------------------------------------------------------------------------
# 里程碑评测的异形状读数（验收 3）——替身与 test_milestone_eval 同款机制


class HeterogeneousLatentSampler:
    """异形采样替身：t1w 条目 latent (4,16,16,8)、flair 条目 (4,8,8,16)。"""

    SHAPES = {"t1w/axial": (4, 16, 16, 8), "flair/axial": (4, 8, 8, 16)}

    def sample(self, entries) -> list[EntrySample]:
        return [
            EntrySample(
                entry=entry,
                target=entry.condition,
                source_case=None,
                terminal=torch.zeros(1, *self.SHAPES[entry.condition]),
            )
            for entry in entries
        ]


class PerConditionDecoder:
    """逐条件解码替身：latent 空间轴 ×4 → 影像体（4× 空间压缩契约）。"""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x, y, z = latents.shape[-3:]
        return torch.rand(
            (latents.shape[0], 1, x * 4, y * 4, z * 4),
            generator=torch.Generator().manual_seed(0),
        )


class PerConditionRealStore:
    """逐条件参照替身：影像体形状随条件（统一网格 resample 后口径）。"""

    def case_ids(self) -> list[str]:
        return ["case-a"]

    def volume(self, case_id: str, modality: str) -> torch.Tensor:
        x, y, z = HeterogeneousLatentSampler.SHAPES[modality][1:]
        return torch.rand(x * 4, y * 4, z * 4)


class TestMilestoneHeterogeneousReadings:
    """里程碑评测按条件产出读数、解码路径形状随条件（验收 3）。"""

    def _config(self) -> CynosureConfig:
        data = {
            "experiment": {"group": "modal-label", "dataset": "MR-RATE"},
            "fixture_mode": True,
            # 强度臂两域锁死（#130/#71）：MR 线须显式 clip=False
            "preprocessing": {"intensity_clip": False},
            "artifacts": {
                "unet_ckpt": "u.pt",
                "vae_ckpt": "v.pt",
                "net_config_json": "n.json",
                "modality_mapping_json": "m.json",
                "dataset_root": "data",
                "condition_vocabulary_json": "vocab.json",
                # prepare 装配输入四件套（#121/#131 schema 必填面）
                "mrrate_metadata_csv": "metadata.csv",
                "mrrate_splits_csv": "splits.csv",
                "eval_manifest_csv": "eval_manifest.csv",
                "mrrate_data_snapshot": "MR-RATE@v1.0",
            },
            "reward": {
                "disc_batch_size_k": 4,
                "replay_buffer_capacity": 64,
                "real_pool_manifest": "p.json",
                "heldout_real_manifest": "h.json",
                "channel_stats_json": "c.json",
                "pretrain_report_json": "r.json",
                "pretrain_gate_auc": 0.51,
                "sampling_manifest_json": "sampling_manifest.json",
            },
            "schedule": {
                "seed": 0, "baseline_samples": 4,
                "milestone_eval_samples": 4,
            },
        }
        return CynosureConfig.model_validate(data)

    def test_per_condition_metrics_with_heterogeneous_volumes(self) -> None:
        entries = [
            ManifestEntry(index=index, condition=name, noise_seed=index)
            for index, name in enumerate(
                ["t1w/axial", "flair/axial", "t1w/axial", "flair/axial"],
            )
        ]
        manifest = BaselineManifest(
            seed=0, group="modal-label",
            conditions=["t1w/axial", "flair/axial"], entries=entries,
        )
        evaluator = MilestoneEvaluator(
            self._config(),
            1,
            HeterogeneousLatentSampler(),
            PerConditionDecoder(),
            StubSliceFeatureExtractor(),
            PerConditionRealStore(),
            manifest,
            torch.device("cpu"),
        )
        metrics = evaluator.evaluate()
        # 按条件产出读数：两条件各自的分层 FID/KID 在案
        assert set(metrics.target_fid) == {"t1w/axial", "flair/axial"}
        assert set(metrics.target_kid) == {"t1w/axial", "flair/axial"}
        assert all(
            value == value and abs(value) < float("inf")
            for value in metrics.target_fid.values()
        )


class ReplayEntryStub:
    """ReplayEntry 的静态替身（resume._capture_zone 只消费两字段）。"""

    def __init__(self, latent: torch.Tensor, modality: str) -> None:
        self.latent = latent
        self.modality = modality


class MarkedLatentSampler:
    """条目序打标采样替身：terminal 以条目序号常值填充（配对对账的
    标记原料）；目标端取组2 双条目条件的 target 位、源病例随条目锁定。"""

    def sample(self, entries: list[ManifestEntry]) -> list[EntrySample]:
        samples = []
        for entry in entries:
            assert isinstance(entry.condition, list)  # 组2 = [源, 目标]
            samples.append(EntrySample(
                entry=entry,
                target=entry.condition[1],
                source_case=entry.source_case,
                terminal=torch.full((1, 4, 16, 16, 8), float(entry.index + 1)),
            ))
        return samples


class MarkingDecoder:
    """打标解码替身：每行 latent 的标记常值 + 确定性微纹理 →
    [K,1,4,4,4] 体（行序与输入批一致；微纹理使 SSIM 方差项良态）。"""

    _TEXTURE = torch.arange(64, dtype=torch.float32).view(1, 1, 4, 4, 4) / 255.0

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        marks = latents.mean(dim=(1, 2, 3, 4))
        return marks.view(-1, 1, 1, 1, 1) + self._TEXTURE


class PlanarExtractor:
    """特征替身：切片体摊平取前 16 维（FID/KID 面的协议最小替身）。"""

    def extract(self, slices: torch.Tensor) -> torch.Tensor:
        return slices.flatten(1)[:, :16].double()


class MarkedRealStore:
    """打标参照替身：参照体 = 锁定条目的标记值 + 同一微纹理——配对
    正确时与合成侧逐位相等（配对对齐断言的原料）。"""

    _TEXTURE = torch.arange(64, dtype=torch.float32).view(4, 4, 4) / 255.0

    def __init__(self) -> None:
        self._marks = {"case-0": 1.0, "case-1": 2.0}

    def case_ids(self) -> list[str]:
        return list(self._marks)

    def volume(self, case_id: str, target: str) -> torch.Tensor:
        return self._marks[case_id] + self._TEXTURE


class TestCrossModalPairAlignment:
    """组2 配对逐位对齐（#129 回归锁）：manifest 条目目标交错且轮转序
    非名字典序——分组解码的 synthetic 必须按**条目原序**回排后才与
    manifest 序的参照栈配对（按条件名序 cat 会逐位错位，SSIM/MAE/
    PSNR 配错对）。"""

    def test_pairing_follows_manifest_entry_order(self, tmp_path: Path) -> None:
        config = Fixture().config(
            tmp_path / "fixtures", group="cross-modal",
        )
        config.schedule.milestone_eval_samples = 2
        # 条目 0 目标 t2w、条目 1 目标 t1c：manifest 序 [t2w, t1c]，
        # 条件名序 [t1c, t2w]——按名序 cat 的错位口径在此必然现行
        entries = [
            ManifestEntry(
                index=0, condition=["t1n", "t2w"], source_case="case-0",
                noise_seed=0,
            ),
            ManifestEntry(
                index=1, condition=["t1n", "t1c"], source_case="case-1",
                noise_seed=1,
            ),
        ]
        manifest = BaselineManifest(
            seed=0, group="cross-modal",
            conditions=["t1n", "t1c", "t2w"], entries=entries,
        )
        evaluator = MilestoneEvaluator(
            config,
            1,
            MarkedLatentSampler(),
            MarkingDecoder(),
            PlanarExtractor(),
            MarkedRealStore(),
            manifest,
            torch.device("cpu"),
        )
        metrics = evaluator.evaluate()
        # 配对正确 = 合成与参照逐位相等：MAE 恰 0、SSIM 1、PSNR 封顶
        assert metrics.mae == 0.0
        assert metrics.ssim == pytest.approx(1.0, abs=1e-3)
        assert metrics.psnr == pytest.approx(VolumePairFidelity.PSNR_CAP)


class TestMilestoneEvaluationBuildGuard:
    """MR 评测装配面（验收 3 的装配侧）：schema 不读词表工件，
    ``milestone_eval_samples`` 对词汇表条件数的下界由 build 期守卫
    承接（fixture 豁免不触发；BraTS 的下界校验仍在 schema 层）。"""

    def test_k_below_vocabulary_rejected_outside_fixture_mode(
        self, tmp_path: Path, fixture_env,
    ) -> None:
        _, _, vocab, sampler, config = fixture_env
        # 生产语义（fixture_mode=False）：生产词表装载须携带 #78 普查
        # 引用——真生产工件（11 条件）拷入，census 相对引用随工件解析
        config.fixture_mode = False
        config.schedule.milestone_eval_samples = 2  # < 生产词表 11 条件
        vocab_path = Path(config.artifacts.condition_vocabulary_json)
        data = json.loads(PRODUCTION_VOCAB_PATH.read_text(encoding="utf-8"))
        census_target = vocab_path.parent / data["census_grid_csv"]
        census_target.parent.mkdir(parents=True, exist_ok=True)
        census_target.write_bytes(PRODUCTION_CENSUS_PATH.read_bytes())
        vocab_path.write_text(json.dumps(data), encoding="utf-8")
        pool_path = Path(config.reward.real_pool_manifest)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(LatentManifest(
            kind="real_pool",
            encoder="fixture",
            latent_shape=(4, 16, 16, 8),
            split_seed=0,
            split_sizes={"train": 0, "val": 0, "test": 0},
            entries=[],
            condition_latent_shapes={
                name: vocab.latent_shape(name) for name in vocab.names()
            },
        ).model_dump_json(), encoding="utf-8")
        with pytest.raises(ValueError, match="未覆盖本域条件词汇表"):
            ManifestEvaluation.build(
                config,
                RunArtifacts.init(config, tmp_path / "run"),
                sampler,
                stage=1,
                manifest=BaselineManifest(
                    seed=0, group="modal-label",
                    conditions=list(vocab.names()),
                    entries=[
                        ManifestEntry(
                            index=0, condition="t1w/axial", noise_seed=0,
                        ),
                        ManifestEntry(
                            index=1, condition="flair/axial", noise_seed=1,
                        ),
                    ],
                ),
                amp=AmpContext(torch.device("cpu"), torch.bfloat16),
                write_enabled=False,
            )

    def test_mr_reference_store_rejected_explicitly(
        self, tmp_path: Path, fixture_env,
    ) -> None:
        """MR-RATE 评测装配在参照影像库处显式拒绝：RealVolumeStore 是
        BraTS 病例布局的参照库（dataset_root 扫描与序列键都是 BraTS
        语义），MR config 此前走到 BratsSeriesLayout 扫描才炸出
        「源数据集根目录不存在」的布局错误——装配期给出能力边界声明。"""
        _, _, vocab, sampler, config = fixture_env
        config.schedule.milestone_eval_samples = len(vocab.names())
        pool_path = Path(config.reward.real_pool_manifest)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool_path.write_text(LatentManifest(
            kind="real_pool",
            encoder="fixture",
            latent_shape=(4, 16, 16, 8),
            split_seed=0,
            split_sizes={"train": 0, "val": 0, "test": 0},
            entries=[],
            condition_latent_shapes={
                name: vocab.latent_shape(name) for name in vocab.names()
            },
        ).model_dump_json(), encoding="utf-8")
        with pytest.raises(ValueError, match="参照影像库"):
            ManifestEvaluation.build(
                config,
                RunArtifacts.init(config, tmp_path / "run"),
                sampler,
                stage=1,
                manifest=BaselineManifest(
                    seed=0, group="modal-label",
                    conditions=list(vocab.names()),
                    entries=[
                        ManifestEntry(
                            index=0, condition="t1w/axial", noise_seed=0,
                        ),
                    ],
                ),
                amp=AmpContext(torch.device("cpu"), torch.bfloat16),
                write_enabled=False,
            )


class MrPretrainArtifactsFixture:
    """MR-RATE 预训练产物夹具（#129 报告守卫与装配守卫共用）：real pool /
    held-out manifest（逐条件形状契约 = 词汇表真值）、channel stats 与
    对齐的报告 provenance——守卫测试里唯一的拒绝来源即被测对照本身。"""

    ENTRIES_PER_CONDITION: int = 4
    """每条件条目数：≥ fixture 的 disc_batch_size_k（4）× world（1）。"""

    @staticmethod
    def write_manifest(
        path: Path, vocabulary: MrConditionVocabulary, kind: str,
        shape_override: tuple[int, int, int, int] | None = None,
        with_condition_shapes: bool = True,
    ) -> Path:
        """按词汇表形状写一份逐条件 manifest（含 latent 本体）。

        ``shape_override`` = 全部条件改用另一套（「旧词表」）形状落盘：
        条目与契约自洽（装载期逐条目对账通过）、但与活动词汇表异形——
        正是「词表工件改动而 manifest 未重建」的形态。
        ``with_condition_shapes=False`` = 工件整体不携带逐条件形状契约：
        「MR 工件缺表」的形态（load_latent 静默回退全局对账的入口）。"""
        shapes = {
            name: vocabulary.latent_shape(name) for name in vocabulary.names()
        }
        if shape_override is not None:
            shapes = {name: shape_override for name in shapes}
        latents_dir = path.parent / f"{path.stem}_latents"
        latents_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for name in vocabulary.names():
            for index in range(MrPretrainArtifactsFixture.ENTRIES_PER_CONDITION):
                file_name = f"{name.replace('/', '_')}-{index}.pt"
                torch.save(torch.randn(*shapes[name]), latents_dir / file_name)
                entries.append({
                    "case_id": f"case-{name}-{index}",
                    "modality": name,
                    "latent": f"{latents_dir.name}/{file_name}",
                    "spacing": [100.0, 100.0, 100.0],
                })
        payload = {
            "kind": kind,
            "encoder": "fixture-mr-pretrain",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"train": len(entries)},
            "entries": entries,
        }
        if with_condition_shapes:
            payload["condition_latent_shapes"] = {
                name: list(shape) for name, shape in shapes.items()
            }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    @staticmethod
    def write_channel_stats(
        path: Path, vocabulary: MrConditionVocabulary,
    ) -> Path:
        """判别器输入的标准化统计量（冷启动装配必读工件；取值不参与本组
        断言——四通道恒等标准化）。"""
        path.write_text(json.dumps({
            "kind": "channel_stats",
            "mean": [0.0, 0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0, 1.0],
            "num_latents": 8,
            "latent_shape": list(
                vocabulary.latent_shape(vocabulary.names()[0]),
            ),
            "source_manifest": "real_pool.json",
        }), encoding="utf-8")
        return path

    @staticmethod
    def report(
        config: CynosureConfig, vocabulary: MrConditionVocabulary,
        **overrides,
    ) -> PretrainReport:
        """与磁盘工件全部对齐的 MR 预训练报告（条件集 = 词汇表条件集、
        provenance 指纹 = 当前工件内容）——多条件线不记单域全局形状
        （形状口径由词表指纹承载）。"""
        reward = config.reward
        provenance = PretrainProvenance(
            real_pool_manifest=str(reward.real_pool_manifest),
            real_pool_manifest_sha256=PretrainProvenance.digest(
                Path(reward.real_pool_manifest),
            ),
            heldout_manifest=str(reward.heldout_real_manifest),
            heldout_manifest_sha256=PretrainProvenance.digest(
                Path(reward.heldout_real_manifest),
            ),
            channel_stats=str(reward.channel_stats_json),
            channel_stats_sha256=PretrainProvenance.digest(
                Path(reward.channel_stats_json),
            ),
            discriminator_config=str(config.artifacts.discriminator_config_json),
            discriminator_config_sha256=PretrainProvenance.digest(
                Path(config.artifacts.discriminator_config_json),
            ),
            discriminator_ckpt="checkpoints/pretrain_discriminator.pt",
            discriminator_ckpt_sha256=PretrainProvenance.digest(
                Path(config.artifacts.discriminator_ckpt),
            ),
            condition_vocabulary=str(config.artifacts.condition_vocabulary_json),
            condition_vocabulary_sha256=PretrainProvenance.digest(
                Path(config.artifacts.condition_vocabulary_json),
            ),
        )
        fields = {
            "group": config.experiment.group,
            "latent_shape": None,
            "condition_auc": {name: 0.7 for name in vocabulary.names()},
            "gate_whitelist": list(vocabulary.names()),
            "steps_completed": 12,
            "gate_auc": 0.51,
            "gate_passed": True,
            "discriminator_ckpt": "checkpoints/pretrain_discriminator.pt",
            "provenance": provenance,
        }
        fields.update(overrides)
        return PretrainReport(**fields)


@pytest.fixture
def mr_pretrain_artifacts(fixture_env):
    """MR 预训练产物面落盘（manifest 用词汇表真值形状 + channel stats）。"""
    _, _, vocab, _, config = fixture_env
    MrPretrainArtifactsFixture.write_manifest(
        Path(config.reward.real_pool_manifest), vocab, kind="real_pool",
    )
    MrPretrainArtifactsFixture.write_manifest(
        Path(config.reward.heldout_real_manifest), vocab,
        kind="heldout_real",
    )
    MrPretrainArtifactsFixture.write_channel_stats(
        Path(config.reward.channel_stats_json), vocab,
    )
    return vocab, config


class TestRealPoolVocabularyShapeGuard:
    """real pool / held-out manifest 的逐条件形状契约与活动词汇表的装配期
    对照（#129）：同名异形（词表工件改动而 manifest 未重建）此前放行到
    首次判别器拼接 real 与 fake 时才炸——装配期显式拒绝。"""

    def test_aligned_manifest_passes(self, mr_pretrain_artifacts) -> None:
        """契约与词汇表逐条件同形：守卫放行（无异常）。"""
        vocab, config = mr_pretrain_artifacts
        LatentManifest.load(
            Path(config.reward.real_pool_manifest), kind="real_pool",
        ).assert_condition_shapes(vocab)

    def test_drifting_condition_named_in_error(
        self, tmp_path: Path, mr_pretrain_artifacts,
    ) -> None:
        """条目与契约自洽的旧词表 manifest 不再静默入训：报错点名漂移
        条件与词汇表侧期望形状（可行动）。"""
        vocab, _ = mr_pretrain_artifacts
        stale = MrPretrainArtifactsFixture.write_manifest(
            tmp_path / "stale_pool.json", vocab, kind="real_pool",
            shape_override=(4, 16, 16, 8),
        )
        manifest = LatentManifest.load(stale, kind="real_pool")
        with pytest.raises(ValueError, match="词汇表") as exc_info:
            manifest.assert_condition_shapes(vocab)
        message = str(exc_info.value)
        assert "flair/axial" in message  # 与词表异形的哪一条件
        assert "[4, 8, 8, 16]" in message  # 词汇表侧期望形状
        assert "按当前词表重建 manifest" in message  # 可行动指引

    def test_assembly_rejects_stale_shape_contract(
        self, mr_pretrain_artifacts,
    ) -> None:
        """装配缝收口：``assemble_rewards`` 在装配期拒绝同名异形的 real
        pool（判别器侧无「两套影像空间拼一批」的窗口）。"""
        vocab, config = mr_pretrain_artifacts
        MrPretrainArtifactsFixture.write_manifest(
            Path(config.reward.real_pool_manifest), vocab, kind="real_pool",
            shape_override=(4, 16, 16, 8),
        )
        dist = DistributedContext.bootstrap()
        with pytest.raises(ValueError, match="词汇表"):
            TrainingRuntime.assemble_rewards(
                config,
                AmpContext(
                    device=torch.device("cpu"),
                    dtype=AMP_DTYPES[config.policy.amp_dtype],
                ),
                TrainingRngStreams(
                    dist.derive_seed(config.schedule.seed),
                ).named(),
                dist,
            )

    def test_missing_contract_rejected_at_assembly(
        self, mr_pretrain_artifacts,
    ) -> None:
        """多条件域 manifest 缺逐条件形状契约：装配期拒绝——缺表则
        ``load_latent`` 静默回退全局 ``latent_shape`` 对账，异形条件在
        判别器 real 采样 / gate 重算期才炸、同形条件带着错误的全局口径
        静默入训（判别器全卷积，形状差异自身不报错）。"""
        vocab, config = mr_pretrain_artifacts
        MrPretrainArtifactsFixture.write_manifest(
            Path(config.reward.real_pool_manifest), vocab,
            kind="real_pool", with_condition_shapes=False,
        )
        dist = DistributedContext.bootstrap()
        with pytest.raises(ValueError, match="逐条件形状契约"):
            TrainingRuntime.assemble_rewards(
                config,
                AmpContext(
                    device=torch.device("cpu"),
                    dtype=AMP_DTYPES[config.policy.amp_dtype],
                ),
                TrainingRngStreams(
                    dist.derive_seed(config.schedule.seed),
                ).named(),
                dist,
            )


class TestPretrainReportVocabularyGuard:
    """预训练报告与活动词汇表的口径对照（#129 装载期守卫）：条件集取值域
    （换域报告不得上岗）+ 词表工件内容指纹（工件漂移即报告实测值对另一份
    fake 分布负责）。"""

    def test_aligned_report_passes(self, mr_pretrain_artifacts) -> None:
        """条件集 = 词汇表条件集、各项指纹 = 当前工件：守卫全链放行。"""
        vocab, config = mr_pretrain_artifacts
        MrPretrainArtifactsFixture.report(
            config, vocab,
        ).assert_data_provenance(config)

    def test_condition_set_mismatch_rejected(
        self, mr_pretrain_artifacts,
    ) -> None:
        """报告条件集 ≠ 本域词汇表条件集（四序列名拿到 MR config 上岗）：
        装载期显式拒绝——白名单不落到另一条件域的判别力上。"""
        vocab, config = mr_pretrain_artifacts
        mislabelled = MrPretrainArtifactsFixture.report(
            config, vocab,
            condition_auc={"t1n": 0.7, "t1c": 0.7},
            gate_whitelist=["t1n"],
        )
        with pytest.raises(ValueError, match="条件集不符"):
            mislabelled.assert_data_provenance(config)

    def test_single_domain_shape_on_multi_condition_report_rejected(
        self, mr_pretrain_artifacts,
    ) -> None:
        """多条件线报告携带单域全局 latent_shape（换域线无语义）：拒绝
        ——形状口径只经词表工件指纹承载。"""
        vocab, config = mr_pretrain_artifacts
        report = MrPretrainArtifactsFixture.report(
            config, vocab, latent_shape=(4, 16, 16, 8),
        )
        with pytest.raises(ValueError, match="不应携带单域全局"):
            report.assert_data_provenance(config)

    def test_vocabulary_content_drift_rejected(
        self, mr_pretrain_artifacts,
    ) -> None:
        """词表工件内容改动（flair/axial 网格口径变更）而 real 侧工件与
        权重未变：报告条件集与名称都对得上，唯有内容指纹不符——拒绝
        （否则白名单与 AUC 是对另一份 fake 分布的测量）。"""
        vocab, config = mr_pretrain_artifacts
        report = MrPretrainArtifactsFixture.report(config, vocab)
        report.assert_data_provenance(config)  # 对齐基线先放行
        vocabulary_path = Path(config.artifacts.condition_vocabulary_json)
        data = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        flair = next(
            condition for condition in data["conditions"]
            if condition["name"] == "flair/axial"
        )
        flair["grid_xyz"] = [64, 64, 32]  # 与 t1w 同网格：口径漂移
        vocabulary_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="条件词汇表指纹不符"):
            report.assert_data_provenance(config)


class RecordingPixelDecoder:
    """测试仪器：记录 decode 输入批并按真 VAE 的输出形态（[B, 1, X, Y, Z]
    单通道像素批）返回固定体——baseline/重采物化路径的观测面。"""

    def __init__(self) -> None:
        self.batches: list[torch.Tensor] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.batches.append(latents)
        return torch.zeros(latents.shape[0], 1, 8, 8, 4)


class TestBaselineVolumeMaterialization:
    """baseline/重采的解码物化（#129 分组解码的落盘契约）：逐条目体是
    [X, Y, Z] 像素体——解码输出 [B, 1, X, Y, Z] 的批维与单通道维在条目
    分离时一并剥离（单例维不外溢进落盘体）；解码批 ≤ decode_batch_size
    且批内同形（异形条件在分组边界分开，不跨形 cat）。"""

    def test_stored_volume_is_pixel_volume_and_batches_are_bounded(
        self, fixture_env, tmp_path: Path,
    ) -> None:
        _, _, vocab, sampler, config = fixture_env
        config.schedule.decode_batch_size = 2
        latent_sampler = ManifestLatentSampler(
            sampler,
            EntryConditionResolver(vocab, torch.device("cpu")),
            AmpContext(device=torch.device("cpu"), dtype=torch.float32),
            vocab,
        )
        entries = [
            ManifestEntry(index=0, condition="t1w/axial", noise_seed=1),
            ManifestEntry(index=1, condition="flair/axial", noise_seed=2),
            ManifestEntry(index=2, condition="t1w/axial", noise_seed=3),
            ManifestEntry(index=3, condition="flair/axial", noise_seed=4),
        ]
        manifest = BaselineManifest(
            seed=0, group="modal-label",
            conditions=list(vocab.names()), entries=entries,
        )
        paths = RunArtifacts.init(config, tmp_path / "run").paths
        decoder = RecordingPixelDecoder()
        ManifestVolumeSampler(
            1, manifest, latent_sampler, decoder, paths,
            decode_batch_size=config.schedule.decode_batch_size,
        ).sample_baseline()
        # 每批 ≤ 块大小、批内同形；每条目恰解码一次
        for batch in decoder.batches:
            assert batch.shape[0] <= config.schedule.decode_batch_size
            assert len({tuple(t.shape) for t in batch}) == 1
        assert sum(batch.shape[0] for batch in decoder.batches) == 4
        for entry in manifest.entries:
            stored = torch.load(
                paths.root / entry.baseline_sample, weights_only=True,
            )
            assert stored.shape == (8, 8, 4)  # 批维与单通道维均已剥离
            assert torch.isfinite(stored).all()
