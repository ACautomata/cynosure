"""MR-RATE prepare 数据链换域测试（#121/#131，spec #125 实现决策 3）。

三个测试面（spec「Testing Decisions」：只测外部行为，主 seam = CLI 命令
级与工件契约级）：

1. **预处理链泛化**（强度臂 clip=False + 逐条件统一网格 resample）；
2. **装配语义**（官方 split join、评估集互斥硬守卫、patient 级 held-out
   二分、逐条件配额抽样、容量守卫）经 CLI prepare 端到端；
3. **工件契约**（三工件 + 抽样留痕可被既有契约装载、provenance 留痕、
   幂等零漂移）。

BraTS 线回归零改动（既有 test_prepare.py / test_preprocessing.py 不动）。"""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.reward.artifacts import (
    ChannelStats,
    LatentManifest,
    SamplingManifest,
)
from cynosure.reward.sampler import RealPoolSampler
from tests.conftest import RAS_AFFINE, CliResult, CliSession, SyntheticMrRateDataset


class MrPrepareScenario:
    """一次 MR prepare 端到端场景：合成 MR-RATE 数据集 + fixture config
    （dataset=MR-RATE）+ CLI prepare 调用。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self._cli = cli
        self.work_dir = tmp_path
        self.config_path = tmp_path / "config.json"

    def build_config(
        self,
        *,
        train_patients: int = 6,
        eval_rows: list[dict[str, str]] | None = None,
        quota: dict[str, int] | None = None,
        heldout_fraction: float | None = None,
    ) -> CynosureConfig:
        fixtures_dir = self.work_dir / "fixtures"
        Fixture().write_condition_vocabulary(fixtures_dir)
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        if quota is not None:
            config.reward.real_pool_quota = quota
        if heldout_fraction is not None:
            config.reward.heldout_fraction = heldout_fraction
        self._write_config(config)
        dataset = SyntheticMrRateDataset(
            config.artifacts.dataset_root,
            train_patients=train_patients,
            eval_rows=eval_rows,
        )
        dataset.write()
        return config

    def run(self, **kwargs) -> CliResult:
        self.build_config(**kwargs)
        return self._cli.run("prepare", "--config", str(self.config_path))

    def run_with_config(self, config: CynosureConfig) -> CliResult:
        self._write_config(config)
        return self._cli.run("prepare", "--config", str(self.config_path))

    def _write_config(self, config: CynosureConfig) -> None:
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )


@pytest.fixture
def mr_scenario(cli: CliSession, tmp_path: Path) -> MrPrepareScenario:
    return MrPrepareScenario(cli, tmp_path)


class TestFixtureFullChain:
    """fixture 全链（#131 AC5）：小配额夹具数据走通 CLI prepare，全套
    工件（real pool / held-out real / per-channel 统计量 + 抽样留痕）
    可被既有契约装载。"""

    def test_full_chain_produces_loadable_artifacts(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        result = mr_scenario.run()
        assert result.code == 0, result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        # Real sample pool：条件分层计数与逐条件形状登记齐备
        #（6 train patients → 2 held-out patients（fraction 0.3）→ pool 侧
        # 4 patients × 每条件 1 series = 每条件 4 卷；配额 8 为上限，
        # 候选不足取全量）
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, "real_pool",
        )
        assert pool.conditions == {"t1w/axial": 4, "flair/axial": 4}
        assert pool.condition_shapes == {
            "t1w/axial": (4, 16, 16, 8), "flair/axial": (4, 16, 16, 8),
        }
        assert pool.latent_shape is None
        # Held-out real：同契约、kind 守卫区分（patient 级二分的另一侧）
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, "heldout_real",
        )
        assert heldout.conditions == {"t1w/axial": 2, "flair/axial": 2}
        # per-channel 统计量 + provenance（#121 AC2）
        stats = ChannelStats.load(config.reward.channel_stats_json)
        assert stats.provenance is not None
        assert stats.provenance.dataset == "MR-RATE"
        assert stats.provenance.intensity_clip is False
        assert stats.provenance.resize_semantics == "uniform-grid"
        assert stats.provenance.data_snapshot == "fixture-snapshot"
        assert stats.num_latents == 8
        # 抽样留痕（#131）：互斥守卫读数落档、逐卷归属齐备
        sampling = SamplingManifest.model_validate(
            json.loads(
                Path(config.reward.sampling_manifest_json).read_text(
                    encoding="utf-8",
                ),
            ),
        )
        assert sampling.eval_exclusion_series_hits == 0
        assert sampling.eval_exclusion_patient_hits == 0
        assert sampling.census_candidates == {
            "t1w/axial": 6, "flair/axial": 6,
        }
        assert sampling.census_quota_taken == {
            "t1w/axial": 4, "flair/axial": 4,
        }
        assert sampling.heldout_counts == {"t1w/axial": 2, "flair/axial": 2}
        assert sampling.quota == {"t1w/axial": 8, "flair/axial": 8}
        assert sampling.heldout_fraction == pytest.approx(0.3)
        roles = {entry.role for entry in sampling.entries}
        assert roles == {"pool", "heldout"}

    def test_loadable_by_reward_pipeline_seam(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """AC1：pool 工件可被 reward 数据管线加载——RealPoolSampler 按
        生成条件采样同形 stack（条件词表一致的序列分层）。"""
        result = mr_scenario.run()
        assert result.code == 0, result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, "real_pool",
        )
        sampler = RealPoolSampler(pool, torch.Generator().manual_seed(0))
        batch = sampler.sample(4, condition="t1w/axial")
        assert tuple(batch.shape) == (4, 4, 16, 16, 8)
        with pytest.raises(ValueError, match="超出.*flair/axial"):
            sampler.sample(99, condition="flair/axial")

    def test_quota_manifest_idempotent(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """同 seed 重跑工件零漂移（#78 抽样机制先例同款断言）：抽样
        manifest 与 pool manifest 逐字节一致。"""
        first = mr_scenario.run()
        assert first.code == 0, first.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        sampling_before = Path(
            config.reward.sampling_manifest_json,
        ).read_bytes()
        pool_before = Path(config.reward.real_pool_manifest).read_bytes()
        second = mr_scenario.run()
        assert second.code == 0, second.stderr
        assert Path(config.reward.sampling_manifest_json).read_bytes() == (
            sampling_before
        )
        assert Path(config.reward.real_pool_manifest).read_bytes() == pool_before

    def test_eval_series_key_collision_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """评估集互斥（series 键，#131 AC2）：train 候选卷键被登记进评估
        清单（口径漂移形态）→ 装配期 fail-fast 可读拒绝。"""
        eval_rows = [
            {
                "stratum": "T1W/AXIAL", "sampling_role": "stratified-n250",
                "split": "val", "batch_id": "batch00",
                "patient_uid": "P00", "study_uid": "ST00",
                "series_id": "t1w-raw-axi", "modality": "T1W",
                "plane": "AXIAL", "array_shape": "[256,256,128]",
                "array_spacing_mm": "[1,1,1]", "array_fov_mm": "[240,240,174]",
            },
        ]
        result = mr_scenario.run(eval_rows=eval_rows)
        assert result.code == 2
        assert "评估集互斥" in result.stderr
        assert "ST00" in result.stderr

    def test_eval_patient_collision_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """评估集互斥（patient 级第二道防线）：评估清单登记了 train
        patient 的异 study 卷 → 同样拒绝（防同 patient 泄漏）。"""
        eval_rows = [
            {
                "stratum": "T1W/AXIAL", "sampling_role": "stratified-n250",
                "split": "test", "batch_id": "batch01",
                "patient_uid": "P00", "study_uid": "ST99",
                "series_id": "t1w-raw-sag", "modality": "T1W",
                "plane": "SAGITTAL", "array_shape": "[256,256,128]",
                "array_spacing_mm": "[1,1,1]", "array_fov_mm": "[240,240,174]",
            },
        ]
        result = mr_scenario.run(eval_rows=eval_rows)
        assert result.code == 2
        assert "评估集互斥" in result.stderr
        assert "P00" in result.stderr

    def test_capacity_guard_small_pool_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """容量装配守卫（#121 AC5 稀疏模态小池触发口径）：某条件实抽
        < K×world → 装配期可读拒绝（开工前失败，非训练中途）；守卫先于
        manifest 落盘——失败时盘上 manifest 明确缺失（工件契约「要么
        全量一致、要么明确缺失」的延续，抽样留痕同样不落）。"""
        result = mr_scenario.run(quota={"t1w/axial": 2, "flair/axial": 8})
        assert result.code == 2
        assert "容量不足" in result.stderr
        assert "t1w/axial×2" in result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        assert not Path(config.reward.real_pool_manifest).exists()
        assert not Path(config.reward.heldout_real_manifest).exists()
        assert not Path(config.reward.sampling_manifest_json).exists()
        assert not Path(config.reward.channel_stats_json).exists()

    def test_missing_volume_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """候选域（元数据）与影像落盘不一致 → 可读拒绝（不静默丢卷）。"""
        config = mr_scenario.build_config()
        orphan = sorted(config.artifacts.dataset_root.glob("*.nii.gz"))[0]
        orphan.unlink()
        result = mr_scenario.run_with_config(config)
        assert result.code == 2
        assert "影像缺失" in result.stderr

    def test_heldout_patient_level_disjoint(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """held-out 互斥（#121 AC3）：pool 与 held-out 在 patient 级不
        相交（病例级二分），逐条件 held-out 非空（per-condition AUC
        可归因），归属随抽样 manifest 落档可查。"""
        result = mr_scenario.run(heldout_fraction=0.3)
        assert result.code == 0, result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        sampling = SamplingManifest.model_validate(json.loads(
            Path(config.reward.sampling_manifest_json).read_text(
                encoding="utf-8",
            ),
        ))
        pool_patients = {
            entry.patient_uid for entry in sampling.entries
            if entry.role == "pool"
        }
        heldout_patients = {
            entry.patient_uid for entry in sampling.entries
            if entry.role == "heldout"
        }
        assert pool_patients and heldout_patients
        assert pool_patients.isdisjoint(heldout_patients)
        # 评估留出池（E* patients）与两侧均不相交（#73 原则）
        assert pool_patients.isdisjoint({f"E{i:02d}" for i in range(4)})
        assert heldout_patients.isdisjoint({f"E{i:02d}" for i in range(4)})
        for condition, count in sampling.heldout_counts.items():
            assert count >= 1, condition


class TestPerConditionLatentShapes:
    """逐条件异形状贯通的 prepare 侧锚（#111 多网格案落地语义）：条件
    统一网格来自词汇表，编码产物逐条件形状唯一（= 统一网格 / 4）；
    异条件异 latent 形状在同一份工件内登记（condition_shapes）。"""

    def test_conditions_with_distinct_grids(self, mr_scenario) -> None:
        fixtures_dir = mr_scenario.work_dir / "fixtures"
        Fixture().write_condition_vocabulary(fixtures_dir)
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        # 测试内专用小词表（fixture_mode 装载通道）：两条件异统一网格
        vocabulary = {
            "kind": "mrrate-condition-vocabulary",
            "issue": "test",
            "upstream_reference": "test",
            "grid_semantics": "test",
            "modality_tokens": {"t1w": 9, "t2w": 10, "flair": 11, "swi": 20, "mra": 16},
            "conditions": [
                {
                    "name": "t1w/axial", "modality": "t1w", "plane": "axial",
                    "fov_mm": [64.0, 64.0, 32.0], "fov_source": "fixture",
                    "grid_xyz": [64, 64, 32],
                },
                {
                    "name": "flair/axial", "modality": "flair", "plane": "axial",
                    "fov_mm": [64.0, 64.0, 64.0], "fov_source": "fixture",
                    "grid_xyz": [64, 64, 64],
                },
            ],
        }
        config.artifacts.condition_vocabulary_json.write_text(
            json.dumps(vocabulary), encoding="utf-8",
        )
        # 夹具影像异原生形状：同条件内统一网格归一（flair 条件的输入
        # 落 64×64×64 网格、latent [4,16,16,16]；t1w 条件保持 fixture
        # 网格、latent [4,16,16,8]）
        dataset = SyntheticMrRateDataset(config.artifacts.dataset_root)
        dataset.write()
        overrides = {
            "flair-raw-axi": (96, 64, 40),  # flair 条件：异原生形状输入
        }
        for volume_path in sorted(
            config.artifacts.dataset_root.glob("*_flair-raw-axi.nii.gz"),
        ):
            study = volume_path.name.split("_")[0]
            volume = np.random.default_rng(hash(study) % 100).standard_normal(
                overrides["flair-raw-axi"],
            ).astype(np.float32)
            nib.save(nib.Nifti1Image(volume, RAS_AFFINE), volume_path)
        result = mr_scenario.run_with_config(config)
        assert result.code == 0, result.stderr
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, "real_pool",
        )
        assert pool.condition_shapes == {
            "t1w/axial": (4, 16, 16, 8),
            "flair/axial": (4, 16, 16, 16),
        }
        for entry in pool.entries:
            latent = pool.load_latent(entry)
            expected = pool.condition_shapes[entry.condition]
            assert tuple(latent.shape) == expected, entry.case_id
        # spacing = 条件属性（spec #125 决策 6）：同条件两卷的条目值严格
        # 同值（= 等效 spacing ×1e2），不构成「spacing 差异」判别捷径
        by_condition: dict[str, set[tuple[float, float, float]]] = {}
        for entry in pool.entries:
            by_condition.setdefault(entry.condition, set()).add(entry.spacing)
        assert all(
            len(spacings) == 1 for spacings in by_condition.values()
        )
        # FOV = 网格（fixture 词表等效 spacing 全 1.0 mm）→ 两条件同值，
        # 异条件异形状的 spacing 同为条件属性值、逐卷不再独立
        assert by_condition["flair/axial"] == {(100.0, 100.0, 100.0)}
        assert by_condition["t1w/axial"] == {(100.0, 100.0, 100.0)}
