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
from cynosure.reward.mrrate import PatientHeldoutSplit
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
        extra_metadata_rows: list[dict[str, str]] | None = None,
        quota: dict[str, int] | None = None,
        heldout_fraction: float | None = None,
        source_commit: str | None = None,
    ) -> CynosureConfig:
        fixtures_dir = self.work_dir / "fixtures"
        Fixture().write_condition_vocabulary(fixtures_dir)
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        if quota is not None:
            config.reward.real_pool_quota = quota
        if heldout_fraction is not None:
            config.reward.heldout_fraction = heldout_fraction
        if source_commit is not None:
            config.artifacts.source_commit = source_commit
        self._write_config(config)
        dataset = SyntheticMrRateDataset(
            config.artifacts.dataset_root,
            train_patients=train_patients,
            eval_rows=eval_rows,
            extra_metadata_rows=extra_metadata_rows,
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
        assert pool.modalities == {"t1w/axial": 4, "flair/axial": 4}
        # 逐条件异形状（fixture 词表两条件网格互换厚薄轴）：t1w/axial
        # [64,64,32] → (4,16,16,8)；flair/axial [32,32,64] → (4,8,8,16)
        assert pool.condition_latent_shapes == {
            "t1w/axial": (4, 16, 16, 8), "flair/axial": (4, 8, 8, 16),
        }
        # 全局 latent_shape 两域恒填（#129）：MR 多条件域该值只作通道数
        # 对账锚，逐条目对账走 condition_latent_shapes 的权威表
        assert pool.latent_shape == tuple(config.latent_shape)
        # Held-out real：同契约、kind 守卫区分（patient 级二分的另一侧）
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, "heldout_real",
        )
        assert heldout.modalities == {"t1w/axial": 2, "flair/axial": 2}
        # per-channel 统计量 + provenance（#121 AC2）
        stats = ChannelStats.load(config.reward.channel_stats_json)
        assert stats.provenance is not None
        assert stats.provenance.dataset == "MR-RATE"
        assert stats.provenance.intensity_clip is False
        assert stats.provenance.resize_semantics == "uniform-grid"
        assert stats.provenance.data_snapshot == "fixture-snapshot"
        # 未声明 source_commit（config 默认 None）→ 字段留空而非报错
        assert stats.provenance.source_commit is None
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
        batch = sampler.sample(4, modality="t1w/axial")
        assert tuple(batch.shape) == (4, 4, 16, 16, 8)
        with pytest.raises(ValueError, match="超出.*flair/axial"):
            sampler.sample(99, modality="flair/axial")

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

    def test_non_train_split_metadata_volumes_skipped(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """生产形态：元数据覆盖全 split——val/test 患者卷是评估留出池，
        不进 real 数据链候选（join 悬挂判定按 splits **全集**，非 train
        集），计数留痕供审计；候选域不被污染、装配照常成功。"""
        extra_rows = [
            {
                "batch_id": "batch90", "patient_uid": "E00",
                "study_uid": "ES00", "series_id": "t1w-raw-axi",
                "modality": "T1W", "plane": "AXIAL",
            },
            {
                "batch_id": "batch91", "patient_uid": "E01",
                "study_uid": "ES01", "series_id": "flair-raw-axi",
                "modality": "FLAIR", "plane": "AXIAL",
            },
        ]
        result = mr_scenario.run(extra_metadata_rows=extra_rows)
        assert result.code == 0, result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        sampling = SamplingManifest.model_validate(json.loads(
            Path(config.reward.sampling_manifest_json).read_text(
                encoding="utf-8",
            ),
        ))
        assert sampling.non_train_volumes == 2
        assert sampling.census_candidates == {
            "t1w/axial": 6, "flair/axial": 6,
        }
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, "real_pool",
        )
        assert pool.modalities == {"t1w/axial": 4, "flair/axial": 4}

    def test_dangling_patient_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """元数据 patient 不在 splits.csv 任何 split → join 完整性破坏，
        可读拒绝（不静默丢卷）。"""
        extra_rows = [
            {
                "batch_id": "batch99", "patient_uid": "ZZZ",
                "study_uid": "ZS00", "series_id": "t1w-raw-axi",
                "modality": "T1W", "plane": "AXIAL",
            },
        ]
        result = mr_scenario.run(extra_metadata_rows=extra_rows)
        assert result.code == 2
        assert "不在官方" in result.stderr

    def test_unknown_quota_key_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """配额键拼错条件名（词汇表外）→ 装配期可读拒绝：静默忽略等于
        该条件全量不设限，与「显式错误值即拒」哲学不符。"""
        result = mr_scenario.run(
            quota={"t1w/axial": 8, "t1w/axiall": 8},
        )
        assert result.code == 2
        assert "词汇表外" in result.stderr

    def test_source_commit_recorded_in_provenance(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """provenance 的来源 commit（#121 AC2）：config 显式声明的代码
        版本标识随工件落档（未声明路径的留空断言见 full_chain 场景）。"""
        result = mr_scenario.run(source_commit="deadbeef")
        assert result.code == 0, result.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        stats = ChannelStats.load(config.reward.channel_stats_json)
        assert stats.provenance.source_commit == "deadbeef"

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
    异条件异 latent 形状在同一份工件内登记（condition_latent_shapes）。"""

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
        assert pool.condition_latent_shapes == {
            "t1w/axial": (4, 16, 16, 8),
            "flair/axial": (4, 16, 16, 16),
        }
        for entry in pool.entries:
            latent = pool.load_latent(entry)
            expected = pool.condition_latent_shapes[entry.modality]
            assert tuple(latent.shape) == expected, entry.case_id
        # spacing = 条件属性（spec #125 决策 6）：同条件两卷的条目值严格
        # 同值（= 等效 spacing ×1e2），不构成「spacing 差异」判别捷径
        by_condition: dict[str, set[tuple[float, float, float]]] = {}
        for entry in pool.entries:
            by_condition.setdefault(entry.modality, set()).add(entry.spacing)
        assert all(
            len(spacings) == 1 for spacings in by_condition.values()
        )
        # FOV = 网格（fixture 词表等效 spacing 全 1.0 mm）→ 两条件同值，
        # 异条件异形状的 spacing 同为条件属性值、逐卷不再独立
        assert by_condition["flair/axial"] == {(100.0, 100.0, 100.0)}
        assert by_condition["t1w/axial"] == {(100.0, 100.0, 100.0)}


MRA_CONDITION: dict = {
    # 测试专用第三条件（fixture_mode 小词表通道）：MRA 全平面单格
    # （#81 读数格），网格与 t1w 同形——稀疏条件场景的构造件
    "name": "mra/all-planes", "modality": "mra", "plane": "all-planes",
    "fov_mm": [64.0, 64.0, 32.0], "fov_source": "fixture",
    "grid_xyz": [64, 64, 32],
}

MRA_SERIES_IDS: tuple[str, ...] = (
    "mra-raw-axi", "mra-raw-sag", "mra-raw-cor", "mra-raw-tra",
)
"""稀疏条件卷数 = disc_batch_size_k（4）：容量守卫恰好过线，只剩
held-out 覆盖一处可失败——两个守卫的口径互不遮蔽。"""


class TestAssemblyInputHardening:
    """装配输入的完整性守卫（PR #163 评审 5224004327）：重复/冲突登记、
    空评估清单、逐条件 held-out 覆盖、守卫与编码的先后——一律在装配期
    fail-fast，不留「静默降级」或「下游才炸」的路径。"""

    def test_duplicate_series_key_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """元数据重复 (study_uid, series_id) 行 → 拒绝：两行编码到同一
        latent 路径，manifest 把同一枚物理卷当作两卷计数与采样（统计量
        被重复计入、容量守卫可被虚增行数骗过）。"""
        duplicate = {
            "batch_id": "batch00", "patient_uid": "P00", "study_uid": "ST00",
            "series_id": "t1w-raw-axi", "modality": "T1W", "plane": "AXIAL",
        }
        result = mr_scenario.run(extra_metadata_rows=[duplicate])
        assert result.code == 2
        assert "ST00/t1w-raw-axi" in result.stderr

    def test_conflicting_duplicate_split_row_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """splits.csv 同一 patient 两行且 split 冲突 → 拒绝：静默「后行
        覆盖」让 train 人群随行序漂移（本意 val/test 的 patient 混进 real
        pool），而 split_sizes 仍把两行都计上。8 个 train patient 起底：
        P00 转 val 后每条件仍有 5 卷 ≥ K，容量守卫不遮蔽本守卫的报错。"""
        config = mr_scenario.build_config(train_patients=8)
        splits_path = config.artifacts.mrrate_splits_csv
        original = splits_path.read_text(encoding="utf-8")
        splits_path.write_text(f"{original}P00,val\n", encoding="utf-8")
        result = mr_scenario.run_with_config(config)
        assert result.code == 2
        assert "P00" in result.stderr
        assert "重复" in result.stderr

    def test_duplicate_split_row_same_value_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """同 patient 重复行（split 值相同）→ 同样拒绝：patient → split
        是映射不是行表，重复键即登记错误，且 split_sizes 留痕会把患者数
        虚增。"""
        config = mr_scenario.build_config()
        splits_path = config.artifacts.mrrate_splits_csv
        original = splits_path.read_text(encoding="utf-8")
        splits_path.write_text(f"{original}P00,train\n", encoding="utf-8")
        result = mr_scenario.run_with_config(config)
        assert result.code == 2
        assert "P00" in result.stderr

    def test_empty_eval_manifest_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """评估清单只有表头、无数据行 → 拒绝：互斥守卫退化为空集比较
        （恒不相交），硬守卫静默失效而抽样留痕还记下 eval_exclusion_keys=0
        的「已执行」假凭据。同仓 #78 的另一读面（``MrRateEvalManifest``）
        对同一形态即显式拒绝。"""
        result = mr_scenario.run(eval_rows=[])
        assert result.code == 2
        assert "无数据行" in result.stderr

    def test_capacity_guard_fails_before_encoding(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """容量守卫先于编码（「开工前失败」的字面口径）：逐条件实抽数
        在装配计划里已经可知，生产体量下先编码整池再报错 = 白烧加速卡
        数小时——失败时盘上不留任何 latent。"""
        result = mr_scenario.run(quota={"t1w/axial": 2, "flair/axial": 8})
        assert result.code == 2
        assert "容量不足" in result.stderr
        assert not list((mr_scenario.work_dir / "fixtures").rglob("*.pt"))

    def test_failed_rerun_invalidates_sampling_manifest(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """失败重跑不留上一轮的抽样留痕：编码期失败的路径已失效
        pool/held-out manifest 与统计量，抽样 manifest 若原地留存，盘上
        就是「一次失败的运行 + 一份描述上一轮归属的审计工件」——违反
        「要么全量一致、要么明确缺失」。"""
        first = mr_scenario.run()
        assert first.code == 0, first.stderr
        config = CynosureConfig.model_validate_json(
            mr_scenario.config_path.read_text(encoding="utf-8"),
        )
        sampling_path = Path(config.reward.sampling_manifest_json)
        assert sampling_path.exists()
        # 损坏一卷影像：重跑在编码期失败（已过装配计划与守卫）
        victim = sorted(config.artifacts.dataset_root.glob("*.nii.gz"))[0]
        victim.write_bytes(b"not a nifti")
        second = mr_scenario.run_with_config(config)
        assert second.code == 2
        assert "影像读取失败" in second.stderr
        assert not sampling_path.exists()
        assert not Path(config.reward.real_pool_manifest).exists()

    def test_condition_without_heldout_coverage_rejected(
        self, mr_scenario: MrPrepareScenario,
    ) -> None:
        """逐条件 held-out 覆盖：某条件全部卷落 pool 侧（patient 级二分
        是条件无关的全局洗牌）→ 拒绝。该条件在 held-out 工件缺条目是
        预训练装配期才炸的「下游错误」（ADR-0008-04 轮转条件集要求每条件
        held-out 非空），根因（二分 seed / heldout_fraction）却在本侧——
        按本仓「开工前失败而非训练中途」口径在此拒绝。"""
        fixtures_dir = mr_scenario.work_dir / "fixtures"
        Fixture().write_condition_vocabulary(fixtures_dir)
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        vocabulary_path = fixtures_dir / "condition_vocabulary.json"
        vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
        vocabulary["conditions"].append(MRA_CONDITION)
        vocabulary_path.write_text(json.dumps(vocabulary), encoding="utf-8")
        train_patients = 6
        patients = {f"P{index:02d}" for index in range(train_patients)}
        pool_patients, _ = PatientHeldoutSplit(
            config.schedule.seed, config.reward.heldout_fraction,
        ).split(patients)
        # 稀疏条件的卷只落在一个 pool 侧 patient 上（held-out 侧 0 卷）：
        # 二分归属用真装配类求，测试不复制洗牌逻辑
        sparse_patient = sorted(pool_patients)[0]
        study_uid = f"ST{int(sparse_patient[1:]):02d}"
        extra_rows = [
            {
                "batch_id": "batchmra", "patient_uid": sparse_patient,
                "study_uid": study_uid, "series_id": series_id,
                "modality": "MRA", "plane": "AXIAL",
            }
            for series_id in MRA_SERIES_IDS
        ]
        SyntheticMrRateDataset(
            config.artifacts.dataset_root, train_patients=train_patients,
            extra_metadata_rows=extra_rows,
        ).write()
        for index, series_id in enumerate(MRA_SERIES_IDS):
            volume = np.random.default_rng(index).standard_normal(
                SyntheticMrRateDataset.SERIES_SHAPE,
            ).astype(np.float32)
            nib.save(
                nib.Nifti1Image(volume, RAS_AFFINE),
                config.artifacts.dataset_root
                / f"{study_uid}_{series_id}.nii.gz",
            )
        result = mr_scenario.run_with_config(config)
        assert result.code == 2
        assert "mra/all-planes" in result.stderr
