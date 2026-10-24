"""run 目录样本图版面的行为验证（issue #295）。

全部经 run 目录公共缝驱动（#293 Testing Decisions 预约 seam）：合成
run 目录布局（RunArtifacts.init 产 config 快照 + manifest，样本 .pt 按
落盘契约落位后回填 manifest 样本路径）→ 一键产出 → 断言产物集合 +
布局元数据。CLI usage 面与产物落位在 test_cli.py（既有唯一 seam）。"""

import copy
import json
from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.run_plates import RunSamplePlates
from cynosure.train import RunArtifacts
from cynosure.train.artifacts import BaselineManifest
from tests.conftest import MINIMAL_CONFIG_DICT
from tests.test_sample_plates import anisotropic_volume


def fill_samples(artifacts: RunArtifacts, source_cases: dict[int, str] | None = None) -> None:
    """全条目 baseline/resample 样本落盘并回填 manifest（与 eval/sampling
    落盘契约同构：``samples/stage{stage}/{phase}/{index:04d}.pt``）；
    ``source_cases`` 按条目 index 注入组2 锁定源病例。"""
    manifest = BaselineManifest.load(artifacts.paths.manifest)
    for entry in manifest.entries:
        entry.source_case = (source_cases or {}).get(entry.index)
        for phase in ("baseline", "resample"):
            relative = (
                f"samples/stage{entry.stage}/{phase}/{entry.index:04d}.pt"
            )
            target = artifacts.paths.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(anisotropic_volume(), target)
            setattr(entry, f"{phase}_sample", relative)
    manifest.write(artifacts.paths.manifest)


def synthetic_run_dir(
    tmp_path: Path, config: CynosureConfig,
    source_cases: dict[int, str] | None = None,
) -> tuple[Path, RunArtifacts]:
    """合成 run 目录：init 契约最小集（config 快照 + manifest）+ 样本
    落盘回填——生产 producer 产出布局，测试只摆数据。"""
    artifacts = RunArtifacts.init(config, tmp_path / "run")
    fill_samples(artifacts, source_cases)
    return artifacts.paths.root, artifacts


def brats_run_dir(tmp_path: Path, baseline_samples: int = 4) -> tuple[Path, RunArtifacts]:
    """合成 BraTS 组1 run 目录（四序列轮转）。"""
    data = copy.deepcopy({**MINIMAL_CONFIG_DICT, "fixture_mode": True})
    data["schedule"] = {"seed": 0, "baseline_samples": baseline_samples}
    return synthetic_run_dir(tmp_path, CynosureConfig.model_validate(data))


def cross_modal_run_dir(
    tmp_path: Path, baseline_samples: int = 2,
    source_cases: dict[int, str] | None = None,
) -> tuple[Path, RunArtifacts]:
    """合成 BraTS 组2 run 目录（12 有序对轮转）。"""
    data = copy.deepcopy({**MINIMAL_CONFIG_DICT, "fixture_mode": True})
    data["experiment"] = {"group": "cross-modal"}
    data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
    data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
    data["schedule"] = {
        "seed": 0, "baseline_samples": baseline_samples,
        "milestone_eval_samples": baseline_samples,
    }
    return synthetic_run_dir(
        tmp_path, CynosureConfig.model_validate(data), source_cases,
    )


# fixture 词汇表的两条件等效 spacing（FOV / 统一网格，物理 mm）：
# t1w/axial = (96,64,64)/(32,32,32) = (3,2,2)；flair/axial = (64,64,32)/
# (64,64,32) = (1,1,1)——数值互异使「经词汇表解析」可断言
MR_SPACING_T1W_AXIAL = (3.0, 2.0, 2.0)
MR_SPACING_FLAIR_AXIAL = (1.0, 1.0, 1.0)


def write_mr_fixture_vocabulary(path: Path) -> Path:
    """fixture 小词汇表工件（fixture_mode 装载；数值与生产无关，只承载
    「逐条件 spacing 互异」的解析语义）。"""
    payload = {
        "kind": "mrrate-condition-vocabulary",
        "issue": "test-fixture",
        "upstream_reference": "test-fixture",
        "grid_semantics": "test-fixture",
        "modality_tokens": {
            "t1w": 9, "t2w": 10, "flair": 11, "swi": 20, "mra": 16,
        },
        "conditions": [
            {
                "name": "t1w/axial", "modality": "t1w", "plane": "axial",
                "fov_mm": [96.0, 64.0, 64.0], "fov_source": "fixture",
                "grid_xyz": [32, 32, 32],
            },
            {
                "name": "flair/axial", "modality": "flair", "plane": "axial",
                "fov_mm": [64.0, 64.0, 32.0], "fov_source": "fixture",
                "grid_xyz": [64, 64, 32],
            },
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def mr_run_dir(tmp_path: Path) -> tuple[Path, RunArtifacts]:
    """合成 MR-RATE 组1 run 目录（fixture 词汇表两条件轮转）。"""
    vocabulary = write_mr_fixture_vocabulary(
        tmp_path / "artifacts" / "condition_vocabulary.json",
    )
    data = copy.deepcopy({**MINIMAL_CONFIG_DICT, "fixture_mode": True})
    data.pop("latent_shape")  # MR 线形状锚逐条件派生（#129 互斥携带）
    data["experiment"] = {"group": "modal-label", "dataset": "MR-RATE"}
    data["artifacts"].update({
        "condition_vocabulary_json": str(vocabulary),
        "mrrate_metadata_csv": "artifacts/metadata.csv",
        "mrrate_splits_csv": "artifacts/splits.csv",
        "eval_manifest_csv": "artifacts/eval_manifest.csv",
        "mrrate_data_snapshot": "fixture-snapshot",
    })
    data["preprocessing"] = {"intensity_clip": False}
    data["reward"]["sampling_manifest_json"] = (
        "artifacts/sampling_manifest.json"
    )
    data["schedule"] = {"seed": 0, "baseline_samples": 4}
    return synthetic_run_dir(tmp_path, CynosureConfig.model_validate(data))


class TestRunDirOneShotPlates:
    """run 目录一键产出：落位契约 + 布局元数据物理口径（AC 1/3）。"""

    def test_produces_both_stage_phase_versions_at_contract_paths(
        self, tmp_path: Path,
    ) -> None:
        run_root, artifacts = brats_run_dir(tmp_path)

        report = RunSamplePlates(run_root).render()

        assert report.output_root == artifacts.paths.plots / "samples"
        for phase in ("baseline", "resample"):
            group = report.group(f"1-{phase}")
            assert [plate.png.name for plate in group.plates] == [
                "0000.png", "0001.png", "0002.png", "0003.png",
            ]
            for plate in group.plates:
                assert plate.png.is_file()
                assert plate.png.parent == (
                    artifacts.paths.plots / "samples" / f"1-{phase}"
                )
        assert report.produced == 8

    def test_spacing_resolved_via_vocabulary_channel_not_downgrade(
        self, tmp_path: Path,
    ) -> None:
        """布局元数据断言物理口径：spacing_origin = vocabulary（BraTS
        线单位间距各向同性常量，非体素等比降级口径）。"""
        run_root, _ = brats_run_dir(tmp_path)

        report = RunSamplePlates(run_root).render()

        for group in report.groups:
            for plate in group.plates:
                assert plate.spacing == (1.0, 1.0, 1.0)
                assert plate.spacing_origin == "vocabulary"
                assert plate.axis_codes == "RAS"


class TestPerConditionDeterministicSelection:
    """每条件按 manifest 序确定性选取（AC 2）：轮转 manifest 下缺省每
    条件 1 张取词汇表首张；同 run 重打产物集合与布局元数据一致。"""

    def test_default_takes_first_entry_of_each_condition(
        self, tmp_path: Path,
    ) -> None:
        """8 条目两轮轮转（t1n,t1c,t2w,t2f,t1n,...）：缺省每条件 1 张 →
        每版本取首轮 0000–0003，不取第二轮条目。"""
        run_root, _ = brats_run_dir(tmp_path, baseline_samples=8)

        report = RunSamplePlates(run_root).render()

        for phase in ("baseline", "resample"):
            group = report.group(f"1-{phase}")
            assert [plate.png.name for plate in group.plates] == [
                "0000.png", "0001.png", "0002.png", "0003.png",
            ]
        assert report.produced == 8

    def test_rerender_reproduces_same_product_set_and_metadata(
        self, tmp_path: Path,
    ) -> None:
        run_root, _ = brats_run_dir(tmp_path)

        first = RunSamplePlates(run_root).render()
        second = RunSamplePlates(run_root).render()

        assert [plate.png for group in first.groups for plate in group.plates] == [
            plate.png for group in second.groups for plate in group.plates
        ]
        assert [
            plate.model_dump(exclude={"png", "panels"})
            for group in first.groups for plate in group.plates
        ] == [
            plate.model_dump(exclude={"png", "panels"})
            for group in second.groups for plate in group.plates
        ]

    def test_per_condition_count_is_adjustable(self, tmp_path: Path) -> None:
        """每条件 2 张：两轮轮转 manifest 全量 8 条目被选中；渲染按词汇表
        条件分组序（每条件内部 manifest 序：t1n 0000/0004 → t1c
        0001/0005 → …）。"""
        run_root, _ = brats_run_dir(tmp_path, baseline_samples=8)

        report = RunSamplePlates(run_root).render(per_condition=2)

        group = report.group("1-baseline")
        assert [plate.png.name for plate in group.plates] == [
            f"{index:04d}.png"
            for index in (0, 4, 1, 5, 2, 6, 3, 7)
        ]

    def test_all_takes_every_filled_entry(self, tmp_path: Path) -> None:
        """``all`` 全量：与数量上限无关，全部已填充条目入选。"""
        run_root, _ = brats_run_dir(tmp_path, baseline_samples=8)

        report = RunSamplePlates(run_root).render(per_condition=None)

        assert report.produced == 16
        assert report.group("1-baseline").plates[0].png.name == "0000.png"

    def test_rerender_with_smaller_count_leaves_no_stale_plates(
        self, tmp_path: Path,
    ) -> None:
        """跨参数重打 = 组目录重建：先全量再缺省，上轮多出的 PNG 不残留
        （磁盘产物集合 == 报告产物集合，评审引用锚定固定路径不被脏文件
        污染）。"""
        run_root, _ = brats_run_dir(tmp_path, baseline_samples=8)

        RunSamplePlates(run_root).render(per_condition=None)
        report = RunSamplePlates(run_root).render()

        group_dir = run_root / "plots" / "samples" / "1-baseline"
        on_disk = sorted(path.name for path in group_dir.glob("*.png"))
        assert on_disk == [
            plate.png.name for plate in report.group("1-baseline").plates
        ]
        assert len(on_disk) == 4


class TestUnfilledEntriesSkipped:
    """未填充样本路径的条目跳过并汇总（AC 5）：不产出空图版，缺哪版
    出哪版。"""

    def test_partial_fill_skips_missing_and_reports(
        self, tmp_path: Path,
    ) -> None:
        run_root, artifacts = brats_run_dir(tmp_path)
        # baseline 只回填 0000/0001、resample 只回填 0003（其余删样本文件
        # 并清空路径——模拟训练中断的半成品 run）
        manifest = BaselineManifest.load(artifacts.paths.manifest)
        for entry in manifest.entries:
            for phase, keep in (("baseline", entry.index < 2), ("resample", entry.index == 3)):
                relative = f"samples/stage1/{phase}/{entry.index:04d}.pt"
                path = artifacts.paths.root / relative
                if not keep:
                    path.unlink()
                    setattr(entry, f"{phase}_sample", None)
        manifest.write(artifacts.paths.manifest)

        report = RunSamplePlates(run_root).render()

        assert report.produced == 3
        assert [plate.png.name for plate in report.group("1-baseline").plates] == [
            "0000.png", "0001.png",
        ]
        assert [plate.png.name for plate in report.group("1-resample").plates] == [
            "0003.png",
        ]
        assert [
            (skip.phase, skip.index) for skip in report.skipped
        ] == [
            ("baseline", 2), ("baseline", 3),
            ("resample", 0), ("resample", 1), ("resample", 2),
        ]

    def test_missing_groups_reports_whole_unfilled_phase(
        self, tmp_path: Path,
    ) -> None:
        """缺失版取数面（#296 train 收尾挂接的告警取数面）：某版样本全
        未落盘 → missing_groups 只含该版 label、stages 仍列全部预期
        stage；两版齐备时为空集。"""
        run_root, artifacts = brats_run_dir(tmp_path)
        assert RunSamplePlates(run_root).render().missing_groups() == []

        manifest = BaselineManifest.load(artifacts.paths.manifest)
        for entry in manifest.entries:
            entry.resample_sample = None
        manifest.write(artifacts.paths.manifest)

        report = RunSamplePlates(run_root).render()
        assert report.stages == [1]
        assert report.missing_groups() == ["1-resample"]


class TestMrRateVocabularySpacing:
    """MR-RATE 线条目的 spacing 经词汇表解析（AC 3）：逐条件等效 spacing
    （FOV / 统一网格）进布局元数据，非体素等比降级。"""

    def test_per_condition_spacing_from_vocabulary(
        self, tmp_path: Path,
    ) -> None:
        """轮转 manifest 按词汇表条件分组渲染：t1w/axial（0000/0002，
        spacing (3,2,2)）→ flair/axial（0001/0003，spacing (1,1,1)）。"""
        run_root, _ = mr_run_dir(tmp_path)

        report = RunSamplePlates(run_root).render(per_condition=None)

        plates = report.group("1-baseline").plates
        assert [plate.png.name for plate in plates] == [
            "0000.png", "0002.png", "0001.png", "0003.png",
        ]
        assert plates[0].spacing == MR_SPACING_T1W_AXIAL
        assert plates[1].spacing == MR_SPACING_T1W_AXIAL
        assert plates[2].spacing == MR_SPACING_FLAIR_AXIAL
        assert plates[3].spacing == MR_SPACING_FLAIR_AXIAL
        for plate in plates:
            assert plate.spacing_origin == "vocabulary"
        # 物理 mm 口径落渲染几何：t1w/axial 的 XY panel 纵横比 = Y/X = 2/3
        assert plates[0].panels[0].aspect == (
            MR_SPACING_T1W_AXIAL[1] / MR_SPACING_T1W_AXIAL[0]
        )

    def test_condition_missing_from_vocabulary_is_readable_error(
        self, tmp_path: Path,
    ) -> None:
        """词汇表工件缺条目条件（清单与工件漂移）：可读 ValueError，
        不裸 KeyError。"""
        run_root, artifacts = mr_run_dir(tmp_path)
        manifest = BaselineManifest.load(artifacts.paths.manifest)
        manifest.conditions.append("t2w/axial")
        manifest.entries[0].condition = "t2w/axial"
        manifest.write(artifacts.paths.manifest)

        with pytest.raises(ValueError, match="不在词汇表"):
            RunSamplePlates(run_root).render()


class TestManifestSelfConsistency:
    """manifest 条目与清单词汇表自洽破坏：可读拒绝，不静默丢弃条目。"""

    def test_entry_condition_outside_vocabulary_rejected(
        self, tmp_path: Path,
    ) -> None:
        run_root, artifacts = brats_run_dir(tmp_path)
        manifest = BaselineManifest.load(artifacts.paths.manifest)
        manifest.entries[0].condition = "t9n"
        manifest.write(artifacts.paths.manifest)

        with pytest.raises(ValueError, match="自洽破坏"):
            RunSamplePlates(run_root).render()


class TestCrossModalEntryCaption:
    """组2 锁定源病例条目的条件解析（AC 4）：信息面板内容断言——有序对
    条件、索引、noise seed、source case 随布局元数据可断言。"""

    def test_caption_carries_pair_condition_and_source_case(
        self, tmp_path: Path,
    ) -> None:
        run_root, artifacts = cross_modal_run_dir(
            tmp_path, source_cases={0: "BraTS-GLI-00001-000"},
        )
        manifest = BaselineManifest.load(artifacts.paths.manifest)
        entry = manifest.entries[0]
        assert isinstance(entry.condition, list)  # 组2 场景自检

        report = RunSamplePlates(run_root).render()

        plate = report.group("1-baseline").plates[0]
        assert f"condition: {' → '.join(entry.condition)}" in plate.info_lines
        assert f"index: {entry.index} (stage {entry.stage})" in plate.info_lines
        assert f"noise seed: {entry.noise_seed}" in plate.info_lines
        assert "source case: BraTS-GLI-00001-000" in plate.info_lines
        assert plate.source == f"{entry.index:04d}.pt"

    def test_group1_caption_has_no_source_case_line(
        self, tmp_path: Path,
    ) -> None:
        run_root, _ = brats_run_dir(tmp_path)

        report = RunSamplePlates(run_root).render()

        plate = report.group("1-baseline").plates[0]
        assert not any(
            line.startswith("source case") for line in plate.info_lines
        )
        assert "condition: t1n" in plate.info_lines
