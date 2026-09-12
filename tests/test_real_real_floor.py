"""real-vs-real 地板半分工具单测（wayfinder #79 的地板验证面）。

覆盖：seed 冻结半分的确定性/互斥/覆盖、重复病例拒绝、MR-RATE
manifest 适配（patient_uid 病例键 + stratum 逐格展开 + 路径模板）、
冻结记录 roundtrip、产物喂 MrFidInstrument 的闭环，以及 CLI
fid-floor 子命令。
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from cynosure.eval.mr_fid import (
    FidResult,
    MrFidConfig,
    MrFidInstrument,
)
from cynosure.eval.real_real_floor import (
    DEFAULT_PATH_TEMPLATE,
    HalvesSplitter,
    RealRealFloorSplit,
    SplitFreezeRecord,
)

_MANIFEST_HEADER = [
    "stratum", "sampling_role", "split", "batch_id", "patient_uid",
    "study_uid", "series_id", "modality", "plane", "array_shape",
    "array_spacing_mm", "array_fov_mm",
]


def _row(stratum, patient, study, series, batch="batch00", modality="T1w", plane="AXIAL"):
    return {
        "stratum": stratum,
        "sampling_role": "stratified-n250",
        "split": "val",
        "batch_id": batch,
        "patient_uid": patient,
        "study_uid": study,
        "series_id": series,
        "modality": modality,
        "plane": plane,
        "array_shape": "[8, 8, 8]",
        "array_spacing_mm": "[1.0, 1.0, 1.0]",
        "array_fov_mm": "[8.0, 8.0, 8.0]",
    }


def _write_manifest(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_MANIFEST_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return path


# 6 病例、含同病例多卷/跨格，覆盖半分的病例级语义
_MANIFEST_ROWS = [
    _row("T1w/AXIAL", "p1", "s1", "t1w-raw-axi"),
    _row("T1w/AXIAL", "p2", "s2", "t1w-raw-axi"),
    _row("T1w/AXIAL", "p3", "s3", "t1w-raw-axi"),
    _row("T1w/SAGITTAL", "p3", "s3", "t1w-raw-sag"),  # p3 同 study 第二卷
    _row("T1w/SAGITTAL", "p4", "s4", "t1w-raw-sag"),
    _row("T1w/SAGITTAL", "p5", "s5", "t1w-raw-sag"),
    _row("MRA/ALL-PLANES", "p5", "s5", "mra-raw-axi", modality="MRA"),  # p5 第二模态
    _row("MRA/ALL-PLANES", "p6", "s6", "mra-raw-axi", modality="MRA"),
]


class TestHalvesSplitter:
    def test_seeded_split_is_deterministic(self):
        cases = [f"case{i}" for i in range(10)]
        first = HalvesSplitter().split(cases, seed=42)
        second = HalvesSplitter().split(list(reversed(cases)), seed=42)
        assert first == second  # 入参顺序无关（内部先排序）

    def test_disjoint_and_covering(self):
        cases = [f"case{i}" for i in range(9)]
        half_a, half_b = HalvesSplitter().split(cases, seed=7)
        assert len(half_a) == 5  # ceil(9/2)
        assert len(half_b) == 4
        assert set(half_a).isdisjoint(half_b)
        assert set(half_a) | set(half_b) == set(cases)

    def test_different_seeds_usually_differ(self):
        cases = [f"case{i}" for i in range(20)]
        assert HalvesSplitter().split(cases, seed=1) != HalvesSplitter().split(cases, seed=2)

    def test_duplicate_cases_rejected(self):
        with pytest.raises(ValueError, match="重复"):
            HalvesSplitter().split(["a", "b", "a"], seed=0)


class TestRealRealFloorSplit:
    def test_floor_artifacts_written(self, tmp_path):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        record = RealRealFloorSplit(manifest, out, seed=42).run()
        assert (out / "split_record.json").is_file()
        assert len(record.half_a) == 3  # 6 病例 → 3/3
        # 逐格双侧清单：2 个格 → 2 名 × 2 半 = 4 份 + 2 份病例清单
        for name in (
            "filelist_half_a.txt", "filelist_half_b.txt",
            "filelist_half_a_T1w_AXIAL.txt", "filelist_half_b_T1w_AXIAL.txt",
            "filelist_half_a_T1w_SAGITTAL.txt", "filelist_half_b_T1w_SAGITTAL.txt",
            "filelist_half_a_MRA_ALL-PLANES.txt", "filelist_half_b_MRA_ALL-PLANES.txt",
        ):
            assert (out / name).is_file(), name

    def test_patient_level_halves_no_split_volumes(self, tmp_path):
        """病例级语义：同一 patient 的全部卷（跨 study/模态/格）同半。"""
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        record = RealRealFloorSplit(manifest, out, seed=42).run()
        side_of = {case: "a" for case in record.half_a}
        side_of.update({case: "b" for case in record.half_b})
        for row in _MANIFEST_ROWS:
            expected_file = out / f"filelist_half_{side_of[row['patient_uid']]}_MRA_ALL-PLANES.txt"
            if row["stratum"] != "MRA/ALL-PLANES":
                expected_file = out / (
                    f"filelist_half_{side_of[row['patient_uid']]}_"
                    f"{row['stratum'].replace('/', '_')}.txt"
                )
            lines = expected_file.read_text().splitlines()
            expanded = DEFAULT_PATH_TEMPLATE.format(**row)
            assert expanded in lines

    def test_per_stratum_lists_partition_volumes(self, tmp_path):
        """每格内两半清单互斥且并集 = 全部卷（地板对比的双侧）。"""
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        RealRealFloorSplit(manifest, out, seed=42).run()
        for stratum in {"T1w/AXIAL", "T1w/SAGITTAL", "MRA/ALL-PLANES"}:
            token = stratum.replace("/", "_")
            lines_a = (out / f"filelist_half_a_{token}.txt").read_text().splitlines()
            lines_b = (out / f"filelist_half_b_{token}.txt").read_text().splitlines()
            expected = {
                DEFAULT_PATH_TEMPLATE.format(**row)
                for row in _MANIFEST_ROWS if row["stratum"] == stratum
            }
            assert set(lines_a).isdisjoint(lines_b)
            assert set(lines_a) | set(lines_b) == expected

    def test_freeze_record_roundtrip(self, tmp_path):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        record = RealRealFloorSplit(manifest, out, seed=42).run()
        loaded = SplitFreezeRecord.load(out / "split_record.json")
        assert loaded == record
        assert loaded.validation_source == str(manifest)
        assert loaded.seed == 42

    def test_same_seed_reproduces_same_halves(self, tmp_path):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        first = RealRealFloorSplit(manifest, tmp_path / "f1", seed=99).run()
        second = RealRealFloorSplit(manifest, tmp_path / "f2", seed=99).run()
        assert first.half_a == second.half_a
        assert first.half_b == second.half_b

    def test_missing_manifest_rejected(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest"):
            RealRealFloorSplit(tmp_path / "absent.csv", tmp_path / "out").run()

    def test_empty_manifest_rejected(self, tmp_path):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", [])
        with pytest.raises(ValueError, match="无数据行"):
            RealRealFloorSplit(manifest, tmp_path / "out").run()

    def test_bad_template_field_rejected(self, tmp_path):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        with pytest.raises(ValueError, match="路径模板"):
            RealRealFloorSplit(
                manifest, tmp_path / "out",
                path_template="{nope}/{study_uid}/img/x.nii.gz",
            ).run()


class TestFloorFeedsMrFid:
    """闭环：地板清单对直接喂 MrFidInstrument（一次地板对比）。"""

    def _channel_mean_backbone(self):
        class _Backbone:
            def forward(self, slices):
                return slices.mean(dim=(2, 3))

        return _Backbone()

    def test_floor_pair_runs_as_fid_comparison(self, tmp_path):
        import nibabel as nib
        import zlib

        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        RealRealFloorSplit(manifest, out, seed=42).run()

        # 按清单行把体数据落位到模板路径（模拟 gauss 卷树）
        volumes_root = tmp_path / "volumes"
        for row in _MANIFEST_ROWS:
            target = volumes_root / DEFAULT_PATH_TEMPLATE.format(**row)
            target.parent.mkdir(parents=True, exist_ok=True)
            seed = zlib.crc32(row["series_id"].encode("utf-8"))
            rng = np.random.default_rng(seed)
            array = rng.uniform(0.0, 100.0, (8, 8, 8)).astype(np.float32)
            nib.save(nib.Nifti1Image(array, np.eye(4)), target)

        config = MrFidConfig.model_validate({
            "real_dataset_root": str(volumes_root),
            "real_filelist": str(out / "filelist_half_a_T1w_AXIAL.txt"),
            "real_features_dir": "floor_a",
            "synth_dataset_root": str(volumes_root),
            "synth_filelist": str(out / "filelist_half_b_T1w_AXIAL.txt"),
            "synth_features_dir": "floor_b",
            "num_images": 3,
            "target_shape": [16, 16, 16],
            "center_slices_ratio": 0.5,
            "radimagenet_weights": str(tmp_path / "unused.pt"),
            "output_root": str(tmp_path / "features"),
            "result_json": str(tmp_path / "floor_result.json"),
            "num_workers": 0,
        })
        result = MrFidInstrument(
            config, backbone=self._channel_mean_backbone(),
        ).run(comparison_tag="floor_half_a_vs_half_b")
        loaded = FidResult.load(tmp_path / "floor_result.json")
        assert loaded.comparison_tag == "floor_half_a_vs_half_b"
        assert np.isfinite(result.fid_avg)


class TestFidFloorCli:
    def test_fid_floor_end_to_end(self, tmp_path, cli):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        result = cli.run(
            "fid-floor", "--manifest", str(manifest),
            "--output-dir", str(out),
        )
        assert result.code == 0, result.stderr
        assert "half_a=3、half_b=3" in result.stdout
        assert "split_record.json" in result.stdout
        record = json.loads((out / "split_record.json").read_text())
        assert record["seed"] == 42

    def test_fid_floor_missing_manifest_exit_2(self, tmp_path, cli):
        result = cli.run(
            "fid-floor", "--manifest", str(tmp_path / "absent.csv"),
            "--output-dir", str(tmp_path / "out"),
        )
        assert result.code == 2
        assert "fid-floor 输入契约违反" in result.stderr

    def test_fid_floor_custom_seed_recorded(self, tmp_path, cli):
        manifest = _write_manifest(tmp_path / "eval_manifest.csv", _MANIFEST_ROWS)
        out = tmp_path / "floor"
        result = cli.run(
            "fid-floor", "--manifest", str(manifest),
            "--output-dir", str(out), "--seed", "20260912",
        )
        assert result.code == 0, result.stderr
        assert json.loads((out / "split_record.json").read_text())["seed"] == 20260912
