"""Baseline manifest 生成与 RL 后重采复用（ticket #25 AC）。

契约面：manifest 条目（seed、条件、噪声种子）在 run 创建时确定性生成；
baseline 采样（冻结只采一次）与 RL 后重采（同 seed 同条件）先后把样本
路径写回**同一 manifest**——两侧逐条目配对，差异唯一归因于 RL。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES
from cynosure.train import BaselineManifest
from tests.conftest import CliSession
from tests.test_train_loop import TrainingLoopScenario


class ManifestRun:
    """一次训练 run 的 manifest 观测面（BaselineManifest 工件装载）。"""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.manifest = BaselineManifest.load(run_dir / "manifest.json")

    def volume(self, relative: str) -> torch.Tensor:
        return torch.load(
            self.run_dir / relative, map_location="cpu", weights_only=True,
        )


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


class TestBaselineManifestGeneration:
    def test_train_fills_baseline_and_resample_sample_paths(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """CLI train 一次运行后：每个条目都有 baseline 与 resample 两条
        样本路径，文件存在、可装载为像素体 [X, Y, Z]。"""
        scenario.write_inputs()
        result = scenario.train()
        assert result.code == 0, result.stderr
        run = ManifestRun(scenario.run_dir)
        assert len(run.manifest.entries) == 4
        for entry in run.manifest.entries:
            assert entry.baseline_sample is not None
            assert entry.resample_sample is not None
            baseline = run.volume(entry.baseline_sample)
            resample = run.volume(entry.resample_sample)
            assert baseline.shape == (64, 64, 32)
            assert resample.shape == (64, 64, 32)
            assert torch.isfinite(baseline).all() and torch.isfinite(resample).all()

    def test_baseline_and_resample_share_entries_pairwise(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """重采复用同一 manifest：两侧逐条目 index/condition/noise_seed
        完全一致（同 seed 同条件），样本路径各自独立。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        run = ManifestRun(scenario.run_dir)
        conditions = [MODALITIES[index % len(MODALITIES)] for index in range(4)]
        for index, entry in enumerate(run.manifest.entries):
            assert entry.index == index
            assert entry.condition == conditions[index]
            assert entry.baseline_sample != entry.resample_sample
            assert entry.baseline_sample == f"samples/stage1/baseline/{index:04d}.pt"
            assert entry.resample_sample == f"samples/stage1/resample/{index:04d}.pt"

    def test_baseline_volume_is_frozen_policy_resample_is_trained(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """baseline 由冻结初始 policy 产出、resample 由 RL 后 policy 产出：
        提高学习率放大单次梯度步的权重移动，两侧体必须可区分。"""
        scenario.write_inputs(seed=1)
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["policy"]["policy_lr"] = 1e-2  # 放大 RL 一步的权重移动
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        assert scenario.train().code == 0, scenario.run_dir
        run = ManifestRun(scenario.run_dir)
        entry = run.manifest.entries[0]
        baseline = run.volume(entry.baseline_sample)
        resample = run.volume(entry.resample_sample)
        assert not torch.equal(baseline, resample)

    def test_manifest_noise_seeds_deterministic_across_runs(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """同 seed 两次 run 的 manifest 条目（含噪声种子）逐位一致——
        重采与里程碑可比性的前提；不同 seed 则噪声种子不同。"""
        seeds: dict[int, list[int]] = {}
        for run_index, seed in enumerate((0, 0, 1)):
            scenario = TrainingLoopScenario(cli, tmp_path / f"run{run_index}")
            scenario.write_inputs(seed=seed)
            data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
            data["schedule"]["max_iterations"] = 1
            scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
            assert scenario.train().code == 0
            run = ManifestRun(scenario.run_dir)
            seeds[run_index] = [entry.noise_seed for entry in run.manifest.entries]
        assert seeds[0] == seeds[1]  # 同 seed → 同清单
        assert seeds[0] != seeds[2]  # 不同 seed → 不同噪声种子


class TestCrossModalManifest:
    def test_source_case_locked_at_baseline_and_shared_by_resample(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """组2：baseline 采样期锁定源病例并写回 manifest；resample 条目
        读回同一病例（配对 SSIM/MAE 与重采的参照一致性）。"""
        scenario.write_inputs(group="cross-modal")
        assert scenario.train().code == 0, scenario.run_dir
        entries = ManifestRun(scenario.run_dir).manifest.entries
        assert all(entry.source_case for entry in entries)
        assert all(
            entry.baseline_sample and entry.resample_sample for entry in entries
        )
        conditions = [tuple(entry.condition) for entry in entries]
        assert all(
            len(pair) == 2 and pair[0] != pair[1] for pair in conditions
        )  # [源序列, 目标序列] 有序对
