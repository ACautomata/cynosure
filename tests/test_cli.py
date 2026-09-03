"""CLI seam 测试：train / eval / prepare 三子命令、config 校验字段级错误、
run 目录与工件契约最小版（config 快照 + metrics.jsonl + manifest + checkpoints）。

测试原则（spec「Testing Decisions」）：只断言经 CLI 边界可观测的外部行为。"""

import io
import json
from pathlib import Path

import pytest

from cynosure.cli import CynosureCli
from cynosure.train import IterEvent, MilestoneEvent, RunArtifacts
from tests.conftest import MINIMAL_CONFIG_DICT


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = CynosureCli(argv, stdout, stderr).run()
    return code, stdout.getvalue(), stderr.getvalue()


def write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    import copy

    data = copy.deepcopy(MINIMAL_CONFIG_DICT)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestTrainCommand:
    def test_creates_run_artifact_layout_under_home(
        self, fake_home: Path, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path)
        code, stdout, _ = run_cli(["train", "--config", str(config_path)])
        assert code == 0
        runs = list((fake_home / ".cynosure" / "runs").iterdir())
        assert len(runs) == 1
        run_dir = runs[0]
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "metrics.jsonl").is_file()
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "checkpoints").is_dir()
        assert "run" in stdout  # stdout 报告 run 目录位置

    def test_config_snapshot_is_normalized_input(
        self, fake_home: Path, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path)
        code, _, _ = run_cli(["train", "--config", str(config_path)])
        assert code == 0
        run_dir = next((fake_home / ".cynosure" / "runs").iterdir())
        from cynosure.config import ConfigLoader

        snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        expected = ConfigLoader.load(config_path).model_dump(mode="json")
        assert snapshot == expected

    def test_manifest_contract_minimal(
        self, fake_home: Path, tmp_path: Path,
    ) -> None:
        """manifest 契约最小集：seed、group、conditions、samples。"""
        config_path = write_config(tmp_path)
        code, _, _ = run_cli(["train", "--config", str(config_path)])
        assert code == 0
        run_dir = next((fake_home / ".cynosure" / "runs").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["seed"] == 0
        assert manifest["group"] == "modal-label"
        assert manifest["conditions"] == ["t1n", "t1c", "t2w", "t2f"]
        assert manifest["samples"] == []

    def test_manifest_conditions_for_cross_modal_group(
        self, fake_home: Path, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path, {
            "experiment": {"group": "cross-modal"},
            "artifacts": {"controlnet_ckpt": "ckpts/controlnet.pt"},
        })
        code, _, _ = run_cli(["train", "--config", str(config_path)])
        assert code == 0
        run_dir = next((fake_home / ".cynosure" / "runs").iterdir())
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["conditions"]) == 12

    def test_explicit_run_dir_is_honored(
        self, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path)
        run_dir = tmp_path / "my-run"
        code, _, _ = run_cli([
            "train", "--config", str(config_path), "--run-dir", str(run_dir),
        ])
        assert code == 0
        assert (run_dir / "metrics.jsonl").is_file()

    def test_invalid_config_is_rejected_with_field_level_error(
        self, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        del data["experiment"]["group"]
        config_path.write_text(json.dumps(data), encoding="utf-8")
        code, _, stderr = run_cli(["train", "--config", str(config_path)])
        assert code == 2
        assert "group" in stderr

    def test_fixed_value_violation_reports_field(
        self, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path, {"policy": {"ratio_clip": 1e-3}})
        code, _, stderr = run_cli(["train", "--config", str(config_path)])
        assert code == 2
        assert "ratio_clip" in stderr

    def test_missing_config_file(self, tmp_path: Path) -> None:
        code, _, stderr = run_cli(["train", "--config", str(tmp_path / "nope.json")])
        assert code == 2
        assert "nope.json" in stderr


class TestEvalCommand:
    def test_valid_config_passes(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path)
        code, stdout, _ = run_cli(["eval", "--config", str(config_path)])
        assert code == 0
        assert "modal-label" in stdout

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path, {"schedule": {"n_plateau": 0}})
        code, _, stderr = run_cli(["eval", "--config", str(config_path)])
        assert code == 2
        assert "n_plateau" in stderr

    def test_run_dir_must_exist(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path)
        code, _, stderr = run_cli([
            "eval", "--config", str(config_path),
            "--run-dir", str(tmp_path / "missing"),
        ])
        assert code == 2
        assert "missing" in stderr


class TestMetricsStream:
    """训练指标流契约（spec「产物工件契约」）：单一 JSONL 两类事件。"""

    def _train_run(self, fake_home: Path, tmp_path: Path) -> Path:
        config_path = write_config(tmp_path)
        code, _, _ = run_cli(["train", "--config", str(config_path)])
        assert code == 0
        return next((fake_home / ".cynosure" / "runs").iterdir())

    def test_iter_event_roundtrip(self, fake_home: Path, tmp_path: Path) -> None:
        artifacts = RunArtifacts(RunArtifacts.layout(
            self._train_run(fake_home, tmp_path),
        ))
        event = IterEvent(
            iteration=0, anchor_eval_reward=-1.5, intra_group_reward_std=0.3,
            heldout_auc=0.62, loss={"policy": 0.01}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, lr=2e-6, elapsed_s=12.3,
        )
        artifacts.append_event(event)
        events = artifacts.read_events()
        assert len(events) == 1
        assert events[0]["event"] == "iter"
        assert events[0]["iteration"] == 0
        assert events[0]["anchor_eval_reward"] == pytest.approx(-1.5)

    def test_milestone_event_roundtrip(self, fake_home: Path, tmp_path: Path) -> None:
        artifacts = RunArtifacts(RunArtifacts.layout(
            self._train_run(fake_home, tmp_path),
        ))
        event = MilestoneEvent(iteration=50, fid=12.3, early_stop=False)
        artifacts.append_event(event)
        events = artifacts.read_events()
        assert events[0]["event"] == "milestone"
        assert events[0]["fid"] == pytest.approx(12.3)
        assert events[0]["ssim"] is None  # 跨模态组才带 SSIM/MAE

    def test_events_coexist_in_single_stream(self, fake_home: Path, tmp_path: Path) -> None:
        artifacts = RunArtifacts(RunArtifacts.layout(
            self._train_run(fake_home, tmp_path),
        ))
        artifacts.append_event(IterEvent(
            iteration=0, anchor_eval_reward=0.0, intra_group_reward_std=0.0,
            heldout_auc=0.5, loss={}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, lr=2e-6, elapsed_s=1.0,
        ))
        artifacts.append_event(MilestoneEvent(iteration=50, fid=1.0))
        assert [e["event"] for e in artifacts.read_events()] == ["iter", "milestone"]


class TestPrepareCommand:
    def test_valid_config_passes_and_reports_artifacts(
        self, tmp_path: Path,
    ) -> None:
        config_path = write_config(tmp_path)
        code, stdout, _ = run_cli(["prepare", "--config", str(config_path)])
        assert code == 0
        assert "real_pool.json" in stdout
        assert "heldout_real.json" in stdout
        assert "channel_stats.json" in stdout

    def test_invalid_config_rejected(self, tmp_path: Path) -> None:
        config_path = write_config(tmp_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        del data["reward"]["disc_batch_size_k"]
        config_path.write_text(json.dumps(data), encoding="utf-8")
        code, _, stderr = run_cli(["prepare", "--config", str(config_path)])
        assert code == 2
        assert "disc_batch_size_k" in stderr
