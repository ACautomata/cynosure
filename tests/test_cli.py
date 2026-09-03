"""CLI seam 测试：train / eval / prepare 三子命令、config 校验字段级错误、
run 目录与工件契约最小版（config 快照 + metrics.jsonl + manifest + checkpoints）。

测试原则（spec「Testing Decisions」）：只断言经 CLI 边界可观测的外部行为。"""

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from cynosure.cli import CynosureCli
from cynosure.config import ConfigLoader
from cynosure.train import IterEvent, MilestoneEvent, RunArtifacts
from tests.conftest import MINIMAL_CONFIG_DICT


@dataclass
class CliResult:
    """一次进程内 CLI 调用的外部可观测结果。"""

    code: int
    stdout: str
    stderr: str


class CliSession:
    """CLI 会话：向 cynosure 命令行提交 argv 并捕获输出。"""

    def run(self, *args: str) -> CliResult:
        stdout, stderr = io.StringIO(), io.StringIO()
        code = CynosureCli(list(args), stdout, stderr).run()
        return CliResult(code, stdout.getvalue(), stderr.getvalue())

    def write_config(self, directory: Path, overrides: dict | None = None) -> Path:
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
        path = directory / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def train(self, config_path: Path, run_dir: Path | None = None) -> CliResult:
        argv = ["train", "--config", str(config_path)]
        if run_dir is not None:
            argv += ["--run-dir", str(run_dir)]
        return self.run(*argv)

    def sole_run_directory(self, home: Path) -> Path:
        """本会话（$HOME 下）唯一一次 train 产出的 run 目录。"""
        return next((home / ".cynosure" / "runs").iterdir())


@pytest.fixture
def cli() -> CliSession:
    return CliSession()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestTrainCommand:
    def test_creates_run_artifact_layout_under_home(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        result = cli.train(cli.write_config(tmp_path))
        assert result.code == 0
        run_dir = cli.sole_run_directory(fake_home)
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "metrics.jsonl").is_file()
        assert (run_dir / "manifest.json").is_file()
        assert (run_dir / "checkpoints").is_dir()
        assert "run" in result.stdout  # stdout 报告 run 目录位置

    def test_config_snapshot_is_normalized_input(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path)
        assert cli.train(config_path).code == 0
        snapshot = json.loads(
            cli.sole_run_directory(fake_home).joinpath("config.json").read_text(
                encoding="utf-8",
            ),
        )
        expected = ConfigLoader.load(config_path).model_dump(mode="json")
        assert snapshot == expected

    def test_manifest_contract_minimal(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        """manifest 契约最小集：seed、group、conditions、samples。"""
        assert cli.train(cli.write_config(tmp_path)).code == 0
        manifest = json.loads(
            cli.sole_run_directory(fake_home)
            .joinpath("manifest.json")
            .read_text(encoding="utf-8"),
        )
        assert manifest["seed"] == 0
        assert manifest["group"] == "modal-label"
        assert manifest["conditions"] == ["t1n", "t1c", "t2w", "t2f"]
        assert manifest["samples"] == []

    def test_manifest_conditions_for_cross_modal_group(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path, {
            "experiment": {"group": "cross-modal"},
            "artifacts": {"controlnet_ckpt": "ckpts/controlnet.pt"},
        })
        assert cli.train(config_path).code == 0
        manifest = json.loads(
            cli.sole_run_directory(fake_home)
            .joinpath("manifest.json")
            .read_text(encoding="utf-8"),
        )
        assert len(manifest["conditions"]) == 12

    def test_explicit_run_dir_is_honored(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "my-run"
        result = cli.train(cli.write_config(tmp_path), run_dir=run_dir)
        assert result.code == 0
        assert (run_dir / "metrics.jsonl").is_file()

    def test_existing_run_dir_is_not_silently_overwritten(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """每次运行一个 run 目录的隔离契约：重跑同一目录被拒。"""
        run_dir = tmp_path / "my-run"
        assert cli.train(cli.write_config(tmp_path), run_dir=run_dir).code == 0
        (run_dir / "metrics.jsonl").write_text("sentinel\n", encoding="utf-8")
        result = cli.train(cli.write_config(tmp_path), run_dir=run_dir)
        assert result.code == 2
        assert "已存在" in result.stderr
        assert (run_dir / "metrics.jsonl").read_text(encoding="utf-8") == "sentinel\n"

    def test_invalid_config_is_rejected_with_field_level_error(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        del data["experiment"]["group"]
        config_path.write_text(json.dumps(data), encoding="utf-8")
        result = cli.train(config_path)
        assert result.code == 2
        assert "group" in result.stderr

    def test_fixed_value_violation_reports_field(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path, {"policy": {"ratio_clip": 1e-3}})
        result = cli.train(config_path)
        assert result.code == 2
        assert "ratio_clip" in result.stderr

    def test_missing_config_file(self, cli: CliSession, tmp_path: Path) -> None:
        result = cli.train(tmp_path / "nope.json")
        assert result.code == 2
        assert "nope.json" in result.stderr


class TestEvalCommand:
    def test_valid_config_passes(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        result = cli.run("eval", "--config", str(cli.write_config(tmp_path)))
        assert result.code == 0
        assert "modal-label" in result.stdout

    def test_invalid_config_rejected(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path, {"schedule": {"n_plateau": 0}})
        result = cli.run("eval", "--config", str(config_path))
        assert result.code == 2
        assert "n_plateau" in result.stderr

    def test_run_dir_must_exist(self, cli: CliSession, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        result = cli.run(
            "eval", "--config", str(cli.write_config(tmp_path)),
            "--run-dir", str(missing),
        )
        assert result.code == 2
        assert "missing" in result.stderr


class TestPrepareCommand:
    def test_valid_config_passes_and_reports_artifacts(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        result = cli.run("prepare", "--config", str(cli.write_config(tmp_path)))
        assert result.code == 0
        assert "real_pool.json" in result.stdout
        assert "heldout_real.json" in result.stdout
        assert "channel_stats.json" in result.stdout

    def test_invalid_config_rejected(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        config_path = cli.write_config(tmp_path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        del data["reward"]["disc_batch_size_k"]
        config_path.write_text(json.dumps(data), encoding="utf-8")
        result = cli.run("prepare", "--config", str(config_path))
        assert result.code == 2
        assert "disc_batch_size_k" in result.stderr


class TestMetricsStream:
    """训练指标流契约（spec「产物工件契约」）：单一 JSONL 两类事件。"""

    def _trained_artifacts(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> RunArtifacts:
        assert cli.train(cli.write_config(tmp_path)).code == 0
        return RunArtifacts(RunArtifacts.layout(cli.sole_run_directory(fake_home)))

    def test_iter_event_roundtrip(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(cli, fake_home, tmp_path)
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

    def test_milestone_event_roundtrip(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(cli, fake_home, tmp_path)
        event = MilestoneEvent(iteration=50, fid=12.3, early_stop=False)
        artifacts.append_event(event)
        events = artifacts.read_events()
        assert events[0]["event"] == "milestone"
        assert events[0]["fid"] == pytest.approx(12.3)
        assert events[0]["ssim"] is None  # 跨模态组才带 SSIM/MAE

    def test_events_coexist_in_single_stream(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(cli, fake_home, tmp_path)
        artifacts.append_event(IterEvent(
            iteration=0, anchor_eval_reward=0.0, intra_group_reward_std=0.0,
            heldout_auc=0.5, loss={}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, lr=2e-6, elapsed_s=1.0,
        ))
        artifacts.append_event(MilestoneEvent(iteration=50, fid=1.0))
        assert [e["event"] for e in artifacts.read_events()] == ["iter", "milestone"]
