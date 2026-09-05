"""CLI seam 测试：train / eval / prepare 三子命令、config 校验字段级错误、
run 目录与工件契约最小版（config 快照 + metrics.jsonl + manifest + checkpoints）。

测试原则（spec「Testing Decisions」）：只断言经 CLI 边界可观测的外部行为。"""

import copy
import json
import threading
from pathlib import Path

import pytest
from pydantic import ValidationError

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.train import IterEvent, MilestoneEvent, RunArtifacts
from tests.conftest import CliSession, MINIMAL_CONFIG_DICT


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestTrainCommand:
    """run 目录契约（直接驱动 RunArtifacts；CLI train 的训练全链路由
    test_train_loop.py 的 fixture 场景覆盖——train 命令自 #21 起真实执行
    训练循环，需要可装载的网络工件）。"""

    @staticmethod
    def _initialized_artifacts(config: CynosureConfig, tmp_path: Path) -> RunArtifacts:
        return RunArtifacts.init(config, tmp_path / "run")

    def test_creates_run_artifact_layout_under_home(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        artifacts = self._initialized_artifacts(config, tmp_path)
        assert (artifacts.paths.config_snapshot).is_file()
        assert (artifacts.paths.metrics).is_file()
        assert (artifacts.paths.manifest).is_file()
        assert (artifacts.paths.checkpoints).is_dir()

    def test_config_snapshot_is_normalized_input(
        self, valid_config_json: Path, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        artifacts = self._initialized_artifacts(config, tmp_path)
        snapshot = json.loads(
            artifacts.paths.config_snapshot.read_text(encoding="utf-8"),
        )
        expected = ConfigLoader.load(valid_config_json).model_dump(mode="json")
        assert snapshot == expected

    def test_manifest_contract_minimal(self, valid_config_dict: dict, tmp_path: Path) -> None:
        """manifest 契约最小集：seed、group、conditions + 采样条目
        （seed、条件、噪声种子；样本路径由 baseline/重采先后填入）。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        artifacts = self._initialized_artifacts(config, tmp_path)
        manifest = json.loads(artifacts.paths.manifest.read_text(encoding="utf-8"))
        assert manifest["seed"] == 0
        assert manifest["group"] == "modal-label"
        assert manifest["conditions"] == ["t1n", "t1c", "t2w", "t2f"]
        entries = manifest["entries"]
        assert len(entries) == config.schedule.baseline_samples  # 每条目一个采样位
        assert [entry["index"] for entry in entries[:4]] == [0, 1, 2, 3]
        assert [entry["condition"] for entry in entries[:4]] == [
            "t1n", "t1c", "t2w", "t2f",
        ]  # 四序列轮转
        assert all(
            isinstance(entry["noise_seed"], int) and entry["noise_seed"] > 0
            for entry in entries
        )
        assert all(entry["baseline_sample"] is None for entry in entries)
        assert all(entry["resample_sample"] is None for entry in entries)

    def test_manifest_conditions_for_cross_modal_group(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["experiment"] = {"group": "cross-modal"}
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"] = {"seed": 0, "milestone_eval_samples": 12}
        config = CynosureConfig.model_validate(data)
        artifacts = self._initialized_artifacts(config, tmp_path)
        manifest = json.loads(
            artifacts.paths.manifest.read_text(encoding="utf-8"),
        )
        assert len(manifest["conditions"]) == 12

    def test_explicit_run_dir_is_honored(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        run_dir = tmp_path / "my-run"
        artifacts = RunArtifacts.init(config, run_dir)
        assert (artifacts.paths.metrics).is_file()

    def test_existing_run_dir_is_not_silently_overwritten(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        """每次运行一个 run 目录的隔离契约：重跑同一目录被拒。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        run_dir = tmp_path / "my-run"
        RunArtifacts.init(config, run_dir)
        (run_dir / "metrics.jsonl").write_text("sentinel\n", encoding="utf-8")
        with pytest.raises(FileExistsError):
            RunArtifacts.init(config, run_dir)
        assert (run_dir / "metrics.jsonl").read_text(
            encoding="utf-8",
        ) == "sentinel\n"

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


class TestSingleProcessGuard:
    """单进程训练循环（#21 tracer bullet）的 rank 守卫：FSDP 梯度聚合与
    rank 0 归并落盘由 orchestration ticket 交付前，非 0 rank 显式拒绝——
    否则每个 rank 各自跑完整循环，重复追加 iter 事件、覆写同一 checkpoint
    文件名（RunArtifacts 的 rank 0 写盘契约被静默破坏）。"""

    def test_nonzero_rank_rejected_even_when_run_dir_ready(
        self, cli: CliSession, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run 目录已就绪（rank 0 已创建）也不进训练循环：显式拒绝、
        消息指明单进程版边界——而非沿用「等待 rank 0」的 barrier 语义
        （barrier 之后各 rank 重复训练才是要防的故障）。"""
        monkeypatch.setenv("RANK", "3")
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "config.json").write_text("{}", encoding="utf-8")
        result = cli.train(cli.write_config(tmp_path), run_dir=run_root)
        assert result.code == 2
        assert "单进程" in result.stderr
        assert "训练输入契约违反" not in result.stderr  # 未进训练循环

    def test_rank0_passes_guard_into_training(
        self, cli: CliSession, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rank 0 通过守卫进入训练装配（生产 config 在装配期因工件缺失
        得到训练契约错误——走到该错误证明守卫未拦截；失败后未产出工件
        的 run 目录已被回滚）。"""
        monkeypatch.setenv("RANK", "0")
        result = cli.train(cli.write_config(tmp_path), run_dir=tmp_path / "run")
        assert "训练输入契约违反" in result.stderr


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
    def test_production_mode_rejected_until_vae_ticket(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """fixture_mode=false 的生产 prepare 须 MONAI VAE 预编码（后续 ticket
        交付），当前显式拒绝而非静默产出。"""
        result = cli.run("prepare", "--config", str(cli.write_config(tmp_path)))
        assert result.code == 2
        assert "fixture_mode" in result.stderr

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


class TestTrajectoryDiagnosticFlag:
    """train --dump-trajectory（fixture 诊断开关）：生产 config 显式拒绝，
    不建 run 目录、不产出残缺工件（spec「产物工件契约」诊断模式）。"""

    def test_production_mode_rejected_before_run_dir_created(
        self, cli: CliSession, fake_home: Path, tmp_path: Path,
    ) -> None:
        result = cli.run(
            "train", "--config", str(cli.write_config(tmp_path)),
            "--run-dir", str(tmp_path / "run"), "--dump-trajectory",
        )
        assert result.code == 2
        assert "fixture_mode" in result.stderr
        assert not (tmp_path / "run").exists()  # fail fast：不留半初始化 run 目录


class TestDistributedRunDir:
    """torchrun 多 rank（RANK env）下的 run 目录初始化：rank 0 创建、
    其余 rank 轮询等待采用（spec：多 rank 下指标由 rank 0 归并写出）。"""

    @pytest.fixture
    def config(self) -> CynosureConfig:
        return CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))

    def test_rank0_creates_run_dir(
        self, config: CynosureConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "8")
        artifacts = RunArtifacts.init(config, tmp_path / "run")
        assert artifacts.paths.config_snapshot.is_file()

    def test_nonzero_rank_adopts_dir_created_by_rank0(
        self, config: CynosureConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """非 0 rank 不创建：rank 0 先行建好目录时，等待方直接采用、不再报 FileExistsError。"""
        monkeypatch.setenv("RANK", "3")
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "config.json").write_text("{}", encoding="utf-8")  # rank 0 已创建
        artifacts = RunArtifacts.init(config, run_root)
        assert artifacts.paths.config_snapshot.is_file()

    def test_nonzero_rank_times_out_when_rank0_never_creates(
        self, config: CynosureConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rank 0 迟迟不建目录 → 等待超时显式失败，而非各 rank 静默自建、分裂 run。"""
        monkeypatch.setenv("RANK", "3")
        with pytest.raises(TimeoutError):
            RunArtifacts.init(config, tmp_path / "never", wait_timeout_s=0.1)

    def test_nonzero_rank_waits_then_adopts(
        self, config: CynosureConfig, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """rank 0 稍后创建：等待方阻塞至目录出现后采用，且自身不落盘。"""
        monkeypatch.setenv("RANK", "3")
        run_root = tmp_path / "run"

        def _create_as_rank0() -> None:
            run_root.mkdir()
            (run_root / "config.json").write_text("{}", encoding="utf-8")

        timer = threading.Timer(0.2, _create_as_rank0)
        timer.start()
        try:
            artifacts = RunArtifacts.init(config, run_root, wait_timeout_s=5.0)
        finally:
            timer.join()
        assert artifacts.paths.root == run_root
        assert artifacts.paths.config_snapshot.is_file()

    def test_train_under_torchrun_requires_explicit_run_dir(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """默认 run 目录按进程时间戳生成、无法跨 rank 对齐：torchrun 下
        （rank 0）也必须显式 --run-dir（非 0 rank 已被单进程守卫更早拒绝）。"""
        monkeypatch.setenv("RANK", "0")
        result = cli.train(cli.write_config(tmp_path))
        assert result.code == 2
        assert "--run-dir" in result.stderr


class TestMetricsStream:
    """训练指标流契约（spec「产物工件契约」）：单一 JSONL 两类事件。"""

    def _trained_artifacts(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> RunArtifacts:
        config = CynosureConfig.model_validate(valid_config_dict)
        return RunArtifacts.init(config, tmp_path / "run")

    def test_iter_event_roundtrip(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(valid_config_dict, tmp_path)
        event = IterEvent(
            iteration=0, modality="t1n", anchor_eval_reward=-1.5,
            intra_group_reward_std=0.3,
            heldout_auc=0.62, loss={"policy": 0.01}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, buffer_base_occupied=32,
            buffer_recent_occupied=12, lr=2e-6, elapsed_s=12.3,
        )
        artifacts.append_event(event)
        events = artifacts.read_events()
        assert len(events) == 1
        assert events[0]["event"] == "iter"
        assert events[0]["iteration"] == 0
        assert events[0]["anchor_eval_reward"] == pytest.approx(-1.5)

    def test_milestone_event_roundtrip(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(valid_config_dict, tmp_path)
        event = MilestoneEvent(iteration=50, fid=12.3, early_stop=False)
        artifacts.append_event(event)
        events = artifacts.read_events()
        assert events[0]["event"] == "milestone"
        assert events[0]["fid"] == pytest.approx(12.3)
        assert events[0]["ssim"] is None  # 跨模态组才带 SSIM/MAE

    @staticmethod
    def _iter_event(**overrides: object) -> IterEvent:
        values: dict = dict(
            iteration=0, modality="t1n", anchor_eval_reward=-1.5,
            intra_group_reward_std=0.3,
            heldout_auc=0.62, loss={"policy": 0.01}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, buffer_base_occupied=32,
            buffer_recent_occupied=12, lr=2e-6, elapsed_s=12.3,
        )
        values.update(overrides)
        return IterEvent(**values)

    def test_iter_event_rejects_non_finite_reward(self) -> None:
        """NaN/Inf 事件会以非标准 token 落盘 JSONL、严格消费方拒读：必须构造期拒绝。"""
        with pytest.raises(ValidationError):
            self._iter_event(anchor_eval_reward=float("nan"))

    def test_iter_event_rejects_non_finite_loss_value(self) -> None:
        with pytest.raises(ValidationError):
            self._iter_event(loss={"policy": float("inf")})

    def test_milestone_event_rejects_non_finite_fid(self) -> None:
        with pytest.raises(ValidationError):
            MilestoneEvent(iteration=50, fid=float("inf"))

    def test_events_coexist_in_single_stream(
        self, valid_config_dict: dict, tmp_path: Path,
    ) -> None:
        artifacts = self._trained_artifacts(valid_config_dict, tmp_path)
        artifacts.append_event(IterEvent(
            iteration=0, modality="t1n", anchor_eval_reward=0.0,
            intra_group_reward_std=0.0,
            heldout_auc=0.5, loss={}, buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5, buffer_base_occupied=32,
            buffer_recent_occupied=0, lr=2e-6, elapsed_s=1.0,
        ))
        artifacts.append_event(MilestoneEvent(iteration=50, fid=1.0))
        assert [e["event"] for e in artifacts.read_events()] == ["iter", "milestone"]
