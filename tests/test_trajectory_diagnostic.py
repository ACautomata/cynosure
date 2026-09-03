"""轨迹诊断工件测试（ticket #19）：train --dump-trajectory 经 fixture 端到端，
只断言诊断工件的外部行为——η=0 parity 双列、sigma 日程锚、log-prob 对、
双样本分布统计量（测试面 #1–#3 的载体，spec「Testing Decisions」）。"""

import json
import math
from pathlib import Path

import pytest

from cynosure.fixtures import Fixture
from tests.conftest import CliSession, CliResult

RUN_DIR = "diag-run"


class DiagnosticScenario:
    """一次 train --dump-trajectory 场景：fixture 网络 + fixture config + CLI。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self._cli = cli
        self._tmp_path = tmp_path
        self._config_path = tmp_path / "config.json"

    def run(self, *, eta: float = 0.7, seed: int = 0) -> CliResult:
        fixture = Fixture()
        fixture.write_artifacts(self._tmp_path / "fixtures")
        config = fixture.config(self._tmp_path / "fixtures")
        config.policy.sde_eta = eta
        config.schedule.seed = seed
        self._config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        return self._cli.run(
            "train", "--config", str(self._config_path),
            "--run-dir", str(self._tmp_path / RUN_DIR), "--dump-trajectory",
        )

    def report(self) -> dict:
        return json.loads(
            (self._tmp_path / RUN_DIR / "trajectory.json").read_text(
                encoding="utf-8",
            ),
        )


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> DiagnosticScenario:
    return DiagnosticScenario(cli, tmp_path)


class TestDiagnosticArtifact:
    """诊断工件契约：结构、sigma 日程锚、扰动步、样本量（AC 4/5）。"""

    def test_dump_writes_artifact_into_run_dir(
        self, scenario: DiagnosticScenario,
    ) -> None:
        result = scenario.run()
        assert result.code == 0
        assert (scenario._tmp_path / RUN_DIR / "trajectory.json").is_file()
        assert "trajectory.json" in result.stdout

    def test_schedule_anchors_monai_actual_output(
        self, scenario: DiagnosticScenario,
    ) -> None:
        """sigma 日程 = MONAI set_timesteps 实际输出（3 步 @2048 数值锚，
        timestep transform 生效；实际 scale=1.0，非 config 字面 1.4）。"""
        assert scenario.run().code == 0
        report = scenario.report()
        assert report["schedule_timesteps"] == [1000.0, 442.0, 165.0]
        assert report["input_img_size_numel"] == 2048

    def test_perturbation_steps_follow_m(self, scenario: DiagnosticScenario) -> None:
        """fixture M={1}：扰动只发生在被优化训练步（3 步日程取 {1}）。"""
        assert scenario.run().code == 0
        report = scenario.report()
        assert report["perturbation_steps"] == [1]
        assert report["eta"] == pytest.approx(0.7)
        assert report["s_max"] == pytest.approx(0.999)

    def test_trajectory_columns_cover_every_step(
        self, scenario: DiagnosticScenario,
    ) -> None:
        assert scenario.run().code == 0
        report = scenario.report()
        for column in ("anchor_trajectory", "monai_reference_trajectory"):
            steps = report[column]
            assert [entry["step_index"] for entry in steps] == [0, 1, 2, 3]
            assert [entry["timestep"] for entry in steps] == [1000.0, 442.0, 165.0, 0.0]
            for entry in steps:
                assert math.isfinite(entry["mean"])
                assert entry["std"] >= 0.0
                assert entry["min"] <= entry["mean"] <= entry["max"]
                assert len(entry["sha256"]) == 64

    def test_logprob_pairs_cover_m_times_g_times_noise(
        self, scenario: DiagnosticScenario,
    ) -> None:
        """|M| × 噪声数 × G = 1 × 4 × 12 组对，每组 recorded == recomputed。"""
        assert scenario.run().code == 0
        pairs = scenario.report()["logprob_pairs"]
        assert len(pairs) == 1 * 4 * 12
        assert {(p["step_index"], p["noise_index"], p["direction"]) for p in pairs} == {
            (1, noise, direction) for noise in range(4) for direction in range(12)
        }
        for pair in pairs:
            assert math.isfinite(pair["recorded"])
            assert pair["recorded"] == pair["recomputed"]

    def test_terminal_sample_stats_contract(
        self, scenario: DiagnosticScenario,
    ) -> None:
        """对照 = 4 条 Anchor 终点；扰动 = 4 噪声 × G=12 = 48 条终点。"""
        assert scenario.run().code == 0
        report = scenario.report()
        control = report["control_terminals"]
        perturbed = report["perturbed_terminals"]
        assert control["count"] == 4
        assert perturbed["count"] == 48
        for column in (control, perturbed):
            assert len(column["channel_mean"]) == len(column["channel_std"]) == 4
            assert all(math.isfinite(v) for v in column["channel_mean"])
            assert all(v >= 0.0 for v in column["channel_std"])

    def test_missing_modality_mapping_rejected(
        self, scenario: DiagnosticScenario,
    ) -> None:
        """modality_mapping 是 spec 输入物：工件缺失显式失败，不静默回退
        到代码内副本。"""
        fixture_dir = scenario._tmp_path / "fixtures"
        Fixture().write_artifacts(fixture_dir)
        (fixture_dir / "modality_mapping.json").unlink()
        config = Fixture().config(fixture_dir)
        config.policy.sde_eta = 0.7
        scenario._config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        result = scenario._cli.run(
            "train", "--config", str(scenario._config_path),
            "--run-dir", str(scenario._tmp_path / RUN_DIR), "--dump-trajectory",
        )
        assert result.code == 2
        assert "modality" in result.stderr


class TestEtaZeroParity:
    """测试面 #1（AC）：η=0 时 policy 采样路径与直接调 MONAI 确定性步
    逐位一致——经诊断工件双列断言 per-step 轨迹。"""

    def test_policy_trajectory_equals_monai_reference(
        self, scenario: DiagnosticScenario,
    ) -> None:
        assert scenario.run(eta=0.7).code == 0  # parity 双列与 η 取值无关
        report = scenario.report()
        assert report["anchor_trajectory"] == report["monai_reference_trajectory"]

    def test_eta0_control_terminals_identical(
        self, scenario: DiagnosticScenario,
    ) -> None:
        """η=0 对照：扰动 + 续跑终点分布与 Anchor 终点一致（无噪声可注入，
        仅 batch 组织的 fp32 归约序差异），log-prob 对为空（确定性步无密度）。"""
        assert scenario.run(eta=0.0).code == 0
        report = scenario.report()
        control = report["control_terminals"]
        perturbed = report["perturbed_terminals"]
        assert control["count"] == 4
        assert perturbed["count"] == 48
        for channel in range(4):
            assert abs(
                perturbed["channel_mean"][channel] - control["channel_mean"][channel],
            ) < 1e-6
            assert abs(
                perturbed["channel_std"][channel] / control["channel_std"][channel] - 1.0,
            ) < 1e-4
        assert report["logprob_pairs"] == []
        assert report["eta"] == pytest.approx(0.0)


class TestNoiseInjectionSanity:
    """测试面 #2（AC）：单步 SDE 扰动 + ODE 续跑的样本分布统计量与原模型
    一致（边缘分布保持；η=0 对照见 TestEtaZeroParity）。"""

    # 容差按 fixture 实测（seed 固定、数值确定）放大：η=0.7 实测 |Δmean|≈1.3e-2、
    # relΔstd≈0.17（单步扰动经 ODE 收缩后的真实分布移动）；量级失控的核
    # 实现（σ 或漂移项错误）会立即击穿
    MEAN_ABS_TOLERANCE = 0.05
    STD_REL_TOLERANCE = 0.30

    def test_perturbed_distribution_tracks_control(
        self, scenario: DiagnosticScenario,
    ) -> None:
        assert scenario.run(eta=0.7).code == 0
        report = scenario.report()
        control = report["control_terminals"]
        perturbed = report["perturbed_terminals"]
        for channel in range(4):
            assert abs(perturbed["channel_mean"][channel] - control["channel_mean"][channel]) < self.MEAN_ABS_TOLERANCE
            base = max(abs(control["channel_std"][channel]), 1e-6)
            relative = (
                abs(perturbed["channel_std"][channel] - control["channel_std"][channel])
                / base
            )
            assert relative < self.STD_REL_TOLERANCE
