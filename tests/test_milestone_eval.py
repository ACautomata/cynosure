"""里程碑评测 + 早停数据流（ticket #25 验收面）。

AC 覆盖：
- fixture 下 eval 产出 FID/KID 数值与 ``milestone`` 事件（入同一指标流）；
- 里程碑间隔 / 早停参数走 config schema（数值口径见 test_config_schema）；
- 解码只发生在里程碑路径，不进逐 iteration 训练循环（结构断言：
  静态 AST 检查 + 注入计数解码器的运行时验证）；
- 早停接线：plateau 判定在 train 进程内消费指标流后中断训练。
"""

import ast
import json
import math
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.eval import EvaluationPhase
from cynosure.eval.decode import LatentDecoder
from cynosure.eval.milestone import MilestoneMetrics
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.numerics import AmpContext
from cynosure.train import BaselineManifest, GranularGrpoTrainer, RunArtifacts
from tests.conftest import CliSession
from tests.test_train_loop import TrainingLoopScenario

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cynosure"


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


class TestMilestoneEventStream:
    """AC：fixture 下 eval 产出 FID/KID 数值与 milestone 事件（同一指标流）。"""

    def test_milestone_event_emitted_into_same_stream(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.set_schedule(max_iterations=1, milestone_interval=1)
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.events()
        assert [event["event"] for event in events] == ["iter", "milestone"]
        milestone = events[1]
        assert milestone["iteration"] == 1
        assert milestone["stage"] == 1
        assert math.isfinite(milestone["fid"]) and milestone["fid"] >= 0.0
        assert math.isfinite(milestone["kid"])
        summary = milestone["criteria_summary"]
        for plane in ("xy", "yz", "zx"):
            assert math.isfinite(summary[f"fid_{plane}"])
            assert math.isfinite(summary[f"kid_{plane}"])
        assert summary["kid_ci_low"] <= summary["kid_ci_high"]
        assert milestone["ssim"] is None  # 跨模态组才带 SSIM/MAE/PSNR
        assert milestone["mae"] is None
        assert milestone["psnr"] is None
        assert milestone["early_stop"] is False

    def test_cross_modal_milestone_carries_ssim_mae(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """跨模态组另加 3D SSIM/MAE：合成 target 与同一病例 ground-truth
        target 配对比较（manifest 锁定源病例，参照取其目标序列）。"""
        scenario.write_inputs(group="cross-modal")
        scenario.set_schedule(max_iterations=1, milestone_interval=1)
        result = scenario.train()
        assert result.code == 0, result.stderr
        milestone = scenario.events()[1]
        assert milestone["ssim"] is not None and -1.0 <= milestone["ssim"] <= 1.0
        assert milestone["mae"] is not None and milestone["mae"] >= 0.0
        assert milestone["psnr"] is not None and 0.0 < milestone["psnr"] <= 100.0
        manifest = json.loads(
            (scenario.run_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        assert all(
            entry["source_case"] for entry in manifest["entries"]
        )  # 源病例锁定（配对参照的依据）

    def test_same_seed_runs_produce_identical_milestone_metrics(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """里程碑评测的 manifest 条目驱动（同 seed 同条件）可复现：
        同一 config 跑两次，里程碑 FID 逐位一致。"""
        fids = []
        for index in range(2):
            scenario = TrainingLoopScenario(cli, tmp_path / f"run{index}")
            scenario.write_inputs()
            scenario.set_schedule(max_iterations=1, milestone_interval=1)
            result = scenario.train()
            assert result.code == 0, result.stderr
            fids.append(scenario.events()[1]["fid"])
        assert fids[0] == fids[1]


class CountingDecoder:
    """测试仪器：计数解码器（decode 只发生在评测路径的运行时观测面）。"""

    def __init__(self, inner: LatentDecoder) -> None:
        self._inner = inner
        self.calls: list[tuple[int, ...]] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(latents.shape))
        return self._inner.decode(latents)


class TestDecodeOnlyInEvaluationPaths:
    """AC（结构断言）：解码只发生在里程碑路径，不进逐 iteration 循环。"""

    def test_train_package_never_calls_decode_or_imports_decoder(self) -> None:
        """静态 AST 检查：train 包源码无 ``.decode`` 调用点、不引用任何
        解码器类型（解码只属于 eval 包；trainer 只经 EvaluationPhase
        的三个评测动作间接触达）。"""
        offenders: list[str] = []
        decoder_names = {"LatentDecoder", "AutoencoderKlMaisi", "VolumeDecoder"}
        for source in sorted((SRC_ROOT / "train").rglob("*.py")):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "decode"
                ):
                    offenders.append(f"{source.name}:{node.lineno} decode 调用")
                if isinstance(node, ast.Name) and node.id in decoder_names:
                    offenders.append(f"{source.name}:{node.lineno} {node.id}")
                if (
                    isinstance(node, ast.Attribute) and node.attr in decoder_names
                ):
                    offenders.append(f"{source.name}:{node.lineno} {node.attr}")
        assert offenders == [], f"train 包出现解码路径引用: {offenders}"

    def test_eval_package_owns_the_decoder(self) -> None:
        """解码实现在 eval 包内（结构断言的正向面，ast 静态解析）。"""
        tree = ast.parse(
            (SRC_ROOT / "eval" / "decode.py").read_text(encoding="utf-8"),
        )
        classes = {
            node.name: {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for node in tree.body if isinstance(node, ast.ClassDef)
        }
        assert "LatentDecoder" in classes
        assert "decode" in classes["LatentDecoder"]

    def test_decode_invocations_confined_to_evaluation_phases(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """运行时验证（计数解码器注入）：3 iteration、里程碑间隔 2 的
        run 恰好 3 次 decode 调用——baseline（4 条目）+ 里程碑（前缀
        milestone_eval_samples=2 条）+ 重采（4 条目）；逐 iteration
        循环中零解码。"""
        scenario.write_inputs()
        scenario.set_schedule(
            max_iterations=3,
            milestone_interval=2,
            milestone_eval_samples=2,
        )
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        inner = LatentDecoder(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.vae_config_json,
                ),
                checkpoint=config.artifacts.vae_ckpt,
            ),
            torch.device("cpu"),
        )
        counter = CountingDecoder(inner)
        evaluation = EvaluationPhase.build(
            config,
            artifacts,
            scenario.standalone_sampler(config),
            stage=1,
            manifest=BaselineManifest.load(artifacts.paths.manifest),
            amp=AmpContext(torch.device("cpu"), torch.bfloat16),
            decoder=counter,
        )
        trainer = GranularGrpoTrainer(config, artifacts, evaluation=evaluation)
        assert trainer.run() == 3
        assert counter.calls == [
            (4, 4, 16, 16, 8),  # baseline：manifest 全部 4 条目一批
            (2, 4, 16, 16, 8),  # 里程碑：manifest 前缀 milestone_eval_samples=2 条
            (4, 4, 16, 16, 8),  # RL 后重采：同 manifest 条目
        ]


class StubEvaluation:
    """测试仪器：里程碑度量可控的评测相替身（早停接线断言用）。"""

    def __init__(self, fids: list[float]) -> None:
        self._fids = list(fids)
        self.baseline_called = False
        self.resample_called = False

    def sample_baseline(self) -> None:
        self.baseline_called = True

    def resample(self) -> None:
        self.resample_called = True

    def milestone_metrics(self) -> MilestoneMetrics:
        return MilestoneMetrics(
            fid=self._fids.pop(0) if self._fids else 5.0,
            kid=0.01,
            kid_ci_low=0.0,
            kid_ci_high=0.02,
            plane_fid={"XY": 5.0, "YZ": 5.0, "ZX": 5.0},
            plane_kid={"XY": 0.01, "YZ": 0.01, "ZX": 0.01},
        )


class TestProductionEvaluationContract:
    """生产装配契约（经公共 ``EvaluationPhase.build`` 口径断言，不直调
    私有装配）：fixture 工件齐全、语义翻转为生产（fixture_mode=false）
    时，缺 RadImageNet 权重 / VAE 网络配置在装配期显式拒绝——不静默
    stub（生产静默 stub 会让里程碑指标变成无意义数字）。"""

    @staticmethod
    def _evaluation_phase(scenario: TrainingLoopScenario, **config_mutations):
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        for key, value in config_mutations.items():
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                target = getattr(target, part)
            setattr(target, parts[-1], value)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        return EvaluationPhase.build(
            config,
            artifacts,
            scenario.standalone_sampler(config),
            stage=1,
            manifest=BaselineManifest.load(artifacts.paths.manifest),
            amp=AmpContext(torch.device("cpu"), torch.bfloat16),
        )

    def test_production_mode_requires_radimagenet_weights(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        with pytest.raises(ValueError, match="radimagenet_weights"):
            self._evaluation_phase(scenario, **{"fixture_mode": False})

    def test_missing_vae_network_config_rejected(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        with pytest.raises(ValueError, match="vae_config_json"):
            self._evaluation_phase(
                scenario, **{"artifacts.vae_config_json": None},
            )


class TestEarlyStopWiring:
    """早停判定在 train 进程内消费指标流：plateau 命中即中断训练。"""

    def test_plateau_stops_training_before_max_iterations(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """主判据连续 N_plateau=3 个里程碑无改善（同值 FID）→ 第 4 个
        里程碑触发早停；完成的 iteration 数 < max_iterations，最终
        checkpoint 停在早停点。"""
        scenario.write_inputs()
        scenario.set_schedule(
            max_iterations=6,
            milestone_interval=1,
            n_plateau=3,
            plateau_tolerance=0.5,
        )
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        stub = StubEvaluation(fids=[5.0] * 6)
        trainer = GranularGrpoTrainer(config, artifacts, evaluation=stub)
        assert trainer.run() == 4  # 里程碑 1 立基准，2/3/4 连续 plateau → 停
        assert stub.baseline_called and stub.resample_called
        events = artifacts.read_events()
        # 每 iteration 先落 iter 事件、里程碑再落 milestone 事件：交错流
        assert [event["event"] for event in events] == ["iter", "milestone"] * 4
        final_milestone = events[-1]
        assert final_milestone["early_stop"] is True
        assert final_milestone["early_stop_reason"] == "plateau"
        assert final_milestone["criteria_summary"]["plateau_stalled"] == 1.0
        checkpoints = scenario.run_dir / "checkpoints"
        assert (checkpoints / "policy_iter4.pt").is_file()  # 里程碑强制落盘
        assert not (checkpoints / "policy_iter5.pt").exists()
        assert not (checkpoints / "policy_iter6.pt").exists()

    def test_no_early_stop_when_criteria_keeps_improving(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.set_schedule(
            max_iterations=3,
            milestone_interval=1,
            n_plateau=3,
            plateau_tolerance=0.5,
        )
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        stub = StubEvaluation(fids=[10.0, 5.0, 2.0])  # 持续显著改善
        trainer = GranularGrpoTrainer(config, artifacts, evaluation=stub)
        assert trainer.run() == 3  # 跑满，不早停
        events = artifacts.read_events()
        assert events[-1]["early_stop"] is False
        assert events[-1]["early_stop_reason"] is None
