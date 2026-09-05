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

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.eval import EvaluationPhase, ManifestEvaluation
from cynosure.eval.decode import LatentDecoder
from cynosure.eval.features import StubSliceFeatureExtractor
from cynosure.eval.milestone import MilestoneEvaluator, MilestoneMetrics
from cynosure.eval.sampling import EntrySample
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.numerics import AmpContext
from cynosure.reward.artifacts import LatentManifest
from cynosure.train import (
    BaselineManifest,
    GranularGrpoTrainer,
    ManifestEntry,
    RunArtifacts,
)
from tests.conftest import MINIMAL_CONFIG_DICT, CliSession
from tests.test_train_loop import TrainingLoopScenario

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cynosure"

SECOND_DEVICE_AVAILABLE = (
    torch.cuda.is_available() or torch.backends.mps.is_available()
)
"""设备错配复现的前提：CPU 之外存在第二计算设备（MPS/CUDA）。"""


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
        解码器类型（解码只属于 eval 包；trainer 只经 EvaluationPhase 接口
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
        evaluation = ManifestEvaluation.build(
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


class TestBoundedVolumeSampling:
    """Baseline/重采的分块流式解码（显存有界）与落盘独立物化。

    生产 N_baseline = 200–500、解码体 256×256×128 fp32：整 manifest
    单批解码的峰值分配 OOM 级；view 序列化会携带整批 backing storage，
    每条目文件膨胀 K 倍。"""

    def _evaluation_with_counter(
        self, scenario: TrainingLoopScenario, decode_batch_size: int,
    ) -> tuple[ManifestEvaluation, CountingDecoder]:
        scenario.write_inputs()
        scenario.set_schedule(decode_batch_size=decode_batch_size)
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
        evaluation = ManifestEvaluation.build(
            config,
            artifacts,
            scenario.standalone_sampler(config),
            stage=1,
            manifest=BaselineManifest.load(artifacts.paths.manifest),
            amp=AmpContext(torch.device("cpu"), torch.bfloat16),
            decoder=counter,
        )
        return evaluation, counter

    def test_baseline_decoding_streams_in_bounded_chunks(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """decode_batch_size=2、4 条目：baseline 解码两次调用、每次批
        ≤ 块大小——峰值显存以块为界；重采同口径。"""
        evaluation, counter = self._evaluation_with_counter(
            scenario, decode_batch_size=2,
        )
        evaluation.sample_baseline()
        chunk_latent_shape = (2, 4, 16, 16, 8)  # 计数面 = decode 输入 latent
        assert counter.calls == [chunk_latent_shape, chunk_latent_shape]
        counter.calls.clear()
        evaluation.resample()
        assert counter.calls == [chunk_latent_shape, chunk_latent_shape]

    def test_saved_volume_owns_its_storage(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """落盘像素体是独立物化张量：条目体是批张量的 view 时序列化
        携带整批 backing storage（load 回读 storage 必须等于张量自身）。"""
        evaluation, _ = self._evaluation_with_counter(
            scenario, decode_batch_size=4,
        )
        evaluation.sample_baseline()
        stored = torch.load(
            scenario.run_dir / "samples" / "stage1" / "baseline" / "0000.pt",
        )
        assert stored.untyped_storage().nbytes() == (
            stored.numel() * stored.element_size()
        )


class StaticLatentSampler:
    """测试仪器：恒零 latent 的采样替身（不依赖网络）。"""

    def __init__(self, latent_shape: tuple[int, ...]) -> None:
        self._latent_shape = latent_shape

    def sample(self, entries) -> list[EntrySample]:
        return [
            EntrySample(
                entry=entry, target="t1n", source_case=None,
                terminal=torch.zeros(1, *self._latent_shape),
            )
            for entry in entries
        ]


class FixedShapeDecoder:
    """测试仪器：输出固定形状像素体的解码替身。"""

    def __init__(self, volume_shape: tuple[int, int, int]) -> None:
        self._volume_shape = volume_shape

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.rand(
            (latents.shape[0], 1, *self._volume_shape),
            generator=torch.Generator().manual_seed(0),
        )


class NativeShapeRealStore:
    """测试仪器：返回原生（未经 prepare 预处理）参照体的替身。"""

    def __init__(self, volume_shape: tuple[int, int, int]) -> None:
        self._volume_shape = volume_shape

    def case_ids(self) -> list[str]:
        return ["case-a", "case-b"]

    def volume(self, case_id: str, modality: str) -> torch.Tensor:
        return torch.rand(self._volume_shape)


class TestReferenceSpaceAlignment:
    """参照/合成影像空间对齐守卫：原生 BraTS NIfTI（240×240×155、原生
    朝向）直入参照库时，跨域比较在组1 FID 路径是**静默**的（特征提取器
    对任意切片尺寸都能出特征）——守卫把未对齐参照库变成显式失败，而非
    无意义的跨域指标（生产参照须经 prepare 预处理到模型影像空间）。"""

    @staticmethod
    def _evaluator(
        reference_shape: tuple[int, int, int],
        synthetic_shape: tuple[int, int, int],
    ) -> MilestoneEvaluator:
        config = CynosureConfig.model_validate({
            **MINIMAL_CONFIG_DICT, "schedule": {"seed": 0},
        })
        manifest = BaselineManifest(
            seed=0, group="modal-label", conditions=["t1n"],
            entries=[
                ManifestEntry(index=index, condition="t1n", noise_seed=index)
                for index in range(4)
            ],
        )
        return MilestoneEvaluator(
            config,
            1,
            StaticLatentSampler((4, 16, 16, 8)),
            FixedShapeDecoder(synthetic_shape),
            StubSliceFeatureExtractor(),
            NativeShapeRealStore(reference_shape),
            manifest,
            torch.device("cpu"),
        )

    def test_mismatched_reference_space_rejected_loudly(self) -> None:
        evaluator = self._evaluator(
            reference_shape=(20, 20, 10),   # 原生空间（未经预处理）
            synthetic_shape=(16, 16, 8),    # 模型影像空间
        )
        with pytest.raises(ValueError, match="影像空间"):
            evaluator.evaluate()

    def test_aligned_reference_space_passes(self) -> None:
        evaluator = self._evaluator(
            reference_shape=(16, 16, 8), synthetic_shape=(16, 16, 8),
        )
        metrics = evaluator.evaluate()
        assert math.isfinite(metrics.fid)


class AcceleratorStubDecoder:
    """测试仪器：产出落在加速器上的像素体（模拟生产 LatentDecoder 的
    设备行为——解码网络在 amp.device；体尺寸与 fixture 场景的
    dataset 参照体同空间 64×64×32）。"""

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.rand(
            (latents.shape[0], 1, 64, 64, 32),
            generator=torch.Generator().manual_seed(0),
        ).to(self._device)


class TestSingleDeviceMilestoneEvaluation:
    """里程碑度量两侧单设备归一：GPU 训练下合成侧解码在加速器、
    参照库装载自 CPU NIfTI、提取器骨干另有装载点——不归一即首个
    里程碑设备错配崩溃（生产 CUDA run 必炸，CPU fixture 测不出）。"""

    @pytest.mark.skipif(
        not SECOND_DEVICE_AVAILABLE,
        reason="需要 CPU 之外的第二计算设备来复现设备错配",
    )
    def test_synthetic_and_reference_reach_metrics_on_one_device(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        accelerator = torch.device(
            "cuda" if torch.cuda.is_available() else "mps",
        )
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        evaluation = ManifestEvaluation.build(
            config,
            artifacts,
            scenario.standalone_sampler(config, accelerator),
            stage=1,
            manifest=BaselineManifest.load(artifacts.paths.manifest),
            amp=AmpContext(accelerator, torch.bfloat16),
            decoder=AcceleratorStubDecoder(accelerator),
        )
        metrics = evaluation.milestone_metrics()
        assert math.isfinite(metrics.fid)


class TestBaselineSamplingIdempotence:
    """「冻结模型只采一次」的工件级幂等：baseline_sample 已填充的条目
    跳过不重采——续训恢复点 policy 已非冻结初始权重，重采会把训练后
    样本污染进基线（experiment-design「对照基线」；trainer 侧 resume
    跳过之外的第二道防线）。"""

    def test_baseline_sampling_skips_prefilled_entries(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        assert scenario.train().code == 0, scenario.train().stderr
        # sentinel 放在保持已填充的条目 1 上（断言不被覆盖）；条目 0 标回
        # 未采——触发补采路径，验证只采缺的、不动已冻结的
        run_paths = RunArtifacts.layout(scenario.run_dir)
        manifest = BaselineManifest.load(run_paths.manifest)
        frozen = manifest.entries[1]
        sentinel_path = scenario.run_dir / frozen.baseline_sample
        sentinel = torch.full_like(torch.load(sentinel_path), -7.0)
        torch.save(sentinel, sentinel_path)
        manifest.entries[0].baseline_sample = None
        manifest.write(run_paths.manifest)

        config = ConfigLoader.load(scenario.config_path)
        counter = CountingDecoder(LatentDecoder(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.vae_config_json,
                ),
                checkpoint=config.artifacts.vae_ckpt,
            ),
            torch.device("cpu"),
        ))
        evaluation = ManifestEvaluation.build(
            config,
            RunArtifacts(run_paths),
            scenario.standalone_sampler(config),
            1,
            BaselineManifest.load(run_paths.manifest),
            amp=AmpContext(torch.device("cpu"), torch.bfloat16),
            decoder=counter,
        )
        evaluation.sample_baseline()
        assert [shape[0] for shape in counter.calls] == [1]  # 只补采条目 0 这 1 条未填充
        assert torch.equal(torch.load(sentinel_path), sentinel)  # 已填条目不覆盖


class StubEvaluation(EvaluationPhase):
    """测试仪器：里程碑度量可控的评测相替身（早停接线断言用）——显式
    实现 ``EvaluationPhase`` 接口（trainer 依赖的契约面）。"""

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
    """生产装配契约（经公共 ``ManifestEvaluation.build`` 口径断言，不直调
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
        return ManifestEvaluation.build(
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


class TestReferencePoolRestriction:
    """组1 参照库也取 real pool train split：_build_reals 装配缝把
    参照病例白名单锁进 pool manifest（dataset_root 全树轮转会混入
    val/test 分区，FID 参照分布分裂泄漏）。"""

    def test_reference_store_restricted_to_real_pool_split(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )
        store = ManifestEvaluation._build_reals(config, pool)
        assert store.case_ids() == sorted({
            entry.case_id for entry in pool.entries
        })
        assert len(store.case_ids()) < 20  # dataset 全树 20 例，pool 是其 train 子集


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
