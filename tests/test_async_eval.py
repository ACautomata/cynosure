"""async 执行序的评测三路径测试档（#237 评测顺迁期，#217 三路径口径）。

档位与挂法（#226 决策 3，沿用 test_async_executor 双 marker 惯例）：

- **CPU fixture 档（slow+gpu）**：评测三路径（Baseline / 里程碑 / 重采）
  进 async 门面的端到端——milestone 事件契约（``phase_seconds`` 的
  decode/fid 面完整不收缩）+ 三路径落盘（manifest 回写 + 像素体文件）
  + 分派逐位安全单元锚（槽分派 vs 同步逐条：同 manifest 同权重下
  terminal 逐位一致）+ 评测零消耗训练 RNG 流（milestone_interval 变化
  不动 iter 事件流——逐位重放不受评测触发点影响）+ 早停接线（plateau
  触发 break + 里程碑强制 checkpoint + 收官重采）+ 组2 配对保真
  （SSIM/MAE/PSNR + 源病例锁定）+ MR 域异形 latent 评测。
- **gauss 多卡 e2e 档（slow+gpu）**：多卡拓扑下三路径绿 + 事件契约
  （采样前向经各卡副本、汇聚卡 0 的多卡形态）。

slow+gpu 挂法：默认跳过、CPU 环境跳过（conftest 双 marker 轴），验证
职责由 gauss ``pytest --run-slow`` 全量承担（仓库纪律：测试一律上集群）。
"""

import json
import math
from pathlib import Path

import pytest
import torch

from cynosure.eval.condition import EntryConditionResolver
from cynosure.eval.sampling import ManifestLatentSampler
from cynosure.reward.artifacts import LatentManifest
from cynosure.train.artifacts import BaselineManifest, RunArtifacts
from cynosure.train.async_eval import SlotDispatchLatentSampler
from cynosure.train.executor import AsyncTrainingExecutor
from cynosure.train.runtime import TrainingRuntime
from tests.conftest import RunTrajectory, CliSession
from tests.test_async_executor import ExecutorScenario
from tests.test_mr_train import CONDITIONS, MrTrainScenario


@pytest.mark.gpu
@pytest.mark.slow
class TestAsyncEvaluationFixtureTier:
    """CPU fixture 档：评测三路径进 async 门面的端到端与分派逐位锚。"""

    @staticmethod
    def _milestone_config(scenario: ExecutorScenario, **kwargs):
        """评测档 config：里程碑间隔 1（单 iteration 即触发，监控相可达）。"""
        config = scenario.prepare(**kwargs)
        config.schedule.milestone_interval = 1
        return config

    def test_milestone_event_contract_and_three_paths(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """三路径端到端：milestone 事件入同一指标流（iter 族之后）+
        事件契约完整（fid/kid 有限、criteria_summary 逐面、SSIM 组1
        缺席、early_stop False）+ ``phase_seconds`` 面不收缩（decode/fid
        两相在场且为正——#237 AC 的契约面）+ baseline/重采落盘回写。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._milestone_config(scenario, seed=0)
        executor = scenario.build(config, coroutines=3)
        executor.run()
        events = scenario.events()
        typed = [
            event["event"] for event in events
            if event["event"] != "overfit_alert"
        ]
        assert typed == ["iter", "iter", "iter", "milestone"]
        milestone = next(
            event for event in events if event["event"] == "milestone"
        )
        assert milestone["iteration"] == 1
        assert milestone["stage"] == 1
        assert math.isfinite(milestone["fid"]) and milestone["fid"] >= 0.0
        assert math.isfinite(milestone["kid"])
        summary = milestone["criteria_summary"]
        for plane in ("xy", "yz", "zx"):
            assert math.isfinite(summary[f"fid_{plane}"])
            assert math.isfinite(summary[f"kid_{plane}"])
        assert summary["kid_ci_low"] <= summary["kid_ci_high"]
        # 契约不收缩（#237 AC）：decode/fid 相位面与旧执行序同口径
        assert set(milestone["phase_seconds"]) == {"decode", "fid"}
        assert milestone["phase_seconds"]["decode"] > 0.0
        assert milestone["phase_seconds"]["fid"] > 0.0
        assert milestone["elapsed_s"] is not None
        assert milestone["elapsed_s"] > 0.0
        assert milestone["ssim"] is None  # 跨模态组才带 SSIM/MAE/PSNR
        assert milestone["mae"] is None
        assert milestone["psnr"] is None
        assert milestone["early_stop"] is False
        # 三路径落盘（baseline + 重采：像素体文件 + manifest 回写）
        run_dir = tmp_path / "run"
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        assert manifest["entries"]
        for entry in manifest["entries"]:
            assert entry["baseline_sample"] is not None
            assert entry["resample_sample"] is not None
            assert (run_dir / entry["baseline_sample"]).is_file()
            assert (run_dir / entry["resample_sample"]).is_file()

    def test_slot_dispatch_bitwise_matches_sequential(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """分派逐位安全单元锚（#217「按槽分派逐位安全」的机器证明）：
        同一 executor（同权重同 manifest 条目）下，槽分派实现与同步逐条
        实现对同一批条目产出逐位一致的 terminal——per-entry noise_seed
        独立 generator + anchor η=0 确定性使分派只影响墙钟不影响数值。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._milestone_config(scenario, seed=0)
        executor = scenario.build(config, coroutines=3)
        for card in executor.cards:
            card.start()
        try:
            vocabulary = TrainingRuntime.assemble_vocabulary(config)
            dispatched = SlotDispatchLatentSampler.assemble(
                executor.cards, executor.allocation.slot_count,
                vocabulary,
                LatentManifest.load(
                    config.reward.real_pool_manifest, kind="real_pool",
                ),
                [
                    TrainingRuntime.amp_context(config, card.device)
                    for card in executor.cards
                ],
            )
            manifest = BaselineManifest.load(executor.artifacts.paths.manifest)
            entries = manifest.entries_for_stage(1)
            device = torch.device("cpu")
            sampler = executor.cards[0].replica.sampler
            for card in executor.cards:
                card.replica.policy.eval_phase()
            sequential = ManifestLatentSampler(
                sampler,
                EntryConditionResolver(vocabulary, device),
                TrainingRuntime.amp_context(config, device),
                vocabulary,
            )
            via_slots = dispatched.sample(entries)
            via_sequential = sequential.sample(entries)
            assert [s.entry.index for s in via_slots] == [
                entry.index for entry in entries
            ]
            for slot_sample, sync_sample in zip(via_slots, via_sequential):
                assert slot_sample.entry.index == sync_sample.entry.index
                assert slot_sample.target == sync_sample.target
                assert slot_sample.source_case == sync_sample.source_case
                assert torch.equal(slot_sample.terminal, sync_sample.terminal)
        finally:
            for card in executor.cards:
                card.stop()

    def test_evaluation_consumes_no_training_rng(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """评测零消耗训练 RNG 流（#218 流纪律在评测路径的延续）：同 seed
        两 run 仅 milestone_interval 不同（评测触发点 = 每个 iter 后 vs 仅
        iter 2 后）——iter 事件流（含告警族）逐位一致（墙钟字段除外）。
        评测采样若碰训练流，首个里程碑评测之后的 iter 事件即分道。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=5)
        config.schedule.max_iterations = 2
        config.schedule.milestone_interval = 1
        scenario.build(config, run_name="run1", coroutines=2).run()
        config_sparse = scenario.prepare(seed=5)
        config_sparse.schedule.max_iterations = 2
        config_sparse.schedule.milestone_interval = 2
        scenario.build(config_sparse, run_name="run2", coroutines=2).run()

        def training_events(run: str) -> RunTrajectory:
            return RunTrajectory([
                event for event in scenario.events(run)
                if event["event"] != "milestone"
            ])

        assert training_events("run1") == training_events("run2")

        def fid_at(run: str, iteration: int) -> float:
            return next(
                event["fid"] for event in scenario.events(run)
                if event["event"] == "milestone"
                and event["iteration"] == iteration
            )

        # 同 iteration 的里程碑主判据逐位：run1 在 iter 0 后已多做过一次
        # 评测，iter 1 后权重的评测 fid 仍与 run2 逐位一致 = 评测采样
        # （噪声独立 generator + anchor 确定性）零消耗训练 RNG 流
        assert fid_at("run1", 2) == fid_at("run2", 2)

    def test_early_stop_plateau_breaks_with_milestone_checkpoint(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """早停接线：plateau 签名命中 → 判定停机（完成数 < max_iterations）
        + ``milestone`` 事件携带 verdict + **里程碑强制 checkpoint**（停时
        态已落盘——config 契约：周期不覆盖时里程碑仍须产出）+ 收官重采
        照常执行（早停也是「执行了训练」）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._milestone_config(scenario, seed=0)
        config.schedule.max_iterations = 3
        config.schedule.n_plateau = 1
        config.schedule.plateau_tolerance = 1e9  # 首里程碑后不可能改善
        completed = scenario.build(config, coroutines=2).run()
        assert completed == 2  # 第二个里程碑 plateau 触发、iter 2 不执行
        milestones = [
            event for event in scenario.events()
            if event["event"] == "milestone"
        ]
        assert len(milestones) == 2
        assert milestones[-1]["early_stop"] is True
        assert milestones[-1]["early_stop_reason"] == "plateau"
        assert milestones[-1]["criteria_summary"]["plateau_stalled"] == 1.0
        assert milestones[0]["early_stop"] is False  # 首里程碑只立基准
        iter_events = [
            event for event in scenario.events()
            if event["event"] == "iter"
        ]
        assert {event["iteration"] for event in iter_events} == {0, 1}
        # 里程碑强制 checkpoint（停时态 = iteration 2 的落盘）
        assert (tmp_path / "run" / "checkpoints" / "policy_iter2.pt").is_file()
        # 收官重采（早停后 policy 未再变，resample 路径照常填充）
        manifest = json.loads(
            (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"),
        )
        assert all(
            entry["resample_sample"] is not None
            for entry in manifest["entries"]
        )

    def test_cross_modal_milestone_carries_pair_fidelity(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """组2 里程碑：跨模态组另加 3D SSIM/MAE/PSNR（合成 target 与
        同一病例 ground-truth target 配对）——源病例锁定写回 manifest
        （baseline 采样期锁定、里程碑读回同一病例）。overfit_alert 合法
        插入流（fixture 小池记忆化），按类型选取。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._milestone_config(scenario, group="cross-modal", seed=0)
        executor = scenario.build(config, coroutines=2)
        executor.run()
        milestone = next(
            event for event in scenario.events()
            if event["event"] == "milestone"
        )
        assert milestone["ssim"] is not None
        assert -1.0 <= milestone["ssim"] <= 1.0
        assert milestone["mae"] is not None and milestone["mae"] >= 0.0
        assert milestone["psnr"] is not None
        assert 0.0 < milestone["psnr"] <= 100.0
        manifest = json.loads(
            (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"),
        )
        assert all(
            entry["source_case"] for entry in manifest["entries"]
        )

    def test_mr_rate_heterogeneous_latent_milestone(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """MR-RATE 域（异形 latent 主场景）：异条件异形词汇表下评测三
        路径走通——条目按槽分派采样、异形 latent 逐条传卡 0、按条件
        分组解码（异形不可 cat 的汇聚重组在端到端的实证），事件契约
        同步保持。"""
        scene = MrTrainScenario(cli, tmp_path)
        prepared = scene.prepare(
            reward_overrides={"pretrain_pass_threshold": 0.01},
        )
        config = scene.pretrain(prepared)
        config.policy.num_inference_steps = 3
        config.policy.train_step_indices_m = {1}
        config.schedule.seed = 0
        config.schedule.max_iterations = 1
        config.schedule.milestone_interval = 1
        config.schedule.milestone_eval_samples = len(CONDITIONS)
        artifacts = RunArtifacts.init(config, tmp_path / "run")
        executor = AsyncTrainingExecutor.build(
            config, artifacts, coroutines=2, owner_check=True,
            devices=[torch.device("cpu")],
        )
        executor.run()
        events = artifacts.read_events()
        milestone = next(
            event for event in events if event["event"] == "milestone"
        )
        assert math.isfinite(milestone["fid"]) and milestone["fid"] >= 0.0
        assert set(milestone["phase_seconds"]) == {"decode", "fid"}
        run_dir = tmp_path / "run"
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        assert all(
            entry["baseline_sample"] is not None
            and entry["resample_sample"] is not None
            for entry in manifest["entries"]
        )


@pytest.mark.gpu
@pytest.mark.slow
class TestAsyncEvaluationMultiCardTier:
    """gauss 多卡 e2e 档：多卡拓扑下评测三路径绿 + 事件契约。"""

    def test_multicard_evaluation_paths(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """单进程多卡：采样前向按槽分派到各卡副本（权重步末逐位一致的
        结构保证下逐位安全）、异形 latent 逐条传卡 0 汇聚、解码 / FID
        在卡 0 主线程单点——里程碑事件契约与三路径落盘在多卡拓扑下
        完整。"""
        card_count = (
            torch.cuda.device_count() if torch.cuda.is_available() else 0
        )
        if card_count < 2:
            pytest.skip("多卡档：需要 ≥2 CUDA 设备（gauss 4×A6000 口径）")
        scenario = ExecutorScenario(cli, tmp_path)
        # K=2（判别器窗口容量守卫 = 逐条件池 ≥ K×卡数，test_async_executor
        # 多卡档同款口径）
        config = scenario.prepare(
            seed=0, reward={"disc_batch_size_k": 2},
        )
        config.schedule.max_iterations = 1
        config.schedule.milestone_interval = 1
        devices = [
            torch.device(f"cuda:{index}") for index in range(card_count)
        ]
        executor = scenario.build(
            config,
            coroutines=min(card_count, 4),
            devices=devices,
        )
        executor.run()
        milestone = next(
            event for event in scenario.events()
            if event["event"] == "milestone"
        )
        assert math.isfinite(milestone["fid"]) and milestone["fid"] >= 0.0
        assert set(milestone["phase_seconds"]) == {"decode", "fid"}
        assert milestone["phase_seconds"]["decode"] > 0.0
        assert milestone["phase_seconds"]["fid"] > 0.0
        run_dir = tmp_path / "run"
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"),
        )
        assert all(
            entry["baseline_sample"] is not None
            and entry["resample_sample"] is not None
            for entry in manifest["entries"]
        )
        for entry in manifest["entries"]:
            assert (run_dir / entry["baseline_sample"]).is_file()
            assert (run_dir / entry["resample_sample"]).is_file()
