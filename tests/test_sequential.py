"""组3 序贯两阶段（issue #23）：单次运行 stage-1（组1 配置）→ stage-2
（base′ 冻结 + 预训练 ControlNet 复用初始化，组2 配置）。

验收面：两阶段顺序执行、stage-1 产物正确传入 stage-2、经 config 指定
既有 stage-1 产物路径跳过 stage-1、每组判别器与 Replay buffer 独立
（互不串扰的断言）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train import GranularGrpoTrainer, RunArtifacts, SequentialTrainer
from tests.test_train_loop import TrainingLoopScenario


@pytest.fixture
def scenario(cli, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


class TestStagePlan:
    """两阶段计划：stage-1 按组1 配置、stage-2 = 组2 配置 + base′ 传递。"""

    def test_plan_runs_modal_label_then_cross_modal(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """stage-1 = 组1 配置（modal-label、无 stage1_run_dir）；stage-2 =
        组2 配置且 base checkpoint 指向本 run 目录的 stage-1 最终产物。"""
        scenario.write_inputs(group="sequential")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        plan = SequentialTrainer(config, artifacts).plan()
        assert [item.stage for item in plan] == [1, 2]
        assert plan[0].config.experiment.group == "modal-label"
        assert plan[0].checkpoint_prefix == ""  # stage-1 产物 = 历史布局（可复用路径）
        assert plan[1].config.experiment.group == "cross-modal"
        assert plan[1].checkpoint_prefix == "stage2_"
        expected_base = scenario.run_dir / "checkpoints" / "policy_iter1.pt"
        assert plan[1].config.artifacts.unet_ckpt == expected_base
        assert config.artifacts.unet_ckpt != expected_base  # 原 config 未被改写

    def test_plan_skips_stage1_with_existing_product(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """指定既有 stage-1 产物路径：计划只剩 stage-2，其 base 指向该
        run 目录的最终 policy checkpoint。"""
        scenario.write_inputs(group="sequential")
        config = ConfigLoader.load(scenario.config_path)
        assert scenario.train().code == 0  # 先跑完一次序贯（产出 stage-1 产物）
        skip_config_path = scenario.tmp_path / "skip_stage1.json"
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["experiment"]["stage1_run_dir"] = str(scenario.run_dir)
        skip_config_path.write_text(json.dumps(data), encoding="utf-8")
        skip_config = ConfigLoader.load(skip_config_path)
        artifacts = RunArtifacts.init(skip_config, scenario.tmp_path / "skip_run")
        plan = SequentialTrainer(skip_config, artifacts).plan()
        assert [item.stage for item in plan] == [2]
        assert plan[0].config.artifacts.unet_ckpt == (
            scenario.run_dir / "checkpoints" / "policy_iter1.pt"
        )

    def test_stage1_product_path_without_checkpoint_rejected(
        self, scenario: TrainingLoopScenario, tmp_path: Path,
    ) -> None:
        """stage1_run_dir 无 policy checkpoint = 输入契约违反（清晰错误，
        非裸 traceback）。"""
        scenario.write_inputs(group="sequential")
        empty_run = tmp_path / "empty_run"
        (empty_run / "checkpoints").mkdir(parents=True)
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["experiment"]["stage1_run_dir"] = str(empty_run)
        (tmp_path / "bad_stage1.json").write_text(
            json.dumps(data), encoding="utf-8",
        )
        result = scenario.cli.run(
            "train", "--config", str(tmp_path / "bad_stage1.json"),
            "--run-dir", str(tmp_path / "bad_run"),
        )
        assert result.code == 2
        assert "stage-1" in result.stderr


class TestSequentialRun:
    """CLI 端到端：两阶段顺序执行、产物布局、事件流按 stage 归因。"""

    def test_single_run_executes_both_stages_in_order(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0, scenario.stderr
        events = scenario.events()
        # stage-1 与 stage-2 各一 iteration，顺序排列、stage 字段区分
        assert [(event["stage"], event["iteration"]) for event in events] == [(1, 0), (2, 0)]
        checkpoints = scenario.run_dir / "checkpoints"
        # stage-1 产物（无前缀 = 与独立组1 run 同布局，可作 stage1_run_dir 复用）
        assert (checkpoints / "policy_iter1.pt").is_file()
        assert (checkpoints / "discriminator_iter1.pt").is_file()
        # stage-2 产物（stage2_ 前缀隔离，不覆写 stage-1）
        assert (checkpoints / "stage2_policy_iter1.pt").is_file()
        assert (checkpoints / "stage2_discriminator_iter1.pt").is_file()

    def test_stage1_product_is_trained_base_loadable(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """stage-1 产物 = 训练后的 base′（相对初始 fixture UNet 已演化）且
        可按 UNet 网络配置重新装载——「stage-1 产出 base′（落盘可复用）」。"""
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0
        config = ConfigLoader.load(scenario.config_path)
        base_prime = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=scenario.run_dir / "checkpoints" / "policy_iter1.pt",
        ))
        initial = torch.load(config.artifacts.unet_ckpt, map_location="cpu")
        assert any(
            not torch.equal(base_prime.state_dict()[name], value)
            for name, value in initial.items()
        )

    def test_stage2_policy_loads_as_controlnet(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """stage-2 的 policy checkpoint = ControlNet（预训练复用初始化后
        训练的可训练对象），按 ControlNet 网络配置可重装载。"""
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0
        config = ConfigLoader.load(scenario.config_path)
        reloaded = NetworkAssembler.controlnet(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.controlnet_config_json,
            ),
            checkpoint=scenario.run_dir / "checkpoints" / "stage2_policy_iter1.pt",
        ))
        assert any(p.requires_grad for p in reloaded.parameters())

    def test_stage2_trainer_assembles_base_prime_weights(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC「stage-1 产物正确传入 stage-2」的权重级断言：按 stage-2 计划
        装配的训练循环，其 base UNet state_dict 逐位等于 stage-1 最终
        checkpoint（路径传递 + 装载全链路）。"""
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts(RunArtifacts.layout(scenario.run_dir))
        plans = SequentialTrainer(config, artifacts).plan()
        assert [item.stage for item in plans] == [1, 2]
        stage2_trainer = GranularGrpoTrainer(plans[1].config, artifacts)
        stage1_checkpoint = torch.load(
            scenario.run_dir / "checkpoints" / "policy_iter1.pt",
            map_location="cpu",
        )
        for name, value in stage2_trainer.unet.state_dict().items():
            assert torch.equal(stage1_checkpoint[name], value), name
        assert all(
            not p.requires_grad for p in stage2_trainer.unet.parameters()
        )  # base′ 在 stage-2 中冻结

    def test_skip_stage1_via_config_runs_stage2_only(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """指定既有 stage-1 产物路径：跳过 stage-1 训练（指标流只有
        stage=2 事件），stage-2 在同一 run 目录正常产出。"""
        scenario.write_inputs(group="sequential")
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        assert scenario.train().code == 0
        first_run = scenario.run_dir
        # 第二次运行：复用第一次的 stage-1 产物，跳过 stage-1
        skip_config = scenario.tmp_path / "skip_stage1.json"
        data["experiment"]["stage1_run_dir"] = str(first_run)
        skip_config.write_text(json.dumps(data), encoding="utf-8")
        second_run = scenario.tmp_path / "second_run"
        result = scenario.cli.run(
            "train", "--config", str(skip_config), "--run-dir", str(second_run),
        )
        assert result.code == 0, result.stderr
        events = RunArtifacts(RunArtifacts.layout(second_run)).read_events()
        assert [event["stage"] for event in events] == [2]
        assert (second_run / "checkpoints" / "stage2_policy_iter1.pt").is_file()


class TestStageIndependence:
    """AC「每组判别器与 Replay buffer 独立（互不串扰）」：组3 两阶段在
    同一次运行内也各持独立判别器与 buffer。"""

    def test_stage_discriminators_train_independently(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """两阶段判别器从同一预训练初始化各自起步、各自在线更新一 iteration
        后权重分叉（fake 分布不同源；共享实例/串扰会得到逐位相同权重）。"""
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0
        checkpoints = scenario.run_dir / "checkpoints"
        stage1_disc = torch.load(
            checkpoints / "discriminator_iter1.pt", map_location="cpu",
        )
        stage2_disc = torch.load(
            checkpoints / "stage2_discriminator_iter1.pt", map_location="cpu",
        )
        assert stage1_disc.keys() == stage2_disc.keys()
        assert any(
            not torch.equal(stage1_disc[name], stage2_disc[name])
            for name in stage1_disc
        )

    def test_stage2_buffer_seeded_fresh_not_from_stage1_fakes(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """stage-2 的 buffer 从零起步：其首 iteration 的近期分区只含本阶段
        新 fake（|M|×G×|Λ| + anchor = 25）——若 stage-1 的 fake 串扰进来，
        该观测会翻倍；base 分区由 stage-2 自己的初始 policy 生成（首 iter
        回放占比 50% 即可用）。"""
        scenario.write_inputs(group="sequential")
        assert scenario.train().code == 0
        stage1_event, stage2_event = scenario.events()
        assert stage1_event["buffer_recent_occupied"] == 25
        assert stage2_event["buffer_recent_occupied"] == 25  # 全新 buffer，非 50
        assert stage2_event["buffer_base_occupied"] == 32  # stage-2 自行生成的 base 分区
        assert stage2_event["buffer_replay_fraction"] == pytest.approx(0.5)
