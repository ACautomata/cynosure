"""断点续训 roundtrip（ticket #22，测试面 #5）。

fixture 下 CLI train 端到端的三条验收：
1. 训练 N iteration → 中断 → 恢复 → 与不中断续跑的轨迹/指标一致
   （iter 事件逐条相等（除 wall-clock elapsed_s）+ 收官 policy/判别器
   checkpoint 逐位一致）；
2. 续训状态清单完整覆盖：两模型权重与 optimizer、buffer 两区、RNG
   （torch/CUDA/numpy/python + 六条命名 generator 流）、iteration 计数、
   LR scheduler 状态槽、EMA 条件项槽；
3. 落盘周期走 config schema（schedule.checkpoint_interval，默认 10）
   且默认值生效——周期未到不产出状态、恢复入口对缺失状态显式拒绝；
   里程碑强制落盘独立于 checkpoint 周期。

「中断」的两条路径都覆盖：
- 干净截断（max_iterations 截短训练后延长续训，收尾兜底落盘）；
- 模拟崩溃（训练循环协作方法中途 KeyboardInterrupt——真实作业边界的
  杀进程在 CLI seam 内无入口，经 monkeypatch 注入；断言面仍是外部工件）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train import GranularGrpoTrainer, RunArtifacts
from tests.test_train_loop import TrainingLoopScenario

RESUME_STATE = "checkpoints/resume_state.pt"


@pytest.fixture
def scenario(cli, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


def _patch_config(scenario: TrainingLoopScenario, **sections: dict) -> None:
    """按 section 覆写训练 config（既有测试的 JSON 补丁惯例）。"""
    data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
    for section, values in sections.items():
        data[section].update(values)
    scenario.config_path.write_text(json.dumps(data), encoding="utf-8")


def _resume_run(scenario: TrainingLoopScenario):
    return scenario.cli.run(
        "train", "--config", str(scenario.config_path),
        "--run-dir", str(scenario.run_dir), "--resume",
    )


def _load_state(scenario: TrainingLoopScenario) -> dict:
    return torch.load(
        scenario.run_dir / RESUME_STATE, map_location="cpu", weights_only=True,
    )


def _trajectory(events: list[dict]) -> list[dict]:
    """iter 事件的轨迹可比面（wall-clock elapsed_s 除外）。"""
    return [
        {key: value for key, value in event.items() if key != "elapsed_s"}
        for event in events
    ]


def _assert_checkpoints_identical(
    left: Path, right: Path, names: list[str],
) -> None:
    for name in names:
        first = torch.load(
            left / "checkpoints" / name, map_location="cpu", weights_only=True,
        )
        second = torch.load(
            right / "checkpoints" / name, map_location="cpu", weights_only=True,
        )
        assert set(first) == set(second)
        for key in first:
            assert torch.equal(first[key], second[key]), f"{name}:{key}"


def _crash_during_iteration(monkeypatch, iteration: int) -> None:
    """把第 ``iteration + 1`` 次 _update_policy 调用替换为 KeyboardInterrupt
    （iteration 为 0 起数的崩溃所在迭代；该迭代不产出事件）。"""
    original = GranularGrpoTrainer._update_policy
    calls = {"count": 0}

    def crashing(self, record):
        calls["count"] += 1
        if calls["count"] == iteration + 1:
            raise KeyboardInterrupt(f"模拟作业边界崩溃（iteration {iteration}）")
        return original(self, record)

    monkeypatch.setattr(GranularGrpoTrainer, "_update_policy", crashing)


class TestRoundtripEquivalence:
    """AC 1：中断 → 恢复后的轨迹/指标与不中断续跑一致。"""

    def test_truncated_run_resumes_to_identical_trajectory(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """干净截断：max_iterations=2 训练（收尾兜底落盘状态@2）→ 延长
        config 到 4 并 --resume → 事件流与收官 checkpoint 和不中断的
        4-iteration run 逐条/逐位一致。"""
        scenario.write_inputs()
        _patch_config(scenario, schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        assert [event["iteration"] for event in scenario.events()] == [0, 1]

        _patch_config(scenario, schedule={"max_iterations": 4})
        result = _resume_run(scenario)
        assert result.code == 0, result.stderr
        resumed_events = scenario.events()
        assert [event["iteration"] for event in resumed_events] == [0, 1, 2, 3]

        baseline_dir = scenario.tmp_path / "run_baseline"
        assert scenario.cli.train(
            scenario.config_path, run_dir=baseline_dir,
        ).code == 0
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert _trajectory(resumed_events) == _trajectory(baseline_events)
        _assert_checkpoints_identical(
            scenario.run_dir, baseline_dir,
            ["policy_iter4.pt", "discriminator_iter4.pt"],
        )
        assert _load_state(scenario)["iteration"] == 4

    def test_mid_run_crash_resumes_from_last_periodic_state(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """模拟崩溃（checkpoint_interval=2、iteration 3 中途崩溃）：恢复点
        = 最近周期落盘 iteration 2；恢复回退指标流中半截事件（iteration 2
        的事件被重执行重写），最终轨迹与不中断 run 一致。"""
        scenario.write_inputs()
        _patch_config(
            scenario,
            schedule={"max_iterations": 4, "checkpoint_interval": 2},
        )
        with monkeypatch.context() as patch:
            _crash_during_iteration(patch, iteration=3)
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        # 崩溃前完整 iteration 0/1/2 已追加事件；状态停在周期点 2
        assert _load_state(scenario)["iteration"] == 2
        assert [event["iteration"] for event in scenario.events()] == [0, 1, 2]

        assert _resume_run(scenario).code == 0
        resumed_events = scenario.events()
        assert [event["iteration"] for event in resumed_events] == [0, 1, 2, 3]

        baseline_dir = scenario.tmp_path / "run_baseline"
        assert scenario.cli.train(
            scenario.config_path, run_dir=baseline_dir,
        ).code == 0
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert _trajectory(resumed_events) == _trajectory(baseline_events)
        _assert_checkpoints_identical(
            scenario.run_dir, baseline_dir,
            ["policy_iter4.pt", "discriminator_iter4.pt"],
        )


class TestResumeStateChecklist:
    """AC 2：续训状态清单完整覆盖（spec #15 续训状态全清单）。"""

    def test_state_covers_full_checklist(self, scenario: TrainingLoopScenario) -> None:
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = _load_state(scenario)
        config = ConfigLoader.load(scenario.config_path)

        assert state["format_version"] == 1
        assert state["iteration"] == 1  # 收尾兜底落盘点 = max_iterations

        # 两模型权重：键形与全新装配的网络一致（组1 policy = UNet 本体）
        unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
        ))
        assert set(state["policy_network"]) == set(unet.state_dict())
        discriminator = NetworkAssembler.discriminator(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
        ))
        assert set(state["discriminator_network"]) == set(discriminator.state_dict())

        # 两 optimizer：AdamW 动量已积累（梯度步真实生效）
        for name in ("policy_optimizer", "discriminator_optimizer"):
            optimizer_state = state[name]["state"]
            assert optimizer_state, name
            assert "exp_avg" in next(iter(optimizer_state.values()))

        # buffer 两区：base 满容量（64//2），recent = |M|×G×|Λ| + anchor = 25
        assert state["replay_buffer"]["base"].shape == (32, 4, 16, 16, 8)
        assert state["replay_buffer"]["recent"].shape == (25, 4, 16, 16, 8)

        # RNG：六条命名流 + 全局 torch/numpy/python（fixture CPU 无 CUDA）
        assert set(state["generators"]) == {
            "rollout", "real_pool", "disc_update",
            "heldout_auc", "fake_shuffle", "base_partition",
        }
        assert all(
            saved.dtype == torch.uint8 for saved in state["generators"].values()
        )
        assert state["rng"]["torch"].dtype == torch.uint8
        assert state["rng"]["cuda"] is None
        assert state["rng"]["numpy"]["keys"].shape == (624,)  # MT19937 键数组
        assert len(state["rng"]["python"]["state"]) == 625

        # LR scheduler 状态槽（常数 LR 实现 = 两 optimizer 的 lr）
        assert state["lr"] == {
            "policy": config.policy.policy_lr,
            "discriminator": config.reward.disc_lr,
        }
        # EMA 条件项槽（升级项未交付，恒 None）
        assert state["ema"] is None

    def test_ema_enabled_rejected_as_undelivered_upgrade(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """ema_anchor_enabled=true 属升级项（ADR-0001）：静默忽略会让续训
        状态清单缺 EMA 权重（条件项失真），装配期显式拒绝。"""
        scenario.write_inputs()
        _patch_config(scenario, grpo={"ema_anchor_enabled": True})
        result = scenario.train()
        assert result.code == 2
        assert "EMA" in result.stderr


class TestCheckpointCadence:
    """AC 3：落盘周期走 config schema 且默认值生效。"""

    def test_default_interval_defers_persistence_and_resume_fails_cleanly(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """默认 checkpoint_interval=10 被消费：3 个完整 iteration 后崩溃
        （周期未到、无收尾兜底）不产出续训状态；恢复入口对缺失状态显式
        拒绝（退出码 2 + 清晰消息），不裸 traceback。"""
        scenario.write_inputs()  # checkpoint_interval 缺省 = 10
        _patch_config(scenario, schedule={"max_iterations": 4})
        with monkeypatch.context() as patch:
            _crash_during_iteration(patch, iteration=3)
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        assert not (scenario.run_dir / RESUME_STATE).is_file()
        result = _resume_run(scenario)
        assert result.code == 2
        assert "续训状态" in result.stderr

    def test_milestone_forces_state_between_checkpoint_intervals(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """「每里程碑强制」独立于 checkpoint 周期：milestone_interval=2、
        checkpoint_interval=5，iteration 3 中途崩溃——状态@2 仅由里程碑
        节奏产出（2 % 5 != 0）。"""
        scenario.write_inputs()
        _patch_config(
            scenario,
            schedule={
                "max_iterations": 4,
                "milestone_interval": 2,
                "checkpoint_interval": 5,
            },
        )
        with monkeypatch.context() as patch:
            _crash_during_iteration(patch, iteration=3)
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        assert _load_state(scenario)["iteration"] == 2


class TestResumeGuards:
    """续训入口的输入契约：错误组合显式拒绝（退出码 2）。"""

    def test_resume_requires_explicit_run_dir(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        result = scenario.cli.run(
            "train", "--config", str(scenario.config_path), "--resume",
        )
        assert result.code == 2
        assert "--run-dir" in result.stderr

    def test_resume_rejects_missing_run_directory(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        result = scenario.cli.run(
            "train", "--config", str(scenario.config_path),
            "--run-dir", str(scenario.tmp_path / "nope"), "--resume",
        )
        assert result.code == 2
        assert "run 目录" in result.stderr

    def test_resume_rejects_config_drift(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """seed 漂移的续训不可复现（RNG 流与恢复状态失配）：除
        max_iterations（延长训练规模的正当地址）外逐字段一致被守卫拒绝。"""
        scenario.write_inputs()
        _patch_config(scenario, schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        _patch_config(scenario, schedule={"seed": 1})
        result = _resume_run(scenario)
        assert result.code == 2
        assert "schedule.seed" in result.stderr

    def test_resume_rejects_sequential_group(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs(group="sequential")
        result = scenario.cli.run(
            "train", "--config", str(scenario.config_path),
            "--run-dir", str(scenario.tmp_path / "seq"), "--resume",
        )
        assert result.code == 2
        assert "sequential" in result.stderr

    def test_resume_past_completion_is_clean_noop(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """恢复点已等于目标 iteration 数：零重执行、零重复事件、退出 0
        （跨实例重置后的重复提交不产生半截执行史）。"""
        scenario.write_inputs()
        _patch_config(scenario, schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        events_before = scenario.events()
        assert _resume_run(scenario).code == 0
        assert scenario.events() == events_before

    def test_resume_past_shrunk_target_rewrites_nothing(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """收缩 max_iterations 的续训 = 无操作：恢复点（4）在目标（2）之后
        时，收尾兜底不得把更后的训练态改写成更小的 iteration 标签——否则
        同一 run 目录的下次续训会从 iter-4 权重按 iteration 2 起步（轨迹
        分支）。"""
        scenario.write_inputs()
        _patch_config(scenario, schedule={"max_iterations": 4, "checkpoint_interval": 4})
        assert scenario.train().code == 0
        events_before = scenario.events()
        _patch_config(scenario, schedule={"max_iterations": 2})
        assert _resume_run(scenario).code == 0
        assert _load_state(scenario)["iteration"] == 4  # 状态未被改写
        assert scenario.events() == events_before  # 事件流未被改写
        assert not (scenario.run_dir / "checkpoints" / "policy_iter2.pt").is_file()

    def test_resume_rejects_cuda_availability_mismatch(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """落盘含 CUDA RNG、恢复环境无 CUDA（跨设备续训）：静默丢弃 =
        轨迹静默漂移，显式拒绝（CPU fixture 环境恒无 CUDA，可构造该方向）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = _load_state(scenario)
        state["rng"]["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
        torch.save(state, scenario.run_dir / RESUME_STATE)
        result = _resume_run(scenario)
        assert result.code == 2
        assert "CUDA" in result.stderr
