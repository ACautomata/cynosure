"""断点续训 roundtrip（ticket #22，测试面 #5）。

fixture 下 CLI train 端到端的三条验收：
1. 训练 N iteration → 中断 → 恢复 → 与不中断续跑的轨迹/指标一致
   （iter 事件逐条相等（除 wall-clock elapsed_s，RunTrajectory）+ 收官
   policy/判别器 checkpoint 逐位一致）；
2. 续训状态清单完整覆盖：两模型权重与 optimizer、buffer 两区、RNG
   （torch/CUDA/numpy/python + 六条命名 generator 流）、iteration 计数、
   LR scheduler 状态槽、EMA 条件项槽；
3. 落盘周期走 config schema（schedule.checkpoint_interval，默认 10）
   且默认值生效——周期未到不产出状态、恢复入口对缺失状态显式拒绝；
   里程碑强制落盘独立于 checkpoint 周期。

「中断」的两条路径都覆盖：
- 干净截断（max_iterations 截短训练后延长续训，收尾兜底落盘）；
- 模拟崩溃（MidRunCrash：训练循环协作方法中途 KeyboardInterrupt——
  真实作业边界的杀进程在 CLI seam 内无入口，经 monkeypatch 注入；
  断言面仍是外部工件）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, MODALITIES
from cynosure.eval import ManifestEvaluation
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train import IterationLoop, PretrainEvent, RunArtifacts
from cynosure.train.policy import GroupPolicy
from tests.conftest import RunTrajectory
from tests.test_train_loop import TrainingLoopScenario

RESUME_STATE = "checkpoints/resume_state.pt"


class MidRunCrash:
    """模拟作业边界崩溃（上下文管理器界定注入范围）：第 ``iteration + 1``
    次 ``update_policy`` 调用替换为 KeyboardInterrupt——该迭代不产出事件，
    训练停在最近周期/里程碑落盘点。"""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, iteration: int) -> None:
        self._monkeypatch = monkeypatch
        self._iteration = iteration
        self._patch = None

    def __enter__(self) -> "MidRunCrash":
        original = IterationLoop.update_policy
        crash_on_call = self._iteration + 1
        calls = {"count": 0}

        def crashing(loop, record):
            calls["count"] += 1
            if calls["count"] == crash_on_call:
                raise KeyboardInterrupt(
                    f"模拟作业边界崩溃（iteration {self._iteration}）"
                )
            return original(loop, record)

        self._patch = self._monkeypatch.context()
        self._patch.__enter__().setattr(
            IterationLoop, "update_policy", crashing,
        )
        return self

    def __exit__(self, *exc_info) -> None:
        self._patch.__exit__(*exc_info)


def prepend_pretrain_events(run_dir: Path, steps: int) -> list[dict]:
    """把 ``steps`` 条预训练事件插到指标流头部（warm-start 先于 RL 执行史
    落盘的同流混存布局），返回插入事件的读回字典——逐字保留对账的基准。

    事件经生产写入口与读取面（``RunArtifacts.append_event`` / ``read_events``）
    落盘，头部让位给既有行：注入内容与生产写出的逐字一致，不另抄一份
    序列化与布局知识。"""
    artifacts = RunArtifacts(RunArtifacts.layout(run_dir))
    existing = artifacts.paths.metrics.read_text(encoding="utf-8")
    artifacts.paths.metrics.write_text("", encoding="utf-8")
    for step in range(steps):
        artifacts.append_event(PretrainEvent(
            step=step,
            loss_discriminator=1.0,
            heldout_auc=0.5 + step * 0.01,
            buffer_base_occupied=32,
            buffer_recent_occupied=4,
            lr=5e-5,
            elapsed_s=0.1,
        ))
    injected = artifacts.read_events()
    with open(artifacts.paths.metrics, "a", encoding="utf-8") as fh:
        fh.write(existing)
    return injected


@pytest.fixture
def scenario(cli, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


class TestRoundtripEquivalence:
    """AC 1：中断 → 恢复后的轨迹/指标与不中断续跑一致。"""

    def test_truncated_run_resumes_to_identical_trajectory(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """干净截断：max_iterations=2 训练（收尾兜底落盘状态@2）→ 延长
        config 到 4 并 --resume → 事件流与收官 checkpoint 和不中断的
        4-iteration run 逐条/逐位一致。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        assert [event["iteration"] for event in scenario.events()] == [0, 1]

        scenario.patch_config(schedule={"max_iterations": 4})
        result = scenario.resume()
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
        assert RunTrajectory(resumed_events) == RunTrajectory(baseline_events)
        scenario.checkpoints_identical(
            baseline_dir, ["policy_iter4.pt", "discriminator_iter4.pt"],
        )
        assert scenario.resume_state()["iteration"] == 4

    def test_mid_run_crash_resumes_from_last_periodic_state(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """模拟崩溃（checkpoint_interval=2、iteration 3 中途崩溃）：恢复点
        = 最近周期落盘 iteration 2；恢复回退指标流中半截事件（iteration 2
        的事件被重执行重写），最终轨迹与不中断 run 一致。"""
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 4, "checkpoint_interval": 2},
        )
        with MidRunCrash(monkeypatch, iteration=3):
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        # 崩溃前完整 iteration 0/1/2 已追加事件；状态停在周期点 2
        assert scenario.resume_state()["iteration"] == 2
        assert [event["iteration"] for event in scenario.events()] == [0, 1, 2]

        assert scenario.resume().code == 0
        resumed_events = scenario.events()
        assert [event["iteration"] for event in resumed_events] == [0, 1, 2, 3]

        baseline_dir = scenario.tmp_path / "run_baseline"
        assert scenario.cli.train(
            scenario.config_path, run_dir=baseline_dir,
        ).code == 0
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert RunTrajectory(resumed_events) == RunTrajectory(baseline_events)
        scenario.checkpoints_identical(
            baseline_dir, ["policy_iter4.pt", "discriminator_iter4.pt"],
        )


class TestResumeStateChecklist:
    """AC 2：续训状态清单完整覆盖（spec #15 续训状态全清单）。"""

    def test_state_covers_full_checklist(self, scenario: TrainingLoopScenario) -> None:
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        config = ConfigLoader.load(scenario.config_path)

        assert state["format_version"] == 3
        assert state["iteration"] == 1  # 收尾兜底落盘点 = max_iterations
        assert state["world_size"] == 1  # 单进程拓扑（多 rank 见 test_distributed）

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

        # buffer 两区（v3：条目带目标模态标签，latents + modalities 成对）：
        # base 满容量（64//2）且每条件配额分布，recent = |M|×G×|Λ| + anchor = 25
        base = state["replay_buffer"]["base"]
        assert base["latents"].shape == (32, 4, 16, 16, 8)
        assert len(base["modalities"]) == 32
        assert set(base["modalities"]) == set(MODALITIES)  # 配额量产全条件覆盖
        recent = state["replay_buffer"]["recent"]
        assert recent["latents"].shape == (25, 4, 16, 16, 8)
        assert len(recent["modalities"]) == 25
        assert all(m in MODALITIES for m in recent["modalities"])

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
        scenario.patch_config(grpo={"ema_anchor_enabled": True})
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
        scenario.patch_config(schedule={"max_iterations": 4})
        with MidRunCrash(monkeypatch, iteration=3):
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        assert not (scenario.run_dir / RESUME_STATE).is_file()
        result = scenario.resume()
        assert result.code == 2
        assert "续训状态" in result.stderr

    def test_milestone_forces_state_between_checkpoint_intervals(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """「每里程碑强制」独立于 checkpoint 周期：milestone_interval=2、
        checkpoint_interval=5，iteration 3 中途崩溃——状态@2 仅由里程碑
        节奏产出（2 % 5 != 0）。"""
        scenario.write_inputs()
        scenario.patch_config(
            schedule={
                "max_iterations": 4,
                "milestone_interval": 2,
                "checkpoint_interval": 5,
            },
        )
        with MidRunCrash(monkeypatch, iteration=3):
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        assert scenario.resume_state()["iteration"] == 2


class TestNoopResumeIntegrity:
    """恢复点已达标的续训 = 完整无操作（第四轮 review 反馈）。

    「半截执行史由重执行重写」的边界 = checkpoint 覆盖面：恢复点 N 的
    checkpoint 已覆盖 iter 0..N-1 与完成数 ≤ N 的里程碑评测——它们是
    已完成执行史，rewind 不得删除（早停 verdict 与 FID 历史都在事件
    里）；milestone 事件以完成数记账，rewind 保留边界对 milestone 用
    ≤、对 iter 用 <。同理，零训练迭代的续训不重执行收官重采、完成数
    报告恢复点本身。"""

    def test_preserves_checkpoint_milestone_events(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """no-op 续训不删 checkpoint 已覆盖的 milestone 事件：恢复点 2
        与 milestone@2 同批落盘，rewind(2) 的 iter 边界（<2）不得波及
        完成数口径的 milestone（≤2）。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2, "milestone_interval": 2})
        assert scenario.train().code == 0
        assert [
            event["iteration"] for event in scenario.events()
            if event.get("event") == "milestone"
        ] == [2]
        events_before = scenario.events()
        assert scenario.resume().code == 0
        assert scenario.events() == events_before

    def test_skips_resample_and_reports_restored_count(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        """零训练迭代的续训不重执行收官重采（policy 未变、重采产物不因
        RNG 流位置漂移被静默改写），完成数报告恢复点而非 0。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        calls = {"resample": 0}
        original = ManifestEvaluation.resample

        def counting(evaluation):
            calls["resample"] += 1
            return original(evaluation)

        monkeypatch.setattr(ManifestEvaluation, "resample", counting)
        result = scenario.resume()
        assert result.code == 0
        assert calls["resample"] == 0
        assert "2 iteration" in result.stdout

    def test_noop_resume_preserves_existing_diagnostic(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """零训练迭代的续训（--dump-trajectory）不得用空 log-prob 对清单
        覆盖既有 training.json——那会把「文档化的 no-op resume」变成对
        一致性证据的破坏。"""
        scenario.write_inputs()
        assert scenario.train(dump=True).code == 0
        diagnostic_path = scenario.run_dir / "training.json"
        original = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        assert original["logprob_pairs"]  # 首轮确有真实采集

        assert scenario.resume(dump=True).code == 0
        after = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        assert after == original


class TestPretrainEventRewindIsolation:
    """预训练事件与 RL 事件同流混存下的续训回退（ticket #59 专属用例）。

    warm-start 的 ``pretrain`` 事件与 RL 的 iter/milestone 事件住在同一份
    metrics.jsonl 契约流里，而续训回退只重写恢复点之后的 RL 半截执行史：
    **误删**方向——预训练事件没有对应的 checkpoint 可重放，被回退波及即
    永久丢失（收敛曲线断点、RM readiness gate 的阈值校准数据不可复现）；
    **漏删**方向——半截 iter 事件必须删干净，否则重执行后同一 iteration
    留下两条事件，污染早停判定与离线曲线。"""

    def test_resume_rewind_preserves_pretrain_events(
        self, scenario: TrainingLoopScenario, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """崩溃（checkpoint 周期 5 / 里程碑间隔 2 → 恢复点 2）→ 指标流混入
        预训练事件 → 续训：预训练事件逐字保留，iter 事件连续无重复。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={
            "max_iterations": 4, "milestone_interval": 2, "checkpoint_interval": 5,
        })
        with MidRunCrash(monkeypatch, iteration=3):
            with pytest.raises(KeyboardInterrupt):
                scenario.train()
        pretrain_events = prepend_pretrain_events(scenario.run_dir, steps=3)
        crashed_iter = [
            event for event in scenario.events() if event["event"] == "iter"
        ]
        # 回退确有其事（非空转用例）：恢复点 = 里程碑强制的 checkpoint@2，
        # 而流里已有 iteration 2 的半截事件——续训必删它并重执行重写
        assert scenario.resume_state()["iteration"] == 2
        assert [event["iteration"] for event in crashed_iter] == [0, 1, 2]

        assert scenario.resume().code == 0
        events = scenario.events()
        # 误删方向：预训练事件全量、逐字保留（含头部位置与字段序）
        assert events[:len(pretrain_events)] == pretrain_events
        assert [
            event["step"] for event in events if event["event"] == "pretrain"
        ] == [0, 1, 2]
        # 漏删方向：半截 iter@2 被重执行重写——号连续、无重复、无旧值残留
        resumed_iter = [
            event for event in events if event["event"] == "iter"
        ]
        assert [event["iteration"] for event in resumed_iter] == [0, 1, 2, 3]
        assert RunTrajectory(resumed_iter[:3]) == RunTrajectory(crashed_iter)
        # 恢复点已覆盖的里程碑评测不被重放、也不被删除
        assert [
            event["iteration"] for event in events if event["event"] == "milestone"
        ] == [2, 4]


class TestLegacyDirCompatibility:
    """代际标记引入后的 world-1 兼容：单分片自身原子替换已保证一致性，
    历史无标记的 run 目录照常恢复（缺标记的拒绝只在多 rank——分片与
    标记必须同代际对账）。"""

    def test_world1_resume_tolerates_dir_without_marker(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        marker = scenario.run_dir / "checkpoints" / "resume_generation.json"
        assert marker.is_file()  # 当前节奏照常产出标记
        marker.unlink()  # 抹掉标记 = 历史 run 目录形态
        scenario.patch_config(schedule={"max_iterations": 4})
        assert scenario.resume().code == 0
        assert scenario.resume_state()["iteration"] == 4


class TestCheckpointCost:
    """checkpoint 的 full-state 导出成本（一次导出、两处共享）：产物
    checkpoint 写盘与续训分片快照消费同一份导出——第二次导出会让每
    rank 在 checkpoint 期同时驻留两份完整 CPU 权重副本（FSDP full state
    是 offload 到 host 的全量），生产规模下是 host 内存尖峰/OOM 源。"""

    def test_checkpoint_exports_policy_state_once(
        self, scenario: TrainingLoopScenario, monkeypatch,
    ) -> None:
        scenario.write_inputs()
        calls = {"count": 0}
        original = GroupPolicy.full_state

        def counting(policy):
            calls["count"] += 1
            return original(policy)

        monkeypatch.setattr(GroupPolicy, "full_state", counting)
        assert scenario.train().code == 0
        assert calls["count"] == 1


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
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        scenario.patch_config(schedule={"seed": 1})
        result = scenario.resume()
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
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        events_before = scenario.events()
        assert scenario.resume().code == 0
        assert scenario.events() == events_before

    def test_resume_past_shrunk_target_rewrites_nothing(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """收缩 max_iterations 的续训 = 无操作：恢复点（4）在目标（2）之后
        时，收尾兜底不得把更后的训练态改写成更小的 iteration 标签——否则
        同一 run 目录的下次续训会从 iter-4 权重按 iteration 2 起步（轨迹
        分支）。"""
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 4, "checkpoint_interval": 4},
        )
        assert scenario.train().code == 0
        events_before = scenario.events()
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.resume().code == 0
        assert scenario.resume_state()["iteration"] == 4  # 状态未被改写
        assert scenario.events() == events_before  # 事件流未被改写
        assert not (scenario.run_dir / "checkpoints" / "policy_iter2.pt").is_file()

    def test_resume_rejects_cuda_availability_mismatch(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """落盘含 CUDA RNG、恢复环境无 CUDA（跨设备续训）：静默丢弃 =
        轨迹静默漂移，显式拒绝（CPU fixture 环境恒无 CUDA，可构造该方向）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        state["rng"]["cuda"] = [torch.zeros(1, dtype=torch.uint8)]
        torch.save(state, scenario.run_dir / RESUME_STATE)
        result = scenario.resume()
        assert result.code == 2
        assert "CUDA" in result.stderr

    def test_resume_rejects_legacy_v2_shard(self, scenario: TrainingLoopScenario) -> None:
        """ADR-0008-01：旧格式（v2，replay buffer 为裸 latent 分区、条目
        不带来源标签）分片被版本对账显式拒绝——跨口径续训不可恢复
        （恢复后回放采样无法按条件过滤），报错须指向格式口径变更。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        # 回写成 v2 形态：分区 = 裸 tensor、format_version = 2
        legacy = dict(state)
        legacy["format_version"] = 2
        legacy["replay_buffer"] = {
            "base": state["replay_buffer"]["base"]["latents"],
            "recent": state["replay_buffer"]["recent"]["latents"],
        }
        torch.save(legacy, scenario.run_dir / RESUME_STATE)
        result = scenario.resume()
        assert result.code == 2
        assert "格式版本" in result.stderr
        assert "ADR-0008" in result.stderr
