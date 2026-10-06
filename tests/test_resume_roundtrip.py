"""断点续训 roundtrip（ticket #22 测试面 #5 的接替面；#226 切换期第一步
起经 async 执行序生产入口驱动，v12 单文件分片）。

fixture 下 CLI train 端到端的验收：
1. 训练 N iteration → 中断 → 恢复 → 与不中断续跑的轨迹/指标一致
   （iter 事件逐条相等（除 wall-clock elapsed_s，RunTrajectory）+ 收官
   policy/判别器 checkpoint 逐位一致）；
2. 续训状态清单完整覆盖（v12 单文件 payload 契约，#218/#222）：两模型
   权重与 optimizer、per-(槽×流) 嵌套 generator 流、iteration 计数、
   lr 槽位、overfit per-condition EMA、seeds 记录性字段；
3. 落盘周期走 config schema（schedule.checkpoint_interval，默认 10）
   且默认值生效——周期未到不产出状态、恢复入口对缺失状态显式拒绝；
   里程碑强制落盘独立于 checkpoint 周期；
4. 版本对账：v12 读写闭环、非 v12 代际（旧 v11 / 更早）分片显式拒绝。

「中断」的两条路径都覆盖：
- 干净截断（max_iterations 截短训练后延长续训，收尾兜底落盘）；
- 模拟崩溃（MidRunCrash：执行序主循环的 iteration 协程中途
  KeyboardInterrupt——真实作业边界的杀进程在 CLI seam 内无入口，经
  monkeypatch 注入；断言面仍是外部工件）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.eval import ManifestEvaluation
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train import PretrainEvent, RunArtifacts
from cynosure.train.async_resume import ASYNC_RESUME_FORMAT_VERSION
from cynosure.train.executor import AsyncTrainingExecutor
from tests.conftest import (
    RunTrajectory,
    execution_slot_count as slot_count,
    slot_tiled,
)
from tests.test_train_loop import TrainingLoopScenario

# 整文件大轮次：每个场景 2-3 次完整训练（截断 train + resume + baseline
# 重放）——标记 gpu：CPU 环境自动跳过（conftest 执行环境分派），验证
# 职责由集群 GPU 口径全量承担（仓库纪律：测试一律上集群）。
pytestmark = [pytest.mark.gpu, pytest.mark.slow]  # slow：默认跳过（--run-slow 显式全量）

RESUME_STATE = "checkpoints/resume_state.pt"

# 跨执行器实例世界对的容差口径（两次独立 AsyncTrainingExecutor.run）：
# GPU 库层 1-2 ulp 重算噪声（确定性模式不可根除）经 AdamW 归一化步长
# 链式放大——gauss 双卡 4-iteration 实测：事件指标最大分叉 1.6e-3、
# 收官 checkpoint 最大分叉 1.5e-5；且共享集群的负载窗口（gauss 卡轴
# 多租户）带来不可控的波动增量（全量档案：带 5e-3/1e-4 容差仍出现
# 低频越线，独立复跑同测试则绿）。atol 取实测噪声的 ~10 倍量级；恢
# 复语义的结构面（iteration 序、事件数、恢复点对账、rewind 边界）
# 全部严格相等是硬锚，数值面只承载大方向恢复缺陷（RNG 流错位/EMA
# 错位 → O(0.1) 级，检测余量 10 倍+）。依据 ADR-0018 决策 9 accepted
# drift + RunTrajectory docstring 的两级语义（同进程重放逐位、跨实例
# 容差）。checkpoint 面不用 checkpoint_parity 的 1e-7（那是同进程
# 重放口径；跨实例分叉实测已到 1e-5）。
_CROSS_INSTANCE_ATOL = 1e-2
_CROSS_INSTANCE_CKPT_ATOL = 1e-3


class MidRunCrash:
    """模拟作业边界崩溃（上下文管理器界定注入范围）：第 ``iteration + 1``
    次 iteration 执行替换为 KeyboardInterrupt——该迭代不产出事件，训练
    停在最近周期/里程碑落盘点。"""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, iteration: int) -> None:
        self._monkeypatch = monkeypatch
        self._iteration = iteration
        self._patch = None

    def __enter__(self) -> "MidRunCrash":
        original = AsyncTrainingExecutor._run_iteration
        crash_on_call = self._iteration + 1
        calls = {"count": 0}

        async def crashing(executor, iteration):
            calls["count"] += 1
            if calls["count"] == crash_on_call:
                raise KeyboardInterrupt(
                    f"模拟作业边界崩溃（iteration {self._iteration}）"
                )
            return await original(executor, iteration)

        self._patch = self._monkeypatch.context()
        self._patch.__enter__().setattr(
            AsyncTrainingExecutor, "_run_iteration", crashing,
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
            modality="t1n",
            loss_discriminator=1.0,
            heldout_auc=0.5 + step * 0.01,
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
        4-iteration run 逐条一致（恢复段数值容差：resume 与 baseline
        是两次独立训练——GPU 库层 1-2 ulp 重算噪声经 AdamW 放大，结构
        连续严格 + 浮点容差，RunTrajectory.atol / checkpoints atol）。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2})
        assert scenario.train().code == 0
        assert [
            event["iteration"] for event in scenario.events()
            if event["event"] == "iter"
        ] == slot_tiled([0, 1])

        scenario.patch_config(schedule={"max_iterations": 4})
        result = scenario.resume()
        assert result.code == 0, result.stderr
        resumed_events = scenario.events()
        assert [
            event["iteration"] for event in resumed_events
            if event["event"] == "iter"
        ] == slot_tiled([0, 1, 2, 3])

        baseline_dir = scenario.tmp_path / "run_baseline"
        assert scenario.cli.train(
            scenario.config_path, run_dir=baseline_dir,
        ).code == 0
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert RunTrajectory(resumed_events, atol=_CROSS_INSTANCE_ATOL) == (
            RunTrajectory(baseline_events, atol=_CROSS_INSTANCE_ATOL)
        )
        scenario.checkpoints_identical(
            baseline_dir, ["policy_iter4.pt", "discriminator_iter4.pt"],
            atol=_CROSS_INSTANCE_CKPT_ATOL,
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
        assert [
            event["iteration"] for event in scenario.events()
            if event["event"] == "iter"
        ] == slot_tiled([0, 1, 2])

        assert scenario.resume().code == 0
        resumed_events = scenario.events()
        assert [
            event["iteration"] for event in resumed_events
            if event["event"] == "iter"
        ] == slot_tiled([0, 1, 2, 3])

        baseline_dir = scenario.tmp_path / "run_baseline"
        assert scenario.cli.train(
            scenario.config_path, run_dir=baseline_dir,
        ).code == 0
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert RunTrajectory(resumed_events, atol=_CROSS_INSTANCE_ATOL) == (
            RunTrajectory(baseline_events, atol=_CROSS_INSTANCE_ATOL)
        )
        scenario.checkpoints_identical(
            baseline_dir, ["policy_iter4.pt", "discriminator_iter4.pt"],
            atol=_CROSS_INSTANCE_CKPT_ATOL,
        )


class TestResumeStateChecklist:
    """AC 2：v12 续训状态清单完整覆盖（#218/#222 终稿键清单）。"""

    def test_state_covers_full_checklist(self, scenario: TrainingLoopScenario) -> None:
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        config = ConfigLoader.load(scenario.config_path)

        assert state["version"] == ASYNC_RESUME_FORMAT_VERSION
        assert state["iteration"] == 1  # 收尾兜底落盘点 = max_iterations
        assert state["slots"] == slot_count()  # 拓扑对账 = 协程数（卡数不进对账）

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

        # 删除面（v12 终稿）：全局 RNG、EMA 预留槽、world_size、
        # replay buffer（v10）、门控状态（v11）、分配表状态（纯函数重导出）
        for retired_key in (
            "rng", "ema", "world_size", "replay_buffer", "gating",
            "allocation",
        ):
            assert retired_key not in state

        # 分叉监控状态（per-condition 嵌套）：单 iteration 逐槽观测
        # （每条件首条观测置值、count=1）
        assert set(state["overfit"]) == {"ema"}
        assert state["overfit"]["ema"]
        for divergence in state["overfit"]["ema"].values():
            assert divergence["count"] == 1
            assert isinstance(divergence["value"], float)

        # RNG：per-(槽×流) 嵌套清单，流状态 CPU generator 的 uint8 张量
        generators = state["generators"]
        assert set(generators) == {f"slot{index}" for index in range(slot_count())}
        for slot_streams in generators.values():
            assert set(slot_streams) == {
                "rollout", "real_pool", "heldout_auc", "recon",
            }
            assert all(
                saved.dtype == torch.uint8 for saved in slot_streams.values()
            )

        # seeds = 派生值记录性字段（恢复走状态回填、不重派生对账）
        assert set(state["seeds"]) == {"base", "per_slot"}
        assert state["seeds"]["base"] == config.schedule.seed
        assert len(state["seeds"]["per_slot"]) == slot_count()

        # LR scheduler 状态槽（常数 LR 实现 = 两 optimizer 的 lr）
        assert state["lr"] == {
            "policy": config.policy.policy_lr,
            "discriminator": config.reward.disc_lr,
        }

    def test_ema_enabled_rejected_as_undelivered_upgrade(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """ema_anchor_enabled=true 属升级项（ADR-0001）：静默忽略会让续训
        状态清单缺 EMA 权重（条件项失真），schema 级显式拒绝。"""
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


class TestPretrainEventRewindIsolation:
    """预训练事件与 RL 事件同流混存下的续训回退（ticket #59 专属用例）。

    warm-start 的 ``pretrain`` 事件与 RL 的 iter/milestone 事件住在同一份
    metrics.jsonl 契约流里，而续训回退只重写恢复点之后的 RL 半截执行史：
    **误删**方向——预训练事件没有对应的 checkpoint 可重放，被回退波及即
    永久丢失（收敛曲线断点、预训练过线阈值的校准数据不可复现）；
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
        crashed_iter = scenario.slot0_iter_events()
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
        resumed_iter = scenario.slot0_iter_events()
        assert [event["iteration"] for event in resumed_iter] == [0, 1, 2, 3]
        assert RunTrajectory(resumed_iter[:3], atol=_CROSS_INSTANCE_ATOL) == (
            RunTrajectory(crashed_iter, atol=_CROSS_INSTANCE_ATOL)
        )
        # 恢复点已覆盖的里程碑评测不被重放、也不被删除
        assert [
            event["iteration"] for event in events if event["event"] == "milestone"
        ] == [2, 4]


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

    def test_resume_rejects_legacy_v2_shard(self, scenario: TrainingLoopScenario) -> None:
        """旧格式代际（version=2 的历史 payload）被 v12 版本对账显式拒绝
        ——跨口径续训不可恢复，报错须指向格式口径变更（对账在版本数字，
        payload 键形无所谓）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        legacy = dict(state)
        legacy["version"] = 2
        torch.save(legacy, scenario.run_dir / RESUME_STATE)
        result = scenario.resume()
        assert result.code == 2
        assert "格式版本" in result.stderr

    def test_resume_rejects_v11_shard(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """旧执行序 v11 分片（v11 号位 + 扁平 generators + world_size 键
        的历史形态）被 v12 版本对账显式拒绝——「跨执行器拒绝由版本号
        承载」（#222）在 v12 单常量收口后的行为锁：v12 读写闭环、v11
        拒载，报错须指向版本口径。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        legacy = dict(state)
        legacy["version"] = 11  # 旧执行序 v11 号位
        torch.save(legacy, scenario.run_dir / RESUME_STATE)
        result = scenario.resume()
        assert result.code == 2
        assert "格式版本" in result.stderr
        assert "v12" in result.stderr

    def test_resume_rejects_flat_generators_shard(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """旧执行序的扁平流名清单（generators 键 = 流名、无 per-槽嵌套）
        被 v12 二次拒绝面拒绝（#218 嵌套结构的双保险，版本号之外的
        形态校验）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        state = scenario.resume_state()
        assert state["version"] == ASYNC_RESUME_FORMAT_VERSION
        state["generators"] = {
            "rollout": torch.zeros(1, dtype=torch.uint8),
        }
        torch.save(state, scenario.run_dir / RESUME_STATE)
        result = scenario.resume()
        assert result.code == 2
        assert "generators" in result.stderr
