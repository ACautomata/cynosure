"""async 执行模型骨架期的测试档（#231）：三层锚 + 门面全宽薄切片。

档位与挂法（#226 决策 3 / #231）：

- **单元锚（无标记，纯 CPU）**：派生公式单元锚（线性步长、槽 0 恒等、
  槽×流不撞位）+ 分配表纯函数锚（轮内置换/覆盖/边际均匀/组2 对数断言）
  + 超时口径解析锚。
- **CPU fixture 档（slow+gpu）**：门面全宽薄切片端到端——分配表 →
  per-槽协程 rollout → 逐 k barrier 收集-同步 → 事件发射，测试进程内
  可断言；多协程重放锚（#218 锚②：同 seed 两次 run 权重 + 判别器
  u/v buffer + RunTrajectory 事件流逐位）；异常 fail-fast 与 barrier
  软/硬超时可触发可断言；两域贯穿（BraTS + MR-RATE + 组2）。
- **gauss 多卡 e2e 档（slow+gpu）**：单进程多卡换发射方式重建的验收
  路径（#217 门面 + M1 单点归约）——跨卡权重/u-v 逐位一致 + 层 3
  同 seed 多卡重放锚薄切片形态（全强度随判别器链期，#226）。

slow+gpu 挂法：默认跳过、CPU 环境跳过（conftest 双 marker 轴），验证
职责由 gauss ``pytest --run-slow`` 全量承担（仓库纪律：测试一律上集群）。
"""

import math
import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.reward.auc import HeldOutAuc
from cynosure.train.allocation import AllocationTable
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.executor import (
    AsyncTrainingExecutor,
    BarrierTimeoutPolicy,
    TrainingAborted,
)
from cynosure.train.rng import (
    SLOT_SEED_STRIDE,
    SlotRngRegistry,
    TrainingRngStreams,
)
from tests.conftest import CliSession, FixtureArtifactLibrary, RunTrajectory
from tests.test_mr_train import MrTrainScenario


class ExecutorScenario:
    """CPU fixture 的门面端到端场景：库工件 → config → 门面直驱（不经
    CLI train——新执行序无生产入口，#226 决策 1；prepare/pretrain 的
    工件链复用 ``FixtureArtifactLibrary``）。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self._cli = cli
        self._tmp_path = tmp_path
        self._tmp_path.mkdir(parents=True, exist_ok=True)

    def prepare(
        self,
        *,
        group: str = "modal-label",
        seed: int = 0,
        num_steps: int = 3,
        train_steps: set[int] = frozenset({1}),
        reward: dict | None = None,
    ) -> CynosureConfig:
        """库工件 + 训练 config（单 iteration 薄切片口径）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(
            self._cli, group,
            num_steps=num_steps, train_steps=frozenset(train_steps),
            seed=seed, reward=reward,
        )
        config = Fixture().config(fixture_dir, group=group)
        config.policy.num_inference_steps = num_steps
        config.policy.train_step_indices_m = set(train_steps)
        config.schedule.seed = seed
        config.schedule.max_iterations = 1
        if reward:
            config.reward = config.reward.model_copy(update=reward)
        return config

    def build(
        self, config: CynosureConfig, run_name: str = "run", **kwargs,
    ) -> AsyncTrainingExecutor:
        """门面装配（run 目录初始化 + build；不启动线程）。fixture 档
        钉单 CPU 设备——CPU fixture 档的语义就是单设备上的多协程机器面
        （设备集裁剪的生产口径 = ``CUDA_VISIBLE_DEVICES``；测试显式传
        设备，与 ``GranularGrpoTrainer`` 的 device 注入同惯例）。"""
        kwargs.setdefault("devices", [torch.device("cpu")])
        artifacts = RunArtifacts.init(config, self._tmp_path / run_name)
        return AsyncTrainingExecutor.build(config, artifacts, **kwargs)

    def events(self, run_name: str = "run") -> list[dict]:
        """run 目录指标流的外部读取面。"""
        return RunArtifacts(
            RunArtifacts.layout(self._tmp_path / run_name),
        ).read_events()

    def assert_state_dicts_bitwise(
        self, first: dict, second: dict, label: str,
    ) -> None:
        """两张 state_dict 的逐位对账（同 seed 重放的数值面）。"""
        assert set(first) == set(second), label
        for key in first:
            assert torch.equal(first[key].cpu(), second[key].cpu()), (
                f"{label}:{key}"
            )

    def spectral_buffers(self, scorer) -> dict[str, torch.Tensor]:
        """判别器 spectral norm 的 ``_u``/``_v`` buffer 观测面（#218 锚
        的断言面：谱归一化幂迭代不变式的观测载体）。"""
        return {
            name: buffer.detach().cpu()
            for name, buffer in scorer.discriminator.named_buffers()
            if name.endswith("._u") or name.endswith("._v")
        }

    def global_rng_snapshot(self, cuda: bool = False) -> dict:
        """进程全局 RNG 状态快照（#218「训练路径禁碰进程全局 RNG」的
        可测断言取数口；观测窗 = 装配完成后 → run 结束）。"""
        snapshot = {
            "torch": torch.get_rng_state().clone(),
            "python": random.getstate(),
            "numpy_keys": np.random.get_state()[1].copy(),
            "numpy_pos": np.random.get_state()[2],
        }
        if cuda:
            snapshot["cuda"] = torch.cuda.get_rng_state_all()
        return snapshot

    def assert_global_rng_untouched(self, snapshot: dict) -> None:
        """全局 RNG 状态逐位不变断言（四套进程全局 RNG 的训练路径禁区）。"""
        assert torch.equal(snapshot["torch"], torch.get_rng_state())
        assert snapshot["python"] == random.getstate()
        assert np.array_equal(snapshot["numpy_keys"], np.random.get_state()[1])
        assert snapshot["numpy_pos"] == np.random.get_state()[2]
        if "cuda" in snapshot:
            states = torch.cuda.get_rng_state_all()
            assert len(snapshot["cuda"]) == len(states)
            for before, after in zip(snapshot["cuda"], states):
                assert torch.equal(before, after)


class TestAllocationTable:
    """静态分配表纯函数锚（#217 决议 1 / #231）：确定性、轮内置换、
    覆盖与边际、组2 每端对数装配断言。"""

    def test_deterministic_same_seed_same_table(self) -> None:
        first = AllocationTable(("a", "b", "c"), 2, seed=7)
        second = AllocationTable(("a", "b", "c"), 2, seed=7)
        for iteration in range(6):
            for slot in range(2):
                assert (
                    first.condition_for(iteration, slot)
                    == second.condition_for(iteration, slot)
                )

    def test_different_seed_permutes_rounds(self) -> None:
        first = AllocationTable(("a", "b", "c", "d"), 2, seed=0)
        second = AllocationTable(("a", "b", "c", "d"), 2, seed=1)
        readings = [
            (first.condition_for(i, s) != second.condition_for(i, s))
            for i in range(4) for s in range(2)
        ]
        assert any(readings), "不同 seed 的轮置换必须分道"

    def test_full_coverage_per_round_and_uniform_marginal(self) -> None:
        """C ≥ D（MR 生产形态 C=11 > D=4 的缩影）：一轮覆盖全条件一次，
        跨轮边际按条件均匀（每条件每轮恰一次）。"""
        conditions = tuple(f"cond{index}" for index in range(5))
        table = AllocationTable(conditions, 2, seed=0)
        assert table.round_length == 3  # ⌈5/2⌉
        for round_index in range(table.round_length):
            round_conditions = [
                table.condition_for(iteration, slot)
                for iteration in range(
                    round_index * 3, (round_index + 1) * 3,
                )
                for slot in range(2)
            ]
            missing = set(conditions) - set(round_conditions)
            assert not missing, f"轮 {round_index} 未覆盖 {missing}"
            # 6 = 5 + 1：恰一条件出现两次、其余一次（tile 补齐的尾段）
            counts = {name: round_conditions.count(name) for name in conditions}
            assert sorted(counts.values()) == [1, 1, 1, 1, 2]

    def test_within_iteration_distinct_for_c_ge_d(self) -> None:
        """同 iteration 内槽间条件恒互异（#217：一轮内恒互异的良性
        分布性质，vs 现行 i.i.d. 同 iteration 互异概率仅 ~9%）。"""
        table = AllocationTable(("a", "b", "c", "d"), 3, seed=2)
        for iteration in range(6):
            picked = [
                table.condition_for(iteration, slot) for slot in range(3)
            ]
            assert len(set(picked)) == 3, picked

    def test_c_lt_d_tiles_conditions_across_slots(self) -> None:
        """C < D（BraTS 4 条件 × 协程 8 的形态）：条件 tile 到槽、每轮
        重洗、每条件 ⌊D/C⌋ 或 ⌈D/C⌉ 槽——单规则覆盖不留两态特例。"""
        conditions = ("a", "b", "c", "d")
        table = AllocationTable(conditions, 8, seed=5)
        assert table.round_length == 1  # ⌈4/8⌉：每 iteration 一轮（重洗）
        picked = [table.condition_for(0, slot) for slot in range(8)]
        counts = {name: picked.count(name) for name in conditions}
        assert set(counts.values()) == {2}  # ⌈8/4⌉ 恰等分

    def test_slot_out_of_range_rejected(self) -> None:
        table = AllocationTable(("a", "b"), 2, seed=0)
        with pytest.raises(ValueError):
            table.condition_for(0, 2)

    def test_pair_symmetry_accepts_default_pairs(self) -> None:
        pairs = [
            (source, target)
            for source in ("t1n", "t1c", "t2w", "t2f")
            for target in ("t1n", "t1c", "t2w", "t2f")
            if source != target
        ]
        targets = tuple(dict.fromkeys(t for _, t in pairs))
        AllocationTable.assert_pair_symmetry(pairs, targets)  # 不抛即过

    def test_pair_symmetry_rejects_asymmetric_counts(self) -> None:
        pairs = [("t1n", "t1c"), ("t2w", "t1c"), ("t1n", "t2f")]
        with pytest.raises(ValueError, match="非对称"):
            AllocationTable.assert_pair_symmetry(pairs, ("t1c", "t2f"))

    def test_pair_symmetry_rejects_target_outside_axis(self) -> None:
        with pytest.raises(ValueError, match="不一致"):
            AllocationTable.assert_pair_symmetry(
                [("t1n", "ghost")], ("t1n",),
            )


class TestSlotRngRegistry:
    """派生公式单元锚（#218 三层锚①，纯 CPU）：线性步长派生、槽 0
    恒等、槽×流不撞位、出口 owner-thread 断言。"""

    STREAMS = (
        TrainingRngStreams.ROLLOUT,
        TrainingRngStreams.REAL_POOL,
        TrainingRngStreams.HELDOUT_AUC,
        TrainingRngStreams.RECON,
    )

    def test_slot0_identity_with_single_process_registry(self) -> None:
        """槽 0 恒等：数值面与现行单进程派生值逐位相同（#218 决策 0
        ——槽 0 恒等 + recon 挪轴 world-1 数值不变）。"""
        seed = 123
        registry = SlotRngRegistry(seed, slots=2)
        legacy = TrainingRngStreams(seed, shared_seed=seed)
        for stream in self.STREAMS:
            expected = torch.randn(8, generator=getattr(legacy, stream))
            actual = torch.randn(
                8, generator=registry.get_stream(0, stream),
            )
            assert torch.equal(expected, actual), stream

    def test_linear_stride_derivation(self) -> None:
        seed = 55
        registry = SlotRngRegistry(seed, slots=3)
        for slot in range(3):
            expected = torch.randn(
                8,
                generator=torch.Generator().manual_seed(
                    seed + slot * SLOT_SEED_STRIDE,
                ),
            )
            actual = torch.randn(
                8, generator=registry.get_stream(slot, "rollout"),
            )
            assert torch.equal(expected, actual), slot

    def test_slot_stream_combinations_do_not_collide(self) -> None:
        """槽×流不撞位（#218 撞位审计的机器面：最大偏移差 19 ≪ 槽间距）。"""
        registry = SlotRngRegistry(9, slots=4)
        readings = {
            (slot, stream): torch.randn(
                4, generator=registry.get_stream(slot, stream),
            ).sum().item()
            for slot in range(4) for stream in self.STREAMS
        }
        assert len(set(readings.values())) == len(readings)

    def test_get_stream_returns_same_instance(self) -> None:
        """同一 (槽, 流) 恒返同一实例（在线零复位的注册表面：无复位
        换装入口，流跨 iteration 连续推进）。"""
        registry = SlotRngRegistry(1, slots=1)
        first = registry.get_stream(0, "rollout")
        assert registry.get_stream(0, "rollout") is first

    def test_stream_state_observation_does_not_consume(self) -> None:
        registry = SlotRngRegistry(1, slots=1)
        before = registry.stream_state(0, "rollout")
        after = registry.stream_state(0, "rollout")
        assert torch.equal(before, after)

    def test_owner_thread_assertion_fails_fast_cross_thread(self) -> None:
        registry = SlotRngRegistry(1, slots=2, owner_check=True)
        registry.get_stream(0, "rollout")  # 当前线程成为 owner
        errors: list[Exception] = []

        class ForeignThread:
            def __call__(self) -> None:
                try:
                    registry.get_stream(0, "rollout")
                except RuntimeError as error:
                    errors.append(error)

        thread = threading.Thread(target=ForeignThread())
        thread.start()
        thread.join()
        assert errors and "异线程" in str(errors[0])

    def test_owner_check_disabled_by_default(self) -> None:
        registry = SlotRngRegistry(1, slots=1)
        registry.get_stream(0, "rollout")
        registry.get_stream(0, "rollout")  # 无 owner 记录、不拒绝

    def test_unknown_stream_and_slot_rejected(self) -> None:
        registry = SlotRngRegistry(1, slots=1)
        with pytest.raises(ValueError):
            registry.get_stream(0, "nope")
        with pytest.raises(ValueError):
            registry.get_stream(5, "rollout")


class TestBarrierTimeoutPolicy:
    """barrier 软/硬超时口径锚（``CYNOSURE_PG_TIMEOUT_MIN`` 语义换绑）。"""

    def test_env_semantics_rebound_to_barrier_hard_timeout(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(BarrierTimeoutPolicy.ENV, "40")
        policy = BarrierTimeoutPolicy.from_env()
        assert policy.hard_seconds == 40 * 60
        assert policy.soft_seconds == 20 * 60

    def test_env_unset_keeps_ten_minute_default(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv(BarrierTimeoutPolicy.ENV, raising=False)
        policy = BarrierTimeoutPolicy.from_env()
        assert policy.hard_seconds == 10 * 60

    def test_env_invalid_value_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(BarrierTimeoutPolicy.ENV, "0")
        with pytest.raises(ValueError):
            BarrierTimeoutPolicy.from_env()

    def test_explicit_construction_scales_soft(self) -> None:
        policy = BarrierTimeoutPolicy(hard_seconds=0.05)
        assert policy.soft_seconds == pytest.approx(0.025)

    def test_soft_fraction_override_decouples_thresholds(self) -> None:
        """软比率构造解耦（测试档旋钮）：软阈值钉到远小于 barrier 实耗时
        ——告警触发不再依赖「总耗时落在软硬阈值之间」的机器裕度。"""
        policy = BarrierTimeoutPolicy(hard_seconds=600.0, soft_fraction=0.0002)
        assert policy.soft_seconds == pytest.approx(0.12)

    def test_soft_fraction_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            BarrierTimeoutPolicy(hard_seconds=1.0, soft_fraction=2.0)


@pytest.mark.gpu
@pytest.mark.slow
class TestExecutorFixtureTier:
    """CPU fixture 档（slow+gpu 挂法）：门面全宽薄切片端到端 + 多协程
    重放锚 + 异常/超时门面。"""

    def test_single_iteration_full_width_brats(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """fixture 尺度端到端一 iteration 走通：分配表 → 逐 k 收集-同步
        → 事件发射，全程测试进程内可断言（#231 AC1）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        executor = scenario.build(config, coroutines=3, owner_check=True)
        executor.run()
        events = scenario.events()
        assert len(events) == 3
        # (iteration, slot) 排序写 + rank 字段语义 = 槽号
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (0, 0), (0, 1), (0, 2),
        ]
        # 事件条件 = 分配表（分配表驱动执行序的机器断言）
        for event in events:
            assert event["modality"] == executor.allocation.condition_for(
                0, event["rank"],
            )
            assert math.isfinite(event["anchor_eval_reward"])
            assert math.isfinite(event["heldout_auc"])
            assert math.isfinite(event["loss"]["policy_step_1"])
            assert event["train_pairwise_acc"] is None  # 判别器链缺席口径
            # 逐例相位分解（#217 PhaseTimer 新口径：per-例子 phase_seconds）
            assert set(event["phase_seconds"]) == {
                "rollout", "policy_update",
            }
            assert event["phase_seconds"]["rollout"] > 0
            assert event["phase_seconds"]["policy_update"] > 0

    def test_two_iterations_round_progression(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=3)
        executor.run()
        events = scenario.events()
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ]
        for event in events:
            assert event["modality"] == executor.allocation.condition_for(
                event["iteration"], event["rank"],
            )

    def test_mr_rate_domain_full_width(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """两域贯穿之 MR-RATE：异条件异形词汇表 + C=2 < D=3 的 tile
        路径（每 iteration 条件 tile 到槽）。"""
        scene = MrTrainScenario(cli, tmp_path)
        prepared = scene.prepare(
            reward_overrides={"pretrain_pass_threshold": 0.01},
        )
        config = scene.pretrain(prepared)
        config.policy.num_inference_steps = 3
        config.policy.train_step_indices_m = {1}
        config.schedule.seed = 0
        config.schedule.max_iterations = 1
        artifacts = RunArtifacts.init(config, tmp_path / "run")
        executor = AsyncTrainingExecutor.build(
            config, artifacts, coroutines=3, owner_check=True,
            devices=[torch.device("cpu")],
        )
        executor.run()
        events = [
            event for event in artifacts.read_events()
            if event["event"] == "iter"
        ]
        assert len(events) == 3
        assert {
            executor.allocation.condition_for(0, rank) for rank in range(3)
        } <= {"t1w/axial", "flair/axial"}
        for event in events:
            assert event["modality"] == executor.allocation.condition_for(
                0, event["rank"],
            )
            assert math.isfinite(event["heldout_auc"])

    def test_cross_modal_group_full_width(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """两域贯穿之组2：条件轴 = 目标端（每端对数断言过闸），源序列
        自由度留槽内流（sample_target 显式传槽流）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(group="cross-modal", seed=0)
        executor = scenario.build(config, coroutines=2, owner_check=True)
        executor.run()
        events = scenario.events()
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (0, 0), (0, 1),
        ]
        targets = executor.allocation.conditions
        assert len(targets) == 4  # 组2 条件轴 = 4 目标端
        for event in events:
            assert event["modality"] in targets

    def test_same_seed_replay_bitwise_multi_coroutine(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """多协程重放锚（#218 三层锚②，协程数 > 1）：同 config 同 seed
        两次 run——权重 + 判别器 u/v buffer + RunTrajectory 事件流逐位
        一致（SN 启用使 u/v buffer 在场）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(
            seed=3, reward={"spectral_norm_enabled": True},
        )
        config.schedule.max_iterations = 2
        first = scenario.build(config, run_name="run1", coroutines=2)
        first.run()
        second = scenario.build(config, run_name="run2", coroutines=2)
        second.run()
        scenario.assert_state_dicts_bitwise(
            first.cards[0].replica.policy.full_state(),
            second.cards[0].replica.policy.full_state(),
            "policy",
        )
        first_buffers = scenario.spectral_buffers(first.cards[0].replica.scorer)
        second_buffers = scenario.spectral_buffers(second.cards[0].replica.scorer)
        assert set(first_buffers) == set(second_buffers)
        assert first_buffers, "SN 启用下 _u/_v buffer 须在场（断言有判别力）"
        scenario.assert_state_dicts_bitwise(
            first_buffers, second_buffers, "spectral",
        )
        assert RunTrajectory(scenario.events("run1")) == RunTrajectory(
            scenario.events("run2"),
        )

    def test_streams_advance_without_reset(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """在线零复位不变式（#218）：流跨 iteration 连续推进——run 后的
        流状态 ≠ 注册时的初始派生状态（槽 0 的线性派生 = seed + 流偏移），
        且无复位换装（同一实例，见 TestSlotRngRegistry）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=4)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=2)
        executor.run()
        for stream, offset in (("rollout", 0), ("heldout_auc", 3)):
            state = executor.rng.stream_state(0, stream)
            initial = torch.Generator().manual_seed(4 + offset)
            assert not torch.equal(state, initial.get_state()), stream

    def test_global_rng_untouched_across_run(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=4)
        executor = scenario.build(config, coroutines=2)
        snapshot = scenario.global_rng_snapshot(cuda=torch.cuda.is_available())
        executor.run()
        scenario.assert_global_rng_untouched(snapshot)

    def test_slot_exception_aborts_fail_fast(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """异常 fail-fast 门面（#217 §4）：槽线程异常 → 训练中止异常
        （原异常挂 cause）+ 线程收尾，进程内可断言。故障注入挂在第二个
        构造的 AUC 实例上（槽装配序 = 槽序，实例在绑卡线程槽装配期
        诞生——构造序是装配面的结构事实）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=5)
        constructed: list = []
        real_init = HeldOutAuc.__init__

        def spy_init(self_auc, *args, **kwargs):
            real_init(self_auc, *args, **kwargs)
            constructed.append(self_auc)

        monkeypatch.setattr(HeldOutAuc, "__init__", spy_init)
        real_compute = HeldOutAuc.compute

        def selective(self_auc, fakes, modality=None):
            if len(constructed) > 1 and self_auc is constructed[1]:
                raise RuntimeError("槽内故障注入")
            return real_compute(self_auc, fakes, modality)

        monkeypatch.setattr(HeldOutAuc, "compute", selective)
        executor = scenario.build(config, coroutines=3, owner_check=True)
        with pytest.raises(TrainingAborted) as caught:
            executor.run()
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert all(card.stopped_within(30.0) for card in executor.cards)

    def test_barrier_soft_timeout_warns_then_completes(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """barrier 软超时（#217 §4）：越过软阈值发
        ``barrier_soft_timeout`` 事件（主线程 = 卡 0 写出口径）后继续
        等待，训练正常完成。软阈值钉到远小于 barrier 实耗时（睡眠主导
        ——触发与机器计算耗时无关），硬阈值放大到不构成中止压力。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=6)
        executor = scenario.build(
            config,
            coroutines=2,
            timeout=BarrierTimeoutPolicy(
                hard_seconds=600.0, soft_fraction=0.0002,  # 软阈值 0.12s
            ),
        )
        updater = executor.cards[0].replica.updater
        real_accumulate = updater.accumulate

        def sluggish(*args, **kwargs):
            time.sleep(1.0)  # barrier 等待必然越过 0.12s 软阈值
            return real_accumulate(*args, **kwargs)

        monkeypatch.setattr(updater, "accumulate", sluggish)
        executor.run()
        warnings = [
            event for event in scenario.events()
            if event["event"] == "barrier_soft_timeout"
        ]
        assert warnings
        assert warnings[0]["iteration"] == 0
        assert warnings[0]["step_index"] == 1
        assert warnings[0]["waited_s"] >= warnings[0]["threshold_s"]
        iter_events = [
            event for event in scenario.events()
            if event["event"] == "iter"
        ]
        assert len(iter_events) == 2  # 训练照常完成（软超时只告警不动作）

    def test_barrier_hard_timeout_aborts(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """barrier 硬超时（#217 §4）：``CYNOSURE_PG_TIMEOUT_MIN`` 换绑
        口径的 fail-fast——硬阈值到即训练中止。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=7)
        executor = scenario.build(
            config,
            coroutines=2,
            timeout=BarrierTimeoutPolicy(hard_seconds=0.05),
        )
        updater = executor.cards[0].replica.updater
        real_accumulate = updater.accumulate

        def stalled(*args, **kwargs):
            # 槽内串行：两槽各睡 0.8s——障碍远超硬阈值 0.05s；退绕余量
            # 见断言前的宽限 join
            time.sleep(0.8)
            return real_accumulate(*args, **kwargs)

        monkeypatch.setattr(updater, "accumulate", stalled)
        with pytest.raises(TrainingAborted, match="硬超时"):
            executor.run()
        warnings = [
            event for event in scenario.events()
            if event["event"] == "barrier_soft_timeout"
        ]
        assert warnings  # 硬超时前先有软超时告警
        # 中止后的线程退绕（槽内串行睡眠 + 运行时间歇停顿）可能超出
        # executor 的限时 join 窗口——宽限观测面收尾后再断言线程已停
        assert all(card.stopped_within(30.0) for card in executor.cards)


@pytest.mark.gpu
@pytest.mark.slow
class TestExecutorMultiCardTier:
    """gauss 多卡 e2e 档（slow+gpu；换发射方式重建，#217 门面 + M1）：
    跨卡一致性 + 层 3 同 seed 多卡重放锚薄切片形态。"""

    @staticmethod
    def _visible_card_count() -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def test_multicard_weights_and_events_across_cards(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """单进程多卡端到端：跨卡权重与 spectral buffer 逐位一致（确定性
        allreduce + 同步 step 序列的结构性保证），事件 (iteration, slot)
        完备有序。"""
        card_count = self._visible_card_count()
        if card_count < 2:
            pytest.skip("多卡档：需要 ≥2 CUDA 设备（gauss 4×A6000 口径）")
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=min(card_count, 4))
        executor.run()
        slots = executor.allocation.slot_count
        reference = executor.cards[0].replica.policy.full_state()
        reference_buffers = scenario.spectral_buffers(
            executor.cards[0].replica.scorer,
        )
        for card in executor.cards[1:]:
            scenario.assert_state_dicts_bitwise(
                reference, card.replica.policy.full_state(),
                f"policy(card{card.index})",
            )
            scenario.assert_state_dicts_bitwise(
                reference_buffers, scenario.spectral_buffers(card.replica.scorer),
                f"spectral(card{card.index})",
            )
        events = scenario.events()
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (iteration, slot)
            for iteration in range(2) for slot in range(slots)
        ]
        for event in events:
            assert event["modality"] == executor.allocation.condition_for(
                event["iteration"], event["rank"],
            )

    def test_multicard_same_seed_replay_bitwise(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """层 3 薄切片锚（#218 三层锚③，gauss 多卡）：同 seed 两 run——
        卡 0 权重 + spectral buffer + RunTrajectory 事件流逐位一致
        （NCCL allreduce 逐位确定，#215 双栈实证；全强度形态随判别器
        链期，#226）。"""
        card_count = self._visible_card_count()
        if card_count < 2:
            pytest.skip("多卡档：需要 ≥2 CUDA 设备（gauss 4×A6000 口径）")
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(
            seed=3, reward={"spectral_norm_enabled": True},
        )
        config.schedule.max_iterations = 2
        first = scenario.build(
            config, run_name="run1", coroutines=min(card_count, 4),
        )
        first.run()
        second = scenario.build(
            config, run_name="run2", coroutines=min(card_count, 4),
        )
        second.run()
        scenario.assert_state_dicts_bitwise(
            first.cards[0].replica.policy.full_state(),
            second.cards[0].replica.policy.full_state(),
            "policy(card0)",
        )
        scenario.assert_state_dicts_bitwise(
            scenario.spectral_buffers(first.cards[0].replica.scorer),
            scenario.spectral_buffers(second.cards[0].replica.scorer),
            "spectral(card0)",
        )
        assert RunTrajectory(scenario.events("run1")) == RunTrajectory(
            scenario.events("run2"),
        )
