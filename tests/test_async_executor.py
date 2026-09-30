"""async 执行模型的测试档（#231 骨架期 → #234 判别器链期）：三层锚 +
门面全宽薄切片。

档位与挂法（#226 决策 3 / #231）：

- **单元锚（无标记，纯 CPU）**：派生公式单元锚（线性步长、槽 0 恒等、
  槽×流不撞位）+ 分配表纯函数锚（轮内置换/覆盖/边际均匀/组2 对数断言）
  + 超时口径解析锚；判别器链纯函数锚（test_discriminator_chain：桶/
  窗口计划/real 抽取/逐桶等权/相位序列）。
- **CPU fixture 档（slow+gpu）**：门面全宽薄切片端到端——分配表 →
  per-槽协程 rollout → 池化 AUC → 逐 k barrier 收集-同步 → 判别器步
  → 事件发射，测试进程内可断言；多协程重放锚（#218 锚②：同 seed 两
  次 run 权重 + 判别器 u/v buffer + RunTrajectory 事件流逐位）；异常
  fail-fast 与 barrier 软/硬超时可触发可断言；两域贯穿（BraTS +
  MR-RATE + 组2）；判别器链锚（#234：窗口任务摊派/real 抽取/recon
  流任务粒度/事件 per-condition 明细/池化 AUC）。
- **gauss 多卡 e2e 档（slow+gpu）**：单进程多卡换发射方式重建的验收
  路径（#217 门面 + M1 单点归约）——跨卡权重/u-v 逐位一致 + 层 3
  同 seed 多卡重放锚（判别器链全强度：梯度 allreduce SUM + 步末
  u/v broadcast 后的跨卡一致进锚）。

slow+gpu 挂法：默认跳过、CPU 环境跳过（conftest 双 marker 轴），验证
职责由 gauss ``pytest --run-slow`` 全量承担（仓库纪律：测试一律上集群）。
"""

import math
import random
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.assembly import PairBatch, ReconstructionAssembler
from cynosure.reward.scorer import ChunkedScorer
from cynosure.train.allocation import AllocationTable
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.discriminator import WindowRealDraw
from cynosure.train.executor import (
    AsyncTrainingExecutor,
    BarrierTimeoutPolicy,
    SlotExample,
    SlotRunner,
    TrainingAborted,
)
from cynosure.train.policy import GroupPolicy
from cynosure.train.rng import (
    SLOT_SEED_STRIDE,
    DropoutGuard,
    SlotRngRegistry,
    TrainingRngStreams,
)
from cynosure.train.runtime import TrainingRuntime
from tests.conftest import (
    WALL_CLOCK_EVENT_FIELDS,
    CliSession,
    FixtureArtifactLibrary,
    RunTrajectory,
)
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

    def spy_window_pairs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> list[tuple[torch.Tensor, str, PairBatch]]:
        """``ReconstructionAssembler.reconstruct_assigned`` 的产出记录
        spy（#234 窗口任务的消费面：供给入口的 (reals, modality) 与
        返回 PairBatch 一并记录——桶构造/任务粒度的观测原料）。单绑卡
        线程内调用序串行，list.append 无竞争。"""
        calls: list[tuple[torch.Tensor, str, PairBatch]] = []
        real_reconstruct = ReconstructionAssembler.reconstruct_assigned

        def recording(assembler, reals, modality):
            pair = real_reconstruct(assembler, reals, modality)
            calls.append((reals, modality, pair))
            return pair

        monkeypatch.setattr(
            ReconstructionAssembler, "reconstruct_assigned", recording,
        )
        return calls

    def spy_slot_examples(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> list[SlotExample]:
        """``SlotRunner.run_example`` 的返回记录 spy：``SlotExample``
        协程内创建、run 后不可达，经 wrap 真实现取返回值观测记账链路
        的消费者端（装配层 spy 与流锚之下，若记账字段静默缺失
        （window_pairs 空载）而装配照调，本 spy 捕获的实例暴露）。"""
        examples: list[SlotExample] = []
        real_run_example = SlotRunner.run_example

        async def recording(
            runner, condition_name, *, window_tasks=(), score_fakes=True,
        ):
            example = await real_run_example(
                runner, condition_name, window_tasks=window_tasks,
                score_fakes=score_fakes,
            )
            examples.append(example)
            return example

        monkeypatch.setattr(SlotRunner, "run_example", recording)
        return examples

    @staticmethod
    def replay_recon_stream(
        before: torch.Tensor,
        *,
        strokes: int,
        latent_shape: tuple[int, ...],
        step_count: int,
    ) -> torch.Tensor:
        """recon 流消耗序的重放终态（#234 任务粒度）：每笔 =
        ``randint(step_count, (1,))``（先 s）+ ``randn((1, *latent))``
        （ε）——窗口任务逐对抽取的消耗形状单点；与注册表
        ``stream_state`` 逐位比较，任一失守（次序/消耗量/复位）即分道。"""
        replay = torch.Generator()
        replay.set_state(before)
        for _ in range(strokes):
            torch.randint(step_count, (1,), generator=replay)
            torch.randn((1, *latent_shape), generator=replay)
        return replay.get_state()

    def assert_state_dicts_bitwise(
        self, first: dict, second: dict, label: str,
    ) -> None:
        """两张 state_dict 的逐位对账（同 seed 重放的数值面）。"""
        assert set(first) == set(second), label
        for key in first:
            assert torch.equal(first[key].cpu(), second[key].cpu()), (
                f"{label}:{key}"
            )

    @staticmethod
    def assert_state_dicts_close(
        first: dict, second: dict, label: str,
    ) -> None:
        """两张 state_dict 的容差对账（rtol=1e-2、atol=5e-5——实测漂移
        包络的一数倍余量：2 iteration 后全张量最大绝对漂移 2.2e-5，近零
        权重张量是绝对漂移的主要失守面、rtol 在其上失效）。多卡档重放
        锚的数值面：kernel 路径的地址/对齐敏感选择使同进程/跨进程逐位
        不可达（#234 评审收敛期探测定谳，见 test_multicard_same_seed_
        replay docstring），协议级非确定性（RNG 流错接、barrier 序乱）
        为 O(1) 仍被本容差捕获。"""
        assert set(first) == set(second), label
        for key in first:
            assert torch.allclose(
                first[key].cpu(), second[key].cpu(), rtol=1e-2, atol=5e-5,
            ), f"{label}:{key}"

    @staticmethod
    def assert_replay_values_close(
        left, right, path: str, *, rel: float,
    ) -> None:
        """重放事件值的容差递归对账：float 叶 rel（调用方按 iteration
        分代——漂移随训练步放大：iter 0 紧锚 rel=1e-2，iter ≥ 1 放松
        rel=1e-1；观测包络：判别器损失分量 2 iteration 后相对漂移
        2.2%）+ abs=1e-3（近零损失 surrogate 抵消放大面）；None/
        int/str/bool 叶与容器键集精确相等——结构面不留容差。"""
        if isinstance(left, float) or isinstance(right, float):
            assert left is not None and right is not None, (
                f"{path}: {left!r} != {right!r}（None 与 float 不可对拍）"
            )
            assert float(left) == pytest.approx(
                float(right), rel=rel, abs=1e-3,
            ), f"{path}: {left!r} != {right!r}"
            return
        if isinstance(left, dict):
            assert isinstance(right, dict) and set(left) == set(right), (
                f"{path}: 键集 {sorted(left)} != {sorted(right)}"
            )
            for key in left:
                ExecutorScenario.assert_replay_values_close(
                    left[key], right[key], f"{path}.{key}", rel=rel,
                )
            return
        assert left == right, f"{path}: {left!r} != {right!r}"

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


class TestDropoutGuard:
    """装配期 dropout 守卫锚（#218 §3：dropout 绕开注册表消耗随机流，
    装配期 fail-fast；policy/scorer 两侧的装配通过由 fixture 档全流程
    隐式覆盖——真实网络树全零 dropout 才能装配）。"""

    def test_rejects_positive_dropout(self) -> None:
        block = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.2),
        )
        with pytest.raises(ValueError, match="dropout=0.2"):
            DropoutGuard.assert_clean(block, origin="探针")

    def test_allows_zero_dropout_and_plain_modules(self) -> None:
        block = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.Dropout(p=0.0),
        )
        DropoutGuard.assert_clean(block, origin="探针")


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
        → 判别器步 → 事件发射，全程测试进程内可断言（#231 AC1）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        executor = scenario.build(config, coroutines=3, owner_check=True)
        executor.run()
        events = [
            event for event in scenario.events()
            if event["event"] == "iter"
        ]
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
            # 判别器链接上（#234，N_d=1 每 iter 判别器步）：per-例
            # 桥接单值 = 本例条件读数 + 逐条件明细在场
            assert event["train_pairwise_acc"] is not None
            assert 0.0 <= event["train_pairwise_acc"] <= 1.0
            detail = event["disc_update"]
            assert detail is not None
            assert detail["global_batch_size"] == (
                config.reward.disc_batch_size_k * 1  # CPU fixture 单卡
            )
            assert event["modality"] in detail["conditions"]
            reading = detail["conditions"][event["modality"]]
            assert reading["train_pairwise_acc"] == event["train_pairwise_acc"]
            assert math.isfinite(reading["loss"])
            # 池化 AUC「每活跃条件恰一次读数」的事件面（#220 决议 12）：
            # per-例桥接单值 = 本例条件在窗口账簿的最后一次池化读数
            # （modal-label 组各例条件不同、桥接值各异为口径本身；对拍
            # 窗口账簿单点 + 同条件跨例唯一 = 「恰一次读数」的判别面）
            assert event["heldout_auc"] == executor.window_run.window_auc(
                event["modality"]
            )
        by_condition: dict[str, set[float]] = {}
        for event in events:
            by_condition.setdefault(event["modality"], set()).add(
                event["heldout_auc"]
            )
        for condition, values in by_condition.items():
            assert len(values) == 1, (
                f"条件 {condition} 同 iter 池化读数不唯一——「恰一次"
                "读数」失守（per-例独立测量的旧口径不可能逐位相等）"
            )
            # 逐例相位分解（#217 PhaseTimer 新口径：per-例子 phase_seconds）
            assert set(event["phase_seconds"]) == {
                "rollout", "policy_update", "discriminator",
            }
            assert event["phase_seconds"]["rollout"] > 0
            assert event["phase_seconds"]["policy_update"] > 0
            assert event["phase_seconds"]["discriminator"] > 0
            assert "discriminator" in event["loss"]

    def test_two_iterations_round_progression(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=3)
        executor.run()
        events = [
            event for event in scenario.events()
            if event["event"] == "iter"
        ]
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
        自由度留槽内流（sample_target 显式传槽流）。判别器链接上后
        （#234）overfit_alert 合法插入流（fixture 小池记忆化越线，
        test_distributed 同款口径）——归并序断言只锁 iter 事件。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(group="cross-modal", seed=0)
        executor = scenario.build(config, coroutines=2, owner_check=True)
        executor.run()
        events = scenario.events()
        iter_events = [e for e in events if e["event"] == "iter"]
        assert [(e["iteration"], e["rank"]) for e in iter_events] == [
            (0, 0), (0, 1),
        ]
        iter_pairs = {(e["iteration"], e["rank"]) for e in iter_events}
        for event in events:
            if event["event"] == "overfit_alert":
                # 主线程写出口径（executor 事件段）：alert rank 恒 0——
                # 原 (iteration, rank) 成员资格断言对 rank 轴无判别力
                # （alert 无槽面、槽 0 每 iter 恒有 iter 事件，恒真）；
                # 显式锁 rank 常量 + iteration 不越调度域两可判别面
                assert event["rank"] == 0
                assert (event["iteration"], 0) in iter_pairs
        targets = executor.allocation.conditions
        assert len(targets) == 4  # 组2 条件轴 = 4 目标端
        for event in iter_events:
            assert event["modality"] in targets

    def test_same_seed_replay_bitwise_multi_coroutine(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """多协程重放锚（#218 三层锚②，协程数 > 1）：同 config 同 seed
        两次 run——权重 + 判别器 u/v buffer + RunTrajectory 事件流逐位
        一致（SN 启用使 u/v buffer 在场）。#233 起日程升 multi-k
        （num_steps=5、M={1,2,3}）——更新相逐 k barrier 收集-同步的
        完整形态进重放锚（单 k 薄切片锚随 #231 结票退役）。

        本档承载**逐位数值重放**的判别力：多卡档因判别器 patch conv
        的 ATen 原生路径对内存对齐敏感（机制与证据见
        test_multicard_same_seed_replay docstring）改为容差对拍，
        逐位数值锚由本档（CPU 单卡多协程、该几何分配模式实测稳定）
        独占承担——协议确定性与设备无关。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(
            seed=3,
            num_steps=5,
            train_steps={1, 2, 3},
            reward={"spectral_norm_enabled": True},
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
        for run in ("run1", "run2"):
            for event in scenario.events(run):
                if event["event"] != "iter":
                    continue  # overfit_alert 等其它族事件无 loss 面
                assert set(event["loss"]) == {
                    "policy_step_1", "policy_step_2", "policy_step_3",
                    "discriminator",  # N_d=1：判别器步随 iter，loss 可扩键
                }, "M={1,2,3} 的每个 k 都须真实发生累积与 step"
        assert RunTrajectory(scenario.events("run1")) == RunTrajectory(
            scenario.events("run2"),
        )

    def test_streams_advance_without_reset(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """在线零复位不变式（#218）：流跨 iteration 连续推进——run 后的
        流状态 ≠ 注册时的初始派生状态（槽 0 的线性派生 = seed + 流偏移），
        且无复位换装（同一实例，见 TestSlotRngRegistry）。rollout 期起
        recon 流随窗口任务进场（#234：任务粒度消耗、N_d 窗口逐 iter
        摊派）；heldout 流随池化 AUC 每判别器步消耗（#234：窗口末
        「每活跃条件恰一次读数」，非判别器步 iter 无消耗）；
        real_pool 流随
        #232 过渡形态退役（#234：real 侧是池数据不是随机流——窗口抽取
        走一次性派生 generator，注册表流零消耗）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=4)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=2)
        executor.run()
        for stream, offset in (
            ("rollout", 0), ("heldout_auc", 3), ("recon", 9),
        ):
            state = executor.rng.stream_state(0, stream)
            initial = torch.Generator().manual_seed(4 + offset)
            assert not torch.equal(state, initial.get_state()), stream
        # real_pool 流零消耗（退役断言：状态 == 初始派生 = 无人触碰）
        pool_state = executor.rng.stream_state(0, TrainingRngStreams.REAL_POOL)
        pool_initial = torch.Generator().manual_seed(4 + 1)
        assert torch.equal(pool_state, pool_initial.get_state()), (
            "real_pool 注册表流应零消耗（#234 窗口抽取不走命名流）——"
            "残留消耗 = 供给语义接线漏改"
        )

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
        构造的 ``ChunkedScorer`` 上（槽装配序 = 槽序、每槽一实例——
        #234 后 rollout 相打分走 ChunkedScorer；HeldOutAuc.compute 的
        per-槽消费面已随 per-例 AUC 槽实例退役）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=5)
        constructed: list = []
        real_init = ChunkedScorer.__init__

        def spy_init(self_scorer, *args, **kwargs):
            real_init(self_scorer, *args, **kwargs)
            constructed.append(self_scorer)

        monkeypatch.setattr(ChunkedScorer, "__init__", spy_init)
        real_scores = ChunkedScorer.scores

        def selective(self_scorer, latents):
            if len(constructed) > 1 and self_scorer is constructed[1]:
                raise RuntimeError("槽内故障注入")
            return real_scores(self_scorer, latents)

        monkeypatch.setattr(ChunkedScorer, "scores", selective)
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
class TestDiscriminatorChainWindow:
    """判别器链期窗口任务锚（#234 / #220 决议 5/10/16）：任务进 rollout
    相（窗口逐 iter 摊派 floor(K/L)、余数补窗口首 iter）+ real 侧窗口
    起点全局无放回抽取（注册表 real_pool 流退役、一次性派生 generator）
    + recon 流任务粒度消耗序（先 s 后 ε、跨 iteration 连续）+ 配对
    fake = rollout 相当前权重 θ_t（drift #1 时点前移保持）。

    BraTS 单域逐条件同形（fixture latent 恒 [4,16,16,8]）：ε 重放形状
    与条件无关——消耗量锚在单域档钉死；MR 异形档的消耗面由既有
    full-width 锚隐式覆盖（N_d=1 默认跑任务在场）。"""

    def test_window_tasks_produce_pairs_per_iteration(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """N_d=1（默认节奏）：窗口长 1、K 任务全落每 iter——供给入口
        调用数 = K × iteration × 1 卡；单对产出属性（批维 1、两侧同形、
        CPU 暂存、inference 无梯度）；任务条件 = 任务槽的分配表条件。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        config.schedule.max_iterations = 2
        calls = scenario.spy_window_pairs(monkeypatch)
        examples = scenario.spy_slot_examples(monkeypatch)
        executor = scenario.build(config, coroutines=2, owner_check=True)
        executor.run()
        batch_k = config.reward.disc_batch_size_k
        # 2 iteration × K 任务/卡（窗口长 1：K 全落窗口首 iter）
        assert len(calls) == 2 * batch_k
        assert len(examples) == 4  # 2 iteration × 2 槽
        assert all(
            example.window_pairs for example in examples
        ), "SlotExample.window_pairs 记账缺失（run_example 返回构造断开）"
        produced = Counter(
            modality for _, modality, _ in calls
        )
        expected = Counter(
            executor.allocation.condition_for(iteration, slot)
            for iteration in range(2) for slot in range(2)
            for _ in range(batch_k // 2)  # 卡内槽轮转：每槽 K/2 任务
        )
        assert produced == expected, "重构任务条件 ≠ 分配表派发条件"
        latent_shape = tuple(config.latent_shape)
        for reals, modality, pair in calls:
            assert pair.reals.shape == (1, *latent_shape)
            assert pair.fakes.shape == pair.reals.shape  # 同源配对同形
            assert pair.fakes.requires_grad is False  # inference 前向无图
            assert torch.isfinite(pair.fakes).all()
            assert pair.reals.device.type == "cpu"

    def test_window_task_spread_across_iterations(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """窗口逐 iter 摊派（#220 决议 5）：N_d=2、K=4——窗口(0)=[0]
        发射 [4]；窗口(2)=[1,2] 发射 [2,2]（floor 均匀）。任务在窗口内
        逐 iter 发射（含非判别器步 iter——窗口摊派形态取代 #232 的
        「判别器步 iteration 才重构」节奏）；recon 流消耗随任务、
        real_pool 注册表流零消耗。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=4)
        config.schedule.max_iterations = 2
        config.reward.disc_update_interval_n_d = 2
        calls = scenario.spy_window_pairs(monkeypatch)
        executor = scenario.build(config, coroutines=2, owner_check=True)
        before = executor.rng.stream_state(0, TrainingRngStreams.RECON)
        executor.run()
        assert len(calls) == 4 + 2  # iter0 窗口(0) 4 笔 + iter1 窗口(2) 首段 2 笔
        after = executor.rng.stream_state(0, TrainingRngStreams.RECON)
        assert torch.equal(after, scenario.replay_recon_stream(
            before,
            strokes=3,  # 槽 0：iter0 2 笔（j%2）+ iter1 1 笔
            latent_shape=tuple(config.latent_shape),
            step_count=len(sorted(config.policy.train_step_indices_m)),
        )), (
            "recon 流消耗 ≠ 窗口任务逐对（先 s 后 ε）——摊派节奏失守或"
            "跨 iteration 复位/额外消耗"
        )
        after_pool = executor.rng.stream_state(
            0, TrainingRngStreams.REAL_POOL,
        )
        pool_initial = torch.Generator().manual_seed(
            4 + 1,
        ).get_state()
        assert torch.equal(after_pool, pool_initial), (
            "real_pool 注册表流应零消耗（#234 窗口抽取不走命名流）"
        )
        # 分叉窗口配对的手工复算（#220 决议 13/14，首观测精确相等）：
        # iter0（窗口(0)=[0] 判别器步）的 divergence_ema = 本条件桶
        # train acc − 窗口内最后一次（= 同 iter）池化 AUC。精确锚落在
        # **本例条件**——事件面 heldout_auc 是本例条件的桥接单值，其
        # 余条件的池化 AUC 不在事件面（per-条件配对的正确性由
        # OverfitMonitor 单测承担），多条件域对其余条件只锚读数在场。
        for event in scenario.events():
            if event["event"] != "iter" or event["disc_update"] is None:
                continue
            assert event["overfit_divergence_ema"] is not None
            own = event["disc_update"]["conditions"][event["modality"]]
            assert own["divergence_ema"] == pytest.approx(
                own["train_pairwise_acc"] - event["heldout_auc"],
                abs=1e-9,
            ), "分叉 ≠ train acc − 池化 AUC（#220 决议 13 口径失守）"
            assert all(
                reading["divergence_ema"] is not None
                for reading in event["disc_update"]["conditions"].values()
            )
            # per-例桥接单值 = 本例条件读数
            assert event["overfit_divergence_ema"] == pytest.approx(
                own["divergence_ema"], abs=1e-12,
            )

    def test_recon_stream_consumption_order_s_before_epsilon(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """per-槽消耗序锚（#218「先 s 后 ε」次序契约的 executor 面，
        #234 任务粒度）：N_d=1 两 iteration 的槽 0 recon 流状态差 =
        (randint(|M|,(1,)) + randn((1,*latent))) × 每槽任务数 × 2——
        消耗量与次序任一失守即逐位分道。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=3)
        config.schedule.max_iterations = 2
        executor = scenario.build(config, coroutines=2, owner_check=True)
        before = executor.rng.stream_state(0, TrainingRngStreams.RECON)
        executor.run()
        after = executor.rng.stream_state(0, TrainingRngStreams.RECON)
        batch_k = config.reward.disc_batch_size_k
        assert torch.equal(after, scenario.replay_recon_stream(
            before,
            strokes=2 * (batch_k // 2),  # 每槽 K/2 任务 × 2 iteration
            latent_shape=tuple(config.latent_shape),
            step_count=len(sorted(config.policy.train_step_indices_m)),
        )), (
            "recon 流全序重放失配——任务粒度消耗序非「先 s 后 ε」或"
            "消耗量非每槽恰 K/2 × iteration 数"
        )

    def test_reconstruction_uses_rollout_phase_weights(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """drift #1 机制锚：窗口任务 fake = rollout 相当前权重 θ_t
        （#217 §3 时点前移）。同 seed 同 config 下，run 内 iter0 的
        任务产出与「未 run 的等价装配（θ_0 权重 + 槽 0 recon 流初始位
        + 窗口起点抽取的同条目 real）」逐位一致——若重构错误地落在
        policy 更新后（θ_1 权重），逐位比较必然分道。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=6)
        config.schedule.max_iterations = 1
        calls = scenario.spy_window_pairs(monkeypatch)
        executor = scenario.build(config, coroutines=1, owner_check=True)
        executor.run()
        assert calls, "iter0（窗口(0)）须摊派 K 个任务"
        real_run, _, pair_run = calls[0]
        # θ_0 权重 + 同源流位 + 窗口抽取条目 的等价装配（同 config 同
        # seed：网络经 checkpoint 装载逐位等价、槽 0 流 = 注册表同派生
        # 公式的初始位、WindowRealDraw 同 (seed, 窗口号, 条件) 派生）
        device = torch.device("cpu")
        policy = GroupPolicy.build(
            config, torch.Generator().manual_seed(config.schedule.seed), device,
        )
        registry = SlotRngRegistry(config.schedule.seed, slots=1)
        manual_assembler = TrainingRuntime.assemble_pair_assembler(
            config,
            real_sampler=None,
            sampler=TrainingRuntime.assemble_sampler(
                config, policy.field, device=device,
            ),
            conditions=policy.conditions,
            generator=registry.get_stream(0, TrainingRngStreams.RECON),
            amp=TrainingRuntime.amp_context(config, device),
        )
        manual_pair = manual_assembler.reconstruct_assigned(
            real_run.to(device), calls[0][1],
        )
        scenario.assert_state_dicts_bitwise(
            {"reals": pair_run.reals, "fakes": pair_run.fakes},
            {"reals": manual_pair.reals, "fakes": manual_pair.fakes},
            "rollout 相窗口任务 vs θ_0 等价装配",
        )

    def test_real_draw_entries_are_window_scoped(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """real 侧窗口抽取（#220 决议 10）：任务 real 来自窗口起点全局
        无放回抽取——同条件任务条目互异，且与 WindowRealDraw.assign
        纯函数重导逐位一致（不经注册表流）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=8)
        config.schedule.max_iterations = 1
        calls = scenario.spy_window_pairs(monkeypatch)
        executor = scenario.build(config, coroutines=2, owner_check=True)
        executor.run()
        real_pool = LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )
        draw = WindowRealDraw(real_pool)
        assigned = draw.assign(
            config.schedule.seed, 0, executor.window,
        )
        fingerprints = [tuple(reals.flatten().tolist()) for reals, _, _ in calls]
        assert len(set(fingerprints)) == len(fingerprints), (
            "同窗口 real 条目必须全局无放回（条条不同）"
        )
        # 任务槽 0 的首任务 real 应等于抽取分配中 (0, 0, 0, j) 任务的条目
        first_task = next(
            task for task in sorted(
                assigned,
                key=lambda t: (t.condition, t.card, t.iteration, t.index),
            )
            if task.slot == 0
        )
        entry = assigned[first_task]
        expected = real_pool.load_latent(entry).unsqueeze(0)
        slot0_call = next(
            reals for reals, _, _ in calls
            if torch.equal(reals, expected)
        )
        assert slot0_call is not None

    def test_empty_card_topology_rejected(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """空卡拓扑（槽数 < 卡数）装配期拒绝：「每窗口每卡恰 K 对」的
        窗口语义前提每卡至少一槽——半绑定拓扑不静默可用。"""
        scenario = ExecutorScenario(cli, tmp_path)
        config = scenario.prepare(seed=0)
        with pytest.raises(ValueError, match="空卡"):
            scenario.build(
                config,
                coroutines=1,
                devices=[torch.device("cpu"), torch.device("cpu")],
            )


@pytest.mark.gpu
@pytest.mark.slow
class TestExecutorMultiCardTier:
    """gauss 多卡 e2e 档（slow+gpu；换发射方式重建，#217 门面 + M1）：
    跨卡一致性 + 层 3 同 seed 多卡重放锚薄切片形态。"""

    @staticmethod
    def _visible_card_count() -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    @staticmethod
    def _multicard_config(scenario: ExecutorScenario, **kwargs):
        """多卡档 config：K=2（#234 窗口容量守卫 = 逐条件池 ≥ K×卡数——
        fixture BraTS 池每序列 14 条，4 卡下 K=4 需 16 触发装配期拒绝；
        K=2 需 8 ≤ 14。守卫语义本身由此档的装配通过隐式覆盖）。"""
        kwargs.setdefault("reward", {"disc_batch_size_k": 2})
        return scenario.prepare(**kwargs)

    @staticmethod
    def _devices(card_count: int) -> list:
        """真多卡设备集（slow+gpu 档本意）：显式供给 CUDA 设备——
        ``ExecutorScenario.build`` 的 fixture 默认是单 CPU 设备，不显式
        供给时多卡档实际跑成单卡多协程（跨卡断言空转恒真，#234 评审
        收敛期暴露），devices 显式化是本档断言有判别力的前提。"""
        return [torch.device(f"cuda:{i}") for i in range(card_count)]

    def test_multicard_weights_and_events_across_cards(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """单进程多卡端到端：跨卡权重与 spectral buffer 逐位一致（确定性
        allreduce + 同步 step 序列的结构性保证），事件 (iteration, slot)
        完备有序。判别器链全强度（#234）：判别器参数跨卡逐位一致进锚
        （梯度 allreduce SUM + 同步 optimizer.step 的实证）+ 每事件
        disc_update 在场（N_d=1）。"""
        card_count = self._visible_card_count()
        if card_count < 2:
            pytest.skip("多卡档：需要 ≥2 CUDA 设备（gauss 4×A6000 口径）")
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._multicard_config(scenario, seed=0)
        config.schedule.max_iterations = 2
        executor = scenario.build(
            config,
            coroutines=min(card_count, 4),
            devices=self._devices(card_count),
        )
        executor.run()
        slots = executor.allocation.slot_count
        reference = executor.cards[0].replica.policy.full_state()
        reference_disc = executor.cards[0].replica.scorer.discriminator.state_dict()
        reference_buffers = scenario.spectral_buffers(
            executor.cards[0].replica.scorer,
        )
        for card in executor.cards[1:]:
            scenario.assert_state_dicts_bitwise(
                reference, card.replica.policy.full_state(),
                f"policy(card{card.index})",
            )
            scenario.assert_state_dicts_bitwise(
                reference_disc,
                card.replica.scorer.discriminator.state_dict(),
                f"discriminator(card{card.index})",
            )
            scenario.assert_state_dicts_bitwise(
                reference_buffers, scenario.spectral_buffers(card.replica.scorer),
                f"spectral(card{card.index})",
            )
        events = [
            event for event in scenario.events()
            if event["event"] == "iter"
        ]
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (iteration, slot)
            for iteration in range(2) for slot in range(slots)
        ]
        for event in events:
            assert event["modality"] == executor.allocation.condition_for(
                event["iteration"], event["rank"],
            )
            assert event["disc_update"] is not None
            assert event["disc_update"]["global_batch_size"] == (
                config.reward.disc_batch_size_k * card_count
            )

    def test_multicard_same_seed_replay(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """层 3 薄切片锚（#218 三层锚③，gauss 多卡）：同 seed 两 run——
        结构面逐位（事件流完备序 + 流终态）+ 数值面容差对拍（卡 0
        权重/事件值；spectral buffer 在场非零、跨 run 不比较，口径
        见下）。#233 起日程升 multi-k（num_steps=5、
        M={1,2,3}）——多卡更新相逐 k 收集-同步（逐 tensor 串行
        allreduce × |M| barrier）进重放锚，验收逐 k 协议的多卡档。

        数值面为**容差对拍**（权重 rtol=1e-2/atol=5e-5；事件按
        iteration 分代——iter 0 rel=1e-2 紧锚、iter ≥ 1 rel=1e-1，
        漂移随训练步放大、判别器损失分量 2 iter 后相对漂移实测 2.2%，
        近零值 abs=1e-3）而非逐位；spectral u/v 缓冲**跨 run 不比较**——幂
        迭代在两判别器步（8 次推进）远未收敛，跨 run 方向是迭代
        basin 抽签（实测一对 |cos|=0.994、另一对 0.887），无跨 run
        契约可言（#220 决议 2 的异构 sigma 近似 accepted drift 同面；
        run **内**卡间一致由 test_multicard_weights_and_events_across_
        cards 逐位覆盖）——本测试只断言缓冲在场且非零（幂迭代在推进）。
        ——#234 评审收敛期探测定谳：流终态（draw 序列）逐位一致、初始
        权重逐位一致、线程数钉死、cudnn heuristic 钉死后，同进程两 run
        与跨进程两 run 的权重在 2 iteration 后仍有 1e-4 级相对漂移
        （实测 117/121 张量分道）；逐张量定位（spy 哈希）钉死机制——
        policy rollout 全链（anchor/directions/log-prob/fakes）跨 run
        逐位相同，**判别器打分**在同 fakes 输入下跨 run 分道：策略
        UNet 卷积走 mkldnn 重排序路径（地址不敏感），判别器 patch
        conv 落 ATen 原生路径（kernel 向量化选择对权重/输入的内存
        对齐敏感，ASLR + 分配器历史使跨 run 地址不可复现）——同输入
        不同地址 → 1e-7 级分数差 → 经奖励/损失/训练两步放大。框架级
        不可归一。容差取观测包络一个数量级上界：协议级非确定性
        （RNG 流错接、barrier 序乱）为 O(1) 仍被捕获。逐位数值重放的
        判别力由 CPU 单卡多协程档（test_same_seed_replay_bitwise_
        multi_coroutine，该几何分配模式实测稳定、多次复跑逐位一致）
        独占承担——协议确定性与设备无关，多卡档验全协议的结构性形态
        + 跨卡逐位一致性（test_multicard_weights_and_events_
        across_cards）。

        结构面仍逐位：事件 (iteration, slot) 完备序、条件 = 分配表、
        loss 键集、disc_update conditions 键集、流终态——流终态跨 run
        逐位相等本身就是 #234 RNG 消耗序（窗口摊派 + 池化恰一次读数）
        的强确定性锚：数值可随 kernel 路径漂移，消耗序不可。

        流终态在比较面（#232 spec 评审补强）：recon 流终态是重构消耗
        序的确定函数；real_pool 流本期**零消耗**（窗口 real 抽取 =
        splitmix64 一次性派生 generator、不进 RNG 注册表）——其终态
        == 初始派生快照，构成「抽取确定性」锚——重构 no_grad 不改权
        重、无事件面，重构在多卡路径被静默跳过或消耗序漂移时权重/
        事件面不敏感，逐位比较把盲区关上（rollout/heldout 流已分别被
        权重与事件 heldout_auc 间接涵盖，不重复入面）。"""
        card_count = self._visible_card_count()
        if card_count < 2:
            pytest.skip("多卡档：需要 ≥2 CUDA 设备（gauss 4×A6000 口径）")
        scenario = ExecutorScenario(cli, tmp_path)
        config = self._multicard_config(
            scenario, seed=3, num_steps=5, train_steps={1, 2, 3},
        )
        config.reward = config.reward.model_copy(
            update={"spectral_norm_enabled": True},
        )
        config.schedule.max_iterations = 2
        first = scenario.build(
            config, run_name="run1", coroutines=min(card_count, 4),
            devices=self._devices(card_count),
        )
        first.run()
        second = scenario.build(
            config, run_name="run2", coroutines=min(card_count, 4),
            devices=self._devices(card_count),
        )
        second.run()
        scenario.assert_state_dicts_close(
            first.cards[0].replica.policy.full_state(),
            second.cards[0].replica.policy.full_state(),
            "policy(card0)",
        )
        for run, executor in (("run1", first), ("run2", second)):
            buffers = scenario.spectral_buffers(
                executor.cards[0].replica.scorer,
            )
            assert buffers, "SN 启用下 _u/_v buffer 须在场（断言有判别力）"
            for key, buffer in buffers.items():
                assert buffer.norm() > 0, (
                    f"{run}:{key} 零向量——谱归一化幂迭代未推进"
                )
        events1 = scenario.events("run1")
        events2 = scenario.events("run2")
        assert len(events1) == len(events2)
        for index, (left, right) in enumerate(zip(events1, events2)):
            assert left["event"] == right["event"], (
                f"事件[{index}] 族失配：{left['event']} != {right['event']}"
            )
            # 分代容差：漂移随训练步放大——iter 0 数值面留紧锚，
            # iter ≥ 1 放宽防 kernel 漂移误报（机制见 docstring）。
            rel = 1e-2 if left["iteration"] == 0 else 1e-1
            for key, value in left.items():
                if key in WALL_CLOCK_EVENT_FIELDS:
                    continue  # 墙钟字段跨 run 必然不同
                scenario.assert_replay_values_close(
                    value, right[key], f"事件[{index}].{key}", rel=rel,
                )
        for run in ("run1", "run2"):
            for event in scenario.events(run):
                if event["event"] != "iter":
                    continue  # overfit_alert 等其它族事件无 loss 面
                policy_keys = {
                    "policy_step_1", "policy_step_2", "policy_step_3",
                }
                assert policy_keys <= set(event["loss"]), (
                    "M={1,2,3} 的每个 k 都须真实发生累积与 step（多卡档）"
                )
                # 判别器链全强度进重放锚（#234）：判别器步 loss 与
                # per-condition 明细每事件在场
                assert "discriminator" in event["loss"]
                assert event["disc_update"] is not None
        for slot in range(first.allocation.slot_count):
            # recon 流任务粒度终态（跨 run 一致 = 窗口任务消耗序确定）
            assert torch.equal(
                first.rng.stream_state(slot, TrainingRngStreams.RECON),
                second.rng.stream_state(slot, TrainingRngStreams.RECON),
            ), f"槽 {slot} recon 流终态失配：窗口任务消耗序在多卡路径漂移"
            # real_pool 注册表流零消耗（#234 窗口抽取不走命名流）：
            # 终态 == 初始派生（seed + slot 步长 + 1）
            expected_pool = torch.Generator().manual_seed(
                3 + slot * SLOT_SEED_STRIDE + 1,
            ).get_state()
            for executor in (first, second):
                assert torch.equal(
                    executor.rng.stream_state(
                        slot, TrainingRngStreams.REAL_POOL,
                    ),
                    expected_pool,
                ), f"槽 {slot} real_pool 流应零消耗（判别器链期退役）"
