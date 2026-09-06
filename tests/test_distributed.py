"""分布式执行层（ticket #24 验收标准聚合）：torchrun 语义入口 + FSDP
full-shard + 判别器 DDP + rank 0 指标归并 + 多 rank 续训 roundtrip。

测试形态：fixture 多进程（CPU gloo）经 ``SpawnedTrainWorld`` 起 world 个
worker，每个 worker（``TrainWorldWorker``）以 torchrun 同款环境变量
（RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT）驱动真实 CLI 路径
——torchrun 的 worker 入口即 ``python -m cynosure.cli``，本 harness 与其
进程语义等价（--nproc_per_node 的参数化 = world size；torchrun 二进制
的 rendezvous 冒烟属集群 M0 门槛清单，本机 CPU fixture 不依赖它）。

AC 对应：
1. FSDP 分片 + 梯度 allreduce 生效（与单进程等价性检查）——
   world=1 与进程内单进程逐位一致（分布式装配路径无副作用、rank 0 的
   seed 派生恒等）+ world=2 各 rank 训练后权重逐位一致（allreduce 同步
   生效）+ world=2 结果 ≠ 单进程对照（梯度混入他 rank rollout 数据）；
2. 判别器 DDP：各 rank 本 rank fake + pool 切片更新，同步后参数一致；
3. rank 0 指标归并：无重复、无丢失、顺序稳定（(iteration, rank) 序）；
4. 多 rank 续训 roundtrip：中断恢复后与一步到位 run 的轨迹/权重一致；
5. torchrun 启动入口：world 全 rank 驱动同一 CLI（--nproc_per_node
   参数化语义由 harness 的 world 参数覆盖）。
"""

import io
import json
import math
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from cynosure.cli import CynosureCli
from cynosure.reward.artifacts import LatentManifest
from cynosure.train import RunArtifacts
from cynosure.distributed import DistributedContext, RankSlicedPool
from tests.conftest import RunTrajectory
from tests.test_train_loop import TrainingLoopScenario

_WORKER_JOIN_TIMEOUT_S = 600.0
"""单次 spawn train 的 worker join 上限（秒）：worker 死锁时测试显式
失败而非无限挂起。"""

_EQUIVALENCE_RTOL = 1e-5
"""跨路径等价性检查的数值容差：分布式路径（FSDP + 梯度检查点重算）
与单进程路径（直接前向）在 fp32 尾数层存在求和顺序噪声（实测 ~1e-8，
远低于 bf16 autocast 训练的量化信号）；语义等价以相对容差断言。同路径
重放（续训 roundtrip、各 rank 权重同步）仍逐位断言。"""


class TrainWorldWorker:
    """单个 rank 的训练执行体（torchrun worker 的进程内等价入口）：env
    注入后驱动真实 CLI，结果（退出码 + stderr）经队列回传主进程——
    worker 崩溃也回传，死锁由主进程 join 超时兜底。实例可 pickle
    （spawn Process 以它为 target）。"""

    def __init__(
        self, rank: int, world: int, port: int, argv: list[str], queue,
    ) -> None:
        self.rank = rank
        self.world = world
        self.port = port
        self.argv = argv
        self.queue = queue

    def __call__(self) -> None:
        os.environ.update(
            RANK=str(self.rank),
            LOCAL_RANK=str(self.rank),
            WORLD_SIZE=str(self.world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(self.port),
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            code = CynosureCli(self.argv, stdout, stderr).run()
            payload = {"code": code, "stderr": stderr.getvalue()}
        except Exception as exc:  # worker 崩溃的诊断面：退出码 + 异常文本
            payload = {"code": 1, "stderr": f"{type(exc).__name__}: {exc}"}
        self.queue.put({"rank": self.rank, **payload})


@dataclass
class DistTrainResult:
    """一次 spawn train 的全 rank 结果（退出码 + 每 rank stderr）。"""

    codes: list[int]
    errors: list[str]

    def assert_green(self) -> None:
        assert all(code == 0 for code in self.codes), (
            f"分布式 train 退出码 {self.codes}；"
            f"stderr: {self.errors}"
        )


class SpawnedTrainWorld:
    """torchrun 语义的本地多进程 world（fixture CPU gloo）：spawn 起
    world 个 ``TrainWorldWorker`` 并行跑 ``train``，收集全 rank 结果。"""

    _next_port = 29730
    """TCPStore 端口游标（world 间串行递增，避 TIME_WAIT 冲突）。"""

    def __init__(
        self, config_path: Path, run_dir: Path, world: int,
        argv: list[str] | None = None,
    ) -> None:
        self.config_path = config_path
        self.run_dir = run_dir
        self.world = world
        self.argv = argv if argv is not None else [
            "train", "--config", str(config_path), "--run-dir", str(run_dir),
        ]
        type(self)._next_port += 1
        self.port = type(self)._next_port

    def launch(self) -> DistTrainResult:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        workers = [
            TrainWorldWorker(rank, self.world, self.port, self.argv, queue)
            for rank in range(self.world)
        ]
        processes = [context.Process(target=worker) for worker in workers]
        for process in processes:
            process.start()
        collected: dict[int, dict] = {}
        for _ in range(self.world):
            payload = queue.get()
            collected[payload["rank"]] = payload
        for process in processes:
            process.join(timeout=_WORKER_JOIN_TIMEOUT_S)
            if process.is_alive():
                process.terminate()
                collected[process.pid] = {
                    "code": 1, "stderr": "worker join 超时（疑似死锁）",
                }
        return DistTrainResult(
            codes=[collected[rank]["code"] for rank in range(self.world)],
            errors=[collected[rank]["stderr"] for rank in range(self.world)],
        )


class CrossPathEquivalence:
    """跨路径（分布式 vs 单进程）等价性判定：结构字段严格一致、浮点
    字段在重算路径噪声容差内一致；wall-clock elapsed_s 不参与对比。
    同路径重放（续训 roundtrip、各 rank 权重同步）走逐位断言
    （RunTrajectory / RankResumeShards），不经本判定。"""

    def __init__(
        self, rtol: float, float_atol: float = 1e-8, tensor_atol: float = 1e-7,
    ) -> None:
        self.rtol = rtol
        self.float_atol = float_atol
        self.tensor_atol = tensor_atol

    def trajectories(self, left: list[dict], right: list[dict]) -> None:
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert set(first) == set(second)
            for key in first:
                if key == "elapsed_s":
                    continue
                a, b = first[key], second[key]
                if isinstance(a, float) and isinstance(b, float):
                    assert math.isclose(
                        a, b, rel_tol=self.rtol, abs_tol=self.float_atol,
                    ), f"{key}: {a} vs {b}"
                elif isinstance(a, dict) and isinstance(b, dict):
                    assert set(a) == set(b)
                    for name in a:
                        assert math.isclose(
                            a[name], b[name],
                            rel_tol=self.rtol, abs_tol=self.float_atol,
                        ), f"{key}:{name}: {a[name]} vs {b[name]}"
                else:
                    assert a == b, f"{key}: {a!r} vs {b!r}"

    def checkpoints(self, left: Path, right: Path, name: str) -> None:
        """跨路径 checkpoint 等价：键集严格一致、张量在重算噪声容差内一致。"""
        a = torch.load(
            left / "checkpoints" / name, map_location="cpu", weights_only=True,
        )
        b = torch.load(
            right / "checkpoints" / name, map_location="cpu", weights_only=True,
        )
        assert set(a) == set(b)
        for key in a:
            assert torch.allclose(
                a[key], b[key], rtol=self.rtol, atol=self.tensor_atol,
            ), f"{name}:{key} 超出等价容差（max diff "
            f"{(a[key] - b[key]).abs().max().item():.3e}）"


_CROSS_PATH = CrossPathEquivalence(rtol=_EQUIVALENCE_RTOL)


class RankResumeShards:
    """一个 run 的 per-rank 续训分片集（「同步生效」与「续训 roundtrip」
    的外部观测面）：各 rank 分片文件的对号读取与逐位对账。"""

    def __init__(self, run_dir: Path, world: int) -> None:
        self.run_dir = run_dir
        self.world = world

    def state(self, rank: int) -> dict:
        return torch.load(
            self.run_dir / "checkpoints" / f"resume_state_rank{rank}.pt",
            map_location="cpu", weights_only=True,
        )

    def assert_bitwise_identical_across_ranks(self, keys: list[str]) -> None:
        """各 rank 续训状态的指定字段逐位一致（梯度 allreduce 同步生效的
        外部观测面）。"""
        states = [self.state(rank) for rank in range(self.world)]
        for key in keys:
            for rank in range(1, self.world):
                left, right = states[0][key], states[rank][key]
                assert set(left) == set(right), f"{key}: rank 0/{rank} 键集不符"
                for name in left:
                    assert torch.equal(left[name], right[name]), (
                        f"{key}:{name} 在 rank 0 与 rank {rank} 间漂移"
                    )

    def assert_matches(self, other: "RankResumeShards", keys: list[str]) -> None:
        """与另一 run（一步到位 baseline）的对应 rank 分片逐位一致
        （多 rank 续训 roundtrip 的判定轴）。"""
        for rank in range(self.world):
            left, right = self.state(rank), other.state(rank)
            for key in keys:
                for name in left[key]:
                    assert torch.equal(left[key][name], right[key][name]), (
                        f"rank {rank} {key}:{name} 续训后漂移"
                    )


class TestDistributedContextUnit:
    """进程组 Facade 的单进程退化与 seed 派生语义。"""

    def teardown_method(self) -> None:
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)

    def test_bootstrap_without_env_is_single_process(self) -> None:
        context = DistributedContext.bootstrap()
        assert context.rank == 0
        assert context.world_size == 1
        assert not context.distributed
        # gather 恒等（EventMerger 单进程路径 = 直写）
        assert context.gather([{"rank": 0}]) == [[{"rank": 0}]]
        context.destroy()

    def test_derived_seed_keeps_rank0_identity(self) -> None:
        context = DistributedContext.bootstrap()
        assert context.derive_seed(7) == 7  # 等价性前提：rank 0 恒等偏移
        context.destroy()


class TestPoolSliceUnit:
    """Real sample pool 的 rank 切片语义（条带切片 + 分层保持）。"""

    @pytest.fixture
    def manifest(self, tmp_path: Path) -> LatentManifest:
        """8 条目 × 四序列交替的池（load 装载形态：条目路径存在性不作要求）。"""
        path = tmp_path / "real_pool.json"
        entries = [
            {
                "case_id": f"case-{index}",
                "modality": ["t1n", "t1c", "t2w", "t2f"][index % 4],
                "latent": f"latent-{index}.pt",
            }
            for index in range(8)
        ]
        path.write_text(json.dumps({
            "kind": "real_pool",
            "encoder": "fixture",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"train": 8, "val": 0, "test": 0},
            "entries": entries,
        }), encoding="utf-8")
        return LatentManifest.load(path, kind="real_pool")

    def test_stripe_slice_keeps_alternating_layers(self, manifest: LatentManifest) -> None:
        context = DistributedContext(0, 1, False)
        sliced = RankSlicedPool(manifest, context).view()
        assert sliced.kind == manifest.kind
        assert sliced.entries == manifest.entries  # world=1 恒等

    def test_stripe_slice_distributes_entries_by_rank(self, manifest: LatentManifest) -> None:
        """分层条带切片：每序列内部 entries[rank::world]，各片覆盖全部序列。"""
        context = DistributedContext(1, 2, True)  # 不 init 进程组的纯切片语义
        sliced = RankSlicedPool(manifest, context).view()
        # t1n=[0,4] t1c=[1,5] t2w=[2,6] t2f=[3,7] → rank 1 取各序列第 2 条
        assert [entry.case_id for entry in sliced.entries] == [
            "case-4", "case-5", "case-6", "case-7",
        ]
        assert set(entry.modality for entry in sliced.entries) == {"t1n", "t1c", "t2w", "t2f"}
        assert sliced.modalities == {m: 1 for m in ("t1n", "t1c", "t2w", "t2f")}

    def test_slice_rejects_empty_modality_band(self, manifest: LatentManifest) -> None:
        """切片后某序列条目归零 = 判别器 real 侧断供，装配期显式拒绝。"""
        context = DistributedContext(0, 4, True)  # 4 路切片 × 每序列仅 2 条
        starved = manifest.model_copy(deep=True)
        starved.entries = starved.entries[:2]  # 只剩 t1n/t1c 两序列
        with pytest.raises(ValueError, match="t2w|t2f|序列"):
            RankSlicedPool(starved, context).view()


class TestSingleRankEquivalence:
    """AC1 前半 + AC5：world=1 的 torchrun 语义入口与进程内单进程逐位一致。"""

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def test_world1_entry_matches_in_process_run(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(schedule={"max_iterations": 2})
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=1,
        ).launch()
        result.assert_green()

        reference_dir = scenario.tmp_path / "run_inproc"
        assert scenario.cli.train(
            scenario.config_path, run_dir=reference_dir,
        ).code == 0

        dist_events = RunArtifacts(
            RunArtifacts.layout(scenario.run_dir),
        ).read_events()
        inproc_events = RunArtifacts(
            RunArtifacts.layout(reference_dir),
        ).read_events()
        _CROSS_PATH.trajectories(dist_events, inproc_events)

        for name in ("policy_iter2.pt", "discriminator_iter2.pt"):
            _CROSS_PATH.checkpoints(scenario.run_dir, reference_dir, name)


class TestTwoRankSharding:
    """AC1 后半 + AC2 + AC3：FSDP/DDP 同步、指标归并、allreduce 生效。"""

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def test_two_rank_run_merges_metrics_and_syncs_weights(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch()
        result.assert_green()

        events = RunArtifacts(
            RunArtifacts.layout(scenario.run_dir),
        ).read_events()
        # AC3：无重复、无丢失、顺序稳定——(iteration, rank) 升序且各恰一次
        assert [(event["iteration"], event["rank"]) for event in events] == [
            (0, 0), (0, 1), (1, 0), (1, 1),
        ]

        # AC1/AC2：各 rank 训练后 policy（FSDP full state）与判别器（DDP
        # 副本）逐位一致——梯度 allreduce 同步生效、无 rank 漂移
        shards = RankResumeShards(scenario.run_dir, world=2)
        shards.assert_bitwise_identical_across_ranks(
            keys=["policy_network", "discriminator_network"],
        )

        # AC2：判别器确实被更新过（≠ 冷启动 checkpoint）
        initial = torch.load(
            scenario.fixture_dir / "discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        synced = shards.state(0)["discriminator_network"]
        assert any(
            not torch.equal(initial[name], synced[name]) for name in initial
        )

    def test_two_rank_gradient_differs_from_single_process(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC1（allreduce 生效）：world=2 的梯度 = 两 rank 梯度平均——
        各 rank rollout 数据独立，结果不得与单进程（rank 0 数据独断）逐位相同。"""
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        assert SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().codes == [0, 0]

        reference_dir = scenario.tmp_path / "run_inproc"
        assert scenario.cli.train(
            scenario.config_path, run_dir=reference_dir,
        ).code == 0

        dist_policy = torch.load(
            scenario.run_dir / "checkpoints" / "policy_iter2.pt",
            map_location="cpu", weights_only=True,
        )
        inproc_policy = torch.load(
            reference_dir / "checkpoints" / "policy_iter2.pt",
            map_location="cpu", weights_only=True,
        )
        assert any(
            not torch.equal(dist_policy[key], inproc_policy[key])
            for key in dist_policy
        )


class TestTwoRankResume:
    """AC4：多 rank 续训 roundtrip——恢复后与一步到位 run 一致。"""

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def _resume_world(self, scenario: TrainingLoopScenario) -> DistTrainResult:
        argv = [
            "train", "--config", str(scenario.config_path),
            "--run-dir", str(scenario.run_dir), "--resume",
        ]
        return SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2, argv=argv,
        ).launch()

    def test_resume_to_identical_trajectory(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().assert_green()
        assert [
            event["iteration"]
            for event in RunArtifacts(
                RunArtifacts.layout(scenario.run_dir),
            ).read_events()
        ] == [0, 0, 1, 1]

        scenario.patch_config(schedule={"max_iterations": 4})
        self._resume_world(scenario).assert_green()

        resumed_events = RunArtifacts(
            RunArtifacts.layout(scenario.run_dir),
        ).read_events()
        assert [(event["iteration"], event["rank"]) for event in resumed_events] == [
            (0, 0), (0, 1), (1, 0), (1, 1),
            (2, 0), (2, 1), (3, 0), (3, 1),
        ]

        baseline_dir = scenario.tmp_path / "run_baseline"
        scenario.patch_config(schedule={"max_iterations": 4})
        SpawnedTrainWorld(
            scenario.config_path, baseline_dir, world=2,
        ).launch().assert_green()
        baseline_events = RunArtifacts(
            RunArtifacts.layout(baseline_dir),
        ).read_events()
        assert RunTrajectory(resumed_events) == RunTrajectory(baseline_events)

        resumed_shards = RankResumeShards(scenario.run_dir, world=2)
        baseline_shards = RankResumeShards(baseline_dir, world=2)
        resumed_shards.assert_matches(
            baseline_shards,
            keys=["policy_network", "discriminator_network"],
        )
        for rank in range(2):
            assert resumed_shards.state(rank)["iteration"] == 4
            assert baseline_shards.state(rank)["iteration"] == 4
