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
4. 多 rank 续训 roundtrip：中断恢复后与一步到位 run 一致——权重逐位
   （RankResumeShards）、事件轨迹在跨路径容差内（独立进程世界的
   观测前向存在 1-2 ulp 重算噪声）；
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
from queue import Empty as _QueueEmpty

import pytest
import torch

from cynosure.cli import CynosureCli
from cynosure.reward.artifacts import LatentManifest
from cynosure.train import RunArtifacts
from cynosure.distributed import DistributedContext, RankSlicedPool
from tests.test_train_loop import TrainingLoopScenario

_WORKER_JOIN_TIMEOUT_S = 600.0
"""单次 spawn train 的 worker join 上限（秒）：worker 死锁时测试显式
失败而非无限挂起。"""

_EQUIVALENCE_RTOL = 1e-5
"""跨路径等价性检查的数值容差：分布式路径（FSDP + 梯度检查点重算）
与单进程路径（直接前向）在 fp32 尾数层存在求和顺序噪声（实测 ~1e-8，
远低于 bf16 autocast 训练的量化信号）；语义等价以相对容差断言。逐位
承重轴：各 rank 权重同步（RankResumeShards）、同进程续训 roundtrip
（RunTrajectory）；跨进程世界对的事件浮点面（含续训 roundtrip 的
两世界对比）走本容差——独立进程实例的打分前向偶发 1-2 ulp 分叉。"""


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
        argv: list[str] | None = None, join_timeout_s: float | None = None,
    ) -> None:
        self.config_path = config_path
        self.run_dir = run_dir
        self.world = world
        self.argv = argv if argv is not None else [
            "train", "--config", str(config_path), "--run-dir", str(run_dir),
        ]
        self.join_timeout_s = (
            join_timeout_s
            if join_timeout_s is not None else _WORKER_JOIN_TIMEOUT_S
        )
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
            # get 的超时是 join 兜底的前置（worker 崩溃未回传时主进程不能
            # 无限等）：超时把已收集的 stderr 带进失败信息，死锁可诊断
            try:
                payload = queue.get(timeout=self.join_timeout_s)
            except _QueueEmpty:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5.0)
                stuck = ", ".join(
                    f"rank {rank}: {collected[rank]['stderr']}"
                    for rank in sorted(collected)
                ) or "（无任何 worker 回传）"
                raise AssertionError(
                    f"worker 回传超时（{self.join_timeout_s}s，"
                    f"疑似某 rank 在集合操作互等后崩溃）: {stuck}"
                )
            collected[payload["rank"]] = payload
        for process in processes:
            process.join(timeout=self.join_timeout_s)
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
    """跨路径等价性判定：结构字段严格一致、浮点字段在重算路径噪声容差
    内一致；wall-clock elapsed_s 不参与对比。适用面 = 任何两个独立执行
    语境的重算对比（分布式 vs 单进程、跨进程世界对——含多 rank 续训
    roundtrip 的两世界事件对比）。逐位断言保留给同进程重放（单进程
    续训 roundtrip，RunTrajectory）与各 rank 权重同步（RankResumeShards）
    ——后者的输入是落盘状态本身，不含重算面。"""

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

    def test_broadcast_flag_and_local_device_degenerate_on_single_process(self) -> None:
        """world-1 恒等：broadcast_flag 原样返回传入值（不构造张量）、
        local_device 回落 CPU。CUDA 下的 cuda:LOCAL_RANK 绑定与 NCCL 的
        广播张量设备正确性属集群 torchrun 冒烟门槛（本机 CPU fixture 只
        锁退化语义与代码路径收敛）。"""
        context = DistributedContext.bootstrap()
        assert context.broadcast_flag(True) is True
        assert context.broadcast_flag(False) is False
        assert context.local_device().type == (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        context.destroy()

    def test_pg_timeout_env_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """watchdog 超时环境变量：未设置 = None（不传参、torch 默认）、
        正整数 = timedelta 分钟、非法值显式拒绝（部署侧拼写错误静默回落
        默认会让 SothisAI 平台的长 watchdog 预期失效）。"""
        from datetime import timedelta

        monkeypatch.delenv("CYNOSURE_PG_TIMEOUT_MIN", raising=False)
        assert DistributedContext._pg_timeout() is None
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "40")
        assert DistributedContext._pg_timeout() == timedelta(minutes=40)
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "0")
        with pytest.raises(ValueError, match="CYNOSURE_PG_TIMEOUT_MIN"):
            DistributedContext._pg_timeout()
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "abc")
        with pytest.raises(ValueError):
            DistributedContext._pg_timeout()


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
                # spacing 侧车（issue #46 契约必填）：BraTS 1mm iso 的
                # header zooms ×1e2（切片语义不消费取值，仅须通过装载校验）
                "spacing": [100.0, 100.0, 100.0],
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

    def test_insufficient_pool_rejected_consistently_on_every_rank(
        self, manifest: LatentManifest,
    ) -> None:
        """pool 不足的拒绝必须全 rank 一致：校验消费**切片前**的 full
        manifest（每 rank 对同一全量判定同一结果）。按切片后本地视图校验
        时 rank 间可见性不同（序列仅 1 条时 rank 0 满额通过、高 rank 条带
        为空才拒绝）——失败方单方面退出装配、其余 rank 进入集合操作互等
        （连接错误/挂死），而非全 rank 一致的输入拒绝。"""
        starved = manifest.model_copy(deep=True)
        starved.entries = starved.entries[:4]  # 每序列恰好 1 条
        for rank in range(2):
            with pytest.raises(ValueError, match="不足"):
                RankSlicedPool(starved, DistributedContext(rank, 2, True)).view()


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

    def test_two_rank_milestone_eval_and_stop_broadcast(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """里程碑评测相在 world=2 下全 rank 集合参与（policy 采样前向是
        FSDP 集合操作）、早停 verdict 经广播原语同步（``milestone`` 事件
        rank 0 独写入流）——早停广播的张量设备语义在多 rank 进程组下的
        集合路径覆盖（NCCL 多卡的设备正确性由集群 torchrun 冒烟门槛
        验证，本机 CPU gloo 等价进程语义）。"""
        scenario.write_inputs()
        scenario.patch_config(schedule={
            "max_iterations": 2,
            "milestone_interval": 1,
            "checkpoint_interval": 2,
        })
        SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().assert_green()

        events = RunArtifacts(
            RunArtifacts.layout(scenario.run_dir),
        ).read_events()
        milestones = [
            event for event in events if event.get("event") == "milestone"
        ]
        assert [event["iteration"] for event in milestones] == [1, 2]
        assert all(not event["early_stop"] for event in milestones)

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


class TestResumeGeneration:
    """续训代际一致性：per-rank 分片各自原子替换（tmp + os.replace），
    保存中途崩溃可留下**混代际**分片（部分 rank 已到 N、其余还在 N-1）
    ——各 rank 只对账自己分片的恢复会从不同 iteration 继续训练：集合
    操作与邻居错配、指标流出现重复事件、allreduce 混入不同逻辑迭代的
    梯度（权重静默分叉）。恢复入口必须对齐共同代际：全 rank 分片均
    持久化到同一 iteration 后才发布代际标记（提交点），恢复对账标记、
    混代际现场显式拒绝。"""

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def test_generation_marker_published_with_shards(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().assert_green()

        marker = json.loads(
            (scenario.run_dir / "checkpoints" / "resume_generation.json")
            .read_text(encoding="utf-8"),
        )
        assert marker["iteration"] == 2
        shards = RankResumeShards(scenario.run_dir, world=2)
        for rank in range(2):
            assert shards.state(rank)["iteration"] == marker["iteration"]

    def test_mixed_generation_shards_are_refused(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """混代际现场（rank 1 分片落后一代 = 其保存未及完成的崩溃现场）
        的续训显式拒绝：分叉续跑不是恢复，是静默损坏。"""
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().assert_green()

        shard = scenario.run_dir / "checkpoints" / "resume_state_rank1.pt"
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        payload["iteration"] -= 1  # 回拨一代：混代际注入
        torch.save(payload, shard)

        scenario.patch_config(schedule={"max_iterations": 4})
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
            argv=[
                "train", "--config", str(scenario.config_path),
                "--run-dir", str(scenario.run_dir), "--resume",
            ],
        ).launch()
        assert result.codes == [2, 2], result.errors
        assert any("代际" in error for error in result.errors)


class TestResumeShardFailureRefusal:
    """单 rank 分片装载/校验失败的集体拒绝：restore 的本地前置（代际
    标记读取、分片装载、payload 契约/拓扑校验）在**任一 rank** 失败时，
    拒绝必须作为报告数据进对账 collective、全体一致返回输入契约错误
    ——本地先抛会让通过校验的邻居停在 all_gather 永等（拒绝方退出、
    通过方挂死），那是作业假死而非干净的输入拒绝。

    红/绿信号用短 join 超时（挂死 → 回传超时红；一致退出 → 秒级绿）。"""

    _JOIN_TIMEOUT_S = 120.0

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def test_single_rank_topology_mismatch_refused_on_every_rank(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(
            schedule={"max_iterations": 2, "checkpoint_interval": 2},
        )
        SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
        ).launch().assert_green()

        shard = scenario.run_dir / "checkpoints" / "resume_state_rank1.pt"
        payload = torch.load(shard, map_location="cpu", weights_only=True)
        payload["world_size"] = 1  # 单 rank 拓扑篡改：仅 rank 1 分片自称 world-1
        torch.save(payload, shard)

        scenario.patch_config(schedule={"max_iterations": 4})
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
            argv=[
                "train", "--config", str(scenario.config_path),
                "--run-dir", str(scenario.run_dir), "--resume",
            ],
            join_timeout_s=self._JOIN_TIMEOUT_S,
        ).launch()
        assert result.codes == [2, 2], result.errors
        assert any("world_size" in error for error in result.errors)


class TestSpawnedUsageContract:
    """分布式启动的 usage 错误契约：rank 非对称的 pre-flight 失败（run
    目录预存、轨迹诊断输入错误）必须经广播裁决让**全 rank 一致**返回
    usage error（exit 2）——rank 0 单方面退出、其余 rank 进 rendezvous
    的作业是挂死/被 launcher 噪声终止，不是干净的输入拒绝；训练装配
    失败的 run 目录回滚只由 rank 0 执行（多 rank 各自 rmtree 同一目录
    是 stat/rmtree 竞态）。

    红/绿信号用短 join 超时（挂死 → 超时红；一致退出 → 秒级绿）。"""

    _JOIN_TIMEOUT_S = 120.0

    @pytest.fixture
    def scenario(self, cli, tmp_path: Path) -> TrainingLoopScenario:
        return TrainingLoopScenario(cli, tmp_path)

    def test_preexisting_run_dir_rejected_on_every_rank(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.run_dir.mkdir(parents=True)
        (scenario.run_dir / "config.json").write_text("{}", encoding="utf-8")
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
            join_timeout_s=self._JOIN_TIMEOUT_S,
        ).launch()
        assert result.codes == [2, 2], result.errors
        assert any("已存在" in error for error in result.errors)

    def test_dump_trajectory_failure_reaches_every_rank(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        scenario.patch_config(artifacts={
            "net_config_json": str(scenario.fixture_dir / "missing.json"),
        })
        result = SpawnedTrainWorld(
            scenario.config_path, scenario.run_dir, world=2,
            argv=[
                "train", "--config", str(scenario.config_path),
                "--run-dir", str(scenario.run_dir), "--dump-trajectory",
            ],
            join_timeout_s=self._JOIN_TIMEOUT_S,
        ).launch()
        assert result.codes == [2, 2], result.errors
        assert any("轨迹诊断" in error for error in result.errors)

    def test_construction_failure_rollback_is_rank0_only(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        # 损坏判别器 checkpoint → 装配期对 torch.load 的对称 RuntimeError
        (scenario.fixture_dir / "discriminator.pt").write_bytes(b"corrupt")
        for _ in range(3):  # 竞态类：重复三次提高捕获率
            result = SpawnedTrainWorld(
                scenario.config_path, scenario.run_dir, world=2,
                join_timeout_s=self._JOIN_TIMEOUT_S,
            ).launch()
            assert result.codes == [2, 2], result.errors
            assert not any(
                "FileNotFoundError" in error for error in result.errors
            )
            assert not scenario.run_dir.exists()  # 未产出工件的目录已回滚


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
        # 事件浮点面走跨路径容差（权重逐位对账在下方分片断言承重）：
        # 两个独立进程世界的打分前向对相同输入偶发 float32 1-2 ulp 分叉
        # （macOS/CPU 实测仅 anchor_eval_reward，rel ~2e-7，且可出现在
        # resume 未触及的段落——训练态全组件逐位一致、噪声纯观测不进
        # 梯度），逐位断言在此本质脆弱而非恢复逻辑缺陷。
        _CROSS_PATH.trajectories(resumed_events, baseline_events)

        resumed_shards = RankResumeShards(scenario.run_dir, world=2)
        baseline_shards = RankResumeShards(baseline_dir, world=2)
        resumed_shards.assert_matches(
            baseline_shards,
            keys=["policy_network", "discriminator_network"],
        )
        for rank in range(2):
            assert resumed_shards.state(rank)["iteration"] == 4
            assert baseline_shards.state(rank)["iteration"] == 4
