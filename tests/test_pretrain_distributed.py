"""预训练 torchrun 主路径测试（ADR-0016，issue #199 的 AC 聚合）。

torchrun 2 卡（gloo/CPU，scripted fixture）端到端：测量批按卷切片
（连续段切片 + σ 轮转偏移 + ε 前缀消耗 ⇒ gather 合并还原全量排列、
rank0 重算全局 recon-AUC）、rank0 gate 四态广播分发（更新/复测/确认/
终止全路径在时限内完成——挂死即集合序列错位的第一信号）、判别器经
既有装配缝自动 DDP（各 rank 更新后权重逐位一致）、事件流/报告/checkpoint
rank0 独写（最小 rank 门）。

测试形态 = train 侧先例（test_distributed.py）：spawn 多进程 world，
worker 以 torchrun 同款环境变量（RANK/LOCAL_RANK/WORLD_SIZE/
MASTER_ADDR/MASTER_PORT）驱动真实 CLI（SpawnedTrainWorld）或直驱动
PretrainDriver（PretrainWorldWorker——DDP 权重逐位一致的观测面不经
落盘，state_dict 全体 all_gather 回传对账）。整文件 gpu + slow 标记：
CPU 环境自动跳过，``--run-slow`` 全量时集群承担（仓库纪律：测试一律
上集群）；「World-1 与分布式化前逐位一致」的回归锚 = 各用例的单进程
对照 run（同共享 prepare 工件、同 config）。
"""

import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty as _QueueEmpty

import pytest
import torch

from cynosure.config import ConfigLoader, MODALITIES
from cynosure.distributed import DistributedContext
from cynosure.netbuild import NetworkAssembler
from cynosure.pretrain.artifacts import PretrainReport, PretrainRun
from cynosure.pretrain.driver import PretrainDriver
from cynosure.reward.artifacts import LatentManifest
from tests.conftest import enforce_deterministic_kernels
from tests.test_distributed import (
    DistTrainResult,
    SpawnedTrainWorld,
    _worker_port_base,
)
from tests.test_pretrain import pretrain_inputs

# 整文件 2 卡多进程 + 端到端预训练：同 test_distributed 的分派口径
# （gpu：CPU 环境自动跳过；slow：默认跳过，--run-slow 集群全量）。
pytestmark = [pytest.mark.gpu, pytest.mark.slow]

_CONFIRM_JOIN_TIMEOUT_S = 300.0
"""确认/更新路径的 worker 回传上限（秒）：绿路径 fixture 规模下数十秒
（每条件测量 = 全量卷重构，2 卡分摊），超时即四态集合序列错位的挂死
形态（train 侧慢档先例的同一语义，取值按 fixture 预训练时长留余量）。"""

_USAGE_JOIN_TIMEOUT_S = 120.0
"""usage 错误路径（预存目录全 rank 一致拒绝）的短超时：一致退出是秒级
绿，任一读不到回传即 rank 非对称挂死的红。"""


def _heldout_volumes(config_path: Path) -> dict[str, int]:
    """共享 prepare 工件的 held-out 每条件卷数（测量/报告的全局口径
    断言基准）：分布式切片后每 rank 只本地驻留 1/N 卷——事件与报告的
    measurement/condition_volumes 必须仍留痕**全量**卷数（ADR-0016
    的成本读数全局口径），否则即为切片卷数（断言即红）。"""
    config = ConfigLoader.load(config_path)
    manifest = LatentManifest.load(
        config.reward.heldout_real_manifest, kind="heldout_real",
    )
    return {
        modality: manifest.modalities[modality] for modality in MODALITIES
    }


class DistPretrainScenario:
    """2 卡预训练用例的场景原料：共享 prepare 工件 + 每用例 config 与
    run 目录（报告路径指向 run 目录——``--run-dir`` 与声明一致的显式
    分布式启动面）。"""

    def __init__(self, inputs: Path, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.tmp_path.mkdir(parents=True, exist_ok=True)
        self.run_dir = tmp_path / "pretrain_run"
        self.config_dict = json.loads(
            (inputs / "config.json").read_text(encoding="utf-8"),
        )
        self.config_dict["reward"]["pretrain_report_json"] = str(
            self.run_dir / "pretrain_report.json",
        )
        self.config_path = tmp_path / "config.json"

    def write_config(self, **reward) -> Path:
        self.config_dict["reward"].update(reward)
        self.config_path.write_text(
            json.dumps(self.config_dict), encoding="utf-8",
        )
        return self.config_path

    def report(self) -> PretrainReport:
        return PretrainReport.load(self.run_dir / "pretrain_report.json")

    def events(self) -> list[dict]:
        return PretrainRun(PretrainRun.layout(self.run_dir)).read_events()


class PretrainWorldWorker:
    """单 rank 的预训练执行体（torchrun worker 的进程内等价入口，
    直驱动 PretrainDriver）：run 目录 init 复刻 CLI 的 rank0 + 广播裁决
    协议 → 端到端 run() → 判别器可装载 state 全体 ``all_gather`` 互见
    → 回传主进程（DDP 各 rank 权重逐位一致的观测面不经落盘：pretrain
    无续训分片，落盘只有 rank0 一份；逐键 ``torch.equal`` 对账在主进程
    完成，见 ``TestTwoRankDenseSteps``）。实例可 pickle（spawn Process
    以它为 target）。"""

    def __init__(
        self,
        rank: int,
        world: int,
        port: int,
        config_path: Path,
        run_dir: Path,
        queue,
        num_threads: int,
    ) -> None:
        self.rank = rank
        self.world = world
        self.port = port
        self.config_path = config_path
        self.run_dir = run_dir
        self.queue = queue
        self.num_threads = num_threads

    def __call__(self) -> None:
        torch.set_num_threads(self.num_threads)
        # spawn 子进程是独立进程：确定性 kernel 口径显式收口（跨 rank
        # 权重逐位对账依赖它，同 test_distributed.TrainWorldWorker）
        enforce_deterministic_kernels()
        os.environ.update(
            RANK=str(self.rank),
            LOCAL_RANK=str(self.rank),
            WORLD_SIZE=str(self.world),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(self.port),
        )
        try:
            config = ConfigLoader.load(self.config_path)
            dist = DistributedContext.bootstrap()
            try:
                run: PretrainRun | None = None
                if dist.rank == 0:
                    try:
                        run = PretrainRun.init(config, self.run_dir)
                    except FileExistsError:
                        pass  # 裁决经广播：非 0 rank 等信号而非轮询文件系统
                if not dist.broadcast_flag(run is not None):
                    self.queue.put({
                        "rank": self.rank, "code": 2,
                        "stderr": "预训练 run 目录已存在（不静默覆盖）",
                        "states": None, "report": None,
                    })
                    return
                if run is None:
                    run = PretrainRun(PretrainRun.layout(self.run_dir))
                driver = PretrainDriver(config, run, dist_context=dist)
                report = driver.run()
                # DDP 同步的观测面：各 rank 解包判别器的可装载 state 全体
                # 互见（all_gather），主进程逐键 torch.equal 对账——
                # RankResumeShards 落盘逐位对账的 worker 内等价（pretrain
                # 无续训分片，不落 rank 分片；不解包摘要哈希：pickle 字节
                # 序非本断言的语义面，逐键相等才是「权重逐位一致」本身）
                state = {
                    name: tensor.contiguous().cpu()
                    for name, tensor in
                    NetworkAssembler.loadable_state_dict(
                        driver.rewards.discriminator,
                    ).items()
                }
                gathered = dist.all_gather([state])
                self.queue.put({
                    "rank": self.rank, "code": 0, "stderr": "",
                    "states": [entry[0] for entry in gathered],
                    "report": None if report is None else report.model_dump(),
                })
            finally:
                dist.destroy()
        except Exception as exc:  # worker 崩溃的诊断面：异常文本回传
            self.queue.put({
                "rank": self.rank, "code": 1,
                "stderr": f"{type(exc).__name__}: {exc}",
                "states": None, "report": None,
            })


class SpawnedPretrainWorld:
    """torchrun 语义的本地 2 卡预训练 world（直驱动 driver 版）：spawn
    起 world 个 ``PretrainWorldWorker`` 并行跑，收集全 rank 回传。"""

    _next_port = _worker_port_base() + 50
    """TCPStore 端口游标：``SpawnedTrainWorld``（test_distributed）与本
    类是同基址的两个独立游标——同一 pytest worker 内两类交替跑会绑到
    同一端口（TIME_WAIT 的 EADDRINUSE flake 面），本类整体偏移半个带
    （50）错开；单带 100 端口内本文件的 world 数 ≪ 50，不与 test_
    distributed 游标的推进区间重叠。"""

    def __init__(
        self,
        config_path: Path,
        run_dir: Path,
        world: int,
        join_timeout_s: float,
    ) -> None:
        self.config_path = config_path
        self.run_dir = run_dir
        self.world = world
        self.join_timeout_s = join_timeout_s
        type(self)._next_port += 1
        self.port = type(self)._next_port

    def launch(self) -> list[dict]:
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        num_threads = torch.get_num_threads()
        workers = [
            PretrainWorldWorker(
                rank, self.world, self.port, self.config_path,
                self.run_dir, queue, num_threads,
            )
            for rank in range(self.world)
        ]
        processes = [context.Process(target=worker) for worker in workers]
        for process in processes:
            process.start()
        collected: dict[int, dict] = {}
        for _ in range(self.world):
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
                    f"worker 回传超时（{self.join_timeout_s}s，疑似四态"
                    f"广播/gather 集合序列错配的死锁）: {stuck}"
                )
            collected[payload["rank"]] = payload
        for process in processes:
            process.join(timeout=self.join_timeout_s)
            if process.is_alive():
                process.terminate()
                collected[process.pid] = {
                    "code": 1, "stderr": "worker join 超时（疑似死锁）",
                    "states": None, "report": None,
                }
        return [collected[rank] for rank in range(self.world)]


class TestTwoRankConfirmationPath:
    """四态的复测/确认/终止路径 + 切片合并还原全量（issue #199 AC2/AC3）。

    门槛 0.01 恒达标：每条件首测过线 → 复测确认 → 末条件终止，全程
    零更新（集合序列 = 每条件两轮「排列广播 + 分数 gather + 态广播」，
    挂死即超时红）。确认路径的判别器权重 = 冷启动（无更新步）——2 卡
    报告的 per-condition AUC 与单进程对照（同工件同 config）的对比
    直接承重「切片合并还原全量排列 + gather 全局 recon-AUC」：分片若
    错位/漏卷，AUC 读数立即偏离（容差外的分布级差）。
    """

    def test_two_rank_confirmation_matches_single_process(
        self,
        pretrain_inputs: Path,
        cli,
        tmp_path: Path,
    ) -> None:
        dist_case = DistPretrainScenario(pretrain_inputs, tmp_path / "dist")
        config_path = dist_case.write_config(pretrain_gate_auc=0.01)
        result = SpawnedTrainWorld(
            config_path, dist_case.run_dir, world=2,
            argv=[
                "pretrain", "--config", str(config_path),
                "--run-dir", str(dist_case.run_dir),
            ],
            join_timeout_s=_CONFIRM_JOIN_TIMEOUT_S,
        ).launch()
        result.assert_green()

        report = dist_case.report()
        assert report.gate_passed is True
        assert report.steps_completed == 0  # 确认步不更新
        assert sorted(report.gate_whitelist) == sorted(MODALITIES)
        assert set(report.condition_auc) == set(MODALITIES)
        # 事件流仅 rank0 落盘（最小 rank 门）：确认路径零更新 → 零事件；
        # 若两个 rank 都写，同一共享 metrics.jsonl 即双份行
        assert dist_case.events() == []
        # 支撑度卷数全局口径：切片后每 rank 只本地驻留 1/N 卷，报告必须
        # 留痕全量卷数（每条件对全量 manifest 逐值对账）
        assert report.condition_volumes == _heldout_volumes(config_path)

        # 单进程对照（同共享工件、同 config）：gather 全局 AUC == 全量
        # 测量的 AUC（σ 轮转偏移 + ε 前缀消耗 ⇒ 切片行与整批逐位同位；
        # 打分前向的卷积 batch-shape 尾差在秩统计上只表现为 ~1e-12 级
        # 读数差——容差外的差即分片错位/漏卷的分布级信号）
        ref_case = DistPretrainScenario(pretrain_inputs, tmp_path / "ref")
        ref_config = ref_case.write_config(pretrain_gate_auc=0.01)
        assert cli.run(
            "pretrain", "--config", str(ref_config),
            "--run-dir", str(ref_case.run_dir),
        ).code == 0
        reference = ref_case.report()
        assert report.condition_auc == pytest.approx(
            reference.condition_auc, abs=1e-9,
        )
        assert reference.gate_passed is True


class TestTwoRankDenseSteps:
    """更新路径 + 判别器 DDP 逐位一致 + 补测循环集合对齐（AC2/AC4）。

    门槛 0.99 不可达：每步首测不过线 → 更新（DDP allreduce），步数
    耗尽 → 补测循环（未确认条件逐个测量/gather——全 rank 走同一目标
    序列，序列错位即挂死转红）。DDP 权重断言 = worker 内 all_gather
    的可装载 state 逐键 ``torch.equal``。

    覆盖取舍（记录在案）：「首测过线 → 复测掉线回落更新」（remeasure
    →update）路径无独立 2 卡用例——fixture 冷启动权重的 per-condition
    AUC 读数带内，门槛选在首测/复测读数之间依赖测量涨落、不可确定性
    构造；该路径与确认路径的集合序列同构（remeasure 一轮的广播/gather
    调用数相同，仅广播值分支不同——分支值来自同一广播消息，全 rank
    天然一致），错位面由两个用例的集合序列联合覆盖 + 实现审查承重。
    """

    def test_two_rank_dense_steps_ddp_bitwise_and_global_cost(
        self,
        pretrain_inputs: Path,
        cli,
        tmp_path: Path,
    ) -> None:
        dist_case = DistPretrainScenario(pretrain_inputs, tmp_path / "dist")
        config_path = dist_case.write_config(
            pretrain_gate_auc=0.99,
            pretrain_max_steps=4,
            disc_lr=2e-4,
        )
        payloads = SpawnedPretrainWorld(
            config_path, dist_case.run_dir, world=2,
            join_timeout_s=_CONFIRM_JOIN_TIMEOUT_S,
        ).launch()
        assert [payload["code"] for payload in payloads] == [0, 0], payloads
        # AC4：判别器 DDP 自动生效——各 rank 更新后权重逐位一致
        # （all_gather 全体互见：任一键的任一分叉都被两两对比捕获）
        states = payloads[0]["states"]
        assert states is not None and len(states) == 2
        assert set(states[0]) == set(states[1])
        mismatched = [
            name for name in states[0]
            if not torch.equal(states[0][name], states[1][name])
        ]
        assert mismatched == [], f"各 rank 判别器权重在 DDP 后分叉: {mismatched}"
        # 非 0 rank 无报告（rank0 唯一写者契约的返回面）
        assert payloads[1]["report"] is None
        report = dist_case.report()
        assert payloads[0]["report"] is not None
        assert report.gate_passed is False
        assert report.steps_completed == 4
        # rank0 独写 checkpoint：可装载且与报告指纹对得上（守卫重载链
        # 在分布式产物上照常成立）
        assert (dist_case.run_dir / "checkpoints" /
                "pretrain_discriminator.pt").is_file()
        report.load_discriminator(ConfigLoader.load(config_path))

        events = dist_case.events()
        # 事件仅 rank0：行数 == 完成步数（双写者会翻倍）
        assert len(events) == 4
        assert [event["step"] for event in events] == [0, 1, 2, 3]
        assert [event["modality"] for event in events] == [
            MODALITIES[step % len(MODALITIES)] for step in range(4)
        ]
        # 成本读数全局口径：measurement_volumes = 全量卷数（非本地切片
        # 的 1/N）；reconstruction_forwards = 各 rank 本地读数 gather
        # 求和（整型推算值与设备/分片无关——与单进程对照严格相等）
        volumes = _heldout_volumes(config_path)
        assert all(
            event["measurement_volumes"] == volumes[event["modality"]]
            for event in events
        )
        ref_case = DistPretrainScenario(pretrain_inputs, tmp_path / "ref")
        ref_config = ref_case.write_config(
            pretrain_gate_auc=0.99,
            pretrain_max_steps=4,
            disc_lr=2e-4,
        )
        assert cli.run(
            "pretrain", "--config", str(ref_config),
            "--run-dir", str(ref_case.run_dir),
        ).code == 0
        reference_events = ref_case.events()
        assert [
            event["reconstruction_forwards"] for event in events
        ] == [
            event["reconstruction_forwards"] for event in reference_events
        ]
        # 补测循环走完：报告覆盖全部轮转条件（步数耗尽路径的补测）
        assert set(report.condition_auc) == set(MODALITIES)
        assert all(
            auc < report.gate_auc for auc in report.condition_auc.values()
        )


class TestPretrainDistributedUsageContract:
    """分布式启动的 usage 错误契约：run 目录预存（rank0 单点可见）必须
    经广播裁决让全 rank 一致返回 usage error（exit 2）——rank0 单方面
    退出、其余 rank 停在广播互等是挂死而非干净拒绝（train 侧
    ``TestSpawnedUsageContract`` 同款语义）。"""

    def test_preexisting_run_dir_rejected_on_every_rank(
        self,
        pretrain_inputs: Path,
        tmp_path: Path,
    ) -> None:
        dist_case = DistPretrainScenario(pretrain_inputs, tmp_path / "dist")
        config_path = dist_case.write_config(pretrain_gate_auc=0.01)
        dist_case.run_dir.mkdir(parents=True)
        (dist_case.run_dir / "config.json").write_text("{}", encoding="utf-8")
        result: DistTrainResult = SpawnedTrainWorld(
            config_path, dist_case.run_dir, world=2,
            argv=[
                "pretrain", "--config", str(config_path),
                "--run-dir", str(dist_case.run_dir),
            ],
            join_timeout_s=_USAGE_JOIN_TIMEOUT_S,
        ).launch()
        assert result.codes == [2, 2], result.errors
        assert any("已存在" in error for error in result.errors)
