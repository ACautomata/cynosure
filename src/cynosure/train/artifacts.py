"""run 目录与产物工件契约（spec「产物工件契约」）：run 目录布局、
config 快照、metrics.jsonl 指标流（事件 schema）、Baseline manifest、
checkpoint 目录。

本模块由 ticket #16（prefactor）交付、#21 起承载训练循环的消费面；
从 ``train`` 包体拆出（循环模块 artifacts/trainer/rollout 的共同依赖，
防 trainer → 包体的循环 import）。
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cynosure.config import CynosureConfig, MODALITIES

_SEQUENTIAL_STAGES: list[str] = ["modal-label", "cross-modal"]
"""组3 序贯 = 先组1 后组2（experiment-design 章），manifest conditions 按两阶段名记录。"""

_RANK_WAIT_TIMEOUT_S: float = 60.0
"""非 0 rank 等待 rank 0 创建 run 目录的超时（秒）。"""

_RANK_POLL_INTERVAL_S: float = 0.05
"""非 0 rank 轮询 run 目录出现的间隔（秒）。"""

POLICY_CHECKPOINT_TEMPLATE = "policy_iter{iteration}.pt"
"""policy checkpoint 文件名模板（契约布局的一部分：组3 stage-1 的复用
查找 ``SequentialTrainer._locate_stage1_product`` 按同一形态解析）。"""


class IterEvent(BaseModel):
    """训练指标流的 per-iteration 事件（契约最小集：施工可扩不可改名）。

    指标 JSONL 是契约工件：NaN/Inf 经默认 json.dumps 会写成非标准 token，
    严格消费方拒读——构造期即拒绝非有限浮点。
    """

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["iter"] = "iter"
    iteration: int
    stage: int = 1
    """组内阶段号（序贯两阶段的归因轴）：单阶段组（组1/组2）恒 1，
    组3 stage-2 事件 = 2——「每组判别器与 buffer 独立」在指标流上的
    观测面（各阶段事件互不混淆）。"""
    modality: str
    """本 iteration 采样的目标序列（条件分布四序列均匀采样）——reward/
    loss/AUC 按目标序列归因的轴（per-sequence 健康监控，experiment-design）。"""
    anchor_eval_reward: float
    intra_group_reward_std: float
    heldout_auc: float
    loss: dict[str, float]
    buffer_current_fraction: float
    """更新批的当前 fake 混合占比（N_d 跳过的 iteration 为 0）。"""
    buffer_replay_fraction: float
    """更新批的回放混合占比（N_d 跳过的 iteration 为 0）。"""
    buffer_base_occupied: int
    """Replay buffer base 分区当前占用（固定分区的状态观测面）。"""
    buffer_recent_occupied: int
    """Replay buffer 近期分区当前占用（FIFO 滚动观测面）。"""
    lr: float
    elapsed_s: float


class MilestoneEvent(BaseModel):
    """训练指标流的里程碑评测事件（解码评测只发生在里程碑）。"""

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["milestone"] = "milestone"
    iteration: int
    fid: float
    kid: float | None = None
    ssim: float | None = None
    mae: float | None = None
    criteria_summary: dict[str, float] = Field(default_factory=dict)
    early_stop: bool = False


@dataclass
class RunPaths:
    """run 目录内各工件文件的固定路径（契约布局）。"""

    root: Path
    config_snapshot: Path
    metrics: Path
    manifest: Path
    checkpoints: Path
    trajectory_diagnostic: Path
    """轨迹诊断工件（fixture 诊断开关 --dump-trajectory 产出；诊断未跑则无此文件）。"""
    training_diagnostic: Path
    """训练诊断工件（--dump-trajectory 时的训练侧 log-prob 对；未开启则无此文件）。"""


class RunArtifacts:
    """run 目录与产物工件契约：config 快照 + metrics.jsonl + manifest +
    checkpoint 目录，落 ``$HOME``（多 rank 下指标由 rank 0 归并写出）。"""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    @classmethod
    def init(
        cls, config: CynosureConfig, root: Path, *,
        wait_timeout_s: float = _RANK_WAIT_TIMEOUT_S,
    ) -> "RunArtifacts":
        """创建 run 目录并落盘契约最小集工件；run 目录已存在则拒绝
        （每次运行一个 run 目录的隔离契约，续训须显式复用并经续训入口）。

        torchrun 环境（``RANK`` env）下 rank 0 创建、其余 rank 轮询等待目录
        出现后原样采用（自身不写盘——多 rank 下指标由 rank 0 归并写出）；
        等待超时抛 ``TimeoutError``，防各 rank 静默分裂 run。
        """
        paths = cls.layout(root)
        rank = os.environ.get("RANK")
        if rank is None or rank == "0":
            if paths.config_snapshot.exists():
                raise FileExistsError(f"run 目录已存在（不静默覆盖）: {root}")
            cls._create_minimal_set(config, paths)
            return cls(paths)
        return cls._await_rank0(paths, wait_timeout_s)

    @classmethod
    def _create_minimal_set(cls, config: CynosureConfig, paths: RunPaths) -> None:
        paths.root.mkdir(parents=True)
        paths.checkpoints.mkdir()
        paths.config_snapshot.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        paths.metrics.touch()
        paths.manifest.write_text(
            json.dumps(cls.manifest(config), indent=2), encoding="utf-8",
        )

    @classmethod
    def _await_rank0(cls, paths: RunPaths, wait_timeout_s: float) -> "RunArtifacts":
        """轻量文件系统 barrier：轮询等待 rank 0 写出 config 快照
        （不引入 torch.distributed 初始化——那属 orchestration ticket）。"""
        deadline = time.monotonic() + wait_timeout_s
        while not paths.config_snapshot.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"等待 rank 0 创建 run 目录超时（{wait_timeout_s}s）: {paths.root}"
                )
            time.sleep(_RANK_POLL_INTERVAL_S)
        return cls(paths)

    @classmethod
    def layout(cls, root: Path) -> RunPaths:
        return RunPaths(
            root=root,
            config_snapshot=root / "config.json",
            metrics=root / "metrics.jsonl",
            manifest=root / "manifest.json",
            checkpoints=root / "checkpoints",
            trajectory_diagnostic=root / "trajectory.json",
            training_diagnostic=root / "training.json",
        )

    @classmethod
    def default_root(cls, config: CynosureConfig) -> Path:
        """默认 run 目录：``$HOME/.cynosure/runs/<UTC 微秒时间戳>-<group>``。"""
        base = Path.home() / ".cynosure" / "runs"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        return base / f"{timestamp}-{config.experiment.group}"

    @classmethod
    def manifest(cls, config: CynosureConfig) -> dict:
        """Baseline 采样清单契约最小集：seed、条件、样本路径。

        Baseline 与 RL 后重采共用同一 manifest（同 seed 同条件）。
        """
        group = config.experiment.group
        if group == "modal-label":
            conditions: list = list(MODALITIES)
        elif group == "cross-modal":
            conditions = [list(pair) for pair in config.experiment.cross_modal_pairs]
        else:
            conditions = list(_SEQUENTIAL_STAGES)
        return {
            "seed": config.schedule.seed,
            "group": group,
            "conditions": conditions,
            "samples": [],
        }

    def append_event(self, event: IterEvent | MilestoneEvent) -> None:
        """向训练指标流追加一行 JSON 事件（按行追加、rank 0 归并）。"""
        with open(self.paths.metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                event.model_dump(), ensure_ascii=False, allow_nan=False,
            ) + "\n")

    def read_events(self) -> list[dict]:
        """读回指标流全部事件（早停判定与评测脚本共同消费）。"""
        lines = self.paths.metrics.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def rewind_events(self, iteration: int, stage: int) -> int:
        """续训回退指标流：删除恢复点之后本 stage 的事件（iteration ≥
        恢复点且 stage 匹配）——被中断的半截执行史由恢复后的重执行重写，
        保住「每 iteration 每 stage 一条事件」的流不变量（重复事件会污染
        早停判定等下游消费者）。milestone 事件暂不带 stage 字段、按 stage 1
        归属（组3 两阶段的 milestone 归属待 eval ticket 随事件 schema 补
        stage）。返回删除的事件数。"""
        events = self.read_events()
        kept = [
            event for event in events
            if event.get("iteration", 0) < iteration
            or event.get("stage", 1) != stage
        ]
        removed = len(events) - len(kept)
        if removed:
            tmp = self.paths.metrics.with_name(
                self.paths.metrics.name + ".rewind.tmp",
            )
            with open(tmp, "w", encoding="utf-8") as fh:
                for event in kept:
                    fh.write(json.dumps(
                        event, ensure_ascii=False, allow_nan=False,
                    ) + "\n")
            # 原子替换（与续训状态同一 durability 口径）：中途崩溃不留
            # 半截重写的指标流
            os.replace(tmp, self.paths.metrics)
        return removed
