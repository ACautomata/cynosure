"""iteration 循环、断点续训状态机、指标工件落盘；本模块承载 run 目录与
产物工件契约（spec「产物工件契约」）。

本 ticket（#16，prefactor）交付契约最小版：run 目录布局、config 快照、
metrics.jsonl 指标流（事件 schema）、Baseline manifest、checkpoint 目录。
训练循环由后续 ticket 填充。
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from cynosure.config import CynosureConfig, MODALITIES

_SEQUENTIAL_STAGES: list[str] = ["modal-label", "cross-modal"]
"""组3 序贯 = 先组1 后组2（experiment-design 章），manifest conditions 按两阶段名记录。"""


class IterEvent(BaseModel):
    """训练指标流的 per-iteration 事件（契约最小集：施工可扩不可改名）。"""

    event: Annotated[Literal["iter"], Field(default="iter")]
    iteration: int
    anchor_eval_reward: float
    intra_group_reward_std: float
    heldout_auc: float
    loss: dict[str, float]
    buffer_current_fraction: float
    buffer_replay_fraction: float
    lr: float
    elapsed_s: float


class MilestoneEvent(BaseModel):
    """训练指标流的里程碑评测事件（解码评测只发生在里程碑）。"""

    event: Annotated[Literal["milestone"], Field(default="milestone")]
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


class RunArtifacts:
    """run 目录与产物工件契约：config 快照 + metrics.jsonl + manifest +
    checkpoint 目录，落 ``$HOME``（多 rank 下指标由 rank 0 归并写出）。"""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    @classmethod
    def init(cls, config: CynosureConfig, root: Path) -> "RunArtifacts":
        """创建 run 目录并落盘契约最小集工件；run 目录已存在则拒绝
        （每次运行一个 run 目录的隔离契约，续训须显式复用并经续训入口）。"""
        paths = cls.layout(root)
        if paths.config_snapshot.exists():
            raise FileExistsError(f"run 目录已存在（不静默覆盖）: {root}")
        paths.root.mkdir(parents=True)
        paths.checkpoints.mkdir()
        paths.config_snapshot.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        paths.metrics.touch()
        paths.manifest.write_text(
            json.dumps(cls.manifest(config), indent=2), encoding="utf-8",
        )
        return cls(paths)

    @classmethod
    def layout(cls, root: Path) -> RunPaths:
        return RunPaths(
            root=root,
            config_snapshot=root / "config.json",
            metrics=root / "metrics.jsonl",
            manifest=root / "manifest.json",
            checkpoints=root / "checkpoints",
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
            fh.write(json.dumps(event.model_dump(), ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict]:
        """读回指标流全部事件（早停判定与评测脚本共同消费）。"""
        lines = self.paths.metrics.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
