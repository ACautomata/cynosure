"""组3 序贯两阶段的单次运行编排（issue #23 / experiment-design「组3 序贯
衔接」）。

单次运行 = stage-1（组1 配置训练并产出 base′）→ stage-2（base′ 冻结 +
预训练 ControlNet 复用初始化，组2 配置）。两阶段各是一次完整的单阶段
训练（GranularGrpoTrainer）：判别器与 Replay buffer 随训练实例隔离——
跨阶段不复用（experiment-design：判别器在线跟踪本阶段 fake 分布），
stage-2 的 buffer base 分区由 stage-2 自己的初始 policy 生成。

产物布局（同一次 run 目录内）：stage-1 无前缀（与独立组1 run 逐字一致，
故 ``stage1_run_dir`` 可指向任意其一）；stage-2 带 ``stage2_`` 前缀。
指标流单文件、事件按 ``stage`` 字段区分两阶段。

stage-1 传递（spec 配置项清单「组3 衔接」）：缺省时同一次运行内先跑
stage-1；config 指定既有 stage-1 产物 run 目录则跳过 stage-1，只跑 stage-2。
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from cynosure.config import CynosureConfig
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.trainer import GranularGrpoTrainer, StageTag

_STAGE2_CHECKPOINT_PREFIX = "stage2_"
"""stage-2 产物前缀：隔离两阶段同名 checkpoint（stage-1 无前缀保持复用布局）。"""

_STAGE1_PRODUCT_SOURCE_GROUPS = ("modal-label", "sequential")
"""可复用 stage-1 产物的来源 run 组别：这两组的无前缀 ``policy_iter*.pt``
按构造是 base′ UNet checkpoint；跨模态 run 的同名文件是 ControlNet
checkpoint（其可训练对象），不得混作 stage-2 的 base′。"""

_POLICY_CKPT_PATTERN = re.compile(r"policy_iter(\d+)\.pt$")
"""stage-1 产物文件名形态（与 artifacts.POLICY_CHECKPOINT_TEMPLATE 同一字面，
此处按名解析 iteration 取最大 = 最终 base′）。"""


@dataclass(frozen=True)
class StagePlan:
    """单阶段执行计划：阶段号、该阶段 config、产物命名前缀。"""

    stage: int
    config: CynosureConfig
    checkpoint_prefix: str


class SequentialTrainer:
    """组3 序贯编排：两阶段计划（base′ 传递路径的解析）与顺序执行。"""

    def __init__(
        self,
        config: CynosureConfig,
        run_artifacts: RunArtifacts,
        *,
        device: torch.device | None = None,
    ) -> None:
        if config.experiment.group != "sequential":
            raise ValueError(
                f"SequentialTrainer 只编排组3（sequential），得到组 "
                f"{config.experiment.group}",
            )
        self._config = config
        self._artifacts = run_artifacts
        self._device = device

    def plan(self) -> list[StagePlan]:
        """两阶段执行计划（stage-1 产物路径在执行前即可解析，计划的
        base′ 传递是外部可断言的公开契约）。"""
        existing_stage1 = self._config.experiment.stage1_run_dir
        if existing_stage1 is not None:
            return [self._stage2_plan(self._locate_stage1_product(existing_stage1))]
        base_prime = self._artifacts.paths.checkpoints / (
            f"policy_iter{self._config.schedule.max_iterations}.pt"
        )
        return [self._stage1_plan(), self._stage2_plan(base_prime)]

    def run(self) -> int:
        """顺序执行全部阶段，返回完成的 iteration 总数（= Σ 各阶段）。"""
        completed = 0
        for item in self.plan():
            completed += GranularGrpoTrainer(
                item.config,
                self._artifacts,
                device=self._device,
                stage=StageTag(item.stage, item.checkpoint_prefix),
            ).run()
        return completed

    def _stage1_plan(self) -> StagePlan:
        """stage-1 = 组1 配置（顺序矩阵第一行；产物名无前缀 = 独立组1
        run 的同布局，故两处 stage-1 产物可互换复用）。"""
        config = self._config.model_copy(deep=True)
        config.experiment.group = "modal-label"
        config.experiment.stage1_run_dir = None
        return StagePlan(stage=1, config=config, checkpoint_prefix="")

    def _stage2_plan(self, base_prime: Path) -> StagePlan:
        """stage-2 = 组2 配置 + base′ 冻结（unet checkpoint 重写为 stage-1
        产物）+ 预训练 ControlNet 复用初始化（artifacts.controlnet_ckpt
        不变，experiment-design「从预训练 ControlNet checkpoint 复用作
        初始化」）。"""
        config = self._config.model_copy(deep=True)
        config.experiment.group = "cross-modal"
        config.experiment.stage1_run_dir = None
        config.artifacts.unet_ckpt = base_prime
        return StagePlan(
            stage=2, config=config, checkpoint_prefix=_STAGE2_CHECKPOINT_PREFIX,
        )

    @staticmethod
    def _locate_stage1_product(run_dir: Path) -> Path:
        """既有 stage-1 run 目录 → 最终 base′ checkpoint（policy_iter 取
        最大 iteration）。来源 run 先验组别（config.json）：仅组1 run 与
        序贯 run 的无前缀 ``policy_iter*.pt`` 按构造是 base′ UNet——跨模态
        run 的同名文件是 ControlNet checkpoint，选定前拒绝（清晰输入契约
        错误），而非 stage-2 装载时的裸 RuntimeError。"""
        config_snapshot = Path(run_dir) / "config.json"
        if not config_snapshot.is_file():
            raise ValueError(
                f"stage1_run_dir 缺 config.json（无法确认来源 run 的组别，"
                f"stage-1 产物须来自组1/modal-label run 或序贯 run）: {run_dir}"
            )
        group = json.loads(
            config_snapshot.read_text(encoding="utf-8"),
        ).get("experiment", {}).get("group")
        if group not in _STAGE1_PRODUCT_SOURCE_GROUPS:
            raise ValueError(
                f"stage1_run_dir 须指向组1（modal-label）run 或序贯 run 的 "
                f"stage-1 产物（得到 experiment.group = {group!r}；跨模态 "
                f"run 的 policy_iter*.pt 是 ControlNet checkpoint 而非 "
                f"base′ UNet）: {run_dir}"
            )
        checkpoints = Path(run_dir) / "checkpoints"
        candidates: list[tuple[int, Path]] = []
        for path in checkpoints.glob("policy_iter*.pt"):
            match = _POLICY_CKPT_PATTERN.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise ValueError(
                f"stage1_run_dir 无 stage-1 产物（期望 "
                f"checkpoints/policy_iter*.pt）: {run_dir}"
            )
        return max(candidates, key=lambda item: item[0])[1]
