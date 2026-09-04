"""iteration 循环、run 目录与产物工件契约（spec「产物工件契约」）。

- 契约层（artifacts）：run 目录布局、config 快照、metrics.jsonl 指标流、
  Baseline manifest、checkpoint 目录（ticket #16 交付）；
- rollout 相（rollout）：eval + no_grad 的 Anchor/扰动/λ 续跑/打分编排
  与训练侧诊断工件 schema；
- 循环编排（trainer）：单进程 Granular-GRPO iteration 循环（ticket #21
  tracer bullet）——MGAI advantage → 逐 k 独立梯度步 → 判别器 Online
  update → iter 事件落盘，train 启动时自动生成 buffer base 分区。
"""

from cynosure.train.artifacts import (
    IterEvent,
    MilestoneEvent,
    RunArtifacts,
    RunPaths,
)
from cynosure.train.rollout import (
    IterationRollout,
    ModalLabelConditionSampler,
    RolloutPhase,
    StepRollout,
)
from cynosure.train.trainer import (
    GranularGrpoTrainer,
    RewardCoordinator,
    TrainingDiagnostic,
    TrainingLogProbPair,
)

__all__ = [
    "GranularGrpoTrainer",
    "IterEvent",
    "IterationRollout",
    "MilestoneEvent",
    "ModalLabelConditionSampler",
    "RewardCoordinator",
    "RolloutPhase",
    "RunArtifacts",
    "RunPaths",
    "StepRollout",
    "TrainingDiagnostic",
    "TrainingLogProbPair",
]
