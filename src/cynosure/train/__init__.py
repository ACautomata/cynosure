"""iteration 循环、run 目录与产物工件契约（spec「产物工件契约」）。

- 契约层（artifacts）：run 目录布局、config 快照、metrics.jsonl 指标流、
  Baseline manifest、checkpoint 目录（ticket #16 交付）；
- rollout 相（rollout）：eval + no_grad 的 Anchor/扰动/λ 续跑/打分编排、
  按组条件分布（组1 四序列 / 组2 12 有序对）与训练侧诊断工件 schema；
- 每组 policy 装配（policy）：可训练网络 + 采样场 + 条件分布 + 优化器的
  单点分派（issue #23 三组实验矩阵的组间差异收敛处）；
- 执行序门面（executor）：async 单进程多卡执行序（#217/#226 切换期
  第一步起生产入口单口径）——静态分配表轮 + per-槽协程 + 逐 k barrier
  收集-同步 + 判别器链 + 评测三路径 + v12 单文件续训分片。
"""

from cynosure.train.artifacts import (
    REWIND_ACCOUNTING,
    BarrierTimeoutAlertEvent,
    BaselineManifest,
    IterEvent,
    ManifestEntry,
    MilestoneEvent,
    OverfitAlertEvent,
    PretrainEvent,
    RewindAccounting,
    RunArtifacts,
    RunPaths,
    WeightDivergenceAlertEvent,
)
from cynosure.train.earlystop import EarlyStopJudge, EarlyStopVerdict
from cynosure.train.executor import AsyncTrainingExecutor
from cynosure.train.rewards import RewardCoordinator
from cynosure.train.rollout import (
    ConditionSampler,
    CrossModalConditionSampler,
    IterationRollout,
    ModalLabelConditionSampler,
    MrConditionSampler,
    RolloutPhase,
    SourceLatentPool,
    StepRollout,
)
from cynosure.train.runtime import AmpContext, TrainingRuntime

__all__ = [
    "REWIND_ACCOUNTING",
    "AmpContext",
    "AsyncTrainingExecutor",
    "BarrierTimeoutAlertEvent",
    "BaselineManifest",
    "ConditionSampler",
    "CrossModalConditionSampler",
    "EarlyStopJudge",
    "EarlyStopVerdict",
    "IterationRollout",
    "IterEvent",
    "ManifestEntry",
    "MilestoneEvent",
    "ModalLabelConditionSampler",
    "MrConditionSampler",
    "OverfitAlertEvent",
    "PretrainEvent",
    "RewardCoordinator",
    "RewindAccounting",
    "RolloutPhase",
    "RunArtifacts",
    "RunPaths",
    "SourceLatentPool",
    "StepRollout",
    "TrainingRuntime",
    "WeightDivergenceAlertEvent",
]
