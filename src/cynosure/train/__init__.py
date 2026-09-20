"""iteration 循环、run 目录与产物工件契约（spec「产物工件契约」）。

- 契约层（artifacts）：run 目录布局、config 快照、metrics.jsonl 指标流、
  Baseline manifest、checkpoint 目录（ticket #16 交付）；
- rollout 相（rollout）：eval + no_grad 的 Anchor/扰动/λ 续跑/打分编排、
  按组条件分布（组1 四序列 / 组2 12 有序对）与训练侧诊断工件 schema；
- 每组 policy 装配（policy）：可训练网络 + 采样场 + 条件分布 + 优化器的
  单点分派（issue #23 三组实验矩阵的组间差异收敛处）；
- 循环编排（trainer）：单进程 Granular-GRPO iteration 循环——MGAI
  advantage → 逐 k 独立梯度步 → 判别器 Online update → iter 事件落盘，
  启动期执行 Baseline 采样等昂贵启动动作，续训状态按周期落盘/恢复；
- 续训状态机（resume）：断点续训全清单（两模型权重与 optimizer、RNG、
  iteration 计数、LR/EMA 槽位）的单文件滚动落盘与整体恢复
  （T07；清单随 ADR-0012 退役面收窄——buffer 不再落盘，RNG 为
  四条命名流）；
- 序贯编排（sequential）：组3 两阶段的单次运行（stage-1 组1 配置 →
  stage-2 base′ 冻结 + 预训练 ControlNet，支持既有产物跳过 stage-1）。
"""

from cynosure.train.artifacts import (
    REWIND_ACCOUNTING,
    BaselineManifest,
    IterEvent,
    ManifestEntry,
    MilestoneEvent,
    OverfitAlertEvent,
    PretrainEvent,
    RewindAccounting,
    RunArtifacts,
    RunPaths,
)
from cynosure.train.earlystop import EarlyStopJudge, EarlyStopVerdict
from cynosure.train.gate import ReadinessGate
from cynosure.train.gating import ConditionAucEma, DynamicWhitelist
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
from cynosure.train.sequential import SequentialTrainer, StagePlan
from cynosure.train.trainer import (
    GranularGrpoTrainer,
    IterationLoop,
    TrainingDiagnostic,
    TrainingLogProbPair,
)
from cynosure.train.whitelist import ConditionWhitelist

__all__ = [
    "REWIND_ACCOUNTING",
    "AmpContext",
    "BaselineManifest",
    "ConditionAucEma",
    "ConditionSampler",
    "ConditionWhitelist",
    "CrossModalConditionSampler",
    "DynamicWhitelist",
    "EarlyStopJudge",
    "EarlyStopVerdict",
    "GranularGrpoTrainer",
    "ReadinessGate",
    "IterationLoop",
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
    "SequentialTrainer",
    "SourceLatentPool",
    "StagePlan",
    "StepRollout",
    "TrainingDiagnostic",
    "TrainingLogProbPair",
    "TrainingRuntime",
]
