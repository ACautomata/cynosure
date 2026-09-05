"""iteration 循环、run 目录与产物工件契约（spec「产物工件契约」）。

- 契约层（artifacts）：run 目录布局、config 快照、metrics.jsonl 指标流、
  Baseline manifest、checkpoint 目录（ticket #16 交付）；
- rollout 相（rollout）：eval + no_grad 的 Anchor/扰动/λ 续跑/打分编排、
  按组条件分布（组1 四序列 / 组2 12 有序对）与训练侧诊断工件 schema；
- 每组 policy 装配（policy）：可训练网络 + 采样场 + 条件分布 + 优化器的
  单点分派（issue #23 三组实验矩阵的组间差异收敛处）；
- 循环编排（trainer）：单进程 Granular-GRPO iteration 循环——MGAI
  advantage → 逐 k 独立梯度步 → 判别器 Online update → iter 事件落盘，
  train 启动时自动生成 buffer base 分区，续训状态按周期落盘/恢复；
- 续训状态机（resume）：断点续训全清单（两模型权重与 optimizer、buffer
  两区、RNG、iteration 计数、LR/EMA 槽位）的单文件滚动落盘与整体恢复
  （T07）；
- 序贯编排（sequential）：组3 两阶段的单次运行（stage-1 组1 配置 →
  stage-2 base′ 冻结 + 预训练 ControlNet，支持既有产物跳过 stage-1）。
"""

from cynosure.train.artifacts import (
    BaselineManifest,
    IterEvent,
    ManifestEntry,
    MilestoneEvent,
    RunArtifacts,
    RunPaths,
)
from cynosure.train.earlystop import EarlyStopJudge, EarlyStopVerdict
from cynosure.train.rollout import (
    ConditionSampler,
    CrossModalConditionSampler,
    IterationRollout,
    ModalLabelConditionSampler,
    RolloutPhase,
    SourceLatentPool,
    StepRollout,
)
from cynosure.train.sequential import SequentialTrainer, StagePlan
from cynosure.train.trainer import (
    GranularGrpoTrainer,
    IterationLoop,
    RewardCoordinator,
    TrainingDiagnostic,
    TrainingLogProbPair,
)

__all__ = [
    "BaselineManifest",
    "ConditionSampler",
    "CrossModalConditionSampler",
    "EarlyStopJudge",
    "EarlyStopVerdict",
    "GranularGrpoTrainer",
    "IterationLoop",
    "IterationRollout",
    "IterEvent",
    "ManifestEntry",
    "MilestoneEvent",
    "ModalLabelConditionSampler",
    "RewardCoordinator",
    "RolloutPhase",
    "RunArtifacts",
    "RunPaths",
    "SequentialTrainer",
    "SourceLatentPool",
    "StagePlan",
    "StepRollout",
    "TrainingDiagnostic",
    "TrainingLogProbPair",
]
