"""policy：scheduler 轨迹游标、Anchor 轨迹、单步 SDE 核、ODE 续跑、
per-group 采样场、log-prob（spec #15 模块划分）。

核心测试面（η=0 parity / 噪声注入 sanity / log-prob 一致性）经 CLI 诊断
工件驱动（spec「Testing Decisions」的唯一 seam）；薄封装单元的数值锚
（η=0 逐位对照、组合场 batch 组织、MONAI 日程实际输出）以 MONAI 库为
真值直接测试——先例同 netbuild 装配测试。fixture 轨迹诊断回路
（TrajectoryDiagnosticRunner）是 ``train --dump-trajectory`` 背后的编排。
"""

from cynosure.policy.condition import ModalityMapping, RolloutCondition
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.diagnostic import (
    LatentFingerprint,
    LogProbPair,
    TerminalSampleStats,
    TrajectoryDiagnosticReport,
    TrajectoryDiagnosticRunner,
    TrajectoryStepStats,
)
from cynosure.policy.field import BareConditionField, CfgCombinedField, VelocityField
from cynosure.policy.kernel import SdeKernel, SdeTransition
from cynosure.policy.sampler import RolloutSampler

__all__ = [
    "BareConditionField",
    "CfgCombinedField",
    "LatentFingerprint",
    "LogProbPair",
    "ModalityMapping",
    "RolloutCondition",
    "RolloutSampler",
    "SdeKernel",
    "SdeTransition",
    "TerminalSampleStats",
    "TrajectoryCursor",
    "TrajectoryDiagnosticReport",
    "TrajectoryDiagnosticRunner",
    "TrajectoryStepStats",
    "VelocityField",
]
