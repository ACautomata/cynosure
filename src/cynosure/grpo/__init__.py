"""GRPO 核心：advantage（组内标准化 + clamp）、MGAI（跨 λ 求和）、
ratio clip loss、逐 k 更新语义。

关键接口（spec #15 模块划分）：advantage 计算 / policy loss。
数值语义见 docs/spec/policy-modeling.md（MGAI、clip 1e-4、无 KL、
无参考模型）；实现由 ticket #21（T06 tracer bullet）交付——
MgaiAdvantage / ClippedPolicyLoss 是可注入的数值策略，
StepwisePolicyUpdate 编排每训练步 k 的独立梯度步（bf16 autocast +
fp32 master，口径与 rollout 相一致保证 π_old 可逐位重算）。
"""

from cynosure.grpo.advantage import MgaiAdvantage
from cynosure.grpo.loss import ClippedPolicyLoss
from cynosure.grpo.update import StepwisePolicyUpdate

__all__ = ["ClippedPolicyLoss", "MgaiAdvantage", "StepwisePolicyUpdate"]
