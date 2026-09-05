"""采样数值口径（bf16 autocast + fp32 master 的单进程落地）。

``AmpContext`` 是装配期单点选定的「设备 + autocast dtype」：所有模型与
rollout/打分张量随 device 放置——autocast(device_type) 只影响前向
dtype，不移动张量。train 循环与 eval 评测相（里程碑/Baseline/重采的
policy 采样）共用同一口径，保证 π_old 可逐位重算、里程碑采样与训练
rollout 同场。本模块位于 policy 包——train 与 eval 两侧共同依赖的
import 环安全位。
"""

from dataclasses import dataclass

import torch

AMP_DTYPES: dict[str, torch.dtype] = {"bf16": torch.bfloat16}
"""config amp_dtype（Literal["bf16"] 定死）→ torch autocast dtype。"""


@dataclass(frozen=True)
class AmpContext:
    """装配期单点选定的数值口径：设备 + autocast dtype（bf16 autocast +
    fp32 master weights 的单进程落地）。所有模型与 rollout/打分张量随
    device 放置——autocast(device_type) 只影响前向 dtype，不移动张量。"""

    device: torch.device
    dtype: torch.dtype

    @property
    def device_type(self) -> str:
        return self.device.type
