"""轨迹游标：RFlowScheduler 实际日程的自持快照（policy-modeling 章）。

sigma 日程一律以 MONAI ``set_timesteps`` 的实际输出为准（ADR-0002：
timestep transform 生效，config 字面 ``scale:1.4`` 是死参数、实际生效 1.0）。
游标在构造期快照 timesteps——共享调度器被后续 ``set_timesteps`` 复写时，
已开出的轨迹日程不受影响（spec：轨迹游标自持）；``next_timesteps`` 按位
前移、末位补 0（MONAI 推理循环的组织）。
"""

import torch
from monai.networks.schedulers import RFlowScheduler


class TrajectoryCursor:
    """一条 rollout 的日程游标：timesteps 快照 + 按位前移的 next_timesteps
    + 噪声水平换算（s = t/1000，1000=纯噪声）。"""

    def __init__(self, scheduler: RFlowScheduler) -> None:
        self._num_train_timesteps = scheduler.num_train_timesteps
        self.timesteps = scheduler.timesteps.clone()
        self.next_timesteps = torch.cat(
            (self.timesteps[1:], self.timesteps.new_zeros(1)),
        )

    @property
    def num_steps(self) -> int:
        return int(self.timesteps.numel())

    def timestep(self, index: int) -> int:
        """第 k 步的实际 timestep（0=最噪端，transform 后的 MONAI 输出）。"""
        return int(self.timesteps[index])

    def next_timestep(self, index: int) -> int:
        return int(self.next_timesteps[index])

    def sigma_level(self, index: int) -> float:
        """噪声水平 s_k = t_k / 1000（policy-modeling 章：前向加噪 x_t =
        (1−s)·x0 + s·noise，速度目标 v = x0 − noise）。"""
        return self.timestep(index) / self._num_train_timesteps

    def delta_s(self, index: int) -> float:
        """步长 Δs = s_k − s_{k+1}，与 MONAI ``step()`` 内部 dt 同式同精度
        （η=0 逐位 parity 的前提）。"""
        return (
            float(self.timestep(index) - self.next_timestep(index))
            / self._num_train_timesteps
        )
