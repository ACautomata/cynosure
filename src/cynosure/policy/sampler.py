"""Rollout 编排（policy-modeling 章「实现接缝」的 policy 薄封装）。

1. **Anchor 轨迹**：同一初始噪声全 ODE（η=0）采出，逐步存下 latent；
2. **被优化训练步 k** 用单步 SDE 核替换确定性步，产生 G 个方向
   （全组共享 anchor：无条件分支 batch=1 一次评估全组复用）；
3. 各方向 **ODE 续跑到 x_0**（确定性）。

除被优化步外全组共享同一确定性轨迹，组内差异唯一来源于该步注入的噪声
——步级 reward 归因的结构前提（research/granular-grpo.md §3）。
"""

import torch

from cynosure.policy.condition import RolloutCondition
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel


class RolloutSampler:
    """Anchor 轨迹 / 单步扰动 / ODE 续跑的 rollout 编排。"""

    def __init__(
        self,
        field: CfgCombinedField,
        kernel: SdeKernel,
        cursor: TrajectoryCursor,
    ) -> None:
        self._field = field
        self._kernel = kernel
        self._cursor = cursor
        self._deterministic = SdeKernel.deterministic(s_max=kernel.s_max)

    def anchor_trajectory(
        self,
        initial_noise: torch.Tensor,
        condition: RolloutCondition,
    ) -> list[torch.Tensor]:
        """同一批初始噪声的 η=0 全 ODE 轨迹：返回逐步 latent
        （trajectory[0] = 初始噪声，长度 = num_steps + 1）；
        batch 维并行（同组条件批量共享日程）。"""
        trajectory = [initial_noise]
        x = initial_noise
        for index in range(self._cursor.num_steps):
            x = self._deterministic_step(x, index, condition)
            trajectory.append(x)
        return trajectory

    def perturb_group(
        self,
        x_k: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """第 k 步 SDE 扰动：x_k（batch=1）→ G 方向 x_{k+1} 与各自 log-prob。

        组内共享 anchor——velocity 由 ``group_velocity`` 以两次 batch=1
        前向得出（无条件分支全组一次评估复用）。η=0 时无高斯密度可记，
        log-prob 返回 ``None``（确定性步不存在策略密度，非缺数据）。
        """
        group_size = noise.shape[0]
        velocity = self._field.group_velocity(
            x_k, self._cursor.timestep(index), condition, group_size,
        )
        transition = self._kernel.transition(
            x_k.expand(group_size, *x_k.shape[1:]),
            velocity,
            self._cursor.sigma_level(index),
            self._cursor.delta_s(index),
            noise=noise,
        )
        if self._kernel.eta <= 0.0:
            return transition.sample, None
        return transition.sample, self._kernel.log_prob(transition.sample, transition)

    def evaluate_log_prob(
        self,
        x_k: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """采样场重算口径的 log-prob（GRPO 更新侧与 rollout 侧同一入口）。

        组织沿用扰动步的全组复用技巧（policy-modeling 章：被优化步上
        无条件分支 batch=1 一次评估全组复用；与 batch=2 前向的口径一致，
        仅 batch 尺寸的 fp32 舍入差）。
        """
        group_size = samples.shape[0]
        velocity = self._field.group_velocity(
            x_k, self._cursor.timestep(index), condition, group_size,
        )
        transition = self._kernel.transition(
            x_k.expand(group_size, *x_k.shape[1:]),
            velocity,
            self._cursor.sigma_level(index),
            self._cursor.delta_s(index),
        )
        return self._kernel.log_prob(samples, transition)

    def continue_to_terminal(
        self,
        latents: torch.Tensor,
        index: int,
        condition: RolloutCondition,
    ) -> torch.Tensor:
        """从第 index+1 步 ODE 续跑到 x_0（确定性），batch 维并行；
        index 为最后一步时原样返回（无续跑空间）。"""
        x = latents
        for step in range(index + 1, self._cursor.num_steps):
            x = self._deterministic_step(x, step, condition)
        return x

    def _deterministic_step(
        self,
        x: torch.Tensor,
        index: int,
        condition: RolloutCondition,
    ) -> torch.Tensor:
        velocity = self._field.velocity(x, self._cursor.timestep(index), condition)
        return self._deterministic.transition(
            x,
            velocity,
            self._cursor.sigma_level(index),
            self._cursor.delta_s(index),
        ).sample
