"""轨迹游标：RFlowScheduler 实际日程的自持快照（policy-modeling 章）。

sigma 日程一律以 MONAI ``set_timesteps`` 的实际输出为准（ADR-0002：
timestep transform 生效，config 字面 ``scale:1.4`` 是死参数、实际生效 1.0）。
游标在构造期快照 timesteps——共享调度器被后续 ``set_timesteps`` 复写时，
已开出的轨迹日程不受影响（spec：轨迹游标自持）；``next_timesteps`` 按位
前移、末位补 0（MONAI 推理循环的组织）。
"""

from collections.abc import Sequence

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

    def delta_s(self, index: int, stride: int = 1) -> float:
        """步长 Δs = (t_k − t_next)/1000，与 MONAI ``step()`` 内部 dt 同式
        同精度（η=0 逐位 parity 的前提）；stride > 1 时 next 取抽稀日程的
        访问位 k+stride（MGAI 粒度续跑的大步），越过日程末端时取 0
        （末段直落 σ=0 终点）。"""
        next_index = index + stride
        next_timestep = (
            self.timestep(next_index) if next_index < self.num_steps else 0
        )
        return (self.timestep(index) - next_timestep) / self._num_train_timesteps

    def continue_start_index(self, sigma: float) -> int:
        """σ 水平 → ``continue_to_terminal`` 的续跑起点下标：σ = s_k 的
        样本位于第 k 步的输入位置（第 k−1 步的输出位置），续跑从第 k 步
        开始积分。s = 0（σ=0 终点之后）走调用方短路、不进本方法；最噪端
        s≈1（下标 0，被优化步集合 M 排除——ADR-0012 决策 2）与日程点外
        的 σ 显式拒绝。重构装配（更新批）与测量模板（测量批）共用本
        换算——无第二套：两处漂移即更新批与测量读数的口径分叉。"""
        for step in range(self.num_steps):
            if self.sigma_level(step) == sigma:
                if step == 0:
                    raise ValueError(
                        f"sigma={sigma} 是最噪端（日程下标 0，s≈1 奇异点）"
                        "——不在重构候选（被优化步集合 M 排除下标 0，"
                        "ADR-0012 决策 2）"
                    )
                return step - 1
        raise ValueError(
            f"sigma={sigma!r} 不是该条件的日程点（重构起点无从定位；"
            "候选 = 被优化步的 sigma 日程点）"
        )

    def assert_reconstruction_candidates(
        self, step_indices: Sequence[int], name: str,
    ) -> None:
        """被优化步集合在本条件日程的**中段**（合法候选 = 1..num_steps−2）：
        首位 s≈1 奇异端、末位零续跑空间（重构退化透传）都非法——小锚
        日程下 M 截尾在此显式暴露（可读报错点名条件与越界位），而非
        ``sigma_level`` 越界的 IndexError 或首步使用才炸。测量面**没有**
        ``reconstruct`` 的 s=0 短路：末位产生的零内容批次静默入测量会把
        上岗判据污染成 chance 带上的噪声。"""
        overflow = [
            step for step in step_indices
            if step >= self.num_steps - 1
        ]
        if overflow:
            raise ValueError(
                f"被优化步 {overflow} 越界条件 {name} 的重构候选"
                f"（num_steps={self.num_steps}：合法候选为 1..num_steps−2"
                "，首位的 s≈1 奇异端与末位的零续跑空间都排除）"
            )
