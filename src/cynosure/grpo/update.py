"""policy 逐 k 独立更新（spec #15 执行序第 2 相：train() 逐 k 更新）。

每个被优化训练步 k 一次独立 forward→backward→optimizer.step（每
iteration 共 |M| 次，参考实现的更新语义）：log π_θ 在当前权重下重算 →
ratio → clipped loss → 反传 → AdamW step。

数值口径（bf16 autocast + fp32 master 的单进程版）：velocity 前向与
log-prob 计算进 ``torch.autocast``（dtype 取 config 的 amp_dtype，定死
bf16）；参数与梯度保持 fp32（fp32 master weights）——AdamW 直接更新
fp32 参数，无 fp16 copy。rollout 相（π_old 记录）使用同一 autocast 口径，
保证同权重下 π_old 可被逐位重算（ratio 分子分母同场，测试面 #3）。

``advantages``（MGAI 融合后的组内方向 advantage）由编排方在 rollout 相
产出后传入——本类只负责「重算→loss→梯度步」的更新语义，不做 reward
融合（MgaiAdvantage 的职责）。
"""

import torch

from cynosure.grpo.loss import ClippedPolicyLoss
from cynosure.policy.condition import RolloutCondition
from cynosure.policy.sampler import RolloutSampler


class StepwisePolicyUpdate:
    """逐 k 更新编排：单训练步的重算 → clipped loss → 优化器步。"""

    def __init__(
        self,
        sampler: RolloutSampler,
        optimizer: torch.optim.Optimizer,
        loss: ClippedPolicyLoss,
        device_type: str = "cpu",
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.sampler = sampler
        self.optimizer = optimizer
        self.loss_fn = loss
        self._device_type = device_type
        self._amp_dtype = amp_dtype

    def step(
        self,
        step_index: int,
        x_k: torch.Tensor,
        condition: RolloutCondition,
        directions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> float:
        """第 k 步的一次独立梯度步：log π_new 重算（当前权重）→ clipped
        surrogate → backward → optimizer.step，返回标量 loss 值。

        ``advantages`` 已是 MGAI 融合后的组内方向 advantage（rollout 相
        产出）；``old_log_probs`` 是 rollout 记录的 π_old、本方法不改动。
        """
        with torch.autocast(self._device_type, dtype=self._amp_dtype):
            new_log_probs = self.sampler.evaluate_log_prob(
                x_k, step_index, condition, directions,
            )
            loss = self.loss_fn.loss(new_log_probs, old_log_probs, advantages)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def accumulate(
        self,
        step_index: int,
        x_k: torch.Tensor,
        condition: RolloutCondition,
        directions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        example_count: int,
    ) -> float:
        """逐 k 收集-同步的单例累积任务（async 执行序，#217 决议 2）：
        log π 重算 → clipped loss → **loss 侧 × 1/example_count** 反传——
        只累积梯度，不 zero_grad、不 optimizer.step（零梯度与优化器步归
        barrier 相：k 内多例本卡累积 → 跨卡 allreduce SUM → 各卡一次
        step；loss 侧 1/N × SUM ≡ 现行 FSDP/DDP AVG 的数学等价形态，
        N = 全 iteration 例子数）。

        返回**未缩放**的 reported loss（上报口径 = 先 ``float(loss.
        detach())`` 取值、再对缩放副本 backward——上报与现行逐位一致，
        不做回乘 D 的浮点往返）。``example_count`` < 1 显式拒绝。"""
        if example_count < 1:
            raise ValueError(
                f"全 iteration 例子数须 ≥ 1，得到 {example_count}"
                "（loss×(1/N)+SUM 口径的缩放分母）"
            )
        with torch.autocast(self._device_type, dtype=self._amp_dtype):
            new_log_probs = self.sampler.evaluate_log_prob(
                x_k, step_index, condition, directions,
            )
            loss = self.loss_fn.loss(new_log_probs, old_log_probs, advantages)
        reported = float(loss.detach())
        (loss / example_count).backward()
        return reported
