"""Rollout 编排（policy-modeling 章「实现接缝」的 policy 薄封装）。

1. **Anchor 轨迹**：同一初始噪声全 ODE（η=0）采出，逐步存下 latent；
2. **被优化训练步 k** 用单步 SDE 核替换确定性步，产生 G 个方向
   （全组共享 anchor：无条件分支 batch=1 一次评估全组复用）；
3. 各方向 **ODE 续跑到 x_0**（确定性）。

除被优化步外全组共享同一确定性轨迹，组内差异唯一来源于该步注入的噪声
——步级 reward 归因的结构前提（research/granular-grpo.md §3）。

sigma 日程经 ``ConditionSchedules`` 按条件名选择（#129 逐条件锚）：
同一批 rollout 共享同一条件 → 同一份日程快照；GRPO 更新侧重算
log-prob 经同一入口，扰动与重算的日程口径逐位一致。
"""

import torch

from cynosure.policy.condition import RolloutCondition
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import VelocityField
from cynosure.policy.kernel import SdeKernel, SdeTransition
from cynosure.policy.schedules import ConditionSchedules


_ACTIVATION_BYTES_PER_LATENT_VOXEL = 4 * 1024
"""单次前向激活的经验系数（字节 / latent 体素 / 前向样本）：#123 首跑
OOM 探针在生产 UNet（180.5M 参数、bf16 autocast、no_grad）实测
≈3.2 KiB（三个 latent 形状 × 三档批量落同一直线），取整上界 4 KiB 留
余量。显存占用与权重值无关；网络结构变化时在此重标。"""

AUTO_FORWARD_ACTIVATION_FRACTION = 0.6
"""自动探测口径（``forward_activation_budget_gib`` 缺省时）：单次前向
激活预算 = 该比例 × 设备总显存——余下四成留给常驻（权重、优化器态、
缓冲）与分配器碎片。按总显存而非当前空闲探测：同设备上可复现；反过来
共享实例的实际可用显存可能远低于总显存（他进程占用不进本口径），此类
场景须显式钉 ``forward_activation_budget_gib``。"""

DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES = 40 * 2**30
"""无设备探测面（CPU fixture / 未传 device）时的回落预算：探针曲线上
最大生产条件的 40 GiB 档（16 样本即 53 GiB，20 以上在 64 GiB 卡上 OOM）。"""


def budget_from_total_memory(total_bytes: int) -> int:
    """设备总显存 → 前向激活预算（纯函数：探测与测试共用）。"""
    return max(1, int(total_bytes * AUTO_FORWARD_ACTIVATION_FRACTION))


def cuda_total_memory(device: "torch.device | None") -> int | None:
    """设备总显存（字节）：无 CUDA 探测面（CPU fixture / 未传设备 / CUDA
    栈不可用）返回 ``None``——自动探测与装配期上界校验共用的唯一取数点
    （同一件五条件判定曾在 sampler 与 TrainingRuntime 各写一份）。"""
    if (
        device is not None
        and device.type == "cuda"
        and torch.cuda.is_available()
    ):
        return torch.cuda.get_device_properties(device).total_memory
    return None


def auto_forward_activation_budget(device: "torch.device | None") -> int:
    """按设备总显存自动探测前向激活预算；无 CUDA 设备（CPU fixture 口径）
    回落 ``DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES``。"""
    total = cuda_total_memory(device)
    if total is None:
        return DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES
    return budget_from_total_memory(total)


class RolloutSampler:
    """Anchor 轨迹 / 单步扰动 / ODE 续跑的 rollout 编排。"""

    def __init__(
        self,
        field: VelocityField,
        kernel: SdeKernel,
        schedules: ConditionSchedules,
        forward_activation_budget: int | None = None,
    ) -> None:
        self._field = field
        self._kernel = kernel
        self._schedules = schedules
        self._deterministic = SdeKernel.deterministic(s_max=kernel.s_max)
        # 前向激活预算（字节）：装配方按 config/设备探测注入（单一解析点
        # = ``TrainingRuntime.forward_activation_budget``）；缺省回落常量
        # ——诊断回路与测试直构同样带分块保护
        self._forward_budget = (
            DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES
            if forward_activation_budget is None
            else forward_activation_budget
        )

    def anchor_trajectory(
        self,
        initial_noise: torch.Tensor,
        condition: RolloutCondition,
    ) -> list[torch.Tensor]:
        """同一批初始噪声的 η=0 全 ODE 轨迹：返回逐步 latent
        （trajectory[0] = 初始噪声，长度 = num_steps + 1）；
        batch 维并行（同组条件批量共享日程）。"""
        cursor = self._schedules.cursor(condition.name)
        trajectory = [initial_noise]
        x = initial_noise
        for index in range(cursor.num_steps):
            x = self._deterministic_step(x, index, condition, cursor)
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
        cursor = self._schedules.cursor(condition.name)
        transition = self._group_transition(
            x_k, index, condition, group_size, cursor, noise=noise,
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
        cursor = self._schedules.cursor(condition.name)
        transition = self._group_transition(
            x_k, index, condition, samples.shape[0], cursor,
        )
        return self._kernel.log_prob(samples, transition)

    def continue_to_terminal(
        self,
        latents: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        stride: int = 1,
    ) -> torch.Tensor:
        """从第 index 步之后 ODE 续跑到 x_0（确定性），batch 维并行；
        index 为最后一步时原样返回（无续跑空间）。

        ``stride`` = Granularity λ（MGAI）：按时间步间隔 λ 抽稀 sigma 日程
        积分——λ=1 逐步（默认，与 MONAI 推理循环同组织）；λ>1 首段走一个
        与 λ=1 同粒度的细步（扰动后 latent 位于 index+1，参考实现的访问
        日程 ``suffix = sigma_schedule[eta_step+2::g]`` 从 index+2 起才
        抽稀——跳过该细步会漏掉普通续跑的第一个端点），之后访问点相隔
        λ（大步 Δs = 相邻访问位 σ 之差、velocity 在段起点评估），末段
        一律从最后位置直达 σ=0 终点。

        分块（#123 首跑 OOM 修复）：G 方向整批 × 大 FOV latent 的单次
        前向在 64 GiB 卡上是 OOM 级分配（实测 2G=20 即超），故超预算的
        批量按 ``_forward_chunk`` 切子批逐块续跑——ODE 逐样本独立，逐块
        与原语义一致（小形状不触发，fixture 口径逐位不变）。"""
        chunk = self._forward_chunk(latents.shape, latents.shape[0])
        if chunk >= latents.shape[0]:
            return self._continue_batch(latents, index, condition, stride)
        return torch.cat(
            [
                self._continue_batch(part, index, condition, stride)
                for part in latents.split(chunk, dim=0)
            ],
            dim=0,
        )

    def _forward_chunk(self, shape: torch.Size, batch: int) -> int:
        """ODE 续跑的分块上限（#123 首跑 OOM 修复）：单次 UNet 前向的激活
        显存随「前向样本数 × 空间体素数」增长（CFG 配对使前向样本 = 2 ×
        本分块），故分块 = 预算 // (2 × 系数 × 体素数)，截到 [1, batch]。

        ``shape`` = 待续跑 latent 的 [C, D, H, W]（消费面传张量形状）。"""
        voxels = int(shape[-3]) * int(shape[-2]) * int(shape[-1])
        cap = self._forward_budget // (
            2 * _ACTIVATION_BYTES_PER_LATENT_VOXEL * voxels
        )
        return max(1, min(batch, cap))

    def _continue_batch(
        self,
        latents: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        stride: int,
    ) -> torch.Tensor:
        """单块续跑（分块调度在 ``continue_to_terminal``；本方法只负责逐块积分）。"""
        cursor = self._schedules.cursor(condition.name)
        x = latents
        current = index + 1
        if stride > 1 and current < cursor.num_steps:
            velocity = self._field.velocity(
                x, cursor.timestep(current), condition,
            )
            x = self._deterministic.transition(
                x,
                velocity,
                cursor.sigma_level(current),
                cursor.delta_s(current),
            ).sample
            current += 1
        while current < cursor.num_steps:
            velocity = self._field.velocity(
                x, cursor.timestep(current), condition,
            )
            x = self._deterministic.transition(
                x,
                velocity,
                cursor.sigma_level(current),
                cursor.delta_s(current, stride),
            ).sample
            current += stride
        return x

    def _group_transition(
        self,
        x_k: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        group_size: int,
        cursor: TrajectoryCursor,
        noise: torch.Tensor | None = None,
    ) -> SdeTransition:
        """共享 anchor 的组内转移：velocity 走全组复用评估，核参数取自
        同一日程位（扰动步与 log-prob 重算共用，保证两侧口径一致）。"""
        velocity = self._field.group_velocity(
            x_k, cursor.timestep(index), condition, group_size,
        )
        return self._kernel.transition(
            x_k.expand(group_size, *x_k.shape[1:]),
            velocity,
            cursor.sigma_level(index),
            cursor.delta_s(index),
            noise=noise,
        )

    def _deterministic_step(
        self,
        x: torch.Tensor,
        index: int,
        condition: RolloutCondition,
        cursor: TrajectoryCursor,
    ) -> torch.Tensor:
        velocity = self._field.velocity(x, cursor.timestep(index), condition)
        return self._deterministic.transition(
            x,
            velocity,
            cursor.sigma_level(index),
            cursor.delta_s(index),
        ).sample
