"""eval() + no_grad 的 Rollout 与打分相（spec #15 执行序第 1 相）。

单条件组的完整 rollout：同组共享初始噪声 → Anchor 轨迹（η=0 逐步存
latent）→ 每个被优化训练步 k 单步 SDE 扰动 G 方向 → 各 Granularity λ
ODE 续跑到终点 → 判别器 raw real-logit 打分 → π_old 记录。

数值口径：采样（Anchor/扰动/续跑的 policy 前向）进 bf16 autocast（与
更新相同口径，保证 π_old 可被逐位重算）；判别器打分在 autocast 外
fp32（T05 已锚定的 reward 数值口径）。
"""

from dataclasses import dataclass

import torch

from cynosure.config import CynosureConfig, MODALITIES
from cynosure.policy.condition import ModalityMapping, RolloutCondition
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.scorer import LatentScorer

_BASE_BATCH = 8
"""base 分区种子生成的 rollout 批量（CFG 组合场 = 2×batch 前向）。"""


@dataclass(frozen=True)
class StepRollout:
    """一个被优化训练步 k 的 rollout 记录（MGAI advantage 与逐 k 更新的输入）。"""

    step_index: int
    anchor_latent: torch.Tensor
    """该步的 Anchor latent x_k（batch=1，更新相重算的采样场输入）。"""
    directions: torch.Tensor
    """单步 SDE 扰动的 G 方向 x_{k+1}。"""
    old_log_probs: torch.Tensor
    """rollout 时记录的 π_old（各自采样场口径，更新相逐位重算的对照）。"""
    rewards: dict[int, torch.Tensor]
    """Granularity λ → 组内 G 方向的 terminal reward（raw real-logit）。"""


@dataclass(frozen=True)
class IterationRollout:
    """一个 RL iteration 的 rollout 相产出（eval + no_grad 的完整记录）。"""

    condition: RolloutCondition
    anchor_eval_reward: float
    """Anchor 全 ODE 终点的判别器 reward（训练曲线信号，不参与 loss）。"""
    steps: list[StepRollout]
    """按 M 升序排列的逐步记录。"""
    new_fakes: torch.Tensor
    """本 iteration 的全部新 fake（各 (k, λ) 终点 + Anchor 终点），判别器
    Online update 与 held-out AUC 的 fake 侧输入。"""
    intra_group_reward_std: float
    """组内 reward std（各 (k, λ) 组内 std 的均值）——非退化观测面。"""


class ModalLabelConditionSampler:
    """组1 条件分布：四序列均匀采样（experiment-design「条件分布按组定义」）。

    条件 c = (modality label, spacing)。spacing 生产语义为数据分布的体素
    间距（×1e2 恒传，基座 include_spacing_input=true）；fixture 无源数据
    分布，取单位间距。"""

    SPACING_X1E2: tuple[float, float, float] = (100.0, 100.0, 100.0)
    """fixture 单位间距（1.0 × 1e2）；生产 spacing 分布由数据侧 ticket 接管。"""

    def __init__(self, mapping: ModalityMapping, generator: torch.Generator) -> None:
        self._mapping = mapping
        self._generator = generator

    def sample(self) -> RolloutCondition:
        """均匀采一个序列的 rollout 条件（label batch=1，组合场负责广播）。"""
        index = int(torch.randint(len(MODALITIES), (1,), generator=self._generator))
        label = self._mapping.label(MODALITIES[index])
        return RolloutCondition(
            label=torch.tensor([label]),
            spacing=torch.tensor([self.SPACING_X1E2]),
        )


class RolloutPhase:
    """rollout 相编排：条件采样 → Anchor → 扰动 → λ 续跑 → 打分。"""

    def __init__(
        self,
        config: CynosureConfig,
        sampler: RolloutSampler,
        scorer: LatentScorer,
        generator: torch.Generator,
        device_type: str = "cpu",
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._scorer = scorer
        self._generator = generator
        self._device_type = device_type
        self._amp_dtype = autocast_dtype
        self._condition_sampler = ModalLabelConditionSampler(
            ModalityMapping.load(config.artifacts.modality_mapping_json),
            generator,
        )

    def run_iteration(self) -> IterationRollout:
        """单条件组的完整 rollout 与打分（执行序第 1 相的单进程版）。"""
        condition = self._condition_sampler.sample()
        with torch.no_grad(), torch.autocast(self._device_type, dtype=self._amp_dtype):
            noise = torch.randn(
                (1, *self._config.latent_shape), generator=self._generator,
            )
            anchor = self._sampler.anchor_trajectory(noise, condition)
            sampled = [
                (
                    step_index,
                    anchor[step_index],
                    *self._perturb(anchor, step_index, condition),
                )
                for step_index in sorted(self._config.policy.train_step_indices_m)
            ]
            anchor_terminal = anchor[-1]

        steps: list[StepRollout] = []
        fakes: list[torch.Tensor] = []
        std_sum = 0.0
        std_count = 0
        with torch.no_grad():  # 打分是 inference（autocast 外、fp32、无图）
            for step_index, x_k, directions, old_log_probs, terminals in sampled:
                rewards = {
                    lam: self._scorer.reward(latents)
                    for lam, latents in terminals.items()
                }
                steps.append(StepRollout(
                    step_index=step_index,
                    anchor_latent=x_k,
                    directions=directions,
                    old_log_probs=old_log_probs,
                    rewards=rewards,
                ))
                fakes.extend(terminals.values())
                std_sum += sum(rewards.std().item() for rewards in rewards.values())
                std_count += len(rewards)
            anchor_eval_reward = float(self._scorer.reward(anchor_terminal)[0])
        fakes.append(anchor_terminal)
        return IterationRollout(
            condition=condition,
            anchor_eval_reward=anchor_eval_reward,
            steps=steps,
            new_fakes=torch.cat(fakes),
            intra_group_reward_std=std_sum / std_count,
        )

    def base_partition_samples(self, total: int) -> torch.Tensor:
        """冻结初始 policy 的 rollout 产出（Anchor 全 ODE 终点）——
        buffer base 分区的种子（train 启动时自动生成，spec 补钉）。"""
        terminals: list[torch.Tensor] = []
        produced = 0
        with torch.no_grad(), torch.autocast(self._device_type, dtype=self._amp_dtype):
            while produced < total:
                count = min(_BASE_BATCH, total - produced)
                condition = self._condition_sampler.sample()
                noise = torch.randn(
                    (count, *self._config.latent_shape), generator=self._generator,
                )
                anchor = self._sampler.anchor_trajectory(noise, condition)
                terminals.append(anchor[-1])
                produced += count
        return torch.cat(terminals)

    def _perturb(
        self,
        anchor: list[torch.Tensor],
        step_index: int,
        condition: RolloutCondition,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """单步 SDE 扰动 G 方向 + 各 λ ODE 续跑到终点（autocast 口径）。

        η=0 无策略密度（perturb_group 的 log-prob 为 None）：训练循环在
        装配期已拒绝 η=0，此处防御性兜底为显式错误。"""
        policy = self._config.policy
        noise = torch.randn(
            (policy.group_size_g, *self._config.latent_shape),
            generator=self._generator,
        )
        directions, old_log_probs = self._sampler.perturb_group(
            anchor[step_index], step_index, condition, noise,
        )
        if old_log_probs is None:
            raise ValueError(
                "η=0 的扰动步无 π_old 可记录（训练循环须 η>0）",
            )
        terminals = {
            lam: self._sampler.continue_to_terminal(
                directions, step_index, condition, stride=lam,
            )
            for lam in sorted(policy.granularity_intervals_lambda)
        }
        return directions, old_log_probs, terminals
