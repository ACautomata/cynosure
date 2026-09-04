"""eval() + no_grad 的 Rollout 与打分相（spec #15 执行序第 1 相）。

单条件组的完整 rollout：同组共享初始噪声 → Anchor 轨迹（η=0 逐步存
latent）→ 每个被优化训练步 k 单步 SDE 扰动 G 方向 → 各 Granularity λ
ODE 续跑到终点 → 判别器 raw real-logit 打分 → π_old 记录。

条件分布按组定义、均匀采样（experiment-design）：组1 = ModalLabelCondition
Sampler（四序列均匀）；组2 = CrossModalConditionSampler（12 有序对均匀，
源影像 latent 按 real sample pool 的序列分层抽取）。RolloutPhase 经构造
注入条件分布——rollout 编排本身组无关。

数值口径：采样（Anchor/扰动/续跑的 policy 前向）进 bf16 autocast（与
更新相同口径，保证 π_old 可被逐位重算）；判别器打分在 autocast 外
fp32（T05 已锚定的 reward 数值口径）。
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from cynosure.config import CynosureConfig, MODALITIES, Modality
from cynosure.policy.condition import ModalityMapping, RolloutCondition
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.scorer import LatentScorer

_BASE_BATCH = 8
"""base 分区种子生成的 rollout 批量（CFG 组合场 = 2×batch 前向）。"""

CONDITION_SPACING_X1E2: tuple[float, float, float] = (100.0, 100.0, 100.0)
"""条件分布的体素间距（1.0 × 1e2，fixture 单位间距；生产 spacing 分布由
数据侧 ticket 接管）——两组条件分布共用同一常量（单一来源）。"""


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
    modality: str
    """本 iteration 采样的目标序列（条件分布均匀采样的目标端）——iter
    事件按目标序列归因 reward/loss/AUC 的依据。"""
    anchor_eval_reward: float
    """Anchor 全 ODE 终点的判别器 reward（训练曲线信号，不参与 loss）。"""
    steps: list[StepRollout]
    """按 M 升序排列的逐步记录。"""
    new_fakes: torch.Tensor
    """本 iteration 的全部新 fake（各 (k, λ) 终点 + Anchor 终点），判别器
    Online update 与 held-out AUC 的 fake 侧输入。"""
    intra_group_reward_std: float
    """组内 reward std（各 (k, λ) 组内 std 的均值）——非退化观测面。"""


class ConditionSampler(Protocol):
    """组条件分布的策略接口（experiment-design「条件分布按组定义、均匀采样」）。

    ``sample()`` 连同采中的目标序列名返回——iter 事件按目标序列归因
    健康指标（per-sequence 健康监控）的依据。
    """

    def sample(self) -> tuple[RolloutCondition, str]:
        """均匀采一个条件的 rollout 条件（batch=1，采样场负责广播）。"""
        ...


class ModalLabelConditionSampler:
    """组1 条件分布：四序列均匀采样（experiment-design「条件分布按组定义」）。

    条件 c = (modality label, spacing)。spacing 生产语义为数据分布的体素
    间距（×1e2 恒传，基座 include_spacing_input=true）；fixture 无源数据
    分布，取单位间距。"""

    def __init__(
        self,
        mapping: ModalityMapping,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self._mapping = mapping
        self._generator = generator
        self._device = device

    def sample(self) -> tuple[RolloutCondition, str]:
        """均匀采一个序列的 rollout 条件（label batch=1，组合场负责广播），
        连同采中的序列名返回——iter 事件按目标序列归因健康指标的依据。
        随机数经 CPU generator 生成（跨设备可复现的 fixture「固定 seed」
        语义）后迁移到 rollout 设备。"""
        index = int(torch.randint(len(MODALITIES), (1,), generator=self._generator))
        label = self._mapping.label(MODALITIES[index])
        return (
            RolloutCondition(
                label=torch.tensor([label], device=self._device),
                spacing=torch.tensor([CONDITION_SPACING_X1E2], device=self._device),
            ),
            MODALITIES[index],
        )


class SourceLatentPool:
    """组2 条件的源影像 latent 库（按源序列分层的均匀抽取）。

    源影像分布 = VAE 预编码 train split——工件复用 Real sample pool
    manifest（同一次 prepare 产出、experiment-design「组2 按 4 序列分层」），
    与判别器 real 侧共用工件、各自独立采样（policy 条件与 reward real 是
    两个消费方，不是同一份采样状态）。"""

    def __init__(self, manifest: LatentManifest, device: torch.device) -> None:
        self._manifest = manifest
        self._device = device
        self._entries: dict[Modality, list[PoolEntry]] = {
            modality: [] for modality in MODALITIES
        }
        for entry in manifest.entries:
            self._entries[entry.modality].append(entry)
        empty = [m for m, entries in self._entries.items() if not entries]
        if empty:
            raise ValueError(
                f"real pool manifest 缺少序列 {empty} 的条目"
                "（组2 源影像条件要求四序列全部分层非空）"
            )

    def size(self, modality: Modality) -> int:
        """该序列的条目数（均匀抽样的总体）。"""
        return len(self._entries[modality])

    def latent(self, modality: Modality, index: int) -> torch.Tensor:
        """按序列取第 index 枚预编码 latent（[C, D, H, W]，已迁移到
        rollout 设备；懒加载与判别器 real 侧同一装载契约）。"""
        entry = self._entries[modality][index]
        return self._manifest.load_latent(entry).to(self._device)


class CrossModalConditionSampler:
    """组2 条件分布：四序列 12 有序 src→tgt 对均匀采样（experiment-design）。

    条件 c = (源影像 latent, 目标序列 label, spacing)——两个条件都带
    （policy-modeling 章 MDP）；源影像 latent 按源序列从 SourceLatentPool
    均匀抽取，scale_factor 缩放发生在组2 采样场（条件的唯一缩放点）。
    ``pairs`` 来自 config（cross_modal_pairs 可配置），不设代码内副本。"""

    def __init__(
        self,
        mapping: ModalityMapping,
        pairs: list[tuple[Modality, Modality]],
        pool: SourceLatentPool,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        if not pairs:
            raise ValueError("组2 条件分布的有序对清单不得为空")
        self._mapping = mapping
        self._pairs = list(pairs)
        self._pool = pool
        self._generator = generator
        self._device = device

    def sample(self) -> tuple[RolloutCondition, str]:
        """均匀采一个有序对（目标 label batch=1 + 源影像 latent batch=1），
        连同目标序列名返回——iter 事件按目标序列归因健康指标的依据。"""
        pair_index = int(torch.randint(len(self._pairs), (1,), generator=self._generator))
        source_modality, target_modality = self._pairs[pair_index]
        source_index = int(torch.randint(
            self._pool.size(source_modality), (1,), generator=self._generator,
        ))
        label = self._mapping.label(target_modality)
        return (
            RolloutCondition(
                label=torch.tensor([label], device=self._device),
                spacing=torch.tensor([CONDITION_SPACING_X1E2], device=self._device),
                source_latent=self._pool.latent(source_modality, source_index).unsqueeze(0),
            ),
            target_modality,
        )


class RolloutPhase:
    """rollout 相编排：条件采样 → Anchor → 扰动 → λ 续跑 → 打分。"""

    def __init__(
        self,
        config: CynosureConfig,
        sampler: RolloutSampler,
        scorer: LatentScorer,
        generator: torch.Generator,
        condition_sampler: ConditionSampler,
        device_type: str = "cpu",
        autocast_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._scorer = scorer
        self._generator = generator
        self._device_type = device_type
        self._amp_dtype = autocast_dtype
        self._device = device
        self._condition_sampler = condition_sampler

    def run_iteration(self) -> IterationRollout:
        """单条件组的完整 rollout 与打分（执行序第 1 相的单进程版）。"""
        condition, modality = self._condition_sampler.sample()
        with torch.no_grad(), torch.autocast(self._device_type, dtype=self._amp_dtype):
            noise = torch.randn(
                (1, *self._config.latent_shape), generator=self._generator,
            ).to(self._device)
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
            modality=modality,
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
                condition = self._condition_sampler.sample()[0]
                noise = torch.randn(
                    (count, *self._config.latent_shape), generator=self._generator,
                ).to(self._device)
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
        ).to(self._device)
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
