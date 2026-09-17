"""eval() + no_grad 的 Rollout 与打分相（spec #15 执行序第 1 相）。

单条件组的完整 rollout：同组共享初始噪声 → Anchor 轨迹（η=0 逐步存
latent）→ 每个被优化训练步 k 单步 SDE 扰动 G 方向 → 各 Granularity λ
ODE 续跑到终点 → 判别器 raw real-logit 打分 → π_old 记录。

条件分布按组与数据域定义、均匀采样（experiment-design）：组1 BraTS =
ModalLabelConditionSampler（四序列均匀）；组2 = CrossModalCondition
Sampler（12 有序对均匀，源影像 latent 按 real sample pool 的序列分层
抽取）；MR-RATE 组1 = MrConditionSampler（词汇表生成条件均匀轮转，
spec #125 决策 5）。RolloutPhase 经构造注入条件分布——rollout 编排
本身组无关、域无关。

latent 形状与 sigma 日程逐条件贯通（#129，spec #125 实现决策 2）：
rollout 初始噪声、扰动噪声的形状从批次条件经 ``ConditionVocabulary``
解析（批内同条件即同形状；GRPO 组内天然同条件同形状，advantage 无
跨形状问题）；sigma 日程由 RolloutSampler 按条件名经 ``Condition
Schedules`` 选择（锚 = 该条件空间 numel）。BraTS 单域 = 单条件词汇
特例：任意条件恒 config ``latent_shape``、共用一份全局锚日程。

ADR-0008-01：base 分区种子按每条件配额量产（``base_condition_quota``）
——``ConditionSampler.sample_target`` 按指定目标条件构造条件（组2 的
源影像自由度仍在合法源上均匀），量产产出逐样本目标条件标签
（fill_base 的标签输入）。

数值口径：采样（Anchor/扰动/续跑的 policy 前向）进 bf16 autocast（与
更新相同口径，保证 π_old 可被逐位重算）；判别器打分在 autocast 外
fp32（T05 已锚定的 reward 数值口径）。
"""

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch

from cynosure.conditions import ConditionVocabulary
from cynosure.config import CynosureConfig, MODALITIES
from cynosure.policy.condition import (
    CONDITION_SPACING_X1E2,
    ModalityMapping,
    RolloutCondition,
)
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.scorer import LatentScorer

_BASE_BATCH = 8
"""base 分区种子生成的 rollout 批量（CFG 组合场 = 2×batch 前向）；
基准口径 = 64³ 空间网格（BraTS 单域锚）。大网格条件经
``RolloutPhase._base_batch_for`` 按空间体积缩批——量产前向的激活
显存随 batch × 体积增长，多网格域（MR-RATE 逐条件统一网格，#111
裁决）直接套单域常数会把大网格条件推向 OOM（#122 首跑集群实录：
t1w/coronal [4,128,64,128] batch=8 前向单次分配 12 GB）。"""


@dataclass(frozen=True)
class StepRollout:
    """一个被优化训练步 k 的 rollout 记录（MGAI advantage 与逐 k 更新的输入）。

    域契约：``anchor_latent``/``directions`` 保持 policy（scaled 采样）
    域——更新相 log-prob 重算在此域消费；``rewards`` 已归位 real pool
    存储域（``RolloutPhase._to_pool_domain``），与 ``IterationRollout.
    new_fakes`` 同域。
    """

    step_index: int
    anchor_latent: torch.Tensor
    """该步的 Anchor latent x_k（batch=1，更新相重算的采样场输入）。"""
    directions: torch.Tensor
    """单步 SDE 扰动的 G 方向 x_{k+1}。"""
    old_log_probs: torch.Tensor
    """rollout 时记录的 π_old（各自采样场口径，更新相逐位重算的对照）。"""
    rewards: dict[int, torch.Tensor]
    """Granularity λ → 组内 G 方向的 terminal reward（raw real-logit，
    pool 存储域）。"""


@dataclass(frozen=True)
class IterationRollout:
    """一个 RL iteration 的 rollout 相产出（eval + no_grad 的完整记录）。"""

    condition: RolloutCondition
    modality: str
    """本 iteration 采样的目标条件键（BraTS = 目标序列名；MR-RATE =
    生成条件名，如 t1w/axial）——iter 事件按目标条件归因 reward/loss/
    AUC 的依据（字段名沿事件契约「可扩不可改名」保留）。"""
    anchor_eval_reward: float
    """Anchor 全 ODE 终点的判别器 reward（训练曲线信号，不参与 loss）。"""
    steps: list[StepRollout]
    """按 M 升序排列的逐步记录。"""
    new_fakes: torch.Tensor
    """本 iteration 的全部新 fake（各 (k, λ) 终点 + Anchor 终点），判别器
    Online update 与 held-out AUC 的 fake 侧输入。域：已除
    ``policy.latent_scale_factor``，即 real pool 存储域（与 real 装载
    值同域比较；rollout 原始产出在 policy 域，见 StepRollout）。"""
    intra_group_reward_std: float
    """组内 reward std（各 (k, λ) 组内 std 的均值）——非退化观测面。"""


class ConditionSampler(Protocol):
    """组条件分布的策略接口（experiment-design「条件分布按组与域定义、
    均匀采样」）。

    ``sample()`` 连同采中的目标条件键返回——iter 事件按目标条件归因
    健康指标（per-condition 健康监控）的依据。条件键即
    ``ConditionVocabulary`` 的域名（BraTS = 序列名；MR-RATE = 生成
    条件名）。
    """

    def sample(
        self, generator: torch.Generator | None = None,
    ) -> tuple[RolloutCondition, str]:
        """均匀采一个条件的 rollout 条件（batch=1，采样场负责广播）。

        ``generator`` 缺省用实现自身的主流；base 分区种子生成传独立流
        （不漂移训练 rollout 的抽样流，各组实现同一约定）。"""
        ...

    def sample_target(
        self, target: str, generator: torch.Generator | None = None,
    ) -> RolloutCondition:
        """按指定目标条件构造 rollout 条件（ADR-0008-01：base 分区
        配额量产的条件源——配额决定目标端分布，条件内的其余自由度
        仍按本组分布均匀抽取）。

        ``generator`` 缺省用实现自身的主流；base 分区种子生成传独立流。"""
        ...

    def targets(self) -> tuple[str, ...]:
        """本组条件分布的目标端全集（ADR-0008-04：预训练 per-condition
        轮转调度的条件枚举——集合知识归条件分布自身，driver 不从
        config 复制按组分派；顺序确定性，轮转序由此而来）。"""
        ...


class ModalLabelConditionSampler:
    """组1 条件分布（BraTS）：四序列均匀采样（experiment-design「条件
    分布按组定义」）。

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

    def sample(
        self, generator: torch.Generator | None = None,
    ) -> tuple[RolloutCondition, str]:
        """均匀采一个序列的 rollout 条件（label batch=1，组合场负责广播），
        连同采中的序列名返回——iter 事件按目标序列归因健康指标的依据。
        随机数经 CPU generator 生成（跨设备可复现的 fixture「固定 seed」
        语义）后迁移到 rollout 设备；``generator`` 缺省用主流，base 分区
        种子生成传独立流（不漂移训练 rollout 的抽样流）。"""
        stream = generator if generator is not None else self._generator
        index = int(torch.randint(len(MODALITIES), (1,), generator=stream))
        label = self._mapping.label(MODALITIES[index])
        return (
            RolloutCondition(
                label=torch.tensor([label], device=self._device),
                spacing=torch.tensor([CONDITION_SPACING_X1E2], device=self._device),
                name=MODALITIES[index],
            ),
            MODALITIES[index],
        )

    def sample_target(
        self, target: str, generator: torch.Generator | None = None,
    ) -> RolloutCondition:
        """组1 的条件无其余自由度：label 恒为 target 的映射值，不耗 RNG。"""
        label = self._mapping.label(target)
        return RolloutCondition(
            label=torch.tensor([label], device=self._device),
            spacing=torch.tensor([CONDITION_SPACING_X1E2], device=self._device),
            name=target,
        )

    def targets(self) -> tuple[str, ...]:
        """组1 目标端全集 = 四序列固定序（experiment-design 的条件分布
        定义；轮转序 = 此序，确定性）。"""
        return tuple(MODALITIES)


class MrConditionSampler:
    """MR-RATE 组1 条件分布：条件词汇表生成条件均匀轮转（spec #125
    决策 5 默认口径——rollout 条件分布默认均匀轮转，稀疏条件加权为
    config knob 默认关，与预训练轮转同口径）。

    条件 c = (条件 token, 等效 spacing ×1e2, 条件名)——token/spacing
    都是条件属性（token 按模态派生、平面不分化；spacing = 统一网格
    下的等效物理分辨率，spec #125 决策 6——real 侧与 fake 侧条件张量
    同值的同源要求），条件名贯通 sigma 日程选择与噪声形状解析（批内
    同条件即同形状，#129）。词表装载产物经构造注入（工件是唯一来源，
    装载面之外不复制词表数据）。
    """

    def __init__(
        self,
        vocabulary: ConditionVocabulary,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        self._vocabulary = vocabulary
        self._generator = generator
        self._device = device

    def sample(
        self, generator: torch.Generator | None = None,
    ) -> tuple[RolloutCondition, str]:
        """均匀采一个生成条件的 rollout 条件（轮转序 = 词汇表登记序，
        确定性），连同条件名返回。随机流语义同 ModalLabelCondition
        Sampler（缺省主流、base 分区独立流）。"""
        stream = generator if generator is not None else self._generator
        names = self._vocabulary.names()
        index = int(torch.randint(len(names), (1,), generator=stream))
        name = names[index]
        return self._condition(name), name

    def sample_target(
        self, target: str, generator: torch.Generator | None = None,
    ) -> RolloutCondition:
        """按指定生成条件构造条件（base 分区配额量产入口）：条件的
        token/spacing 都是条件属性、无其余自由度，不耗 RNG。未知条件
        名即拒绝（词汇表装载面守卫的取数前置）。"""
        return self._condition(target)

    def targets(self) -> tuple[str, ...]:
        """MR-RATE 目标端全集 = 词汇表条件集（轮转序 = 登记序）。"""
        return self._vocabulary.names()

    def _condition(self, name: str) -> RolloutCondition:
        """条件名 → rollout 条件：token 与 spacing 从条件五元组取数
        （单一来源：词汇表工件经装载产物，协议取数面）。"""
        return RolloutCondition(
            label=torch.tensor(
                [self._vocabulary.token(name)], device=self._device,
            ),
            spacing=torch.tensor(
                [self._vocabulary.spacing_condition(name)], device=self._device,
            ),
            name=name,
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
        self._entries: dict[str, list[PoolEntry]] = {
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

    def size(self, modality: str) -> int:
        """该序列的条目数（均匀抽样的总体）。"""
        return len(self._entries[modality])

    def spacing(self, modality: str, index: int) -> tuple[float, float, float]:
        """该条目的 per-case spacing（manifest 侧车值原样透传，×1e2 条件
        单位；issue #46：组2 源影像条件的 spacing 与源 latent 同条目同源）。"""
        return self._entries[modality][index].spacing

    def latent(self, modality: str, index: int) -> torch.Tensor:
        """按序列取第 index 枚预编码 latent（[C, D, H, W]，已迁移到
        rollout 设备；懒加载与判别器 real 侧同一装载契约）。"""
        entry = self._entries[modality][index]
        return self._manifest.load_latent(entry).to(self._device)


class CrossModalConditionSampler:
    """组2 条件分布：四序列 12 有序 src→tgt 对均匀采样（experiment-design）。

    条件 c = (源影像 latent, 源序列 label, 目标序列 label, spacing)——
    两个 label 位各收其职（issue #115：ControlNet 收源、UNet 收目标）；
    源影像 latent 按源序列从 SourceLatentPool 均匀抽取，scale_factor 缩放
    发生在组2 采样场（条件的唯一缩放点）；spacing 为源影像条目的 manifest
    per-case 侧车值（issue #46：条件来自数据而非写死常量，与源 latent 同
    条目同源）——源 latent、源 spacing、源 label 同源于同一源模态条目。
    ``pairs`` 来自 config（cross_modal_pairs 可配置），不设代码内副本。"""

    def __init__(
        self,
        mapping: ModalityMapping,
        pairs: list[tuple[str, str]],
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

    def sample(
        self, generator: torch.Generator | None = None,
    ) -> tuple[RolloutCondition, str]:
        """均匀采一个有序对（目标 label batch=1 + 源影像 latent batch=1），
        连同目标序列名返回——iter 事件按目标序列归因健康指标的依据；
        ``generator`` 缺省用主流（base 分区种子生成传独立流）。"""
        stream = generator if generator is not None else self._generator
        pair_index = int(torch.randint(len(self._pairs), (1,), generator=stream))
        source_modality, target_modality = self._pairs[pair_index]
        return (
            self._condition_for(source_modality, target_modality, stream),
            target_modality,
        )

    def sample_target(
        self, target: str, generator: torch.Generator | None = None,
    ) -> RolloutCondition:
        """目标端固定为 ``target``（ADR-0008-01 配额量产的条件源），
        源序列自由度按组2 分布在「目标端为 target 的有序对」上均匀
        抽取；注入清单无该目标端的有序对时显式拒绝（cross_modal_pairs
        可配置，不静默回退全目标采样）。"""
        stream = generator if generator is not None else self._generator
        candidates = [pair for pair in self._pairs if pair[1] == target]
        if not candidates:
            raise ValueError(
                f"组2 条件分布的有序对清单无目标端为 {target} 的对"
                "（sample_target 是配额量产的条件源，清单来自 "
                "cross_modal_pairs 配置）"
            )
        pair_index = int(torch.randint(len(candidates), (1,), generator=stream))
        source_modality, _ = candidates[pair_index]
        return self._condition_for(source_modality, target, stream)

    def targets(self) -> tuple[str, ...]:
        """组2 目标端全集 = 有序对清单的目标端去重保序（cross_modal_pairs
        可配置：清单不产的目标端不在预训练轮转集——条件分布不产的
        条件不参与 per-condition 归因）。"""
        return tuple(dict.fromkeys(target for _, target in self._pairs))

    def _condition_for(
        self,
        source_modality: str,
        target_modality: str,
        stream: torch.Generator,
    ) -> RolloutCondition:
        """按 (源序列, 目标序列) 构造组2 条件：源影像 latent 按源序列
        均匀抽取；per-case spacing 与源 latent 同条目同源（manifest 侧车，
        issue #46），控制网络条件路径的消费端取值；双 label 按 (源, 目标)
        对同时产出（issue #115：源 label 随 ControlNet、目标 label 随
        UNet），三个源位与 label 同源于同一 (源, 目标) 对。"""
        source_index = int(torch.randint(
            self._pool.size(source_modality), (1,), generator=stream,
        ))
        return RolloutCondition(
            label=torch.tensor(
                [self._mapping.label(target_modality)], device=self._device,
            ),
            spacing=torch.tensor(
                [self._pool.spacing(source_modality, source_index)],
                device=self._device,
            ),
            source_latent=self._pool.latent(source_modality, source_index).unsqueeze(0),
            source_label=torch.tensor(
                [self._mapping.label(source_modality)], device=self._device,
            ),
            name=target_modality,
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
        vocabulary: ConditionVocabulary,
        device_type: str = "cpu",
        autocast_dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device("cpu"),
        base_generator: torch.Generator | None = None,
    ) -> None:
        self._config = config
        self._sampler = sampler
        self._scorer = scorer
        self._generator = generator
        self._device_type = device_type
        self._amp_dtype = autocast_dtype
        self._device = device
        self._condition_sampler = condition_sampler
        self._vocabulary = vocabulary
        # base 分区种子生成的独立流：其抽取数随 replay_buffer_capacity
        # 变化，与训练 rollout 共流会让 buffer 容量实验漂移 policy 样本流
        self._base_generator = base_generator

    @property
    def vocabulary(self) -> ConditionVocabulary:
        """本运行时的条件词汇表（rollout 形状解析与续训 buffer 逐条目
        对账的共同取数面，#129）。"""
        return self._vocabulary

    def run_iteration(self) -> IterationRollout:
        """单条件组的完整 rollout 与打分（执行序第 1 相的单进程版）。"""
        condition, condition_name = self._condition_sampler.sample()
        shape = self._vocabulary.latent_shape(condition_name)
        with torch.no_grad(), torch.autocast(self._device_type, dtype=self._amp_dtype):
            noise = torch.randn(
                (1, *shape), generator=self._generator,
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
        # fake 域归一（T12 探针定谳）：rollout 终点在 checkpoint scaled
        # 采样域，real pool 按 data-preparation 契约存 encode 原始输出
        # （seeded 后验采样）——打分与入 buffer 前归位 pool 域（见
        # _to_pool_domain），判别器比较两侧同域。打分输入与 new_fakes
        # 的消费面（Online update / 回放 / AUC fake 侧）因此域一致。
        with torch.no_grad():  # 打分是 inference（autocast 外、fp32、无图）
            for step_index, x_k, directions, old_log_probs, terminals in sampled:
                rewards = {
                    lam: self._scorer.reward(self._to_pool_domain(latents))
                    for lam, latents in terminals.items()
                }
                steps.append(StepRollout(
                    step_index=step_index,
                    anchor_latent=x_k,
                    directions=directions,
                    old_log_probs=old_log_probs,
                    rewards=rewards,
                ))
                fakes.extend(self._to_pool_domain(latents) for latents in terminals.values())
                std_sum += sum(rewards.std().item() for rewards in rewards.values())
                std_count += len(rewards)
            anchor_eval_reward = float(
                self._scorer.reward(self._to_pool_domain(anchor_terminal))[0]
            )
        fakes.append(self._to_pool_domain(anchor_terminal))
        return IterationRollout(
            condition=condition,
            modality=condition_name,
            anchor_eval_reward=anchor_eval_reward,
            steps=steps,
            new_fakes=torch.cat(fakes),
            intra_group_reward_std=std_sum / std_count,
        )

    def base_partition_samples(
        self, quota: Mapping[str, int],
    ) -> tuple[list[torch.Tensor], list[str]]:
        """冻结初始 policy 的 rollout 产出（Anchor 全 ODE 终点）——
        buffer base 分区的种子（train 启动时自动生成，spec 补钉）。

        ADR-0008-01：按每条件配额量产（``base_condition_quota``）——
        逐目标条件产满配额（条件键 = 序列名/生成条件名），产出逐样本
        目标条件标签（fill_base 的标签输入；组2 条目按目标端归因）。
        各条件的噪声形状从条件键经词汇表解析；**cat 仅同条件批内发生**
        （跨条件异形状无从 cat，#129），返回值 = 逐条目张量清单 +
        逐条条件标签（与 ``fill_base`` 的消费面对齐）。走独立 base 流
        （构造注入 base_generator）：其抽取数随 buffer 容量/配额变化，
        不占训练 rollout 的抽样流（同 seed 下容量实验的 rollout 流
        保持不变）。"""
        if not quota:
            raise ValueError("base 分区量产的每条件配额不得为空")
        generator = (
            self._base_generator
            if self._base_generator is not None else self._generator
        )
        latents: list[torch.Tensor] = []
        condition_names: list[str] = []
        with torch.no_grad(), torch.autocast(self._device_type, dtype=self._amp_dtype):
            for condition_name, count in quota.items():
                shape = self._vocabulary.latent_shape(condition_name)
                produced = 0
                while produced < count:
                    batch = min(
                        self._base_batch_for(shape), count - produced,
                    )
                    condition = self._condition_sampler.sample_target(
                        condition_name, generator,
                    )
                    noise = torch.randn(
                        (batch, *shape), generator=generator,
                    ).to(self._device)
                    anchor = self._sampler.anchor_trajectory(noise, condition)
                    terminal = self._to_pool_domain(anchor[-1])
                    latents.extend(terminal[i] for i in range(batch))
                    condition_names.extend([condition_name] * batch)
                    produced += batch
        # base 分区与近期分区同一 reward 域（real pool 存储域）
        return latents, condition_names

    @staticmethod
    def _base_batch_for(
        shape: tuple[int, int, int, int], reference: int = _BASE_BATCH,
    ) -> int:
        """条件形状的量产批量（体积感知缩放，#122 首跑 OOM 修复）：
        基准 = 64³ 空间（BraTS 单域锚）× ``_BASE_BATCH`` 批；实际批 =
        基准批 × (基准体积 / 本条件空间体积) 截到 [1, 基准]——前向激活
        显存随 batch × 体积增长，缩批以单次前向体积近似守恒；小网格
        只截顶、不放大（基准批量本身是 CFG 双前向的标定口径）。"""
        reference_spatial = 64 * 64 * 32
        spatial = shape[1] * shape[2] * shape[3]
        return max(1, min(reference, reference * reference_spatial // spatial))

    def _to_pool_domain(self, latent: torch.Tensor) -> torch.Tensor:
        """rollout 终点（policy scaled 采样域）→ real pool 存储域：
        除 ``policy.latent_scale_factor``（``LatentDecoder`` 解码前除回
        同构）——判别器比较与 buffer 存取的单一归一点。"""
        return latent / self._config.policy.latent_scale_factor

    def _perturb(
        self,
        anchor: list[torch.Tensor],
        step_index: int,
        condition: RolloutCondition,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """单步 SDE 扰动 G 方向 + 各 λ ODE 续跑到终点（autocast 口径）。

        扰动噪声形状随本批条件（词汇表解析，#129——与初始噪声、sigma
        日程同一条件键）。η=0 无策略密度（perturb_group 的 log-prob
        为 None）：训练循环在装配期已拒绝 η=0，此处防御性兜底为显式
        错误。"""
        policy = self._config.policy
        shape = self._vocabulary.latent_shape(condition.name_or_raise())
        noise = torch.randn(
            (policy.group_size_g, *shape),
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
