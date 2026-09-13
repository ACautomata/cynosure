"""两区 Replay buffer（reward-model 章「在线更新机制」+ ADR-0008 决策 2/4）。

封顶 FIFO 回放缓冲 = 固定 base 分区（初始冻结 policy 产出，填满即锁）
+ FIFO 近期分区（新 fake 滚动挤出最老），防判别器随 policy 变好而
灾难性遗忘「明显假」长什么样。容量对半切分（奇数余数归近期分区——
近当前分布一侧多一张）；回放采样在 base / recent 间均匀分配（各取一半、
奇数余数归 recent），某区样本不足时由另一区补足（训练首步 recent 为空，
回放全量由 base 承担——spec 未明文的唯一可行退化路径，测试固化其行为）。

条目带目标模态标签（ADR-0008 决策 2）：rollout 单 iteration 单条件，
整批 fake 同标签；组2 跨模态条目按**目标模态**归因（source 影像不在
Real sample pool，配对语义不存在）。回放采样可按本 iteration 条件过滤：
该条件候选充足则在该条件内维持两区各半、可互补语义；不足则显式拒绝
（可区分「条件不足」与「总数不足」，绝不静默回退全池混采）。base 分区
种子按每条件配额量产（``base_condition_quota``），装配期守卫每条件
配额 ≥ 回放半区需求（``assert_replay_supply``，train 装配与预训练
driver 同口径，ADR-0008 决策 4）。
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

import torch

from cynosure.config import MODALITIES, Modality, RewardConfig


@dataclass(frozen=True)
class ReplayEntry:
    """单条回放条目：fake latent + 目标模态标签（ADR-0008 决策 2）。"""

    latent: torch.Tensor
    modality: Modality
    """目标模态标签——组2 跨模态条目按目标模态归因（ADR-0008 决策 1）。"""


@dataclass(frozen=True)
class ZoneSizes:
    """两区当前占用的观测记录。"""

    base: int
    recent: int


@dataclass(frozen=True)
class ZoneModalities:
    """两区占用的按条件观测面：每目标模态 × 两区条目数（ADR-0008
    的 zone 标签观测——回放条件充足性的审计数据）。"""

    base: dict[Modality, int]
    recent: dict[Modality, int]


@dataclass(frozen=True)
class ReplayDraw:
    """一次回放采样的结果：样本批 + 逐样本目标模态标签（与行对齐）
    + 两区来源数（混采占比审计数据）。"""

    samples: torch.Tensor
    modalities: list[Modality]
    num_base: int
    num_recent: int


class ReplayStore(Protocol):
    """fake latent 回放存库的策略接口（glossary「Replay buffer」）。

    Online update 依赖本接口而非具体两区实现。
    """

    @property
    def base_capacity(self) -> int:
        """base 分区容量（train 启动期 base 分区种子生成的数量依据）。"""
        ...

    def zone_sizes(self) -> ZoneSizes:
        """两区当前占用（诊断/测试观测面，iter 事件的 buffer 占比来源）。"""
        ...

    def zone_modalities(self) -> ZoneModalities:
        """两区占用的按条件观测面（每目标模态 × 两区条目数）。"""
        ...

    def condition_supply(self, modality: Modality) -> int:
        """该条件当前的全部回放候选数（两区合计）——Online update 的
        回放退化判定查询面（候选 < 回放半区需求 → 该步退化纯 current
        半区，ADR-0008-03），与 ``sample_replay`` 同走条件过滤。"""
        ...

    def base_samples(self) -> list[ReplayEntry]:
        """base 分区当前内容快照（只读观测面，条目带目标模态标签）。"""
        ...

    def recent_samples(self) -> list[ReplayEntry]:
        """recent 分区当前内容快照（只读观测面，插入序、条目带标签）。"""
        ...

    def fill_base(
        self, latents: torch.Tensor, modalities: Sequence[Modality],
    ) -> None:
        """初始冻结 policy 产出填充 base 分区（逐样本目标模态标签对齐）。"""
        ...

    def push(self, latents: torch.Tensor, modality: Modality) -> None:
        """新 fake 入近期分区（整批同条件：rollout 单 iteration 单条件）。"""
        ...

    def sample_replay(
        self, count: int, generator: torch.Generator,
        modality: Modality | None = None,
    ) -> ReplayDraw:
        """回放采样（两区混采；``modality`` 给定时候选收窄为该条件）。"""
        ...


def base_condition_quota(capacity: int) -> dict[Modality, int]:
    """base 分区每条件配额（参数 = buffer 总容量，与
    ``config.replay_buffer_capacity`` 同口径；内部对半取 base 容量后
    均匀分派到各目标模态，余数按 MODALITIES 顺序逐个 +1——确定性，
    配额和恒等于 base 容量，fill_base 按配额量产后恰好填满）。

    ADR-0008 决策 4 的量产依据：base 分区种子须每条件覆盖回放半区
    需求（装配守卫见 ``assert_replay_supply``），否则首个判别器更新
    在某条件上将无回放候选可用。
    """
    base_capacity = capacity // 2
    per, extra = divmod(base_capacity, len(MODALITIES))
    return {
        modality: per + (1 if index < extra else 0)
        for index, modality in enumerate(MODALITIES)
    }


def assert_replay_supply(config: RewardConfig) -> None:
    """装配期回放供给守卫（train 装配与预训练 driver 同口径，
    ADR-0008 决策 4）：无效组合在装配期显式拒绝，而非让昂贵 rollout
    先行、更新时才缺样本。

    两条线：回放半区非零（K ≥ 2）；base 分区每条件配额 ≥ 回放半区
    需求——首次判别器更新时近期分区为空，回放全量由 base 承担，按
    条件过滤后某条件的回放候选不得少于半区需求。
    """
    replay_count = config.disc_batch_size_k - math.ceil(
        config.disc_batch_size_k * config.replay_current_fraction,
    )
    if replay_count < 1:
        raise ValueError(
            f"回放供给不足：disc_batch_size_k={config.disc_batch_size_k}"
            " 的回放半区为 0 条（K 须 ≥2）"
        )
    quota = base_condition_quota(config.replay_buffer_capacity)
    weakest = min(quota.values())
    if weakest < replay_count:
        raise ValueError(
            f"回放供给不足：base 分区每条件配额 {weakest} 条"
            f"（replay_buffer_capacity={config.replay_buffer_capacity} → base "
            f"分区 {config.replay_buffer_capacity // 2} 条均匀分派 "
            f"{len(MODALITIES)} 目标模态）< 判别器更新回放半区 "
            f"{replay_count} 条（disc_batch_size_k="
            f"{config.disc_batch_size_k}）：ADR-0008 决策 4 装配期守卫——"
            "某条件回放候选不足半区需求，按条件过滤的回放将无米下锅；"
            "增大 replay_buffer_capacity 或减小 disc_batch_size_k"
        )


class ReplayBuffer:
    """fake latent 的两区回放缓冲：base 分区固定 + recent 分区 FIFO，
    条目带目标模态标签（ADR-0008 决策 2）。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError(
                f"Replay buffer 容量须 ≥2（两区各至少 1），得到 {capacity}"
            )
        self._base_capacity = capacity // 2
        self._recent_capacity = capacity - self._base_capacity
        self._base: list[ReplayEntry] = []
        self._recent: deque[ReplayEntry] = deque(maxlen=self._recent_capacity)

    @property
    def base_capacity(self) -> int:
        return self._base_capacity

    @property
    def recent_capacity(self) -> int:
        return self._recent_capacity

    def zone_sizes(self) -> ZoneSizes:
        """两区当前占用（诊断/测试观测面）。"""
        return ZoneSizes(base=len(self._base), recent=len(self._recent))

    def zone_modalities(self) -> ZoneModalities:
        """两区占用的按条件观测面（每目标模态 × 两区条目数）。"""
        return ZoneModalities(
            base=self._modality_counts(self._base),
            recent=self._modality_counts(self._recent),
        )

    def condition_supply(self, modality: Modality) -> int:
        """该条件当前的全部回放候选数（两区合计——退化判定查询面，
        ADR-0008-03）。"""
        base_pool, recent_pool = self._condition_candidates(modality)
        return len(base_pool) + len(recent_pool)

    def base_samples(self) -> list[ReplayEntry]:
        """base 分区当前内容快照（只读观测面，条目带目标模态标签）。"""
        return list(self._base)

    def recent_samples(self) -> list[ReplayEntry]:
        """recent 分区当前内容快照（只读观测面，插入序、条目带标签）。"""
        return list(self._recent)

    def fill_base(
        self, latents: torch.Tensor, modalities: Sequence[Modality],
    ) -> None:
        """初始冻结 policy 产出填满 base 分区（一次性，逐样本标签对齐）。

        不足容量或重复填充显式拒绝——base 分区固定语义是防遗忘的根基，
        不静默接受残缺或覆写。
        """
        if self._base:
            raise ValueError("base 分区已填满（固定语义：一次填充、之后不可变）")
        if latents.shape[0] < self._base_capacity:
            raise ValueError(
                f"base 分区容量 {self._base_capacity}，"
                f"初始 fake {latents.shape[0]} 条不足"
            )
        if len(modalities) != latents.shape[0]:
            raise ValueError(
                f"base 分区填充的标签清单 {len(modalities)} 条与样本数 "
                f"{latents.shape[0]} 条不符（逐样本一一对应）"
            )
        self._base = [
            ReplayEntry(latents[index].detach().clone(), modalities[index])
            for index in range(self._base_capacity)
        ]

    def push(self, latents: torch.Tensor, modality: Modality) -> None:
        """新 fake 入近期分区（FIFO：超容自动挤出最老；整批同条件——
        rollout 单 iteration 单条件）。"""
        self._recent.extend(
            ReplayEntry(latent.detach().clone(), modality)
            for latent in latents
        )

    def sample_replay(
        self, count: int, generator: torch.Generator,
        modality: Modality | None = None,
    ) -> ReplayDraw:
        """回放采样：base / recent 均匀分配（奇数余数归 recent）。

        ``modality`` 给定时候选收窄为该条件的条目（ADR-0008 决策 2：
        回放抽取按本 iteration 条件过滤），两区互补语义在该条件内不变；
        该条件候选不足需求时显式拒绝——可区分「条件不足」（全池够、
        该条件不够，点名条件与两区可用量）与「总数不足」（不过滤也
        不够），绝不静默回退全池混采。
        """
        if count < 1:
            raise ValueError(f"回放采样数须 ≥1，得到 {count}")
        base_pool, recent_pool = self._condition_candidates(modality)
        if len(base_pool) + len(recent_pool) < count:
            if modality is not None and (
                len(self._base) + len(self._recent) >= count
            ):
                raise ValueError(
                    f"条件 {modality} 的回放候选不足：base {len(base_pool)} 条 + "
                    f"recent {len(recent_pool)} 条 = "
                    f"{len(base_pool) + len(recent_pool)} 条 < 需求 {count}；"
                    "绝不静默回退全池混采（ADR-0008 决策 2：回放按本 "
                    "iteration 条件过滤，条件不足须显式暴露，由调用方决定"
                    "退化或拒绝）"
                )
            raise ValueError(
                f"回放样本不足：需求 {count}，缓冲仅 "
                f"{len(self._base) + len(self._recent)} 条"
            )
        base_take, recent_take = self._split_counts(
            count, len(base_pool), len(recent_pool),
        )
        base_idx = torch.randperm(len(base_pool), generator=generator)[:base_take]
        recent_idx = torch.randperm(
            len(recent_pool), generator=generator,
        )[:recent_take]
        entries = [base_pool[i] for i in base_idx.tolist()]
        entries += [recent_pool[i] for i in recent_idx.tolist()]
        return ReplayDraw(
            samples=torch.stack([entry.latent for entry in entries]),
            modalities=[entry.modality for entry in entries],
            num_base=base_take,
            num_recent=recent_take,
        )

    def _condition_candidates(
        self, modality: Modality | None,
    ) -> tuple[list[ReplayEntry], list[ReplayEntry]]:
        """两区候选：``modality`` 给定时各区收窄为该条件的条目。"""
        if modality is None:
            return list(self._base), list(self._recent)
        return (
            [entry for entry in self._base if entry.modality == modality],
            [entry for entry in self._recent if entry.modality == modality],
        )

    @staticmethod
    def _modality_counts(entries: Sequence[ReplayEntry]) -> dict[Modality, int]:
        counts: dict[Modality, int] = {}
        for entry in entries:
            counts[entry.modality] = counts.get(entry.modality, 0) + 1
        return counts

    @staticmethod
    def _split_counts(
        count: int, base_available: int, recent_available: int,
    ) -> tuple[int, int]:
        """回放需求在 base / recent 间的分配（各自无放回、可互相补足）。"""
        base_take = min(count // 2, base_available)
        recent_take = min(count - count // 2, recent_available)
        while base_take + recent_take < count:
            if base_take < base_available:
                base_take += 1
            elif recent_take < recent_available:
                recent_take += 1
            else:
                break  # 两区耗尽：由 sample_replay 的容量校验显式拒绝
        return base_take, recent_take
