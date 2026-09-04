"""两区 Replay buffer（reward-model 章「在线更新机制」）。

封顶 FIFO 回放缓冲 = 固定 base 分区（初始冻结 policy 产出，填满即锁）
+ FIFO 近期分区（新 fake 滚动挤出最老），防判别器随 policy 变好而
灾难性遗忘「明显假」长什么样。容量对半切分（奇数余数归近期分区——
近当前分布一侧多一张）；回放采样在 base / recent 间均匀分配（各取一半、
奇数余数归 recent），某区样本不足时由另一区补足（训练首步 recent 为空，
回放全量由 base 承担——spec 未明文的唯一可行退化路径，测试固化其行为）。
"""

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ZoneSizes:
    """两区当前占用的观测记录。"""

    base: int
    recent: int


@dataclass(frozen=True)
class ReplayDraw:
    """一次回放采样的结果：样本批 + 两区来源数（混采占比审计数据）。"""

    samples: torch.Tensor
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

    def fill_base(self, latents: torch.Tensor) -> None:
        """初始冻结 policy 产出填充 base 分区。"""
        ...

    def push(self, latents: torch.Tensor) -> None:
        """新 fake 入近期分区。"""
        ...

    def sample_replay(self, count: int, generator: torch.Generator) -> ReplayDraw:
        """回放采样（两区混采）。"""
        ...


class ReplayBuffer:
    """fake latent 的两区回放缓冲：base 分区固定 + recent 分区 FIFO。"""

    def __init__(self, capacity: int) -> None:
        if capacity < 2:
            raise ValueError(
                f"Replay buffer 容量须 ≥2（两区各至少 1），得到 {capacity}"
            )
        self._base_capacity = capacity // 2
        self._recent_capacity = capacity - self._base_capacity
        self._base: list[torch.Tensor] = []
        self._recent: deque[torch.Tensor] = deque(maxlen=self._recent_capacity)

    @property
    def base_capacity(self) -> int:
        return self._base_capacity

    @property
    def recent_capacity(self) -> int:
        return self._recent_capacity

    def zone_sizes(self) -> ZoneSizes:
        """两区当前占用（诊断/测试观测面）。"""
        return ZoneSizes(base=len(self._base), recent=len(self._recent))

    def base_samples(self) -> list[torch.Tensor]:
        """base 分区当前内容快照（只读观测面）。"""
        return list(self._base)

    def recent_samples(self) -> list[torch.Tensor]:
        """recent 分区当前内容快照（只读观测面，插入序）。"""
        return list(self._recent)

    def fill_base(self, latents: torch.Tensor) -> None:
        """初始冻结 policy 产出填满 base 分区（一次性）。

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
        self._base = [
            latents[index].detach().clone() for index in range(self._base_capacity)
        ]

    def push(self, latents: torch.Tensor) -> None:
        """新 fake 入近期分区（FIFO：超容自动挤出最老）。"""
        self._recent.extend(t.detach().clone() for t in latents)

    def sample_replay(
        self, count: int, generator: torch.Generator,
    ) -> ReplayDraw:
        """回放采样：base / recent 均匀分配（奇数余数归 recent）。

        某区样本不足时由另一区补足；两区总计不足需求时显式拒绝
        （不静默重复采样）。
        """
        if count < 1:
            raise ValueError(f"回放采样数须 ≥1，得到 {count}")
        base_take, recent_take = self._split_counts(count)
        if base_take + recent_take < count:
            raise ValueError(
                f"回放样本不足：需求 {count}，缓冲仅 "
                f"{len(self._base) + len(self._recent)} 条"
            )
        base_idx = torch.randperm(len(self._base), generator=generator)[:base_take]
        recent_idx = torch.randperm(
            len(self._recent), generator=generator,
        )[:recent_take]
        samples = [self._base[i] for i in base_idx.tolist()]
        samples += [self._recent[i] for i in recent_idx.tolist()]
        return ReplayDraw(
            samples=torch.stack(samples), num_base=base_take, num_recent=recent_take,
        )

    def _split_counts(self, count: int) -> tuple[int, int]:
        """回放需求在 base / recent 间的分配（各自无放回、可互相补足）。"""
        base_take = min(count // 2, len(self._base))
        recent_take = min(count - count // 2, len(self._recent))
        while base_take + recent_take < count:
            if base_take < len(self._base):
                base_take += 1
            elif recent_take < len(self._recent):
                recent_take += 1
            else:
                break  # 两区耗尽：由 sample_replay 的容量校验显式拒绝
        return base_take, recent_take
