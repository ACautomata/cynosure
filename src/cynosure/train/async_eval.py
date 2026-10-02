"""async 执行序的评测采样前向（#237 评测顺迁期，#217 评测三路径口径）。

三路径（Baseline 采样 / 里程碑解码评测 / RL 后重采）的 policy latent
采样段走新执行器：manifest 条目按槽分派（槽 = 调度轴，槽协程经绑卡
线程多路复用落卡），汇聚缝 = **异形 latent 逐条传卡 0 + 按 entry index
字典保序重组**（异条件条目 latent 形状各异、不可 cat，#129 形状按条件
贯通）；解码 / FID / 特征提取不进本模块——评测编排（``ManifestEvaluation``
三路径 + ``MilestoneEvaluator`` decode/fid 相位打点）经
``EntryLatentSampler`` 缝注入本实现后零改动。

逐位安全性（按槽分派不漂数值的机器面）：per-entry ``noise_seed`` 独立
generator（噪声与任何共享流位置无关，#218 评测 RNG 数值域分离假设）+
anchor 轨迹 η=0 确定性（``anchor_trajectory`` 纯前向、零随机流消耗）+
各卡副本权重步末逐位一致（#217 执行序不变式）→ 条目无论分派到哪个
槽/卡都得到同一 terminal；分派只影响墙钟，不影响数值。评测全程零消耗
训练 RNG 流与进程全局 RNG（训练路径禁碰进程全局 RNG 的既有不变式）。"""

from typing import TYPE_CHECKING

import torch

from cynosure.conditions import ConditionVocabulary
from cynosure.eval.condition import EntryConditionResolver
from cynosure.eval.sampling import EntrySample
from cynosure.policy.numerics import AmpContext
from cynosure.reward.artifacts import LatentManifest

if TYPE_CHECKING:
    from cynosure.train.artifacts import ManifestEntry
    from cynosure.train.executor import CardWorker


class SlotDispatchLatentSampler:
    """manifest 条目的 policy latent 采样前向（新执行序
    ``EntryLatentSampler`` 实现）：条目按槽轮转分派 → 每卡一个采样协程
    在绑卡线程逐条执行（no_grad + autocast 与训练 rollout 同数值口径）
    → 异形 latent 逐条迁卡 0 → 主线程按 entry index 字典保序重组。"""

    def __init__(
        self,
        cards: "list[CardWorker]",
        slot_count: int,
        resolvers: list[EntryConditionResolver],
        amps: list[AmpContext],
        vocabulary: ConditionVocabulary,
        gather_device: torch.device,
    ) -> None:
        self._cards = cards
        self._slot_count = slot_count
        self._resolvers = resolvers
        self._amps = amps
        self._vocabulary = vocabulary
        self._gather_device = gather_device

    @classmethod
    def assemble(
        cls,
        cards: "list[CardWorker]",
        slot_count: int,
        vocabulary: ConditionVocabulary,
        pool: LatentManifest,
        amps: list[AmpContext],
    ) -> "SlotDispatchLatentSampler":
        """门面装配单点：调用方（executor.build）单点装载词汇表、
        real pool 与数值口径后注入（评审：pool manifest 不双读、词表
        不重复装配，与 ``ManifestEvaluation.build`` 共享同一份）；本面
        只做 per-卡条件解析器构造与组装——条件 tensor / 前向都落本卡
        设备，跨卡复用单实例解析器会让条件张量与执行前向错设备。"""
        return cls(
            cards=cards,
            slot_count=slot_count,
            resolvers=[
                EntryConditionResolver(vocabulary, card.device, pool=pool)
                for card in cards
            ],
            amps=amps,
            vocabulary=vocabulary,
            gather_device=cards[0].device,
        )

    def sample(
        self, entries: list["ManifestEntry"],
    ) -> list[EntrySample]:
        """三路径共用的条目采样（``EntryLatentSampler`` 缝）：条目按槽
        round-robin 分派（评测分派不消耗分配表轮——分配表是训练 iteration
        的条件轴，评测条目分派只承担负载均衡），每卡一个采样协程提交绑
        卡线程；future 回传后按 entry index 字典重组回条目输入序。

        同步等待形态：评测路径在主线程同步调用（``MilestoneEvaluator`` /
        ``ManifestVolumeSampler`` 编排是同步方法），绑卡线程独立完成采样
        不依赖主线程推进——无死锁面；评测采样不套 barrier 超时口径（与
        rollout 相派发同为非 k 段），异常经 future 直接传播（评测路径无
        fail-fast 语义，收尾由门面 run 的 finally 承担）。"""
        per_card: dict[int, list["ManifestEntry"]] = {}
        for position, entry in enumerate(entries):
            card_index = (position % self._slot_count) % len(self._cards)
            per_card.setdefault(card_index, []).append(entry)
        by_index: dict[int, EntrySample] = {}
        futures = [
            card.submit(self._sample_on_card(
                card, per_card.get(card.index, []),
            ))
            for card in self._cards
        ]
        for future in futures:
            for sample in future.result():
                by_index[sample.entry.index] = sample
        return [by_index[entry.index] for entry in entries]

    async def _sample_on_card(
        self, card: "CardWorker", assigned: list["ManifestEntry"],
    ) -> list[EntrySample]:
        """本卡分派条目的采样段（绑卡线程执行）：逐条 resolve → per-entry
        独立 generator 生成噪声（CPU generator、跨设备可复现）→ 本卡
        ``anchor_trajectory`` 取终点 → **逐条**迁汇聚设备（卡 0；异形不可
        cat，逐条迁移不引入批组织）。与 ``ManifestLatentSampler.sample``
        同一调用序（resolve → 形状 → 噪声 → anchor → source_case 读取
        ——组2 源病例锁定的写回发生在 resolve 内，读取在其后）。"""
        replica = card.replica
        amp = self._amps[card.index]
        resolver = self._resolvers[card.index]
        samples: list[EntrySample] = []
        with torch.no_grad(), torch.autocast(
            amp.device_type, dtype=amp.dtype,
        ):
            for entry in assigned:
                condition, target = resolver.resolve(entry)
                shape = self._vocabulary.latent_shape(target)
                noise = torch.randn(
                    (1, *shape),
                    generator=torch.Generator().manual_seed(entry.noise_seed),
                ).to(replica.device)
                terminal = replica.sampler.anchor_trajectory(
                    noise, condition,
                )[-1]
                source_case = (
                    resolver.source_case(entry)
                    if not isinstance(entry.condition, str) else None
                )
                samples.append(EntrySample(
                    entry, target, source_case,
                    terminal.to(self._gather_device),
                ))
        return samples
