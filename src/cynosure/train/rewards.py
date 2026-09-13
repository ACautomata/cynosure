"""判别器侧协作者组（Facade）：聚合两区缓冲、Online update 与 held-out
AUC——trainer 只面对「种植/更新/AUC」三个动作与判别器引用。

fake 供给与判别器相位约定：更新批的当前半区从全批 fake 随机抽取
（rollout 产出按 (k, λ) 有序堆叠，确定性取头部会使 K=4 的当前半区
永远只见最小 step、λ=1 的头部方向）；判别器默认保持 eval 相（打分
与监控前向不得推进 spectral norm power iteration），仅更新一步
期间短暂 train。

分布式（ADR-0003）：fake 批是本 rank rollout 产出、real 批来自 rank
切片 pool、梯度经 DDP allreduce 平均——本类的更新语义逐字不变。
"""

import torch

from cynosure.config import Modality
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayStore
from cynosure.reward.update import OnlineUpdate, UpdateReport


class RewardCoordinator:
    """判别器侧动作面（种植/更新/AUC）的单点持有。"""

    def __init__(
        self, update: OnlineUpdate, auc: HeldOutAuc,
        generator: torch.Generator,
    ) -> None:
        self.update = update
        self.auc = auc
        self._generator = generator

    @property
    def buffer(self) -> "ReplayStore":
        """两区回放缓冲（Online update 的混采源，单点持有）。"""
        return self.update.buffer

    @property
    def discriminator(self) -> torch.nn.Module:
        """底层判别器（checkpoint 落盘用；DDP 装配下为解包后的裸网络）。"""
        return self.update.scorer.discriminator

    def seed_base(
        self, samples: torch.Tensor, modalities: list[Modality],
    ) -> None:
        """冻结初始 policy 的产出填充 base 分区（train 启动期一次，
        逐样本目标模态标签对齐——ADR-0008-01 的配额量产标签输入）。"""
        self.buffer.fill_base(samples, modalities)

    def update_step(
        self, current_fakes: torch.Tensor, modality: Modality,
    ) -> UpdateReport:
        """判别器 Online update 一步：全批 fake 随机置换后交更新
        （50% 当前 / 50% 回放的混采由 update 消费置换批的头部；该条件
        回放候选不足时该步退化纯 current 半区，退化标记随报告透出，
        ADR-0008-03），更新期间判别器 train 相、结束后恢复 eval 相。
        置换过的整批照常入近期分区（近期分布记录是集合语义，次序无关）。

        ``modality`` = 本 iteration 的目标模态：整批 fake 的条件标签
        （入近期分区）、回放半区的过滤条件与 real 批的条件匹配采样
        （ADR-0008 决策 1/2）同源。
        """
        order = torch.randperm(current_fakes.shape[0], generator=self._generator)
        shuffled = current_fakes[order]
        self.discriminator.train()
        try:
            return self.update.step(shuffled, modality)
        finally:
            self.discriminator.eval()

    def heldout_auc(
        self, current_fakes: torch.Tensor, modality: Modality,
    ) -> float:
        """held-out real vs 当前 fake 的判别器 AUC（hacking 监控信号）。

        real 侧按本 iteration 采样的目标序列过滤——iter 事件按序列归因
        reward/loss/AUC，混采会让其他序列的判别器分数偏移伪装成本序列
        realism 变化（per-target-sequence 健康监控）。"""
        return self.auc.compute(current_fakes, modality=modality)
