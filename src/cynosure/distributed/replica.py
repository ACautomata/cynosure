"""判别器的每 rank 完整副本 + DDP 梯度 allreduce（ADR-0003：判别器不分片）。

几层 3D conv 的在线更新不值得引入分片复杂度：每 rank 完整副本、本
rank fake + Real sample pool 切片更新，梯度经标准 DDP allreduce 平均
（各 rank 数据不同 → 梯度 = 全 rank 梯度平均 → 更新后参数逐位一致）。
打分与 held-out AUC 的前向在 no_grad/eval 相下经 DDP wrapper 直通
（无梯度 hook、无集合通信），判别器 checkpoint 契约不变——
``RewardScorer.discriminator`` 解包 DDP 返回裸网络，state_dict 键形
与单进程逐字一致。
"""

import os

import torch
from torch.nn.parallel import DistributedDataParallel

from cynosure.distributed.process import DistributedContext
from cynosure.reward.scorer import RewardScorer


class ReplicatedDiscriminator:
    """判别器副本语义的装配入口（scorer 内升级为 DDP、其余零改动）。"""

    @classmethod
    def replicate(
        cls, scorer: RewardScorer, context: DistributedContext,
    ) -> RewardScorer:
        """scorer 的判别器升级为 DDP 副本（单进程恒等返回）。"""
        if not context.distributed:
            return scorer
        device_ids = cls._device_ids(context)
        scorer.adopt_distributed(DistributedDataParallel(
            scorer.discriminator, device_ids=device_ids,
        ))
        return scorer

    @staticmethod
    def _device_ids(context: DistributedContext) -> list[int] | None:
        """DDP 的 device_ids：加速器 rank 卡号；CPU（gloo fixture）为 None。"""
        if context.local_device().type != "cuda":
            return None
        return [int(os.environ.get("LOCAL_RANK", context.rank))]
