"""判别器的每 rank 完整副本 + DDP 梯度 allreduce（ADR-0003：判别器不分片）。

几层 3D conv 的在线更新不值得引入分片复杂度：每 rank 完整副本、本
rank fake + Real sample pool 切片更新，梯度经标准 DDP allreduce 平均
（各 rank 数据不同 → 梯度 = 全 rank 梯度平均 → 更新后参数逐位一致）。
打分与 held-out AUC 的前向在 no_grad/eval 相下经 DDP wrapper 直通
（无梯度 hook、**无集合通信**），判别器 checkpoint 契约不变——
``RewardScorer.discriminator`` 解包 DDP 返回裸网络，state_dict 键形
与单进程逐字一致。

前向期 buffer 广播显式关闭（不是 DDP 默认）：判别器的 buffer 无状态
语义——GroupNorm 无 running stats，谱归一化的 ``_u``/``_v`` 由装载面
（checkpoint 携带参数化状态，或冷启动的 fork_rng 定死初始化）与
「各 rank 相同权重上的确定性幂迭代」逐位一致，广播是语义空操作。幂
迭代只在 train 相前向就地推进，而推进点只有在线更新（各 rank 前向
次数对称、权重经梯度 allreduce 逐位一致）；打分与监控前向恒 eval 相
（``RewardCoordinator``），不推进。而**打开时会引入一个隐藏的集合
点**：首次前向触发一次跨 rank 广播，启动期集合裁决（RM readiness
gate）的失败 rank 会绕过前向直奔 all_gather，其余 rank 卡在那次广播
里互等（实测 world=2 死锁）。关闭后 no_grad 前向是纯本地计算：失败
rank 与健康 rank 都必然抵达显式裁决点。
"""

import inspect
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
            **cls._no_forward_buffer_sync(),
        ))
        return scorer

    @staticmethod
    def _no_forward_buffer_sync() -> dict:
        """关闭「前向期 buffer 广播」的版本自适应构造参数（见模块
        docstring：该广播是语义空操作，却是一次会让失败 rank 的邻居
        互等的隐藏集合通信）。

        torch 2.14 起 ``broadcast_buffers=False`` 弃用（新名
        ``forward_sync_buffers=False`` 只关前向同步、保留构造期同步）；
        旧版本无新参数，用弃用名（语义即「前向不同步」）。两条分支都
        保留 DDP 构造期的参数同步——各 rank 副本起步一致仍是装配前提。
        """
        if "forward_sync_buffers" in inspect.signature(
            DistributedDataParallel,
        ).parameters:
            return {"forward_sync_buffers": False}
        return {"broadcast_buffers": False}

    @staticmethod
    def _device_ids(context: DistributedContext) -> list[int] | None:
        """DDP 的 device_ids：加速器 rank 卡号；CPU（gloo fixture）为 None。"""
        if context.local_device().type != "cuda":
            return None
        return [int(os.environ.get("LOCAL_RANK", context.rank))]
