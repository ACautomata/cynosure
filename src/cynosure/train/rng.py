"""训练循环的命名随机流注册表（续训状态机按名保存/恢复的消费面）。

六条 ``torch.Generator`` 流由主 seed 各自偏移派生、互不漂移：rollout 相
与条件分布共享主流；real 采样 / 判别器更新 / held-out AUC / fake 置换 /
base 分区生成各用独立流——一条流的抽取数变化不扰动其余流的序列（容量
实验等不漂移样本流）。续训状态按**流名**保存/恢复（resume 模块经
``named()`` 枚举），注册表结构一变即续训状态清单失配、显式拒绝。
"""

import torch


class TrainingRngStreams:
    """六条命名 ``torch.Generator`` 流的注册表（训练循环的随机性注入口）。"""

    ROLLOUT = "rollout"
    REAL_POOL = "real_pool"
    DISC_UPDATE = "disc_update"
    HELDOUT_AUC = "heldout_auc"
    FAKE_SHUFFLE = "fake_shuffle"
    BASE_PARTITION = "base_partition"

    def __init__(self, seed: int) -> None:
        self.rollout = torch.Generator().manual_seed(seed)
        self.real_pool = torch.Generator().manual_seed(seed + 1)
        self.disc_update = torch.Generator().manual_seed(seed + 2)
        self.heldout_auc = torch.Generator().manual_seed(seed + 3)
        self.fake_shuffle = torch.Generator().manual_seed(seed + 4)
        self.base_partition = torch.Generator().manual_seed(seed + 5)

    def named(self) -> dict[str, torch.Generator]:
        """流名 → Generator 的映射视图（续训状态的保存/恢复枚举面）。"""
        return {
            self.ROLLOUT: self.rollout,
            self.REAL_POOL: self.real_pool,
            self.DISC_UPDATE: self.disc_update,
            self.HELDOUT_AUC: self.heldout_auc,
            self.FAKE_SHUFFLE: self.fake_shuffle,
            self.BASE_PARTITION: self.base_partition,
        }
