"""训练循环的命名随机流注册表（续训状态机按名保存/恢复的消费面）。

八条 ``torch.Generator`` 流由主 seed 各自偏移派生、互不漂移：rollout 相
与条件分布共享主流；real 采样 / 判别器更新 / 判别器噪声注入 / held-out
AUC / fake 置换 / base 分区生成 / 同源重构加噪各用独立流——一条流的抽
取数变化不扰动其余流的序列（容量实验等不漂移样本流；噪声注入的 σ_max
取值同理不漂移回放抽样序列，ADR-0009-α）。续训状态按**流名**保存/恢复
（resume 模块经 ``named()`` 枚举），注册表结构一变即续训状态清单失配、
显式拒绝。

退役中（ADR-0012）：``disc_noise``（训练期噪声注入）与 ``disc_update``
（回放抽样）、``fake_shuffle``（fake 全批置换）的**消费面**已随判别器
更新批换为同源重构配对批而消失——流保留在注册表是为了续训分片的
``named()`` 清单稳定（退役删除与 config schema 清理同票进行）；
``recon``（同源重构加噪，先抽 s 后抽 ε）是 ADR-0012 的新流。
"""

import torch


class TrainingRngStreams:
    """八条命名 ``torch.Generator`` 流的注册表（训练循环的随机性注入口）。"""

    ROLLOUT = "rollout"
    REAL_POOL = "real_pool"
    DISC_UPDATE = "disc_update"
    DISC_NOISE = "disc_noise"
    HELDOUT_AUC = "heldout_auc"
    FAKE_SHUFFLE = "fake_shuffle"
    BASE_PARTITION = "base_partition"
    RECON = "recon"

    def __init__(self, seed: int) -> None:
        self.rollout = torch.Generator().manual_seed(seed)
        self.real_pool = torch.Generator().manual_seed(seed + 1)
        self.disc_update = torch.Generator().manual_seed(seed + 2)
        self.heldout_auc = torch.Generator().manual_seed(seed + 3)
        self.fake_shuffle = torch.Generator().manual_seed(seed + 4)
        self.base_partition = torch.Generator().manual_seed(seed + 5)
        # seed+6/7 已被注册表之外的派生用途占用（+6 = 冷启动判别器
        # 初始化的全局 fork seed、+7 = 预训练 SupportRule 的 bootstrap 流
        # ——PretrainDriver），噪声流顺延 +8，同源重构流顺延 +9
        self.disc_noise = torch.Generator().manual_seed(seed + 8)
        self.recon = torch.Generator().manual_seed(seed + 9)

    def named(self) -> dict[str, torch.Generator]:
        """流名 → Generator 的映射视图（续训状态的保存/恢复枚举面）。"""
        return {
            self.ROLLOUT: self.rollout,
            self.REAL_POOL: self.real_pool,
            self.DISC_UPDATE: self.disc_update,
            self.DISC_NOISE: self.disc_noise,
            self.HELDOUT_AUC: self.heldout_auc,
            self.FAKE_SHUFFLE: self.fake_shuffle,
            self.BASE_PARTITION: self.base_partition,
            self.RECON: self.recon,
        }
