"""训练循环的命名随机流注册表（续训状态机按名保存/恢复的消费面）。

八条 ``torch.Generator`` 流、两条 seeding 轴：七条**数据侧**流由
rank 派生后的主 seed 各自偏移派生（rollout 相与条件分布共享主流；
real 采样 / 判别器更新 / 判别器噪声注入 / held-out AUC / fake 置换 /
base 分区生成各用独立流——一条流的抽取数变化不扰动其余流的序列，
容量实验等不漂移样本流，ADR-0009-α）；``recon``（同源重构加噪，
ADR-0012 的新流）由 rank 无关的 shared seed 派生——其 s 抽样的调用
结构是分布式集合序列的一部分，必须跨 rank 一致。

续训状态按**流名**保存/恢复（resume 模块经 ``named()`` 枚举），注册
表结构一变即续训状态清单失配、显式拒绝。

退役中（ADR-0012）：``disc_noise``（训练期噪声注入）与 ``disc_update``
（回放抽样）、``fake_shuffle``（fake 全批置换）的**消费面**已随判别器
更新批换为同源重构配对批而消失——流保留在注册表是为了续训分片的
``named()`` 清单稳定（退役删除与 config schema 清理同票进行）。
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

    def __init__(self, seed: int, shared_seed: int | None = None) -> None:
        """``seed`` = 本 rank 派生后的主 seed（数据侧流的多样性来源）；
        ``shared_seed`` = rank 无关的原 seed，``recon`` 流的派生来源
        （缺省用 ``seed``——单进程两参同值，行为不变）。"""
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
        # recon 用 rank 无关的 shared seed：s 抽样决定重构续跑
        # ``continue_to_terminal`` 的调用次数与逐次批量——分布式下它是
        # FSDP 集合序列的一部分（#165 review P1 的同一不变式），逐 rank
        # 相异会让首个判别器更新步集合错位死锁。ε 与 s 同流（先 s 后 ε
        # 的抽取次序契约）；rank 本地的只有真实样本库切片，重构加噪不是。
        self.recon = torch.Generator().manual_seed(
            (shared_seed if shared_seed is not None else seed) + 9,
        )

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
