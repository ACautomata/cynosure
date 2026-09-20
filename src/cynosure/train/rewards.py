"""判别器侧协作者组（Facade）：聚合配对批装配原语、Online update 与
held-out AUC——trainer 只面对「装配/更新/AUC」三个动作与判别器引用。

判别器相位约定：判别器默认保持 eval 相（打分与监控前向不得推进
spectral norm power iteration），仅更新一步期间短暂 train。配对批的
重构前向是 policy 侧 no_grad 推理（装配原语自持 no_grad + autocast
口径），不受判别器相位影响。

分布式（ADR-0003）：配对批是本 rank 装配产出（real 批来自 rank 切片
pool）、梯度经 DDP allreduce 平均——本类的更新语义逐字不变。
"""

import torch

from cynosure.reward.assembly import PairBatch, ReconstructionAssembler
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.overfit import OverfitMonitor
from cynosure.reward.update import OnlineUpdate, UpdateReport
from cynosure.train.gating import DynamicWhitelist
from cynosure.train.whitelist import ConditionWhitelist


class RewardCoordinator:
    """判别器侧动作面（装配/更新/AUC/分叉监控）与条件白名单的单点持有。"""

    def __init__(
        self, update: OnlineUpdate, auc: HeldOutAuc,
        gating: DynamicWhitelist,
        overfit: OverfitMonitor,
        assembler: ReconstructionAssembler | None,
    ) -> None:
        self.update = update
        self.auc = auc
        # 条件白名单的动态运行时对象（ADR-0008 决策 5/8）：readiness
        # gate 判定与 train 循环逐 iteration 门控查询的同源消费面
        # （经 whitelist 快照视图）；名单变更（EMA 动态恢复）由它以
        # 快照替换驱动，判定为全 rank 集体口径
        self.gating = gating
        # 过拟合分叉监控器（ADR-0009 决策 4/5）：per-condition 分叉 EMA
        # 的 rank 本地单点——train 循环逐判别器步喂入两侧干净域读数、
        # 消费越线判定落 overfit_alert 事件；按 rank 独立（无集合通信），
        # 报警不动作（白名单不被它联动）
        self.overfit = overfit
        # 判别器更新批装配原语（ADR-0012 唯一新缝）：两阶段装配缝注入；
        # None = 替身测试场景，生产装配恒注入（trainer 装配期校验）
        self.assembler = assembler

    @property
    def whitelist(self) -> ConditionWhitelist:
        """条件白名单当前快照（ADR-0008 决策 5 的接线面）：readiness
        gate 的上岗判定与循环侧 ``modality in whitelist`` 逐 iteration
        查询读同一来源——动态恢复变更名单后，查询面即时见到新快照
        （上岗名单与更新开关永不分叉）。"""
        return self.gating.whitelist

    @property
    def discriminator(self) -> torch.nn.Module:
        """底层判别器（checkpoint 落盘用；DDP 装配下为解包后的裸网络）。"""
        return self.update.scorer.discriminator

    def update_step(self, pair: PairBatch) -> UpdateReport:
        """判别器 Online update 一步：消费配对批（real 与 fake 同源，
        装配原语供批——ADR-0012），更新期间判别器 train 相、结束后
        恢复 eval 相。

        ``pair.modality`` = 本 iteration 的目标条件（条件标记随批上行，
        ADR-0008 决策 1 的归因轴；fake 侧条件匹配由同源自动满足）。"""
        self.discriminator.train()
        try:
            return self.update.step(pair)
        finally:
            self.discriminator.eval()

    def heldout_auc(
        self, current_fakes: torch.Tensor, modality: str,
    ) -> float:
        """held-out real vs 当前 fake 的判别器 AUC（hacking 监控信号）。

        fake 侧维持 rollout 终点（gate 的运行语义 = 判别器对打分对象
        的分辨力，ADR-0012 决策 4）；real 侧按本 iteration 采样的目标
        条件过滤——iter 事件按序列归因 reward/loss/AUC，混采会让其他
        序列的判别器分数偏移伪装成本序列 realism 变化
        （per-target-sequence 健康监控）。"""
        return self.auc.compute(current_fakes, modality=modality)
