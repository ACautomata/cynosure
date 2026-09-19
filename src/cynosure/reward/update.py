"""判别器 Online update 一步（reward-model 章「在线更新机制」+ ADR-0012）。

每 RL iteration 更新一步（N_d=1、D:G 更新比 ≈ 1:1）：消费**配对批**
（real 批 + 同源重构 fake 批 + 条件标记，装配原语
``ReconstructionAssembler`` 供批——fake 与 real 同内容、仅生成伪影不
同）→ LSGAN 损失 → AdamW（默认 lr 5e-5）。real 侧不再内部自抽、fake
侧不再回放混采（ADR-0012 决策 6 替换而非并存：判别器 fake 侧只有重构
体，replay buffer 判别器链路退役）；损失、优化器、梯度流零改动。

判别器输入恒干净域（ADR-0012 决策 3）：参数更新前向走干净域打分入口
``patch_logits``——加噪只发生在 fake 构造的输入端（装配原语的重构加
噪），训练、打分、held-out AUC 与过拟合监控共享同一输入语义，不再有
带噪/干净两套输入路径。

条件口径：批的条件标记（目标条件）随 ``PairBatch`` 上行——real 侧条
件匹配采样与 fake 侧同源匹配在装配原语完成（ADR-0008 决策 1 的归因轴
不动，fake 侧条件匹配由同源自动满足——ADR-0012 决策 8）。

本类是「一步」原语：``disc_update_interval_n_d`` 的迭代节奏（每 N_d 个
RL iteration 调用一次 step）由编排方（train 循环）消费；fixture 场景
N_d=1 即每 iter 一步。
"""

from dataclasses import dataclass

import torch

from cynosure.config import RewardConfig
from cynosure.reward.assembly import PairBatch
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.scorer import LatentScorer


@dataclass(frozen=True)
class UpdateReport:
    """一步 online update 的可观测结果（指标流与测试断言的数据）。"""

    loss_discriminator: float
    loss_real_term: float
    loss_fake_term: float
    batch_size: int
    """本步配对批的批量（real 与 fake 两侧同量——同源配对的结构性质）。"""
    modality: str
    """本步更新的条件（目标条件）——update 归因轴与配对批的条件标记
    同源（ADR-0008 决策 1）。"""
    train_pairwise_acc: float
    """train 侧干净域 pairwise 准确率（ADR-0009-β 决策 4）：本步更新批
    上的干净域 no_grad 复算——更新前快照（参数未动）与同 iteration 的
    held-out AUC（更新前测得）同刻，per-condition 分叉 =
    EMA(train pairwise acc − held-out AUC) 的 train 侧原料。估计量与
    held-out AUC 同一 Mann-Whitney pairwise 占比（不同采样平面）。"""


class OnlineUpdate:
    """判别器在线更新一步的编排（消费配对批 → LSGAN → AdamW）。

    配对批由调用方经装配原语产出后注入（预训练 = 冻结基座重构、在线 =
    当前 policy 重构）——构造同构、两阶段同一原语（ADR-0012 决策 7），
    更新管线对批来源无感知。
    """

    def __init__(self, scorer: LatentScorer, config: RewardConfig) -> None:
        self.scorer = scorer
        # weight_decay 显式落位（ADR-0007 卫生项）：与 policy 侧同值口径
        # （config.disc_weight_decay，默认 1e-4）——此前隐式取 PyTorch
        # 默认 0.01，与 policy 侧 1e-4 的事实不对称
        self.optimizer = torch.optim.AdamW(
            scorer.discriminator.parameters(), lr=config.disc_lr,
            weight_decay=config.disc_weight_decay,
        )

    def step(self, pair: PairBatch) -> UpdateReport:
        """一步更新：干净域前向 → LSGAN loss → AdamW step。

        ``pair`` = 装配原语产出的配对批（real 与 fake 同源、同条件、
        同量）——本方法对批的构造（重构链、随机流）无感知。train 侧
        干净域复算发生在参数更新之前（与同 iteration 的 held-out AUC
        同刻，ADR-0009-β）。"""
        # train 侧干净域复算（ADR-0009-β）：参数更新前、同一批上重算一次
        # 干净域准确率随报告上行
        train_pairwise_acc = self._clean_pairwise_accuracy(pair.reals, pair.fakes)
        # 参数更新前向 = 干净域打分入口（ADR-0012 决策 3）：打分、训练、
        # AUC、监控共享同一输入语义（带噪入口随注入退役）
        logits_real = self.scorer.patch_logits(pair.reals)
        logits_fake = self.scorer.patch_logits(pair.fakes)
        terms = self.scorer.discriminator_terms(logits_real, logits_fake)
        self.optimizer.zero_grad()
        terms.total.backward()
        self.optimizer.step()
        return UpdateReport(
            loss_discriminator=terms.total.item(),
            loss_real_term=terms.real_term.item(),
            loss_fake_term=terms.fake_term.item(),
            batch_size=pair.reals.shape[0],
            modality=pair.modality,
            train_pairwise_acc=train_pairwise_acc,
        )

    def _clean_pairwise_accuracy(
        self, reals: torch.Tensor, fakes: torch.Tensor,
    ) -> float:
        """本步更新批的 train 侧干净域 pairwise 准确率（ADR-0009-β）。

        干净域打分入口（``patch_logits``）no_grad 复算——与参数更新
        前向同批（real/fake 两侧）、同判别器快照（发生在 optimizer.step
        之前，参数未动），故与同 iteration 的 held-out AUC（更新前测得）
        同刻可比，分叉 = 同一刻 in-sample 训练批 vs out-of-sample 池的
        判别力差。估计量复用 ``HeldOutAuc.auc_from_scores``（Mann-Whitney
        pairwise 占比，并列计 0.5）：与 held-out AUC 同一估计量、不同
        采样平面，分叉才有「同尺可比」语义，且分数非有限的 fail-fast
        闸口单点共用。监控前向恒 eval 相（spectral norm 的 power
        iteration 不被监控推进，与打分/监控相位约定一致）。
        """
        discriminator = self.scorer.discriminator
        was_training = discriminator.training
        discriminator.eval()
        try:
            with torch.no_grad():
                real_scores = self.scorer.patch_logits(reals).flatten()
                fake_scores = self.scorer.patch_logits(fakes).flatten()
        finally:
            if was_training:
                discriminator.train()
        return HeldOutAuc.auc_from_scores(real_scores, fake_scores)
