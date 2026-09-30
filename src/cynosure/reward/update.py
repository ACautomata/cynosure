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

**UpdateReport 升格（#220 决议 11，判别器链期）**：混合条件配对批的
per-condition 明细升格——``conditions`` 逐条件明细（桶序恒条件名排
序）、``batch_size`` = 本卡 Σ桶对数、``global_batch_size`` = 全局
N_total（另立字段）。单条件一步（本类、预训练 driver）= 单桶特例。
``modality``/``loss_real_term``/``loss_fake_term``/``train_pairwise_acc``
单值字段失效为 property：单桶派生、多桶显式拒绝（升格不静默二义）。
"""

from dataclasses import dataclass

import torch

from cynosure.config import RewardConfig
from cynosure.reward.assembly import PairBatch
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.scorer import LatentScorer


@dataclass(frozen=True)
class ConditionUpdateDetail:
    """单条件（单桶）更新明细：per-condition 归因的数据面（#220 决议
    11 升格）。loss 组件均为**未缩放**本桶值（加权是步级聚合的语义）。"""

    condition: str
    loss_discriminator: float
    loss_real_term: float
    loss_fake_term: float
    pair_count: int
    """本桶对数——逐对等权全局 mean 的加权因子（loss×pair_count/N_total，
    #220 决议 8）。"""
    train_pairwise_acc: float


@dataclass(frozen=True)
class UpdateReport:
    """一步 online update 的可观测结果（指标流与测试断言的数据）。

    升格后（#220 决议 11）的三个量面：

    - ``conditions`` = per-condition 明细（单条件一步 = 单桶特例）；
    - ``batch_size`` = 本卡 Σ桶对数（单卡即全量）；
    - ``global_batch_size`` = 全局 N_total（K×卡数，逐对等权分母）；
    - ``loss_discriminator`` = 本步上报 loss——单条件一步（DDP AVG
      语义）为**未缩放** terms.total；混合条件步（新执行序 SUM 语义）
      为 ``Σ_b loss_b×(n_b/N_total)`` 的全局加权值（
      ``DiscriminatorPhase`` 构造）。口径随执行序，以构造点 docstring
      为准。
    """

    conditions: tuple[ConditionUpdateDetail, ...]
    batch_size: int
    global_batch_size: int
    loss_discriminator: float

    @property
    def modality(self) -> str:
        """单值字段失效为 property（#220 决议 11）：单桶派生条件名；
        多桶显式拒绝（混合条件批无单一归因）。"""
        return self._single().condition

    @property
    def loss_real_term(self) -> float:
        """单桶派生 real 项；多桶拒绝。"""
        return self._single().loss_real_term

    @property
    def loss_fake_term(self) -> float:
        """单桶派生 fake 项；多桶拒绝。"""
        return self._single().loss_fake_term

    @property
    def train_pairwise_acc(self) -> float:
        """单桶派生 train 侧干净域复算准确率；多桶拒绝。

        （ADR-0009-β 决策 4 的原料：与同 iteration held-out AUC 合成
        per-condition 分叉观测——升格后逐条件明细各自携带，消费面按
        条件取。）"""
        return self._single().train_pairwise_acc

    def _single(self) -> ConditionUpdateDetail:
        if len(self.conditions) != 1:
            raise ValueError(
                f"单值字段在混合条件步失效（#220 决议 11 升格）：本步 "
                f"{len(self.conditions)} 桶 {[d.condition for d in self.conditions]}——"
                "per-condition 读数经 conditions 明细按条件消费"
            )
        return self.conditions[0]


class OnlineUpdate:
    """判别器在线更新一步的编排（消费配对批 → LSGAN → AdamW）。

    配对批由调用方经装配原语产出后注入（预训练 = 冻结基座重构、在线 =
    当前 policy 重构）——构造同构、两阶段同一原语（ADR-0012 决策 7），
    更新管线对批来源无感知。
    """

    def __init__(self, scorer: LatentScorer, config: RewardConfig) -> None:
        self.scorer = scorer
        # 优化器经共享装配缝（``assemble_optimizer``）构建——两执行序
        # （OnlineUpdate / async 门面 DiscriminatorPhase）同一 AdamW 形态
        self.optimizer = self.assemble_optimizer(scorer, config)

    @staticmethod
    def assemble_optimizer(
        scorer: LatentScorer, config: RewardConfig,
    ) -> torch.optim.Optimizer:
        """判别器 AdamW 的单一装配缝（两执行序同一实例形态）：
        lr = ``config.disc_lr``、weight_decay 显式落位 = ``config.
        disc_weight_decay``（ADR-0007 卫生项——与 policy 侧同值口径，
        此前隐式取 PyTorch 默认 0.01 的事实不对称）。Online update
        （旧执行序/pretrain）与 ``DiscriminatorPhase``（async 执行序）
        经本缝共享，消除双写漂移面。"""
        return torch.optim.AdamW(
            scorer.discriminator.parameters(), lr=config.disc_lr,
            weight_decay=config.disc_weight_decay,
        )

    def step(self, pair: PairBatch) -> UpdateReport:
        """一步更新：干净域前向 → LSGAN loss → AdamW step。

        ``pair`` = 装配原语产出的配对批（real 与 fake 同源、同条件、
        同量）——本方法对批的构造（重构链、随机流）无感知。train 侧
        干净域复算发生在参数更新之前（与同 iteration 的 held-out AUC
        同刻，ADR-0009-β）。单条件一步（旧执行序/pretrain）：上报
        loss 保持未缩放（DDP AVG 口径，与升格前逐位一致），
        ``global_batch_size`` = 本批对数（单进程口径——多卡 N_total
        由新执行序 ``DiscriminatorPhase`` 在构造点直接供给，不经本
        路径）。"""
        count = pair.reals.shape[0]
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
        detail = ConditionUpdateDetail(
            condition=pair.modality,
            loss_discriminator=terms.total.item(),
            loss_real_term=terms.real_term.item(),
            loss_fake_term=terms.fake_term.item(),
            pair_count=count,
            train_pairwise_acc=train_pairwise_acc,
        )
        return UpdateReport(
            conditions=(detail,),
            batch_size=count,
            global_batch_size=count,
            loss_discriminator=terms.total.item(),
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
