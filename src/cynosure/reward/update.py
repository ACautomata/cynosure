"""判别器 Online update 一步（reward-model 章「在线更新机制」）。

每 RL iteration 更新一步（N_d=1、D:G 更新比 ≈ 1:1）：50% 当前 fake +
50% 回放混采 → real 批（Real sample pool 按本 iteration 条件匹配采样）
→ LSGAN 损失 → AdamW（默认 lr 5e-5）→ 当前 fake 全部入近期分区。先采
回放、后 push：当前批不会在本步回放中立即重复出现。

条件口径（ADR-0008 决策 1/2/3，归因轴 = 目标模态，与 held-out AUC 同
源）：回放半区按该条件过滤（条目标签见 buffer 模块），real 批同条件
匹配采样，当前 fake 入近期分区带同条件标签。该条件回放候选不足回放
半区需求时（``condition_supply`` 查询面），该步退化为纯 current 半区
（回放 0 条）、real 侧与退化后批同量匹配，报告带 ``replay_degraded``
标记落盘——绝不静默回退全池混采或半途截断回放（退化判定的装配期
防线 = ``assert_replay_supply`` + ``LatentManifest.assert_condition_capacity``）。

本类是「一步」原语：``disc_update_interval_n_d`` 的迭代节奏（每 N_d 个
RL iteration 调用一次 step）由编排方（train 循环，后续 ticket）消费；
fixture 场景 N_d=1 即每 iter 一步。
"""

import math
from dataclasses import dataclass

import torch

from cynosure.config import RewardConfig
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayDraw, ReplayStore
from cynosure.reward.sampler import RealSampling
from cynosure.reward.scorer import LatentScorer


@dataclass(frozen=True)
class UpdateReport:
    """一步 online update 的可观测结果（指标流与测试断言的数据）。"""

    loss_discriminator: float
    loss_real_term: float
    loss_fake_term: float
    num_current: int
    num_replay: int
    num_base_replay: int
    num_recent_replay: int
    modality: str
    """本步更新的条件（目标模态）——update 归因轴与回放/real 两侧的
    过滤条件同源（ADR-0008 决策 1）。"""
    replay_degraded: bool
    """回放退化标记（ADR-0008 决策 2）：该条件回放候选不足半区需求，
    本步退化为纯 current 半区（回放 0 条、real 侧与退化后批同量）。"""
    train_pairwise_acc: float
    """train 侧干净域 pairwise 准确率（ADR-0009-β 决策 4）：本步更新批
    上的干净域 no_grad 复算——更新前快照（参数未动）与同 iteration 的
    held-out AUC（更新前测得）同刻，per-condition 分叉 =
    EMA(train pairwise acc − held-out AUC) 的 train 侧原料。估计量与
    held-out AUC 同一 Mann-Whitney pairwise 占比（不同采样平面）。"""


class OnlineUpdate:
    """判别器在线更新一步的编排（组合 scorer / buffer / real 采样 / 优化器）。

    fake 源由调用方注入（生产 = 当前 policy rollout 输出；fixture = 预置
    固定 latent 批）——fake 侧与生产解耦，更新管线完全同一。

    参数更新前向走 scorer 的训练专用带噪入口（ADR-0009-α：real/fake 两侧
    逐样本对称噪声注入，打分路径不动）；噪声采样经 ``noise_generator``
    注入的专属随机流（TrainingRngStreams 的 ``disc_noise`` 流——与回放
    抽样等训练采样流不交叉，σ_max 取值不漂移其余流的序列），σ_max = 0
    时该流零消耗（回归锚：全链路与无注入逐位一致）。
    """

    def __init__(
        self,
        scorer: LatentScorer,
        buffer: ReplayStore,
        real_sampler: RealSampling,
        config: RewardConfig,
        generator: torch.Generator,
        noise_generator: torch.Generator,
    ) -> None:
        self.scorer = scorer
        self.buffer = buffer
        # weight_decay 显式落位（ADR-0007 卫生项）：与 policy 侧同值口径
        # （config.disc_weight_decay，默认 1e-4）——此前隐式取 PyTorch
        # 默认 0.01，与 policy 侧 1e-4 的事实不对称
        self.optimizer = torch.optim.AdamW(
            scorer.discriminator.parameters(), lr=config.disc_lr,
            weight_decay=config.disc_weight_decay,
        )
        self._real_sampler = real_sampler
        self._generator = generator
        self._noise_generator = noise_generator
        self._batch_size_k = config.disc_batch_size_k
        self._current_fraction = config.replay_current_fraction

    def step(
        self, current_fakes: torch.Tensor, modality: str,
    ) -> UpdateReport:
        """一步更新：混采 fake 批（50% 当前 / 50% 回放）→ real 批 →
        LSGAN loss → AdamW step → 当前 fake 入近期分区。

        当前半区取 fake 批前 current_count 条（确定性口径）；整批当前
        fake 全部入近期分区（记录近期 policy 分布）。``modality`` = 本
        iteration 的目标模态：回放按条件过滤（ADR-0008 决策 2），real
        批同条件匹配采样（决策 1），入区 fake 带同标签。该条件回放候选
        不足（``condition_supply`` < 回放半区需求）时本步退化纯 current
        半区：回放 0 条、real 侧与退化后批同量匹配、报告带
        ``replay_degraded``（ADR-0008-03）。
        """
        current_count = math.ceil(self._batch_size_k * self._current_fraction)
        replay_count = self._batch_size_k - current_count
        if current_fakes.shape[0] < current_count:
            raise ValueError(
                f"当前 fake {current_fakes.shape[0]} 条 < 当前半区需求"
                f" {current_count} 条（batch_size_k={self._batch_size_k}）"
            )
        degraded = self.buffer.condition_supply(modality) < replay_count
        if degraded:
            # 退化：纯 current 半区（回放 0 条），real 侧与退化后批同量
            fakes = current_fakes[:current_count]
            real_count = current_count
            num_replay = num_base_replay = num_recent_replay = 0
        else:
            draw: ReplayDraw = self.buffer.sample_replay(
                replay_count, self._generator, modality,
            )
            fakes = torch.cat([current_fakes[:current_count], draw.samples])
            real_count = self._batch_size_k
            num_replay, num_base_replay, num_recent_replay = (
                replay_count, draw.num_base, draw.num_recent,
            )
        reals = self._real_sampler.sample(real_count, modality=modality)
        # train 侧干净域复算（ADR-0009-β）：参数更新前、同一批上重算一次
        # 干净域准确率随报告上行——带噪训练前向的 loss 伴生量系统性低估
        # 分叉（带噪输入使任务天然更难），不得复用
        train_pairwise_acc = self._clean_pairwise_accuracy(reals, fakes)
        # 参数更新前向 = 训练专用带噪入口（ADR-0009-α）：real/fake 两侧
        # 同一入口、同一噪声流（对称性 = 同一采样机制、同分布）；打分
        # 路径（patch_logits / reward / AUC）不经过本入口、恒干净域
        logits_fake = self.scorer.training_patch_logits(
            fakes, self._noise_generator,
        )
        logits_real = self.scorer.training_patch_logits(
            reals, self._noise_generator,
        )
        terms = self.scorer.discriminator_terms(logits_real, logits_fake)
        self.optimizer.zero_grad()
        terms.total.backward()
        self.optimizer.step()
        self.buffer.push(current_fakes, modality)
        return UpdateReport(
            loss_discriminator=terms.total.item(),
            loss_real_term=terms.real_term.item(),
            loss_fake_term=terms.fake_term.item(),
            num_current=current_count,
            num_replay=num_replay,
            num_base_replay=num_base_replay,
            num_recent_replay=num_recent_replay,
            modality=modality,
            replay_degraded=degraded,
            train_pairwise_acc=train_pairwise_acc,
        )

    def _clean_pairwise_accuracy(
        self, reals: torch.Tensor, fakes: torch.Tensor,
    ) -> float:
        """本步更新批的 train 侧干净域 pairwise 准确率（ADR-0009-β）。

        干净域打分入口（``patch_logits``）no_grad 复算——与带噪训练
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
