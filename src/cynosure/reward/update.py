"""判别器 Online update 一步（reward-model 章「在线更新机制」）。

每 RL iteration 更新一步（N_d=1、D:G 更新比 ≈ 1:1）：50% 当前 fake +
50% 回放混采 → real 批（Real sample pool 均匀采样）→ LSGAN 损失 →
AdamW（默认 lr 5e-5）→ 当前 fake 全部入近期分区。先采回放、后 push：
当前批不会在本步回放中立即重复出现。

fake 侧条件口径（ADR-0008 决策 2）：本 iteration 目标模态随 fake 批
穿入——回放半区按该条件过滤（条目标签见 buffer 模块），当前 fake
入近期分区带同条件标签；real 侧条件匹配采样归 ADR-0008-03。

本类是「一步」原语：``disc_update_interval_n_d`` 的迭代节奏（每 N_d 个
RL iteration 调用一次 step）由编排方（train 循环，后续 ticket）消费；
fixture 场景 N_d=1 即每 iter 一步。
"""

import math
from dataclasses import dataclass

import torch

from cynosure.config import Modality, RewardConfig
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


class OnlineUpdate:
    """判别器在线更新一步的编排（组合 scorer / buffer / real 采样 / 优化器）。

    fake 源由调用方注入（生产 = 当前 policy rollout 输出；fixture = 预置
    固定 latent 批）——fake 侧与生产解耦，更新管线完全同一。
    """

    def __init__(
        self,
        scorer: LatentScorer,
        buffer: ReplayStore,
        real_sampler: RealSampling,
        config: RewardConfig,
        generator: torch.Generator,
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
        self._batch_size_k = config.disc_batch_size_k
        self._current_fraction = config.replay_current_fraction

    def step(
        self, current_fakes: torch.Tensor, modality: Modality,
    ) -> UpdateReport:
        """一步更新：混采 fake 批（50% 当前 / 50% 回放）→ real 批 →
        LSGAN loss → AdamW step → 当前 fake 入近期分区。

        当前半区取 fake 批前 current_count 条（确定性口径）；整批当前
        fake 全部入近期分区（记录近期 policy 分布）。``modality`` = 本
        iteration 的目标模态：回放按条件过滤（ADR-0008 决策 2，条件候选
        不足显式拒绝——退化路径归 ADR-0008-03），入区 fake 带同标签。
        real 侧暂为全池采样（条件匹配归 ADR-0008-03）。
        """
        current_count = math.ceil(self._batch_size_k * self._current_fraction)
        replay_count = self._batch_size_k - current_count
        if current_fakes.shape[0] < current_count:
            raise ValueError(
                f"当前 fake {current_fakes.shape[0]} 条 < 当前半区需求"
                f" {current_count} 条（batch_size_k={self._batch_size_k}）"
            )
        draw: ReplayDraw = self.buffer.sample_replay(
            replay_count, self._generator, modality,
        )
        fakes = torch.cat([current_fakes[:current_count], draw.samples])
        reals = self._real_sampler.sample(self._batch_size_k)
        logits_fake = self.scorer.patch_logits(fakes)
        logits_real = self.scorer.patch_logits(reals)
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
            num_replay=replay_count,
            num_base_replay=draw.num_base,
            num_recent_replay=draw.num_recent,
        )
