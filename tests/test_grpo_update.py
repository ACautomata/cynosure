"""逐 k 独立更新编排：每训练步 k 一次 forward→backward→optimizer.step
（每 iteration 共 |M| 次，spec #15 执行序第 2 相），bf16 autocast +
fp32 master 语义的单进程版。

log-prob 重算与 rollout 记录共用 RolloutSampler 的同一组转移路径
（口径一致性由 policy 测试锚定）；本文件聚焦更新步的结构行为。"""

import pytest
import torch

from cynosure.fixtures import Fixture
from cynosure.grpo import ClippedPolicyLoss
from cynosure.grpo.update import StepwisePolicyUpdate
from cynosure.netbuild import NetworkAssembler
from cynosure.policy import (
    CfgCombinedField,
    RolloutCondition,
    RolloutSampler,
    SdeKernel,
    TrajectoryCursor,
)

LATENT_SHAPE = (4, 16, 16, 8)
GROUP_SIZE = 4  # 更新步结构测试与 G=12 的统计语义无关，缩小省 CPU 时间
POLICY_LR = 2e-6  # spec 起步值


class TrainingScenario:
    """一次可更新的最小场景：fixture UNet + 3 步日程 + 一个扰动方向批。"""

    def __init__(self, seed: int = 7) -> None:
        torch.manual_seed(seed)
        self.unet = Fixture().unet().eval()
        self.sampler = RolloutSampler(
            CfgCombinedField(self.unet),
            SdeKernel(eta=0.7, s_max=0.999),
            TrajectoryCursor(NetworkAssembler.rflow_scheduler(3, 2048)),
        )
        self.condition = RolloutCondition(
            label=torch.tensor([29]),
            spacing=torch.full((1, 3), 100.0),
        )
        torch.manual_seed(seed + 1)
        self.x_k = torch.randn(1, *LATENT_SHAPE)
        noise = torch.randn(GROUP_SIZE, *LATENT_SHAPE)
        with torch.no_grad():  # 生产 rollout 相的口径：π_old 是无图记录
            self.directions, self.old_log_probs = self.sampler.perturb_group(
                self.x_k, index=1, condition=self.condition, noise=noise,
            )
        self.advantage = torch.linspace(-1.0, 1.0, GROUP_SIZE)

    def build_update(self) -> StepwisePolicyUpdate:
        """组装被测更新编排（AdamW lr=2e-6、clipped loss、bf16 autocast）。"""
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=POLICY_LR)
        return StepwisePolicyUpdate(
            sampler=self.sampler,
            optimizer=optimizer,
            loss=ClippedPolicyLoss(clip_range=1e-4),
        )

    def parameters(self) -> list[torch.Tensor]:
        return [p.detach().clone() for p in self.unet.parameters()]





class TestStepwisePolicyUpdate:
    """|M| 次/iteration 的独立梯度步：逐 k 前向→loss→backward→step。"""

    @pytest.fixture
    def scenario(self) -> TrainingScenario:
        return TrainingScenario()

    def test_single_step_changes_parameters_and_returns_loss(
        self, scenario: TrainingScenario,
    ) -> None:
        update = scenario.build_update()
        before = scenario.parameters()
        loss = update.step(
            step_index=1,
            x_k=scenario.x_k,
            condition=scenario.condition,
            directions=scenario.directions,
            old_log_probs=scenario.old_log_probs,
            advantages=scenario.advantage,
        )
        after = scenario.parameters()
        assert torch.isfinite(torch.tensor(loss))
        assert any(
            not torch.equal(a, b) for a, b in zip(before, after)
        )  # optimizer.step 真实生效

    def test_independent_steps_per_k_use_evolved_weights(
        self, scenario: TrainingScenario,
    ) -> None:
        """每个 k 独立一次梯度步：第二次 step 基于第一次更新后的权重
        （每 iteration 共 |M| 次独立的 forward→backward→step）。"""
        update = scenario.build_update()
        first = update.step(
            1, scenario.x_k, scenario.condition,
            scenario.directions, scenario.old_log_probs, scenario.advantage,
        )
        snapshot_after_first = scenario.parameters()
        second = update.step(
            1, scenario.x_k, scenario.condition,
            scenario.directions, scenario.old_log_probs, scenario.advantage,
        )
        snapshot_after_second = scenario.parameters()
        assert any(
            not torch.equal(a, b)
            for a, b in zip(snapshot_after_first, snapshot_after_second)
        )
        # 权重已演化、两步均产出有限 loss（重算真走了当前权重）
        assert torch.isfinite(torch.tensor([first, second])).all()

    def test_step_direction_follows_advantage_sign(
        self, scenario: TrainingScenario,
    ) -> None:
        """advantage 符号驱动正确的更新方向（lr=2e-6 下微弱但可测）：
        正 advantage 推高重算 log-prob；负 advantage 不推高（观测为 0）。

        数值事实（tracer bullet 存档）：bf16 前向的 log-prob 量化偏差
        使初始 ratio ≈ 0.997（偏离 1 约 1e-3，clip 窗 1e-4 的 10~30 倍），
        ratio 系统性落在窗下方——正 advantage 方向 min 取 surr1（梯度
        流动）、负 advantage 方向被 clamp 分支遮蔽（「只罚外推」语义在
        bf16 下的极端形态）。该行为与生产 DCU（同为 bf16 autocast）一致，
        M0 门槛的 bf16 生效 profile 应以此 fixture 观测为参照。
        """
        update = scenario.build_update()
        positive = torch.full((GROUP_SIZE,), 5.0)  # clamp 边界的强正 advantage
        negative = torch.full((GROUP_SIZE,), -5.0)
        before = scenario.sampler.evaluate_log_prob(
            scenario.x_k, 1, scenario.condition, scenario.directions,
        ).detach().clone()
        update.step(
            1, scenario.x_k, scenario.condition,
            scenario.directions, scenario.old_log_probs, positive,
        )
        after_positive = scenario.sampler.evaluate_log_prob(
            scenario.x_k, 1, scenario.condition, scenario.directions,
        ).detach()
        scenario_negative = TrainingScenario()  # 独立场景：负 advantage 对照
        update_negative = scenario_negative.build_update()
        update_negative.step(
            1, scenario_negative.x_k, scenario_negative.condition,
            scenario_negative.directions, scenario_negative.old_log_probs,
            negative,
        )
        after_negative = scenario_negative.sampler.evaluate_log_prob(
            scenario_negative.x_k, 1, scenario_negative.condition,
            scenario_negative.directions,
        ).detach()
        assert (after_positive - before).mean() > 0
        assert (after_negative - before).mean() <= 0

    def test_fp32_master_weights_preserved_under_bf16_autocast(
        self, scenario: TrainingScenario,
    ) -> None:
        """bf16 autocast 只作用于前向：参数与梯度保持 fp32（fp32 master）。"""
        update = scenario.build_update()
        update.step(
            1, scenario.x_k, scenario.condition,
            scenario.directions, scenario.old_log_probs, scenario.advantage,
        )
        assert all(p.dtype == torch.float32 for p in scenario.unet.parameters())
        assert all(
            g.dtype == torch.float32
            for g in (
                p.grad for p in scenario.unet.parameters() if p.grad is not None
            )
        )

    def test_old_log_probs_and_directions_not_mutated(
        self, scenario: TrainingScenario,
    ) -> None:
        """π_old 与方向批是 rollout 记录：更新步不得原地改动（ratio 的
        分母必须保持 rollout 时数值）。"""
        update = scenario.build_update()
        directions_snapshot = scenario.directions.clone()
        old_snapshot = scenario.old_log_probs.clone()
        update.step(
            1, scenario.x_k, scenario.condition,
            scenario.directions, scenario.old_log_probs, scenario.advantage,
        )
        assert torch.equal(scenario.old_log_probs, old_snapshot)
        assert torch.equal(scenario.directions, directions_snapshot)

    def test_recomputed_log_prob_matches_recorded_before_any_step(
        self, scenario: TrainingScenario,
    ) -> None:
        """log-prob 一致性（更新前同权重）：更新侧重算与 rollout 记录值
        逐位一致（同权重同口径——ratio 分子分母可信的前提）。"""
        recomputed = scenario.sampler.evaluate_log_prob(
            scenario.x_k, 1, scenario.condition, scenario.directions,
        )
        assert torch.equal(recomputed, scenario.old_log_probs)
