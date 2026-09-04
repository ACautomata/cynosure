"""policy 薄封装测试：轨迹游标、单步 SDE 核（η=0 parity 锚点）、组1 CFG=10
组合场（batch 组织逐字复刻）、rollout 编排（Anchor 轨迹 / 单步扰动 / ODE 续跑）。

真值锚 = MONAI 库本身（step()、set_timesteps 输出），非基座仓库（零依赖）。"""

import pytest
import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.schedulers import RFlowScheduler

from cynosure.fixtures import Fixture, FIXTURE_UNET_CONFIG
from cynosure.netbuild import NetworkAssembler
from cynosure.policy import (
    CfgCombinedField,
    RolloutCondition,
    RolloutSampler,
    SdeKernel,
    TrajectoryCursor,
)

LATENT_SHAPE = (4, 16, 16, 8)
SPACING = torch.full((1, 3), 100.0)  # 体素间距 ×1e2，恒传（fixture 单位间距）


@pytest.fixture
def fixture_unet() -> DiffusionModelUNetMaisi:
    # Fixture.unet() 对全零卷积重初始化（zero-init 会把 temb 条件通道归零、
    # 输出恒 0）：测试需要条件敏感性真实存在（timestep/label 可分辨）
    torch.manual_seed(7)
    return Fixture().unet().eval()


@pytest.fixture
def scheduler() -> RFlowScheduler:
    return NetworkAssembler.rflow_scheduler(
        num_inference_steps=3, input_img_size_numel=2048,
    )


class RecordingUnet:
    """测试仪器：委托真实 UNet 前向、记录每次调用的输入组织。"""

    def __init__(self, unet: DiffusionModelUNetMaisi) -> None:
        self._unet = unet
        self.calls: list[dict] = []

    def __call__(self, **kwargs: object) -> torch.Tensor:
        self.calls.append({
            "batch": int(kwargs["x"].shape[0]),
            "timesteps": kwargs["timesteps"].tolist(),
            "labels": kwargs["class_labels"].tolist(),
        })
        return self._unet(**kwargs)


class TestTrajectoryCursor:
    """轨迹游标自持：快照防共享调度器被复写（spec：轨迹游标自持）。"""

    @pytest.fixture
    def cursor(self, scheduler: RFlowScheduler) -> TrajectoryCursor:
        return TrajectoryCursor(scheduler)

    def test_snapshots_timesteps_against_scheduler_rewrite(
        self, scheduler: RFlowScheduler,
    ) -> None:
        cursor = TrajectoryCursor(scheduler)
        scheduler.timesteps += 1  # 共享调度器被原地复写
        scheduler.set_timesteps(num_inference_steps=3, input_img_size_numel=32**3)
        assert cursor.timesteps.tolist() == [1000, 442, 165]  # 快照不受影响

    def test_next_timesteps_shift_with_zero_padding(
        self, cursor: TrajectoryCursor,
    ) -> None:
        assert cursor.next_timesteps.tolist() == [442, 165, 0]  # 按位前移、末位补 0

    def test_sigma_level_and_delta(self, cursor: TrajectoryCursor) -> None:
        assert cursor.num_steps == 3
        assert cursor.sigma_level(0) == pytest.approx(1.0)
        assert cursor.sigma_level(1) == pytest.approx(0.442)
        assert cursor.delta_s(1) == pytest.approx(0.277)
        assert cursor.timestep(2) == 165
        assert cursor.next_timestep(2) == 0

    def test_delta_s_matches_monai_step_dt_bitwise(
        self, cursor: TrajectoryCursor,
    ) -> None:
        """Δs 与 MONAI step() 内部 dt 同式同精度（η=0 逐位 parity 的前提）。"""
        expected = float(442 - 165) / 1000
        assert cursor.delta_s(1).hex() == expected.hex()


class TestSdeKernel:
    """单步 SDE 高斯核：η 参数化、s_max 钳制、η=0 精确退化为 MONAI step()。"""

    @pytest.fixture
    def kernel(self) -> SdeKernel:
        return SdeKernel(eta=0.7, s_max=0.999)

    @pytest.fixture
    def sample_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(3)
        return torch.randn(2, 4, 16, 16, 8), torch.randn(2, 4, 16, 16, 8)

    def test_eta0_matches_monai_step_bitwise(
        self, scheduler: RFlowScheduler, sample_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """η=0 时单步核与 MONAI RFlowScheduler 确定性步 fp32 逐位一致（AC）。"""
        x, v = sample_state
        noise = torch.randn_like(x)
        expected, _ = scheduler.step(v, 442, x, 165)
        outcome = SdeKernel(eta=0.0, s_max=0.999).transition(
            x, v, sigma_level=0.442, delta_s=float(442 - 165) / 1000, noise=noise,
        )
        assert torch.equal(outcome.sample, expected)

    def test_eta0_mean_is_deterministic_step(
        self, sample_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, v = sample_state
        outcome = SdeKernel(eta=0.0, s_max=0.999).transition(
            x, v, sigma_level=0.442, delta_s=0.277,
        )
        assert torch.equal(outcome.mean, x + v * 0.277)
        assert outcome.std == 0.0

    def test_s_max_clamps_singular_point(self, kernel: SdeKernel) -> None:
        """s=1（纯噪端）经 s_max 钳制后 g 有限、转移无 NaN/Inf。"""
        x = torch.randn(1, 4, 16, 16, 8)
        v = torch.randn(1, 4, 16, 16, 8)
        outcome = kernel.transition(x, v, sigma_level=1.0, delta_s=0.021)
        assert torch.isfinite(outcome.sample).all()
        expected_std = 0.7 * (0.999 / 0.001) ** 0.5 * 0.021 ** 0.5
        assert outcome.std == pytest.approx(expected_std)

    def test_std_matches_gaussian_kernel_formula(
        self, kernel: SdeKernel,
    ) -> None:
        x = torch.randn(1, 4, 16, 16, 8)
        v = torch.randn(1, 4, 16, 16, 8)
        outcome = kernel.transition(x, v, sigma_level=0.442, delta_s=0.277)
        expected_std = 0.7 * (0.442 / 0.558) ** 0.5 * 0.277 ** 0.5
        assert outcome.std == pytest.approx(expected_std)

    def test_log_prob_of_mean_is_gaussian_peak(
        self, kernel: SdeKernel, sample_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """log π(μ) = −log σ − ½log 2π（高斯核密度对非 batch 维取均值）。"""
        x, v = sample_state
        outcome = kernel.transition(x, v, sigma_level=0.442, delta_s=0.277)
        log_prob = kernel.log_prob(outcome.mean, outcome)
        expected = -torch.log(torch.tensor(outcome.std)) - 0.5 * torch.log(
            torch.tensor(2.0 * torch.pi),
        )
        assert log_prob.shape == (2,)
        assert torch.allclose(log_prob, expected.expand(2))

    def test_log_prob_decays_with_squared_offset(
        self, kernel: SdeKernel, sample_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        x, v = sample_state
        outcome = kernel.transition(x, v, sigma_level=0.442, delta_s=0.277)
        at_mean = kernel.log_prob(outcome.mean, outcome)
        one_sigma_off = kernel.log_prob(outcome.mean + outcome.std, outcome)
        assert torch.allclose(at_mean - one_sigma_off, torch.full((2,), 0.5))

    def test_log_prob_at_eta0_rejected(
        self, sample_state: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """η=0 是确定性步、无高斯密度可求：显式拒绝而非静默返回。"""
        x, v = sample_state
        outcome = SdeKernel(eta=0.0, s_max=0.999).transition(
            x, v, sigma_level=0.442, delta_s=0.277,
        )
        with pytest.raises(ValueError, match="确定性步"):
            SdeKernel(eta=0.0, s_max=0.999).log_prob(outcome.sample, outcome)

    def test_zero_sigma_level_rejected(self, kernel: SdeKernel) -> None:
        x = torch.randn(1, 4, 16, 16, 8)
        with pytest.raises(ValueError, match="sigma"):
            kernel.transition(x, x, sigma_level=0.0, delta_s=0.0)


class TestCfgCombinedField:
    """组1 CFG=10 组合场：batch 组织、序、全零 label、全组复用逐字对齐基座。"""

    LABEL = 29  # t1n 的 modality token

    @pytest.fixture
    def recording(self, fixture_unet: DiffusionModelUNetMaisi) -> RecordingUnet:
        return RecordingUnet(fixture_unet)

    @pytest.fixture
    def field(self, recording: RecordingUnet) -> CfgCombinedField:
        return CfgCombinedField(recording)

    @pytest.fixture
    def condition(self) -> RolloutCondition:
        return RolloutCondition(label=torch.tensor([self.LABEL]), spacing=SPACING)

    def test_velocity_single_paired_forward_cond_then_uncond(
        self, field: CfgCombinedField, recording: RecordingUnet,
        condition: RolloutCondition,
    ) -> None:
        """batch=2 单次前向按 chunk(2) 拆分、序 [cond, uncond]、无条件=全零 label。"""
        x = torch.randn(1, *LATENT_SHAPE)
        field.velocity(x, timesteps=442, condition=condition)
        assert len(recording.calls) == 1
        call = recording.calls[0]
        assert call["batch"] == 2  # batch=2 单次前向（基座组织）
        assert call["labels"] == [self.LABEL, 0]  # 序 [cond, uncond]、uncond=全零 label
        assert call["timesteps"] == [442, 442]

    def test_velocity_matches_spec_formula(
        self, field: CfgCombinedField, fixture_unet: DiffusionModelUNetMaisi,
        condition: RolloutCondition,
    ) -> None:
        """数值 = v_uncond + 10·(v_cond − v_uncond)（组合公式逐字复刻）。"""
        torch.manual_seed(11)
        x = torch.randn(1, *LATENT_SHAPE)
        with torch.no_grad():
            paired = fixture_unet(
                x=torch.cat((x, x)),
                timesteps=torch.tensor([442, 442]),
                class_labels=torch.tensor([self.LABEL, 0]),
                spacing_tensor=torch.cat((SPACING, SPACING)),
            )
            v_cond, v_uncond = torch.chunk(paired, 2)
            expected = v_uncond + 10.0 * (v_cond - v_uncond)
            actual = field.velocity(x, timesteps=442, condition=condition)
        assert torch.equal(actual, expected)

    def test_group_velocity_two_batch1_forwards_reused_across_group(
        self, field: CfgCombinedField, recording: RecordingUnet,
        condition: RolloutCondition,
    ) -> None:
        """全组复用：无条件分支 batch=1 一次评估（共两次 batch=1 前向），
        组合结果 expand 成 G——不随条件分支复制 G 倍（G²RPO 效率技巧）。"""
        x = torch.randn(1, *LATENT_SHAPE)
        v = field.group_velocity(x, timesteps=442, condition=condition, group_size=12)
        assert v.shape == (12, *LATENT_SHAPE)
        assert len(recording.calls) == 2  # cond + uncond 各一次，均 batch=1
        assert all(call["batch"] == 1 for call in recording.calls)
        assert [call["labels"] for call in recording.calls] == [
            [self.LABEL], [0],
        ]

    def test_group_velocity_values_match_combined_field(
        self, field: CfgCombinedField, fixture_unet: DiffusionModelUNetMaisi,
        condition: RolloutCondition,
    ) -> None:
        """复用路径与 batch=2 组合场数值一致（batch 尺寸仅引入 fp32 舍入差）。"""
        torch.manual_seed(5)
        x = torch.randn(1, *LATENT_SHAPE)
        expanded_condition = RolloutCondition(
            label=torch.tensor([self.LABEL] * 12), spacing=SPACING.expand(12, -1),
        )
        with torch.no_grad():
            expanded = field.velocity(
                x.expand(12, -1, -1, -1, -1), timesteps=442,
                condition=expanded_condition,
            )
            grouped = field.group_velocity(
                x, timesteps=442, condition=condition, group_size=12,
            )
        # batch 组织的 fp32 归约序差（实测 max |Δ| ≈ 1e-5）
        assert torch.allclose(grouped, expanded, rtol=1e-4, atol=2e-5)


class TestRolloutSampler:
    """Anchor 轨迹（η=0 逐步存 latent）→ 第 k 步 SDE 扰动 G 方向 → ODE 续跑。"""

    @pytest.fixture
    def sampler(
        self, fixture_unet: DiffusionModelUNetMaisi, scheduler: RFlowScheduler,
    ) -> RolloutSampler:
        torch.manual_seed(7)
        cursor = TrajectoryCursor(scheduler)
        field = CfgCombinedField(fixture_unet)
        return RolloutSampler(
            field, SdeKernel(eta=0.7, s_max=0.999), cursor,
        )

    @pytest.fixture
    def conditions(self) -> RolloutCondition:
        return RolloutCondition(label=torch.tensor([29]), spacing=SPACING)

    def test_anchor_trajectory_stores_every_step(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        torch.manual_seed(0)
        initial = torch.randn(1, *LATENT_SHAPE)
        trajectory = sampler.anchor_trajectory(initial, conditions)
        assert len(trajectory) == 4  # 初始噪声 + 3 步（3 步日程）
        assert torch.equal(trajectory[0], initial)
        assert all(torch.isfinite(step).all() for step in trajectory)

    def test_anchor_trajectory_is_deterministic(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        torch.manual_seed(0)
        initial = torch.randn(1, *LATENT_SHAPE)
        first = sampler.anchor_trajectory(initial, conditions)
        second = sampler.anchor_trajectory(initial, conditions)
        assert all(torch.equal(a, b) for a, b in zip(first, second))

    def test_perturb_group_produces_distinct_directions_with_log_prob(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        torch.manual_seed(0)
        initial = torch.randn(1, *LATENT_SHAPE)
        anchor = sampler.anchor_trajectory(initial, conditions)
        noise = torch.randn(12, *LATENT_SHAPE)
        directions, log_probs = sampler.perturb_group(
            anchor[1], index=1, condition=conditions, noise=noise,
        )
        assert directions.shape == (12, *LATENT_SHAPE)
        assert log_probs.shape == (12,)
        assert torch.isfinite(log_probs).all()
        for i in range(12):  # G 个方向各不相同（噪声注入生效）
            for j in range(i + 1, 12):
                assert not torch.equal(directions[i], directions[j])

    def test_eta0_perturbation_path_matches_anchor_step(
        self, fixture_unet: DiffusionModelUNetMaisi, scheduler: RFlowScheduler,
        conditions: RolloutCondition,
    ) -> None:
        """η=0 的扰动路径退化为确定性步：group_velocity + η=0 核的组合
        输出与 anchor 下一步一致（batch 组织差异仅 fp32 舍入级；
        log-prob 在 η=0 无定义、由 kernel 显式拒绝）。"""
        torch.manual_seed(0)
        cursor = TrajectoryCursor(scheduler)
        field = CfgCombinedField(fixture_unet)
        kernel = SdeKernel(eta=0.0, s_max=0.999)
        sampler = RolloutSampler(field, kernel, cursor)
        initial = torch.randn(1, *LATENT_SHAPE)
        anchor = sampler.anchor_trajectory(initial, conditions)
        velocity = field.group_velocity(
            anchor[1], cursor.timestep(1), conditions, group_size=12,
        )
        noise = torch.randn(12, *LATENT_SHAPE)
        outcome = kernel.transition(
            anchor[1].expand(12, -1, -1, -1, -1), velocity,
            cursor.sigma_level(1), cursor.delta_s(1), noise=noise,
        )
        # η=0 扰动路径退化为确定性步；batch 组织的 fp32 归约序差（≈1e-5）
        assert torch.allclose(
            outcome.sample, anchor[2].expand(12, -1, -1, -1, -1),
            rtol=1e-4, atol=2e-5,
        )

    def test_evaluate_log_prob_matches_recorded(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        """采样场重算口径：重算 log-prob 与扰动时记录值逐位一致（同权重）。"""
        torch.manual_seed(0)
        initial = torch.randn(1, *LATENT_SHAPE)
        anchor = sampler.anchor_trajectory(initial, conditions)
        noise = torch.randn(12, *LATENT_SHAPE)
        directions, recorded = sampler.perturb_group(
            anchor[1], index=1, condition=conditions, noise=noise,
        )
        recomputed = sampler.evaluate_log_prob(
            anchor[1], index=1, condition=conditions, samples=directions,
        )
        assert torch.equal(recorded, recomputed)

    def test_continue_to_terminal_is_deterministic_ode(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        torch.manual_seed(0)
        initial = torch.randn(1, *LATENT_SHAPE)
        anchor = sampler.anchor_trajectory(initial, conditions)
        noise = torch.randn(12, *LATENT_SHAPE)
        directions, _ = sampler.perturb_group(
            anchor[1], index=1, condition=conditions, noise=noise,
        )
        first = sampler.continue_to_terminal(directions, index=1, condition=conditions)
        second = sampler.continue_to_terminal(directions, index=1, condition=conditions)
        assert first.shape == (12, *LATENT_SHAPE)
        assert torch.equal(first, second)  # 续跑确定性（ODE，无随机性）
        assert not torch.equal(first, anchor[3].expand(12, -1, -1, -1, -1))  # 扰动已改道

    def test_continue_from_final_step_is_identity(
        self, sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        latents = torch.randn(3, *LATENT_SHAPE)
        terminal = sampler.continue_to_terminal(latents, index=2, condition=conditions)
        assert torch.equal(terminal, latents)  # 末步之后无续跑空间


class TestGranularContinuation:
    """λ 粒度续跑（MGAI，research/granular-grpo.md §3）：按时间步间隔 λ
    抽稀 sigma 日程做确定性 ODE 积分——λ=1 逐步（既有行为），λ>1 先走一个
    逐步细步再按 λ 抽稀。参考实现的访问日程 ``suffix =
    sigma_schedule[eta_step+2::g]`` 从 k+2 起才抽稀：扰动后 latent 位于
    k+1，首段细步与 λ=1 同粒度，之后访问点相隔 λ，末段一律直达 σ=0 终点。"""

    NUM_STEPS = 5

    @pytest.fixture
    def wide_scheduler(self) -> RFlowScheduler:
        return NetworkAssembler.rflow_scheduler(
            num_inference_steps=self.NUM_STEPS, input_img_size_numel=2048,
        )

    @pytest.fixture
    def wide_cursor(self, wide_scheduler: RFlowScheduler) -> TrajectoryCursor:
        return TrajectoryCursor(wide_scheduler)

    @pytest.fixture
    def wide_sampler(
        self, fixture_unet: DiffusionModelUNetMaisi, wide_scheduler: RFlowScheduler,
    ) -> RolloutSampler:
        cursor = TrajectoryCursor(wide_scheduler)
        field = CfgCombinedField(fixture_unet)
        return RolloutSampler(field, SdeKernel(eta=0.7, s_max=0.999), cursor)

    @pytest.fixture
    def conditions(self) -> RolloutCondition:
        return RolloutCondition(label=torch.tensor([29]), spacing=SPACING)

    def test_delta_s_with_stride_matches_schedule_arithmetic(
        self, wide_cursor: TrajectoryCursor,
    ) -> None:
        """跨 stride 的 Δs 与 MONAI dt 同式：(t_prev − t_visit)/1000。"""
        by_hand = (
            wide_cursor.timestep(0) - wide_cursor.timestep(2)
        ) / 1000
        assert wide_cursor.delta_s(0, stride=2) == by_hand

    def test_delta_s_terminal_segment_reaches_zero_sigma(
        self, wide_cursor: TrajectoryCursor,
    ) -> None:
        """末段（visit 越过日程末端）：Δs = t_prev/1000（直落 σ=0）。"""
        assert wide_cursor.delta_s(
            self.NUM_STEPS - 1, stride=2,
        ) == wide_cursor.timestep(self.NUM_STEPS - 1) / 1000

    def test_delta_s_default_stride_is_monai_dt(
        self, wide_cursor: TrajectoryCursor,
    ) -> None:
        """stride=1 与既有逐位口径一致（MONAI step 内部 dt）。"""
        assert wide_cursor.delta_s(1).hex() == (
            float(wide_cursor.timestep(1) - wide_cursor.timestep(2)) / 1000
        ).hex()

    def test_stride1_matches_stepwise_continuation(
        self, wide_sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        """λ=1 与逐步续跑逐位一致（默认参数回归：不破坏既有行为）。"""
        torch.manual_seed(0)
        latents = torch.randn(3, *LATENT_SHAPE)
        stepped = wide_sampler.continue_to_terminal(latents, 1, conditions)
        strided = wide_sampler.continue_to_terminal(
            latents, 1, conditions, stride=1,
        )
        assert torch.equal(stepped, strided)

    def test_stride2_follows_decimated_schedule(
        self, wide_sampler: RolloutSampler, wide_cursor: TrajectoryCursor,
        fixture_unet: DiffusionModelUNetMaisi, conditions: RolloutCondition,
    ) -> None:
        """λ=2 续跑（index=2）：访问日程 suffix = σ[4::2] = [σ4]——首段
        细步 (3→4) 与 λ=1 同粒度，末段 (4→σ=0) 单段大步直达终点。"""
        torch.manual_seed(0)
        latents = torch.randn(2, *LATENT_SHAPE)
        outcome = wide_sampler.continue_to_terminal(latents, 2, conditions, stride=2)
        deterministic = SdeKernel.deterministic(s_max=0.999)
        field = CfgCombinedField(fixture_unet)
        x = latents
        velocity = field.velocity(x, wide_cursor.timestep(3), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(3),
            wide_cursor.sigma_level(3) - wide_cursor.sigma_level(4),
        ).sample
        velocity = field.velocity(x, wide_cursor.timestep(4), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(4),
            wide_cursor.sigma_level(4),
        ).sample
        assert torch.allclose(outcome, x, rtol=1e-6, atol=1e-7)

    def test_stride2_fine_step_then_decimated_visits(
        self, wide_sampler: RolloutSampler, wide_cursor: TrajectoryCursor,
        fixture_unet: DiffusionModelUNetMaisi, conditions: RolloutCondition,
    ) -> None:
        """λ=2 续跑（index=0）：访问日程 suffix = σ[2::2] = [σ2, σ4]——
        首段细步 (1→2) 与 λ=1 同粒度（参考实现从 k+2 起才抽稀，漏掉该步
        会跳过普通续跑的第一个端点、改变全部粗粒度终点），之后大步
        (2→4)，末段 (4→σ=0)。"""
        torch.manual_seed(0)
        latents = torch.randn(2, *LATENT_SHAPE)
        outcome = wide_sampler.continue_to_terminal(latents, 0, conditions, stride=2)
        deterministic = SdeKernel.deterministic(s_max=0.999)
        field = CfgCombinedField(fixture_unet)
        x = latents
        velocity = field.velocity(x, wide_cursor.timestep(1), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(1),
            wide_cursor.sigma_level(1) - wide_cursor.sigma_level(2),
        ).sample
        velocity = field.velocity(x, wide_cursor.timestep(2), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(2),
            wide_cursor.sigma_level(2) - wide_cursor.sigma_level(4),
        ).sample
        velocity = field.velocity(x, wide_cursor.timestep(4), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(4),
            wide_cursor.sigma_level(4),
        ).sample
        assert torch.allclose(outcome, x, rtol=1e-6, atol=1e-7)

    def test_stride2_leads_with_fine_step(
        self, wide_sampler: RolloutSampler, wide_cursor: TrajectoryCursor,
        fixture_unet: DiffusionModelUNetMaisi, conditions: RolloutCondition,
    ) -> None:
        """λ=2 续跑（index=1）：访问日程 suffix = σ[3::2] = [σ3]——首段
        细步 (2→3) 与 λ=1 同粒度，末段 (3→σ=0)（粒度与 λ=1 真实分叉）。"""
        torch.manual_seed(0)
        latents = torch.randn(2, *LATENT_SHAPE)
        outcome = wide_sampler.continue_to_terminal(latents, 1, conditions, stride=2)
        deterministic = SdeKernel.deterministic(s_max=0.999)
        field = CfgCombinedField(fixture_unet)
        x = latents
        velocity = field.velocity(x, wide_cursor.timestep(2), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(2),
            wide_cursor.sigma_level(2) - wide_cursor.sigma_level(3),
        ).sample
        velocity = field.velocity(x, wide_cursor.timestep(3), conditions)
        x = deterministic.transition(
            x, velocity, wide_cursor.sigma_level(3),
            wide_cursor.sigma_level(3),
        ).sample
        assert torch.allclose(outcome, x, rtol=1e-6, atol=1e-7)

    def test_strides_diverge_after_schedule_shortens(
        self, wide_sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        """不同 λ 的续跑终点不同（粒度差异真实存在——MGAI 的前提）。

        index=1（剩余 3 步日程）：λ=1 走 (2→3)→(3→4)→(4→σ0)，λ=2 走
        细步 (2→3) + 大步 (3→σ0)——首个细步之后分叉。（index=2 的剩余
        2 步日程下两粒度访问日程重合：suffix = σ[4::2] = [σ4] 与逐步
        同路径，分叉须剩余日程 > λ 才存在。）"""
        torch.manual_seed(0)
        latents = torch.randn(2, *LATENT_SHAPE)
        stride1 = wide_sampler.continue_to_terminal(latents, 1, conditions, stride=1)
        stride2 = wide_sampler.continue_to_terminal(latents, 1, conditions, stride=2)
        assert not torch.equal(stride1, stride2)

    def test_stride_beyond_schedule_is_single_terminal_leap(
        self, wide_sampler: RolloutSampler, conditions: RolloutCondition,
    ) -> None:
        """stride 越过剩余日程：单段直达 σ=0（无中间访问点）。"""
        torch.manual_seed(0)
        latents = torch.randn(1, *LATENT_SHAPE)
        leaped = wide_sampler.continue_to_terminal(latents, 1, conditions, stride=8)
        single = wide_sampler.continue_to_terminal(latents, 1, conditions, stride=4)
        assert torch.equal(leaped, single)  # 同为 (1 → σ=0) 单段
