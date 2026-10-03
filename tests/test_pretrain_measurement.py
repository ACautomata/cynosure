"""预训练测量面测试（#221 决议 5-9 + #198 锚平移接棒，ADR-0018）。

主控单点测量模板（``MeasurementTemplate``：seed+19 显式直锚、复位
→ 条件构造 → 按卡序逐段抽 ε）与卡轴连续段分片（``ShardPlan``：前余
均分纯函数）的契约面。#198 三组锚的测量批 ε 语义在此**平移宿主**：
「同复位状态主控逐段抽 vs 全量对应行逐位」——顺序流等价性的唯一
机器守护面（行宽 ≡ 0 (mod 16) 的装配期断言见
``LatentManifest.assert_measurement_row_width`` 的契约锁）。

零速度 UNet 桩使确定性 ODE 退化为恒等映射——测量批输出可手工复算为
x_s = (1−s)·x + s·ε（复用 test_assembly 的桩与场景件）。
"""

import pytest
import torch

from cynosure.policy.schedules import SingleConditionSchedules
from cynosure.pretrain.measurement import (
    MEASUREMENT_TEMPLATE_OFFSET,
    MeasurementTemplate,
)
from cynosure.pretrain.sharding import ShardPlan
from cynosure.reward.artifacts import LatentManifest, PoolEntry

from tests.test_assembly import (
    CONDITION,
    NUM_STEPS,
    SHAPE,
    TRAIN_STEPS,
    AssemblyScenario,
    StubConditions,
)

TEMPLATE_SEED = 77


@pytest.fixture
def scenario(tmp_path) -> AssemblyScenario:
    return AssemblyScenario(tmp_path)


def template(
    scenario: AssemblyScenario,
    seed: int = TEMPLATE_SEED,
    train_steps: tuple[int, ...] = TRAIN_STEPS,
    num_steps: int = NUM_STEPS,
    conditions=None,
) -> MeasurementTemplate:
    """主控测量模板（StubConditions 的组1 条件构造缺省——不耗 RNG）。"""
    return MeasurementTemplate(
        seed,
        SingleConditionSchedules(
            num_inference_steps=num_steps, input_img_size_numel=2048,
        ),
        conditions if conditions is not None else StubConditions(),
        train_steps,
    )


class TestShardPlan:
    """卡轴连续段前余均分（#221 决议 8：现行 ``_shard_bounds`` 纯函数化）。

    连续段使主控按卡序拼接各卡产出还原全量排列序；K×D 项切 D 段
    = base K rem 0 是 real 侧每卡配额的等分特例。"""

    def test_front_remainder_split(self) -> None:
        """前余均分：余数摊前段（7 卷 3 卡 = (3, 2, 2)），段连续无缝。"""
        plan = ShardPlan.split(7, 3)
        assert plan.bounds == ((0, 3), (3, 5), (5, 7))
        assert plan.total == 7

    def test_exact_division_is_base_k_rem_zero(self) -> None:
        """K×D 切 D 段 = base K rem 0（每卡恰 K——real 侧配额的等分
        特例）。"""
        plan = ShardPlan.split(16, 4)
        assert plan.bounds == ((0, 4), (4, 8), (8, 12), (12, 16))

    def test_single_card_is_full_span(self) -> None:
        """单卡退化为全量一段（分片机制的 World-1 特例）。"""
        plan = ShardPlan.split(5, 1)
        assert plan.bounds == ((0, 5),)

    def test_matches_legacy_shard_bounds_formula(self) -> None:
        """与现行 ``_shard_bounds`` 公式逐卡恒等（``rank * base +
        min(rank, remainder)`` 起点、base + 前余指示终点的同一函数）。"""
        for total, cards in ((14, 4), (8, 3), (5, 5), (9, 2), (12, 5)):
            base, remainder = divmod(total, cards)
            plan = ShardPlan.split(total, cards)
            for card in range(cards):
                start = card * base + min(card, remainder)
                stop = start + base + (1 if card < remainder else 0)
                assert plan.slice_of(card) == (start, stop), (total, cards)

    def test_empty_tail_segments_legal(self) -> None:
        """全量 < 卡数的空尾段合法（消费面由装配期「每条件 held-out
        ≥ 卡数」守卫把住——计划纯函数不重复该守卫）。"""
        plan = ShardPlan.split(2, 4)
        assert plan.bounds == ((0, 1), (1, 2), (2, 2), (2, 2))
        assert plan.slice_of(3) == (2, 2)

    def test_continuity_violation_rejected(self) -> None:
        """手工构造的非连续计划拒绝（计划值的结构不变式构造期断言）。"""
        with pytest.raises(ValueError, match="连续"):
            ShardPlan(bounds=((0, 3), (4, 7)))

    def test_negative_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="≥ 0"):
            ShardPlan.split(-1, 2)

    def test_zero_cards_rejected(self) -> None:
        with pytest.raises(ValueError, match="≥ 1"):
            ShardPlan.split(4, 0)


class TestMeasurementTemplate:
    """主控单点测量模板（seed+19 显式直锚 + 逐段 ε 顺序流等价）。"""

    def test_template_state_is_seed_plus_nineteen(self) -> None:
        """模板复位状态 = ``seed+19`` 显式直锚（#221 决议 9 恒等面读数：
        数值与现行 ``shared_seed+19`` 恒等；偏移**值**由本测钉住——
        生产常量被误改时此处显式红）。模板构造只需注入替身（schedules/
        conditions），不经场景工件。"""
        assert MEASUREMENT_TEMPLATE_OFFSET == 19
        holder = MeasurementTemplate(
            TEMPLATE_SEED,
            SingleConditionSchedules(
                num_inference_steps=NUM_STEPS, input_img_size_numel=2048,
            ),
            StubConditions(),
            TRAIN_STEPS,
        )
        probe = torch.Generator().manual_seed(TEMPLATE_SEED + 19)
        reference = torch.randn((3, *SHAPE), generator=probe)
        draw = holder.draw(CONDITION, SHAPE, ShardPlan.split(3, 1))
        assert torch.equal(draw.noises[0], reference)

    def test_repeated_draw_bitwise_identical(
        self, scenario: AssemblyScenario,
    ) -> None:
        """测量的可复算性：同输入逐次 draw 逐位同输出——模板复位到同
        一状态（不受「本 run 此前测量过几次」影响；报告值可复算与 gate
        判定可比的结构前提）。"""
        holder = template(scenario)
        plan = ShardPlan.split(5, 1)
        first = holder.draw(CONDITION, SHAPE, plan)
        for _ in range(3):
            again = holder.draw(CONDITION, SHAPE, plan)
            assert torch.equal(again.noises[0], first.noises[0])
            assert again.sigmas == first.sigmas

    def test_sigma_rotation_over_full_positions(
        self, scenario: AssemblyScenario,
    ) -> None:
        """σ 定序轮转按**全量位次**：全量列表 = 候选第 i % |M| 位，
        分段切片 = 全量列表切片（零偏移算术——``volume_offset`` 退役
        后轮转偏移的实现形态）。"""
        holder = template(scenario, train_steps=(1, 2, 3), num_steps=5)
        cursor = SingleConditionSchedules(
            num_inference_steps=5, input_img_size_numel=2048,
        ).cursor(CONDITION)
        candidates = [cursor.sigma_level(step) for step in (1, 2, 3)]
        plan = ShardPlan.split(7, 3)
        draw = holder.draw(CONDITION, SHAPE, plan)
        assert draw.sigmas == tuple(
            candidates[index % len(candidates)] for index in range(7)
        )
        for card in range(3):
            start, stop = plan.slice_of(card)
            assert draw.sigmas_of(card, plan) == tuple(
                draw.sigmas[start:stop],
            )

    def test_segmented_noise_matches_full_batch_bitwise(
        self, scenario: AssemblyScenario,
    ) -> None:
        """#198 平移锚（决议 7）：**同复位状态主控逐段抽 ≡ 全量对应行
        逐位**——float32 ``randn`` 的 16 元素块 Box-Muller 顺序流性质
        （行宽 ≡ 0 (mod 16) 前提，manifest 装配期断言守护）。分段抽
        （D 卡连续段）与全量一段抽的 ε 在对应行逐位一致——多卡测量批
        对单卡可复算的数值前提。"""
        volumes = 8
        full_plan = ShardPlan.split(volumes, 1)
        multi_plan = ShardPlan.split(volumes, 3)
        full = template(scenario).draw(CONDITION, SHAPE, full_plan)
        segmented = template(scenario).draw(CONDITION, SHAPE, multi_plan)
        assert torch.equal(
            torch.cat(list(segmented.noises)), full.noises[0],
        )
        for card, (start, stop) in enumerate(multi_plan.bounds):
            assert torch.equal(
                segmented.noises[card], full.noises[0][start:stop],
            ), card

    def test_forward_count_sums_segmentwise(
        self, scenario: AssemblyScenario,
    ) -> None:
        """成本读数的分段可加性（#221 决议 18「Σ卡任务读数」）：全量
        σ 列表的步数和 = 各连续段步数和之和（加法结合恒等）。"""
        holder = template(scenario, train_steps=(1, 3), num_steps=5)
        plan = ShardPlan.split(7, 3)
        draw = holder.draw(CONDITION, SHAPE, plan)
        total = holder.forward_count(CONDITION, draw.sigmas)
        parts = sum(
            holder.forward_count(
                CONDITION, draw.sigmas_of(card, plan),
            )
            for card in range(3)
        )
        assert total == parts > 0

    def test_forward_count_strictly_below_full_ode(
        self, scenario: AssemblyScenario,
    ) -> None:
        """每卷步数严格小于「量产全 ODE」（num_steps 步）——量产退役
        的数值口径（重构从日程中段起步）。"""
        holder = template(scenario, train_steps=(1, 3), num_steps=5)
        volumes = 4
        draw = holder.draw(CONDITION, SHAPE, ShardPlan.split(volumes, 1))
        forwards = holder.forward_count(CONDITION, draw.sigmas)
        assert forwards < volumes * 5

    def test_named_registry_streams_untouched(
        self, scenario: AssemblyScenario,
    ) -> None:
        """测量模板独立于 RNG 注册表（测量不参与续训——现行口径）：
        构造与 draw 不消耗四条命名流的任何位置（模板 seed+19 显式直锚，
        与 recon 流解耦）。"""
        from cynosure.train.rng import TrainingRngStreams

        streams = TrainingRngStreams(seed=0)
        before = {
            name: generator.get_state().clone()
            for name, generator in streams.named().items()
        }
        holder = template(scenario, seed=0)
        holder.draw(CONDITION, SHAPE, ShardPlan.split(3, 2))
        for name, state in before.items():
            assert torch.equal(streams.named()[name].get_state(), state), name

    def test_condition_draw_stays_off_policy_stream(
        self, scenario: AssemblyScenario,
    ) -> None:
        """组2（跨模态）条件构造穿模板流（源对 + 源条目抽取消耗**测量
        模板**，不漂移 policy 主流——流隔离契约）；逐次 draw 同起手
        复位 ⇒ 条件与 ε 逐位同输出（组2 测量同款可复算）。"""
        from cynosure.policy.condition import ModalityMapping
        from cynosure.train.rollout import CrossModalConditionSampler, SourceLatentPool
        from cynosure.train.rng import TrainingRngStreams

        cross_modal = CrossModalConditionSampler(
            ModalityMapping({"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}),
            [("t1n", "t1c")],
            SourceLatentPool(
                LatentManifest.load(scenario.pool_path, kind="real_pool"),
                torch.device("cpu"),
            ),
            torch.Generator().manual_seed(3),
            torch.device("cpu"),
        )
        streams = TrainingRngStreams(seed=0)
        rollout_before = streams.rollout.get_state().clone()
        holder = template(scenario, conditions=cross_modal)
        plan = ShardPlan.split(3, 1)
        first = holder.draw("t1c", SHAPE, plan)
        second = holder.draw("t1c", SHAPE, plan)
        assert torch.equal(first.noises[0], second.noises[0])
        assert torch.equal(
            first.condition.label, second.condition.label,
        )
        assert torch.equal(first.condition.source_latent,
                           second.condition.source_latent)
        assert torch.equal(streams.rollout.get_state(), rollout_before)

    def test_m_term_guard_rejects_terminal_step(
        self, scenario: AssemblyScenario,
    ) -> None:
        """日程末位不是合法候选（σ 恒 > 0 的中段日程点——末位的零步
        档位在测量期即拒）。"""
        for illegal in ((1, 4), (1, 5)):
            with pytest.raises(ValueError, match="越界"):
                template(
                    scenario, train_steps=illegal, num_steps=5,
                ).draw(CONDITION, SHAPE, ShardPlan.split(2, 1))


class TestMeasurementReconstruction:
    """测量批端到端手工复算（draw → 重构核）：零速度 ODE 下 fake =
    (1−σ)·real + σ·ε 可逐元素复算——主控抽取与卡任务重构的组合契约。"""

    def test_reconstruction_matches_hand_computation(
        self, scenario: AssemblyScenario,
    ) -> None:
        assembler = scenario.assembler(real_sampler=None)
        holder = template(scenario, train_steps=(1, 2))
        volumes = 6
        plan = ShardPlan.split(volumes, 2)
        draw = holder.draw(CONDITION, SHAPE, plan)
        reals = torch.randn(volumes, *SHAPE)
        candidates = assembler.candidate_sigmas(CONDITION)
        for card, (start, stop) in enumerate(plan.bounds):
            noise = draw.noises[card]
            # 手工复算：σ 经 float32 张量参与运算（生产路径的 levels
            # 张量口径——Python float 字面量走双精度标量广播，差 1 ulp）
            for row, index in enumerate(range(start, stop)):
                sigma = torch.tensor(candidates[index % len(candidates)])
                expected = (
                    reals[index] * (1.0 - sigma) + noise[row] * sigma
                )
                fakes = assembler.reconstruct(
                    reals[index:index + 1],
                    draw.condition,
                    [draw.sigmas[index]],
                    noise[row:row + 1],
                )
                assert torch.equal(fakes[0], expected), index

    def test_segmented_reconstruction_equals_full_rows(
        self, scenario: AssemblyScenario,
    ) -> None:
        """分段重构 ≡ 全量重构对应行（逐位）：重构逐样本独立 + σ/ε 按
        全量位次对齐——多卡测量批拼接后与单卡全量测量逐位一致。"""
        assembler = scenario.assembler(real_sampler=None)
        volumes = 6
        reals = torch.randn(volumes, *SHAPE)
        full = template(scenario).draw(
            CONDITION, SHAPE, ShardPlan.split(volumes, 1),
        )
        full_fakes = assembler.reconstruct(
            reals, full.condition, list(full.sigmas), full.noises[0],
        )
        plan = ShardPlan.split(volumes, 3)
        segmented = template(scenario).draw(CONDITION, SHAPE, plan)
        for card, (start, stop) in enumerate(plan.bounds):
            part = assembler.reconstruct(
                reals[start:stop],
                segmented.condition,
                list(segmented.sigmas_of(card, plan)),
                segmented.noises[card],
            )
            assert torch.equal(part, full_fakes[start:stop]), card


class TestMeasurementRowWidth:
    """行宽 ≡ 0 (mod 16) 的装配期断言（#221 决议 6）：顺序流等价性的
    机器前提——违者 fail-fast、文案引导词表/manifest 工件异常。"""

    @staticmethod
    def _manifest(
        shape: tuple[int, int, int, int] | None,
        global_shape: tuple[int, int, int, int] = (4, 16, 16, 8),
    ) -> LatentManifest:
        entry = PoolEntry(
            case_id="case-000",
            modality="t1n",
            latent="latents/0.pt",
            spacing=(100.0, 100.0, 100.0),
        )
        return LatentManifest(
            kind="heldout_real",
            encoder="fixture",
            latent_shape=global_shape,
            split_seed=0,
            split_sizes={"train": 1, "val": 0, "test": 0},
            entries=[entry],
            condition_latent_shapes=(
                None if shape is None else {"t1n": shape}
            ),
        )

    def test_non_multiple_of_sixteen_rejected(self) -> None:
        manifest = self._manifest((4, 5, 5, 5))  # numel = 500
        with pytest.raises(ValueError, match="16 的倍数"):
            manifest.assert_measurement_row_width()

    def test_multiple_of_sixteen_passes(self) -> None:
        self._manifest((4, 16, 16, 8)).assert_measurement_row_width()

    def test_single_domain_global_shape_checked(self) -> None:
        """单域（BraTS）工件无逐条件表：全局 ``latent_shape`` 同款断言
        （两态形状来源与 ``load_latent`` 同款解析）。"""
        self._manifest(
            None, global_shape=(4, 8, 8, 8),  # numel = 2048：通过
        ).assert_measurement_row_width()
        bad = self._manifest(
            None, global_shape=(3, 5, 5, 5),  # numel = 375：拒绝
        )
        with pytest.raises(ValueError, match="16 的倍数"):
            bad.assert_measurement_row_width()
