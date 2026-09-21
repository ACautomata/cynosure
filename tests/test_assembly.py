"""判别器更新批装配原语测试（ADR-0012，issue #170 AC）。

验收锚：配对批性质（fake ≠ real、同源配对、条件匹配）、s=0 重构恒等
（确定性 ODE 终点 = 起点）、同 seed 重放逐位一致（先抽 s 后抽 ε 的采
样契约）、s 抽样镜像日程（候选步点与 ConditionSchedules 按被优化步导
出一致、s=1 不在候选集）、专属随机流与训练/评测/AUC 流不交叉。

零速度 UNet 桩使确定性 ODE 退化为恒等映射——重构输出可手工复算为
x_s = (1−s)·x + s·ε（ODE 续跑零贡献），同源配对与抽取次序由此可观测。
"""

from pathlib import Path

import pytest
import torch

from cynosure.distributed import DistributedContext
from cynosure.fixtures import Fixture
from cynosure.policy.condition import ModalityMapping, RolloutCondition
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.numerics import AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.policy.schedules import SingleConditionSchedules
from cynosure.reward.assembly import PairBatch, ReconstructionAssembler
from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.sampler import RealPoolSampler
from cynosure.train.rollout import CrossModalConditionSampler, SourceLatentPool
from cynosure.train.rng import TrainingRngStreams

SHAPE = (4, 16, 16, 8)
SPACING = torch.tensor([[100.0, 100.0, 100.0]])
NUM_STEPS = 4
"""小锚日程步数（M 候选 = {1, 2}，s=1 奇异端与 σ=0 端都被排除）。"""
TRAIN_STEPS = (1, 2)
CONDITION = "t1n"
MEASUREMENT_OFFSET = 10
"""测量流 seed 偏移的字面量钉子（刻意不 import 生产常量：本测钉的是
偏移**值**——生产常量被误改时此处显式红，而非随改随过）。"""


def _expected_measurement_fakes(assembler, reals: torch.Tensor) -> torch.Tensor:
    """测量批的零速度 ODE 手工复算：重构 = 插值加噪（ODE 零贡献），
    同 seed 同 σ 定序（测量流批次起手复位 → 独立生成器取同一 noise）。"""
    candidates = assembler.candidate_sigmas(CONDITION)
    generator = torch.Generator().manual_seed(
        assembler._generator.initial_seed() + MEASUREMENT_OFFSET,
    )
    noise = torch.randn(reals.shape, generator=generator)
    expected = torch.empty_like(reals)
    for index in range(reals.shape[0]):
        # σ 经 float32 张量参与运算（生产路径的 levels 张量口径——
        # Python float 字面量走双精度标量广播，差 1 ulp）
        sigma = torch.tensor(candidates[index % len(candidates)])
        expected[index] = reals[index] * (1.0 - sigma) + noise[index] * sigma
    return expected


class ZeroVelocityUnet:
    """零速度前向桩：确定性 ODE 每步 x' = x（重构 = 插值加噪本体的观测面）。"""

    def __call__(self, **kwargs: object) -> torch.Tensor:
        return torch.zeros_like(kwargs["x"])


class StubConditions:
    """条件构造替身（ConditionSampler 协议，组1 sample_target 语义）：
    目标条件 = 固定 label/spacing + 条件名，确定性不耗 RNG。"""

    def sample(
        self, generator: torch.Generator | None = None,
    ) -> tuple[RolloutCondition, str]:
        return self.sample_target(CONDITION), CONDITION

    def sample_target(
        self, target: str, generator: torch.Generator | None = None,
    ) -> RolloutCondition:
        return RolloutCondition(
            label=torch.tensor([29]), spacing=SPACING, name=target,
        )

    def targets(self) -> tuple[str, ...]:
        return ("t1n", "t1c", "t2w", "t2f")


class FixedRealSampler:
    """real 采样替身：第 i 行恒填 i（可辨识条目——同源配对的行对应观测面）。"""

    def __init__(self, count: int, shape: tuple[int, ...]) -> None:
        self._count = count
        self._shape = shape
        self.calls: list[str] = []

    @property
    def size(self) -> int:
        return 64

    def sample(
        self, count: int, *, modality: str | None = None,
    ) -> torch.Tensor:
        assert count == self._count
        self.calls.append(modality or "")
        batch = torch.zeros(count, *self._shape)
        for index in range(count):
            batch[index] = float(index)
        return batch


class RecordingRealSampler:
    """real 采样替身：记录 (count, modality) 穿参并返回零批
    （装配原语把目标条件穿给 real 侧采样的观测缝）。"""

    def __init__(self, count: int, shape: tuple[int, ...]) -> None:
        self._count = count
        self._shape = shape
        self.calls: list[tuple[int, str]] = []

    @property
    def size(self) -> int:
        return 64

    def sample(
        self, count: int, *, modality: str | None = None,
    ) -> torch.Tensor:
        self.calls.append((count, modality or ""))
        return torch.zeros(count, *self._shape)


class WrittenPool:
    """不经 prepare 直写的最小 Real sample pool 工件（四模态各 4 条）。"""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.manifest_path = root / "real_pool.json"

    def write(self) -> Path:
        latent_dir = self._root / "real_pool_latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        entries: list[PoolEntry] = []
        for index in range(16):
            modality = ("t1n", "t1c", "t2w", "t2f")[index % 4]
            generator = torch.Generator().manual_seed(1 + index)
            noise = torch.randn(1, 64, 64, 32, generator=generator)
            pooled = torch.nn.functional.avg_pool3d(noise, 4)
            weights = torch.tensor([0.5, 0.75, 1.0, 1.25]).view(4, 1, 1, 1)
            latent_path = latent_dir / f"{index}.pt"
            torch.save(pooled * weights, latent_path)
            entries.append(PoolEntry(
                case_id=f"case-{index:03d}",
                modality=modality,
                latent=f"real_pool_latents/{index}.pt",
                spacing=(100.0, 100.0, 100.0),
            ))
        manifest = LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=SHAPE,
            split_seed=1,
            split_sizes={"train": 16, "val": 2, "test": 4},
            entries=entries,
        )
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )
        return self.manifest_path


class AssemblyScenario:
    """装配原语场景：小锚日程 + 零速度 ODE + 注入式替身。"""

    K = 4

    def __init__(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        fixture = Fixture()
        fixture.write_artifacts(tmp_path)  # config 依赖的工件面（不用其网络）
        self.pool_path = WrittenPool(tmp_path).write()

    @staticmethod
    def schedules(num_steps: int = NUM_STEPS) -> SingleConditionSchedules:
        return SingleConditionSchedules(
            num_inference_steps=num_steps, input_img_size_numel=2048,
        )

    def assembler(
        self,
        *,
        seed: int = 11,
        real_seed: int = 5,
        train_steps: tuple[int, ...] = TRAIN_STEPS,
        num_steps: int = NUM_STEPS,
        scale_factor: float = 1.0,
        real_sampler=None,
        conditions=None,
    ) -> ReconstructionAssembler:
        schedules = self.schedules(num_steps)
        sampler = RolloutSampler(
            CfgCombinedField(ZeroVelocityUnet()),
            SdeKernel(eta=0.7, s_max=0.999),
            schedules,
        )
        pool = LatentManifest.load(self.pool_path, kind="real_pool")
        return ReconstructionAssembler(
            real_sampler=(
                real_sampler if real_sampler is not None
                else RealPoolSampler(
                    pool, torch.Generator().manual_seed(real_seed),
                )
            ),
            sampler=sampler,
            schedules=schedules,
            conditions=conditions if conditions is not None else StubConditions(),
            train_step_indices=train_steps,
            batch_size_k=self.K,
            latent_scale_factor=scale_factor,
            generator=torch.Generator().manual_seed(seed),
            amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
        )


@pytest.fixture
def scenario(tmp_path: Path) -> AssemblyScenario:
    return AssemblyScenario(tmp_path)


class TestPairBatchProperties:
    """AC：装配原语 fixture 域端到端出配对批——fake ≠ real、同源配对、
    条件匹配、批量 = K。"""

    def test_fake_differs_from_real_and_shapes_match(
        self, scenario: AssemblyScenario,
    ) -> None:
        pair = scenario.assembler().assemble(CONDITION)
        assert isinstance(pair, PairBatch)
        assert pair.reals.shape == pair.fakes.shape == (
            scenario.K, *SHAPE,
        )
        assert not torch.equal(pair.reals, pair.fakes)  # 重构非恒等（s > 0）
        assert pair.modality == CONDITION

    def test_fake_is_paired_with_same_source_real(
        self, scenario: AssemblyScenario,
    ) -> None:
        """同源配对（零速度 ODE 恒等下可手工复算）：fake[i] = (1−s_i)·
        real[i] + s_i·ε_i——第 i 条 fake 由第 i 条 real 与第 i 条 ε 构造。"""
        fixed = FixedRealSampler(scenario.K, SHAPE)
        assembler = scenario.assembler(
            seed=11, real_sampler=fixed,
        )
        pair = assembler.assemble(CONDITION)
        # 同 seed 手工重放抽取序：先 randint 抽日程位（= s）、后 randn 抽 ε
        generator = torch.Generator().manual_seed(11)
        position = torch.randint(
            len(TRAIN_STEPS), (scenario.K,), generator=generator,
        )
        steps = [TRAIN_STEPS[i] for i in position.tolist()]
        noise = torch.randn(
            (scenario.K, *SHAPE), generator=generator,
        )
        cursor = scenario.schedules().cursor(CONDITION)
        sigmas = torch.tensor(
            [cursor.sigma_level(step) for step in steps],
        ).view(-1, 1, 1, 1, 1)
        expected = pair.reals * (1.0 - sigmas) + noise * sigmas
        assert torch.equal(pair.fakes, expected)

    def test_real_side_is_sampled_with_target_condition(
        self, scenario: AssemblyScenario,
    ) -> None:
        """real 侧按目标条件过滤采样（无放回、容量硬守卫照旧——装配
        原语把条件穿给 RealSampling）。"""
        recording = RecordingRealSampler(scenario.K, SHAPE)
        assembler = scenario.assembler(real_sampler=recording)
        pair = assembler.assemble("t2w")
        assert recording.calls == [(scenario.K, "t2w")]
        assert pair.modality == "t2w"

    def test_pool_sourced_reals_are_without_replacement(
        self, scenario: AssemblyScenario,
    ) -> None:
        """real 批来自真实样本库（无放回——条条不同、值来自 pool）。"""
        pair = scenario.assembler().assemble(CONDITION)
        fingerprints = {
            tuple(real.flatten().tolist()) for real in pair.reals
        }
        assert len(fingerprints) == scenario.K


class TestDeterministicReconstruction:
    """AC：s=0 重构恒等 + 同 seed 重放逐位一致。"""

    def test_zero_sigma_reconstruction_is_identity(
        self, scenario: AssemblyScenario,
    ) -> None:
        """s=0：加噪恒等 + σ=0 位于日程终点之后（零步续跑）——重构
        输出 = 输入（确定性 ODE 终点 = 起点）。"""
        assembler = scenario.assembler()
        reals = assembler._real_sampler.sample(scenario.K, modality=CONDITION)
        condition = StubConditions().sample_target(CONDITION)
        noise = torch.randn(scenario.K, *SHAPE)
        fakes = assembler.reconstruct(
            reals, condition, [0.0] * scenario.K, noise,
        )
        assert torch.equal(fakes, reals)

    def test_zero_sigma_identity_survives_domain_scale(
        self, scenario: AssemblyScenario,
    ) -> None:
        """s=0 恒等在非中性域换算系数下同样逐位成立（短路透传不走
        乘除往返——生产 scale_factor ≠ 1 时回归锚不降级为近似）。"""
        assembler = scenario.assembler(scale_factor=1.7)
        reals = assembler._real_sampler.sample(scenario.K, modality=CONDITION)
        condition = StubConditions().sample_target(CONDITION)
        fakes = assembler.reconstruct(
            reals, condition, [0.0] * scenario.K, torch.randn(scenario.K, *SHAPE),
        )
        assert torch.equal(fakes, reals)

    def test_same_seed_replay_is_bitwise_identical(
        self, scenario: AssemblyScenario,
    ) -> None:
        """同 seed 双装配逐位一致：real 采样流与重构流（先 s 后 ε）的
        确定性重放 + ODE 确定性（ADR-0011）。"""
        first = scenario.assembler().assemble(CONDITION)
        second = scenario.assembler().assemble(CONDITION)
        assert torch.equal(first.reals, second.reals)
        assert torch.equal(first.fakes, second.fakes)

    def test_reconstruction_of_fixed_sigmas_is_deterministic(
        self, scenario: AssemblyScenario,
    ) -> None:
        """重构核对固定 (real, σ, ε) 输入无随机性：两次调用逐位一致。"""
        assembler = scenario.assembler()
        reals = assembler._real_sampler.sample(scenario.K, modality=CONDITION)
        condition = StubConditions().sample_target(CONDITION)
        sigma = scenario.schedules().cursor(CONDITION).sigma_level(1)
        noise = torch.randn(scenario.K, *SHAPE)
        first = assembler.reconstruct(
            reals, condition, [sigma] * scenario.K, noise,
        )
        second = assembler.reconstruct(
            reals, condition, [sigma] * scenario.K, noise,
        )
        assert torch.equal(first, second)


class TestSigmaScheduleMirroring:
    """AC：s 抽样镜像日程——候选步点与 ConditionSchedules 按被优化步
    导出一致，s=1 不在候选集。"""

    def test_candidate_sigmas_mirror_condition_schedules(
        self, scenario: AssemblyScenario,
    ) -> None:
        """候选 σ = 该条件 sigma 日程在 M 上的导出（同锚同源）。"""
        expected = tuple(
            scenario.schedules().cursor(CONDITION).sigma_level(step)
            for step in TRAIN_STEPS
        )
        assert scenario.assembler().candidate_sigmas(CONDITION) == expected

    def test_singular_sigma_one_is_not_a_candidate(
        self, scenario: AssemblyScenario,
    ) -> None:
        """s=1（纯噪声端）不在候选集：M 排除日程下标 0，日程导出的
        候选 σ 全部严格小于 1（也严格大于 0——σ=0 端无去噪步）。"""
        candidates = scenario.assembler().candidate_sigmas(CONDITION)
        assert all(0.0 < sigma < 1.0 for sigma in candidates)

    def test_unknown_condition_is_rejected(self, scenario) -> None:
        """单域日程对未知名容忍（BraTS 特例）——逐条件候选导出经
        ``ConditionSchedules.cursor`` 的语义分派；本场景单域替身下
        任意名恒等。逐条件形态（PerConditionSchedules）的缺名/未知名
        拒绝由其自身测试面把守，此处不重复。"""
        assembler = scenario.assembler()
        assert assembler.candidate_sigmas("t2f") == (
            assembler.candidate_sigmas(CONDITION)
        )

    def test_step_indices_beyond_schedule_rejected(
        self, scenario: AssemblyScenario,
    ) -> None:
        """M 含超出日程的步位：候选导出显式拒绝（可读报错），不静默
        截尾。"""
        assembler = scenario.assembler(train_steps=(1, 2, NUM_STEPS + 3))
        with pytest.raises(ValueError, match="越界"):
            assembler.candidate_sigmas(CONDITION)


class TestReconStreamIsolation:
    """AC：重构构造的专属随机流与训练/评测/AUC 流不交叉（先 s 后 ε
    只消耗 recon 流；real 侧照旧消费 real_pool 流）。"""

    def test_assembly_consumes_only_recon_and_real_pool_streams(
        self, scenario: AssemblyScenario,
    ) -> None:
        streams = TrainingRngStreams(seed=0)
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        assembler = ReconstructionAssembler(
            real_sampler=RealPoolSampler(
                pool, streams.real_pool,
            ),
            sampler=RolloutSampler(
                CfgCombinedField(ZeroVelocityUnet()),
                SdeKernel(eta=0.7, s_max=0.999),
                scenario.schedules(),
            ),
            schedules=scenario.schedules(),
            conditions=StubConditions(),
            train_step_indices=TRAIN_STEPS,
            batch_size_k=scenario.K,
            latent_scale_factor=1.0,
            generator=streams.recon,
            amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
        )
        before = {
            name: generator.get_state().clone()
            for name, generator in streams.named().items()
        }
        assembler.assemble(CONDITION)
        assert not torch.equal(
            streams.recon.get_state(), before["recon"],
        )  # 重构加噪（s 与 ε）走 recon 流
        assert not torch.equal(
            streams.real_pool.get_state(), before["real_pool"],
        )  # real 侧采样照旧走 real_pool 流
        for name in ("rollout", "heldout_auc"):
            assert torch.equal(
                streams.named()[name].get_state(), before[name],
            ), name


class TestAssemblerContract:
    """构造期契约：M 非空、不含最噪端下标 0、K 为正。"""

    @pytest.mark.gpu  # 加速器口径（集群全量）：CPU 常量的 device mismatch 只在非 CPU 输入上暴露
    def test_reconstruction_aligns_sigma_tensor_with_input_device(
        self, scenario: AssemblyScenario,
    ) -> None:
        """σ 水平张量随输入设备落位（生产路径 real 批在加速器上）：
        重构核在加速器输入上不抛 device mismatch、输出与输入同设备
        （fixture 全 CPU 口径覆盖不到的路径——生产可用性的回归锚）。"""
        device = torch.device("cuda")
        assembler = scenario.assembler()
        reals = assembler._real_sampler.sample(
            scenario.K, modality=CONDITION,
        ).to(device)
        condition = RolloutCondition(
            label=torch.tensor([29], device=device),
            spacing=SPACING.to(device),
            name=CONDITION,
        )
        sigma = scenario.schedules().cursor(CONDITION).sigma_level(1)
        fakes = assembler.reconstruct(
            reals, condition, [sigma] * scenario.K,
            torch.randn(scenario.K, *SHAPE).to(device),
        )
        # 与输入同设备（测点本意）：``torch.device("cuda")`` 是无索引
        # 字面量、张量落位为 ``cuda:0``，两者不相等——断言对齐到
        # 「输出跟随输入设备」本身，不绑设备索引写法
        assert fakes.device == reals.device
        assert fakes.device.type == "cuda"  # 确在加速器上（非静默回落 CPU）

    def test_empty_step_indices_rejected(self, scenario) -> None:
        with pytest.raises(ValueError, match="不得为空"):
            scenario.assembler(train_steps=())

    def test_singular_step_zero_rejected(self, scenario) -> None:
        """M 含日程下标 0（s≈1 奇异端）：构造期拒绝（ADR-0012 决策 2
        的 s=1 天然排除在装配面的落点）。"""
        with pytest.raises(ValueError, match="奇异点"):
            scenario.assembler(train_steps=(0, 1, 2))

    def test_non_positive_batch_size_rejected(self, scenario) -> None:
        schedules = scenario.schedules()
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        with pytest.raises(ValueError, match="batch_size_k"):
            ReconstructionAssembler(
                real_sampler=RealPoolSampler(
                    pool, torch.Generator().manual_seed(5),
                ),
                sampler=RolloutSampler(
                    CfgCombinedField(ZeroVelocityUnet()),
                    SdeKernel(eta=0.7, s_max=0.999), schedules,
                ),
                schedules=schedules,
                conditions=StubConditions(),
                train_step_indices=TRAIN_STEPS,
                batch_size_k=0,
                latent_scale_factor=1.0,
                generator=torch.Generator().manual_seed(11),
                amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
            )

    def test_sigma_outside_schedule_rejected(
        self, scenario: AssemblyScenario,
    ) -> None:
        """重构核收到日程点外的 σ：显式拒绝（起点无从定位）。"""
        assembler = scenario.assembler()
        reals = assembler._real_sampler.sample(scenario.K, modality=CONDITION)
        condition = StubConditions().sample_target(CONDITION)
        with pytest.raises(ValueError, match="日程点"):
            assembler.reconstruct(
                reals, condition, [0.42] * scenario.K,
                torch.randn(scenario.K, *SHAPE),
            )

    def test_singular_sigma_one_rejected_in_kernel(
        self, scenario: AssemblyScenario,
    ) -> None:
        """σ=s_0（最噪端日程点）：重构核拒绝（无「下标 −1」的起点——
        s≈1 只可由 M 排除的日程下标 0 产生）。"""
        assembler = scenario.assembler()
        reals = assembler._real_sampler.sample(scenario.K, modality=CONDITION)
        condition = StubConditions().sample_target(CONDITION)
        sigma_zero = scenario.schedules().cursor(CONDITION).sigma_level(0)
        with pytest.raises(ValueError, match="奇异点"):
            assembler.reconstruct(
                reals, condition, [sigma_zero] * scenario.K,
                torch.randn(scenario.K, *SHAPE),
            )


class TestRankStructuralAlignment:
    """分布式集合序列对齐（#165 review P1 的装配原语延伸）：
    ``continue_to_terminal`` 携带 ``chunk_sync`` 集合通信且前向过
    FSDP——其**调用次数与逐次批量**是跨 rank 集合序列的一部分。s 抽样
    若逐 rank 相异，分组结构（每 σ 的组存在性与组大小）相异，首个判别
    器更新步即集合序列错位死锁。生产 seeding 规则 = 流注册表按 rank
    派生（``DistributedContext.derive_seed``，runtime 装配位的同一规
    则）——本测试按同一规则模拟两 rank，锁定重构续跑的调用结构逐位
    一致。"""

    @staticmethod
    def _rank_assembler(
        scenario: "AssemblyScenario", rank: int,
    ) -> tuple[ReconstructionAssembler, list[tuple[int, int]]]:
        # 生产 seeding 规则（runtime/driver 装配位同款）：数据侧流按
        # rank 派生主 seed、recon 流按 rank 无关 shared seed 派生
        streams = TrainingRngStreams(
            DistributedContext(rank, 2, True).derive_seed(11),
            shared_seed=11,
        )
        schedules = scenario.schedules()
        sampler = RolloutSampler(
            CfgCombinedField(ZeroVelocityUnet()),
            SdeKernel(eta=0.7, s_max=0.999),
            schedules,
        )
        calls: list[tuple[int, int]] = []
        original = sampler.continue_to_terminal

        def recording(noised, start, condition, _original=original):
            calls.append((start, noised.shape[0]))
            return _original(noised, start, condition)

        sampler.continue_to_terminal = recording
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        assembler = ReconstructionAssembler(
            real_sampler=RealPoolSampler(pool, streams.real_pool),
            sampler=sampler,
            schedules=schedules,
            conditions=StubConditions(),
            train_step_indices=TRAIN_STEPS,
            batch_size_k=scenario.K,
            latent_scale_factor=1.0,
            generator=streams.recon,
            amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
        )
        return assembler, calls

    def test_continue_call_structure_is_rank_invariant(
        self, scenario: AssemblyScenario,
    ) -> None:
        structures = []
        for rank in (0, 1):
            assembler, calls = self._rank_assembler(scenario, rank)
            assembler.assemble(CONDITION)
            structures.append(calls)
        assert structures[0] == structures[1]

    def test_recon_stream_shared_and_data_streams_rank_derived(self) -> None:
        """注册表级不变量：recon 流跨 rank 同 seed（集合序列一致性的
        源头），数据侧流跨 rank 异 seed（多样性语义不被误伤）。"""
        base = 11
        registries = [
            TrainingRngStreams(
                DistributedContext(rank, 2, True).derive_seed(base),
                shared_seed=base,
            )
            for rank in (0, 1)
        ]
        assert (
            registries[0].recon.initial_seed()
            == registries[1].recon.initial_seed()
            == base + 9
        )
        assert (
            registries[0].rollout.initial_seed()
            != registries[1].rollout.initial_seed()
        )
        assert (
            registries[0].real_pool.initial_seed()
            != registries[1].real_pool.initial_seed()
        )


class TestCrossModalConditionStream:
    """组2 条件抽取的随机流归位（流隔离契约的装配原语侧）：生产接线把
    ``CrossModalConditionSampler`` 建在 policy 主流上（train/policy.py
    的构造位）——``sample_target`` 的缺省语义是「缺省用主流」，装配原
    语若不显式穿流，每个判别器更新步的源对与源条目抽取都会消耗
    **rollout** 流，rollout 样本序列从此依赖判别器更新节奏。"""

    @staticmethod
    def _cross_modal_conditions(
        scenario: "AssemblyScenario", rollout: torch.Generator,
    ) -> CrossModalConditionSampler:
        return CrossModalConditionSampler(
            ModalityMapping({"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}),
            [("t1n", "t1c")],
            SourceLatentPool(
                LatentManifest.load(scenario.pool_path, kind="real_pool"),
                torch.device("cpu"),
            ),
            rollout,
            torch.device("cpu"),
        )

    def test_condition_draw_stays_off_rollout_stream(
        self, scenario: AssemblyScenario,
    ) -> None:
        streams = TrainingRngStreams(seed=0)
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        assembler = ReconstructionAssembler(
            real_sampler=RealPoolSampler(pool, streams.real_pool),
            sampler=RolloutSampler(
                CfgCombinedField(ZeroVelocityUnet()),
                SdeKernel(eta=0.7, s_max=0.999),
                scenario.schedules(),
            ),
            schedules=scenario.schedules(),
            conditions=self._cross_modal_conditions(
                scenario, streams.rollout,
            ),
            train_step_indices=TRAIN_STEPS,
            batch_size_k=scenario.K,
            latent_scale_factor=1.0,
            generator=streams.recon,
            amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
        )
        rollout_before = streams.rollout.get_state().clone()
        recon_before = streams.recon.get_state().clone()
        pair = assembler.assemble("t1c")
        assert pair.modality == "t1c"
        assert torch.equal(streams.rollout.get_state(), rollout_before)
        assert not torch.equal(streams.recon.get_state(), recon_before)


class TestMeasurementConditionReconstruction:
    """gate 测量批（#171）：ADR-0012 决策 5 的 recon-AUC 构造面——测量批
    的 fake 由**调用方给出的 real** 同源重构而来（非量产 rollout）、σ 按
    候选步点定序轮转、ε 走批次起手复位的测量流。

    测量是上岗判据的原料（报告值与白名单都由它出），所以本类的断言面
    是「可复算」而非「可重放」：同输入 → 逐位同输出，与调用序无关。
    """

    def _assembler(self, scenario: AssemblyScenario, **kwargs):
        return scenario.assembler(**kwargs)

    def test_measurement_reconstruction_is_paired_with_given_reals(
        self, scenario: AssemblyScenario,
    ) -> None:
        """配对语义：返回批的 real 与入参**同一对象**、fake 逐样本由它
        重构——零速度 ODE 下 fake = x_s = (1−s)·x + s·ε 可手工复算。"""
        assembler = self._assembler(scenario)
        reals = torch.randn(6, *SHAPE)
        pair = assembler.measure_condition(reals, CONDITION)
        assert pair.reals is reals  # 同源配对的结构保证
        assert pair.modality == CONDITION
        assert pair.fakes.shape == reals.shape
        assert not torch.equal(pair.fakes, reals)  # s > 0 → 加噪本体在场
        # 手工复算（同 seed 同 σ 序），共享 helper 见模块头
        assert torch.equal(
            pair.fakes, _expected_measurement_fakes(assembler, reals),
        )

    def test_sigma_assignment_rotates_over_candidate_steps(
        self, scenario: AssemblyScenario,
    ) -> None:
        """σ 定序轮转：第 i 枚卷取候选步点的第 i % |M| 位——与调度无关的
        确定性排布（``assemble`` 的逐样本抽签在此被替换为固定序）。"""
        assembler = self._assembler(scenario, train_steps=(1, 2, 3), num_steps=5)
        candidates = assembler.candidate_sigmas(CONDITION)
        assert len(candidates) == 3
        reals = torch.randn(7, *SHAPE)
        pair = assembler.measure_condition(reals, CONDITION)
        # 逐卷手工复算：第 i 枚卷的 σ 必须是候选轮转序的第 i % |M| 位
        # ——σ 排布错位（如全部取首位、或按 |M| 之外的步长轮转）会让
        # 对应行的逐位等式破掉
        expected = _expected_measurement_fakes(assembler, reals)
        for index in range(reals.shape[0]):
            assert torch.equal(expected[index], pair.fakes[index]), index

    def test_repeated_measurement_is_bitwise_identical(
        self, scenario: AssemblyScenario,
    ) -> None:
        """测量的可复算性：同输入逐次测量逐位同输出——不受「本 run 此前
        测量过几次」影响（测量流批次起手复位）。这是报告值可复算与
        gate 判定可比的结构前提。"""
        assembler = self._assembler(scenario)
        reals = torch.randn(5, *SHAPE)
        first = assembler.measure_condition(reals, CONDITION)
        for _ in range(3):
            again = assembler.measure_condition(reals, CONDITION)
            assert torch.equal(again.fakes, first.fakes)

    def test_measurement_consumes_no_named_stream(
        self, scenario: AssemblyScenario,
    ) -> None:
        """测量不消耗 recon 流（续训分片的流位置不被测量次数搅动）、也不
        漂移 real_pool 流之外的数据侧流；随后的更新批与「一次测量都没
        发生」时逐位一致（流隔离契约在两条入口并存下仍成立）。"""
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")

        def build(streams: TrainingRngStreams) -> ReconstructionAssembler:
            return ReconstructionAssembler(
                real_sampler=RealPoolSampler(pool, streams.real_pool),
                sampler=RolloutSampler(
                    CfgCombinedField(ZeroVelocityUnet()),
                    SdeKernel(eta=0.7, s_max=0.999),
                    scenario.schedules(),
                ),
                schedules=scenario.schedules(),
                conditions=StubConditions(),
                train_step_indices=TRAIN_STEPS,
                batch_size_k=scenario.K,
                latent_scale_factor=1.0,
                generator=streams.recon,
                amp=AmpContext(
                    device=torch.device("cpu"), dtype=torch.bfloat16,
                ),
            )

        streams = TrainingRngStreams(seed=0)
        assembler = build(streams)
        reals = torch.randn(4, *SHAPE)
        measured = assembler.measure_condition(reals, CONDITION)
        # 测量不动 recon 流的位置 → 对照装配（连测三次后的同一流位置）
        # 产出的测量批逐位一致（测量次数不进 recon 序列）
        twin = build(TrainingRngStreams(seed=0))
        for _ in range(3):
            twin.measure_condition(reals, CONDITION)
        assert torch.equal(
            measured.fakes, twin.measure_condition(reals, CONDITION).fakes,
        )
        # 命名流全员不被测量消耗（数据侧六条与 recon 都不动）
        streams2 = TrainingRngStreams(seed=0)
        probe = build(streams2)
        before = {
            name: generator.get_state().clone()
            for name, generator in streams2.named().items()
        }
        probe.measure_condition(reals, CONDITION)
        for name, state in before.items():
            assert torch.equal(streams2.named()[name].get_state(), state), name
        # 更新批照旧只消耗 recon / real_pool（两条入口并存不互相搅动）
        probe.assemble(CONDITION)
        assert not torch.equal(streams2.recon.get_state(), before["recon"])
        assert not torch.equal(streams2.real_pool.get_state(), before["real_pool"])

    def test_empty_real_batch_rejected(self, scenario: AssemblyScenario) -> None:
        """空 real 卷显式拒绝（测量批的源不得为空——AUC 配对统计需要
        非空两侧）。"""
        assembler = self._assembler(scenario)
        with pytest.raises(ValueError, match="非空"):
            assembler.measure_condition(torch.zeros(0, *SHAPE), CONDITION)

    def test_forward_count_mirrors_round_robin_schedule(
        self, scenario: AssemblyScenario,
    ) -> None:
        """成本读数 = 轮转排布下逐卷续跑步数之和（#171 AC5：无 30 步全
        ODE 量产的可核对面）；σ=0 档位零步短路。"""
        assembler = self._assembler(scenario, train_steps=(1, 3), num_steps=5)
        cursor = scenario.schedules(5).cursor(CONDITION)
        # 测量批的规模 = 入参卷数（装配原语不假定调用方的 real 来源：
        # 判别器 real 池与 held-out 池是两个采样器）
        volumes = 4
        reals = torch.randn(volumes, *SHAPE)
        # 轮转序 (1, 3, 1, 3, …)：逐卷余量 = num_steps − 1 − 起点下标
        # （步数由 MONAI 实际日程定，不按 num_inference_steps 手算）
        steps = assembler._step_indices
        assert steps == (1, 3)
        expected = sum(
            cursor.num_steps - 1 - (steps[index % len(steps)] - 1)
            for index in range(volumes)
        )
        forwards = assembler.measurement_forward_count(reals, CONDITION)
        assert forwards == expected
        # 每卷步数严格小于「量产全 ODE」（num_steps 步）——量产退役的
        # 数值口径（重构从日程中段起步）
        assert forwards < volumes * cursor.num_steps

    def test_schedule_terminal_step_is_rejected_as_candidate(
        self, scenario: AssemblyScenario,
    ) -> None:
        """日程末位不是合法候选（σ 恒 > 0 的中段日程点——末位的零步档位
        在装配期即拒，成本读数因此只处理正步数）。"""
        # (1, 3)：3 = num_steps − 2，合法中段上界
        assembler = self._assembler(scenario, train_steps=(1, 3), num_steps=5)
        cursor = scenario.schedules(5).cursor(CONDITION)
        assert assembler.candidate_sigmas(CONDITION)[-1] > 0.0
        assert assembler.candidate_sigmas(CONDITION)[-1] == cursor.sigma_level(3)
        # 末位（4 = num_steps − 1）与越界位都被拒：末位之后无续跑空间，
        # 重构退化为透传（fake ≡ real）污染测量。守卫在候选导出面触发
        # （构造期不查日程——装配与判定同一缝，见 _assert_indices_…）
        for illegal in ((1, 4), (1, 5)):
            with pytest.raises(ValueError, match="越界"):
                self._assembler(
                    scenario, train_steps=illegal, num_steps=5,
                ).candidate_sigmas(CONDITION)

    def test_cross_modal_measurement_is_deterministic(
        self, scenario: AssemblyScenario,
    ) -> None:
        """组2（跨模态）测量同款可复算：条件构造（源对 + 源条目抽取）走
        复位测量流，且不漂移 rollout 流——机制缝对组2 开放、语义留待
        专项票（ADR-0012 非目标）。"""
        streams = TrainingRngStreams(seed=0)
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        assembler = ReconstructionAssembler(
            real_sampler=RealPoolSampler(pool, streams.real_pool),
            sampler=RolloutSampler(
                CfgCombinedField(ZeroVelocityUnet()),
                SdeKernel(eta=0.7, s_max=0.999),
                scenario.schedules(),
            ),
            schedules=scenario.schedules(),
            conditions=TestCrossModalConditionStream._cross_modal_conditions(
                scenario, streams.rollout,
            ),
            train_step_indices=TRAIN_STEPS,
            batch_size_k=scenario.K,
            latent_scale_factor=1.0,
            generator=streams.recon,
            amp=AmpContext(device=torch.device("cpu"), dtype=torch.bfloat16),
        )
        reals = torch.randn(3, *SHAPE)
        rollout_before = streams.rollout.get_state().clone()
        recon_before = streams.recon.get_state().clone()
        first = assembler.measure_condition(reals, "t1c")
        second = assembler.measure_condition(reals, "t1c")
        assert torch.equal(first.fakes, second.fakes)
        assert torch.equal(streams.rollout.get_state(), rollout_before)
        assert torch.equal(streams.recon.get_state(), recon_before)

    def test_epsilon_slice_generation_matches_full_batch_bitwise(
        self, scenario: AssemblyScenario,
    ) -> None:
        """ε 切片生成与整批生成同位逐位一致（issue #198；分布式票
        ADR-0016 决策 4 的数值前提）：测量流批次起手复位的**同一状态**
        下，「先消耗前 s 行 ε、再生成 (stop−s) 行」与整批 ε 的 [s:stop)
        行逐位一致——torch CPU generator 的顺序流性质。测量批 ε 的生成
        点在 measure_condition 的复位测量流内，多卡下每 rank 只生成本地
        切片的 ε 依赖本性质成立；生成点若迁移设备（非 CPU 流）或改换
        非顺序流语义，此处即红。"""
        assembler = self._assembler(scenario)
        reals = torch.randn(8, *SHAPE)
        # 整批 ε：测量流复位模板直接注入（measure_condition 每批起手的
        # 同一状态——测试不经私有方法、以模板状态为复位语义的观测面）
        full = torch.Generator()
        full.set_state(assembler._measurement_template)
        batch_noise = torch.randn(reals.shape, generator=full)
        for start, stop in ((0, 3), (2, 5), (5, 8), (0, 8)):
            sliced = torch.Generator()
            sliced.set_state(assembler._measurement_template)
            if start:
                # 前缀消耗：跳过排在前面 rank 的 ε 行（复位流的顺序位）
                torch.randn((start, *SHAPE), generator=sliced)
            assert torch.equal(
                torch.randn((stop - start, *SHAPE), generator=sliced),
                batch_noise[start:stop],
            ), (start, stop)

    def test_prefix_slice_measurement_matches_full_batch(
        self, scenario: AssemblyScenario,
    ) -> None:
        """前缀切片测量批 = 整批测量批的前缀行（逐位，issue #198）：
        ε 侧走复位流前缀（同位逐位一致）+ 单候选步下 σ 轮转退化为恒位
        （无切片错位面）+ 重构逐样本独立——分布式 rank0（首切片）的
        端到端数值前提。任意偏移切片的 σ 轮转偏移（第 i 枚卷取第
        i % |M| 位须按全量位次计）是分布式票的轮转偏移缝，本锚不越权。"""
        assembler = self._assembler(scenario, train_steps=(1,))
        assert len(assembler.candidate_sigmas(CONDITION)) == 1
        reals = torch.randn(6, *SHAPE)
        full = assembler.measure_condition(reals, CONDITION)
        for cut in (1, 3, 6):
            sliced = assembler.measure_condition(reals[:cut], CONDITION)
            assert torch.equal(sliced.fakes, full.fakes[:cut]), cut
            assert torch.equal(sliced.reals, full.reals[:cut]), cut

    def test_offset_slice_measurement_matches_full_batch_rows(
        self, scenario: AssemblyScenario,
    ) -> None:
        """任意偏移切片测量批 = 整批对应行（逐位，ADR-0016 决策 4 的
        分布式数值前提）：σ 轮转按全量位次偏移（本地第 j 卷取候选第
        (offset+j) % |M| 位）+ 复位测量流**前缀消耗** offset 行 ε——
        「同排列、切加载」下 gather 拼回的全量测量批与单卡 rank0 全量
        测量逐位一致（多卡 gate 报告值对单卡可复算）。与 #198 前缀锚
        的分工：彼锚「前缀切片的 σ 退化为恒位」（单候选步），本锚锁
        **任意偏移 + 多候选步** 的轮转偏移与前缀消耗组合。"""
        assembler = self._assembler(scenario, train_steps=(1, 2, 3), num_steps=5)
        reals = torch.randn(7, *SHAPE)
        full = assembler.measure_condition(reals, CONDITION)
        for start, stop in ((0, 3), (2, 5), (5, 7), (3, 4)):
            sliced = assembler.measure_condition(
                reals[start:stop], CONDITION, volume_offset=start,
            )
            assert torch.equal(sliced.fakes, full.fakes[start:stop]), (
                start, stop
            )
            assert torch.equal(sliced.reals, full.reals[start:stop])

    def test_measurement_forward_count_tracks_volume_offset(
        self, scenario: AssemblyScenario,
    ) -> None:
        """成本读数同源 σ 排布：偏移切片的逐卷步数和 = 整批对应段的
        步数和（事件流 ``reconstruction_forwards`` 全局口径 = 各 rank
        本地读数 gather 求和的前提——求和合法当且仅当分段推算与整批
        推算在对应段上同值）。"""
        assembler = self._assembler(scenario, train_steps=(1, 2, 3), num_steps=5)
        reals = torch.randn(7, *SHAPE)
        bounds = ((0, 3), (3, 5), (5, 7))
        parts = [
            assembler.measurement_forward_count(
                reals[start:stop], CONDITION, volume_offset=start,
            )
            for start, stop in bounds
        ]
        assert sum(parts) == assembler.measurement_forward_count(
            reals, CONDITION,
        )
        for (start, stop), part in zip(bounds, parts):
            assert part == (
                assembler.measurement_forward_count(reals[:stop], CONDITION)
                - assembler.measurement_forward_count(
                    reals[:start], CONDITION,
                )
            )

    def test_negative_volume_offset_rejected(
        self, scenario: AssemblyScenario,
    ) -> None:
        """切片偏移是「本地批首卷在全量批中的位次」——负位次无语义，
        显式拒绝而非静默取模（负偏移会把轮转与 ε 消耗导向错误槽位）。"""
        assembler = self._assembler(scenario)
        reals = torch.randn(2, *SHAPE)
        with pytest.raises(ValueError, match="volume_offset"):
            assembler.measure_condition(reals, CONDITION, volume_offset=-1)
        with pytest.raises(ValueError, match="volume_offset"):
            assembler.measurement_forward_count(
                reals, CONDITION, volume_offset=-1,
            )
