"""Online update 一步测试（ticket #20 AC + ADR-0012 配对批接口）：消费
配对批（real + 同源重构 fake，装配原语供批）→ LSGAN 损失随更新下降、
AdamW 5e-5、干净域前向与 train 侧干净域复算。

ADR-0012：real 侧不再内部自抽（装配原语负责条件匹配采样与容量硬守卫）、
fake 侧不再回放混采（替换而非并存，判别器链路的 buffer 消费退役）；
判别器输入恒干净域（带噪训练入口随注入退役——打分、训练、AUC、监控
共享同一入口）；损失、优化器、梯度流零改动。real 池逐模态容量 ≥ K 的
装配期守卫语义由 LatentManifest/RankSlicedPool 测试段把守（本文件尾部）。
"""

from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES
from cynosure.distributed import DistributedContext, RankSlicedPool
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats, LatentManifest, PoolEntry
from cynosure.reward.assembly import PairBatch
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.scorer import LatentScorer, RewardScorer
from cynosure.reward.update import OnlineUpdate

SHAPE = (4, 16, 16, 8)


class WrittenPool:
    """不经 prepare 直写的最小 Real sample pool 工件（held-out 测试等
    消费 manifest 工件的场景共用；四模态各 4 条）。"""

    def __init__(self, root: Path, num_cases: int = 16, seed: int = 1) -> None:
        self._root = root
        self._num_cases = num_cases
        self._seed = seed
        self.manifest_path = root / "real_pool.json"

    def write(self) -> Path:
        latent_dir = self._root / "real_pool_latents"
        latent_dir.mkdir(parents=True, exist_ok=True)
        entries: list[PoolEntry] = []
        for index in range(self._num_cases):
            case_id = f"case-{index:03d}"
            modality = ("t1n", "t1c", "t2w", "t2f")[index % 4]
            generator = torch.Generator().manual_seed(self._seed + index)
            noise = torch.randn(1, 64, 64, 32, generator=generator)
            pooled = torch.nn.functional.avg_pool3d(noise, 4)
            weights = torch.tensor([0.5, 0.75, 1.0, 1.25]).view(4, 1, 1, 1)
            latent_path = latent_dir / f"{index}.pt"
            torch.save(pooled * weights, latent_path)
            entries.append(PoolEntry(
                case_id=case_id,
                modality=modality,
                latent=f"real_pool_latents/{index}.pt",
                spacing=(100.0, 100.0, 100.0),
            ))
        manifest = LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=SHAPE,
            split_seed=self._seed,
            split_sizes={"train": self._num_cases, "val": 2, "test": 4},
            entries=entries,
        )
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )
        return self.manifest_path


class RecordingCleanScorer:
    """测试仪器：委托真 scorer 并记录 patch_logits 的调用序列（入口、
    调用时 grad 开关与输入批——干净域语义的观测缝）与 training_patch_logits
    的调用（ADR-0012 后必须为零——带噪入口退役）。"""

    def __init__(self, inner: LatentScorer) -> None:
        self._inner = inner
        self.clean_entries: list[tuple[tuple[int, ...], bool]] = []
        """patch_logits 的 (输入形状, 调用时 grad 是否开启) 序列。"""
        self.clean_batches: list[torch.Tensor] = []
        """patch_logits 的输入批序列（干净域复算的重放对账原料）。"""
        self.noisy_calls: int = 0

    @property
    def discriminator(self):
        return self._inner.discriminator

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        self.clean_entries.append(
            (tuple(latents.shape), torch.is_grad_enabled()),
        )
        self.clean_batches.append(latents)
        return self._inner.patch_logits(latents)

    def reward(self, latents: torch.Tensor) -> torch.Tensor:
        return self._inner.reward(latents)

    def discriminator_terms(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor,
    ):
        return self._inner.discriminator_terms(logits_real, logits_fake)

    def training_patch_logits(
        self, latents: torch.Tensor, generator: torch.Generator,
    ) -> torch.Tensor:
        self.noisy_calls += 1
        return self._inner.training_patch_logits(latents, generator)


class UpdateScenario:
    """OnlineUpdate 单元场景：fixture 判别器 + 配对批注入（附带 pool
    工件与随机批工厂——held-out AUC 测试等共享场景消费）。"""

    def __init__(self, tmp_path: Path) -> None:
        torch.manual_seed(0)  # fixture 网络工件确定性（随机初始化随 seed 定死）
        fixture = Fixture()
        fixture.write_artifacts(tmp_path)
        self.config = fixture.config(tmp_path)
        self.pool_path = WrittenPool(tmp_path).write()

    def stats(self) -> ChannelStats:
        return ChannelStats(
            mean=[0.0, 0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0, 1.0],
            num_latents=1,
            latent_shape=SHAPE,
            source_manifest="real_pool.json",
        )

    def scorer(self) -> RewardScorer:
        config = self.config
        return RewardScorer(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.discriminator_config_json,
                ),
                checkpoint=config.artifacts.discriminator_ckpt,
            ),
            config.reward,
            self.stats(),
        )

    def generator(self, seed: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(seed)

    def fakes(self, count: int, seed: int = 3) -> torch.Tensor:
        """随机 latent 批工厂（fake 侧输入的替身形态；held-out 测试
        等共享场景的消费面）。"""
        return torch.randn(count, *SHAPE, generator=self.generator(seed))

    def pair(
        self, count: int = 4, seed: int = 3, modality: str = "t2w",
    ) -> PairBatch:
        """配对批替身（装配原语产出的形态等价物）：real/fake 同形同量，
        内容独立（fixture 口径不关心同源性——那是装配原语的测试面）。"""
        reals = torch.randn(
            count, *SHAPE, generator=self.generator(seed),
        )
        fakes = torch.randn(
            count, *SHAPE, generator=self.generator(seed + 1),
        )
        return PairBatch(reals=reals, fakes=fakes, modality=modality)

    def update(
        self, scorer: LatentScorer | None = None,
    ) -> OnlineUpdate:
        return OnlineUpdate(
            scorer=scorer if scorer is not None else self.scorer(),
            config=self.config.reward,
        )


@pytest.fixture
def scenario(tmp_path: Path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class TestUpdateStep:
    def test_report_loss_decomposes_into_real_and_fake_terms(
        self, scenario: UpdateScenario,
    ) -> None:
        """loss = mean((D(real)−1)²) + mean(D(fake)²)：报告两项分解一致
        （损失语义零改动——配对批只是输入来源换缝）。"""
        update = scenario.update()
        report = update.step(scenario.pair())
        assert report.loss_discriminator == pytest.approx(
            report.loss_real_term + report.loss_fake_term, abs=1e-6,
        )

    def test_report_carries_batch_size_and_condition(
        self, scenario: UpdateScenario,
    ) -> None:
        """报告带配对批批量（real 与 fake 同量——同源配对的结构性质）
        与条件标记（update 归因轴）。"""
        update = scenario.update()
        report = update.step(scenario.pair(count=4, modality="t1c"))
        assert report.batch_size == 4
        assert report.modality == "t1c"

    def test_optimizer_is_adamw_with_configured_lr(
        self, scenario: UpdateScenario,
    ) -> None:
        """AdamW、lr = disc_lr（默认 5e-5）。"""
        update = scenario.update()
        assert type(update.optimizer).__name__ == "AdamW"
        assert update.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

    def test_optimizer_weight_decay_is_explicit(
        self, scenario: UpdateScenario,
    ) -> None:
        """卫生项（ADR-0007）：判别器 AdamW 的 weight_decay 显式取 config
        值（与 policy 侧同值口径 1e-4），不再是隐式 PyTorch 默认 0.01。"""
        update = scenario.update()
        decay = update.optimizer.param_groups[0]["weight_decay"]
        assert decay == pytest.approx(1e-4)
        assert decay == pytest.approx(scenario.config.policy.policy_weight_decay)

    def test_step_changes_discriminator_weights(
        self, scenario: UpdateScenario,
    ) -> None:
        """一步更新后判别器权重发生变化（online update 生效——backward
        与梯度流零改动，配对批真实进入损失）。"""
        update = scenario.update()
        weight = next(
            p for p in update.scorer.discriminator.parameters() if p.ndim == 5
        )
        before = weight.detach().clone()
        update.step(scenario.pair())
        assert not torch.equal(before, weight.detach())

    def test_identical_generator_sequences_reproduce_losses(
        self, scenario: UpdateScenario,
    ) -> None:
        """固定 seed 配对批重放：同场景 → 损失轨迹逐位一致。"""
        first = scenario.update()
        second = scenario.update()
        losses_first = [
            first.step(scenario.pair()).loss_discriminator for _ in range(3)
        ]
        losses_second = [
            second.step(scenario.pair()).loss_discriminator for _ in range(3)
        ]
        assert losses_first == pytest.approx(losses_second)


class TestCleanDomainInput:
    """ADR-0012 决策 3：判别器输入恒干净域——参数更新前向与 train 侧
    复算共享干净域打分入口（patch_logits），带噪训练入口退役（零调用）。"""

    def test_training_forward_uses_clean_entry_only(
        self, scenario: UpdateScenario,
    ) -> None:
        """一步更新的四次前向（复算 2 + 训练 2）全走 patch_logits；
        training_patch_logits 零调用（带噪入口退役的观测缝）。"""
        recording = RecordingCleanScorer(scenario.scorer())
        update = scenario.update(scorer=recording)
        update.step(scenario.pair())
        assert recording.noisy_calls == 0
        assert len(recording.clean_entries) == 4  # 复算 real/fake + 训练 real/fake

    def test_recompute_uses_no_grad_before_training_forward(
        self, scenario: UpdateScenario,
    ) -> None:
        """复算两次前向（序首）在 no_grad 下进行（AUC/准确率非可微、
        永不 backward）；训练前向（序后）带图（梯度流向判别器参数）。"""
        recording = RecordingCleanScorer(scenario.scorer())
        update = scenario.update(scorer=recording)
        update.step(scenario.pair())
        flags = [grad_on for _, grad_on in recording.clean_entries]
        assert flags == [False, False, True, True]

    def test_report_carries_train_pairwise_accuracy(
        self, scenario: UpdateScenario,
    ) -> None:
        """报告带 train pairwise 准确率，值 = 干净域 patch logit 的
        Mann-Whitney 占比（optimizer 冻结使复算值可离线重放——报告值与
        同批同权重的独立重放对账）。"""
        recording = RecordingCleanScorer(scenario.scorer())
        update = scenario.update(scorer=recording)
        update.optimizer.step = lambda: None  # 冻结权重：复算值可离线重放
        pair = scenario.pair()
        report = update.step(pair)
        assert 0.0 <= report.train_pairwise_acc <= 1.0
        real_batch, fake_batch = recording.clean_batches[:2]  # 复算序 = real 先
        expected = HeldOutAuc.auc_from_scores(
            recording._inner.patch_logits(real_batch).flatten(),
            recording._inner.patch_logits(fake_batch).flatten(),
        )
        assert report.train_pairwise_acc == pytest.approx(expected)

    def test_recompute_precedes_parameter_update(
        self, scenario: UpdateScenario,
    ) -> None:
        """复算发生在 optimizer.step 之前（与同 iteration 的 held-out AUC
        同刻——更新前判别器快照，ADR-0009-β 的时序语义）。观测缝：替换
        optimizer.step 注入快照比较——step 触发时刻的复算已完成。"""
        recording = RecordingCleanScorer(scenario.scorer())
        update = scenario.update(scorer=recording)
        step_markers: list[int] = []
        real_step = update.optimizer.step

        def spying_step() -> None:
            step_markers.append(len(recording.clean_entries))
            real_step()

        update.optimizer.step = spying_step
        update.step(scenario.pair())
        assert step_markers == [4]  # step 时刻四次干净域前向均已完成


class TestLossDecreases:
    def test_lsgan_loss_decreases_on_fixed_source(
        self, scenario: UpdateScenario,
    ) -> None:
        """AC：fixture 固定配对批上 LSGAN 损失随 Online update 下降
        （尾半段均值 < 头半段均值，且末步 < 首步）。"""
        update = scenario.update()
        pair = scenario.pair()
        losses = [
            update.step(pair).loss_discriminator for _ in range(30)
        ]
        half = len(losses) // 2
        assert losses[-1] < losses[0]
        assert sum(losses[half:]) / half < sum(losses[:half]) / half


class TestRealCapacityGuard:
    """ADR-0008-03 AC 5：装配期 real 容量守卫——逐 (rank 切片或全池,
    模态) real 容量 ≥ K，不足 fail-fast 可读报错（RealPoolSampler 无放回
    采样语义不动，不引入有放回采样补洞）。守卫是 manifest 对本视图
    逐模态计数的校验（装配缝先切片、后守卫）；消费面 = 装配原语的
    real 侧采样（ADR-0012 后判别器更新步不再内部自抽，守卫语义不变）。"""

    @staticmethod
    def _manifest(per_modality: dict[str, int]) -> LatentManifest:
        entries = [
            PoolEntry(
                case_id=f"case-{modality}-{index}",
                modality=modality,
                latent=f"real_pool_latents/{modality}-{index}.pt",
                spacing=(100.0, 100.0, 100.0),
            )
            for modality, count in per_modality.items()
            for index in range(count)
        ]
        return LatentManifest(
            kind="real_pool",
            encoder="guard-test",
            latent_shape=SHAPE,
            split_seed=0,
            split_sizes={"train": len(entries)},
            entries=entries,
        )

    def test_capacity_at_exact_k_passes(self) -> None:
        """每模态恰好 K 条：守卫放行（无放回采 K 条可行）。"""
        self._manifest({m: 4 for m in MODALITIES}).assert_condition_capacity(4, 1, list(MODALITIES))

    def test_capacity_above_k_passes(self) -> None:
        self._manifest({m: 110 for m in MODALITIES}).assert_condition_capacity(8, 1, list(MODALITIES))

    def test_starved_modality_rejected_with_readable_error(self) -> None:
        """单条件不足即拒（总量够、单条件不够不是放行理由——条件匹配
        采样后每条件独立供满 real 批）：报错点名条件、可用量与 K。"""
        manifest = self._manifest(
            {"t1n": 2, "t1c": 4, "t2w": 4, "t2f": 4},
        )
        with pytest.raises(ValueError, match="容量不足") as exc_info:
            manifest.assert_condition_capacity(4, 1, list(MODALITIES))
        message = str(exc_info.value)
        assert "t1n" in message and "2" in message and "4" in message
        assert "disc_batch_size_k" in message  # 可行动：点名 config knob

    def test_zero_capacity_modality_rejected(self) -> None:
        """某模态 0 条（稀疏模态切片断供的极端）：显式拒绝，不静默空采。"""
        manifest = self._manifest({"t1n": 4, "t1c": 4, "t2w": 4, "t2f": 0})
        with pytest.raises(ValueError, match="t2f"):
            manifest.assert_condition_capacity(4, 1, list(MODALITIES))

    def test_multi_rank_demand_is_k_times_world_size(self) -> None:
        """逐（rank 切片, 模态）语义：判定按全量做、需量 = K×world_size
        （条带切片每 rank 视图 ≥ K 的等价条件，且失败路径全 rank 一致）——
        世界 2 路下每模态 6 条 < 8 被拒（rank 切片后最弱视图 3 < 4）。"""
        manifest = self._manifest({m: 6 for m in MODALITIES})
        manifest.assert_condition_capacity(4, 1, list(MODALITIES))  # 单进程：6 ≥ 4 放行
        with pytest.raises(ValueError, match="容量不足") as exc_info:
            manifest.assert_condition_capacity(4, 2, list(MODALITIES))
        message = str(exc_info.value)
        assert "world_size=2" in message and "8" in message

    def test_rank_sliced_views_keep_k_under_passing_guard(self) -> None:
        """等价性锁：守卫放行的全量在真实条带切片后每 rank 视图每模态
        ≥ K（K=4、world=2、每模态 9 条 → 切片 5/4——最弱视图恰过线）。"""
        manifest = self._manifest({m: 9 for m in MODALITIES})
        manifest.assert_condition_capacity(4, 2, list(MODALITIES))
        for rank in (0, 1):
            view = RankSlicedPool(
                manifest, DistributedContext(rank, 2, True), MODALITIES,
            ).view()
            for modality in MODALITIES:
                assert view.modalities[modality] >= 4


class TestConditionLayeredSlicing:
    """rank 切片的分层轴 = 活动条件集（#129）：MR-RATE 多条件池按词汇表
    条件名分层——按 BraTS 四序列代码内副本切片会让每条目条件计数为零，
    换域线多 rank 运行在装配期被全 rank 一致地拒绝（不可达）。"""

    CONDITIONS: tuple[str, ...] = (
        "t1w/axial", "t1w/coronal", "flair/axial", "swi/axial",
    )

    def _mr_manifest(self) -> LatentManifest:
        """每条件 2 条、条件异形状的 MR 池（world=2 条带切片的输入面）。"""
        entries = [
            PoolEntry(
                case_id=f"case-{name}-{index}",
                modality=name,
                latent=f"real_pool_latents/{name}-{index}.pt",
                spacing=(100.0, 100.0, 100.0),
            )
            for name in self.CONDITIONS for index in range(2)
        ]
        return LatentManifest(
            kind="real_pool",
            encoder="mr-slice-test",
            latent_shape=(4, 16, 16, 8),
            split_seed=0,
            split_sizes={"train": len(entries)},
            entries=entries,
            condition_latent_shapes={
                name: (4, 16, 16, 8) for name in self.CONDITIONS
            },
        )

    def test_condition_bands_distribute_entries_by_rank(self) -> None:
        """每条件内部 entries[rank::world]：各片覆盖全部条件、逐条件 1 条。"""
        view = RankSlicedPool(
            self._mr_manifest(), DistributedContext(1, 2, True),
            self.CONDITIONS,
        ).view()
        assert [entry.case_id for entry in view.entries] == [
            f"case-{name}-1" for name in self.CONDITIONS
        ]
        assert view.modalities == {name: 1 for name in self.CONDITIONS}

    def test_brats_condition_copy_rejects_mr_pool(self) -> None:
        """分层轴退回 BraTS 四序列副本 → 每条目条件计数为零、全 rank
        一致拒绝（报错点名缺的正是池里的条件）——代码内副本是换域线多
        rank 不可达的根因，本测试锁定分层轴的注入面。"""
        with pytest.raises(
            ValueError, match="不足以支撑 2-路 rank 切片",
        ) as exc_info:
            RankSlicedPool(
                self._mr_manifest(), DistributedContext(0, 2, True),
                MODALITIES,
            ).view()
        message = str(exc_info.value)
        assert "t1n×0" in message and "t2f×0" in message
