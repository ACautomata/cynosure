"""Online update 一步测试（ticket #20 AC）：50% 当前 / 50% 回放混采、
LSGAN 损失随更新下降、AdamW 5e-5、real 批采样、更新后 fake 入 buffer。

fake 源（Fixture 策略）：固定 seed + 预置固定 fake + policy 不参与更新
（等效 policy lr=0）——OnlineUpdate 的 fake 批由调用方注入（生产 =
policy rollout 输出，fixture = 预置固定 latent 批），一步 = 采 real 批 →
混采 fake 批 → LSGAN loss → AdamW step → 当前 fake push 入近期分区。

ADR-0008-01：step 穿本 iteration 目标模态——回放半区按该条件过滤，
入区 fake 带同标签（real 侧条件匹配归 ADR-0008-03）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES, CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats, LatentManifest, PoolEntry
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.sampler import RealPoolSampler, assert_real_capacity
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate

SHAPE = (4, 16, 16, 8)


class WrittenPool:
    """不经 prepare 直写的最小 Real sample pool 工件（manifest + latent 文件）。"""

    def __init__(self, root: Path, num_cases: int, seed: int) -> None:
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
            # 模拟 prepare 合成预编码：影像 [1,64,64,32] → 4×4×4 块均值 → 通道加权
            noise = torch.randn(1, 64, 64, 32, generator=generator)
            pooled = torch.nn.functional.avg_pool3d(noise, 4)  # [1,16,16,8]
            weights = torch.tensor([0.5, 0.75, 1.0, 1.25]).view(4, 1, 1, 1)
            latent = pooled * weights  # [4,16,16,8]
            latent_path = latent_dir / f"{index}.pt"
            torch.save(latent, latent_path)
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


class RecordingRealSampler:
    """测试仪器：记录 real 侧采样的 (count, modality) 并返回确定性批
    （RealSampling 协议替身——real 侧条件匹配穿参的观测缝）。"""

    def __init__(self, latent_shape: tuple[int, ...]) -> None:
        self._shape = latent_shape
        self.calls: list[tuple[int, str]] = []

    @property
    def size(self) -> int:
        return 1024

    def sample(self, count: int, *, modality: str | None = None) -> torch.Tensor:
        self.calls.append((count, modality))
        return torch.zeros(count, *self._shape)


class UpdateScenario:
    """OnlineUpdate 单元场景：pool 工件 + fixture 判别器 + 满 buffer。"""

    def __init__(self, tmp_path: Path) -> None:
        torch.manual_seed(0)  # fixture 网络工件确定性（随机初始化随 seed 定死）
        fixture = Fixture()
        fixture.write_artifacts(tmp_path)
        self.config = fixture.config(tmp_path)
        self.pool_path = WrittenPool(tmp_path, num_cases=16, seed=1).write()

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
        return torch.randn(count, *SHAPE, generator=self.generator(seed))

    def update(
        self, real_sampler: RealPoolSampler | RecordingRealSampler | None = None,
    ) -> tuple[OnlineUpdate, ReplayBuffer]:
        buffer = ReplayBuffer(self.config.reward.replay_buffer_capacity)
        buffer.fill_base(
            self.fakes(32, seed=11),
            [MODALITIES[i % 4] for i in range(32)],  # 每模态 8 条
        )
        buffer.push(self.fakes(12, seed=22), "t2w")  # 预填 recent：混采即可 1+1
        pool = LatentManifest.load(self.pool_path, kind="real_pool")
        update = OnlineUpdate(
            scorer=self.scorer(),
            buffer=buffer,
            real_sampler=(
                real_sampler if real_sampler is not None
                else RealPoolSampler(pool, self.generator(5))
            ),
            config=self.config.reward,
            generator=self.generator(6),
        )
        return update, buffer

    def depleted_condition_update(
        self, modality: str,
        real_sampler: RealPoolSampler | RecordingRealSampler | None = None,
    ) -> tuple[OnlineUpdate, ReplayBuffer]:
        """该条件回放候选耗尽的场景（退化路径专用）：base 全部另一条件
        填充、recent 不动——``condition_supply(modality)`` = 0 <
        回放半区需求。"""
        buffer = ReplayBuffer(self.config.reward.replay_buffer_capacity)
        buffer.fill_base(
            self.fakes(32, seed=11),
            ["t1n"] * 32,
        )
        pool = LatentManifest.load(self.pool_path, kind="real_pool")
        update = OnlineUpdate(
            scorer=self.scorer(),
            buffer=buffer,
            real_sampler=(
                real_sampler if real_sampler is not None
                else RealPoolSampler(pool, self.generator(5))
            ),
            config=self.config.reward,
            generator=self.generator(6),
        )
        assert buffer.condition_supply(modality) == 0
        return update, buffer


@pytest.fixture
def scenario(tmp_path: Path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class TestMixComposition:
    def test_half_current_half_replay(self, scenario: UpdateScenario) -> None:
        """AC：混采占比 50/50（K=4 → 2 当前 + 2 回放）。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t2w")
        assert report.num_current == 2
        assert report.num_replay == 2

    def test_replay_half_split_across_zones(self, scenario: UpdateScenario) -> None:
        """AC：回放半区跨两区均匀（K=4 → base 1 + recent 1）。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t2w")
        assert report.num_base_replay == 1
        assert report.num_recent_replay == 1

    def test_step_pushes_current_fakes_into_recent(self, scenario: UpdateScenario) -> None:
        """当前 fake 全部入近期分区（buffer 记录近期 policy 分布）。"""
        update, buffer = scenario.update()
        fakes = scenario.fakes(12)
        recent_before = buffer.zone_sizes().recent
        update.step(fakes, "t2w")
        assert buffer.zone_sizes().recent == recent_before + 12
        assert any(
            torch.equal(fakes[0], entry.latent)
            for entry in buffer.recent_samples()
        )

    def test_step_labels_pushed_fakes_with_condition(self, scenario: UpdateScenario) -> None:
        """ADR-0008-01：入区 fake 带本 iteration 条件标签（目标模态）。"""
        update, buffer = scenario.update()
        update.step(scenario.fakes(12), "t1n")
        pushed = buffer.recent_samples()[-12:]
        assert all(entry.modality == "t1n" for entry in pushed)

    def test_replay_draw_carries_condition_labels(self, scenario: UpdateScenario) -> None:
        """ADR-0008-01：回放采样结果的标签观测与过滤条件一致。"""
        update, _ = scenario.update()
        draw = update.buffer.sample_replay(
            2, scenario.generator(9), "t2w",
        )
        assert draw.modalities == ["t2w", "t2w"]

    def test_replay_shortage_raises_for_condition(self, scenario: UpdateScenario) -> None:
        """ADR-0008-01：该条件回放候选不足显式拒绝（可区分于总数不足，
        绝不静默回退全池混采）——recent 预填的 t2w 不在 t1n 候选内。"""
        update, _ = scenario.update()
        with pytest.raises(ValueError, match="t1n"):
            # base 每条件 8 条：需求 9 超出 t1n 候选（base 8 + recent 0）
            update.buffer.sample_replay(9, scenario.generator(9), "t1n")


class TestConditionMatchedRealSide:
    """ADR-0008-03 AC 1：在线更新 real 侧按本 iteration 目标模态匹配
    采样（与 held-out AUC 同条件归因口径同源）——real 侧候选不足属装配
    守卫（assert_real_capacity）的拒绝面，采样语义本身不动。"""

    def test_real_batch_matches_iteration_condition(
        self, scenario: UpdateScenario,
    ) -> None:
        """real 批以本 iteration 目标模态过滤采样（观测缝：RealSampling
        替身记录 (count, modality)）。"""
        sampler = RecordingRealSampler(SHAPE)
        update, _ = scenario.update(real_sampler=sampler)
        update.step(scenario.fakes(12), "t1n")
        assert sampler.calls == [(4, "t1n")]  # K=4、条件 = 步条件

    def test_report_carries_condition(self, scenario: UpdateScenario) -> None:
        """AC：UpdateReport 带条件——本步更新归因的目标模态。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t1c")
        assert report.modality == "t1c"

    def test_condition_filtering_keeps_mix_ratio_semantics(
        self, scenario: UpdateScenario,
    ) -> None:
        """AC：两区混采配比语义（各半、可互补）在条件匹配下保持——
        real 侧条件化不改变 fake 侧混采构成。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t2f")
        assert report.num_current == 2
        assert report.num_replay == 2
        assert (report.num_base_replay, report.num_recent_replay) == (1, 1)


class TestReplayShortageDegradation:
    """ADR-0008-03 AC 2：回放不足退化路径——该步纯 current 半区（回放
    0 条）+ 可观测标记，real 侧与退化后批同量匹配，不静默漂移。"""

    def test_short_condition_degrades_to_pure_current_half(
        self, scenario: UpdateScenario,
    ) -> None:
        """该条件候选 < 回放半区需求：回放 0 条、批 = 当前半区、报告带
        退化标记（不静默漂移为全池混采或半途截断的回放）。"""
        update, _ = scenario.depleted_condition_update("t2w")
        report = update.step(scenario.fakes(12), "t2w")
        assert report.replay_degraded is True
        assert report.num_replay == 0
        assert (report.num_base_replay, report.num_recent_replay) == (0, 0)
        assert report.num_current == 2  # 批 = 当前半区（K=4 → 2 条）
        assert report.modality == "t2w"

    def test_degraded_real_batch_matches_degraded_size(
        self, scenario: UpdateScenario,
    ) -> None:
        """real 侧与退化后批同量匹配（不再抽满 K）——真两侧批等量，
        LSGAN 损失的 real/fake 配对语义保持。"""
        sampler = RecordingRealSampler(SHAPE)
        update, _ = scenario.depleted_condition_update("t2w", sampler)
        update.step(scenario.fakes(12), "t2w")
        assert sampler.calls == [(2, "t2w")]  # 与退化后批（当前半区 2 条）同量

    def test_degraded_step_still_trains_and_pushes(
        self, scenario: UpdateScenario,
    ) -> None:
        """退化步判别器照常更新、当前 fake 照常带标签入近期分区——
        回放缺失不停摆更新（gated 条件的判别器持续受训语义）。"""
        update, buffer = scenario.depleted_condition_update("t2w")
        weight = next(
            p for p in update.scorer.discriminator.parameters() if p.ndim == 5
        )
        before = weight.detach().clone()
        fakes = scenario.fakes(12)
        report = update.step(fakes, "t2w")
        assert not torch.equal(before, weight.detach())
        pushed = buffer.recent_samples()
        assert len(pushed) == 12
        assert all(entry.modality == "t2w" for entry in pushed)

    def test_sufficient_condition_reports_no_degradation(
        self, scenario: UpdateScenario,
    ) -> None:
        """正常混采步退化标记为 False（观测面零歧义：0 回放占比只可能
        出现在退化步或 N_d 跳过，标记区分二者）。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t2w")
        assert report.replay_degraded is False
        assert report.num_replay == 2


class TestUpdateStep:
    def test_report_loss_decomposes_into_real_and_fake_terms(
        self, scenario: UpdateScenario,
    ) -> None:
        """loss = mean((D(real)−1)²) + mean(D(fake)²)：报告两项分解一致。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12), "t2w")
        assert report.loss_discriminator == pytest.approx(
            report.loss_real_term + report.loss_fake_term, abs=1e-6,
        )

    def test_optimizer_is_adamw_with_configured_lr(
        self, scenario: UpdateScenario,
    ) -> None:
        """AdamW、lr = disc_lr（默认 5e-5）。"""
        update, _ = scenario.update()
        assert type(update.optimizer).__name__ == "AdamW"
        assert update.optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)

    def test_optimizer_weight_decay_is_explicit(
        self, scenario: UpdateScenario,
    ) -> None:
        """卫生项（ADR-0007）：判别器 AdamW 的 weight_decay 显式取 config
        值（与 policy 侧同值口径 1e-4），不再是隐式 PyTorch 默认 0.01。"""
        update, _ = scenario.update()
        decay = update.optimizer.param_groups[0]["weight_decay"]
        assert decay == pytest.approx(1e-4)
        assert decay == pytest.approx(scenario.config.policy.policy_weight_decay)

    def test_step_changes_discriminator_weights(
        self, scenario: UpdateScenario,
    ) -> None:
        """一步更新后判别器权重发生变化（online update 生效）。"""
        update, _ = scenario.update()
        weight = next(
            p for p in update.scorer.discriminator.parameters() if p.ndim == 5
        )
        before = weight.detach().clone()
        update.step(scenario.fakes(12), "t2w")
        assert not torch.equal(before, weight.detach())

    def test_identical_generator_sequences_reproduce_losses(
        self, scenario: UpdateScenario,
    ) -> None:
        """固定 seed + 预置固定 fake：同场景重放 → 损失轨迹逐位一致。"""
        first, _ = scenario.update()
        second, _ = scenario.update()
        losses_first = [first.step(scenario.fakes(12), "t2w").loss_discriminator for _ in range(3)]
        losses_second = [second.step(scenario.fakes(12), "t2w").loss_discriminator for _ in range(3)]
        assert losses_first == pytest.approx(losses_second)

    def test_real_batch_comes_from_pool(self, scenario: UpdateScenario) -> None:
        """real 批来自 Real sample pool 采样（K 条、形状契约）。"""
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        sampler = RealPoolSampler(pool, scenario.generator(9))
        real = sampler.sample(4)
        assert tuple(real.shape) == (4, *SHAPE)
        assert len({t[0, 0, 0, 0].item() for t in real}) == 4  # 无放回、条条不同

    def test_real_sample_beyond_pool_rejected(self, scenario: UpdateScenario) -> None:
        pool = LatentManifest.load(scenario.pool_path, kind="real_pool")
        sampler = RealPoolSampler(pool, scenario.generator(9))
        with pytest.raises(ValueError, match="pool"):
            sampler.sample(len(pool.entries) + 1)


class TestRealCapacityGuard:
    """ADR-0008-03 AC 5：装配期 real 容量守卫——逐 (rank 切片或全池,
    模态) real 容量 ≥ K，不足 fail-fast 可读报错（RealPoolSampler 无放回
    采样语义不动，不引入有放回采样补洞）。"""

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
        assert_real_capacity(self._manifest({m: 4 for m in MODALITIES}), 4)

    def test_capacity_above_k_passes(self) -> None:
        assert_real_capacity(self._manifest({m: 110 for m in MODALITIES}), 8)

    def test_starved_modality_rejected_with_readable_error(self) -> None:
        """单条件不足即拒（总量够、单条件不够不是放行理由——条件匹配
        采样后每条件独立供满 real 批）：报错点名条件、可用量与 K。"""
        manifest = self._manifest(
            {"t1n": 2, "t1c": 4, "t2w": 4, "t2f": 4},
        )
        with pytest.raises(ValueError, match="容量不足") as exc_info:
            assert_real_capacity(manifest, 4)
        message = str(exc_info.value)
        assert "t1n" in message and "2" in message and "4" in message
        assert "disc_batch_size_k" in message  # 可行动：点名 config knob

    def test_zero_capacity_modality_rejected(self) -> None:
        """某模态 0 条（稀疏模态切片断供的极端）：显式拒绝，不静默空采。"""
        manifest = self._manifest({"t1n": 4, "t1c": 4, "t2w": 4, "t2f": 0})
        with pytest.raises(ValueError, match="t2f"):
            assert_real_capacity(manifest, 4)

    def test_guard_consumes_rank_sliced_view_by_contract(self) -> None:
        """守卫消费「采样器实际看到的 manifest」（rank 切片视图或全池）：
        函数对传入 manifest 的逐模态计数判定，切片语义归 RankSlicedPool
        （装配缝先切片后守卫——本测试锁守卫面与切片视图的契约方向）。"""
        manifest = self._manifest({m: 4 for m in MODALITIES})
        sliced = manifest.with_entries(manifest.entries[:2])  # 模拟切片视图
        with pytest.raises(ValueError, match="容量不足"):
            assert_real_capacity(sliced, 4)


class TestLossDecreases:
    def test_lsgan_loss_decreases_on_fixed_source(
        self, scenario: UpdateScenario,
    ) -> None:
        """AC：fixture 固定 real/fake 上 LSGAN 损失随 Online update 下降
        （尾半段均值 < 头半段均值，且末步 < 首步）。"""
        update, _ = scenario.update()
        fakes = scenario.fakes(12)
        losses = [update.step(fakes, "t2w").loss_discriminator for _ in range(30)]
        half = len(losses) // 2
        assert losses[-1] < losses[0]
        assert sum(losses[half:]) / half < sum(losses[:half]) / half
