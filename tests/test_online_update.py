"""Online update 一步测试（ticket #20 AC）：50% 当前 / 50% 回放混采、
LSGAN 损失随更新下降、AdamW 5e-5、real 批采样、更新后 fake 入 buffer。

fake 源（Fixture 策略）：固定 seed + 预置固定 fake + policy 不参与更新
（等效 policy lr=0）——OnlineUpdate 的 fake 批由调用方注入（生产 =
policy rollout 输出，fixture = 预置固定 latent 批），一步 = 采 real 批 →
混采 fake 批 → LSGAN loss → AdamW step → 当前 fake push 入近期分区。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats, LatentManifest, PoolEntry
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.sampler import RealPoolSampler
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

    def update(self) -> tuple[OnlineUpdate, ReplayBuffer]:
        buffer = ReplayBuffer(self.config.reward.replay_buffer_capacity)
        buffer.fill_base(self.fakes(32, seed=11))
        buffer.push(self.fakes(12, seed=22))  # 预填 recent：混采即可 1+1
        pool = LatentManifest.load(self.pool_path, kind="real_pool")
        update = OnlineUpdate(
            scorer=self.scorer(),
            buffer=buffer,
            real_sampler=RealPoolSampler(pool, self.generator(5)),
            config=self.config.reward,
            generator=self.generator(6),
        )
        return update, buffer


@pytest.fixture
def scenario(tmp_path: Path) -> UpdateScenario:
    return UpdateScenario(tmp_path)


class TestMixComposition:
    def test_half_current_half_replay(self, scenario: UpdateScenario) -> None:
        """AC：混采占比 50/50（K=4 → 2 当前 + 2 回放）。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12))
        assert report.num_current == 2
        assert report.num_replay == 2

    def test_replay_half_split_across_zones(self, scenario: UpdateScenario) -> None:
        """AC：回放半区跨两区均匀（K=4 → base 1 + recent 1）。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12))
        assert report.num_base_replay == 1
        assert report.num_recent_replay == 1

    def test_step_pushes_current_fakes_into_recent(self, scenario: UpdateScenario) -> None:
        """当前 fake 全部入近期分区（buffer 记录近期 policy 分布）。"""
        update, buffer = scenario.update()
        fakes = scenario.fakes(12)
        recent_before = buffer.zone_sizes().recent
        update.step(fakes)
        assert buffer.zone_sizes().recent == recent_before + 12
        assert any(
            torch.equal(fakes[0], sample) for sample in buffer.recent_samples()
        )


class TestUpdateStep:
    def test_report_loss_decomposes_into_real_and_fake_terms(
        self, scenario: UpdateScenario,
    ) -> None:
        """loss = mean((D(real)−1)²) + mean(D(fake)²)：报告两项分解一致。"""
        update, _ = scenario.update()
        report = update.step(scenario.fakes(12))
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

    def test_step_changes_discriminator_weights(
        self, scenario: UpdateScenario,
    ) -> None:
        """一步更新后判别器权重发生变化（online update 生效）。"""
        update, _ = scenario.update()
        weight = next(
            p for p in update.scorer.discriminator.parameters() if p.ndim == 5
        )
        before = weight.detach().clone()
        update.step(scenario.fakes(12))
        assert not torch.equal(before, weight.detach())

    def test_identical_generator_sequences_reproduce_losses(
        self, scenario: UpdateScenario,
    ) -> None:
        """固定 seed + 预置固定 fake：同场景重放 → 损失轨迹逐位一致。"""
        first, _ = scenario.update()
        second, _ = scenario.update()
        losses_first = [first.step(scenario.fakes(12)).loss_discriminator for _ in range(3)]
        losses_second = [second.step(scenario.fakes(12)).loss_discriminator for _ in range(3)]
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


class TestLossDecreases:
    def test_lsgan_loss_decreases_on_fixed_source(
        self, scenario: UpdateScenario,
    ) -> None:
        """AC：fixture 固定 real/fake 上 LSGAN 损失随 Online update 下降
        （尾半段均值 < 头半段均值，且末步 < 首步）。"""
        update, _ = scenario.update()
        fakes = scenario.fakes(12)
        losses = [update.step(fakes).loss_discriminator for _ in range(30)]
        half = len(losses) // 2
        assert losses[-1] < losses[0]
        assert sum(losses[half:]) / half < sum(losses[:half]) / half
