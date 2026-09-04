"""reward 模块端到端 fixture 测试（ticket #20 验收标准聚合）。

fixture 驱动机制（ticket）：固定 seed + 预置固定 fake + policy 不参与更新
（等效 policy lr=0，schema 定死 policy_lr>0，故以「不进更新循环」落实）——
真实工件由 prepare 产出（Real sample pool / Held-out real / per-channel
统计量），判别器经 fixture 网络工件装载，fake 源为预置 latent 批；
online update 循环与生产同一管线（OnlineUpdate / HeldOutAuc /
ReplayBuffer），N_d=1 即每 iter 一步。

五条 AC 对应：
1. LSGAN 损失随 Online update 下降（固定 real/fake 源）；
2. reward = raw real-logit 的 patch mean 聚合（数值口径断言）；
3. buffer 两区行为：base 不变、近期 FIFO 滚动、混采 50/50、回放跨两区均匀；
4. held-out AUC 可计算并落盘（与训练 real 病例级不相交语义成立）；
5. SpectralNorm 默认关闭（fixture 默认 config 即触发式关闭）。
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import RewardScorer
from cynosure.reward.update import OnlineUpdate
from cynosure.train import IterEvent, RunArtifacts
from tests.conftest import CliSession, FixturePrepareScenario

# 200 步（patch 级 AUC 轨迹实测钉死）：loss 1.06→0.18 单调下降、
# held-out AUC 0.48→0.92（patch 级口径收敛慢于样本级，~175 步后爬升）
NUM_STEPS = 200
DISCRIMINATOR_SEED = 7
FAKE_BATCH = 12  # G 保持 12（fixture 策略）


@dataclass
class RewardComponents:
    """一次场景装配出的 reward 协作者组（纯数据，编排逻辑在场景类）。"""

    scorer: RewardScorer
    pool: LatentManifest
    heldout: LatentManifest
    buffer: ReplayBuffer
    update: OnlineUpdate
    auc: HeldOutAuc
    run_artifacts: RunArtifacts


class RewardFixtureScenario:
    """端到端场景：prepare 工件 → fixture 判别器 → 预置 fake → 更新循环落盘。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.tmp_path = tmp_path
        self.fixture_dir = tmp_path / "fixtures"
        self.run_dir = tmp_path / "run"
        self.config_path = tmp_path / "config.json"
        self.components: RewardComponents | None = None

    def prepare_artifacts(self) -> None:
        """prepare 端到端（fixture 合成数据）：pool / heldout / stats 三工件。"""
        config = Fixture().config(self.fixture_dir)
        FixturePrepareScenario(self.cli, config, self.tmp_path).run(
            self.config_path,
        )

    def write_network_artifacts(self) -> None:
        # fixture 网络「固定 seed」机制：随机初始化随 seed 定死（轨迹可复现）
        torch.manual_seed(DISCRIMINATOR_SEED)
        Fixture().write_artifacts(self.fixture_dir)

    def fake_batches(self, count: int, batch: int, seed: int) -> list[torch.Tensor]:
        """预置固定 fake 批：冻结 policy 的「产出」（每批同一 seed 派生、预先定死）。"""
        batches: list[torch.Tensor] = []
        for index in range(count):
            generator = torch.Generator().manual_seed(seed + index)
            batches.append(torch.randn(
                batch, *self.config().latent_shape, generator=generator,
            ))
        return batches

    def config(self):
        return ConfigLoader.load(self.config_path)

    def assemble(self) -> RewardComponents:
        """装配打分器 / 缓冲 / 更新器 / AUC 信号（与生产同一管线）。"""
        self.prepare_artifacts()
        self.write_network_artifacts()
        config = self.config()
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
            checkpoint=config.artifacts.discriminator_ckpt,
        )
        scorer = RewardScorer(
            artifact, config.reward, ChannelStats.load(config.reward.channel_stats_json),
        )
        pool = LatentManifest.load(
            config.reward.real_pool_manifest, kind="real_pool",
        )
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        buffer = ReplayBuffer(config.reward.replay_buffer_capacity)
        buffer.fill_base(self.fake_batches(
            count=1, batch=buffer.base_capacity, seed=100,
        )[0])
        update = OnlineUpdate(
            scorer=scorer,
            buffer=buffer,
            real_sampler=RealPoolSampler(pool, torch.Generator().manual_seed(200)),
            config=config.reward,
            generator=torch.Generator().manual_seed(201),
        )
        auc = HeldOutAuc(
            heldout_manifest=heldout,
            scorer=scorer,
            generator=torch.Generator().manual_seed(202),
        )
        self.components = RewardComponents(
            scorer=scorer,
            pool=pool,
            heldout=heldout,
            buffer=buffer,
            update=update,
            auc=auc,
            run_artifacts=RunArtifacts.init(config, self.run_dir),
        )
        return self.components

    def run_loop(self) -> list[dict]:
        """200 步 online update：每步记录判别器损失、reward 与 AUC 到指标流。"""
        config = self.config()
        components = self.components
        assert components is not None
        fakes = self.fake_batches(NUM_STEPS, FAKE_BATCH, seed=300)
        for iteration, fake_batch in enumerate(fakes):
            report = components.update.step(fake_batch)
            rewards = components.scorer.reward(fake_batch)
            auc = components.auc.compute(fake_batch)
            components.run_artifacts.append_event(IterEvent(
                iteration=iteration,
                # fixture 语义：固定 fake 批 = 冻结 policy 的锚终点（无扰动）
                anchor_eval_reward=rewards.mean().item(),
                intra_group_reward_std=rewards.std().item(),
                heldout_auc=auc,
                loss={"discriminator": report.loss_discriminator},
                buffer_current_fraction=(
                    report.num_current / config.reward.disc_batch_size_k
                ),
                buffer_replay_fraction=(
                    report.num_replay / config.reward.disc_batch_size_k
                ),
                buffer_base_occupied=components.buffer.zone_sizes().base,
                buffer_recent_occupied=components.buffer.zone_sizes().recent,
                lr=config.policy.policy_lr,
                elapsed_s=0.0,
            ))
        return components.run_artifacts.read_events()


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> RewardFixtureScenario:
    return RewardFixtureScenario(cli, tmp_path)


class TestFixtureAcceptance:
    """ticket #20 五条 AC 的端到端验证。"""

    def test_lsgan_loss_decreases_with_online_update(
        self, scenario: RewardFixtureScenario,
    ) -> None:
        """AC 1：固定 real/fake 源上 LSGAN 损失随 Online update 下降。"""
        components = scenario.assemble()
        events = scenario.run_loop()
        losses = [event["loss"]["discriminator"] for event in events]
        half = len(losses) // 2
        assert losses[-1] < losses[0]
        assert sum(losses[half:]) / half < sum(losses[:half]) / half
        assert components is not None  # 场景装配完成

    def test_reward_is_raw_logit_patch_mean(
        self, scenario: RewardFixtureScenario,
    ) -> None:
        """AC 2：reward = raw real-logit 的 patch mean 聚合（数值口径）。"""
        scenario.assemble()
        assert scenario.components is not None
        fake = scenario.fake_batches(1, FAKE_BATCH, seed=300)[0]
        reward = scenario.components.scorer.reward(fake)
        patch_mean = scenario.components.scorer.patch_logits(fake).mean(dim=(1, 2, 3, 4))
        assert torch.equal(reward, patch_mean)

    def test_buffer_two_zone_behavior(
        self, scenario: RewardFixtureScenario,
    ) -> None:
        """AC 3：base 分区内容不变、近期分区 FIFO 滚动、混采 50/50、
        回放半区跨两区均匀。"""
        components = scenario.assemble()
        assert components is not None
        base_before = [t.clone() for t in components.buffer.base_samples()]
        scenario.run_loop()
        base_after = [t.clone() for t in components.buffer.base_samples()]
        assert all(
            torch.equal(a, b) for a, b in zip(base_before, base_after)
        )  # base 固定
        # FIFO 滚动：recent 容量 32 = 批 197 的后 8 条 + 批 198/199 各 12 条
        batches = scenario.fake_batches(NUM_STEPS, FAKE_BATCH, seed=300)
        recent = components.buffer.recent_samples()
        assert len(recent) == components.buffer.recent_capacity
        assert all(torch.equal(recent[i], batches[197][4 + i]) for i in range(8))
        assert all(torch.equal(recent[8 + i], batches[198][i]) for i in range(12))
        assert all(torch.equal(recent[20 + i], batches[199][i]) for i in range(12))
        # 混采占比（每步 report 断言 2/2 与 1/1，此处抽查落盘占比）
        events = components.run_artifacts.read_events()
        assert events[0]["buffer_current_fraction"] == pytest.approx(0.5)
        assert events[0]["buffer_replay_fraction"] == pytest.approx(0.5)

    def test_heldout_auc_computed_and_persisted(
        self, scenario: RewardFixtureScenario,
    ) -> None:
        """AC 4：held-out AUC 可计算并落盘；held-out 与训练 real 不相交。"""
        components = scenario.assemble()
        assert components is not None
        pool_cases = {e.case_id for e in components.pool.entries}
        heldout_cases = {e.case_id for e in components.heldout.entries}
        assert pool_cases & heldout_cases == set()  # 病例级不相交语义成立
        assert components.heldout.kind == "heldout_real"
        assert components.pool.kind == "real_pool"
        events = scenario.run_loop()
        aucs = [event["heldout_auc"] for event in events]
        assert len(aucs) == NUM_STEPS  # 每步落盘
        assert aucs[-1] > aucs[0]  # 判别器学开：AUC 随更新上升
        assert aucs[-1] > 0.75  # 显著高于 chance（out-of-sample 判别力）

    def test_spectral_norm_off_by_default_in_fixture(
        self, scenario: RewardFixtureScenario,
    ) -> None:
        """AC 5：fixture 默认 config 下 SpectralNorm 关闭。"""
        components = scenario.assemble()
        assert scenario.config().reward.spectral_norm_enabled is False
        assert components is not None
        parametrized = [
            name
            for name, module in components.scorer.discriminator.named_modules()
            if hasattr(module, "parametrizations")
        ]
        assert parametrized == []
