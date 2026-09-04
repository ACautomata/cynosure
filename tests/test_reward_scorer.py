"""RewardScorer 数值口径测试（ticket #20）：raw real-logit patch 聚合、
LSGAN 损失、per-channel 标准化、SpectralNorm 触发开关。

数值口径（reward-model 章 + ADR-0001）：
- reward = patch logit 图的聚合（mean 为主 / min 消融），不过 sigmoid；
- LSGAN 判别器损失 = mean((D(real) − 1)²) + mean(D(fake)²)；
- 输入 latent 先做 per-channel 标准化（消费 prepare 的统计量工件）；
- SpectralNorm 默认关闭、经 config 触发启用（叠在 conv 上）。
"""

from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats
from cynosure.reward.scorer import ChannelNormalizer, RewardScorer


class ScorerScenario:
    """scorer 单元测试场景：fixture 判别器工件 + 合成统计量 + 固定 latent 批。"""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        fixture = Fixture()
        fixture.write_artifacts(tmp_path)
        self.config = fixture.config(tmp_path)

    def artifact(self) -> NetworkArtifact:
        config = self.config
        return NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.discriminator_config_json),
            checkpoint=config.artifacts.discriminator_ckpt,
        )

    def stats(self, mean: list[float] | None = None,
              std: list[float] | None = None) -> ChannelStats:
        return ChannelStats(
            mean=mean or [0.0, 0.0, 0.0, 0.0],
            std=std or [1.0, 1.0, 1.0, 1.0],
            num_latents=1,
            latent_shape=self.config.latent_shape,
            source_manifest="real_pool.json",
        )

    def scorer(self, **reward_overrides) -> RewardScorer:
        config: CynosureConfig = self.config
        reward_config = config.reward.model_copy(update=reward_overrides)
        return RewardScorer(self.artifact(), reward_config, self.stats())

    def latents(self, batch: int = 3) -> torch.Tensor:
        return torch.randn(batch, *self.config.latent_shape)


@pytest.fixture
def scenario(tmp_path: Path) -> ScorerScenario:
    return ScorerScenario(tmp_path)


class TestRewardNumericContract:
    def test_reward_is_patch_mean_of_raw_logits(self, scenario: ScorerScenario) -> None:
        """AC：reward = raw real-logit 的 patch mean 聚合（不过 sigmoid）。

        独立参照：patch logit 图逐样本 mean，与 reward 数值全等。
        """
        scorer = scenario.scorer()
        latents = scenario.latents()
        patch_logits = scorer.patch_logits(latents)
        assert patch_logits.dim() == 5  # [B,1,D',H',W'] patch 图
        assert patch_logits.shape[0] == latents.shape[0]
        assert patch_logits.shape[1] == 1
        expected = patch_logits.mean(dim=(1, 2, 3, 4))
        reward = scorer.reward(latents)
        assert reward.shape == (latents.shape[0],)
        assert torch.allclose(reward, expected)

    def test_reward_is_not_squashed_by_sigmoid(self, scenario: ScorerScenario) -> None:
        """raw logit 不过 sigmoid：reward 与 patch logit 图 mean 逐值全等，
        不存在任何有界映射（sigmoid/概率口径会被此断言排除）。"""
        scorer = scenario.scorer()
        latents = scenario.latents()
        patch_mean = scorer.patch_logits(latents).mean(dim=(1, 2, 3, 4))
        assert torch.equal(scorer.reward(latents), patch_mean)

    def test_min_aggregation_ablation(self, scenario: ScorerScenario) -> None:
        """消融轴维 C：patch_aggregation="min" → 逐样本 min 聚合。"""
        scorer = scenario.scorer(patch_aggregation="min")
        latents = scenario.latents()
        patch_logits = scorer.patch_logits(latents)
        expected = patch_logits.amin(dim=(1, 2, 3, 4))
        assert torch.allclose(scorer.reward(latents), expected)

    def test_tanh_bounding_when_triggered(self, scenario: ScorerScenario) -> None:
        """触发式有界化：reward_tanh_bounding=true → reward = tanh(patch mean)。"""
        scorer = scenario.scorer(reward_tanh_bounding=True)
        latents = scenario.latents()
        patch_mean = scorer.patch_logits(latents).mean(dim=(1, 2, 3, 4))
        reward = scorer.reward(latents)
        assert torch.allclose(reward, torch.tanh(patch_mean))
        assert torch.all(reward.abs() < 1.0)  # (−1,1) 廉价保险语义


class TestLsganLoss:
    def test_loss_is_squared_errors_on_patch_logits(self, scenario: ScorerScenario) -> None:
        """LSGAN 数值口径：mean((D(real) − 1)²) + mean(D(fake)²)，逐 patch 元素。"""
        scorer = scenario.scorer()
        logits_real = torch.tensor([[[[[0.5, -0.5]]]]])  # [1,1,1,1,2]
        logits_fake = torch.tensor([[[[[0.25, 0.75]]]]])
        expected_real = float(((logits_real - 1.0) ** 2).mean())
        expected_fake = float((logits_fake ** 2).mean())
        terms = scorer.discriminator_terms(logits_real, logits_fake)
        assert terms.total.item() == pytest.approx(expected_real + expected_fake)
        assert terms.total.shape == ()  # 标量
        assert terms.real_term.item() == pytest.approx(expected_real)
        assert terms.fake_term.item() == pytest.approx(expected_fake)

    def test_perfect_discriminator_has_zero_loss(self, scenario: ScorerScenario) -> None:
        """D(real)=1、D(fake)=0 时 LSGAN 损失为 0（目标点口径）。"""
        scorer = scenario.scorer()
        terms = scorer.discriminator_terms(
            torch.ones(2, 1, 2, 2, 2), torch.zeros(2, 1, 2, 2, 2),
        )
        assert terms.total.item() == pytest.approx(0.0, abs=1e-6)


class TestChannelNormalization:
    def test_scorer_consumes_channel_stats_artifact(
        self, scenario: ScorerScenario,
    ) -> None:
        """per-channel 标准化消费统计量工件：std=2 时 x 与 2x 标准化后相同，
        判别器输出必须全等（否则统计量未生效）。"""
        plain = scenario.scorer()  # mean=0、std=1：normalize(x) = x
        config = scenario.config
        scaled = RewardScorer(
            scenario.artifact(),  # 同一 ckpt：权重与 plain 全等
            config.reward.model_copy(update={}),
            scenario.stats(std=[2.0, 2.0, 2.0, 2.0]),  # normalize(2x) = x
        )
        latents = scenario.latents()
        assert torch.allclose(
            scaled.patch_logits(latents * 2.0),
            plain.patch_logits(latents),
        )

    def test_channel_count_guard(self, scenario: ScorerScenario) -> None:
        """latent 通道数定死 4（VAE latent_channels=4）：5 通道输入显式拒绝。"""
        scorer = scenario.scorer()
        bad = torch.randn(2, 5, 16, 16, 8)
        with pytest.raises(ValueError, match="通道"):
            scorer.patch_logits(bad)


class TestSpectralNorm:
    def test_disabled_by_default(self, scenario: ScorerScenario) -> None:
        """AC：SpectralNorm 默认关闭——conv 层无谱归一化 parametrization。"""
        scorer = scenario.scorer()
        parametrized = [
            name for name, module in scorer.discriminator.named_modules()
            if hasattr(module, "parametrizations")
        ]
        assert parametrized == []

    def test_enabled_via_config(self, scenario: ScorerScenario) -> None:
        """AC：spectral_norm_enabled=true 触发——conv 带谱归一化且前向正常。"""
        scorer = scenario.scorer(spectral_norm_enabled=True)
        parametrized = [
            name for name, module in scorer.discriminator.named_modules()
            if hasattr(module, "parametrizations")
        ]
        assert parametrized, "启用后 conv 应带谱归一化 parametrization"
        assert all(name.endswith(".conv") for name in parametrized)
        reward = scorer.reward(scenario.latents())
        assert torch.isfinite(reward).all()


class TestModuleAggregation:
    """设备迁移结构（nn.Module 聚合）：统计量注册为 buffer、scorer 单点 .to()。

    reward 侧的 device 迁移由 torch 的 ``.to()`` 递归机制单点接管：
    ChannelNormalizer 的 mean/std 是持久统计量（register_buffer），与
    判别器参数同属 RewardScorer 的模块树；trainer 装配只调一次
    ``scorer.to(device)``。fixture 测试面 CPU-only，cuda 分支留 M0 门槛。
    """

    def test_normalizer_registers_stats_as_buffers(
        self, scenario: ScorerScenario,
    ) -> None:
        """mean/std 是注册 buffer（named_buffers 可见），不是普通属性。"""
        normalizer = ChannelNormalizer(scenario.stats())
        buffers = dict(normalizer.named_buffers())
        assert set(buffers) == {"mean", "std"}
        assert buffers["mean"].shape == (4, 1, 1, 1)
        assert buffers["std"].shape == (4, 1, 1, 1)

    def test_scorer_aggregates_discriminator_and_normalizer_as_submodules(
        self, scenario: ScorerScenario,
    ) -> None:
        """normalizer 与 discriminator 都是 scorer 的注册子模块（属性赋值
        即注册）——``.to()`` 沿模块树递归即可触达两者。"""
        scorer = scenario.scorer()
        modules = dict(scorer.named_modules())
        assert modules.get("_normalizer") is not None
        assert modules["_discriminator"] is scorer.discriminator

    def test_scorer_to_moves_parameters_and_buffers(
        self, scenario: ScorerScenario,
    ) -> None:
        """``scorer.to(device)`` 单点迁移：判别器参数与统计量 buffer 同 device。"""
        scorer = scenario.scorer()
        returned = scorer.to(torch.device("cpu"))
        assert returned is scorer  # nn.Module.to 链式契约
        assert scorer.discriminator.parameters().__next__().device.type == "cpu"
        buffers = dict(scorer.named_buffers())
        assert set(buffers) >= {"_normalizer.mean", "_normalizer.std"}
        assert all(b.device.type == "cpu" for b in buffers.values())

    def test_reward_invariant_under_to(self, scenario: ScorerScenario) -> None:
        """纯结构迁移：``.to()`` 前后同批 reward 逐位不变（数值口径不动）。"""
        scorer = scenario.scorer()
        latents = scenario.latents()
        before = scorer.reward(latents)
        scorer.to(torch.device("cpu"))
        assert torch.equal(before, scorer.reward(latents))

    def test_device_mismatch_rejected_explicitly(
        self, scenario: ScorerScenario,
    ) -> None:
        """normalize 的 device 契约 fail-fast：输入与统计量 buffer device
        不符时显式拒绝（不再静默 ``.to()`` 对齐——错位输入即装配契约违例）。"""
        normalizer = ChannelNormalizer(scenario.stats())
        meta_latents = torch.empty(2, 4, 8, 8, 4, device=torch.device("meta"))
        with pytest.raises(ValueError, match="device"):
            normalizer.normalize(meta_latents)


class TestDepthAnchor:
    def test_network_json_depth_must_match_config(self, scenario: ScorerScenario) -> None:
        """数值锚：网络配置 JSON 的 num_layers_d 与 RewardConfig.disc_num_layers_d
        必须一致（消融扫深度 = 换网络配置，静默错位即拒绝）。"""
        artifact = scenario.artifact()
        artifact.config["num_layers_d"] = 2  # fixture config 为 1
        with pytest.raises(ValueError, match="num_layers_d"):
            RewardScorer(artifact, scenario.config.reward, scenario.stats())

    def test_network_json_must_declare_depth(self, scenario: ScorerScenario) -> None:
        """JSON 缺 num_layers_d 键同样拒绝：默认 3 会让锚静默失效。"""
        artifact = scenario.artifact()
        del artifact.config["num_layers_d"]
        with pytest.raises(ValueError, match="num_layers_d"):
            RewardScorer(artifact, scenario.config.reward, scenario.stats())


class TestNormAndScaleContract:
    def test_batch_norm_json_rejected(self, scenario: ScorerScenario) -> None:
        """GroupNorm 定死（ADR-0001）：JSON 用 MONAI 默认 BATCH 被拒绝。"""
        artifact = scenario.artifact()
        artifact.config["norm"] = "BATCH"
        with pytest.raises(ValueError, match="GroupNorm"):
            RewardScorer(artifact, scenario.config.reward, scenario.stats())

    def test_missing_norm_rejected(self, scenario: ScorerScenario) -> None:
        """JSON 缺 norm 键 = MONAI 默认 BATCH：同样拒绝，不静默放行。"""
        artifact = scenario.artifact()
        del artifact.config["norm"]
        with pytest.raises(ValueError, match="GroupNorm"):
            RewardScorer(artifact, scenario.config.reward, scenario.stats())

    def test_multi_scale_config_rejected_until_ablation_ticket(
        self, scenario: ScorerScenario,
    ) -> None:
        """disc_num_scales>1 是假的消融轴（装配恒单尺度）：显式拒绝而非静默退化。"""
        reward_config = scenario.config.reward.model_copy(
            update={"disc_num_scales": 2},
        )
        with pytest.raises(ValueError, match="多尺度"):
            RewardScorer(scenario.artifact(), reward_config, scenario.stats())
