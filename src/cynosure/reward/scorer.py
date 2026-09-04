"""Reward model 打分核心（reward-model 章 + ADR-0001）：MONAI
PatchDiscriminator 封装。

- per-channel 标准化消费 prepare 的统计量工件（ChannelStats）；
- patch logit 图 ``[B,1,D',H',W']`` → raw real-logit 标量聚合（mean 为主 /
  min 消融），不过 sigmoid；tanh 压 (−1,1) 是触发式保险；
- LSGAN 判别器损失 = mean((D(real) − 1)²) + mean(D(fake)²)；
- SpectralNorm 默认关闭、经 config 触发启用（叠在 conv 上，Lipschitz 约束）；
- 数值锚：网络配置 JSON 的 norm（GroupNorm 定死）/ num_layers_d /
  in_channels 与 config 契约一致（静默错位即拒绝）；
- RewardScorer / ChannelNormalizer 是 nn.Module 聚合（统计量为 buffer）：
  device 迁移由 ``.to()`` 沿模块树递归单点接管；
- 多尺度判别器（``disc_num_scales>1``）属消融矩阵施工，显式拒绝而非静默
  退化为单尺度。
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from monai.networks.nets import PatchDiscriminator

from cynosure.config import RewardConfig
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats


@dataclass(frozen=True)
class LsganTerms:
    """LSGAN 判别器损失的分解记录（总损失 + real/fake 两项）。"""

    total: torch.Tensor
    real_term: torch.Tensor
    fake_term: torch.Tensor


class LatentScorer(Protocol):
    """给 rollout 的 latent 打分的策略接口（glossary「Reward model」：
    给 rollout 的 latent 打标量分的在线判别器）。

    RewardScorer 是 MONAI PatchDiscriminator 实现；编排方（Online update、
    AUC 信号、后续 train 循环）依赖本接口而非具体类。
    """

    @property
    def discriminator(self) -> torch.nn.Module:
        """底层判别器网络（online update 取参数/优化器用）。"""
        ...

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        """[B,4,D,H,W] → patch logit 图 [B,1,D',H',W']（raw，不过 sigmoid）。"""
        ...

    def reward(self, latents: torch.Tensor) -> torch.Tensor:
        """patch logit 图聚合为标量 reward（[B]，raw real-logit）。"""
        ...

    def discriminator_terms(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor,
    ) -> LsganTerms:
        """LSGAN 判别器损失分解。"""
        ...


class ChannelNormalizer(torch.nn.Module):
    """per-channel 标准化（判别器输入前处理，消费 prepare 的统计量工件）。

    统计量来自 Real sample pool 所用训练集；判别器在线小 batch 更新、
    fake 分布逐 iter 漂移，输入先标准化使打分面稳定。mean/std 是持久
    统计量，注册为 buffer：device 随 ``nn.Module.to()`` 递归迁移
    （RewardScorer 聚合本模块后单点 ``.to()`` 接管，见 trainer 装配）。
    """

    def __init__(self, stats: ChannelStats) -> None:
        super().__init__()
        # 显式标注：mypy 无法从 register_buffer 收窄 __getattr__ 的返回类型
        self.mean: torch.Tensor
        self.std: torch.Tensor
        mean = torch.tensor(stats.mean, dtype=torch.float32)
        std = torch.tensor(stats.std, dtype=torch.float32)
        if torch.any(std == 0):
            raise ValueError(f"统计量 std 含 0（不可标准化）: {stats.std}")
        self.register_buffer("mean", mean.view(-1, 1, 1, 1))
        self.register_buffer("std", std.view(-1, 1, 1, 1))

    def normalize(self, latents: torch.Tensor) -> torch.Tensor:
        """[B,C,D,H,W] → (x − mean) / std（通道数与 device 须与统计量一致）。

        buffer 已随 ``scorer.to()`` 迁至判别器所在 device，输入须同源：
        device 不符 fail-fast（错位输入即装配契约违例，不再静默对齐）。
        """
        if latents.dim() != 5:
            raise ValueError(
                f"latent 批须为 [B,C,D,H,W]，得到 {tuple(latents.shape)}"
            )
        if latents.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"latent 通道数 {latents.shape[1]} 与统计量通道数 "
                f"{self.mean.shape[0]} 不符（VAE latent_channels 定死 4）"
            )
        if latents.device != self.mean.device:
            raise ValueError(
                f"latent device {latents.device} 与统计量 buffer device "
                f"{self.mean.device} 不符（统计量随 scorer.to() 迁移，"
                "输入须同源对齐）"
            )
        return (latents - self.mean) / self.std


class RewardScorer(torch.nn.Module):
    """Reward model 打分核心（Facade）：标准化 → PatchDiscriminator →
    patch logit 图 → raw real-logit 聚合，产出 GRPO 的 reward 标量。

    MONAI PatchDiscriminator 输出 patch logit 图（Pix2PixHD 式，
    forward 返回中间特征 list，末元素为判别输出）；聚合后的标量即 reward。

    聚合判别器与 normalizer 为子模块（属性赋值即注册）：device 迁移由
    torch 的 ``.to()`` 沿模块树递归接管——trainer 装配只调一次
    ``scorer.to(device)``（判别器参数 + 统计量 buffer 一并迁移）。
    """

    def __init__(
        self,
        artifact: NetworkArtifact,
        config: RewardConfig,
        stats: ChannelStats,
    ) -> None:
        super().__init__()
        self._check_network_contract(artifact.config, config)
        discriminator = NetworkAssembler.discriminator(artifact)
        if config.spectral_norm_enabled:
            self._apply_spectral_norm(discriminator)
        self._discriminator = discriminator
        self._normalizer = ChannelNormalizer(stats)
        self._aggregation = config.patch_aggregation
        self._tanh_bounding = config.reward_tanh_bounding

    @property
    def discriminator(self) -> PatchDiscriminator:
        """底层的 MONAI 判别器（Online update 取参数/优化器用）。"""
        return self._discriminator

    def patch_logits(self, latents: torch.Tensor) -> torch.Tensor:
        """[B,4,D,H,W] → patch logit 图 [B,1,D',H',W']（raw，不过 sigmoid）。"""
        normalized = self._normalizer.normalize(latents)
        return self._discriminator(normalized)[-1]

    def reward(self, latents: torch.Tensor) -> torch.Tensor:
        """patch logit 图聚合为标量 reward（raw real-logit，保留组内分辨率）。

        mean = 主口径（advantage 组内标准化 scale-invariant，要分辨率而非
        绝对有界性）；min = 消融维 C（对局部伪影更敏感）；tanh 触发时压 (−1,1)。
        """
        logits = self.patch_logits(latents)
        if self._aggregation == "mean":
            values = logits.mean(dim=(1, 2, 3, 4))
        else:  # config Literal 已限死 {"mean", "min"}
            values = logits.amin(dim=(1, 2, 3, 4))
        if self._tanh_bounding:
            return torch.tanh(values)
        return values

    def discriminator_terms(
        self, logits_real: torch.Tensor, logits_fake: torch.Tensor,
    ) -> LsganTerms:
        """LSGAN 判别器损失分解，逐 patch 元素（非饱和梯度）。

        real 项 = mean((D(real) − 1)²)；fake 项 = mean(D(fake)²)。
        """
        real_term = ((logits_real - 1.0) ** 2).mean()
        fake_term = (logits_fake ** 2).mean()
        return LsganTerms(real_term + fake_term, real_term, fake_term)

    @staticmethod
    def _apply_spectral_norm(discriminator: PatchDiscriminator) -> None:
        """SpectralNorm 叠加（触发式）：对所有 conv 层施加谱归一化。"""
        for module in discriminator.modules():
            if isinstance(module, torch.nn.Conv3d):
                torch.nn.utils.parametrizations.spectral_norm(module)

    @staticmethod
    def _check_network_contract(net_config: dict, config: RewardConfig) -> None:
        """数值锚：网络配置 JSON（键 = MONAI 构造参数名）与 config 契约一致。"""
        norm = net_config.get("norm")
        norm_name = norm[0] if isinstance(norm, (list, tuple)) else norm
        if norm_name is None or str(norm_name).upper() != "GROUP":
            raise ValueError(
                f"网络配置 JSON norm={norm!r}，判别器归一化定死 GroupNorm"
                "（弃默认 BatchNorm：在线小 batch 不泄漏 batch 统计，ADR-0001）"
            )
        depth = net_config.get("num_layers_d")
        if depth != config.disc_num_layers_d:
            raise ValueError(
                f"网络配置 JSON num_layers_d={depth} 与 config "
                f"disc_num_layers_d={config.disc_num_layers_d} 不符"
                "（消融扫深度 = 换网络配置，静默错位不可接受）"
            )
        if net_config.get("in_channels") != 4:
            raise ValueError(
                f"网络配置 JSON in_channels={net_config.get('in_channels')}，"
                "判别器输入定死 4 通道（VAE latent_channels=4）"
            )
        if config.disc_num_scales != 1:
            raise ValueError(
                f"disc_num_scales={config.disc_num_scales}：多尺度判别器"
                "（MultiScalePatchDiscriminator，各尺度 mean 后跨尺度相加）"
                "属消融矩阵施工，未交付前显式拒绝而非静默退化为单尺度"
                "（reward-model 章消融矩阵节）"
            )
