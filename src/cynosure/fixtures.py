"""微型合成 fixture 生成器（spec「Fixture 策略」）。

fixture 让全循环在本地 CPU 跑，与生产走同一 netbuild 装载契约：

- MONAI 迷你 UNet（~0.1M 参数）+ 缩小 latent ``[4,16,16,8]`` + 3 步 ODE；
- **G 保持 12**——G 是组内方向数、与网络尺寸无关；G=2 会使组内标准化
  退化为恒 ±1、advantage clamp 永不触发，统计行为失真；
- |M| 按日程长度缩放：3 步日程取 ``{1}``，避开 s≈1 奇异端；
- ``input_img_size_numel`` 按 fixture latent 的同语义值传
  （``prod((16,16,8)) = 2048``，数值锚口径见 ADR-0002）；
- 判别器 ``num_layers_d=2`` 在 fixture 第三维（8）上空间不足（MONAI 的
  Pix2PixHD 式层叠在 8 体素上不足两次 kernel-4 卷积），fixture 取消融轴
  另一端 ``num_layers_d=1``——这是 spec「需确认」项的确认结果；
- VAE 不进 fixture 循环（解码只发生在里程碑评测），config 中相应工件为
  占位路径；modality mapping 作为真实输入工件落盘（诊断经它装载标签）。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.nets import PatchDiscriminator

from cynosure.config import CynosureConfig

# fixture 迷你 UNet 构造参数（键 = MONAI 构造参数名，netbuild 直接消费）
FIXTURE_UNET_CONFIG: dict = {
    "spatial_dims": 3,
    "in_channels": 4,
    "out_channels": 4,
    "num_res_blocks": [1, 1],
    "num_channels": [8, 16],
    "attention_levels": [False, False],
    "norm_num_groups": 8,
    "num_head_channels": 8,
    "with_conditioning": False,
    "num_class_embeds": 128,
    "include_spacing_input": True,
}

# fixture 判别器构造参数（GroupNorm、num_layers_d=1，见模块 docstring）
FIXTURE_DISCRIMINATOR_CONFIG: dict = {
    "spatial_dims": 3,
    "channels": 8,
    "in_channels": 4,
    "num_layers_d": 1,
    "norm": ["GROUP", {"num_groups": 8}],
}

# fixture ControlNet 构造参数（与 fixture UNet 同构：残差注入要求
# num_channels / num_res_blocks / attention_levels 逐级对齐；
# conditioning_embedding_in_channels = 源影像 latent 通道数 4，
# conditioning_embedding_num_channels=[8]（单层 stride-1 卷积）使条件嵌入与
# UNet conv_in 输出同形；use_checkpointing 关闭（fixture CPU 全循环无显存压力））
FIXTURE_CONTROLNET_CONFIG: dict = {
    "spatial_dims": 3,
    "in_channels": 4,
    "num_res_blocks": [1, 1],
    "num_channels": [8, 16],
    "attention_levels": [False, False],
    "norm_num_groups": 8,
    "num_head_channels": 8,
    "with_conditioning": False,
    "num_class_embeds": 128,
    "conditioning_embedding_in_channels": 4,
    "conditioning_embedding_num_channels": [8],
    "use_checkpointing": False,
}

FIXTURE_MODALITY_MAPPING: dict[str, int] = {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}
"""基座 modality_mapping 的文档值（config Artifacts.modality_mapping_json
同口径：t1n/t1c/t2w/t2f → 29/34/30/31），fixture 以工件形式落盘。"""

_ZERO_CONV_REINIT_STD = 0.02
"""全零卷积权重的重初始化尺度（见 Fixture.unet 的说明）。"""


@dataclass
class FixtureArtifacts:
    """fixture 网络工件的落盘路径（与生产 artifact 同构：ckpt + 网络 JSON）。"""

    unet_ckpt: Path
    unet_config_json: Path
    controlnet_ckpt: Path
    controlnet_config_json: Path
    discriminator_ckpt: Path
    discriminator_config_json: Path
    modality_mapping_json: Path


class Fixture:
    """微型合成 fixture：MONAI 迷你网络 + 缩小 latent 的合法全循环配置。"""

    LATENT_SHAPE: tuple[int, int, int, int] = (4, 16, 16, 8)
    NUM_INFERENCE_STEPS: int = 3
    TRAIN_STEP_INDICES_M: frozenset[int] = frozenset({1})
    GROUP_SIZE_G: int = 12
    INPUT_IMG_SIZE_NUMEL: int = 16 * 16 * 8  # = 2048，数值锚与 latent 同语义

    def unet(self) -> DiffusionModelUNetMaisi:
        """随机初始化的 MONAI 迷你 UNet（CPU、seed 由调用方固定以保证复现）。

        MONAI 扩散 UNet 把每个 resnet 末层卷积与输出卷积 zero-init（训练期
        恒等起步的惯例）——未训练的零卷积会把 resnet 的 temb 条件通道
        （label / timestep / spacing）数值归零，使 CFG 组合场在 fixture 上
        退化为常量场。fixture 消费的是采样机制数值而非训练初值，故对所有
        全零卷积权重重新随机初始化（小尺度正态、bias 保持零），条件敏感性
        真实化；网络类与确定性（seed 固定）不变。
        """
        unet = DiffusionModelUNetMaisi(**FIXTURE_UNET_CONFIG)
        for module in unet.modules():
            if isinstance(module, nn.Conv3d) and not module.weight.abs().any():
                nn.init.normal_(module.weight, std=_ZERO_CONV_REINIT_STD)
        return unet

    def discriminator(self) -> PatchDiscriminator:
        """随机初始化的 MONAI PatchDiscriminator（GroupNorm、fixture 深度）。"""
        config = dict(FIXTURE_DISCRIMINATOR_CONFIG)
        config["norm"] = tuple(config["norm"])
        return PatchDiscriminator(**config)

    def controlnet(self) -> ControlNetMaisi:
        """随机初始化的 MONAI ControlNetMaisi（与 fixture UNet 同构）。

        ControlNet 惯例零初始化（controlnet 块与条件嵌入 conv_out）：未训练
        时残差恒零、源影像条件不参与前向——与 unet 相同的全零卷积重初始化
        使条件敏感性真实化（网络类与确定性不变）。"""
        controlnet = ControlNetMaisi(**FIXTURE_CONTROLNET_CONFIG)
        for module in controlnet.modules():
            if isinstance(module, nn.Conv3d) and not module.weight.abs().any():
                nn.init.normal_(module.weight, std=_ZERO_CONV_REINIT_STD)
        return controlnet

    def write_artifacts(self, directory: Path) -> FixtureArtifacts:
        """把 fixture 网络写成 ckpt + 网络配置 JSON（netbuild 可直接装载）。"""
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = FixtureArtifacts(
            unet_ckpt=directory / "unet.pt",
            unet_config_json=directory / "unet_config.json",
            controlnet_ckpt=directory / "controlnet.pt",
            controlnet_config_json=directory / "controlnet_config.json",
            discriminator_ckpt=directory / "discriminator.pt",
            discriminator_config_json=directory / "discriminator_config.json",
            modality_mapping_json=directory / "modality_mapping.json",
        )
        torch.save(self.unet().state_dict(), artifacts.unet_ckpt)
        artifacts.unet_config_json.write_text(
            json.dumps(FIXTURE_UNET_CONFIG, indent=2), encoding="utf-8",
        )
        # 判别器先于 ControlNet 构造：write_artifacts 的权重消费调用方的
        # 环境 RNG 流，判别器保持历史流位（unet 之后第二个构造）——reward
        # fixture 的 held-out AUC 门槛测试对该初始化敏感，不因新增工件重掷
        torch.save(self.discriminator().state_dict(), artifacts.discriminator_ckpt)
        artifacts.discriminator_config_json.write_text(
            json.dumps(FIXTURE_DISCRIMINATOR_CONFIG, indent=2), encoding="utf-8",
        )
        torch.save(self.controlnet().state_dict(), artifacts.controlnet_ckpt)
        artifacts.controlnet_config_json.write_text(
            json.dumps(FIXTURE_CONTROLNET_CONFIG, indent=2), encoding="utf-8",
        )
        artifacts.modality_mapping_json.write_text(
            json.dumps(FIXTURE_MODALITY_MAPPING, indent=2), encoding="utf-8",
        )
        return artifacts

    def config(
        self, artifacts_dir: Path, group: str = "modal-label",
    ) -> CynosureConfig:
        """合法的缩小版全量 config（schema 全字段通过；CPU 全循环可跑）。

        ``group`` 选实验组（三选一）；组2/组3 的 config 携带 ControlNet
        工件对（schema 强制），组1 携带亦无害（modal-label 训练不消费）。"""
        return CynosureConfig.model_validate({
            "experiment": {"group": group},
            "latent_shape": list(self.LATENT_SHAPE),
            "fixture_mode": True,  # 缩小采样日程（3 步 ODE）的显式声明通道
            "artifacts": {
                "unet_ckpt": str(artifacts_dir / "unet.pt"),
                # VAE / modality mapping / 源数据集不进 fixture 循环，占位路径
                "vae_ckpt": str(artifacts_dir / "vae.pt"),
                "net_config_json": str(artifacts_dir / "unet_config.json"),
                "modality_mapping_json": str(artifacts_dir / "modality_mapping.json"),
                "dataset_root": str(artifacts_dir / "dataset"),
                # ControlNet 工件与 write_artifacts 同构：组2/组3 的 policy 装配源
                "controlnet_ckpt": str(artifacts_dir / "controlnet.pt"),
                "controlnet_config_json": str(artifacts_dir / "controlnet_config.json"),
                # 判别器工件与 write_artifacts 同构：fixture 打分/在线更新装载源
                "discriminator_config_json": str(artifacts_dir / "discriminator_config.json"),
                "discriminator_ckpt": str(artifacts_dir / "discriminator.pt"),
            },
            "policy": {
                "num_inference_steps": self.NUM_INFERENCE_STEPS,
                "input_img_size_numel": self.INPUT_IMG_SIZE_NUMEL,
                "group_size_g": self.GROUP_SIZE_G,
                "train_step_indices_m": sorted(self.TRAIN_STEP_INDICES_M),
            },
            "reward": {
                "disc_num_layers_d": 1,
                "disc_batch_size_k": 4,
                "replay_buffer_capacity": 64,
                "real_pool_manifest": str(artifacts_dir / "real_pool.json"),
                "heldout_real_manifest": str(artifacts_dir / "heldout_real.json"),
                "channel_stats_json": str(artifacts_dir / "channel_stats.json"),
            },
            "schedule": {"seed": 0},
        })
