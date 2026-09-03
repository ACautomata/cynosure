"""模块骨架：从网络配置 JSON + checkpoint 文件构建并装载 MONAI 网络。

职责（spec #15 模块划分）：UNet / VAE / ControlNet / RFlowScheduler /
PatchDiscriminator 的按 artifact 构建与装载、per-channel 标准化统计量。
本 ticket（#16，prefactor）交付 UNet / PatchDiscriminator / RFlowScheduler
的最小装配面（fixture 与测试 seam 消费）；生产网络配置 JSON
（NV-Generate 字段名 → MONAI 构造参数）的完整映射与 AutoencoderKlMaisi /
ControlNetMaisi 装配由后续 ticket 在真实工件可得后深化。
"""

import inspect
import json
from pathlib import Path
from typing import Any

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import RFlowScheduler


class NetworkAssembler:
    """按 artifact 装配可前向 MONAI 网络（零依赖原则：网络类全部来自 MONAI）。

    网络配置 JSON 的键即 MONAI 构造参数名（fixture 生成器写出的就是这个约定）；
    非构造参数（如基座 config 字面 ``scale`` 死参数）静默过滤，不复刻其语义
    （ADR-0002：对齐实际生效行为，不照抄 config 字面值）。
    """

    @classmethod
    def load_json(cls, path: Path) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def unet(cls, net_config: dict, ckpt: Path | None = None) -> DiffusionModelUNetMaisi:
        """按网络配置构建 UNet 并装载 checkpoint。"""
        model = DiffusionModelUNetMaisi(
            **cls._known_kwargs(DiffusionModelUNetMaisi, net_config),
        )
        cls._load_state_dict(model, ckpt)
        return model

    @classmethod
    def discriminator(
        cls, disc_config: dict, ckpt: Path | None = None,
    ) -> PatchDiscriminator:
        """按配置构建 PatchDiscriminator（GroupNorm 等 norm 参数随 dict 传入）。"""
        model = PatchDiscriminator(**cls._known_kwargs(PatchDiscriminator, disc_config))
        cls._load_state_dict(model, ckpt)
        return model

    @classmethod
    def rflow_scheduler(
        cls,
        num_inference_steps: int,
        input_img_size_numel: int,
        use_timestep_transform: bool = True,
    ) -> RFlowScheduler:
        """装配 RFlowScheduler 并读入实际 timesteps（sigma 日程以 MONAI 实际
        输出为准，timestep transform 的实际 scale=1.0）。"""
        scheduler = RFlowScheduler(use_timestep_transform=use_timestep_transform)
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=input_img_size_numel,
        )
        return scheduler

    @classmethod
    def _known_kwargs(cls, target: type, config: dict) -> dict:
        """过滤出 target 构造器接受的键（网络配置 JSON 的其余键静默忽略）。"""
        params = inspect.signature(target.__init__).parameters
        return {key: value for key, value in config.items() if key in params}

    @classmethod
    def _load_state_dict(cls, model: Any, ckpt: Path | None) -> None:
        if ckpt is None:
            return
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)


__all__ = ["NetworkAssembler"]
