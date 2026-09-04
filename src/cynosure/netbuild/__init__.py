"""模块骨架：从网络配置 JSON + checkpoint 文件构建并装载 MONAI 网络。

职责（spec #15 模块划分）：UNet / VAE / ControlNet / RFlowScheduler /
PatchDiscriminator 的按 artifact 构建与装载、per-channel 标准化统计量。
已交付 UNet / VAE / ControlNet / PatchDiscriminator / RFlowScheduler 装配面；
生产网络配置 JSON（NV-Generate 字段名 → MONAI 构造参数）的完整映射
由后续 ticket 在真实工件可得后深化。
"""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import RFlowScheduler


@dataclass
class NetworkArtifact:
    """网络工件（零依赖原则的「唯一接口」）：网络配置 + checkpoint 文件对。"""

    config: dict
    checkpoint: Path | None = None


class NetworkAssembler:
    """按 artifact 装配可前向 MONAI 网络（零依赖原则：网络类全部来自 MONAI）。

    网络配置的键即 MONAI 构造参数名（fixture 生成器写出的就是这个约定）；
    非构造参数（如基座 config 字面 ``scale`` 死参数）静默过滤，不复刻其语义
    （ADR-0002：对齐实际生效行为，不照抄 config 字面值）。
    """

    @classmethod
    def load_json(cls, path: Path) -> dict:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    @classmethod
    def unet(cls, artifact: NetworkArtifact) -> DiffusionModelUNetMaisi:
        """按网络工件构建 UNet 并装载 checkpoint。"""
        model = DiffusionModelUNetMaisi(
            **cls._known_kwargs(DiffusionModelUNetMaisi, artifact.config),
        )
        cls._load_state_dict(model, artifact.checkpoint)
        return model

    @classmethod
    def vae(cls, artifact: NetworkArtifact) -> AutoencoderKlMaisi:
        """按网络工件构建图像 VAE（AutoencoderKlMaisi）并装载 checkpoint
        （生产预编码与里程碑评测解码的装载契约；latent 域 RL 循环本身不经它）。"""
        model = AutoencoderKlMaisi(
            **cls._known_kwargs(AutoencoderKlMaisi, artifact.config),
        )
        cls._load_state_dict(model, artifact.checkpoint)
        return model

    @classmethod
    def controlnet(cls, artifact: NetworkArtifact) -> ControlNetMaisi:
        """按网络工件构建 ControlNet（ControlNetMaisi）并装载 checkpoint
        （组2/组3 的 policy：残差每次前向注入 frozen base UNet，
        policy-modeling 章「实现接缝」第 5 条）。"""
        model = ControlNetMaisi(
            **cls._known_kwargs(ControlNetMaisi, artifact.config),
        )
        cls._load_state_dict(model, artifact.checkpoint)
        return model

    @classmethod
    def discriminator(cls, artifact: NetworkArtifact) -> PatchDiscriminator:
        """按网络工件构建 PatchDiscriminator（GroupNorm 等 norm 参数随配置传入）。"""
        model = PatchDiscriminator(
            **cls._known_kwargs(PatchDiscriminator, artifact.config),
        )
        cls._load_state_dict(model, artifact.checkpoint)
        return model

    @classmethod
    def loadable_state_dict(cls, model: Any) -> dict:
        """模型 state_dict 的可重载形式（与 ``_load_state_dict`` 的严格
        装载成对的导出面）：parametrization（如 spectral norm）的视图键
        ``<prefix>.parametrizations.<attr>.original`` 与其内部状态
        （power iteration buffer ``_u``/``_v``）不落盘，取而代之固化该
        参数化属性的**有效权重**（parametrization 输出，即训练前向实际
        使用的张量）到原键 ``<prefix>.<attr>``——本方法产出的 checkpoint
        经本类装配路径严格重载后，重建的裸网络判别函数与保存时逐位
        一致（固化原始参数则装载静默成功但前向漂移；无参数化模型逐键
        同一）。"""
        state = model.state_dict()
        if not any(".parametrizations." in key for key in state):
            return state
        loadable: dict = {}
        for key, value in state.items():
            prefix, marker, rest = key.partition(".parametrizations.")
            if not marker:
                loadable[key] = value
                continue
            attribute, _, tail = rest.partition(".")
            if tail == "original":
                parametrized = getattr(model.get_submodule(prefix), attribute)
                loadable[f"{prefix}.{attribute}"] = parametrized.detach()
        return loadable

    @classmethod
    def rflow_scheduler(
        cls, num_inference_steps: int, input_img_size_numel: int,
    ) -> RFlowScheduler:
        """装配 RFlowScheduler 并读入实际 timesteps。

        sigma 日程以 MONAI 实际输出为准（use_timestep_transform=true、
        实际 scale=1.0，均为基座行为定死，不设开关）。
        """
        scheduler = RFlowScheduler(use_timestep_transform=True)
        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=input_img_size_numel,
        )
        return scheduler

    @classmethod
    def _known_kwargs(cls, target: type, config: dict) -> dict:
        """过滤出 target 构造器接受的键（网络配置 JSON 的其余键静默忽略）。"""
        params = inspect.signature(target).parameters  # 类签名：等价 __init__ 且剔除 self
        return {key: value for key, value in config.items() if key in params}

    @classmethod
    def _load_state_dict(cls, model: Any, ckpt: Path | None) -> None:
        if ckpt is None:
            return
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)


__all__ = ["NetworkArtifact", "NetworkAssembler"]
