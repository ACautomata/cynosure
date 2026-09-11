"""模块骨架：从网络配置 JSON + checkpoint 文件构建并装载 MONAI 网络。

职责（spec #15 模块划分）：UNet / VAE / ControlNet / RFlowScheduler /
PatchDiscriminator 的按 artifact 构建与装载、per-channel 标准化统计量。
已交付 UNet / VAE / ControlNet / PatchDiscriminator / RFlowScheduler 装配面；
生产网络配置 JSON（NV-Generate 字段名 → MONAI 构造参数）的完整映射
由后续 ticket 在真实工件可得后深化。

判别器的 checkpoint 有**形态**之分（裸权重 / 谱归一化形态）：装载按形态
分派、导出面 ``loadable_state_dict`` 与之成对（形态契约与分派理由见
``discriminator``）。
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
    def discriminator(
        cls, artifact: NetworkArtifact, *, spectral_norm: bool = False,
    ) -> PatchDiscriminator:
        """按网络工件构建 PatchDiscriminator（GroupNorm 等 norm 参数随配置传入）。

        装载按 checkpoint 的**形态**分派（形态 = ``loadable_state_dict``
        的导出面）：

        - **谱归一化形态**（含 ``.parametrizations.`` 键：original 权重与
          幂迭代 buffer ``_u``/``_v`` 整份在内）→ 先叠加谱归一化、再严格
          装载：参数化状态**逐位还原**（同权重的前向余差只剩浮点执行
          路径的末位噪声），且装载不消费 RNG（重新叠加会以随机 u/v 起步
          做 15 次幂迭代、再归一化一次——上岗的判别函数随 ambient RNG
          漂移）。``spectral_norm=False`` 时显式拒绝：开关翻转会静默丢弃
          文件里的谱归一化状态，换 regime 须重新预训练；
        - **裸权重形态**（MAISI 发布权重 / fixture 网络工件）→ 先严格
          装载，``spectral_norm=True`` 时再叠加（谱归一化从这份权重起步
          的冷启动语义）。
        """
        model = PatchDiscriminator(
            **cls._known_kwargs(PatchDiscriminator, artifact.config),
        )
        state = cls._read_state_dict(artifact.checkpoint)
        if state is not None and cls._is_spectral_norm_state(state):
            if not spectral_norm:
                raise ValueError(
                    "判别器 checkpoint 是谱归一化形态（含 "
                    ".parametrizations. 键），但本装配未启用谱归一化："
                    "开关翻转会静默丢弃文件里的参数化状态（original 权重 "
                    "+ 幂迭代 u/v），上岗的判别函数将不是落盘的那一份——"
                    "换 regime 须重新预训练"
                )
            cls.apply_spectral_norm(model)
            model.load_state_dict(state, strict=True)
            return model
        if state is not None:
            model.load_state_dict(state, strict=True)
        if spectral_norm:
            cls.apply_spectral_norm(model)
        return model

    @classmethod
    def apply_spectral_norm(cls, model: Any) -> None:
        """谱归一化叠加（触发式装配动作）：对所有 Conv3d 施加谱归一化
        （Lipschitz 约束；判别器默认关闭、经 config 开关触发）。"""
        for module in model.modules():
            if isinstance(module, torch.nn.Conv3d):
                torch.nn.utils.parametrizations.spectral_norm(module)

    @classmethod
    def loadable_state_dict(cls, model: Any) -> dict:
        """模型 state_dict 的可装载形式（与装载面形态分派成对的导出面）。

        **参数化状态整份落盘**：parametrization 键
        ``<prefix>.parametrizations.<attr>.original``（原始权重）与幂迭代
        buffer ``_u``/``_v`` 同盘——装载面（``discriminator``）先叠形态、
        再严格装载，状态逐位还原，且装载不消费 RNG。
        物化有效权重的形态看似兼容（装载静默成功）实则**再归一化一次**
        （随机 u/v 起步的幂迭代估计）：消费面拿到的不再是保存的那一份
        判别函数，故不采用。无参数化模型逐键同一（``state_dict`` 直通）；
        续训分片（``ResumeStore``）存的也是这一形态，两处同形。
        """
        return model.state_dict()

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
    def _read_state_dict(cls, ckpt: Path | None) -> dict | None:
        """读 checkpoint 的 state_dict（None → 无 checkpoint = 随机初始化；
        weights_only 严格反序列化）。"""
        if ckpt is None:
            return None
        return torch.load(ckpt, map_location="cpu", weights_only=True)

    @classmethod
    def _is_spectral_norm_state(cls, state: dict) -> bool:
        """形态判定：含 parametrization 键 = 谱归一化形态
        （``loadable_state_dict`` 导出面的判别特征）。"""
        return any(".parametrizations." in key for key in state)

    @classmethod
    def _load_state_dict(cls, model: Any, ckpt: Path | None) -> None:
        state = cls._read_state_dict(ckpt)
        if state is not None:
            model.load_state_dict(state, strict=True)


__all__ = ["NetworkArtifact", "NetworkAssembler"]
