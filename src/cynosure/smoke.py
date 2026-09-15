"""基座 checkpoint 装载与前向 smoke（wayfinder #120；零依赖只读转写）。

装载的是上游发布件本身——HF ``nvidia/NV-Generate-MR-Brain`` 的
``models/diff_unet_3d_rflow-mr-brain_v1.pt`` + ``models/autoencoder_v1.pt``
——网络架构参数从上游只读转写为网络配置 JSON（``configs/mrrate-base/``，
来源留档见该目录 README），上游代码只读参照、永不 import。

自检的四步读数（``BaseSmokeRunner.run`` 的落盘报告）：

1. **装载**：UNet/VAE 经 ``netbuild`` 装配面严格装载（上游 UNet 权重是
   训练 checkpoint **容器**形态，容器解包与 MetaTensor 白名单登记在
   ``netbuild`` 装载面）；模型参数量与权重文件张量总量对账；
2. **定点前向**：定点 latent（seed 派生）+ 定点模态 token + 定点
   timestep 走组1 采样场（``CfgCombinedField``，CFG=10 组合场，ADR-0002）
   ——两次前向逐位一致（确定性 kernel 口径），指纹（sha256 + 标量统计）
   落报告供跨运行比对；
3. **VAE 往返**：定点影像体 encode → 生产 latent 尺寸 → decode 回像素域
   ——encode 走 ``MaisiLatentEncoder``（生产预编码口径）、decode 走
   ``LatentDecoder``（#98 冻结的 fp16 autocast 口径 + 官方滑窗编排）；
4. **转写完整性**：网络配置 JSON 的键必须逐键被 MONAI 构造器消费（装载面
   对未知键静默过滤——转写誊抄错误会让架构参数静默偏离发布件）。

基准网格 ``[4,64,64,32]``（= 影像体 ``[1,256,256,128]`` ÷4）；多网格承接
由 #111 网格裁决后另行表达，不在本票。

显存画像（DCU 实测，sugon torch-dcu 2.9.0）：装载 + 定点前向 ~2 GiB、
生产尺寸 encode ~5 GiB；**decode 在确定性模式（CLI 入口的
``use_deterministic_algorithms``）下峰值 ~46 GiB**——确定性 trilinear
上采样走 torch 分解实现，是非确定性模式（~8 GiB）的 5 倍以上。同卡
共享跑会 OOM，解码读数要整卡独占；这条对全部 decode 消费点
（里程碑评测 / Baseline 采样）同样成立。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cynosure.config import CFG_MODAL_LABEL, SpecField
from cynosure.eval.decode import LatentDecoder
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import RolloutCondition
from cynosure.policy.diagnostic import LatentFingerprint
from cynosure.policy.field import CfgCombinedField
from cynosure.reward.encoder import MaisiLatentEncoder

_DEVICE_CHOICES = Literal["cuda", "cpu", "mps"]
"""自检设备白名单（同 fid 仪器先例：torch-dcu 上 ``cuda`` 为别名）。"""

_SCALE_FACTOR_SOURCE = Literal["checkpoint", "config"]
"""解码域缩放因子的来源标注：``checkpoint``（容器元数据，生产口径）或
``config``（显式给定，裸 state_dict 形态工件）。"""

_LATENT_SPATIAL_COMPRESSION = 4
"""VAE 的 4× 空间压缩（latent 空间维 = 影像体空间维 ÷4）。"""

_SPACING_CONDITION_UNIT = 100.0
"""体素间距的条件单位换算（policy-modeling 章：spacing ×1e2 恒传，
基座 ``include_spacing_input=true``）。"""


class BaseSmokeConfig(BaseModel):
    """基座装载自检的单次执行配置：独立 schema（同 fid 仪器先例），与训练
    config 完全分离——装载层自检不需要 real 池 / 判别器 / 数据链工件。

    schema 层不查文件存在性（仓库惯例：存在性属运行时输入契约）。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    unet_ckpt: Path = SpecField(
        "运行时", "wayfinder #120",
        "基座 UNet checkpoint（HF nvidia/NV-Generate-MR-Brain 的 "
        "models/diff_unet_3d_rflow-mr-brain_v1.pt；上游训练容器形态，"
        "装载面按容器键解包）",
    )
    unet_config_json: Path = SpecField(
        "运行时", "configs/mrrate-base",
        "基座 UNet 网络配置 JSON（上游架构参数的只读转写，键 = MONAI "
        "DiffusionModelUNetMaisi 构造参数名）",
    )
    vae_ckpt: Path = SpecField(
        "运行时", "wayfinder #120",
        "基座 VAE checkpoint（autoencoder_v1.pt，AutoencoderKlMaisi；"
        "上游发布件为裸 state_dict 形态）",
    )
    vae_config_json: Path = SpecField(
        "运行时", "configs/mrrate-base",
        "基座 VAE 网络配置 JSON（上游架构参数的只读转写，键 = MONAI "
        "AutoencoderKlMaisi 构造参数名）",
    )
    output_json: Path = SpecField(
        "运行时", "wayfinder #120",
        "自检报告落盘路径（读数跨运行比对与 #121/#122 的装载依据）",
    )
    latent_shape: tuple[int, int, int, int] = SpecField(
        "运行时", "policy-modeling",
        "定点 latent 形状 [C, D, H, W]（缺省 = 生产基准网格 [4,64,64,32]，"
        "对应影像体 [1,256,256,128]；多网格承接见 #111 网格裁决）",
        default=(4, 64, 64, 32),
    )
    modality_token: int = SpecField(
        "运行时", "上游 modality_mapping",
        "定点模态 token（MR-RATE 权威映射：t1w/t2w/flair/swi/mra → "
        "9/10/11/20/16，缺省 9 = t1w）；token 词表的 config 化由 #119 承接，"
        "本字段是装载自检的定点输入",
        default=9, ge=0,
    )
    spacing: tuple[float, float, float] = SpecField(
        "运行时", "上游 config_maisi_diff_model_rflow-mr-brain",
        "定点体素间距（mm，基座推理 config 的 spacing [0.94,0.94,1.36]；"
        "前向按基座口径 ×1e2 传入）",
        default=(0.94, 0.94, 1.36),
    )
    timestep: int = SpecField(
        "运行时", "policy-modeling",
        "定点 timestep（RFlow 日程 0..1000 域，1000 = 纯噪声；缺省 500 = "
        "日程中点）",
        default=500, ge=0, le=1000,
    )
    cfg_weight: float = SpecField(
        "运行时", "policy-modeling",
        "组1 采样场的 CFG 引导强度（缺省 = ADR-0002 定死的训练口径 10.0）",
        default=CFG_MODAL_LABEL, ge=0.0,
    )
    seed: int = SpecField(
        "运行时", "wayfinder #120",
        "定点输入派生 seed（latent / 影像体由它派生；encode 的 seeded 后验"
        "采样噪声同用）",
        default=0,
    )
    device: _DEVICE_CHOICES = SpecField(
        "运行时", "wayfinder #120",
        "自检设备（缺省平台检测：有 DCU/CUDA 用 0 号卡——生产 latent 尺寸的 "
        "编解码在 CPU 上无实际可用性；无加速设备的环境回退 CPU）",
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    )
    latent_scale_factor: float | None = SpecField(
        "运行时", "policy-modeling",
        "latent 域缩放因子（解码前除回 encoder 域）：缺省 None = 从基座 "
        "checkpoint 容器的 scale_factor 元数据读（生产单一来源，杜绝手抄"
        "漂移）；显式给出时以 config 为准（裸 state_dict 形态工件无元数据）",
        default=None, gt=0.0,
    )
    decode_roi_size: tuple[int, int, int] = SpecField(
        "运行时", "本票",
        "VAE 解码滑窗的 latent 空间 roi（官方 config_infer 口径 [48,48,48]；"
        "单通道空间体素数 ≤ roi 元素数走整前向豁免）",
        default=(48, 48, 48),
    )
    decode_overlap: float = SpecField(
        "运行时", "本票",
        "VAE 解码滑窗重叠比（官方字面 0.6666 = 2/3 的满精度取值）",
        default=2 / 3, ge=0.0, lt=1.0,
    )

    @field_validator("latent_shape")
    @classmethod
    def _latent_shape_is_encodable(
        cls, value: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if len(value) != 4 or any(dimension <= 0 for dimension in value):
            raise ValueError("latent_shape 必须是 4 个正整数（C, D, H, W）")
        if any(dimension % _LATENT_SPATIAL_COMPRESSION for dimension in value[1:]):
            raise ValueError(
                f"latent_shape 的空间维须被 {_LATENT_SPATIAL_COMPRESSION} 整除"
                f"（VAE 空间压缩），得到 {value[1:]}——否则 VAE 往返的"
                "像素域形状无定义"
            )
        return value

    @field_validator("spacing")
    @classmethod
    def _spacing_is_positive(
        cls, value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if len(value) != 3 or any(item <= 0.0 for item in value):
            raise ValueError(f"spacing 须为 3 个正数（mm），得到 {value}")
        return value

    @field_validator("decode_roi_size")
    @classmethod
    def _roi_is_positive(
        cls, value: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if len(value) != 3 or any(dimension <= 0 for dimension in value):
            raise ValueError(f"decode_roi_size 须为 3 个正整数，得到 {value}")
        return value

    @model_validator(mode="after")
    def _decode_window_fits_vaes_upsampling(self) -> "BaseSmokeConfig":
        """滑窗 overlap 与 roi 的整除约束（MONAI 要求 ``overlap×roi×
        zoom_scale`` 逐维为整数，VAE 4× 上采样下即 ``overlap×roi×4``）：
        不整除时 MONAI 在解码期才炸，错误信息与配置意图脱节。浮点比对
        留容差——官方字面 2/3 是四位截断，``2/3×48×4`` 在二进制浮点下
        是 127.999…，数学上整除。"""
        scaled = [
            self.decode_overlap * dimension * _LATENT_SPATIAL_COMPRESSION
            for dimension in self.decode_roi_size
        ]
        if any(abs(value - round(value)) > 1e-6 for value in scaled):
            raise ValueError(
                f"decode_overlap {self.decode_overlap} × roi "
                f"{self.decode_roi_size} × {_LATENT_SPATIAL_COMPRESSION} 须"
                f"逐维为整数（MONAI 滑窗缩放约束），得到 {scaled}"
            )
        return self

    @property
    def image_shape(self) -> tuple[int, int, int, int]:
        """定点影像体形状（latent 空间维 ×4 的 4× 空间压缩逆映射，
        单通道：``[1, 4D, 4H, 4W]``）。"""
        _, depth, height, width = self.latent_shape
        return (
            1,
            depth * _LATENT_SPATIAL_COMPRESSION,
            height * _LATENT_SPATIAL_COMPRESSION,
            width * _LATENT_SPATIAL_COMPRESSION,
        )


@dataclass(frozen=True)
class BaseSmokeReport:
    """基座装载自检读数（值对象，落 ``output_json``）。

    语义轴 = **同一定点输入**下的逐位复现：``loading.velocity_sha256``
    跨运行比对（同设备/同 torch 版本）；装载段参数量字段是装载完整性
    的对账面（模型侧 vs 权重文件侧）。VAE 侧只记权重文件张量总量——
    模型侧计数由装载面严格装载隐式保证（``LatentDecoder`` /
    ``MaisiLatentEncoder`` 各自持一份实例，不对外暴露）。
    """

    device: str
    seed: int
    loading: BaseLoadReadout
    """装载段读数（参数量对账 + 域缩放因子及来源）。"""
    latent_shape: tuple[int, ...]
    modality_token: int
    spacing: tuple[float, ...]
    timestep: int
    cfg_weight: float
    velocity_shape: tuple[int, ...]
    velocity_sha256: str
    velocity_mean: float
    velocity_std: float
    velocity_min: float
    velocity_max: float
    velocity_repeat_identical: bool
    """两次定点前向的逐位一致判定（确定性 kernel 口径的自检读数）。"""
    image_shape: tuple[int, ...]
    encoded_shape: tuple[int, ...]
    encoded_sha256: str
    decoded_shape: tuple[int, ...]
    decode_autocast_dtype: str
    """解码 autocast dtype 留痕（读自 ``LatentDecoder.AUTOCAST_DTYPE``
    单一来源）。"""
    decode_roi_size: tuple[int, ...]
    decode_overlap: float

    def write(self, path: Path) -> None:
        """报告落盘（父目录缺则建；UTF-8 JSON，字段序稳定）。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


@dataclass(frozen=True)
class BaseLoadReadout:
    """装载期读数（构造期 fail-fast 判定的载体，``BaseSmokeReport`` 的
    装载段）：参数量对账 + 域缩放因子及其来源。"""

    unet_parameters: int
    """装载出的 UNet 模型参数量。"""
    unet_checkpoint_parameters: int
    """基座 UNet 权重文件的张量总量（须与上一项相等——容器解包完整、
    架构转写自洽）。"""
    vae_checkpoint_parameters: int
    """基座 VAE 权重文件的张量总量（VAE 侧以文件为口径：编解码器各持
    一份实例、不对外暴露，严格装载隐式保证两侧一致）。"""
    latent_scale_factor: float
    latent_scale_factor_source: _SCALE_FACTOR_SOURCE
    """缩放因子来源：``checkpoint``（容器元数据，生产口径）或
    ``config``（显式给定，裸 state_dict 形态工件）。"""

    def __post_init__(self) -> None:
        if self.unet_parameters != self.unet_checkpoint_parameters:
            raise ValueError(
                f"基座 UNet 参数量对账不符：模型 {self.unet_parameters} vs "
                f"权重文件 {self.unet_checkpoint_parameters}（架构转写与"
                "发布件不是同一份网络）"
            )


class BaseSmokeRunner:
    """基座装载自检的执行器：装载（构造期，fail-fast）→ 定点前向指纹 →
    VAE 生产尺寸往返 → 报告落盘。

    校验分工：真实工件的形状/参数量错配都在**构造期**暴露（装载面严格
    装载 + 转写键完整性），运行期只产读数（``fid`` 仪器的装载期/执行期
    划分同款）。
    """

    def __init__(self, config: BaseSmokeConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        unet_config = self._transcribed_config(
            DiffusionModelUNetMaisi, config.unet_config_json,
        )
        vae_config = self._transcribed_config(
            AutoencoderKlMaisi, config.vae_config_json,
        )
        self._unet = NetworkAssembler.unet(
            NetworkArtifact(config=unet_config, checkpoint=config.unet_ckpt),
        ).to(self._device)
        # 容器只整读一次（装载面装配内部另有一次按路径的读——那是
        # 生产装配 seam 的 API 形态；本层把元数据与参数量对账合并到
        # 同一次读上）：缩放因子与权重张量总量都从这里出
        container = NetworkAssembler.read_checkpoint(config.unet_ckpt)
        scale_factor, scale_factor_source = self._resolve_scale_factor(
            NetworkAssembler.checkpoint_scale_factor(container),
        )
        self._readout = BaseLoadReadout(
            unet_parameters=sum(
                parameter.numel() for parameter in self._unet.parameters()
            ),
            unet_checkpoint_parameters=sum(
                tensor.numel()
                for tensor in NetworkAssembler.unwrap_container(
                    container,
                ).values()
            ),
            vae_checkpoint_parameters=self._checkpoint_parameters(
                config.vae_ckpt,
            ),
            latent_scale_factor=scale_factor,
            latent_scale_factor_source=scale_factor_source,
        )
        self._field = CfgCombinedField(self._unet, config.cfg_weight)
        vae_artifact = NetworkArtifact(
            config=vae_config, checkpoint=config.vae_ckpt,
        )
        self._encoder = MaisiLatentEncoder(vae_artifact, self._device)
        self._decoder = LatentDecoder(
            vae_artifact,
            self._device,
            scale_factor,
            config.decode_roi_size,
            config.decode_overlap,
        )

    def run(self) -> BaseSmokeReport:
        """跑完四步读数并落盘报告（返回同一份值对象）。"""
        velocity = self._velocity()
        if not bool(torch.isfinite(velocity).all()):
            raise ValueError(
                "基座定点前向输出含非有限值（NaN/Inf）：装载出的网络不可用"
            )
        repeat_identical = bool(torch.equal(velocity, self._velocity()))
        if not repeat_identical:
            raise ValueError(
                "基座定点前向输出逐位不可复现：同一定点 latent + 同一定点"
                "模态 token 的两次前向出现位级差异（确定性 kernel 口径未生效"
                "——CLI 入口的 CUBLAS_WORKSPACE_CONFIG / deterministic "
                "algorithms 是此契约的运行时前提）"
            )
        # 指纹读数在 CPU 上取（LatentFingerprint 的 numpy 序列化口径；
        # 张量本体留在设备，取哈希的搬运是一次性读数成本）
        fingerprint = LatentFingerprint(velocity.detach().cpu()).to_step_stats(
            step_index=0, timestep=float(self._config.timestep),
        )
        image = self._fixed_image()
        encoded = self._encoder.encode(image, noise_seed=self._config.seed)
        # 指纹在 CPU 原件上取（encode 契约 = CPU fp32），解码前搬运到
        # 设备——生产解码器的输入契约是设备驻留 latent（里程碑评测同款）
        encoded_sha256 = LatentFingerprint(encoded).to_step_stats(
            step_index=0, timestep=float(self._config.timestep),
        ).sha256
        decoded = self._decoder.decode(encoded.to(self._device).unsqueeze(0))
        report = BaseSmokeReport(
            device=str(self._device),
            seed=self._config.seed,
            loading=self._readout,
            latent_shape=tuple(self._config.latent_shape),
            modality_token=self._config.modality_token,
            spacing=tuple(self._config.spacing),
            timestep=self._config.timestep,
            cfg_weight=self._config.cfg_weight,
            velocity_shape=tuple(velocity.shape),
            velocity_sha256=fingerprint.sha256,
            velocity_mean=fingerprint.mean,
            velocity_std=fingerprint.std,
            velocity_min=fingerprint.min,
            velocity_max=fingerprint.max,
            velocity_repeat_identical=repeat_identical,
            image_shape=tuple(image.shape),
            encoded_shape=tuple(encoded.shape),
            encoded_sha256=encoded_sha256,
            decoded_shape=tuple(decoded.shape),
            decode_autocast_dtype=str(
                LatentDecoder.AUTOCAST_DTYPE,
            ).removeprefix("torch."),
            decode_roi_size=tuple(self._config.decode_roi_size),
            decode_overlap=self._config.decode_overlap,
        )
        report.write(self._config.output_json)
        return report

    def _velocity(self) -> torch.Tensor:
        """定点 latent + 定点条件（模态 token + 间距）经组1 采样场的前向
        velocity：``CfgCombinedField`` 的组合场（batch=2 [cond, uncond]、
        无条件分支全零 label，基座组织逐字复刻）。

        no_grad 内执行——装载自检只做前向读数，不建 autograd 图
        （采样场本身不带 no_grad：训练路径要梯度；自检若不关，批次=2 的
        前向图会跟着返回值整份驻留显存）。"""
        with torch.no_grad():
            return self._field.velocity(
                self._fixed_latent(),
                self._config.timestep,
                self._fixed_condition(),
            )

    def _fixed_latent(self) -> torch.Tensor:
        """定点 latent：seed 派生的标准正态，形状 [1, *latent_shape]。
        生成器留在 CPU 再搬运——跨设备同值（同 ``MaisiLatentEncoder``
        的后验采样噪声口径）。"""
        generator = torch.Generator().manual_seed(self._config.seed)
        return torch.randn(
            (1, *self._config.latent_shape), generator=generator,
        ).to(self._device)

    def _fixed_image(self) -> torch.Tensor:
        """定点影像体：[0,1] 均匀（上游 MR 强度臂归一域），形状
        ``image_shape``。"""
        generator = torch.Generator().manual_seed(self._config.seed)
        return torch.rand(
            self._config.image_shape, generator=generator,
        ).to(self._device)

    def _fixed_condition(self) -> RolloutCondition:
        """定点条件：单条 rollout 的目标模态 token + 体素间距（×1e2，
        基座 ``include_spacing_input=true`` 的恒传口径）。"""
        return RolloutCondition(
            label=torch.tensor(
                [self._config.modality_token], dtype=torch.int64,
                device=self._device,
            ),
            spacing=torch.tensor(
                [self._config.spacing], dtype=torch.float32,
                device=self._device,
            ) * _SPACING_CONDITION_UNIT,
        )

    def _resolve_scale_factor(
        self, checkpoint_value: float | None,
    ) -> tuple[float, _SCALE_FACTOR_SOURCE]:
        """解码域缩放因子：config 显式值优先，否则取容器元数据。

        两者皆缺席即显式拒绝——随手取 1.0 会让解码落在错误量级
        （读数看着「跑通」实则口径失真）。"""
        if self._config.latent_scale_factor is not None:
            return self._config.latent_scale_factor, "config"
        if checkpoint_value is None:
            raise ValueError(
                f"latent scale factor 缺席：config 未显式给出，且 "
                f"{self._config.unet_ckpt} 不是带 scale_factor 元数据的上游"
                "训练容器——解码域缩放无从确定（请在 config 显式给出）"
            )
        return checkpoint_value, "checkpoint"

    @staticmethod
    def _checkpoint_parameters(ckpt: Path) -> int:
        """权重文件的张量总量（容器形态经装载面解包后计数）。"""
        state = NetworkAssembler.read_state_dict(ckpt)
        return sum(tensor.numel() for tensor in state.values())

    @staticmethod
    def _transcribed_config(target: type, path: Path) -> dict:
        """读转写网络配置并校验键完整性：装载面对未知键静默过滤，
        转写誊抄错误（键名拼错/字段漏转）会让架构参数静默偏离发布件。"""
        config = NetworkAssembler.load_json(path)
        unconsumed = NetworkAssembler.unconsumed_keys(target, config)
        if unconsumed:
            raise ValueError(
                f"转写网络配置 {path} 含 {target.__name__} 构造器不接受的键 "
                f"{sorted(unconsumed)}：转写誊抄错误（装载面会静默丢弃这些"
                "键，架构参数随之偏离上游发布件）"
            )
        return config


__all__ = [
    "BaseLoadReadout",
    "BaseSmokeConfig",
    "BaseSmokeReport",
    "BaseSmokeRunner",
]
