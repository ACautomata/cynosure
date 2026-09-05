"""latent → 像素体的 VAE 解码（eval 模块职责；解码只发生在里程碑评测
与 Baseline/重采路径，不进逐 iteration 训练循环——ADR-0004）。

生产 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt`` + ``vae_config_json``
工件对经 netbuild 装载），解码走官方 NV-Generate-CTMR 的**滑动窗口
策略**（``SlidingWindowInferer`` 高斯聚合 + 小体豁免，latent 空间 roi；
生产大体积整前向解码是 OOM 级分配）；fixture/测试可注入实现同一
``VolumeDecoder`` 接口的替身（如计数解码器，结构断言的运行时观测面）。
"""

import math
from typing import Protocol

import torch
from monai.inferers.inferer import SlidingWindowInferer

from cynosure.netbuild import NetworkArtifact, NetworkAssembler


class VolumeDecoder(Protocol):
    """VAE 解码策略接口（里程碑评测与 Baseline/重采采样共同依赖）。"""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latent 批 [B, C, D, H, W] → 像素体批 [B, 1, X, Y, Z]。"""
        ...


class LatentDecoder:
    """生产解码器：AutoencoderKlMaisi 工件装载、eval 相、no_grad 前向。

    域语义：policy（官方基座权重）工作在 **scaled** latent 域
    （checkpoint scale_factor = 1/std(z)），prepared latents 按存储契约
    是 encode 原始输出（未乘，data-preparation「latent 存储域」）——
    解码前除回 scale factor 把 policy 域 latent 归位到 VAE 期望的
    encoder 域（官方 ``ReconModel.decode_stage_2_outputs(z/scale)``
    同一语义）。生产因子随基座 checkpoint 核对（config
    ``policy.latent_scale_factor``）；fixture 取中性 1.0（除法恒等）。

    解码编排（官方 NV-Generate-CTMR 同策略）：单样本元素数 ≤ roi 元素数
    时整前向（官方 ``dynamic_infer`` 小体豁免——fixture 夹具尺寸恒走
    此路）；否则 ``SlidingWindowInferer`` 按 latent 空间 roi 高斯加权
    分块解码（``sw_batch_size=1``、``mode="gaussian"``，官方口径）。
    """

    def __init__(
        self,
        artifact: NetworkArtifact,
        device: torch.device,
        latent_scale_factor: float,
        roi_size: tuple[int, int, int],
        overlap: float,
    ) -> None:
        if latent_scale_factor <= 0.0:
            raise ValueError(
                f"latent scale factor 须为正（域缩放除数），得到 "
                f"{latent_scale_factor}"
            )
        if any(dimension <= 0 for dimension in roi_size):
            raise ValueError(
                f"滑动窗口 roi 须逐维为正（latent 空间窗口），得到 {roi_size}"
            )
        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"滑动窗口重叠比须在 [0, 1)，得到 {overlap}"
            )
        self._vae = NetworkAssembler.vae(artifact).to(device)
        self._device = device
        self._latent_scale_factor = latent_scale_factor
        self._roi_size = roi_size
        self._overlap = overlap
        self._vae.eval()  # 解码是评测前向：恒 eval 相，不推进任何训练语义

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latent 批 → 像素体批 [B, 1, X, Y, Z]（fp32 口径，指标质量优先；
        输入先除 scale factor 归位 encoder 域——逐元素算子，整批除与
        官方逐 patch 除等价）。"""
        scaled = latents / self._latent_scale_factor
        with torch.no_grad():
            if scaled[0].numel() <= math.prod(self._roi_size):
                decoded = self._vae.decode(scaled)
            else:
                decoded = SlidingWindowInferer(
                    roi_size=list(self._roi_size),
                    sw_batch_size=1,
                    progress=False,
                    mode="gaussian",
                    overlap=self._overlap,
                    sw_device=self._device,
                    device=self._device,
                )(inputs=scaled, network=self._vae.decode)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() != 5 or decoded.shape[1] != 1:
            raise ValueError(
                f"解码输出须为 [B, 1, X, Y, Z]，得到 {tuple(decoded.shape)}"
                "（里程碑评测面向单通道影像体）"
            )
        return decoded
