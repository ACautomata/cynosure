"""latent → 像素体的 VAE 解码（eval 模块职责；解码只发生在里程碑评测
与 Baseline/重采路径，不进逐 iteration 训练循环——ADR-0004）。

生产 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt`` + ``vae_config_json``
工件对经 netbuild 装载）；fixture/测试可注入实现同一 ``VolumeDecoder``
接口的替身（如计数解码器，结构断言的运行时观测面）。
"""

from typing import Protocol

import torch

from cynosure.netbuild import NetworkArtifact, NetworkAssembler


class VolumeDecoder(Protocol):
    """VAE 解码策略接口（里程碑评测与 Baseline/重采采样共同依赖）。"""

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latent 批 [B, C, D, H, W] → 像素体批 [B, 1, X, Y, Z]。"""
        ...


class LatentDecoder:
    """生产解码器：AutoencoderKlMaisi 工件装载、eval 相、no_grad 前向。"""

    def __init__(self, artifact: NetworkArtifact, device: torch.device) -> None:
        self._vae = NetworkAssembler.vae(artifact).to(device)
        self._vae.eval()  # 解码是评测前向：恒 eval 相，不推进任何训练语义

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """latent 批 → 像素体批 [B, 1, X, Y, Z]（fp32 口径，指标质量优先）。"""
        with torch.no_grad():
            decoded = self._vae.decode(latents)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
        if decoded.dim() != 5 or decoded.shape[1] != 1:
            raise ValueError(
                f"解码输出须为 [B, 1, X, Y, Z]，得到 {tuple(decoded.shape)}"
                "（里程碑评测面向单通道影像体）"
            )
        return decoded
