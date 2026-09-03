"""影像体 → VAE 预编码 latent 的策略接口与 fixture 合成实现。

生产预编码 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt``，latent [4,64,64,32] =
[256,256,128] 影像体的 4× 空间压缩）；其 checkpoint 语义（scale_factor 等）
待基座 checkpoint 落地后的 ticket 校准交付。本 ticket（#18）交付接口与
fixture 合成策略：fixture 让 prepare 全循环在本地 CPU 跑，与生产同一管线。
"""

from typing import Protocol

import torch


class LatentEncoder(Protocol):
    """影像体 → latent 的预编码策略（Strategy 接口，prepare 管线依赖此抽象）。"""

    name: str
    """预编码来源标识，随 manifest 落盘（provenance，消费方可区分出处）。"""

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """[1, D, H, W] 影像体 → [4, D/4, H/4, W/4] latent（4× 空间压缩）。"""
        ...


class SyntheticLatentEncoder:
    """fixture 合成预编码：4×4×4 均值池化 + 固定通道权重。

    确定性（无随机、无网络权重）是 prepare 幂等的前提；固定通道权重使
    per-channel 统计量通道间可区分。空间映射与生产 VAE 的 4× 压缩同语义。
    """

    name = "synthetic"

    CHANNEL_WEIGHTS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"影像体须为 [1, D, H, W]，得到 {tuple(image.shape)}"
            )
        channels, depth, height, width = image.shape
        if depth % 4 or height % 4 or width % 4:
            raise ValueError(
                f"影像体空间维须被 4 整除（4× 空间压缩），得到 {tuple(image.shape)}"
            )
        pooled = image.view(
            channels, depth // 4, 4, height // 4, 4, width // 4, 4,
        ).mean(dim=(2, 4, 6))  # [1, D/4, H/4, W/4]
        weights = torch.tensor(
            self.CHANNEL_WEIGHTS, dtype=pooled.dtype,
        ).view(4, 1, 1, 1)
        return pooled * weights
