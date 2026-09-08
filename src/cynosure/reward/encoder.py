"""影像体 → VAE 预编码 latent 的策略接口与 fixture 合成实现。

生产预编码 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt`` + ``vae_config_json``
工件对经 netbuild 装载，latent [4,64,64,32] = [256,256,128] 影像体的
4× 空间压缩）；fixture 合成策略让 prepare 全循环在本地 CPU 跑，与生产
同一管线。
"""

import math
from typing import Protocol

import torch

from cynosure.netbuild import NetworkArtifact, NetworkAssembler


class LatentEncoder(Protocol):
    """影像体 → latent 的预编码策略（Strategy 接口，prepare 管线依赖此抽象）。"""

    name: str
    """预编码来源标识，随 manifest 落盘（provenance，消费方可区分出处）。"""

    def encode(
        self, image: torch.Tensor, noise_seed: int = 0,
    ) -> torch.Tensor:
        """[1, D, H, W] 影像体 → [4, D/4, H/4, W/4] latent（4× 空间压缩）。

        ``noise_seed``：后验采样噪声的内容寻址种子（生产语义，见
        ``MaisiLatentEncoder``）；合成实现忽略（自身无随机）。"""
        ...


class SyntheticLatentEncoder:
    """fixture 合成预编码：4×4×4 均值池化 + 固定通道权重。

    确定性（无随机、无网络权重）是 prepare 幂等的前提；固定通道权重使
    per-channel 统计量通道间可区分。空间映射与生产 VAE 的 4× 压缩同语义。
    """

    name = "synthetic"

    CHANNEL_WEIGHTS: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25)

    def encode(
        self, image: torch.Tensor, noise_seed: int = 0,
    ) -> torch.Tensor:
        """合成预编码（确定性，无随机；``noise_seed`` 忽略）。"""
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


class MaisiLatentEncoder:
    """生产预编码器：AutoencoderKlMaisi 工件装载、eval 相、no_grad 前向
    （风格镜像 ``eval/decode.py::LatentDecoder`` 的生产解码器）。

    存储域：manifest 按存储契约存 **encode 原始输出（未乘 scale_factor**，
    data-preparation「latent 存储域」）——缩放语义归 policy/decode 侧
    （policy 装载点乘入、``LatentDecoder`` 解码前除回）。

    编码存 **seeded 后验采样 z**：``z = z_mu + eps(noise_seed)·z_sigma``
    （上游 create_training_data 同语义，``encode_stage_2_inputs`` 的
    确定性重写）。T12 集群探针定谳：policy rollout 终点在 checkpoint
    scaled 采样域（std≈0.94），raw z_mu（std≈0.48、后验噪声是均值的
    3 倍量级）与它是**分布级错配**——判别器 real/fake 比较要求两侧
    同为后验采样分布；噪声种子按（schedule seed, 病例, 序列）内容寻址
    派生（CPU generator，跨设备确定），RNG 层的重跑零漂移幂等契约
    不因此破缺——逐位复现另依赖同设备/同 torch 版本（fp16 卷积
    autotune 的算法选择）。存储域仍按 data-preparation 契约为 encode
    原始输出（未乘 scale_factor）——policy 域归一在 reward/decode 消费点。

    数值口径：CUDA 上 fp16 autocast（上游 create_training_data 同款；
    上游 config ``norm_float16=true`` 使纯 fp32 前向 dtype mismatch
    崩溃——autocast 是结构性必需，非省显存优化），CPU 纯 fp32（fixture
    与本地测试路径）。

    编排：**恒整前向**——单样本元素数 ≤ roi 元素数才编码（上游
    ``dynamic_infer`` 小体豁免同语义，``roi_size`` 即豁免阈值）。
    BraTS [1,1,256,256,128] = 8.39M ≤ 影像空间阈值 [320,320,160] 的
    16.38M，生产全语料恒整前向（VAE 无注意力层、fp16 峰值 GB 级）。
    超过阈值的体积显式拒绝而非滑窗：MONAI ``SlidingWindowInferer`` 的
    多分辨率拼合是为上采样分割网络设计的，对下采样 encoder 实测把
    通道维折进空间维（[4,16,16,8] 产出 [1,16,16,8]）——静默产出语义
    错误的 latent 比显式失败危险得多；真出现超大体的需求时先裁剪，
    或以「输出网格直接加权拼合」实现 encoder 专用滑窗（随施工验证）。
    """

    name = "autoencoderkl_maisi"

    def __init__(
        self,
        artifact: NetworkArtifact,
        device: torch.device,
        roi_size: tuple[int, int, int] = (320, 320, 160),
    ) -> None:
        if any(dimension <= 0 for dimension in roi_size):
            raise ValueError(
                f"整前向豁免阈值 roi 须逐维为正（影像空间），得到 {roi_size}"
            )
        self._vae = NetworkAssembler.vae(artifact).to(device)
        self._device = device
        self._roi_size = roi_size
        self._vae.eval()  # 预编码是推断前向：恒 eval 相，不推进任何训练语义

    def encode(
        self, image: torch.Tensor, noise_seed: int = 0,
    ) -> torch.Tensor:
        """[1, D, H, W] 影像体 → [4, D/4, H/4, W/4] latent（CPU fp32，
        seeded 后验采样 z，不乘 scale_factor）。"""
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"影像体须为 [1, D, H, W]，得到 {tuple(image.shape)}"
            )
        batch = image.unsqueeze(0).to(self._device)
        if batch[0].numel() > math.prod(self._roi_size):
            raise ValueError(
                f"影像体元素数 {batch[0].numel()} 超过整前向豁免阈值 "
                f"{math.prod(self._roi_size)}（roi={self._roi_size}）："
                "encoder 滑窗未交付（MONAI SlidingWindowInferer 对下采样"
                "网络拼合通道错乱），超大体积请先裁剪"
            )
        with torch.no_grad():
            z_mu, z_sigma = self._encode_batch(batch)
            eps = torch.randn(
                z_mu.shape,
                generator=torch.Generator().manual_seed(noise_seed),
            ).to(z_mu.device, z_mu.dtype)
            latent = z_mu + eps * z_sigma
        return latent.squeeze(0).to(device="cpu", dtype=torch.float32)

    def _encode_batch(
        self, batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单批编码前向：encode 输出元组（z_mu, z_sigma）。"""
        if self._device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                z_mu, z_sigma = self._vae.encode(batch)
        else:
            z_mu, z_sigma = self._vae.encode(batch)
        return z_mu, z_sigma
