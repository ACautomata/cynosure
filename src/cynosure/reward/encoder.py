"""影像体 → VAE 预编码 latent 的策略接口与 fixture 合成实现。

生产预编码 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt`` + ``vae_config_json``
工件对经 netbuild 装载，latent [4,64,64,32] = [256,256,128] 影像体的
4× 空间压缩）；fixture 合成策略让 prepare 全循环在本地 CPU 跑，与生产
同一管线。
"""

import math
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import torch
from monai.inferers.inferer import SlidingWindowInferer

from cynosure.config import UPSTREAM_ENCODE_OVERLAP, UPSTREAM_ENCODE_ROI
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
        ``MaisiLatentEncoder``）；合成实现忽略（自身无随机）。生产
        调用点恒显式传种子（``PreparePipeline.noise_seed``）——缺省 0
        仅为无随机实现的签名兼容，生产路径漏传会让全语料静默共享
        一条 eps 流。"""
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

    编排（上游 ``dynamic_infer`` 判定式逐字同构，#143 交付滑窗分支）：
    单样本**单通道空间体素数**（``batch[0:1, 0:1]``）≤ roi 元素数走
    **整前向**（官方小体豁免；BraTS [1,1,256,256,128] = 8.39M ≤ 影像
    空间阈值 [320,320,160] 的 16.38M，BraTS 全语料恒整前向，行为与
    豁免口径不变）；超过 → roi 逐轴 clamp 到图像尺寸后
    ``SlidingWindowInferer`` 高斯滑窗（sw_batch_size=1，参数锚 NVIDIA
    ``diff_model_create_training_data``：roi [320,320,160] 影像空间、
    overlap 0.4）。T12 复核探针改判（#140）：原「SlidingWindowInferer
    对下采样 encoder 把通道维折进空间维」的裁决在 MONAI 1.6 的
    ``z_scale`` 路径（下采样网络输出网格原生拼合）上不复现，全部目标
    网格滑窗输出形状逐格正确。

    滑窗走 **b 语义**：逐窗 (z_mu, z_sigma) 经 MONAI 在 latent 网格
    高斯加权拼合后，与整前向同款以**单一内容寻址种子采样一次** eps。
    对上游「逐窗采样后拼 z」的记录在案偏离（上游 a 语义接缝方差收缩
    最高 0.5×；b 语义接缝方差与体心均匀）——偏离理由：幂等契约 +
    real pool 分布卫生；偏离的 ADR 落档随 #144。
    """

    name = "autoencoderkl_maisi"

    def __init__(
        self,
        artifact: NetworkArtifact,
        device: torch.device,
        roi_size: tuple[int, int, int] = UPSTREAM_ENCODE_ROI,
        overlap: float = UPSTREAM_ENCODE_OVERLAP,
    ) -> None:
        if any(dimension <= 0 for dimension in roi_size):
            raise ValueError(
                f"滑动窗口 roi 须逐维为正（影像空间豁免阈值），得到 {roi_size}"
            )
        if not 0.0 <= overlap < 1.0:
            raise ValueError(
                f"滑动窗口重叠比须在 [0, 1)，得到 {overlap}"
            )
        self._vae = NetworkAssembler.vae(artifact).to(device)
        self._device = device
        self._roi_size = roi_size
        self._overlap = overlap
        self._vae.eval()  # 预编码是推断前向：恒 eval 相，不推进任何训练语义

    def encode(
        self, image: torch.Tensor, noise_seed: int = 0,
    ) -> torch.Tensor:
        """[1, D, H, W] 影像体 → [4, D/4, H/4, W/4] latent（CPU fp32，
        seeded 后验采样 z，不乘 scale_factor；超界影像体走滑窗拼合，
        形状契约不变）。"""
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"影像体须为 [1, D, H, W]，得到 {tuple(image.shape)}"
            )
        batch = image.unsqueeze(0).to(self._device)
        with torch.no_grad():
            # 豁免判定单通道口径（batch[0:1, 0:1]）：与 LatentDecoder 及
            # 上游 dynamic_infer 判定式逐字同构；入口契约已锁单通道
            # （上方 shape[0] != 1 校验），两口径逐位等值
            if batch[0:1, 0:1].numel() <= math.prod(self._roi_size):
                z_mu, z_sigma = self._encode_batch(batch)
            else:
                z_mu, z_sigma = self._encode_sliding(batch)
            eps = torch.randn(
                z_mu.shape,
                generator=torch.Generator().manual_seed(noise_seed),
            ).to(z_mu.device, z_mu.dtype)
            latent = z_mu + eps * z_sigma
        return latent.squeeze(0).to(device="cpu", dtype=torch.float32)

    def _numerics(self) -> AbstractContextManager:
        """数值口径上下文（类 docstring「数值口径」段；整前向与滑窗两路
        共用同一口径，不得分叉）。"""
        if self._device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    def _encode_batch(
        self, batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """整前向编码：encode 输出元组（z_mu, z_sigma）。"""
        with self._numerics():
            return self._vae.encode(batch)

    def _encode_sliding(
        self, batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """滑窗编码（b 语义参数锚 NVIDIA）：roi 逐轴 clamp 后高斯滑窗，
        MONAI 1.6 z_scale 路径在 latent 网格逐成员拼合 (z_mu, z_sigma)。"""
        clamped = [
            min(roi, size) for roi, size in zip(self._roi_size, batch.shape[2:])
        ]
        inferer = SlidingWindowInferer(
            roi_size=clamped,
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=self._overlap,
            sw_device=self._device,
            device=self._device,
        )
        with self._numerics():
            z_mu, z_sigma = inferer(inputs=batch, network=self._vae.encode)
        return z_mu, z_sigma
