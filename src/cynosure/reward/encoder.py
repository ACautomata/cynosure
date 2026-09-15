"""影像体 → VAE 预编码 latent 的策略接口与 fixture 合成实现。

生产预编码 = MONAI ``AutoencoderKlMaisi``（``vae_ckpt`` + ``vae_config_json``
工件对经 netbuild 装载，latent [4,64,64,32] = [256,256,128] 影像体的
4× 空间压缩）；fixture 合成策略让 prepare 全循环在本地 CPU 跑，与生产
同一管线。
"""

import math
from typing import Protocol

import torch
from monai.inferers.inferer import SlidingWindowInferer

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

    编排（上游 ``dynamic_infer`` 同语义分派，#143）：单样本空间体素数
    ≤ roi 元素数（影像空间阈值 [320,320,160]，prod=16.38M）恒整前向
    （BraTS [1,1,256,256,128]=8.39M ≤ 阈值，生产全语料不触发滑窗）；
    超出则 **滑窗分支（「b 语义」）**——MONAI 1.6 ``SlidingWindowInferer``
    （z_scale 路径，下采样网络原生支持）包 encoder 确定性前向，roi 逐轴
    clamp 到影像尺寸、gaussian、overlap 0.4、sw_batch_size 1（逐项锚
    NVIDIA create_training_data），逐窗 (z_mu, z_sigma) 在 latent 网格
    高斯加权拼合，**拼合后**以单一内容寻址种子采样一次 eps。T12 原裁决
    （MONAI 滑窗对下采样 encoder 通道折乱）经集群探针在 MONAI 1.6 上
    证伪（z_scale 路径逐格产出期望 latent 形状），超界显式拒绝随裁决
    改判移除（错误契约变更，#143）。

    对 NVIDIA 的**记录在案偏离**（a 语义 vs b 语义）：上游逐窗采样后
    拼接 z（接缝方差收缩最高 ~0.5×）；本仓拼 z_mu/z_sigma 后单次采样
    （接缝带方差与体心均匀）。偏离理由：重跑零漂移幂等契约（单次采样
    不引入窗口枚举序 RNG 依赖）+ real pool 分布卫生（接缝收缩方差会把
    artifact 喂进判别器的「真」侧）。滑窗分解对卷积感受野的截断差异
    （机制固有、非接缝集中）见 T12 复核探针报告。
    """

    name = "autoencoderkl_maisi"

    def __init__(
        self,
        artifact: NetworkArtifact,
        device: torch.device,
        roi_size: tuple[int, int, int] = (320, 320, 160),
        overlap: float = 0.4,
    ) -> None:
        if any(dimension <= 0 for dimension in roi_size):
            raise ValueError(
                f"滑动窗口 roi 须逐维为正（影像空间窗口），得到 {roi_size}"
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
        """[1, D, H, W] 影像体 → [4, D/4, H/4, W/4] latent（seeded 后验
        采样 z，不乘 scale_factor）。"""
        if image.dim() != 4 or image.shape[0] != 1:
            raise ValueError(
                f"影像体须为 [1, D, H, W]，得到 {tuple(image.shape)}"
            )
        batch = image.unsqueeze(0).to(self._device)
        with torch.no_grad():
            # 豁免判定与上游 dynamic_infer 逐字同构：单样本单通道空间
            # 体素数 ≤ prod(roi)（输入通道恒 1，与 batch[0].numel() 同值）
            if batch[0:1, 0:1].numel() <= math.prod(self._roi_size):
                z_mu, z_sigma = self._encode_batch(batch)
            else:
                z_mu, z_sigma = self._encode_windows(batch)
            # b 语义：eps 在拼合后的 z_mu 上采一次——同 seed 同输出、
            # 重跑零漂移，不随窗口枚举序分叉
            eps = torch.randn(
                z_mu.shape,
                generator=torch.Generator().manual_seed(noise_seed),
            ).to(z_mu.device, z_mu.dtype)
            latent = z_mu + eps * z_sigma
        return latent.squeeze(0).to(device="cpu", dtype=torch.float32)

    def _encode_windows(
        self, batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """滑窗分支（NVIDIA 语义）：roi 逐轴 clamp 到影像尺寸后
        SlidingWindowInferer 高斯加权分块——MONAI 1.6 z_scale 路径对下采样
        网络的首窗输出自动探测缩放比、逐输出成员（z_mu/z_sigma 各自）在
        latent 网格拼合（T12 复核探针验证：输出形状逐格等于期望 latent）。"""
        clamped = [
            min(roi, size)
            for roi, size in zip(self._roi_size, batch.shape[2:])
        ]
        z_mu, z_sigma = SlidingWindowInferer(
            roi_size=clamped,
            sw_batch_size=1,
            progress=False,
            mode="gaussian",
            overlap=self._overlap,
            sw_device=self._device,
            device=self._device,
        )(inputs=batch, network=self._encode_batch)
        return z_mu, z_sigma

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
