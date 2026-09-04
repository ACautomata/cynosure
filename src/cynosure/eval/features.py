"""2D 切片特征提取策略（像素域 2.5D FID/KID 的特征空间来源）。

生产 = RadImageNet-ResNet50（MONAI resnet50 拓扑、2048 维池化特征，
RadImageNet 公开发布权重的下载属施工——权重文件经 config
``artifacts.radimagenet_weights`` 注入）；fixture/测试 = 确定性 stub
（ticket #25：RadImageNet 权重可下载或 stub 化注入）。两者实现同一
``SliceFeatureExtractor`` 接口，度量核不感知特征来源。
"""

from pathlib import Path
from typing import Protocol

import torch
from monai.networks.nets import ResNet as MonaiResNet
from monai.networks.nets import resnet50

_RADIMAGENET_SLICE_SIZE = 224
"""RadImageNet 预处理口径：切片缩放至 224×224 后进 ResNet50。"""

_RADIMAGENET_CHANNELS = 3
"""RadImageNet 以 3 通道训练：灰度切片复制成 3 通道（医学影像标准做法）。"""


class SliceFeatureExtractor(Protocol):
    """切片 → 特征向量的策略接口（2.5D FID/KID 的特征空间）。"""

    feature_dim: int
    """特征维数（契约：FID/KID 两侧必须同维）。"""

    def extract(self, slices: torch.Tensor) -> torch.Tensor:
        """[N, 1, H, W] 灰度切片批 → [N, feature_dim] 特征矩阵。"""
        ...


class StubSliceFeatureExtractor:
    """fixture 确定性特征器：固定权重的 3×3 卷积 + mean/std 池化。

    无随机、无预训练依赖——同一输入必得同一特征（fixture 确定性契约），
    让 FID/KID 全链路在本地 CPU 无网络权重可跑。"""

    FEATURE_DIM = 8
    feature_dim = FEATURE_DIM

    def extract(self, slices: torch.Tensor) -> torch.Tensor:
        if slices.dim() != 4 or slices.shape[1] != 1:
            raise ValueError(
                f"切片批须为 [N, 1, H, W]，得到 {tuple(slices.shape)}"
            )
        weight = torch.tensor([
            [[[0.0, 0.25, 0.0], [0.25, 1.0, 0.25], [0.0, 0.25, 0.0]]],
        ], dtype=slices.dtype)
        smoothed = torch.nn.functional.conv2d(slices, weight, padding=1)
        flattened = smoothed.reshape(smoothed.shape[0], -1)
        mean = flattened.mean(dim=1)
        std = flattened.std(dim=1, unbiased=False)
        moments = flattened.abs().mean(dim=1)
        columns = [
            mean, std, moments, mean * std,
            flattened.min(dim=1).values, flattened.max(dim=1).values,
            (flattened > 0).float().mean(dim=1), flattened.square().mean(dim=1),
        ]
        return torch.stack(columns, dim=1)


class RadImageNetFeatureExtractor:
    """生产特征器：MONAI ResNet50（2D、2048 维池化特征）+ RadImageNet 权重。

    权重文件须与 ``build_backbone`` 的 MONAI 拓扑同构（``state_dict`` 严格
    装载）——RadImageNet 公开发布为 torchvision 命名格式，键名重映射随
    施工 ticket 落地；缺文件 / 键不匹配都是显式失败，不静默随机权重。
    """

    def __init__(self, weights: Path | str) -> None:
        weights_path = Path(weights)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"RadImageNet-ResNet50 权重文件不存在: {weights_path}"
                "（config artifacts.radimagenet_weights；公开发布权重的"
                "下载脚本属施工）"
            )
        self._backbone = self.build_backbone()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        try:
            self._backbone.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"RadImageNet 权重与 MONAI resnet50 拓扑键名不匹配: "
                f"{weights_path}（公开发布为 torchvision 命名格式，键名"
                "重映射随施工 ticket 落地；不静默随机初始化）"
            ) from exc
        self._backbone.eval()
        self.feature_dim = self._backbone(torch.zeros(1, _RADIMAGENET_CHANNELS, 64, 64)).shape[1]

    @staticmethod
    def build_backbone() -> MonaiResNet:
        """RadImageNet 同拓扑的特征骨干（feed_forward=False = 池化特征出口）。"""
        return resnet50(
            pretrained=False,
            spatial_dims=2,
            n_input_channels=_RADIMAGENET_CHANNELS,
            feed_forward=False,
        )

    def extract(self, slices: torch.Tensor) -> torch.Tensor:
        if slices.dim() != 4 or slices.shape[1] != 1:
            raise ValueError(
                f"切片批须为 [N, 1, H, W]，得到 {tuple(slices.shape)}"
            )
        resized = torch.nn.functional.interpolate(
            slices, size=(_RADIMAGENET_SLICE_SIZE, _RADIMAGENET_SLICE_SIZE),
            mode="bilinear", align_corners=False,
        )
        channels = resized.repeat(1, _RADIMAGENET_CHANNELS, 1, 1)
        with torch.no_grad():
            return self._backbone(channels)
