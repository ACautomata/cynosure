"""上游训练 recipe 的 BraTS 读图预处理链（data-preparation spec + ADR-0006）。

六步链是 NV-Generate-CTMR fork ``create_training_data.py`` 训练数据创建链的
语义重写——零依赖原则：上游代码只读参照、永不 import，对齐的是数据处理语义：

1. ``LoadImage``（NIfTI 读取）
2. ``EnsureChannelFirst``（通道在前）
3. ``Orientation(axcodes="RAS")``（方向重定向；BraTS 原生 LPS 靠翻转达成）
4. ``EnsureType(dtype=float32)``
5. ``ScaleIntensityRangePercentiles(0–99.5 → [0,1], clip=True)``——**clip=True
   是 fork 对 MONAI 上游 MAISI 默认（clip=False）的有意 recipe 偏差**
   （fork issue #251），照抄 fork 而非 MONAI，两版 embedding 不可互用；
6. ``Resize(mode="trilinear")`` 到每轴 128 的倍数——目标尺寸按 ``resize_target``
   的 dim 公式从 **RAS 重定向后**的空间形状读取（fork issue #312：从 NIfTI
   header 存储轴读取对轴置换方向会错；BraTS flip-only 幸免）。

链中明确不存在的步骤同样是对齐结论（ADR-0006）：spacing 重采样、foreground
crop、z-score 归一化——补上任何一步都等于换基座训练分布，须先重论证。

末端保留 ``MetaTensor``（affine 随 RAS 重定向更新），供方向断言与
spacing 侧车（``SpacingSidecar``，issue #46）读取；工件落盘前由管线
剥离元数据。
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    Resize,
    ScaleIntensityRangePercentiles,
)
from monai.utils import MetaKeys

from cynosure.config import UPSTREAM_RESIZE_BASE

SPACING_CONDITION_SCALE: float = 1e2
"""header zooms（mm）→ spacing 条件张量单位的换算因子（×1e2）：policy-modeling
章「体素间距 ×1e2」与 data-preparation spec「spacing 侧车」的单一来源。"""


class UpstreamPreprocessChain:
    """NIfTI 路径 → 上游 recipe 预处理影像体（``[1, D, H, W]`` float32）。

    resize 基数可注入：生产用上游基数（``UPSTREAM_RESIZE_BASE``，权威定义在
    config）；fixture config 注入小基数使夹具影像尺寸不变——fixture 与生产
    走同一条链逻辑、参数各自独立（fixture 不是对齐对象，其基数不取上游值）。
    """

    def __init__(self, resize_base: int = UPSTREAM_RESIZE_BASE) -> None:
        self._resize_base = resize_base
        self._steps = Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            EnsureType(dtype=torch.float32),
            ScaleIntensityRangePercentiles(
                lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True,
            ),
        ])

    @staticmethod
    def resize_target(
        spatial_shape: Iterable[int], resize_base: int,
    ) -> tuple[int, ...]:
        """dim 公式：每轴 ``max(round(size / base), 1) * base``（round 到最近
        倍数；不足半基数的轴受下界保护不塌到 0）。"""
        return tuple(
            max(round(size / resize_base), 1) * resize_base
            for size in spatial_shape
        )

    def __call__(self, series_path: Path) -> MetaTensor:
        image = self._steps(series_path)
        # resize 目标在运行时从 RAS 重定向后的形状计算（第 6 步依赖第 3 步结果）
        target = self.resize_target(tuple(image.shape[1:]), self._resize_base)
        return Resize(spatial_size=target, mode="trilinear")(image)


class SpacingSidecar:
    """per-case spacing 侧车读取器（data-preparation spec「spacing 侧车」+ issue #46）。

    从链末端 ``MetaTensor`` 的 meta 读 **raw NIfTI header zooms**（``LoadImage``
    原样保留的 ``original_pixdim``，nibabel ``get_zooms()[:3]`` 同值）——RAS
    重定向只改 affine、不动 raw zooms（BraTS flip-only 下顺序也不变），与上游
    「sidecar raw zooms、loader 端 ×1e2」的语义一致。"""

    def read(self, image: MetaTensor) -> tuple[float, float, float]:
        """单序列的 per-case spacing（manifest 条目值，×1e2 条件单位）。

        ``pixdim[1:4]`` 即三个存储轴的 zooms；meta 缺 raw zooms（非 NIfTI
        输入、读图契约破坏）显式失败。"""
        if MetaKeys.ORIGINAL_PIXDIM not in image.meta:
            raise ValueError(
                "链末端 meta 缺少 raw header zooms（original_pixdim）："
                "spacing 侧车要求 NIfTI 读图链（dataset 契约）",
            )
        zooms = np.asarray(image.meta[MetaKeys.ORIGINAL_PIXDIM], dtype=float)[1:4]
        i, j, k = (float(zoom) for zoom in zooms)
        return (
            i * SPACING_CONDITION_SCALE,
            j * SPACING_CONDITION_SCALE,
            k * SPACING_CONDITION_SCALE,
        )
