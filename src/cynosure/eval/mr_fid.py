"""fork 口径 MR FID 仪器（wayfinder #79：fork ``scripts/compute_fid_2-5d_ct.py``
MR 分支的移植，6 处改造原样承接）。

口径权威（不得偏离，源自 #69 数值验证 + #73 双轨裁决）：

- 切片**原尺寸**直进 ResNet50（**不** resize 224——那是 ``features.py``
  里程碑设施口径；#73 双轨：本仪器只出裁决性读数，里程碑维持现状）；
- 变换链顺序 crop → 强度 → pad（fork 改造 #2：MR 百分位统计只看真实
  内容不被 padding 稀释，自洽性已数值验证，
  research/upstream-eval-protocol.md §4.3）；
- MR 强度臂 ``ScaleIntensityRangePercentilesd(0, 99.5 → 0, 1000,
  clip=False)``、padding_value=0（生成侧 (0,1000) 输出域，与
  utils_infer 落盘域同口径）；
- 归一链照抄：逐卷×逐平面全局 min-max（``radimagenet_intensity_
  normalisation`` 的 norm2d=False 4D 分支）→ 通道翻转 [2,1,0] →
  减 ImageNet 均值 [0.406, 0.456, 0.485]（翻转后序）；
- 三正交面切片轴语义照抄：XY=沿 D 轴（unbind dim=-1，得 H×W 面）、
  YZ=沿 H 轴（dim=2，得 W×D 面）、ZX=沿 W 轴（dim=3，得 H×D 面）——
  **不按平面名字推断轴**（fork 命名陷阱，upstream-eval-protocol §1.3）；
- 中心窗口 ``start = int((1−r)/2·N)`` / ``end = int((1+r)/2·N)``，
  ratio 在 config 钉死（冻结变量 #4：论文 0.5 vs docs 示例 0.4 不一致，
  必须逐次对比显式声明）；
- FID = 三面**算术平均**；``drop_empty`` 未启用（fork 主流程同，
  其 ``empty_threshold=-700`` 为 HU 语义）；squeezenet torchvision
  fallback 臂裁掉（import 白名单 + #79 依赖补齐裁定）；torchrun 分布式
  all-gather 不移植——单进程执行（预训练同先例），分片并行留给
  #80 成本读数后的后续票。

9 项冻结变量（upstream-eval-protocol §5.1）全部落 ``MrFidConfig``
schema：必填即钉死、范围校验、``FidResult`` provenance 全字段落盘
（落盘纪律 #6）。特征缓存带几何口径 fingerprint 防护（§1.7 缓存陷阱的
结构化兜底：换口径不重算即硬错误，不再靠 ``--ignore_existing`` 操作
纪律）。

零依赖原则：上游脚本只读参照、永不 import——本模块是其数据处理语义
的 cynosure 重写（同 ``reward/preprocessing.py`` 的承接方式）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol

import monai
import torch
from monai.transforms import Compose
from pydantic import BaseModel, ConfigDict, field_validator

from cynosure.config import SpecField
from cynosure.eval.features import RadImageNetBackbone
from cynosure.eval.frechet import FrechetDistance

_FID_MODALITIES = Literal["ct", "mr"]
"""移植继承 fork 的 modality 表（CT 臂为表结构的一部分；本线生产用 mr）。"""

_MODEL_NAMES = Literal["radimagenet_resnet50"]
"""特征网络钉死 RadImageNet-ResNet50：squeezenet torchvision fallback 臂
已裁（import 白名单之外 + #69 冻结变量 #6——两个网络是不同特征空间，
数字不可互比）。"""

_EXTRACT_DEVICES = Literal["cpu", "cuda", "mps"]
"""特征提取设备白名单。冻结变量 #8：设备浮点差异会进特征值，每次对比
必须在 config 显式声明并随 FidResult 落盘（torch-dcu 上 ``cuda`` 为
别名，与上游脚本同语义）。"""

_SUBTRACT_MEAN = [0.406, 0.456, 0.485]
"""通道翻转后的 ImageNet 均值（标准序 [0.485, 0.456, 0.406] 配合
[2,1,0] 翻转自洽）——fork ``subtract_mean`` 逐字常量。"""

_NIFTI_SUFFIX = ".nii.gz"
_FEATURE_SUFFIX = ".pt"


class MrFidConfig(BaseModel):
    """MR FID 仪器的单次对比配置：9 项冻结变量的 schema 化（#79）。

    必填即钉死（``target_shape`` / ``center_slices_ratio`` / ``num_images``
    等无默认值）：每次对比的 config 文件就是冻结记录的载体，
    ``FidResult`` 从同一 config 落 provenance，杜绝口头口径。schema 层
    不做文件存在性检查（仓库惯例：存在性属运行时输入契约）。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    real_dataset_root: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "real 侧体数据根目录（filelist 行相对于它解析）",
    )
    real_filelist: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "real 侧清单（纯文本每行一相对路径；集合决定读数——脚本 sort()，"
        "行序无关）",
    )
    real_features_dir: str = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "real 侧特征缓存子目录名（位于 output_root/<modality>/ 下）",
    )
    synth_dataset_root: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "synth 侧体数据根目录",
    )
    synth_filelist: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "synth 侧清单（同 real 口径）",
    )
    synth_features_dir: str = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "synth 侧特征缓存子目录名",
    )
    num_images: int = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "每侧体数据上限（双侧各自截断，非合计；两侧基数变化同时改变 "
        "FID 的有偏程度，逐次对比必须一致）",
        ge=2,
    )
    target_shape: tuple[int, int, int] = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "几何口径之 padding/crop 目标形状（无默认——每次对比显式钉死）",
    )
    center_slices_ratio: float = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "中心切片窗口比例（无默认——论文 0.5 与 docs 示例 0.4 不一致，"
        "必须逐次对比显式钉死；0 < r ≤ 1）",
        gt=0.0, le=1.0,
    )
    resample_spacing: tuple[float, float, float] | None = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "Spacingd 重采样体素间距（None = 不重采样，fork 默认）",
        default=None,
    )
    enable_padding: bool = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "SpatialPadd 到 target_shape（value = padding_value，MR=0）",
        default=True,
    )
    enable_center_cropping: bool = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "CenterSpatialCropd 到 target_shape（chain 中先于强度变换）",
        default=True,
    )
    ignore_existing: bool = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "True = 无视缓存强制重提特征（几何口径变更后的换血通道；缓存"
        "目录带 fingerprint 防护，口径未变的普通复用保持 False）",
        default=False,
    )
    modality: _FID_MODALITIES = SpecField(
        "定死(表)", "upstream-eval-protocol §4.1",
        "强度域预处理分支（mr = 百分位动态窗口 → (0,1000)；ct 臂为 "
        "fork 表结构的一部分，本线生产用 mr）",
        default="mr",
    )
    model_name: _MODEL_NAMES = SpecField(
        "定死", "upstream-eval-protocol §5.1",
        "特征网络钉死 radimagenet_resnet50（squeezenet fallback 臂已裁）",
        default="radimagenet_resnet50",
    )
    radimagenet_weights: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "RadImageNet-ResNet50 权重文件本体（冻结变量 #7：同版本同文件；"
        "本地注入——torch.hub 需外网，集群阻断）",
    )
    device: _EXTRACT_DEVICES = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "特征提取设备（冻结变量 #8：设备浮点差异会进特征值）",
        default="cpu",
    )
    dtype: Literal["float32"] = SpecField(
        "定死", "upstream-eval-protocol §5.1",
        "特征提取 dtype 钉死 float32（fork 模型默认口径）",
        default="float32",
    )
    output_root: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "特征缓存/结果根目录（缓存实际位 "
        "output_root/<modality>/<features_dir>/，改造 #5 防跨模态复用）",
    )
    result_json: Path = SpecField(
        "运行时", "upstream-eval-protocol §5.1",
        "FidResult provenance 落盘路径（落盘纪律：每次运行必写，"
        "必填无默认）",
    )
    num_workers: int = SpecField(
        "运行时", "本 spec 补钉",
        "DataLoader 读图/变换进程数（不改数值，仅吞吐；fork 用 6）",
        default=4, ge=0,
    )

    @field_validator("target_shape")
    @classmethod
    def _target_shape_positive(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(dim < 1 for dim in value):
            raise ValueError(f"target_shape 各维须为正，得到 {value}")
        return value

    @field_validator("resample_spacing")
    @classmethod
    def _spacing_positive(
        cls, value: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        if value is not None and any(dim <= 0 for dim in value):
            raise ValueError(f"resample_spacing 各维须为正，得到 {value}")
        return value


@dataclass(frozen=True)
class ModalityPreprocessing:
    """单模态强度域常量与预处理链（fork ``ModalityPreprocessing`` 移植）。

    MR 用训练管线的 (0, 99.5) 百分位窗口动态映射到生成侧 (0, 1000)
    输出域（utils_infer 落盘域同口径），clip 关——MR 强度 scanner/序列
    依赖，real 体可远超 1000，固定窗会截断亮尾。``compose`` 产出的
    **单一 Compose 实例必须同时服务 real 与 synth 两个 loader**
    （FID 可比性要求）。
    """

    name: str
    padding_value: float
    output_range: tuple[float, float]
    percentile_range: tuple[float, float] | None = None

    def feature_cache_dir(self, output_root: Path, features_dir: str) -> Path:
        """缓存目录 = ``output_root/<modality>/<features_dir>``（fork
        改造 #5：按模态命名空间隔离，防跨模态复用 .pt 特征）。"""
        return Path(output_root) / self.name / features_dir

    def compose(
        self,
        target_shape: tuple[int, int, int],
        resample_spacing: tuple[float, float, float] | None = None,
        center_crop: bool = True,
        pad: bool = True,
    ) -> Compose:
        """逐卷预处理链（fork 改造 #2/#3/#4：crop → 强度 → pad）。

        强度夹在 crop 与 pad 之间是 MR 正确性的关键：MONAI 的
        CenterSpatialCropd 在 roi 大于影像时原样返回不补零，百分位
        统计只看真实内容；若按官方 pad→crop→强度 顺序，padding 的 0
        会稀释百分位，缩放倍数随体积大小漂移（§4.3 数值验证）。
        """
        transform_list = [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Orientationd(keys=["image"], axcodes="RAS"),
        ]
        if resample_spacing is not None:
            transform_list.append(
                monai.transforms.Spacingd(
                    keys=["image"], pixdim=resample_spacing, mode=["bilinear"],
                ),
            )
        if center_crop:
            transform_list.append(
                monai.transforms.CenterSpatialCropd(
                    keys=["image"], roi_size=target_shape,
                ),
            )
        if self.percentile_range is not None:
            lower, upper = self.percentile_range
            b_min, b_max = self.output_range
            transform_list.append(
                monai.transforms.ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=lower, upper=upper,
                    b_min=b_min, b_max=b_max, clip=False,
                ),
            )
        else:
            a_min, a_max = self.output_range
            transform_list.append(
                monai.transforms.ScaleIntensityRanged(
                    keys=["image"], a_min=a_min, a_max=a_max,
                    b_min=a_min, b_max=a_max, clip=True,
                ),
            )
        if pad:
            transform_list.append(
                monai.transforms.SpatialPadd(
                    keys=["image"], spatial_size=target_shape,
                    mode="constant", value=self.padding_value,
                ),
            )
        return Compose(transform_list)


MODALITY_PREPROCESSING: dict[str, ModalityPreprocessing] = {
    "ct": ModalityPreprocessing(
        name="ct", padding_value=-1000, output_range=(-1000, 1000),
    ),
    "mr": ModalityPreprocessing(
        name="mr", padding_value=0, output_range=(0, 1000),
        percentile_range=(0.0, 99.5),
    ),
}
"""fork ``MODALITY_PREPROCESSING`` 常量表逐字移植。"""


@dataclass(frozen=True)
class FidResult:
    """一次 FID 调用的数字 + provenance（fork ``FidResult`` 移植，改造 #6）。

    冻结记录的落盘形态：每个数字携带识别本次运行所需的全部口径——
    双侧 filelist、模态预处理、采样/几何旗标（``enable_*`` 三个字段
    fork 里默认 None 读作「未记录」，本移植落 config 实际值）。
    """

    comparison_tag: str
    fid_xy: float
    fid_yz: float
    fid_zx: float
    fid_avg: float
    modality: str
    model_name: str
    num_images: int
    real_filelist: str
    synth_filelist: str
    target_shape: str
    center_slices_ratio: float | None
    enable_padding: bool | None = None
    enable_center_cropping: bool | None = None
    enable_resampling_spacing: str | None = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load(cls, path: Path) -> FidResult:
        with path.open() as file:
            payload = json.load(file)
        return cls(**payload)


class PlaneFeatureBackbone(Protocol):
    """仪器面对的特征骨干窄接口：[N, 3, H, W] 已归一化切片 → [N, F]。

    生产 = ``RadImageNetBackbone``；测试注入确定性替身（Strategy，
    与 ``SliceFeatureExtractor`` 同构的接缝）。
    """

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """已归一化的 3 通道切片批 → 池化特征矩阵。"""
        ...


class VolumePlaneSlicer:
    """fork ``get_features_2p5d`` 的口径移植：单卷三正交面切片 → 逐面特征。

    切片轴语义照抄 fork（不按名字推断轴）：XY = 沿 D 轴 unbind（得
    H×W 面）、YZ = 沿 H 轴（得 W×D 面）、ZX = 沿 W 轴（得 H×D 面）。
    单通道复制 3 通道后翻转 [2,1,0]，逐卷×逐平面全局 min-max 归一
    （**非逐切片**——norm2d=False 4D 分支，DataLoader batch=1 下等价
    「每卷×每平面」一个 min/max），再减通道均值。
    """

    def __init__(
        self, backbone: PlaneFeatureBackbone, center_slices_ratio: float,
    ) -> None:
        if not 0.0 < center_slices_ratio <= 1.0:
            raise ValueError(
                f"中心切片比例须在 (0, 1]，得到 {center_slices_ratio}"
            )
        self._backbone = backbone
        self._ratio = center_slices_ratio

    def plane_features(
        self, volume: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单卷 [C, H, W, D] → (XY, YZ, ZX) 逐面特征矩阵 [N_i, F]。"""
        if volume.dim() != 4:
            raise ValueError(
                f"体数据须为 [C, H, W, D]（单卷），得到 {tuple(volume.shape)}"
            )
        # fork 5D [B, C, H, W, D] 语义逐字保留（B=1）：unbind 后切片仍
        # 携带 batch 维，cat 得 4D [N, 3, x, y]——去掉 batch 维会让
        # unbind 产出 3D 切片破坏后续归一/前向
        image = volume.unsqueeze(0)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1, 1)
        if image.shape[1] != 3:
            raise ValueError(
                f"体数据通道数须为 1 或 3，得到 {image.shape[1]}"
            )
        # 'RGB'→(R,G,B) 翻转为 (B,G,R)（fork 逐字语义）
        image = image[:, [2, 1, 0], ...]
        _, _, height, width, depth = image.shape
        start_d, end_d = self._center_window(depth)
        start_h, end_h = self._center_window(height)
        start_w, end_w = self._center_window(width)
        with torch.no_grad():
            feature_xy = self._plane_forward(
                image[..., start_d:end_d], dim=-1,
            )
            feature_yz = self._plane_forward(
                image[:, :, start_h:end_h, :, :], dim=2,
            )
            feature_zx = self._plane_forward(
                image[:, :, :, start_w:end_w, :], dim=3,
            )
        return feature_xy, feature_yz, feature_zx

    def _center_window(self, axis_length: int) -> tuple[int, int]:
        """fork 中心窗口公式：int((1±r)/2·N)（截断语义照抄）。"""
        start = int((1.0 - self._ratio) / 2.0 * axis_length)
        end = int((1.0 + self._ratio) / 2.0 * axis_length)
        return start, end

    def _plane_forward(self, image_sub: torch.Tensor, dim: int) -> torch.Tensor:
        """单平面切片批 → 归一 → 骨干前向（分块在骨干内部）。"""
        images_2d = torch.cat(torch.unbind(image_sub, dim=dim), dim=0)
        images_2d = self._radimagenet_normalise(images_2d)
        return self._backbone.forward(images_2d)

    @staticmethod
    def _radimagenet_normalise(volume: torch.Tensor) -> torch.Tensor:
        """fork ``radimagenet_intensity_normalisation`` 4D 分支逐字移植：
        全局 min-max（逐卷×逐平面，+1e-10 防除零）→ 减翻转后通道均值。"""
        max3d = torch.max(volume)
        min3d = torch.min(volume)
        volume = (volume - min3d) / (max3d - min3d + 1e-10)
        volume[:, 0, ...] -= _SUBTRACT_MEAN[0]
        volume[:, 1, ...] -= _SUBTRACT_MEAN[1]
        volume[:, 2, ...] -= _SUBTRACT_MEAN[2]
        return volume


class FeatureCacheGuard:
    """``.pt`` 特征缓存 + 几何口径 fingerprint 防护（§1.7 陷阱的结构化兜底）。

    缓存路径 = ``cache_root/<体数据相对路径>.pt``（``cache_root`` 已含
    ``output_root/<modality>/<features_dir>`` 前缀）。fingerprint 覆盖
    一切影响逐卷特征的口径字段（几何/模态/网络/权重本体/设备/dtype）；
    目录已有 fingerprint 且与本次不符时**硬错误**——除非
    ``ignore_existing`` 显式换血（换血同时覆写 fingerprint）。
    ``num_images``/filelist 不参与 fingerprint：逐卷特征与取多少卷、
    取哪些卷无关，缓存必须可跨截断复用。
    """

    FINGERPRINT_NAME = "fingerprint.json"

    def __init__(
        self, cache_root: Path, fingerprint: dict, ignore_existing: bool,
    ) -> None:
        self._cache_root = cache_root
        self._fingerprint = fingerprint
        self._ignore_existing = ignore_existing
        recorded_path = cache_root / self.FINGERPRINT_NAME
        if recorded_path.is_file():
            with recorded_path.open() as file:
                recorded = json.load(file)
            if recorded != fingerprint and not ignore_existing:
                divergent = sorted(
                    key for key in set(recorded) | set(fingerprint)
                    if recorded.get(key) != fingerprint.get(key)
                )
                raise ValueError(
                    "特征缓存口径指纹不匹配（upstream-eval-protocol §1.7 "
                    "缓存陷阱防护）：缓存目录 "
                    f"{cache_root} 由另一套口径产出，复用会静默给出错误"
                    f"数字。差异字段: {divergent}。"
                    "换几何/模态/权重口径后请改 features_dir 或置 "
                    "ignore_existing=true 强制重提"
                )
        cache_root.mkdir(parents=True, exist_ok=True)
        with recorded_path.open("w") as file:
            json.dump(fingerprint, file, indent=2)

    def cache_path(self, dataset_root: Path, volume_path: Path) -> Path:
        """体数据路径 → 缓存路径（``.nii.gz`` 后缀换 ``.pt``）。"""
        relative = volume_path.relative_to(dataset_root)
        name = relative.name
        if name.endswith(_NIFTI_SUFFIX):
            name = name[: -len(_NIFTI_SUFFIX)] + _FEATURE_SUFFIX
        else:
            name = name + _FEATURE_SUFFIX
        return self._cache_root / relative.parent / name

    def load_or_extract(
        self,
        cache_path: Path,
        extract: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """命中缓存直接装载；否则提取并落盘（``ignore_existing`` 强制后者）。"""
        if not self._ignore_existing and cache_path.is_file():
            feats = torch.load(cache_path, weights_only=True)
            return feats[0], feats[1], feats[2]
        feats = extract()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
        return feats


class MrFidInstrument:
    """单次 FID 对比的执行编排（fork ``main()`` 的单进程移植）。

    流程照抄：双侧清单 sort()+截断 → 单一 Compose 双 loader 共享 →
    逐卷三平面特征提取（缓存/fingerprint 防护）→ 逐面 FID → 算术
    平均 → ``FidResult`` 落盘。距离核用 cynosure 标准的
    ``FrechetDistance``（float64 谱分解口径，ADR-0004；fork 的 MONAI
    FIDMetric 依赖 scipy 不在 import 白名单，且跨实现数字本就不可比）。
    """

    def __init__(
        self,
        config: MrFidConfig,
        backbone: PlaneFeatureBackbone | None = None,
    ) -> None:
        self._config = config
        self._preprocessing = MODALITY_PREPROCESSING[config.modality]
        self._device = torch.device(config.device)
        self._backbone = (
            backbone
            if backbone is not None
            else RadImageNetBackbone(config.radimagenet_weights, self._device)
        )
        self._slicer = VolumePlaneSlicer(
            self._backbone, config.center_slices_ratio,
        )

    def run(self, comparison_tag: str = "") -> FidResult:
        """执行一次对比：提取双侧特征 → 逐面 FID → 落盘并返回结果。"""
        config = self._config
        fingerprint = self._fingerprint()
        transforms = self._preprocessing.compose(
            target_shape=config.target_shape,
            resample_spacing=config.resample_spacing,
            center_crop=config.enable_center_cropping,
            pad=config.enable_padding,
        )
        real_xy, real_yz, real_zx = self._extract_side(
            dataset_root=config.real_dataset_root,
            filelist=config.real_filelist,
            features_dir=config.real_features_dir,
            transforms=transforms,
            fingerprint=fingerprint,
        )
        synth_xy, synth_yz, synth_zx = self._extract_side(
            dataset_root=config.synth_dataset_root,
            filelist=config.synth_filelist,
            features_dir=config.synth_features_dir,
            transforms=transforms,
            fingerprint=fingerprint,
        )
        frechet = FrechetDistance()
        fid_xy = frechet.score(synth_xy.cpu(), real_xy.cpu())
        fid_yz = frechet.score(synth_yz.cpu(), real_yz.cpu())
        fid_zx = frechet.score(synth_zx.cpu(), real_zx.cpu())
        fid_avg = (fid_xy + fid_yz + fid_zx) / 3.0
        result = FidResult(
            comparison_tag=comparison_tag,
            fid_xy=float(fid_xy),
            fid_yz=float(fid_yz),
            fid_zx=float(fid_zx),
            fid_avg=float(fid_avg),
            modality=config.modality,
            model_name=config.model_name,
            num_images=config.num_images,
            real_filelist=str(config.real_filelist),
            synth_filelist=str(config.synth_filelist),
            target_shape="x".join(str(dim) for dim in config.target_shape),
            center_slices_ratio=config.center_slices_ratio,
            enable_padding=config.enable_padding,
            enable_center_cropping=config.enable_center_cropping,
            enable_resampling_spacing=(
                None
                if config.resample_spacing is None
                else "x".join(str(dim) for dim in config.resample_spacing)
            ),
        )
        result.save(config.result_json)
        return result

    def _fingerprint(self) -> dict:
        """缓存防护指纹：一切影响逐卷特征的口径字段（见类 docstring）。

        权重本体项 = 权重文件 sha256；文件不在（注入替身的测试场景——
        装载本就不经本运行）记 None——指纹仍随设备/几何/口径隔离，
        生产路径的默认装配已在构造期强制权重存在。
        """
        config = self._config
        digest = hashlib.sha256()
        weights_sha256: str | None = None
        if config.radimagenet_weights.is_file():
            with config.radimagenet_weights.open("rb") as file:
                for chunk in iter(lambda: file.read(1 << 20), b""):
                    digest.update(chunk)
            weights_sha256 = digest.hexdigest()
        return {
            "modality": config.modality,
            "target_shape": list(config.target_shape),
            "resample_spacing": (
                None if config.resample_spacing is None
                else list(config.resample_spacing)
            ),
            "enable_padding": config.enable_padding,
            "enable_center_cropping": config.enable_center_cropping,
            "center_slices_ratio": config.center_slices_ratio,
            "model_name": config.model_name,
            "radimagenet_weights_sha256": weights_sha256,
            "device": config.device,
            "dtype": config.dtype,
        }

    def _extract_side(
        self,
        dataset_root: Path,
        filelist: Path,
        features_dir: str,
        transforms: Compose,
        fingerprint: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """单侧清单 → 逐卷三平面特征（vstack 后 [总切片数, F]）。"""
        if not filelist.is_file():
            raise FileNotFoundError(f"清单文件不存在: {filelist}")
        with filelist.open() as file:
            lines = [line.strip() for line in file.readlines()]
        lines = sorted(line for line in lines if line)
        lines = lines[: self._config.num_images]
        if not lines:
            raise ValueError(f"清单为空（截断后无体数据）: {filelist}")
        missing = [
            line for line in lines
            if not (dataset_root / line).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"清单 {filelist} 有 {len(missing)} 行在 {dataset_root} 下"
                f"不存在，如 {missing[:2]}"
            )
        guard = FeatureCacheGuard(
            self._preprocessing.feature_cache_dir(
                self._config.output_root, features_dir,
            ),
            fingerprint,
            self._config.ignore_existing,
        )
        dataset = monai.data.Dataset(
            data=[{"image": str(dataset_root / line)} for line in lines],
            transform=transforms,
        )
        loader = monai.data.DataLoader(
            dataset,
            num_workers=self._config.num_workers,
            batch_size=1,
            shuffle=False,
        )
        features_xy: list[torch.Tensor] = []
        features_yz: list[torch.Tensor] = []
        features_zx: list[torch.Tensor] = []
        for batch_data in loader:
            image = batch_data["image"].to(self._device)
            filename = batch_data["image"].meta["filename_or_obj"][0]
            volume_path = Path(filename)
            feats = guard.load_or_extract(
                guard.cache_path(dataset_root, volume_path),
                lambda: self._slicer.plane_features(image.as_tensor()[0]),
            )
            features_xy.append(feats[0])
            features_yz.append(feats[1])
            features_zx.append(feats[2])
        return (
            torch.vstack(features_xy),
            torch.vstack(features_yz),
            torch.vstack(features_zx),
        )


__all__ = [
    "FidResult",
    "FeatureCacheGuard",
    "MODALITY_PREPROCESSING",
    "ModalityPreprocessing",
    "MrFidConfig",
    "MrFidInstrument",
    "PlaneFeatureBackbone",
    "VolumePlaneSlicer",
]
