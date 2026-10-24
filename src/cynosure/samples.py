"""样本图版（sample plate）渲染入口（issue #294，与 plots.py 指标绘图
平级的公共渲染面）：渲染就绪体 + 渲染参数 → 2×2 医学三视图 PNG + 布局
元数据（元数据同时是测试缝的断言面）。

一个样本一张 PNG：XY/YZ/ZX 三正交面（``OrthoPlane`` 同约定——朝向单点）
+ 第四格信息面板；三格定位线互标彼此切层位置；panel 纵横比 = 显示两维
spacing 比（物理 mm 口径，各向异性网格下层厚轴不视觉失真）。逐体
percentile 窗（0.5–99.5）缺省、显式窗参数覆盖；切片缺省中心层、切层
偏移可调。

体来源分派（Strategy，spec「三实现、单一消费面」）：裸 ``.pt``（预处
理链/样本落盘契约——无头裸张量、已 RAS；spacing 显式或缺省体素等比
降级）、NIfTI（MONAI 装载即显式重定向 RAS，header spacing 经 affine
列范数读取——方向无关口径，轴置换朝向下 raw pixdim 顺序不可靠）与
run 目录条目（manifest 样本路径 + 条件词汇表 spacing，#295）。

matplotlib Agg 后端直出 PNG（无显示环境可跑）；依赖零新增。
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, Sequence

import matplotlib
import nibabel as nib
import numpy as np
import torch
from matplotlib.figure import Figure
from monai.transforms import Compose, EnsureChannelFirst, LoadImage, Orientation
from pydantic import BaseModel

from cynosure.eval.volumes import OrthoPlane

matplotlib.use("Agg")

WINDOW_PERCENTILES = (0.5, 99.5)
"""逐体 percentile 窗的缺省分位（issue #293 决策）。"""

SPACING_ORIGIN_EXPLICIT = "explicit"
"""spacing 来源：调用方显式指定（.pt 参数）。"""

SPACING_ORIGIN_HEADER = "header"
"""spacing 来源：NIfTI header 自动读取。"""

SPACING_ORIGIN_ISOTROPIC = "isotropic-default"
"""spacing 来源：不可得时的降级体素等比（CLI 面 stdout 明示）。"""

SPACING_ORIGIN_VOCABULARY = "vocabulary"
"""spacing 来源：run 目录条目经条件词汇表解析（issue #295；MR-RATE 线
逐条件等效 spacing、BraTS 线单位间距各向同性常量——物理口径，非体素
等比降级，布局元数据断言面）。"""

SPACING_ORIGIN_LABELS = {
    SPACING_ORIGIN_EXPLICIT: "显式指定",
    SPACING_ORIGIN_HEADER: "NIfTI header",
    SPACING_ORIGIN_ISOTROPIC: "体素等比降级",
    SPACING_ORIGIN_VOCABULARY: "条件词汇表",
}
"""spacing 来源的 stdout 读数标签（口径降级可见）。"""

_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

_AXIS_CODE_ENDPOINTS = {"X": ("L", "R"), "Y": ("P", "A"), "Z": ("I", "S")}
"""RAS 轴向约定：体轴两端的解剖字母（轴码 0 端, 1 端）。"""


class Crosshair(BaseModel):
    """一条定位线：来源 panel 的切层在本 panel 显示坐标上的位置。"""

    from_plane: str
    direction: str
    position: int


class PanelLayout(BaseModel):
    """单个三视图 panel 的布局元数据（断言面）。"""

    plane: str
    slice_index: int
    normal_axis: str
    row_axis: str
    col_axis: str
    display_shape: tuple[int, int]
    aspect: float
    window: tuple[float, float]
    corner_labels: dict[str, str]
    crosshairs: list[Crosshair]


class PlateLayout(BaseModel):
    """一张样本图版的完整布局元数据（渲染入口返回值的元数据半边）。"""

    source: str
    volume_shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    spacing_origin: str
    axis_codes: str
    png: Path
    png_size_px: tuple[int, int]
    panels: list[PanelLayout]
    info_lines: list[str]
    """信息面板绘出的全部文本行（溯源块 + 几何读数；内容断言面）。"""


@dataclass
class LoadedVolume:
    """渲染就绪的像素体：[X, Y, Z] 张量 + spacing + 轴码 + 来源标识。"""

    data: torch.Tensor
    spacing: tuple[float, float, float]
    axis_codes: str
    source_label: str
    spacing_origin: str


class VolumeSource(Protocol):
    """体来源策略（spec「体来源分派，单一消费面」）：文件面 → 渲染就绪体。"""

    def load(self) -> LoadedVolume:
        """装载体文件（不存在/损坏/形状非法显式失败，不静默降级）。"""
        ...


class PtVolumeSource:
    """裸 ``.pt`` 体来源（预处理链/样本落盘契约：无头裸张量，已 RAS）。

    spacing 经构造参数显式给出；未给时降级体素等比 (1,1,1) 并打降级
    标记（消费面据此在 stdout 明示口径降级——降级可见，不静默）。"""

    def __init__(
        self, path: Path, spacing: tuple[float, float, float] | None = None,
    ) -> None:
        self._path = Path(path)
        self._spacing = spacing

    def load(self) -> LoadedVolume:
        data = torch.load(self._path, map_location="cpu", weights_only=True)
        if data.dim() == 4 and data.shape[0] == 1:
            # 预处理链落盘形状 [1, D, H, W] 的单通道维
            data = data[0]
        if data.dim() != 3:
            raise ValueError(
                f".pt 体须为 [X, Y, Z]（或单通道 [1, X, Y, Z]），得到 "
                f"{tuple(data.shape)}: {self._path}"
            )
        if self._spacing is not None:
            spacing: tuple[float, float, float] = tuple(
                float(v) for v in self._spacing
            )
            origin = SPACING_ORIGIN_EXPLICIT
        else:
            spacing = (1.0, 1.0, 1.0)
            origin = SPACING_ORIGIN_ISOTROPIC
        return LoadedVolume(
            data=data.float(),
            spacing=spacing,
            axis_codes="RAS",
            source_label=self._path.name,
            spacing_origin=origin,
        )


class NiftiVolumeSource:
    """NIfTI 体来源：装载即显式重定向 RAS（GLOSSARY「RAS 方向」），header
    spacing 经 affine 列范数读取。"""

    _STEPS = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
    ])

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    def load(self) -> LoadedVolume:
        image = self._STEPS(self._path)
        affine = image.affine
        spacing = tuple(float(v) for v in np.linalg.norm(affine[:3, :3], axis=0))
        codes = "".join(nib.orientations.aff2axcodes(affine))
        return LoadedVolume(
            data=image[0].float(),
            spacing=spacing,
            axis_codes=codes,
            source_label=self._path.name,
            spacing_origin=SPACING_ORIGIN_HEADER,
        )


class RunEntryVolumeSource:
    """run 目录条目体来源（issue #295，体来源三实现的收官）：manifest
    样本路径指向的无头裸 ``.pt``（样本落盘契约，已 RAS），spacing 由
    调用方经条件词汇表解析注入（MR-RATE 线逐条件等效 spacing / BraTS
    线单位间距各向同性），口径标记 ``vocabulary``——布局元数据据此
    断言物理口径（非体素等比降级）。"""

    def __init__(
        self, path: Path, spacing: tuple[float, float, float],
    ) -> None:
        self._delegate = PtVolumeSource(path, spacing=spacing)

    def load(self) -> LoadedVolume:
        loaded = self._delegate.load()
        return replace(loaded, spacing_origin=SPACING_ORIGIN_VOCABULARY)


class SamplePlateRenderer:
    """样本图版渲染器：LoadedVolume → 输出目录一张 2×2 PNG + PlateLayout。

    显示朝向约定（RAS 轴码）：panel 行轴 1 端朝上（含 S 者优先于 A）、
    列轴 0 端朝左——XY 上 A 下 P 左 L 右 R、YZ 上 S 下 I 左 P 右 A、
    ZX 上 S 下 I 左 L 右 R；轴字母标注画在 panel 四边中点，任何来源
    的体经 RAS 归一后同一观感。PNG 尺寸固定 figsize×dpi（确定性渲染）。
    """

    FIG_SIZE_INCHES = (10.0, 10.0)
    FIG_DPI = 100

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)

    def render(
        self,
        volume: LoadedVolume,
        slice_offset: int = 0,
        window: tuple[float, float] | None = None,
        caption: Sequence[str] | None = None,
    ) -> PlateLayout:
        """渲染单样本图版：缺省中心层 + 偏移、缺省 percentile 窗 + 覆盖；
        ``caption`` 为信息面板溯源行（run 目录条目的条件/索引/seed，#295），
        绘出来源标识之下并随布局元数据回传。"""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        data = volume.data
        win = self._resolve_window(data, window)
        slice_indices = {
            "X": self._center_index(data.shape[0], slice_offset),
            "Y": self._center_index(data.shape[1], slice_offset),
            "Z": self._center_index(data.shape[2], slice_offset),
        }
        label = volume.source_label
        stem = label[:-7] if label.endswith(".nii.gz") else Path(label).stem
        png_path = self._output_dir / f"{stem}.png"
        figure = Figure(figsize=self.FIG_SIZE_INCHES, dpi=self.FIG_DPI)
        axes = figure.subplots(2, 2)
        panels = [
            self._render_panel(axes[0, 0], volume, OrthoPlane.XY, slice_indices, win),
            self._render_panel(axes[0, 1], volume, OrthoPlane.YZ, slice_indices, win),
            self._render_panel(axes[1, 0], volume, OrthoPlane.ZX, slice_indices, win),
        ]
        info_lines = self._render_info_panel(axes[1, 1], volume, win, caption)
        figure.savefig(png_path)
        return PlateLayout(
            source=volume.source_label,
            volume_shape=tuple(data.shape),
            spacing=volume.spacing,
            spacing_origin=volume.spacing_origin,
            axis_codes=volume.axis_codes,
            png=png_path,
            png_size_px=(
                round(self.FIG_SIZE_INCHES[0] * self.FIG_DPI),
                round(self.FIG_SIZE_INCHES[1] * self.FIG_DPI),
            ),
            panels=panels,
            info_lines=info_lines,
        )

    def _render_panel(
        self,
        ax: matplotlib.axes.Axes,
        volume: LoadedVolume,
        plane: OrthoPlane,
        slice_indices: dict[str, int],
        win: tuple[float, float],
    ) -> PanelLayout:
        """三正交面 panel：切层、物理纵横比 imshow、定位线、轴字母标注。

        显示数组（imshow 行=数组 0 轴、列=数组 1 轴）：XY 切片 [X, Y]
        转置为 [Y, X]；YZ 切片 [Y, Z] 转置为 [Z, Y]；ZX 切片 [Z, X]
        原样。origin="lower" 使行 0（轴码 0 端）落画面底端、行 1 端朝
        上——与 corner_labels 的 top=1 端约定一致（upper 会垂直翻转，
        标注与像素矛盾）。"""
        data = volume.data
        if plane is OrthoPlane.XY:
            display = plane.single_slice(data, slice_indices["Z"]).T
            row_axis, col_axis = "Y", "X"
            crosshairs = [
                Crosshair(from_plane="YZ", direction="vertical",
                          position=slice_indices["X"]),
                Crosshair(from_plane="ZX", direction="horizontal",
                          position=slice_indices["Y"]),
            ]
            normal_axis, slice_index = "Z", slice_indices["Z"]
        elif plane is OrthoPlane.YZ:
            display = plane.single_slice(data, slice_indices["X"]).T
            row_axis, col_axis = "Z", "Y"
            crosshairs = [
                Crosshair(from_plane="XY", direction="horizontal",
                          position=slice_indices["Z"]),
                Crosshair(from_plane="ZX", direction="vertical",
                          position=slice_indices["Y"]),
            ]
            normal_axis, slice_index = "X", slice_indices["X"]
        else:
            display = plane.single_slice(data, slice_indices["Y"])
            row_axis, col_axis = "Z", "X"
            crosshairs = [
                Crosshair(from_plane="XY", direction="horizontal",
                          position=slice_indices["Z"]),
                Crosshair(from_plane="YZ", direction="vertical",
                          position=slice_indices["X"]),
            ]
            normal_axis, slice_index = "Y", slice_indices["Y"]
        aspect = (
            volume.spacing[_AXIS_INDEX[row_axis]]
            / volume.spacing[_AXIS_INDEX[col_axis]]
        )
        ax.imshow(
            display.numpy(), cmap="gray", origin="lower",
            vmin=win[0], vmax=win[1], aspect=aspect,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for crosshair in crosshairs:
            if crosshair.direction == "horizontal":
                ax.axhline(crosshair.position, color="gold", linewidth=0.8)
            else:
                ax.axvline(crosshair.position, color="gold", linewidth=0.8)
        top, bottom = _AXIS_CODE_ENDPOINTS[row_axis][1], _AXIS_CODE_ENDPOINTS[row_axis][0]
        left, right = _AXIS_CODE_ENDPOINTS[col_axis]
        for label, x, y in (
            (top, 0.5, 0.04), (bottom, 0.5, 0.96),
            (left, 0.04, 0.5), (right, 0.96, 0.5),
        ):
            ax.text(
                x, y, label, transform=ax.transAxes,
                color="gold", ha="center", va="center", fontsize=11,
            )
        return PanelLayout(
            plane=plane.value,
            slice_index=slice_index,
            normal_axis=normal_axis,
            row_axis=row_axis,
            col_axis=col_axis,
            display_shape=tuple(display.shape),
            aspect=float(aspect),
            window=win,
            corner_labels={"top": top, "bottom": bottom, "left": left, "right": right},
            crosshairs=crosshairs,
        )

    def _render_info_panel(
        self,
        ax: matplotlib.axes.Axes,
        volume: LoadedVolume,
        win: tuple[float, float],
        caption: Sequence[str] | None,
    ) -> list[str]:
        """第四格信息面板：来源标识 + 溯源行（run 目录条目条件/索引/
        seed，散文件模式为空）+ 体形状、spacing、窗范围；绘出文本行
        原样回传（布局元数据断言面）。"""
        ax.axis("off")
        shape = volume.data.shape
        spacing = " × ".join(f"{v:.3g}" for v in volume.spacing)
        lines = [
            volume.source_label,
            *(caption or []),
            f"shape: X={shape[0]} Y={shape[1]} Z={shape[2]}",
            f"spacing (mm): {spacing}",
            f"window: {win[0]:.4g} – {win[1]:.4g}",
            "axes: RAS (X→L/R, Y→P/A, Z→I/S)",
        ]
        ax.text(
            0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=11, family="monospace",
        )
        return lines

    @staticmethod
    def _resolve_window(
        data: torch.Tensor, window: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if window is not None:
            return float(window[0]), float(window[1])
        lo, hi = np.percentile(data.numpy(), list(WINDOW_PERCENTILES))
        return float(lo), float(hi)

    @staticmethod
    def _center_index(size: int, offset: int) -> int:
        """中心层 + 偏移，clamp 进 [0, size-1]（查看器语义：越界偏移
        停在端层，不拒绝）。"""
        return min(max(size // 2 + offset, 0), size - 1)
