"""样本图版渲染核心的行为验证（issue #294）。

全部经由公共渲染入口缝（合成体 → 断言 PNG 产物 + 布局元数据）驱动，
先例 test_metrics_plots.py；CLI usage 面在 test_cli.py（既有唯一 seam，
issue #293 Testing Decisions）。"""

import struct
from pathlib import Path

import matplotlib
import matplotlib.image
import nibabel as nib
import numpy as np
import pytest
import torch

from cynosure.samples import (
    Crosshair,
    NiftiVolumeSource,
    PtVolumeSource,
    SamplePlateRenderer,
)


def anisotropic_volume() -> torch.Tensor:
    """合成各向异性体 [X=8, Y=12, Z=16]：沿轴递变梯度（percentile 窗
    有定义域、三面切片有可分辨内容）。"""
    x = torch.linspace(0.0, 1.0, 8).view(8, 1, 1)
    y = torch.linspace(0.0, 1.0, 12).view(1, 12, 1)
    z = torch.linspace(0.0, 1.0, 16).view(1, 1, 16)
    return 0.5 * (x + y + z)


def read_png_size(path: Path) -> tuple[int, int]:
    """PNG IHDR 的 (宽, 高) 像素读数（固定偏移 16/20，大端 uint32）。"""
    header = path.read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    return width, height


class TestPtSourceExplicitSpacing:
    """裸 .pt + 显式 spacing 的图版渲染：物理 mm 口径经布局元数据与
    PNG 尺寸读数验证（AC 1）。"""

    def test_mm_true_plate_with_layout_metadata(self, tmp_path: Path) -> None:
        pt = tmp_path / "case01.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()
        output = tmp_path / "plates"

        layout = SamplePlateRenderer(output).render(volume)

        assert layout.axis_codes == "RAS"
        assert layout.volume_shape == (8, 12, 16)
        assert layout.spacing == (1.0, 2.0, 3.0)
        assert layout.spacing_origin == "explicit"
        assert [panel.plane for panel in layout.panels] == ["XY", "YZ", "ZX"]
        by_plane = {panel.plane: panel for panel in layout.panels}
        # 物理 mm 口径：panel 纵横比 = 显示行轴 spacing / 列轴 spacing
        assert by_plane["XY"].aspect == pytest.approx(2.0)  # Y(s=2) / X(s=1)
        assert by_plane["YZ"].aspect == pytest.approx(1.5)  # Z(s=3) / Y(s=2)
        assert by_plane["ZX"].aspect == pytest.approx(3.0)  # Z(s=3) / X(s=1)
        # 切片缺省中心层：法轴长的一半
        assert by_plane["XY"].slice_index == 8  # 法轴 Z，长 16
        assert by_plane["YZ"].slice_index == 4  # 法轴 X，长 8
        assert by_plane["ZX"].slice_index == 6  # 法轴 Y，长 12
        # 显示轴序（行, 列）：行轴含 S 者优先、列轴 0 端朝左——显示
        # 几何与轴标注/定位线/纵横比的公共锚
        assert (by_plane["XY"].row_axis, by_plane["XY"].col_axis) == ("Y", "X")
        assert (by_plane["YZ"].row_axis, by_plane["YZ"].col_axis) == ("Z", "Y")
        assert (by_plane["ZX"].row_axis, by_plane["ZX"].col_axis) == ("Z", "X")
        assert by_plane["XY"].display_shape == (12, 8)
        assert by_plane["YZ"].display_shape == (16, 12)
        assert by_plane["ZX"].display_shape == (16, 8)
        # 逐体 percentile 窗（0.5–99.5）
        lo, hi = np.percentile(anisotropic_volume().numpy(), [0.5, 99.5])
        assert by_plane["XY"].window == pytest.approx((lo, hi))
        # PNG 产物与尺寸读数一致
        png = output / "case01.png"
        assert png.is_file()
        assert layout.png_size_px == read_png_size(png)


class TestPtSourceSpacingDowngrade:
    """裸 .pt 的 spacing 缺省降级（AC 4）：体素等比 + 降级标记（CLI 面
    据此 stdout 明示）。"""

    def test_missing_spacing_downgrades_to_isotropic(self, tmp_path: Path) -> None:
        pt = tmp_path / "case02.pt"
        torch.save(anisotropic_volume(), pt)

        volume = PtVolumeSource(pt).load()

        assert volume.spacing == (1.0, 1.0, 1.0)
        assert volume.spacing_origin == "isotropic-default"

    def test_channel_first_pt_unwrapped_to_3d(self, tmp_path: Path) -> None:
        """预处理链落盘形状 [1, D, H, W]（单通道）同样可打图。"""
        pt = tmp_path / "case03.pt"
        torch.save(anisotropic_volume().unsqueeze(0), pt)

        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()

        assert tuple(volume.data.shape) == (8, 12, 16)


class TestSliceOffsetAndWindow:
    """切层偏移与显式窗覆盖（AC 3）。"""

    def test_slice_offset_moves_all_planes_with_clamp(self, tmp_path: Path) -> None:
        pt = tmp_path / "case04.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()
        renderer = SamplePlateRenderer(tmp_path / "plates")

        layout = renderer.render(volume, slice_offset=3)
        by_plane = {panel.plane: panel for panel in layout.panels}
        assert by_plane["XY"].slice_index == 11  # 8 + 3（法轴 Z）
        assert by_plane["YZ"].slice_index == 7   # 4 + 3（法轴 X）
        assert by_plane["ZX"].slice_index == 9   # 6 + 3（法轴 Y）

        clamped = renderer.render(volume, slice_offset=100)
        clamped_by_plane = {panel.plane: panel for panel in clamped.panels}
        assert clamped_by_plane["XY"].slice_index == 15  # clamp 到 Z 末端

    def test_explicit_window_overrides_percentile(self, tmp_path: Path) -> None:
        pt = tmp_path / "case05.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()

        layout = SamplePlateRenderer(tmp_path / "plates").render(
            volume, window=(0.1, 0.9),
        )

        for panel in layout.panels:
            assert panel.window == pytest.approx((0.1, 0.9))


class TestCrosshairs:
    """三格定位线互标切层位置（AC 3）：每 panel 收另两 panel 的切层。"""

    def test_crosshairs_mark_other_planes_slice_positions(
        self, tmp_path: Path,
    ) -> None:
        pt = tmp_path / "case06.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()

        layout = SamplePlateRenderer(tmp_path / "plates").render(volume)

        by_plane = {panel.plane: panel for panel in layout.panels}
        # 中心层读数（法轴长之半）：z*=8、x*=4、y*=6
        assert by_plane["XY"].crosshairs == [
            Crosshair(from_plane="YZ", direction="vertical", position=4),
            Crosshair(from_plane="ZX", direction="horizontal", position=6),
        ]
        assert by_plane["YZ"].crosshairs == [
            Crosshair(from_plane="XY", direction="horizontal", position=8),
            Crosshair(from_plane="ZX", direction="vertical", position=6),
        ]
        assert by_plane["ZX"].crosshairs == [
            Crosshair(from_plane="XY", direction="horizontal", position=8),
            Crosshair(from_plane="YZ", direction="vertical", position=4),
        ]


class TestNiftiSource:
    """NIfTI 输入：装载即显式重定向 RAS、header spacing 自动读取（AC 2）。"""

    def test_lps_nifti_redirected_to_ras_with_header_spacing(
        self, tmp_path: Path,
    ) -> None:
        """原生 LPS 朝向（diag 负 affine）的 NIfTI：RAS 重定向生效
        （布局元数据轴码断言），spacing = header 步长 (2, 3, 4)。"""
        path = tmp_path / "ref.nii.gz"
        affine = np.diag([-2.0, -3.0, 4.0, 1.0])  # LPS 朝向
        data = anisotropic_volume().numpy()
        nib.save(nib.Nifti1Image(data, affine), path)

        volume = NiftiVolumeSource(path).load()

        assert volume.axis_codes == "RAS"
        assert volume.spacing == pytest.approx((2.0, 3.0, 4.0))
        assert volume.spacing_origin == "header"
        assert tuple(volume.data.shape) == (8, 12, 16)

        layout = SamplePlateRenderer(tmp_path / "plates").render(volume)
        assert layout.axis_codes == "RAS"
        assert (tmp_path / "plates" / "ref.png").is_file()
        by_plane = {panel.plane: panel for panel in layout.panels}
        assert by_plane["XY"].aspect == pytest.approx(1.5)  # Y(3) / X(2)


class TestOrientationLabelsAndDeterminism:
    """轴字母标注（L/R/A/P/I/S，RAS 惯例）与确定性重打（AC 3 / spec
    「产物集合与确定性：同输入重打同产物」）。"""

    def test_corner_labels_follow_ras_convention(self, tmp_path: Path) -> None:
        pt = tmp_path / "case07.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()

        layout = SamplePlateRenderer(tmp_path / "plates").render(volume)

        by_plane = {panel.plane: panel for panel in layout.panels}
        assert by_plane["XY"].corner_labels == {
            "top": "A", "bottom": "P", "left": "L", "right": "R",
        }
        assert by_plane["YZ"].corner_labels == {
            "top": "S", "bottom": "I", "left": "P", "right": "A",
        }
        assert by_plane["ZX"].corner_labels == {
            "top": "S", "bottom": "I", "left": "L", "right": "R",
        }

    def test_same_input_renders_identical_png_bytes(self, tmp_path: Path) -> None:
        pt = tmp_path / "case08.pt"
        torch.save(anisotropic_volume(), pt)
        volume = PtVolumeSource(pt, spacing=(1.0, 2.0, 3.0)).load()
        renderer = SamplePlateRenderer(tmp_path / "plates")

        first = renderer.render(volume)
        second = renderer.render(volume)

        assert first.png.read_bytes() == second.png.read_bytes()
        assert first.model_dump(exclude={"png"}) == second.model_dump(exclude={"png"})


class TestPixelOrientationAnchor:
    """像素级关键几何锚（spec「仅锚关键几何」）：RAS 朝向落到画面——
    元数据断言不了 imshow 行序与画面的映射（origin 翻转类缺陷），
    Y 两端亮度极端的合成体经 XY panel 画面上半/下半读数拦截。"""

    def test_xy_panel_top_is_anterior_end(self, tmp_path: Path) -> None:
        volume = torch.zeros(4, 12, 4)
        volume[:, :4, :] = 0.0    # Y=0..3 → P 端全黑
        volume[:, 4:8, :] = 0.3   # 中段中灰
        volume[:, -4:, :] = 0.7   # Y=8..11 → A 端亮灰（背景白=1.0 之外）
        pt = tmp_path / "orient.pt"
        torch.save(volume, pt)
        loaded = PtVolumeSource(pt, spacing=(1.0, 1.0, 1.0)).load()
        layout = SamplePlateRenderer(tmp_path / "plates").render(loaded)

        image = matplotlib.image.imread(layout.png)
        quadrant = image[:500, :500]  # XY panel = 左上象限
        content = quadrant < 0.95  # 剔除 letterbox/画布白背景
        rows = torch.from_numpy(content.any(axis=1).nonzero()[0])
        middle = (int(rows[0]) + int(rows[-1])) // 2
        top = quadrant[:middle][content[:middle]].mean()
        bottom = quadrant[middle:][content[middle:]].mean()
        assert top > bottom + 0.1  # 上=A 端（0.7）亮于 下=P 端（0.0）
