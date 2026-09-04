"""像素体侧的评测材料：三正交面切片、真实参照影像库、配对保真度。

- **三正交面**（ADR-0004）：XY/YZ/ZX 三平面各自把体切成 2D 切片，
  逐面提取特征算 FID/KID 后汇总——2.5D 的「2.5」即三维体的三组正交
  二维视图；
- **真实参照**（experiment-design）：dataset_root 的真实影像体（BraTS
  病例目录布局与 prepare 同一扫描器），里程碑评测的 FID 参照侧与
  跨模态组 SSIM/MAE/PSNR 的 ground-truth 侧都取自它；
- **配对保真度**（跨模态组另加 3D SSIM/MAE/PSNR）：合成 target 影像与
  **同一病例 ground-truth 的 target 序列影像**逐例配对比较——配对数据
  集里病例全序列齐全，参照取 entry 锁定源病例的目标序列（非 source
  与 target 直接比较）。
"""

import enum
from pathlib import Path

import torch
from monai.metrics import MAEMetric, PSNRMetric, SSIMMetric
from monai.transforms import LoadImage

from cynosure.config import Modality
from cynosure.reward.dataset import BratsSeriesLayout, CaseSeries


class OrthoPlane(enum.Enum):
    """正交切面：成员知道自己如何把 [K, X, Y, Z] 体栈切成 [N, 1, H, W]。"""

    XY = "XY"
    YZ = "YZ"
    ZX = "ZX"

    def slice(self, volumes: torch.Tensor) -> torch.Tensor:
        """体栈 [K, X, Y, Z] → 该平面全部切片 [K×法轴长, 1, H, W]。
        三平面三侧（合成/参照）共用同一约定，切片朝向一致。"""
        if volumes.dim() != 4:
            raise ValueError(
                f"体栈须为 [K, X, Y, Z]，得到 {tuple(volumes.shape)}"
            )
        if self is OrthoPlane.XY:
            slices = volumes.permute(3, 0, 1, 2)  # [Z, K, X, Y]
        elif self is OrthoPlane.YZ:
            slices = volumes.permute(1, 0, 2, 3)  # [X, K, Y, Z]
        else:
            slices = volumes.permute(2, 0, 3, 1)  # [Y, K, Z, X]
        count = slices.shape[0] * slices.shape[1]
        return slices.reshape(count, 1, *slices.shape[2:])

    @classmethod
    def all_planes(cls) -> tuple["OrthoPlane", ...]:
        """spec 钉死的三正交面（XY/YZ/ZX，缺一即不是 2.5D）。"""
        return (cls.XY, cls.YZ, cls.ZX)


class RealVolumeStore:
    """真实参照影像库：dataset_root 病例目录布局的装载与缓存。

    里程碑评测每个里程碑都取同一批参照体——按 (case, modality) 缓存
    装载结果，重复评测不重复读盘。
    """

    def __init__(self, dataset_root: Path | str) -> None:
        self._layout = BratsSeriesLayout(Path(dataset_root))
        self._cases: dict[str, CaseSeries] = {
            case.case_id: case for case in self._layout.scan()
        }
        self._cache: dict[tuple[str, Modality], torch.Tensor] = {}
        self._reader = NiftiVolumeReader()

    def case_ids(self) -> list[str]:
        """全部病例 id（排序稳定——参照配对的确定性基础）。"""
        return sorted(self._cases)

    def volume(self, case_id: str, modality: Modality) -> torch.Tensor:
        """病例某序列的真实影像体 [X, Y, Z]（缓存）。"""
        if case_id not in self._cases:
            raise ValueError(
                f"参照库无病例 {case_id!r}（dataset_root 布局与 prepare 扫描器不符）"
            )
        key = (case_id, modality)
        if key not in self._cache:
            path = self._cases[case_id].series[modality]
            self._cache[key] = self._reader.read(path)
        return self._cache[key]


class NiftiVolumeReader:
    """NIfTI → [X, Y, Z] float 张量（MONAI LoadImage，与 prepare 同一口径）。"""

    def __init__(self) -> None:
        self._load = LoadImage(image_only=True)

    def read(self, path: Path) -> torch.Tensor:
        image = self._load(path)
        volume = torch.as_tensor(image).float()
        if volume.dim() != 3:
            raise ValueError(
                f"参照影像须为 3D 体，{path} 得到 {tuple(volume.shape)}"
            )
        return volume


class VolumePairFidelity:
    """配对保真度：对齐体栈的 3D SSIM、MAE 与 PSNR（跨模态组另加的三指标）。

    输入两侧逐例对齐（同一病例同一序列位）；data_range 取两侧联合强度
    范围——VAE 解码输出的强度尺度不归一，固定 data_range 会让 SSIM/PSNR
    随输出尺度漂移。
    """

    PSNR_CAP = 100.0
    """PSNR 封顶（dB）：全等体 MSE=0 数学上无穷，≥100 dB 即感知完美
    （指标流契约拒 inf，封顶保事件可落盘）。"""

    def score(
        self, synthetic: torch.Tensor, reference: torch.Tensor,
    ) -> tuple[float, float, float]:
        """逐例配对的 (mean SSIM, MAE, PSNR)。"""
        if synthetic.shape != reference.shape:
            raise ValueError(
                f"配对体栈形状不符：{tuple(synthetic.shape)} vs "
                f"{tuple(reference.shape)}（须逐例对齐）"
            )
        if synthetic.shape[0] < 1:
            raise ValueError("配对保真度需要非空体栈")
        if synthetic.shape[1] != 1:
            raise ValueError(
                f"体栈须为 [K, 1, X, Y, Z]（单通道影像），得到通道数 "
                f"{synthetic.shape[1]}"
            )
        data_range = float(
            torch.maximum(synthetic.max(), reference.max())
            - torch.minimum(synthetic.min(), reference.min())
        )
        if data_range <= 0.0:
            data_range = 1.0  # 常量体（全等）退化为单位量程，SSIM=1
        win_size = self._win_size(synthetic.shape[2:])
        ssim = SSIMMetric(spatial_dims=3, data_range=data_range, win_size=win_size)
        mae = MAEMetric()
        psnr = PSNRMetric(max_val=data_range)
        with torch.no_grad():
            ssim_value = float(ssim(synthetic, reference).mean())
            mae_value = float(mae(synthetic, reference).mean())
            psnr_value = min(float(psnr(synthetic, reference).mean()), self.PSNR_CAP)
            # 全等体 MSE=0 → PSNR 数学上无穷；指标流契约拒 inf，
            # ≥100 dB 即感知完美，封顶即可
        return ssim_value, mae_value, psnr_value

    @staticmethod
    def _win_size(spatial: torch.Size) -> int:
        """SSIM 高斯窗：MONAI 默认 11³，不超过体最短边且取奇数。"""
        win = min(11, min(spatial))
        return win - 1 if win % 2 == 0 else win
