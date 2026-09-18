"""像素体侧的评测材料：三正交面切片、真实参照影像库、配对保真度。

- **三正交面**（ADR-0004）：XY/YZ/ZX 三平面各自把体切成 2D 切片，
  逐面提取特征算 FID/KID 后汇总——2.5D 的「2.5」即三维体的三组正交
  二维视图；
- **真实参照**（experiment-design）：dataset_root 的真实影像体经上游
  recipe 预处理链读入（与 prepare 预编码同一口径，ADR-0006），里程碑
  评测的 FID 参照侧与跨模态组 SSIM/MAE/PSNR 的 ground-truth 侧都取自
  它——参照库按域分派两实现（``RealVolumeStore`` BraTS 病例目录布局 /
  ``MrReferenceVolumeStore`` MR-RATE 平铺影像树，#124），共同满足
  ``ReferenceVolumes`` 契约；
- **配对保真度**（跨模态组另加 3D SSIM/MAE/PSNR）：合成 target 影像与
  **同一病例 ground-truth 的 target 序列影像**逐例配对比较——配对数据
  集里病例全序列齐全，参照取 entry 锁定源病例的目标序列（非 source
  与 target 直接比较）。
"""

import enum
from pathlib import Path
from typing import Protocol

import torch
from monai.metrics import MAEMetric, PSNRMetric, SSIMMetric

from cynosure.conditions import MrConditionVocabulary
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.dataset import BratsSeriesLayout, CaseSeries
from cynosure.reward.preprocessing import UpstreamPreprocessChain


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


class ReferenceVolumes(Protocol):
    """里程碑参照侧的取数契约（``MilestoneEvaluator`` 的注入面，两域实现）。

    参照体选择 = 条目序号（或锁定病例）× 目标条件的影像体：组2（配对）
    ``source_case`` 优先——同一病例 ground-truth target；组1 按条目序号
    在**该条件的参照卷池**内确定性轮转（BraTS 病例全序列 → 轮转池为全
    病例；MR-RATE 一卷一条件 → 轮转池为该条件的卷——两域条件维度的
    语义差异封装在各实现内，消费面单一路径）。
    """

    def reference_volume(
        self, condition: str, entry_index: int, source_case: str | None,
    ) -> torch.Tensor:
        """条目的参照影像体 [X, Y, Z]（预处理后，与合成侧同影像空间）。"""
        ...

    def volume(self, case_id: str, condition: str) -> torch.Tensor:
        """指定病例（卷）某条件的参照影像体 [X, Y, Z]（缓存）。"""
        ...


class RealVolumeStore:
    """真实参照影像库：dataset_root 病例目录布局的装载与缓存。

    参照影像经**上游 recipe 预处理链**（``UpstreamPreprocessChain``，与
    prepare 预编码同一口径——RAS 重定向 + percentile 强度 + 128 倍数
    resize）读入：里程碑合成侧（VAE 解码的预处理空间）与参照侧必须在
    同一影像空间与强度域，裸读原生 NIfTI 构成跨域比较。
    ``case_ids`` 白名单把参照病例锁进 real pool 的 train split（spec
    「real 样本库」：real = 病例级 70% train split）——dataset_root 全树
    含 val/test 分区，不筛就构成参照分布的分割泄漏。
    里程碑评测每个里程碑都取同一批参照体——按 (case, modality) 缓存
    装载结果（缓存界 = 里程碑条目数个 (case, modality) 对），重复评测
    不重复读盘。
    """

    def __init__(
        self,
        dataset_root: Path | str,
        case_ids: set[str] | None = None,
        preprocess: UpstreamPreprocessChain | None = None,
    ) -> None:
        self._layout = BratsSeriesLayout(Path(dataset_root))
        cases: dict[str, CaseSeries] = {
            case.case_id: case for case in self._layout.scan()
        }
        if case_ids is not None:
            unknown = sorted(set(case_ids) - set(cases))
            if unknown:
                raise ValueError(
                    f"参照病例白名单含 dataset_root 不存在的病例: {unknown}"
                )
            cases = {
                case_id: case for case_id, case in cases.items()
                if case_id in case_ids
            }
            if not cases:
                raise ValueError("参照病例白名单过滤后无可用病例")
        self._cases: dict[str, CaseSeries] = cases
        self._cache: dict[tuple[str, str], torch.Tensor] = {}
        # 缺省按上游基数构造（评测侧无 config 注入点时的独立使用面；
        # EvaluationPhase 装配时随 config.preprocessing.resize_base 传入）
        self._preprocess = (
            preprocess if preprocess is not None else UpstreamPreprocessChain()
        )

    def case_ids(self) -> list[str]:
        """全部病例 id（排序稳定——参照配对的确定性基础）。"""
        return sorted(self._cases)

    def reference_volume(
        self, condition: str, entry_index: int, source_case: str | None,
    ) -> torch.Tensor:
        """条目的参照体（组1 轮转语义的 BraTS 实现——与原
        ``MilestoneEvaluator._reference_case`` 逐位同语义，职责自评测
        编排下沉至参照库）：锁定病例优先，否则按条目序号在全部病例
        （BraTS 病例四序列齐全——任意病例的该序列都存在）内确定性轮转。"""
        if source_case is not None:
            return self.volume(source_case, condition)
        case_ids = self.case_ids()
        return self.volume(case_ids[entry_index % len(case_ids)], condition)

    def volume(self, case_id: str, condition: str) -> torch.Tensor:
        """病例某序列的**预处理后**影像体 [X, Y, Z]（缓存）。

        ``condition`` 在 BraTS 语义 = 序列名（``Modality`` 四值）。"""
        if case_id not in self._cases:
            raise ValueError(
                f"参照库无病例 {case_id!r}（dataset_root 布局与 prepare 扫描器不符）"
            )
        key = (case_id, condition)
        if key not in self._cache:
            path = self._cases[case_id].series[condition]
            preprocessed = torch.as_tensor(
                self._preprocess(path),
            ).float()[0]  # 链产物 [1, D, H, W] → [X, Y, Z]
            self._cache[key] = preprocessed
        return self._cache[key]


class MrReferenceVolumeStore:
    """MR-RATE 参照影像库：real pool train split 的**条件内**参照卷（#124）。

    两域差异的封装体（契约见 ``ReferenceVolumes``）：

    - **参照卷集 = pool manifest 条目**（病例级 70% train split——与
      BraTS 侧同一泄漏守卫：held-out split 与评估集永不进参照分布）。
      卷 → 条件的映射唯一来源 = 条目的 ``modality``（生成条件键）；
    - **参照轮转按目标条件过滤**：MR 一卷一条件，BraTS 式「全病例池
      轮转」在条件维度上不成立——轮转到非该条件的卷即跨域比较。组1
      条目在**该条件的卷池**内按条目序号确定性轮转；
    - **装载 = 平铺 NIfTI + 逐条件预处理链**：dataset_root 布局
      ``{study_uid}_{series_id}.nii.gz``（``MrRateAssembly._task`` 同款
      落位）；预处理链与 prepare 预编码同口径（强度臂 clip 随 config、
      resize 目标 = 该条件统一网格 ``grid_xyz``）——合成侧（VAE 解码的
      预处理空间）与参照侧同影像空间同强度域；
    - 跨条件取卷显式拒绝：一卷一条件下「卷 × 错条件」只可能是轮转/
      装配错位，静默跨域 FID 在取数面挡下。

    参照卷按 (case, condition) 缓存——每个里程碑取同一批参照体，重复
    评测不重复读盘与预处理。
    """

    def __init__(
        self,
        dataset_root: Path | str,
        pool: LatentManifest,
        vocabulary: MrConditionVocabulary,
        *,
        clip_intensity: bool,
    ) -> None:
        self._root = Path(dataset_root)
        # 逐条件预处理链：resize 目标 = 条件统一网格（绝对目标，RAS 轴序），
        # 与 prepare 侧 ``_chain_for`` 同一构造口径
        self._chains: dict[str, UpstreamPreprocessChain] = {
            spec.name: UpstreamPreprocessChain(
                clip_intensity=clip_intensity,
                target_grid=spec.grid_xyz,
            )
            for spec in vocabulary.conditions
        }
        condition_cases: dict[str, list[str]] = {
            name: [] for name in self._chains
        }
        case_conditions: dict[str, str] = {}
        for entry in pool.entries:
            if entry.modality not in condition_cases:
                raise ValueError(
                    f"参照库条目条件 {entry.modality!r} 不在本域条件"
                    "词汇表（pool manifest 与词汇表工件口径不符）"
                )
            if (
                entry.case_id in case_conditions
                and case_conditions[entry.case_id] != entry.modality
            ):
                raise ValueError(
                    f"参照卷 {entry.case_id!r} 在 pool manifest 中携带"
                    f"两个条件（{case_conditions[entry.case_id]!r} 与 "
                    f"{entry.modality!r}）——一卷一条件契约破坏"
                )
            case_conditions[entry.case_id] = entry.modality
            condition_cases[entry.modality].append(entry.case_id)
        self._condition_cases: dict[str, list[str]] = {
            name: sorted(cases) for name, cases in condition_cases.items()
        }
        self._case_conditions: dict[str, str] = case_conditions
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def reference_volume(
        self, condition: str, entry_index: int, source_case: str | None,
    ) -> torch.Tensor:
        """条目的参照体：锁定卷优先，否则按条目序号在**该条件的卷池**
        内确定性轮转（排序稳定——参照配对的确定性基础）。"""
        if source_case is not None:
            return self.volume(source_case, condition)
        cases = self._condition_cases.get(condition, [])
        if not cases:
            raise ValueError(
                f"参照库条件 {condition!r} 无可用卷（pool manifest 无该"
                "条件的条目——参照分布为空集，里程碑评测不可进行）"
            )
        return self.volume(cases[entry_index % len(cases)], condition)

    def volume(self, case_id: str, condition: str) -> torch.Tensor:
        """指定卷的**预处理后**影像体 [X, Y, Z]（缓存）。

        卷的归属条件与请求条件不符即拒绝（错误消息含两侧条件，供
        轮转错位归因）。"""
        if condition not in self._chains:
            raise ValueError(
                f"参照库请求条件 {condition!r} 不在本域条件词汇表"
            )
        actual = self._case_conditions.get(case_id)
        if actual is None:
            raise ValueError(
                f"参照库无卷 {case_id!r}（参照卷集 = real pool train "
                "split 的病例级白名单，dataset_root 全树不是合法来源）"
            )
        if actual != condition:
            raise ValueError(
                f"参照卷 {case_id!r} 属条件 {actual!r}，与请求条件 "
                f"{condition!r} 不符——一卷一条件下跨条件取卷只可能是"
                "轮转/装配错位"
            )
        key = (case_id, condition)
        if key not in self._cache:
            path = self._root / f"{case_id.replace('/', '_')}.nii.gz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"参照卷影像缺失: {path}（pool manifest 与 "
                    "dataset_root 落位不符——检查数据落位）"
                )
            preprocessed = torch.as_tensor(
                self._chains[condition](path),
            ).float()[0]  # 链产物 [1, D, H, W] → [X, Y, Z]
            self._cache[key] = preprocessed
        return self._cache[key]


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
