"""源数据集扫描与病例级划分（experiment-design「real 样本库」节）。

- **BraTS2023 布局**：``dataset_root/<case_id>/<case_id>-<modality>.nii.gz``，
  每病例四序列（t1n/t1c/t2w/t2f）齐全；
- **病例级 70/10/20**：real = train split（70%）全量预编码，held-out =
  val split（10%），test split（20%）不进 prepare 工件；
- **确定性划分**：病例名排序后按 ``schedule.seed`` 洗牌再切片——prepare
  幂等（重跑工件零漂移）的前提，seed 落入 manifest 留痕。
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cynosure.config import MODALITIES, Modality

SERIES_SUFFIX = {modality: f"-{modality}.nii.gz" for modality in MODALITIES}

SplitPart = Literal["train", "val", "test"]
"""病例级三段名（manifest split_sizes 的键；Literal 钉死防拼错键）。"""


@dataclass
class CaseSeries:
    """一病例的四序列文件路径。"""

    case_id: str
    series: dict[Modality, Path]


@dataclass
class CaseSplit:
    """病例级三段划分：train → Real sample pool，val → Held-out real，
    test → 不进 prepare 工件（评测域归属 eval/基座数据侧）。"""

    train: list[str]
    val: list[str]
    test: list[str]

    def sizes(self) -> dict[SplitPart, int]:
        """三段病例数（split 全貌留痕，随 manifest 落盘）。"""
        return {
            "train": len(self.train), "val": len(self.val), "test": len(self.test),
        }


class BratsSeriesLayout:
    """BraTS2023 病例目录布局：扫描 dataset_root、校验四序列齐全。"""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def scan(self) -> list[CaseSeries]:
        """按病例名排序返回全部病例；缺序列或空数据集显式失败
        （不静默产出残缺工件）。"""
        if not self._root.is_dir():
            raise FileNotFoundError(f"源数据集根目录不存在: {self._root}")
        cases: list[CaseSeries] = []
        for case_dir in sorted(path for path in self._root.iterdir() if path.is_dir()):
            series = {
                modality: case_dir / f"{case_dir.name}{suffix}"
                for modality, suffix in SERIES_SUFFIX.items()
            }
            missing = [
                modality for modality, path in series.items() if not path.is_file()
            ]
            if missing:
                raise ValueError(
                    f"病例 {case_dir.name} 缺序列 {missing}"
                    f"（BraTS2023 四序列布局: <case>-<modality>.nii.gz）"
                )
            cases.append(CaseSeries(case_id=case_dir.name, series=series))
        if not cases:
            raise FileNotFoundError(f"源数据集根目录下无病例目录: {self._root}")
        return cases


class CaseSplitter:
    """病例级 70/10/20 确定性划分（experiment-design：BraTS 病例级 70/10/20）。

    排序 + 按种子洗牌 + 切片：同一数据集与 seed 必得同一划分；held-out
    （val）为空即失去 out-of-sample 信号语义，显式拒绝。
    """

    TRAIN_FRACTION = 0.7
    VAL_FRACTION = 0.1

    def __init__(self, seed: int) -> None:
        self._seed = seed

    def split(self, case_ids: list[str]) -> CaseSplit:
        cases = sorted(case_ids)
        random.Random(self._seed).shuffle(cases)
        num_train = round(len(cases) * self.TRAIN_FRACTION)
        num_val = round(len(cases) * self.VAL_FRACTION)
        if num_train < 1 or num_val < 1:
            raise ValueError(
                f"病例数 {len(cases)} 切不出非空 train/val split"
                f"（70/10/20，seed={self._seed}）：held-out 为空即失去"
                " out-of-sample 信号语义"
            )
        return CaseSplit(
            train=cases[:num_train],
            val=cases[num_train:num_train + num_val],
            test=cases[num_train + num_val:],
        )
