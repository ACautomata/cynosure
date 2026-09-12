"""real-vs-real 地板的病例级半分互比工具（wayfinder #79：fork
``scripts/split_real_real_halves.py`` 的口径移植）。

地板语义（#73 裁决八）：同仪器下 real_half_a vs real_half_b 的 FID，
作为「该仪器在该数据上的噪声下限」——没有地板，绝对 FID 无法解读
（MAISI v1 论文 Table 2 的方法学）。拆分在**病例级**：同一病例的
全部卷留在同一半（fork 为 BraTS 病例；MR-RATE 的病例 = patient_uid），
半分经 seed 冻结（``SplitFreezeRecord`` 落盘 = 冻结工件，重算必须
同 seed 同 manifest），逐格展开成 per-stratum 双侧清单，直接喂
``MrFidInstrument``（每对同格清单 = 一次地板对比）。

与 fork 原版的差异（都是输入适配，拆分口径不变）：

- 病例来源 = MR-RATE 评估 manifest（#78 的 ``eval_manifest.csv``），
  病例键 = ``patient_uid``（fork 读 BraTS dataset.json 的 validation
  键）；
- per-label 展开 = manifest 的 ``stratum`` 列（T1w/AXIAL 等 10 头部格
  + MRA/ALL-PLANES；fork 为 BraTS 四序列后缀映射）；
- 卷路径按模板从 manifest 行展开，默认
  ``{batch_id}/{study_uid}/img/{study_uid}_{series_id}.nii.gz``（
  gauss 落位 ``/data72/junran/mrrate_eval_volumes/`` 的相对路径，
  research/mrrate-data-spec.md §zip 形态）。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH_TEMPLATE = "{batch_id}/{study_uid}/img/{study_uid}_{series_id}.nii.gz"
"""MR-RATE study zip 内的影像相对布局（mrrate-data-spec.md）：提取到
评估卷根目录后去掉 ``mri/`` 前缀即此形态。"""

DEFAULT_SEED = 42
"""fork ``DEFAULT_SEED`` 逐字继承。"""


@dataclass(frozen=True)
class SplitFreezeRecord:
    """冻结的半分指派：seed + 两半 + 病例来源（fork 逐字移植）。

    落盘一次、永不重算（#73：地板数字冻结复用）；复现 = 同 seed +
    同 manifest。
    """

    seed: int
    half_a: list[str]
    half_b: list[str]
    validation_source: str

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(asdict(self), file, indent=2)

    @classmethod
    def load(cls, path: Path) -> SplitFreezeRecord:
        with path.open() as file:
            payload = json.load(file)
        return cls(
            seed=payload["seed"],
            half_a=payload["half_a"],
            half_b=payload["half_b"],
            validation_source=payload["validation_source"],
        )


class HalvesSplitter:
    """seed 冻结的病例级半分：确定性、互斥、覆盖全部病例（fork 逐字移植）。

    half_a = seed 洗牌后前 ceil(n/2) 个；重复病例显式拒绝（输入数据
    契约违反，不静默拆半）。
    """

    def split(self, cases: list[str], seed: int) -> tuple[list[str], list[str]]:
        """病例清单 → (half_a, half_b)。入参顺序无关（内部先排序再洗牌）。"""
        if len(set(cases)) != len(cases):
            raise ValueError("病例清单含重复病例（病例级半分的前提破坏）")
        shuffled = sorted(cases)
        random.Random(seed).shuffle(shuffled)
        midpoint = (len(shuffled) + 1) // 2
        return shuffled[:midpoint], shuffled[midpoint:]


class MrRateEvalManifest:
    """MR-RATE 评估 manifest（eval_manifest.csv，#78 工件）的读取面。

    病例键 = ``patient_uid``（MR-RATE 的「病例」；同一患者的全部
    study/series 必须同半——对应 fork 的「同一 subject 的全部模态
    卷同半」）。``stratum`` 列即地板展开的格标签（#73 裁决三的
    读数粒度）。
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """manifest 文件路径（冻结记录的 validation_source）。"""
        return self._path

    def volume_rows(self) -> list[dict[str, str]]:
        """全部卷行（manifest 数据行，键 = 表头）。"""
        if not self._path.is_file():
            raise FileNotFoundError(f"评估 manifest 不存在: {self._path}")
        with self._path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"评估 manifest 无数据行: {self._path}")
        return rows

    def cases(self) -> list[str]:
        """全部病例（patient_uid 去重，排序）。"""
        return sorted({row["patient_uid"] for row in self.volume_rows()})


class FloorFilelistWriter:
    """冻结清单的落盘：两半的病例清单 + 逐格卷清单（per-label 展开）。

    产出直接是 ``MrFidConfig`` 的 ``real_filelist``/``synth_filelist``
    输入——每对同格 ``filelist_half_a_<格>.txt`` vs
    ``filelist_half_b_<格>.txt`` 构成一次地板对比。
    """

    def __init__(self, output_dir: Path, path_template: str) -> None:
        self._output_dir = output_dir
        self._template = path_template

    def write(self, record: SplitFreezeRecord, rows: list[dict[str, str]]) -> list[Path]:
        """病例半分 + 全部卷行 → 落盘，返回写出的清单路径。"""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        record.save(self._output_dir / "split_record.json")
        written: list[Path] = []
        for name, cases in (("a", record.half_a), ("b", record.half_b)):
            written.append(self._write_lines(f"filelist_half_{name}.txt", cases))
            case_set = set(cases)
            strata = sorted({row["stratum"] for row in rows})
            for stratum in strata:
                lines = [
                    self._expand(row)
                    for row in rows
                    if row["stratum"] == stratum
                    and row["patient_uid"] in case_set
                ]
                written.append(self._write_lines(
                    f"filelist_half_{name}_{self._stratum_token(stratum)}.txt",
                    lines,
                ))
        return written

    def _expand(self, row: dict[str, str]) -> str:
        try:
            return self._template.format(**row)
        except KeyError as exc:
            raise ValueError(
                f"路径模板 {self._template!r} 含 manifest 没有的字段: {exc}"
            ) from exc

    @staticmethod
    def _stratum_token(stratum: str) -> str:
        """格标签 → 文件名形态：路径分隔符 ``/`` 换 ``_``（格标签的其余
        字符（如 MRA 的连字符）保留，人读清单名友好）。"""
        return stratum.replace("/", "_")

    def _write_lines(self, filename: str, lines: list[str]) -> Path:
        path = self._output_dir / filename
        with path.open("w") as file:
            file.writelines(f"{line}\n" for line in lines)
        return path


class RealRealFloorSplit:
    """地板半分的执行编排：manifest → 病例级 seed 半分 → 冻结落盘。

    单次调用的全部产物（冻结记录 + 双侧清单族）在同一个输出目录，
    目录即冻结工件——#80 的地板对比按格取清单对喂
    ``MrFidInstrument``。
    """

    def __init__(
        self,
        manifest_path: Path,
        output_dir: Path,
        seed: int = DEFAULT_SEED,
        path_template: str = DEFAULT_PATH_TEMPLATE,
    ) -> None:
        self._manifest = MrRateEvalManifest(manifest_path)
        self._writer = FloorFilelistWriter(output_dir, path_template)
        self._seed = seed
        self._output_dir = output_dir

    def run(self) -> SplitFreezeRecord:
        """执行半分并落盘；返回冻结记录（两半名单 + seed + 来源）。"""
        rows = self._manifest.volume_rows()
        cases = sorted({row["patient_uid"] for row in rows})
        half_a, half_b = HalvesSplitter().split(cases, self._seed)
        record = SplitFreezeRecord(
            seed=self._seed,
            half_a=half_a,
            half_b=half_b,
            validation_source=str(self._manifest.path),
        )
        self._writer.write(record, rows)
        return record


__all__ = [
    "DEFAULT_PATH_TEMPLATE",
    "DEFAULT_SEED",
    "FloorFilelistWriter",
    "HalvesSplitter",
    "MrRateEvalManifest",
    "RealRealFloorSplit",
    "SplitFreezeRecord",
]
