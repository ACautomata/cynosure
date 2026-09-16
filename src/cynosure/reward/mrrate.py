"""MR-RATE real 数据链的装配语义（#121/#131，spec #125 实现决策 3）。

BraTS 线的装配 = 病例目录扫描 + 70/10/20 seed 洗牌（``reward.dataset``）；
MR-RATE 线的装配 = 官方 patient 级 split join → 评估集互斥硬守卫 →
train split 内 patient 级 held-out 二分 → 逐条件配额抽样——四步全部
确定性（固定 seed、排序后抽样，#78 抽样机制同款），逐卷归属随抽样
manifest 落档可审计。

与上游 / 数据集的关系：

- **train split 判定 = 官方 splits.csv**（patient 级，同患者所有 study
  同 split；mrrate-data-spec §6：自己乱切会因同患者多 study 泄漏）；
- **评估集互斥**：候选域显式排除 #78 评估清单全部卷（study_uid + series_id
  series 键守卫 + patient 集合第二道防线）——官方 split 下 train 与
  val/test 天然不相交，守卫防的是未来口径漂移（fail-fast 在装配期而非
  训练期）；
- **held-out real 在 train split 内二分**（病例级不相交、按条件分层落档、
  永不参与判别器更新）——与评估留出池（官方 val+test）的不相交由
  train split 边界 + 互斥守卫共同保证（#73 原则、#121 AC3）；
- **配额为上限**：头部模态各数千条、MRA 全量 ≈ 110——候选不足取全量，
  硬下限由装配期容量守卫（``assert_condition_capacity``）把守。

零依赖原则不变：影像路径约定 = ``<dataset_root>/<study_uid>_<series_id>
.nii.gz``（官方 zip 内文件名的平铺落盘布局，#132 生产落位同构）。
"""

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from cynosure.conditions import MrConditionVocabulary
from cynosure.config import CynosureConfig, MODALITIES
from cynosure.reward.artifacts import SamplingEntry, SamplingRole
from cynosure.reward.dataset import BratsSeriesLayout, CaseSplitter


@dataclass
class AssemblyTask:
    """单卷编码任务：装配策略产出、编排骨架消费的计划条目。"""

    case_id: str
    """条目键：BraTS = 病例名；MR-RATE = ``<study_uid>/<series_id>`` 卷键
    （噪声种子内容寻址与 latent 文件名共用）。"""
    stratification: str
    """分层键：BraTS = 序列名；MR-RATE = 生成条件名（manifest 条目序与
    分层计数的依据）。"""
    image_path: Path
    spacing: tuple[float, float, float] | None
    """spacing 值：MR-RATE = 条件属性（等效 spacing ×1e2，同条件严格
    同值）；BraTS = None（编码期读 per-case raw header zooms 侧车）。"""
    target_grid: tuple[int, int, int] | None
    """resize 目标：MR-RATE = 条件统一网格（词汇表携带）；BraTS = None
    （RAS 后逐轴 round 到基数倍数的公式口径）。"""


@dataclass
class SamplingTrace:
    """配额抽样的全程留痕（SamplingManifest 的装配侧数据）——prepare
    幂等与 held-out 互斥的可审计读数。逐卷归属（entries）由装配策略
    生成——它是 pool / held-out 归属的唯一裁决者。"""

    census_candidates: dict[str, int]
    """逐条件候选域计数（互斥守卫后、二分与抽样前）。"""
    census_quota_taken: dict[str, int]
    """逐条件 pool 侧实抽计数（配额为上限——候选不足取全量）。"""
    heldout_counts: dict[str, int]
    """逐条件 held-out 侧计数（per-condition AUC 归因的支撑留痕）。"""
    out_of_vocabulary_volumes: int
    """train split 内白名单条件域外卷数（信息性留痕，不进任何工件）。"""
    non_train_volumes: int
    """val/test split 的元数据卷数（评估留出池，不进 real 数据链候选；
    生产元数据覆盖全 split 的常态——计数留痕供审计）。"""
    eval_exclusion_keys: int
    """评估集互斥守卫的 series 键基数（#78 评估清单行数）。"""
    eval_exclusion_series_hits: int
    """候选域与评估集 series 键的命中数（合法装配恒 0——守卫 fail-fast
    在先，0 读数即守卫已执行的凭据）。"""
    eval_exclusion_patient_hits: int
    """候选域与评估集 patient 集的命中数（第二道防线，合法装配恒 0）。"""
    entries: list[SamplingEntry]
    """逐卷装配归属登记（pool / heldout，条件内排序 + 词汇表条件序）。"""


@dataclass
class AssemblyPlan:
    """一次 prepare 的装配计划（两域共用编排骨架的中间产物）。"""

    pool: list[AssemblyTask]
    """Real sample pool 编码任务（判别器「真」训练侧；per-channel 统计
    量的归约域）。"""
    heldout: list[AssemblyTask]
    """Held-out real 编码任务（out-of-sample 监控侧，永不参与判别器
    更新）。"""
    split_sizes: dict[str, int]
    """split 全貌留痕：BraTS = 病例级三段病例数；MR-RATE = 官方三分
    patient 数。"""
    sampling_trace: SamplingTrace | None = None
    """配额抽样留痕（MR-RATE 域必非 None；BraTS 域 None）。"""

    @property
    def is_mr_rate(self) -> bool:
        return self.sampling_trace is not None


@dataclass
class SeriesRecord:
    """一条候选卷（元数据 join 产物，进抽样前的最小登记）。"""

    patient_uid: str
    study_uid: str
    series_id: str
    modality: str
    plane: str
    condition: str

    @property
    def series_key(self) -> tuple[str, str]:
        """评估集互斥的 series 键（study_uid + series_id，#131 口径）。"""
        return (self.study_uid, self.series_id)

    @property
    def case_id(self) -> str:
        """条目键（latent 文件名与噪声种子派生键）。"""
        return f"{self.study_uid}/{self.series_id}"

    def sort_key(self) -> tuple[str, str]:
        return (self.study_uid, self.series_id)


class MrRateSeriesCatalog:
    """series 级元数据 join 官方 split → train split 候选域。

    列契约（本仓规范列名，#78 eval_manifest 同款键列；官方元数据 CSV 的
    列名映射由数据落位面完成，#132）：

    - metadata CSV：``study_uid`` / ``series_id`` / ``patient_uid`` /
      ``modality`` / ``plane``（必需列，缺列可读拒绝）；
    - splits CSV：``patient_uid`` / ``split``（train/val/test）。
    """

    METADATA_COLUMNS: tuple[str, ...] = (
        "study_uid", "series_id", "patient_uid", "modality", "plane",
    )
    SPLITS_COLUMNS: tuple[str, ...] = ("patient_uid", "split")

    def __init__(
        self, metadata_csv: Path, splits_csv: Path,
        vocabulary: MrConditionVocabulary,
    ) -> None:
        self._metadata_csv = Path(metadata_csv)
        self._splits_csv = Path(splits_csv)
        self._vocabulary = vocabulary

    def split_patients(self) -> dict[str, str]:
        """官方 splits.csv 的 patient → split 映射（patient 级，同患者
        所有 study 同 split——mrrate-data-spec §6；值 casefold 归一）。

        重复 patient 行（值相同或冲突）即拒绝：patient → split 是映射不是
        行表，后行覆盖会让 train 人群随行序漂移（本意 val/test 的患者混进
        real pool），且 ``split_sizes`` 把重复行也计上、留痕虚增。
        """
        mapping: dict[str, str] = {}
        with open(self._splits_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self._assert_columns(
                reader.fieldnames, self.SPLITS_COLUMNS, self._splits_csv,
                "官方 split",
            )
            for row in reader:
                patient = row["patient_uid"].strip()
                part = row["split"].strip().casefold()
                previous = mapping.get(patient)
                if previous is not None:
                    raise ValueError(
                        f"官方 splits.csv 重复登记 patient={patient!r}"
                        f"（已登记 {previous!r}、本行 {part!r}）: "
                        f"{self._splits_csv}——patient → split 是映射不是行表："
                        "重复行让 train 人群随行序漂移（本意 val/test 的患者"
                        "混进 real pool）、split_sizes 把患者数虚增；"
                        "请用去重后的官方 split 文件"
                    )
                mapping[patient] = part
        if not mapping:
            raise ValueError(
                f"官方 splits.csv 无 patient 记录: {self._splits_csv}"
                "（real 数据链的 train split 取数域为空）"
            )
        if "train" not in mapping.values():
            raise ValueError(
                f"官方 splits.csv 无 train split 记录: {self._splits_csv}"
                "（real 数据链只取官方 train split）"
            )
        return mapping

    @staticmethod
    def _assert_columns(
        fieldnames: list[str] | None, required: tuple[str, ...], path: Path,
        label: str,
    ) -> None:
        """CSV 必需列守卫（文件错版走可读拒绝，不裸 KeyError）。"""
        missing = [c for c in required if c not in (fieldnames or [])]
        if missing:
            raise ValueError(
                f"{label} CSV 缺必需列 {missing}: {path}（本仓规范键列，"
                "#78 同款）"
            )

    def train_candidates(self) -> tuple[list[SeriesRecord], int, int]:
        """train split × 生成条件词汇表的候选域（排序后返回）+
        （白名单条件域外卷数, 非 train split 卷数）。

        - patient 悬挂（元数据里的 patient 不在 splits.csv **任何** split
          ——join 完整性破坏，口径漂移与文件错版在这里暴露）→ 可读拒绝、
          不静默丢卷；
        - val/test split 的卷（评估留出池）不进 real 数据链候选，计数
          留痕（合法常态——生产元数据覆盖全 split）；
        - 白名单条件域外的卷（如 swi/sagittal）不进候选，计数留痕；
        - 重复卷键（同一 ``(study_uid, series_id)`` 两行）即拒绝——两行编码
          到同一 latent 路径，manifest 会把同一枚物理卷当两卷计数与采样
          （统计量重复计入、容量守卫可被虚增行数骗过）。

        卷键唯一性按**候选行**把守（非 train / 域外的重复行不进任何工件，
        无实害）；键与条目键、latent 文件名同源（``SeriesRecord.case_id``）。
        """
        splits = self.split_patients()
        candidates: list[SeriesRecord] = []
        seen_keys: set[tuple[str, str]] = set()
        out_of_vocabulary = 0
        non_train = 0
        with open(self._metadata_csv, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self._assert_columns(
                reader.fieldnames, self.METADATA_COLUMNS, self._metadata_csv,
                "MR-RATE 元数据",
            )
            for row in reader:
                patient_uid = row["patient_uid"].strip()
                if patient_uid not in splits:
                    raise ValueError(
                        f"元数据卷 patient={patient_uid!r} 不在官方 "
                        f"splits.csv: {self._splits_csv}（join 完整性破坏"
                        "——元数据与 splits 错版/口径漂移，拒绝装配）"
                    )
                if splits[patient_uid] != "train":
                    non_train += 1  # 评估留出池卷：不进 real 数据链候选
                    continue
                modality = row["modality"].strip()
                plane = row["plane"].strip()
                condition = self._vocabulary.resolve_condition(modality, plane)
                if condition is None:
                    out_of_vocabulary += 1
                    continue
                record = SeriesRecord(
                    patient_uid=patient_uid,
                    study_uid=row["study_uid"].strip(),
                    series_id=row["series_id"].strip(),
                    modality=modality.casefold(),
                    plane=plane.casefold(),
                    condition=condition,
                )
                if record.series_key in seen_keys:
                    raise ValueError(
                        f"MR-RATE 元数据重复登记卷键 {record.case_id!r}: "
                        f"{self._metadata_csv}——两行编码到同一 latent 路径，"
                        "manifest 把同一枚物理卷当两卷计数与采样（per-channel"
                        " 统计量重复计入、逐条件容量守卫可被虚增行数骗过）；"
                        "请用去重后的元数据文件"
                    )
                seen_keys.add(record.series_key)
                candidates.append(record)
        candidates.sort(key=SeriesRecord.sort_key)
        return candidates, out_of_vocabulary, non_train

    def split_sizes(self) -> dict[str, int]:
        """官方三分的 patient 数全貌（manifest split_sizes 留痕）。"""
        counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        with open(self._splits_csv, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                part = row["split"].strip().casefold()
                if part in counts:
                    counts[part] += 1
        return counts


class EvalSetExclusion:
    """评估集互斥硬守卫（#131 AC2）：候选域与 #78 评估清单在 series 键
    （study_uid + series_id）与 patient 集合两个粒度上必须零交集——
    任一命中即 fail-fast（可读报错点名命中卷），不静默剔除。

    官方 split 下 train 与评估留出池（val+test）天然不相交；守卫防的是
    口径漂移（评估清单扩充、split 重划、元数据错版）静默吃掉互斥性。
    """

    EVAL_COLUMNS: tuple[str, ...] = ("study_uid", "series_id", "patient_uid")

    def __init__(self, eval_manifest_csv: Path) -> None:
        self._path = Path(eval_manifest_csv)

    def read(self) -> tuple[set[tuple[str, str]], set[str]]:
        """评估清单的（series 键集, patient 集）。

        空清单（有表头、无数据行）即拒绝：互斥硬守卫会退化为空集比较
        （恒不相交）静默失效，抽样留痕还记下 ``eval_exclusion_keys=0``
        的「已执行」假凭据——清单截断/错版必须在此暴露（同仓 #78 的
        另一读面 ``MrRateEvalManifest.volume_rows`` 对同一形态即拒绝）。
        """
        series_keys: set[tuple[str, str]] = set()
        patients: set[str] = set()
        rows = 0
        with open(self._path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            MrRateSeriesCatalog._assert_columns(
                reader.fieldnames, self.EVAL_COLUMNS, self._path,
                "#78 评估清单",
            )
            for row in reader:
                rows += 1
                series_keys.add(
                    (row["study_uid"].strip(), row["series_id"].strip()),
                )
                patients.add(row["patient_uid"].strip())
        if rows == 0:
            raise ValueError(
                f"#78 评估清单无数据行: {self._path}（评估集互斥硬守卫会"
                "退化为空集比较、恒不相交而静默失效，抽样留痕却记下键基数 "
                "0 的「已执行」假凭据；清单截断/错版即拒绝——train split "
                "与评估留出池的互斥是 #73 原则的硬前提）"
            )
        return series_keys, patients

    def assert_disjoint(
        self, candidates: list[SeriesRecord],
    ) -> tuple[int, int, int]:
        """候选域零交集校验：通过返回守卫读数（键基数、两粒度命中数
        均为 0）随抽样 manifest 落档；命中即可读拒绝。"""
        series_keys, patients = self.read()
        series_hits = sorted(
            record.series_key for record in candidates
            if record.series_key in series_keys
        )
        patient_hits = sorted({
            record.patient_uid for record in candidates
            if record.patient_uid in patients
        })
        if series_hits or patient_hits:
            raise ValueError(
                "评估集互斥破坏：候选域命中 #78 评估清单"
                f"（series 键命中 {series_hits[:5]}、patient 命中 "
                f"{patient_hits[:5]}，清单 {self._path}）——real 数据链与"
                "评估留出池必须病例级不相交（#73 原则）：train split 与"
                "守卫双保险下仍出现交集 = 口径漂移，拒绝装配"
            )
        return len(series_keys), 0, 0


class PatientHeldoutSplit:
    """train split 内 patient 级二分（#125 spec 决策 3、#73 原则）。

    patients 排序 + seed 洗牌 + 按 ``heldout_fraction`` 切片：同 patient
    全部卷同侧（病例级不相交的判定键 = patient_uid）；held-out 侧为空
    （份额过小、patients 过少）显式拒绝——空 held-out 失去 out-of-sample
    信号语义（同 CaseSplitter 空拒绝哲学）。
    """

    def __init__(self, seed: int, fraction: float) -> None:
        self._seed = seed
        self._fraction = fraction

    def split(self, patients: set[str]) -> tuple[set[str], set[str]]:
        """（pool patients, held-out patients）：排序 + seed 洗牌后按份额
        切前段为 held-out。"""
        ordered = sorted(patients)
        random.Random(self._seed).shuffle(ordered)
        num_heldout = round(len(ordered) * self._fraction)
        if num_heldout < 1:
            raise ValueError(
                f"patients {len(ordered)} 按份额 {self._fraction} 切不出"
                "非空 held-out（held-out 为空即失去 out-of-sample 信号"
                "语义；增大 heldout_fraction 或候选 patients）"
            )
        heldout = set(ordered[:num_heldout])
        pool = set(ordered) - heldout
        return pool, heldout


class MrRateAssembly:
    """MR-RATE 域装配策略（prepare 编排骨架的 MR 分派实现）。

    plan() 流程：候选域（catalog）→ 互斥守卫（exclusion）→ patient 二分
    （heldout split）→ pool 侧逐条件配额抽样（排序后 seed 洗牌截取，配额
    为上限）→ 编码任务计划。逐卷归属与全部守卫读数随计划留痕（pipeline
    消费落抽样 manifest）。"""

    def __init__(
        self,
        config: CynosureConfig,
        vocabulary: MrConditionVocabulary,
    ) -> None:
        self._config = config
        self._vocabulary = vocabulary
        unknown_quota = sorted(
            set(config.reward.real_pool_quota) - set(vocabulary.names()),
        )
        if unknown_quota:
            raise ValueError(
                f"real_pool_quota 含词汇表外的条件键 {unknown_quota}：拼错的"
                "条件名会被静默忽略（该条件全量不设限，与「显式错误值即拒」"
                f"哲学不符）；在册条件: {list(vocabulary.names())}"
            )
        self._catalog = MrRateSeriesCatalog(
            config.artifacts.mrrate_metadata_csv,
            config.artifacts.mrrate_splits_csv,
            vocabulary,
        )
        self._exclusion = EvalSetExclusion(config.artifacts.eval_manifest_csv)
        self._heldout_split = PatientHeldoutSplit(
            config.schedule.seed, config.reward.heldout_fraction,
        )

    def plan(self) -> AssemblyPlan:
        candidates, out_of_vocabulary, non_train = (
            self._catalog.train_candidates()
        )
        if not candidates:
            raise ValueError(
                "train split 候选域为空（元数据 × 官方 split × 生成条件"
                "词汇表三重过滤后无候选卷）——检查元数据与 splits 覆盖"
            )
        exclusion_keys, series_hits, patient_hits = (
            self._exclusion.assert_disjoint(candidates)
        )
        patients = {record.patient_uid for record in candidates}
        pool_patients, heldout_patients = self._heldout_split.split(patients)
        pool_records, heldout_records = self._partition_by_patient(
            candidates, heldout_patients,
        )
        pool_selected = self._apply_quota(pool_records)
        ordered_pool = self._ordered(pool_selected)
        ordered_heldout = self._ordered(heldout_records)
        root = Path(self._config.artifacts.dataset_root)
        trace = SamplingTrace(
            census_candidates=self._census(candidates),
            census_quota_taken=self._census(pool_selected),
            heldout_counts=self._census(heldout_records),
            out_of_vocabulary_volumes=out_of_vocabulary,
            non_train_volumes=non_train,
            eval_exclusion_keys=exclusion_keys,
            eval_exclusion_series_hits=series_hits,
            eval_exclusion_patient_hits=patient_hits,
            entries=[
                self._sampling_entry(record, "pool")
                for record in ordered_pool
            ] + [
                self._sampling_entry(record, "heldout")
                for record in ordered_heldout
            ],
        )
        return AssemblyPlan(
            pool=[self._task(record, root) for record in ordered_pool],
            heldout=[self._task(record, root) for record in ordered_heldout],
            split_sizes=self._catalog.split_sizes(),
            sampling_trace=trace,
        )

    @staticmethod
    def _sampling_entry(
        record: SeriesRecord, role: SamplingRole,
    ) -> SamplingEntry:
        return SamplingEntry(
            patient_uid=record.patient_uid,
            study_uid=record.study_uid,
            series_id=record.series_id,
            modality=record.modality,
            plane=record.plane,
            condition=record.condition,
            role=role,
        )

    def _partition_by_patient(
        self,
        candidates: list[SeriesRecord],
        heldout_patients: set[str],
    ) -> tuple[list[SeriesRecord], list[SeriesRecord]]:
        pool: list[SeriesRecord] = []
        heldout: list[SeriesRecord] = []
        for record in candidates:
            target = (
                heldout if record.patient_uid in heldout_patients else pool
            )
            target.append(record)
        return pool, heldout

    def _apply_quota(
        self, pool_records: list[SeriesRecord],
    ) -> list[SeriesRecord]:
        """逐条件配额抽样（#78 机制同款）：条件内排序 + seed 洗牌 + 截取
        配额上限（候选不足取全量；洗牌只决定选谁，条目序由 ``_ordered``
        统一重排保持确定性可读）。"""
        quota = self._config.reward.real_pool_quota
        selected: list[SeriesRecord] = []
        for condition in self._vocabulary.names():
            records = sorted(
                (r for r in pool_records if r.condition == condition),
                key=SeriesRecord.sort_key,
            )
            limit = quota.get(condition)
            if limit is not None:
                shuffled = list(records)
                random.Random(
                    f"{self._config.schedule.seed}|{condition}",
                ).shuffle(shuffled)
                records = sorted(shuffled[:limit], key=SeriesRecord.sort_key)
            selected.extend(records)
        return selected

    def _ordered(self, records: list[SeriesRecord]) -> list[SeriesRecord]:
        """manifest 条目序 = 条件（词汇表登记序）× 卷键排序——与 BraTS
        「（序列, 病例）双键排序」同构的确定性分层顺序。"""
        condition_order = {
            name: index
            for index, name in enumerate(self._vocabulary.names())
        }
        return sorted(
            records,
            key=lambda record: (
                condition_order[record.condition], *record.sort_key(),
            ),
        )

    def _task(self, record: SeriesRecord, root: Path) -> AssemblyTask:
        condition = self._vocabulary.by_name(record.condition)
        image_path = root / f"{record.study_uid}_{record.series_id}.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(
                f"候选卷影像缺失: {image_path}（候选域来自元数据，影像落盘"
                "不完整——检查数据落位）"
            )
        return AssemblyTask(
            case_id=record.case_id,
            stratification=record.condition,
            image_path=image_path,
            # spacing 条件属性（#130 解析面，同条件严格同值）
            spacing=self._vocabulary.spacing_condition(record.condition),
            target_grid=condition.grid_xyz,
        )

    @staticmethod
    def _census(records: list[SeriesRecord]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            counts[record.condition] = counts.get(record.condition, 0) + 1
        return counts


class BratsAssembly:
    """BraTS 域装配策略（prepare 编排骨架的 BraTS 实现）：病例目录扫描 +
    病例级 70/10/20 seed 洗牌（语义从 PreparePipeline 逐位保留——train →
    pool、val → held-out、test 不进工件；manifest 条目序 = （序列, 病例）
    双键排序）。"""

    def __init__(self, config: CynosureConfig) -> None:
        self._config = config

    def plan(self) -> AssemblyPlan:
        cases = BratsSeriesLayout(self._config.artifacts.dataset_root).scan()
        split = CaseSplitter(self._config.schedule.seed).split(
            [case.case_id for case in cases],
        )
        pool_tasks = [
            self._task(case_id, modality)
            for modality in MODALITIES
            for case_id in sorted(split.train)
        ]
        heldout_tasks = [
            self._task(case_id, modality)
            for modality in MODALITIES
            for case_id in sorted(split.val)
        ]
        return AssemblyPlan(
            pool=pool_tasks,
            heldout=heldout_tasks,
            split_sizes=split.sizes(),
        )

    def _task(self, case_id: str, modality: str) -> AssemblyTask:
        return AssemblyTask(
            case_id=case_id,
            stratification=modality,
            image_path=(
                Path(self._config.artifacts.dataset_root) / case_id
                / f"{case_id}-{modality}.nii.gz"
            ),
            spacing=None,
            target_grid=None,
        )
