"""run 目录样本图版面（issue #295，样本图版 CLI 的 run 目录输入面）：
读 manifest 与 config 快照，spacing 经条件词汇表逐条件解析（MR-RATE
线逐条件等效 spacing；BraTS 线单位间距各向同性），每条件按 manifest
序确定性选取（缺省每条件 1 张、数量可调、``all`` 全量），baseline/
resample 两 stage/phase 版本枚举，产物落 run 目录 plots/ 附属产物位之
下 ``samples/{stage}-{phase}/``（文件名 ``{index:04d}.png``，#293 落位
契约）。同 run 同参数重打同产物集合。

未填充样本路径的条目跳过并进报告（stdout 汇总是 CLI 面职责），不产出
空图版；条件不在词汇表、MR 线出现 pair 条件等解析失败给可读错误。train
正常收尾的自动出图挂接（#296）复用本面（``RunSamplePlates.render``
单一编排缝），缺失版告警取数面 = ``RunPlateReport.missing_groups``。
信息面板承载条件与索引、noise seed（组2 另有锁定源病例），内容随
``PlateLayout.info_lines`` 进布局元数据断言面。
"""

import shutil
from pathlib import Path

from pydantic import BaseModel

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.conditions import MrConditionVocabulary
from cynosure.samples import (
    PlateLayout,
    RunEntryVolumeSource,
    SamplePlateRenderer,
    SPACING_ORIGIN_LABELS,
)
from cynosure.train.artifacts import (
    BaselineManifest,
    ManifestEntry,
    RunArtifacts,
)

PHASES: tuple[str, ...] = ("baseline", "resample")
"""run 目录条目的两 stage/phase 版本枚举序（manifest 样本路径的填充侧）。"""


class SkippedSample(BaseModel):
    """未填充样本路径而跳过的 manifest 条目（stdout 汇总的数据面）。"""

    stage: int
    phase: str
    index: int
    condition: str | list[str]


class RunPlateGroup(BaseModel):
    """一个 (stage, phase) 版本的图版产物组（确定性 label = 目录名）。"""

    label: str
    stage: int
    phase: str
    plates: list[PlateLayout]


class RunPlateReport(BaseModel):
    """run 目录一键产出的报告（产物集合 + 跳过汇总，断言面）。"""

    output_root: Path
    stages: list[int]
    """manifest 涉及的采样 stage（升序）——版本完整性的预期面：每个
    stage 的 baseline/resample 两版都是预期产出（train 收尾挂接的缺失
    告警取数面；个别条目未填充在 skipped，整版无样本才缺组）。"""
    groups: list[RunPlateGroup]
    skipped: list[SkippedSample]

    @property
    def produced(self) -> int:
        return sum(len(group.plates) for group in self.groups)

    def missing_groups(self) -> list[str]:
        """预期 stage × phase 版本中未产出组的 label（``{stage}-{phase}``，
        升序）——某版样本全未落盘时该版缺组（train 收尾挂接据此告警）。"""
        produced_labels = {group.label for group in self.groups}
        return [
            f"{stage}-{phase}"
            for stage in self.stages
            for phase in PHASES
            if f"{stage}-{phase}" not in produced_labels
        ]

    def summary_line(self) -> str:
        """stdout 汇总行（CLI 面与 train 收尾挂接共用的单一消息源）：
        产出路径 + 逐组计数 + spacing 口径。"""
        counts = "、".join(
            f"{group.label} {len(group.plates)} 张" for group in self.groups
        )
        origin = SPACING_ORIGIN_LABELS[self.spacing_origin]
        return (
            f"样本图版已落盘: {self.output_root}"
            f"（{counts}；spacing 口径: {origin}）"
        )

    @property
    def spacing_origin(self) -> str | None:
        """本 run 图版的 spacing 口径标记（全部图版同源；无产出为 None）。

        CLI stdout 读数的取数面——口径标签的单一读取点，CLI 不深挖
        groups[0].plates[0] 链。"""
        for group in self.groups:
            for plate in group.plates:
                return plate.spacing_origin
        return None

    def skipped_by_group(self) -> dict[str, int]:
        """跳过条目按组 label（``{stage}-{phase}``）的计数（升序；stdout
        汇总的取数面）。"""
        counts: dict[str, int] = {}
        for skip in self.skipped:
            label = f"{skip.stage}-{skip.phase}"
            counts[label] = counts.get(label, 0) + 1
        return dict(sorted(counts.items()))

    def group(self, label: str) -> RunPlateGroup:
        """按确定性 label（``{stage}-{phase}`` 目录名）取产物组。"""
        for group in self.groups:
            if group.label == label:
                return group
        raise KeyError(
            f"无 {label!r} 图版组（产出组：{[group.label for group in self.groups]}）"
        )


class ConditionSpacingResolver:
    """manifest 条目条件 → 物理 mm spacing 解析器（#295）：MR-RATE 线经
    config 快照声明的条件词汇表工件逐条件取等效 spacing（
    ``MrConditionSpec.spacing_mm`` = FOV / 统一网格）；BraTS 线取单位
    间距各向同性常量（预处理重采样网格近似各向同性的物理口径，组2
    pair 条件与组1 同网格）。两线都是词汇表锚定的物理口径——与散文件
    ``.pt`` 的体素等比降级不同源，装载期失败可读拒绝。"""

    ISOTROPIC_MM: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """BraTS 线的物理 spacing 口径：组1 spacing 条件常量
    ``CONDITION_SPACING_X1E2``（×1e2 条件单位）的 mm 原值——预处理重采样
    网格近似各向同性的词汇锚定常量（非散文件 ``.pt`` 的体素等比降级）。"""

    def __init__(self, config: CynosureConfig) -> None:
        self._is_mr = config.experiment.dataset == "MR-RATE"
        self._vocabulary = None
        if self._is_mr:
            path = config.artifacts.condition_vocabulary_json
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"条件词汇表工件不存在: {path}"
                    "（config 快照 artifacts.condition_vocabulary_json 声明；"
                    "spacing 解析的取数域缺失）"
                )
            self._vocabulary = MrConditionVocabulary.load(
                path, fixture_mode=config.fixture_mode,
            )

    def spacing_mm(
        self, condition: str | list[str],
    ) -> tuple[float, float, float]:
        if not self._is_mr:
            return self.ISOTROPIC_MM
        if not isinstance(condition, str):
            raise ValueError(
                f"MR-RATE 线 manifest 条目条件须为单条件名，得到 {condition!r}"
                "（组2 有序对只属 BraTS 线）"
            )
        try:
            spec = self._vocabulary.by_name(condition)
        except KeyError as exc:
            raise ValueError(str(exc)) from None
        return tuple(spec.spacing_mm)


class RunSamplePlates:
    """run 目录一键样本图版产出（#295）：manifest 条目 → 渲染就绪体 →
    落位契约目录的图版产物集合 + 报告。渲染复用 ``SamplePlateRenderer``
    （体来源注入 ``RunEntryVolumeSource``），本类只承载 run 目录侧的
    装载、确定性选取与落位编排。"""

    def __init__(self, run_root: Path) -> None:
        self._paths = RunArtifacts.layout(Path(run_root))

    def render(
        self,
        per_condition: int | None = 1,
        slice_offset: int = 0,
        window: tuple[float, float] | None = None,
    ) -> RunPlateReport:
        """一键产出全部样本图版：``per_condition`` = 每条件选取张数
        （None = 全量），切层偏移与窗同散文件面语义（逐图版生效）。"""
        manifest = BaselineManifest.load(self._paths.manifest)
        config = ConfigLoader.load(self._paths.config_snapshot)
        resolver = ConditionSpacingResolver(config)
        stages = sorted({entry.stage for entry in manifest.entries})
        groups: list[RunPlateGroup] = []
        skipped: list[SkippedSample] = []
        for stage in stages:
            for phase in PHASES:
                skipped.extend(
                    SkippedSample(
                        stage=entry.stage, phase=phase,
                        index=entry.index, condition=entry.condition,
                    )
                    for entry in self._unfilled(manifest, stage, phase)
                )
                selected = self._select(manifest, stage, phase, per_condition)
                if not selected:
                    continue
                group_dir = self._paths.plots / "samples" / f"{stage}-{phase}"
                if group_dir.exists():
                    # 组目录重建语义：跨参数重打不残留上轮 PNG（产物集合
                    # = 报告集合，确定性引用锚）
                    shutil.rmtree(group_dir)
                renderer = SamplePlateRenderer(group_dir)
                groups.append(RunPlateGroup(
                    label=f"{stage}-{phase}", stage=stage, phase=phase,
                    plates=[
                        self._render_entry(
                            renderer, entry, phase, resolver,
                            slice_offset, window,
                        )
                        for entry in selected
                    ],
                ))
        return RunPlateReport(
            output_root=self._paths.plots / "samples",
            stages=stages,
            groups=groups,
            skipped=skipped,
        )

    @staticmethod
    def _condition_key(condition: str | list[str]) -> object:
        """条件 → 分组键（pair 条件元组化、单条件名原样；_select 的单一
        归一点）。"""
        return tuple(condition) if isinstance(condition, list) else condition

    @staticmethod
    def _sample_path(entry: ManifestEntry, phase: str) -> str | None:
        """条目在本 phase 的已填充样本路径（显式分支，不字符串拼属性）。"""
        if phase == "baseline":
            return entry.baseline_sample
        return entry.resample_sample

    @staticmethod
    def _select(
        manifest: BaselineManifest,
        stage: int,
        phase: str,
        per_condition: int | None,
    ) -> list[ManifestEntry]:
        """每条件按 manifest 序确定性选取：词汇表序逐条件取前 N 张已
        填充条目（N=None 全量）——同 run 同参数必得同选取集。词汇表外
        条目是 manifest 自洽破坏，可读拒绝（不静默丢弃）。"""
        keys = [
            RunSamplePlates._condition_key(condition)
            for condition in manifest.conditions
        ]
        by_key: dict[object, list[ManifestEntry]] = {}
        for entry in manifest.entries:
            if entry.stage != stage or not RunSamplePlates._sample_path(entry, phase):
                continue
            key = RunSamplePlates._condition_key(entry.condition)
            if key not in keys:
                raise ValueError(
                    f"manifest 条件 {entry.condition!r} 不在清单词汇表 "
                    f"{manifest.conditions}（条目与词汇表自洽破坏）"
                )
            by_key.setdefault(key, []).append(entry)
        return [
            entry
            for key in keys
            for entry in (
                by_key.get(key, [])
                if per_condition is None
                else by_key.get(key, [])[:per_condition]
            )
        ]

    @staticmethod
    def _unfilled(
        manifest: BaselineManifest, stage: int, phase: str,
    ) -> list[ManifestEntry]:
        return [
            entry for entry in manifest.entries
            if entry.stage == stage
            and not RunSamplePlates._sample_path(entry, phase)
        ]

    def _render_entry(
        self,
        renderer: SamplePlateRenderer,
        entry: ManifestEntry,
        phase: str,
        resolver: ConditionSpacingResolver,
        slice_offset: int,
        window: tuple[float, float] | None,
    ) -> PlateLayout:
        relative = RunSamplePlates._sample_path(entry, phase)
        volume = RunEntryVolumeSource(
            self._paths.root / relative,
            spacing=resolver.spacing_mm(entry.condition),
        ).load()
        return renderer.render(
            volume, slice_offset=slice_offset, window=window,
            caption=self._caption(entry),
        )

    @staticmethod
    def _caption(entry: ManifestEntry) -> list[str]:
        """信息面板溯源行：条件与索引、noise seed（组2 锁定源病例语义
        另有 source case 行）；内容随布局元数据 info_lines 断言。"""
        condition = (
            " → ".join(entry.condition)
            if isinstance(entry.condition, list)
            else entry.condition
        )
        lines = [
            f"condition: {condition}",
            f"index: {entry.index} (stage {entry.stage})",
            f"noise seed: {entry.noise_seed}",
        ]
        if entry.source_case is not None:
            lines.append(f"source case: {entry.source_case}")
        return lines
