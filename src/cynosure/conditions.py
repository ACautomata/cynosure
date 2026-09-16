"""MR-RATE 条件词汇表工件（#127，spec #125 实现决策 1 的 prefactor）。

生成条件 = (modality token, plane, 推荐 FOV, 统一网格, 等效 spacing) 的
五元组清单，以工件为唯一来源装载——替代代码内定死的词表常量与 token
映射副本（映射单一来源：条件 token 由工件的 modality_tokens 派生，文件
内不重复登记）。生产工件逐条件对账 #78 普查工件的期望网格（众数），
「条件 → latent 形状」从此有单一权威对照；缺格、网格不符、字段缺失在
装载期被字段级拒绝。

与 BraTS 线 ``ModalityMapping`` 同款机制（工件装载、不设代码内副本）：
config 只携带路径（``artifacts.condition_vocabulary_json``），装载发生在
消费点。fixture 模式（显式声明纪律，同 resize_base 先例）放行小词汇表：
非 11 条件全量的词汇表只能经 ``fixture_mode=True`` 装载。
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cynosure.config import SPACING_CONDITION_SCALE

MrPlane = Literal["axial", "sagittal", "coronal", "all-planes"]
"""生成条件的采集平面；all-planes 是 #81 读数格的并池口径（T2w 三格
并池读数、MRA 全平面一格），ML 层面四值全域，格级允许集由词表定死。"""

MR_MODALITIES: tuple[str, ...] = ("t1w", "t2w", "flair", "swi", "mra")
"""MR-RATE 五模态域名单（模态「名字」的封闭集；token 数值不在本模块
登记——映射唯一来源是工件文件，#127 单一来源验收）。"""

SPACING_DOMAIN_MM: tuple[float, float] = (0.4, 5.0)
"""等效 spacing 的合法物理域（mm/voxel；上游 check_input_mr 的 spacing
∈ [0.4, 5.0] 约束，mrrate-data-spec §4.1）。"""

LATENT_CHANNELS: int = 4
"""latent 通道数（VAE latent_channels=4，config ``latent_shape`` 同锚）。"""

LATENT_SPATIAL_DOWNSAMPLE: int = 4
"""VAE 空间压缩率：latent 空间轴 = 统一网格 / 4（两级 stride-2 下采样；
与 ``LATENT_CHANNELS`` 的同值是巧合不是同一概念——通道数是张量维度，
压缩率是几何换算）。"""

PRODUCTION_CONDITION_COUNT: int = 11
"""生产词汇表的条件基数 = #81 终审白名单全量（10 头部 (modality, plane)
格 + MRA）；缺格、多格即拒绝。"""

GRID_MULTIPLE: int = 32
"""统一网格的逐轴倍数下界（影像域 32 倍数 ⇔ latent 逐轴 8 倍数，4 级
UNet 跳连约束；#80 实测 latent 44 硬错的口径修正）。"""


class MrConditionEntry(BaseModel):
    """条件词汇表工件的一条目：五元组中除 token 外的文件登记形态。

    token 不入条目——由工件 ``modality_tokens`` 按模态派生（单一来源，
    沿用 #119「分组 token 不单独定死」语义）；等效 spacing 亦不入文件
    ——由 FOV / 网格派生（条件属性的定义式，spec #125 决策 6）。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="条件分组键（小写「模态/平面」，如 t1w/axial）")
    modality: str = Field(description="目标模态（token 取数键，须在 modality_tokens 内）")
    plane: MrPlane = Field(description="采集平面（all-planes 为并池口径）")
    fov_mm: tuple[float, float, float] = Field(
        description="推荐 FOV（mm，RAS 轴序；官方表或评估 manifest 中位）",
    )
    fov_source: str = Field(description="FOV 数值来源标注（official / manifest-median 等）")
    grid_xyz: tuple[int, int, int] = Field(
        description="统一网格（影像域，RAS 轴序；= 普查众数 latent 网格 ×4）",
    )

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        """等效 spacing = 推荐 FOV / 统一网格（条件属性的定义式，spec #125
        决策 6）；validator 与装载共用同一定义，不设第二份计算。"""
        return tuple(
            fov / grid for fov, grid in zip(self.fov_mm, self.grid_xyz)
        )


class MrConditionVocabularyManifest(BaseModel):
    """条件词汇表工件的文件 schema（字段级拒绝面：缺字段/多字段即
    pydantic 字段级错误）。模式无关校验在本 validator；生产模式的普查
    对账与 11 条件全量校验在 ``MrConditionVocabulary.load``（fixture_mode
    是装载参数，不属于文件）。"""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["mrrate-condition-vocabulary"] = Field(
        description="工件类型标识（防误装他类工件）",
    )
    issue: str = Field(
        description="工件的施工票溯源标注",
    )
    upstream_reference: str = Field(
        description="上游权威来源标注（modality_mapping.json + 推荐 FOV 表）",
    )
    grid_semantics: str = Field(
        description="统一网格与等效 spacing 的语义说明（普查对账口径）",
    )
    census_grid_csv: str | None = Field(
        default=None,
        description="#78 普查网格分布 CSV 的路径（相对本工件文件）；生产装载必填，"
        "fixture 小词汇表豁免",
    )
    modality_tokens: dict[str, int] = Field(
        description="whole-brain token 映射（上游权威；恰好覆盖五模态）",
    )
    conditions: list[MrConditionEntry] = Field(
        description="生成条件清单（生产模式 = #81 白名单全量 11 个）",
    )

    @model_validator(mode="after")
    def _structure_is_self_consistent(self) -> "MrConditionVocabularyManifest":
        """结构自洽对账（无需外部 IO 的全部校验）：

        - token 映射恰好覆盖五模态（缺模态/多模态可读拒绝）；
        - 条件名唯一且与 (modality, plane) 一致（name 是 per-condition
          统计与轮转的消费键，ADR-0008）；
        - 条件模态在映射内有 token；
        - 统一网格逐轴为 32 倍数（latent 逐轴 8 倍数的 UNet 跳连约束）；
        - FOV 薄轴与网格薄轴同位（轴序登记错误守卫）；
        - 等效 spacing = FOV / 网格落在上游物理域内。
        """
        missing = [m for m in MR_MODALITIES if m not in self.modality_tokens]
        if missing:
            raise ValueError(
                f"modality_tokens 缺少模态 {missing}（须恰好覆盖五模态 "
                f"{MR_MODALITIES}，上游 modality_mapping.json 权威）"
            )
        unexpected = [
            modality for modality in self.modality_tokens if modality not in MR_MODALITIES
        ]
        if unexpected:
            raise ValueError(
                f"modality_tokens 含五模态之外的键 {unexpected}（词表域为 "
                f"{MR_MODALITIES}；skull-stripped 等派生码不进本轮词汇）"
            )
        names = [entry.name for entry in self.conditions]
        if len(names) != len(set(names)):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"条件名重复：{duplicates}（每条件一条目）")
        for entry in self.conditions:
            expected_name = f"{entry.modality}/{entry.plane}"
            if entry.name != expected_name:
                raise ValueError(
                    f"条件名 {entry.name!r} 与 (modality, plane) = "
                    f"({entry.modality!r}, {entry.plane!r}) 不一致：name 须为 "
                    f"{expected_name!r}（分组名是 ADR-0008 按条件统计的键）"
                )
            if entry.modality not in self.modality_tokens:
                raise ValueError(
                    f"条件 {entry.name} 的模态 {entry.modality!r} 在 "
                    "modality_tokens 中无 token（映射取数键缺失）"
                )
            for axis, grid in enumerate(entry.grid_xyz):
                if grid < GRID_MULTIPLE or grid % GRID_MULTIPLE != 0:
                    raise ValueError(
                        f"条件 {entry.name} 统一网格第 {axis} 轴 = {grid}："
                        f"须为 {GRID_MULTIPLE} 的非零倍数（latent 逐轴 8 倍数"
                        "的 UNet 跳连约束，#80 实测口径）"
                    )
            fov_thin = entry.fov_mm.index(min(entry.fov_mm))
            grid_thin = entry.grid_xyz.index(min(entry.grid_xyz))
            if (
                entry.fov_mm.count(min(entry.fov_mm)) == 1
                and fov_thin != grid_thin
            ):
                raise ValueError(
                    f"条件 {entry.name} 的 FOV 薄轴（第 {fov_thin} 轴，"
                    f"{min(entry.fov_mm)} mm）与统一网格薄轴（第 {grid_thin} "
                    f"轴，{min(entry.grid_xyz)}）错位：RAS 轴序登记自洽破坏"
                )
            low, high = SPACING_DOMAIN_MM
            out_of_domain = [
                (axis, value) for axis, value in enumerate(entry.spacing_mm)
                if not low <= value <= high
            ]
            if out_of_domain:
                raise ValueError(
                    f"条件 {entry.name} 等效 spacing = {list(entry.spacing_mm)} "
                    f"mm/voxel，第 {out_of_domain[0][0]} 轴越出上游物理域 "
                    f"{SPACING_DOMAIN_MM}（FOV / 统一网格；上游 check_input_mr "
                    "约束）"
                )
        return self

@dataclass(frozen=True)
class MrConditionSpec:
    """一个生成条件的五元组（装载产物值对象，token 已派生、spacing
    已按定义式计算）。

    组1 采样的条件单位：rollout 条件 label 取 ``token``、初始噪声形状取
    ``latent_grid``（#129 消费面）、预处理统一网格取 ``grid_xyz``、
    spacing 条件属性取 ``spacing_mm``（#130/#131 消费面）。
    """

    name: str
    modality: str
    plane: MrPlane
    token: int
    fov_mm: tuple[float, float, float]
    grid_xyz: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]

    @property
    def latent_grid(self) -> tuple[int, int, int]:
        """latent 空间网格 = 统一网格 / 压缩率（VAE 空间压缩）。"""
        return tuple(axis // LATENT_SPATIAL_DOWNSAMPLE for axis in self.grid_xyz)


class MrConditionVocabulary:
    """装载后的条件词汇表（不可变视图；每次 ``load`` 独立实例，同进程
    先后装载互不污染）。"""

    def __init__(
        self,
        conditions: tuple[MrConditionSpec, ...],
        tokens: dict[str, int],
    ) -> None:
        self._conditions = tuple(conditions)
        self._tokens: MappingProxyType[str, int] = MappingProxyType(dict(tokens))
        self._by_name = {condition.name: condition for condition in self._conditions}

    @classmethod
    def load(
        cls, path: str | Path, *, fixture_mode: bool = False,
    ) -> "MrConditionVocabulary":
        """装载词汇表工件。

        - **生产模式**（默认）：条件集 = #81 白名单全量 11 个，且逐条件
          网格与 #78 普查工件期望网格（众数）一致；缺格、多格、网格
          不符、普查引用缺失/不可读均拒绝。
        - **fixture 模式**（``fixture_mode=True`` 显式声明）：小词汇表
          通道——条件集非空即可，普查对账豁免（fixture 网格无普查对应）。
          生产模式装载到非全量词表即拒绝：缩小词汇表属 fixture，同
          ``resize_base`` 的显式声明纪律。
        """
        manifest_path = Path(path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = MrConditionVocabularyManifest.model_validate(data)
        if not fixture_mode:
            cls._audit_production(manifest, manifest_path)
        conditions = tuple(
            MrConditionSpec(
                name=entry.name,
                modality=entry.modality,
                plane=entry.plane,
                token=manifest.modality_tokens[entry.modality],
                fov_mm=entry.fov_mm,
                grid_xyz=entry.grid_xyz,
                spacing_mm=entry.spacing_mm,
            )
            for entry in manifest.conditions
        )
        return cls(conditions, manifest.modality_tokens)

    @classmethod
    def census_latent_modes(cls, census_path: Path) -> dict[str, tuple[int, ...]]:
        """#78 普查工件的逐条件期望网格导出：每 stratum 取 n_volumes
        最大的 latent 网格（评估集分布的众数，轴序为 native resize 后的
        多重集语义——Orientationd 只置换/翻转，与 RAS 序的多重集等值）。

        生产装载的权威对照输入（``_audit_production`` 消费）。"""
        counts: dict[str, dict[tuple[int, ...], int]] = {}
        with open(census_path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                stratum = row["stratum"]
                grid = tuple(
                    int(part) for part in row["latent_grid_xyz_ch4"].split("x")[1:]
                )
                counts.setdefault(stratum, {})[grid] = int(row["n_volumes"])
        return {
            stratum.casefold(): max(per_grid, key=lambda grid: per_grid[grid])
            for stratum, per_grid in counts.items()
        }

    @classmethod
    def _audit_production(
        cls, manifest: MrConditionVocabularyManifest, manifest_path: Path,
    ) -> None:
        """生产模式对账：#81 白名单全量 + #78 普查期望网格一致。"""
        if manifest.census_grid_csv is None:
            raise ValueError(
                f"生产装载（fixture_mode=false）须携带普查引用 "
                f"census_grid_csv（#78 普查工件对账，{manifest_path}）："
                "条件 → latent 形状的权威对照不可缺席"
            )
        census_path = (manifest_path.parent / manifest.census_grid_csv).resolve()
        modes = cls.census_latent_modes(census_path)
        vocabulary_strata = {
            condition.name.casefold() for condition in manifest.conditions
        }
        missing = sorted(
            stratum for stratum in modes.keys() - vocabulary_strata
        )
        extra = sorted(vocabulary_strata - modes.keys())
        if missing or extra:
            raise ValueError(
                "条件词汇表与 #78 普查工件的格集不一致："
                f"工件缺格 {missing}、多格 {extra}（普查 stratum 集 = "
                f"{sorted(modes.keys())}，#81 白名单全量 "
                f"{PRODUCTION_CONDITION_COUNT} 格）"
            )
        if len(manifest.conditions) != PRODUCTION_CONDITION_COUNT:
            raise ValueError(
                f"生产条件词汇表定死 #81 白名单全量 "
                f"{PRODUCTION_CONDITION_COUNT} 条件（10 头部 (modality, plane) "
                f"格 + MRA），得到 {len(manifest.conditions)} 个；缩小词汇表"
                "属 fixture，须经 fixture_mode=true 显式装载"
            )
        for entry in manifest.conditions:
            mode = modes[entry.name.casefold()]
            expected = sorted(axis * LATENT_SPATIAL_DOWNSAMPLE for axis in mode)
            actual = sorted(entry.grid_xyz)
            if actual != expected:
                raise ValueError(
                    f"条件 {entry.name} 统一网格 {tuple(actual)} 与 #78 普查"
                    f"期望网格（众数 latent {mode} ×"
                    f"{LATENT_SPATIAL_DOWNSAMPLE} = {tuple(expected)}）不一致："
                    "条件 → latent 形状以普查工件为权威对照（装载期校验，"
                    "spec #125 决策 1）"
                )

    @property
    def conditions(self) -> tuple[MrConditionSpec, ...]:
        """全部生成条件（工件登记序）。"""
        return self._conditions

    @property
    def tokens(self) -> MappingProxyType[str, int]:
        """whole-brain token 映射（只读视图；唯一来源是工件）。"""
        return self._tokens

    def names(self) -> tuple[str, ...]:
        """条件名清单（分组键；per-condition 统计与轮转的取数面）。"""
        return tuple(condition.name for condition in self._conditions)

    def by_name(self, name: str) -> MrConditionSpec:
        """按分组键取条件（缺名即 KeyError，错误信息列出在册条件）。"""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"条件 {name!r} 不在词汇表（在册：{list(self._by_name)}）"
            ) from None

    def latent_shape(self, name: str) -> tuple[int, int, int, int]:
        """条件的 latent 形状 (4, X, Y, Z)：rollout 初始噪声与判别器
        输入的形状解析来源（#129 消费面）。"""
        return (LATENT_CHANNELS, *self.by_name(name).latent_grid)

    def spacing_condition(self, name: str) -> tuple[float, float, float]:
        """条件的 spacing 条件张量值（等效 spacing ×1e2，与 per-case 侧车
        同一换算因子与条件单位）：real 侧 manifest 条目与 fake 侧 rollout
        条件张量的同值来源（#130 消费面；spec #125 决策 6——spacing 是
        条件属性而非逐卷侧车，值只依赖条件名，同条件任意两卷严格同值，
        堵死「spacing 差异」判别捷径）。"""
        i, j, k = (
            value * SPACING_CONDITION_SCALE
            for value in self.by_name(name).spacing_mm
        )
        return (i, j, k)
