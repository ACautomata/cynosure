"""prepare 数据工件契约（spec「产物工件契约」的 reward 侧最小集）。

三个工件均可被 train / eval 装载消费：

- **Real sample pool manifest**（``kind="real_pool"``）：train split 全量
  VAE 预编码 latent 的索引，按条件分层（BraTS = 序列；MR-RATE = 生成
  条件名）；
- **Held-out real manifest**（``kind="heldout_real"``）：val split 预编码
  latent，与 pool 病例级不相交、永不参与判别器更新；
- **per-channel 标准化统计量**（``kind="channel_stats"``）：判别器输入
  标准化所用 mean/std，来自 Real sample pool 所用训练集。

latent 张量本体不经 JSON：每条目一个 ``torch.save`` 文件，manifest 以相对
路径索引（``PoolEntry.latent``），训练侧可按条目懒加载切片。
字段名为契约最小集：施工可扩不可改名。

异形状口径（#129「latent 形状按条件贯通」）：MR-RATE 线的同条件 latent
同形状（统一网格）、跨条件异形状——manifest 携带逐条件形状契约
（``condition_latent_shapes``，条件名 → latent 形状），装载期逐条目
对账；BraTS 单域不带该字段（单形状 = 单条件词汇特例，全局
``latent_shape`` 对账照旧）。
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cynosure.reward.dataset import SplitPart

if TYPE_CHECKING:
    # 条件词汇表协议（本域条件集与形状的权威）；仅作类型标注使用——
    # 运行时 import 会让工件契约反向依赖词汇表模块（消费方向是
    # 词汇表 → 工件装载面）
    from cynosure.conditions import ConditionVocabulary


ManifestKind = Literal["real_pool", "heldout_real"]
"""预编码 latent manifest 的两种语义（pool = 判别器「真」训练侧；held-out =
out-of-sample 监控侧），装载时以 kind 守卫互换使用。"""


class PoolEntry(BaseModel):
    """manifest 条目：一病例一条件的一枚预编码 latent + spacing 侧车。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    case_id: str
    modality: str
    """条件键（BraTS = 序列名；MR-RATE = 生成条件名，如 t1w/axial）——
    判别器条件匹配采样与回放条件过滤的归因轴；同条件 latent 同形状
    （#129，装载期对账）。"""
    latent: str
    """latent 文件路径，相对 manifest 文件所在目录。"""
    spacing: tuple[float, float, float]
    """per-case spacing 侧车（issue #46）：raw NIfTI header zooms ×1e2——
    prepare 从 header 读出、随条目可审计；rollout 组2 源影像条件的
    spacing tensor 按条目取值（BraTS 1mm iso → (100.0, 100.0, 100.0)，
    值来自数据而非写死常量）。"""

    @field_validator("spacing")
    @classmethod
    def _spacing_positive(
        cls, value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError(f"spacing 须为正（raw header zooms ×1e2）: {value}")
        return value


class LatentManifest(BaseModel):
    """预编码 latent 索引（Real sample pool 与 Held-out real 共用契约，
    以 ``kind`` 区分）。

    形状契约两态（#129）：单域（BraTS）全局 ``latent_shape`` 对账——单
    条件词汇特例；多条件（MR-RATE）逐条件 ``condition_latent_shapes``
    对账——同条件同形状、跨条件异形状。两态互斥（携带逐条件表即拒绝
    再依赖全局形状对账，防两口径静默分叉）。"""

    model_config = ConfigDict(extra="forbid")

    _path: Path | None = PrivateAttr(default=None)
    """manifest 文件自身位置：条目相对路径的解析基准（load 时记录）。"""

    kind: ManifestKind
    encoder: str
    """预编码来源标识（fixture 合成 / 生产 MONAI VAE），随工件留痕——
    消费方可从工件本身区分 latent 出处。"""
    latent_shape: tuple[int, int, int, int]
    split_seed: int
    split_sizes: dict[SplitPart, int]
    """病例级 70/10/20 的三段病例数（train/val/test），split 全貌留痕。"""
    entries: list[PoolEntry]
    modalities: dict[str, int] = Field(default_factory=dict)
    """条件分层计数：不传入时由 entries 派生（单一来源），传入则须一致。"""
    condition_latent_shapes: dict[str, tuple[int, int, int, int]] | None = None
    """逐条件 latent 形状契约（MR-RATE 多条件工件携带，#129）：条件名 →
    latent 形状；装载期逐条目对账（条件形状与条目 latent 不符即拒绝——
    防「网格差异」捷径在 real 侧静默混入）。BraTS 单域工件不带（全局
    ``latent_shape`` 对账即单条件词汇特例）。"""

    @classmethod
    def load(cls, path: Path, kind: ManifestKind) -> "LatentManifest":
        """装载并校验 kind：train 侧不得把 held-out manifest 当 pool 消费
        （held-out 永不参与判别器更新，在装载层守住）。"""
        path = Path(path)
        manifest = cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if manifest.kind != kind:
            raise ValueError(
                f"工件 kind 不符：期望 {kind}，{path} 实为 {manifest.kind}"
            )
        manifest._path = path
        return manifest

    def with_entries(self, entries: list[PoolEntry]) -> "LatentManifest":
        """同源 manifest 的条目子集视图：modalities 分层计数重派生、
        装载路径基准保留——条带切片（分布式 real 侧）等消费方的装配入口。"""
        sliced = LatentManifest(
            kind=self.kind,
            encoder=self.encoder,
            latent_shape=self.latent_shape,
            split_seed=self.split_seed,
            split_sizes=self.split_sizes,
            entries=entries,
            condition_latent_shapes=self.condition_latent_shapes,
        )
        sliced._path = self._path
        return sliced

    def assert_condition_capacity(
        self, batch_size_k: int, world_size: int,
        conditions: list[str] | tuple[str, ...],
    ) -> None:
        """逐（目标条件, 本 manifest 全量）容量 ≥ ``batch_size_k`` 的
        装配期守卫（ADR-0008 决策 4 / ADR-0008-03）：判别器 real 侧按本
        iteration 条件匹配采样后每条件独立供满无放回 real 批——判定
        按**全量**做、需量为 ``batch_size_k × world_size``（条带切片
        ``entries[rank::world]`` 下「每 rank 视图 ≥ K」的等价条件；判定
        放全量保失败路径全 rank 一致——与 RankSlicedPool 的切片前校验
        同款理由，按切片后本地视图判定会让失败方单方面退出、其余 rank
        互等）；单进程 world_size=1 判据退化为全池 ≥ K。无放回采样语义
        不动，不引入有放回采样补洞。条件集经注入（#129：BraTS = 四序列、
        MR-RATE = 词汇表条件集），本类不设代码内副本。
        """
        self.assert_capacity_counts(
            self.modalities, batch_size_k, world_size, conditions,
        )

    @classmethod
    def assert_capacity_counts(
        cls, counts: dict[str, int], batch_size_k: int, world_size: int,
        conditions: list[str] | tuple[str, ...],
    ) -> None:
        """容量守卫的算术本体（``assert_condition_capacity`` 与 prepare
        装配计划期守卫共用的唯一实现）：manifest 侧传自身分层计数，
        prepare 侧传装配计划计数——**编码之前**即可判定（生产体量下先
        编码整池再报错是白烧加速卡；spec「开工前失败而非训练中途」的
        字面口径），两入口的判据与报错不设第二份。
        """
        required = batch_size_k * world_size
        starved: list[tuple[str, int]] = []
        for condition in conditions:
            count = counts.get(condition, 0)
            if count < required:
                starved.append((condition, count))
        if starved:
            detail = ", ".join(
                f"{condition}×{count}" for condition, count in starved
            )
            raise ValueError(
                f"Real sample pool 容量不足：逐条件 real 容量须 ≥ "
                f"disc_batch_size_k={batch_size_k} × world_size={world_size}"
                f" = {required} 条（条件匹配采样后每 rank 独立供满无放回 "
                "real 批——ADR-0008 决策 4 装配期守卫；无放回采样语义"
                f"不变，不引入有放回采样补洞）；不足: {detail}。"
                "增大 real pool（或减小 disc_batch_size_k / 切片路数）"
            )

    def assert_condition_shapes(
        self, vocabulary: "ConditionVocabulary",
    ) -> None:
        """逐条件形状契约与**活动词汇表**的装配期对照（#129 消费侧守卫）：
        携带逐条件表的工件须与本域词汇表逐条件同形——同名异形（词表工件
        改动/换域而 manifest 未重建，或 manifest 来自另一词表）在装配期
        显式拒绝，而非首次判别器拼接 real 与 fake 时才炸（fake 侧形状经
        ``vocabulary.latent_shape(name)`` 解析、real 侧经本表解析，两来源
        不一致即错位对；全卷积判别器对此形状差异不报错）。单域（BraTS）
        工件不带逐条件表：全局 ``latent_shape`` 对账 = 单条件词汇特例，
        本守卫不适用（缺表即返回）。缺表放行仅限单条件域（
        ``vocabulary.single_condition``）：多条件域缺表即拒绝——
        ``load_latent`` 会静默回退全局对账，异形条件在 real 采样/gate
        重算期才炸、同形条件带着错误的全局口径静默入训。
        """
        if self.condition_latent_shapes is None:
            if not vocabulary.single_condition:
                raise ValueError(
                    f"多条件域的 {self.kind} 工件须携带逐条件形状契约 "
                    "condition_latent_shapes：缺表则 load_latent 静默回退"
                    "全局 latent_shape 对账——异形条件在判别器 real 采样/"
                    "gate 重算期才炸、同形条件带着错误的全局口径静默入训；"
                    "请按当前条件词汇表重建 manifest（携带逐条件契约）后入训"
                )
            return
        expected = {
            name: vocabulary.latent_shape(name) for name in vocabulary.names()
        }
        drifted = [
            f"{name}（工件 {list(shape)} ≠ 词汇表 "
            f"{list(expected[name]) if name in expected else '未在册'}）"
            for name, shape in self.condition_latent_shapes.items()
            if expected.get(name) != shape
        ]
        missing = [
            name for name in expected
            if name not in self.condition_latent_shapes
        ]
        if drifted or missing:
            raise ValueError(
                f"manifest 逐条件形状契约与活动词汇表不符（kind={self.kind}）: "
                + "; ".join(
                    drifted + [f"缺条件 {name}" for name in missing],
                )
                + "——形状的权威是本域条件词汇表（fake 侧同源），"
                "同名异形会让判别器把两套影像空间的样本拼进同一批；"
                "请按当前词表重建 manifest 后入训"
            )

    def load_latent(self, entry: PoolEntry) -> torch.Tensor:
        """装载单条 latent（按条目懒加载，供判别器 real 切片采样）。
        形状对账两态（#129）：逐条件表携带时按条目条件查表对账，否则
        全局 ``latent_shape`` 对账（BraTS 单域）。"""
        if self._path is None:
            raise ValueError(
                "本 manifest 非经 load() 装载，无条目路径解析基准"
                "（落盘侧用 save()，装载侧一律走 load()）"
            )
        if entry not in self.entries:
            raise ValueError(f"条目不属于本 manifest: {entry}")
        latent = torch.load(
            self._path.parent / entry.latent,
            map_location="cpu", weights_only=True,
        )
        if self.condition_latent_shapes is not None:
            if entry.modality not in self.condition_latent_shapes:
                raise ValueError(
                    f"manifest 逐条件形状契约缺条目条件 {entry.modality!r}"
                    f"（在册：{sorted(self.condition_latent_shapes)}；"
                    "条件名拼错或契约不全即拒绝——逐条件对账不容静默回退"
                    "全局形状）"
                )
            expected = self.condition_latent_shapes[entry.modality]
        else:
            expected = self.latent_shape
        if tuple(latent.shape) != expected:
            raise ValueError(
                f"latent 形状 {tuple(latent.shape)} 与契约 {expected} 不符"
                f"（条件 {entry.modality!r}）: {entry.latent}"
            )
        return latent

    @model_validator(mode="after")
    def _modalities_from_entries(self) -> "LatentManifest":
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.modality] = counts.get(entry.modality, 0) + 1
        if self.modalities and self.modalities != counts:
            raise ValueError(
                f"modalities 计数 {self.modalities} 与条目实际分布 {counts} 不符"
            )
        self.modalities = counts
        return self

    @model_validator(mode="after")
    def _condition_shapes_cover_entries(self) -> "LatentManifest":
        """逐条件形状契约的装载期对账（#129）：契约须覆盖全部在册条件、
        通道数与全局 latent_shape 的通道数一致（4）；契约外条件（条目
        有条件名而契约无键）在 load_latent 时逐条拒绝（此处给全量
        概览式拒绝，fail-fast）。"""
        if self.condition_latent_shapes is None:
            return self
        entry_conditions = {entry.modality for entry in self.entries}
        missing = sorted(entry_conditions - set(self.condition_latent_shapes))
        if missing:
            raise ValueError(
                f"逐条件形状契约缺条目条件 {missing}（在册："
                f"{sorted(self.condition_latent_shapes)}；MR 多条件工件的"
                "条件集须覆盖全部条目——防条件名拼错静默按全局形状对账）"
            )
        for name, shape in self.condition_latent_shapes.items():
            if shape[0] != self.latent_shape[0]:
                raise ValueError(
                    f"条件 {name!r} 的形状 {shape} 通道数与全局契约通道数 "
                    f"{self.latent_shape[0]} 不一致（latent 通道数定死 4）"
                )
        return self


class PrepareProvenance(BaseModel):
    """prepare 工件的 provenance 留痕（#121 AC2：来源快照 + 预处理口径
    随工件落盘——统计量可追溯其数据域、快照与编码链口径）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    dataset: str
    """数据域登记名（REGISTERED_DATASETS）。"""
    data_snapshot: str | None = None
    """数据 release 快照标识（MR-RATE = 与评估集 #78 同一冻结快照；
    BraTS 线 None）。"""
    source_commit: str | None = None
    """来源 commit（产出工件的代码版本标识，config 显式声明注入；None =
    未声明——集群 rsync 部署无 .git，不设运行时自读的隐式通道）。"""
    intensity_clip: bool
    """强度臂口径（BraTS True = ADR-0006 fork 锚；MR-RATE False = NVIDIA
    v1 官方口径，#71/#130 裁决）。"""
    resize_semantics: Literal["formula-round-base", "uniform-grid"]
    """resize 口径：BraTS = RAS 后逐轴 round 到基数倍数；MR-RATE = 逐条件
    统一网格（#111 网格裁决的多网格案落地，与 rollout 口径一致）。"""
    upstream_anchor: str
    """上游锚标注（recipe 级分线：BraTS 锚 fork、MR-RATE 锚 NVIDIA v1，
    CONTEXT.md「上游锚」词条）。"""


class ChannelStats(BaseModel):
    """判别器输入 per-channel 标准化统计量（来自 Real sample pool 所用训练集）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["channel_stats"] = "channel_stats"
    mean: list[float]
    std: list[float]
    num_latents: int
    latent_shape: tuple[int, int, int, int]
    source_manifest: str
    """来源 pool manifest 的路径（相对本文件）。"""
    provenance: PrepareProvenance | None = None
    """数据域 + 快照 + 预处理口径（#121 AC2；既有 BraTS 工件无此字段
    照常装载——可扩不可改名）。"""

    @field_validator("std")
    @classmethod
    def _std_non_negative(cls, value: list[float]) -> list[float]:
        if any(component < 0 for component in value):
            raise ValueError(f"std 不得为负: {value}")
        return value

    @model_validator(mode="after")
    def _channels_match_shape(self) -> "ChannelStats":
        num_channels = self.latent_shape[0]
        if len(self.mean) != num_channels or len(self.std) != num_channels:
            raise ValueError(
                f"mean/std 长度必须等于 latent 通道数 {num_channels}，"
                f"得到 mean={len(self.mean)} std={len(self.std)}"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        return cls.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8")),
        )


SamplingRole = Literal["pool", "heldout"]
"""抽样条目的装配归属：pool = 判别器「真」训练侧；heldout = out-of-sample
监控侧（train split 内 patient 级二分，永不参与判别器更新）。"""


class SamplingEntry(BaseModel):
    """配额抽样留痕的单卷条目（#131：#78 抽样机制同款的逐卷归属登记）。"""

    model_config = ConfigDict(extra="forbid")

    patient_uid: str
    study_uid: str
    series_id: str
    modality: str
    """模态名（casefold 后的元数据原值，MR-RATE 五模态域）。"""
    plane: str
    """采集平面（casefold 后的元数据原值；MRA 卷不作条件判定依据）。"""
    condition: str
    """归属生成条件（词汇表 11 格名）。"""
    role: SamplingRole


class SamplingManifest(BaseModel):
    """MR-RATE 配额抽样留痕工件（#131，#78 抽样机制同款）：固定 seed、
    排序后抽样、逐卷归属与守卫读数落档——prepare 幂等（同 seed 重跑
    零漂移）与 held-out 互斥（病例级不相交、与评估留出池不相交）的
    可审计登记面。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["mrrate-sampling-manifest"] = "mrrate-sampling-manifest"
    seed: int
    """抽样与二分 seed（= schedule.seed，随工件留痕）。"""
    data_snapshot: str
    """数据 release 快照标识（与评估集同一冻结快照，#78 口径一致）。"""
    quota: dict[str, int]
    """逐条件配额上限（config reward.real_pool_quota 原样留痕；未登记
    条件 = 全量不设限）。"""
    heldout_fraction: float
    """train split 内 patient 级二分的 held-out 份额。"""
    heldout_quota_volumes: int
    """逐条件 held-out 侧卷数上限（config reward.heldout_quota_volumes
    原样留痕——held-out 池是监控集，预训练按条件读全量卷级聚类，
    无上限即按 10% patient 二分的体量膨胀）。"""
    census_candidates: dict[str, int]
    """逐条件候选域计数（互斥守卫后、抽样前；held-out 二分基数的
    同源口径）。"""
    census_quota_taken: dict[str, int]
    """逐条件 pool 侧实抽计数（配额为上限——候选不足取全量）。"""
    heldout_counts: dict[str, int]
    """逐条件 held-out 侧**实取**计数（配额为上限——候选不足取全量；
    支撑度判定与 per-condition AUC 归因的留痕）。"""
    out_of_vocabulary_volumes: int
    """train split 内白名单条件域外卷数（信息性留痕，不进任何工件）。"""
    non_train_volumes: int = 0
    """val/test split 的元数据卷数（评估留出池，不进 real 数据链候选；
    生产元数据覆盖全 split 的常态——计数留痕供审计）。"""
    eval_exclusion_keys: int
    """评估集互斥守卫的 series 键基数（#78 评估清单行数）。"""
    eval_exclusion_series_hits: int
    """候选域与评估集 series 键（study_uid, series_id）的命中数——合法
    装配恒 0（守卫 fail-fast 在先），非 0 读数即守卫未执行。"""
    eval_exclusion_patient_hits: int
    """候选域与评估集 patient 集的命中数（第二道防线，合法装配恒 0）。"""
    entries: list[SamplingEntry]
