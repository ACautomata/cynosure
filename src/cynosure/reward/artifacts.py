"""prepare 数据工件契约（spec「产物工件契约」的 reward 侧最小集）。

三个工件均可被 train / eval 装载消费：

- **Real sample pool manifest**（``kind="real_pool"``）：train split 全量
  VAE 预编码 latent 的索引，按序列分层；
- **Held-out real manifest**（``kind="heldout_real"``）：val split 预编码
  latent，与 pool 病例级不相交、永不参与判别器更新；
- **per-channel 标准化统计量**（``kind="channel_stats"``）：判别器输入
  标准化所用 mean/std，来自 Real sample pool 所用训练集。

latent 张量本体不经 JSON：每条目一个 ``torch.save`` 文件，manifest 以相对
路径索引（``PoolEntry.latent``），训练侧可按条目懒加载切片。
字段名为契约最小集：施工可扩不可改名。
"""

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cynosure.config import MODALITIES, Modality
from cynosure.reward.dataset import SplitPart


ManifestKind = Literal["real_pool", "heldout_real"]
"""预编码 latent manifest 的两种语义（pool = 判别器「真」训练侧；held-out =
out-of-sample 监控侧），装载时以 kind 守卫互换使用。"""


class PoolEntry(BaseModel):
    """manifest 条目：一卷预编码 latent + spacing 侧车。

    分层键两域互斥（#121/#131）：BraTS 条目带 ``modality``（序列名），
    MR-RATE 条目带 ``condition``（生成条件名，11 格）——恰好其一非空，
    混合即拒绝；``stratification_key`` 是分层计数与条件匹配采样共用的
    取数键。MR-RATE 卷键 = ``<study_uid>/<series_id>``（case_id 承载）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    case_id: str
    modality: Modality | None = None
    """BraTS 序列名（t1n/t1c/t2w/t2f）；MR-RATE 条目为 None（模态名在
    生成条件内，不单列分层）。"""
    condition: str | None = None
    """MR-RATE 生成条件名（如 t1w/axial，词汇表工件 11 格）；BraTS
    条目为 None。"""
    latent: str
    """latent 文件路径，相对 manifest 文件所在目录。"""
    spacing: tuple[float, float, float]
    """spacing 侧车（×1e2 条件单位）：BraTS = per-case raw header zooms
    （issue #46）；MR-RATE = 条件属性值（等效 spacing = 推荐 FOV / 统一
    网格，spec #125 决策 6——同条件严格同值，不构成判别捷径）。"""

    @field_validator("spacing")
    @classmethod
    def _spacing_positive(
        cls, value: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if any(component <= 0 for component in value):
            raise ValueError(f"spacing 须为正（raw header zooms ×1e2）: {value}")
        return value

    @model_validator(mode="after")
    def _stratification_keys_mutually_exclusive(self) -> "PoolEntry":
        if (self.modality is None) == (self.condition is None):
            raise ValueError(
                "manifest 条目的分层键须恰好其一：modality（BraTS 序列）"
                f"或 condition（MR-RATE 生成条件），得到 modality="
                f"{self.modality} condition={self.condition}"
            )
        return self

    @property
    def stratification_key(self) -> str:
        """分层键：BraTS = 序列名、MR-RATE = 生成条件名——分层计数、
        容量守卫与条件匹配采样的单一取数键。"""
        key = self.condition if self.condition is not None else self.modality
        assert key is not None  # validator 已保证恰一非空
        return key


class LatentManifest(BaseModel):
    """预编码 latent 索引（Real sample pool 与 Held-out real 共用契约，
    以 ``kind`` 区分）。

    分层键两域（#121/#131）：BraTS = 序列（``modalities`` 计数 + 单一
    ``latent_shape`` 契约）；MR-RATE = 生成条件（``conditions`` 计数 +
    ``condition_shapes`` 逐条件形状登记，``latent_shape`` 恒 None——逐
    条件异形状是词汇表统一网格的落地语义，单一形状口径会静默错位）。
    一份 manifest 一个域，混合域分层语义不可判读、显式拒绝。"""

    model_config = ConfigDict(extra="forbid")

    _path: Path | None = PrivateAttr(default=None)
    """manifest 文件自身位置：条目相对路径的解析基准（load 时记录）。"""

    kind: ManifestKind
    encoder: str
    """预编码来源标识（fixture 合成 / 生产 MONAI VAE），随工件留痕——
    消费方可从工件本身区分 latent 出处。"""
    latent_shape: tuple[int, int, int, int] | None = None
    """单一 latent 形状契约（BraTS 域必填，既有工件零变化）；MR-RATE
    域恒 None（形状权威 = condition_shapes 逐条件登记）。"""
    split_seed: int
    split_sizes: dict[SplitPart, int]
    """split 全貌留痕：BraTS = 病例级 70/10/20 三段病例数；MR-RATE =
    官方 split 的 patient 数全貌（real 数据链只消费 train split）。"""
    entries: list[PoolEntry]
    modalities: dict[Modality, int] = Field(default_factory=dict)
    """序列分层计数（BraTS 域）：不传入时由 entries 派生（单一来源），
    传入则须一致。"""
    conditions: dict[str, int] = Field(default_factory=dict)
    """生成条件分层计数（MR-RATE 域）：不传入时由 entries 派生，传入
    则须一致。"""
    condition_shapes: dict[str, tuple[int, int, int, int]] = Field(
        default_factory=dict,
    )
    """逐条件 latent 形状登记（MR-RATE 域必非空且键集 = conditions 键集）
    ——load_latent 的异形状装载校验对照表。"""

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
        """同源 manifest 的条目子集视图：分层计数重派生、装载路径基准
        保留——条带切片（分布式 real 侧）等消费方的装配入口。"""
        sliced = LatentManifest(
            kind=self.kind,
            encoder=self.encoder,
            latent_shape=self.latent_shape,
            split_seed=self.split_seed,
            split_sizes=self.split_sizes,
            entries=entries,
        )
        sliced._path = self._path
        return sliced

    def assert_condition_capacity(
        self, batch_size_k: int, world_size: int = 1,
        *, expected_keys: tuple[str, ...] | None = None,
    ) -> None:
        """逐（分层键, 本 manifest 全量）容量 ≥ ``batch_size_k`` 的
        装配期守卫（ADR-0008 决策 4 / ADR-0008-03）：判别器 real 侧按本
        iteration 条件匹配采样后每分层键独立供满无放回 real 批——判定
        按**全量**做、需量为 ``batch_size_k × world_size``（条带切片
        ``entries[rank::world]`` 下「每 rank 视图 ≥ K」的等价条件；判定
        放全量保失败路径全 rank 一致——与 RankSlicedPool 的切片前校验
        同款理由，按切片后本地视图判定会让失败方单方面退出、其余 rank
        互等）；单进程 world_size=1 判据退化为全池 ≥ K。无放回采样语义
        不动，不引入有放回采样补洞。

        分层键全集（0 条 = 饿死，显式拒绝）：BraTS 域 = ``MODALITIES``
        代码常量（封闭四序列集）；MR-RATE 域 = ``expected_keys``（词汇表
        条件全集，prepare 装配期传入；缺省 = 计数表键集——0 条件不在
        表内的全量口径由调用方决定）。"""
        required = batch_size_k * world_size
        is_mr_rate = bool(self.conditions)
        if is_mr_rate:  # MR-RATE 域：逐条件
            keys = (
                expected_keys if expected_keys is not None
                else tuple(sorted(self.conditions))
            )
            counts = {key: self.conditions.get(key, 0) for key in keys}
        else:  # BraTS 域：封闭四序列全集（0 条模态 = 饿死）
            counts = {m: self.modalities.get(m, 0) for m in MODALITIES}
        starved = [
            (key, count) for key, count in sorted(counts.items())
            if count < required
        ]
        if starved:
            detail = ", ".join(f"{key}×{count}" for key, count in starved)
            scope = "逐条件" if is_mr_rate else "逐模态"
            hint = (
                "增大 real pool 配额" if is_mr_rate else "增大 real pool"
            )
            raise ValueError(
                f"Real sample pool 容量不足：{scope} real 容量须 ≥ "
                f"disc_batch_size_k={batch_size_k} × world_size={world_size}"
                f" = {required} 条（条件匹配采样后每 rank 独立供满无放回 "
                "real 批——ADR-0008 决策 4 装配期守卫；无放回采样语义"
                f"不变，不引入有放回采样补洞）；不足: {detail}。"
                f"{hint}（或减小 disc_batch_size_k / 切片路数）"
            )

    def expected_shape(self, entry: PoolEntry) -> tuple[int, int, int, int]:
        """条目的期望 latent 形状：BraTS = 单一契约；MR-RATE = 条件形状
        登记表（缺登记 = 工件契约缺口，显式失败）。"""
        if entry.condition is not None:
            try:
                return self.condition_shapes[entry.condition]
            except KeyError:
                raise ValueError(
                    f"条件 {entry.condition} 在 condition_shapes 无形状"
                    "登记（MR-RATE 工件契约要求逐条件登记，缺登记 = "
                    "工件不完整）"
                ) from None
        if self.latent_shape is None:
            raise ValueError(
                "本 manifest 缺单一 latent_shape 契约（BraTS 域工件必填）"
            )
        return self.latent_shape

    def load_latent(self, entry: PoolEntry) -> torch.Tensor:
        """装载单条 latent（按条目懒加载，供判别器 real 切片采样）；
        形状按条目域对照（单一契约 / 条件形状登记）。"""
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
        expected = self.expected_shape(entry)
        if tuple(latent.shape) != expected:
            raise ValueError(
                f"latent 形状 {tuple(latent.shape)} 与契约 {expected} 不符:"
                f" {entry.latent}"
            )
        return latent

    @model_validator(mode="after")
    def _stratification_consistent(self) -> "LatentManifest":
        """分层计数派生与两域自洽（域判定 = 条目分层键的构成）：

        - 混合域条目（modality 与 condition 并存于一份 manifest）拒绝；
        - MR-RATE 域（condition 条目）：modalities 恒空、conditions 由
          条目派生（传入则须一致）、condition_shapes 键集 = conditions
          键集、latent_shape 恒 None（单一形状与逐条件异形状矛盾）；
        - BraTS 域（modality 条目）：conditions/condition_shapes 恒空、
          latent_shape 必填、modalities 由条目派生。
        """
        has_condition = [entry.condition is not None for entry in self.entries]
        if any(has_condition) and not all(has_condition):
            raise ValueError(
                "manifest 条目域混合（modality 与 condition 并存）：一份 "
                "manifest 一个数据域，分层语义不可判读"
            )
        if self.entries and all(has_condition):  # MR-RATE 域
            if self.latent_shape is not None:
                raise ValueError(
                    "MR-RATE manifest 禁止单一 latent_shape（逐条件异形状"
                    "由 condition_shapes 登记，单一形状口径会静默错位）"
                )
            if self.modalities:
                raise ValueError(
                    "MR-RATE manifest 的 modalities 计数须为空"
                    f"（序列分层不适用），得到 {self.modalities}"
                )
            counts: dict[str, int] = {}
            for entry in self.entries:
                key = entry.stratification_key
                counts[key] = counts.get(key, 0) + 1
            if self.conditions and self.conditions != counts:
                raise ValueError(
                    f"conditions 计数 {self.conditions} 与条目实际分布 "
                    f"{counts} 不符"
                )
            self.conditions = counts
            missing = sorted(counts.keys() - self.condition_shapes.keys())
            extra = sorted(self.condition_shapes.keys() - counts.keys())
            if missing or extra:
                raise ValueError(
                    "condition_shapes 登记与条件分层不符：缺登记 "
                    f"{missing}、多登记 {extra}（逐条件形状是异形状装载"
                    "校验的对照表，登记必须完备）"
                )
        else:  # BraTS 域
            if self.conditions or self.condition_shapes:
                raise ValueError(
                    "BraTS manifest 携带条件分层（conditions/"
                    "condition_shapes 仅对 MR-RATE 域有语义）"
                )
            if self.latent_shape is None:
                raise ValueError(
                    "BraTS manifest 缺单一 latent_shape 契约（既有工件"
                    "形态必填；逐条件形状登记属 MR-RATE 域）"
                )
            modality_counts: dict[str, int] = {}
            for entry in self.entries:
                key = entry.stratification_key
                modality_counts[key] = modality_counts.get(key, 0) + 1
            if self.modalities and self.modalities != modality_counts:
                raise ValueError(
                    f"modalities 计数 {self.modalities} 与条目实际分布 "
                    f"{modality_counts} 不符"
                )
            self.modalities = modality_counts  # type: ignore[assignment]
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
    latent_shape: tuple[int, int, int, int] | None = None
    """单一 latent 形状（BraTS 域）；MR-RATE 域逐条件异形状、恒 None。"""
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
        if self.latent_shape is not None:
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
    census_candidates: dict[str, int]
    """逐条件候选域计数（互斥守卫后、抽样前；held-out 二分基数的
    同源口径）。"""
    census_quota_taken: dict[str, int]
    """逐条件 pool 侧实抽计数（配额为上限——候选不足取全量）。"""
    heldout_counts: dict[str, int]
    """逐条件 held-out 侧计数（per-condition AUC 归因的支撑留痕）。"""
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
