"""MR-RATE 工件契约泛化测试（#121/#131，spec #125 实现决策 3）。

分层键泛化（BraTS = 序列 modality、MR-RATE = 生成条件名）下的装载契约：
旧 BraTS 工件零变化照常装载（可扩不可改名），MR 工件按条件分层登记
（逐条件计数 + 逐条件 latent 形状）——异形状贯通（#129）前工件自洽的
单一权威。容量装配守卫按分层计数表泛化（prepare 装配期与 train 装配期
同一条判定路径，ADR-0008 决策 4）。"""

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from cynosure.reward.artifacts import (
    ChannelStats,
    LatentManifest,
    PoolEntry,
    PrepareProvenance,
    SamplingEntry,
    SamplingManifest,
)

LATENT_SHAPE_A = (4, 16, 16, 8)
LATENT_SHAPE_B = (4, 16, 16, 16)


class PoolEntryFactory:
    """两域 manifest 条目的测试构造器（分层键各一：BraTS 序列 / MR 条件）。"""

    @staticmethod
    def brats(case_id: str, modality: str = "t1n") -> PoolEntry:
        return PoolEntry(
            case_id=case_id,
            modality=modality,  # type: ignore[arg-type]
            latent=f"real_pool_latents/{case_id}-{modality}.pt",
            spacing=(100.0, 100.0, 100.0),
        )

    @staticmethod
    def mr(series_id: str, condition: str = "t1w/axial") -> PoolEntry:
        return PoolEntry(
            case_id=f"S{series_id}/{series_id}",
            condition=condition,
            latent=f"real_pool_latents/{condition}/{series_id}.pt",
            spacing=(37.5, 37.5, 50.0),
        )


class TestPoolEntryStratificationKeys:
    """条目分层键（序列 modality / 生成条件 condition）恰一非空。"""

    def test_brats_entry_modality_only(self) -> None:
        entry = PoolEntryFactory.brats("BraTS-00001")
        assert entry.condition is None
        assert entry.stratification_key == "t1n"

    def test_mr_entry_condition_only(self) -> None:
        entry = PoolEntryFactory.mr("S1")
        assert entry.modality is None
        assert entry.stratification_key == "t1w/axial"

    def test_both_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PoolEntry(
                case_id="X",
                modality="t1n",  # type: ignore[arg-type]
                condition="t1w/axial",
                latent="l.pt",
                spacing=(1.0, 1.0, 1.0),
            )

    def test_neither_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PoolEntry(case_id="X", latent="l.pt", spacing=(1.0, 1.0, 1.0))


class TestMrLatentManifest:
    """MR-RATE manifest 装载契约：条件分层计数 + 逐条件形状登记。"""

    @staticmethod
    def manifest(
        entries: list[PoolEntry],
        conditions: dict[str, int],
        condition_shapes: dict[str, tuple],
        **overrides,
    ) -> LatentManifest:
        fields = dict(
            kind="real_pool",
            encoder="synthetic",
            split_seed=0,
            split_sizes={"train": 6, "val": 2, "test": 0},
            entries=entries,
            conditions=conditions,
            condition_shapes=condition_shapes,
        )
        fields.update(overrides)
        return LatentManifest(**fields)

    def test_mr_manifest_derives_conditions(self) -> None:
        manifest = self.manifest(
            [
                PoolEntryFactory.mr("S1"),
                PoolEntryFactory.mr("S2", condition="flair/axial"),
            ],
            conditions={"t1w/axial": 1, "flair/axial": 1},
            condition_shapes={
                "t1w/axial": LATENT_SHAPE_A,
                "flair/axial": LATENT_SHAPE_B,
            },
        )
        assert manifest.modalities == {}
        assert manifest.conditions == {"t1w/axial": 1, "flair/axial": 1}

    def test_mr_manifest_rejects_single_latent_shape(self) -> None:
        """MR 域 manifest 禁止单一 latent_shape：形状权威 = condition_shapes
        （逐条件异形状是 #111 多网格案的落地语义，单值口径会静默错位）。"""
        with pytest.raises(ValidationError) as exc_info:
            self.manifest(
                [PoolEntryFactory.mr("S1")],
                conditions={"t1w/axial": 1},
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
                latent_shape=LATENT_SHAPE_A,
            )
        assert "latent_shape" in str(exc_info.value)

    def test_mr_manifest_requires_shape_registration(self) -> None:
        """条件计数有键而形状登记缺键 → 拒绝：异形状装载校验的对照表
        不完整等于契约缺口。"""
        with pytest.raises(ValidationError) as exc_info:
            self.manifest(
                [
                PoolEntryFactory.mr("S1"),
                PoolEntryFactory.mr("S2", condition="flair/axial"),
            ],
                conditions={"t1w/axial": 1, "flair/axial": 1},
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
            )
        assert "flair/axial" in str(exc_info.value)

    def test_mr_manifest_rejects_stratification_mismatch(self) -> None:
        """传入的 conditions 计数与条目实际分布不符 → 拒绝（同 modalities
        先例）。"""
        with pytest.raises(ValidationError):
            self.manifest(
                [PoolEntryFactory.mr("S1")],
                conditions={"t1w/axial": 3},
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
            )

    def test_mr_manifest_rejects_mixed_domain_entries(self) -> None:
        """BraTS 条目（modality）与 MR 条目（condition）混装 → 拒绝：
        一份 manifest 一个域，混合域分层语义不可判读。"""
        with pytest.raises(ValidationError):
            self.manifest(
                [
                    PoolEntryFactory.mr("S1"),
                    PoolEntryFactory.brats("BraTS-00001"),
                ],
                conditions={"t1w/axial": 1},
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
            )

    def test_brats_manifest_unchanged(self) -> None:
        """BraTS 旧形态零变化：latent_shape 必有、条件字段空——既有
        工件与构造点（test_online_update 等）不受泛化影响。"""
        manifest = LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=LATENT_SHAPE_A,
            split_seed=0,
            split_sizes={"train": 2, "val": 1, "test": 1},
            entries=[
                PoolEntryFactory.brats("C1", "t1n"),
                PoolEntryFactory.brats("C2", "t2w"),
            ],
        )
        assert manifest.modalities == {"t1n": 1, "t2w": 1}
        assert manifest.conditions == {}
        assert manifest.condition_shapes == {}

    def test_brats_manifest_rejects_conditions(self) -> None:
        """BraTS manifest 携带条件分层 → 拒绝（域自洽双向守卫）。"""
        with pytest.raises(ValidationError):
            LatentManifest(
                kind="real_pool",
                encoder="synthetic",
                latent_shape=LATENT_SHAPE_A,
                split_seed=0,
                split_sizes={"train": 1, "val": 1, "test": 1},
                entries=[PoolEntryFactory.brats("C1")],
                conditions={"t1w/axial": 1},
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
            )

    def test_load_latent_per_condition_shape(self, tmp_path: Path) -> None:
        """MR 工件 load_latent 按条件形状校验：异条件异形状各自对照，
        错位形状拒绝（工件自洽在装载端闭环）。"""
        entries = [
            PoolEntryFactory.mr("S1"),
            PoolEntryFactory.mr("S2", condition="flair/axial"),
        ]
        manifest = self.manifest(
            entries,
            conditions={"t1w/axial": 1, "flair/axial": 1},
            condition_shapes={
                "t1w/axial": LATENT_SHAPE_A,
                "flair/axial": LATENT_SHAPE_B,
            },
        )
        manifest_path = tmp_path / "real_pool.json"
        manifest._path = manifest_path
        for entry, shape in (
            (entries[0], LATENT_SHAPE_A), (entries[1], LATENT_SHAPE_B),
        ):
            latent_path = tmp_path / entry.latent
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.zeros(shape), latent_path)
        assert manifest.load_latent(entries[0]).shape == LATENT_SHAPE_A
        assert manifest.load_latent(entries[1]).shape == LATENT_SHAPE_B
        # 形状错位：S1 的文件落成 flair 形状 → 对照条件形状拒绝
        torch.save(torch.zeros(LATENT_SHAPE_B), tmp_path / entries[0].latent)
        with pytest.raises(ValueError) as exc_info:
            manifest.load_latent(entries[0])
        assert "t1w/axial" in str(exc_info.value)

    def test_capacity_guard_per_condition(self) -> None:
        """容量守卫按条件计数表判定（MR 域）：逐条件 ≥ K×world——
        稀疏条件（小池）先饿，报错列出饥荒条件明细。"""
        manifest = self.manifest(
            [PoolEntryFactory.mr(f"S{i}") for i in range(3)],
            conditions={"t1w/axial": 3},
            condition_shapes={"t1w/axial": LATENT_SHAPE_A},
        )
        manifest.assert_condition_capacity(batch_size_k=3, world_size=1)
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(batch_size_k=4, world_size=1)
        assert "t1w/axial×3" in str(exc_info.value)

    def test_capacity_guard_brats_uses_modalities(self) -> None:
        """BraTS manifest 容量守卫照旧按 modalities 判定（回归锚）：
        全集 = MODALITIES 常量（缺序 = 0 条 = 饿死）。"""
        manifest = LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=LATENT_SHAPE_A,
            split_seed=0,
            split_sizes={"train": 4, "val": 1, "test": 1},
            entries=[PoolEntryFactory.brats("C1", m) for m in ("t1n", "t1c", "t2w", "t2f")],
        )
        manifest.assert_condition_capacity(batch_size_k=1, world_size=1)
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(batch_size_k=2, world_size=1)
        assert "t1n×1" in str(exc_info.value)

    def test_capacity_guard_expected_keys_catch_zero_conditions(self) -> None:
        """MR 域传词汇表全集：计数表外 0 条件被抓（装配期全量口径由
        调用方决定——prepare 传 vocab.names() 的语义）。"""
        manifest = self.manifest(
            [PoolEntryFactory.mr(f"S{i}") for i in range(3)],
            conditions={"t1w/axial": 3},
            condition_shapes={"t1w/axial": LATENT_SHAPE_A},
        )
        manifest.assert_condition_capacity(3, 1)  # 缺省：表内键全过
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(
                3, 1,
                expected_keys=("t1w/axial", "flair/axial", "mra/all-planes"),
            )
        assert "flair/axial×0" in str(exc_info.value)


class TestSamplingManifest:
    """配额抽样留痕工件（#131：#78 抽样机制同款、幂等与互斥的可审计落档）。"""

    @staticmethod
    def manifest() -> SamplingManifest:
        return SamplingManifest(
            kind="mrrate-sampling-manifest",
            seed=20260912,
            data_snapshot="MR-RATE@v1.0",
            quota={"t1w/axial": 2},
            heldout_fraction=0.3,
            census_candidates={"t1w/axial": 4},
            census_quota_taken={"t1w/axial": 2},
            heldout_counts={"t1w/axial": 1},
            out_of_vocabulary_volumes=1,
            eval_exclusion_keys=250,
            eval_exclusion_series_hits=0,
            eval_exclusion_patient_hits=0,
            entries=[
                SamplingEntry(
                    patient_uid="P0",
                    study_uid="ST1",
                    series_id="t1w-raw-axi",
                    modality="t1w",
                    plane="axial",
                    condition="t1w/axial",
                    role="pool",
                ),
                SamplingEntry(
                    patient_uid="P1",
                    study_uid="ST2",
                    series_id="t1w-raw-axi",
                    modality="t1w",
                    plane="axial",
                    condition="t1w/axial",
                    role="heldout",
                ),
            ],
        )

    def test_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "sampling_manifest.json"
        path.write_text(self.manifest().model_dump_json(), encoding="utf-8")
        revived = SamplingManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8")),
        )
        assert revived == self.manifest()

    def test_invalid_role_rejected(self) -> None:
        data = self.manifest().model_dump()
        data["entries"][0]["role"] = "train"
        with pytest.raises(ValidationError):
            SamplingManifest.model_validate(data)


class TestChannelStatsProvenance:
    """per-channel 统计量的 provenance 留痕（#121 AC2：来源快照 + 预处理口径）。"""

    def test_provenance_roundtrip(self) -> None:
        stats = ChannelStats(
            mean=[0.1, 0.2, 0.3, 0.4],
            std=[1.0, 1.0, 1.0, 1.0],
            num_latents=4,
            latent_shape=LATENT_SHAPE_A,
            source_manifest="real_pool.json",
            provenance=PrepareProvenance(
                dataset="MR-RATE",
                data_snapshot="MR-RATE@v1.0",
                intensity_clip=False,
                resize_semantics="uniform-grid",
                upstream_anchor="NVIDIA v1 clip=False",
            ),
        )
        revived = ChannelStats.model_validate_json(stats.model_dump_json())
        assert revived.provenance == stats.provenance

    def test_provenance_optional(self) -> None:
        """BraTS 既有 stats 工件（无 provenance）照常装载（可扩不改名）。"""
        stats = ChannelStats(
            mean=[0.0] * 4,
            std=[1.0] * 4,
            num_latents=1,
            latent_shape=LATENT_SHAPE_A,
            source_manifest="real_pool.json",
        )
        assert stats.provenance is None
