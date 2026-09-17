"""MR-RATE 工件契约测试（#121/#131，spec #125 实现决策 3）。

条件键两域同名（#129 统一面：BraTS = 序列名、MR-RATE = 生成条件名）下
prepare 工件的装载契约：旧 BraTS 工件零变化照常装载（可扩不可改名），
MR 工件按条件分层登记（``modalities`` 计数 + ``condition_latent_shapes``
逐条件形状契约）。容量装配守卫经条件集注入（ADR-0008 决策 4）。
"""

import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from cynosure.config import MODALITIES
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
    """两域 manifest 条目的测试构造器（条件键各取一格：BraTS 序列 /
    MR 生成条件——字段同为 ``modality``）。"""

    @staticmethod
    def brats(case_id: str, modality: str = "t1n") -> PoolEntry:
        return PoolEntry(
            case_id=case_id,
            modality=modality,
            latent=f"real_pool_latents/{case_id}-{modality}.pt",
            spacing=(100.0, 100.0, 100.0),
        )

    @staticmethod
    def mr(series_id: str, condition: str = "t1w/axial") -> PoolEntry:
        return PoolEntry(
            case_id=f"S{series_id}/{series_id}",
            modality=condition,
            latent=(
                f"real_pool_latents/{condition.replace('/', '_')}/{series_id}.pt"
            ),
            spacing=(37.5, 37.5, 50.0),
        )


class TestConditionKey:
    """条目条件键（#129 统一面）：两域同为 ``modality`` 字段，语义按域
    解释；键必填、未知字段拒绝。"""

    def test_brats_entry_carries_series_name(self) -> None:
        assert PoolEntryFactory.brats("BraTS-00001").modality == "t1n"

    def test_mr_entry_carries_condition_name(self) -> None:
        assert PoolEntryFactory.mr("S1").modality == "t1w/axial"

    def test_missing_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PoolEntry(case_id="X", latent="l.pt", spacing=(1.0, 1.0, 1.0))

    def test_unknown_field_rejected(self) -> None:
        """旧双字段形态（condition=）已收敛为条件键单字段：携带即拒绝。"""
        with pytest.raises(ValidationError):
            PoolEntry(
                case_id="X", modality="t1w/axial", condition="t1w/axial",
                latent="l.pt", spacing=(1.0, 1.0, 1.0),
            )


class TestMrLatentManifest:
    """MR-RATE manifest 装载契约：逐条件计数派生 + 逐条件形状契约对账。"""

    @staticmethod
    def manifest(
        entries: list[PoolEntry],
        condition_shapes: dict[str, tuple],
        **overrides,
    ) -> LatentManifest:
        fields = dict(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=LATENT_SHAPE_A,
            split_seed=0,
            split_sizes={"train": 6, "val": 2, "test": 0},
            entries=entries,
            condition_latent_shapes=condition_shapes,
        )
        fields.update(overrides)
        return LatentManifest(**fields)

    def test_mr_manifest_derives_condition_counts(self) -> None:
        manifest = self.manifest(
            [
                PoolEntryFactory.mr("S1"),
                PoolEntryFactory.mr("S2", condition="flair/axial"),
            ],
            condition_shapes={
                "t1w/axial": LATENT_SHAPE_A,
                "flair/axial": LATENT_SHAPE_B,
            },
        )
        assert manifest.modalities == {"t1w/axial": 1, "flair/axial": 1}

    def test_count_mismatch_rejected(self) -> None:
        """传入的 modalities 计数与条目实际分布不符 → 拒绝。"""
        with pytest.raises(ValidationError, match="modalities 计数"):
            self.manifest(
                [PoolEntryFactory.mr("S1")],
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
                modalities={"t1w/axial": 3},
            )

    def test_shape_contract_must_cover_entries(self) -> None:
        """条目条件不在逐条件形状契约内 → 拒绝：条件名拼错会静默回退
        全局形状对账，异形条件到采样期才炸。"""
        with pytest.raises(ValidationError) as exc_info:
            self.manifest(
                [PoolEntryFactory.mr("S1", condition="flair/axial")],
                condition_shapes={"t1w/axial": LATENT_SHAPE_A},
            )
        assert "flair/axial" in str(exc_info.value)

    def test_condition_shape_channel_count_checked(self) -> None:
        """逐条件形状的通道数须与全局 latent_shape 通道数一致（latent
        通道数定死 4）——异通道数即拒绝。"""
        with pytest.raises(ValidationError) as exc_info:
            self.manifest(
                [PoolEntryFactory.mr("S1")],
                condition_shapes={"t1w/axial": (8, 16, 16, 8)},
            )
        assert "通道数" in str(exc_info.value)

    def test_brats_manifest_unchanged(self) -> None:
        """BraTS 旧形态零变化：全局 latent_shape 必有、不带逐条件表——
        既有工件与构造点（test_online_update 等）不受泛化影响。"""
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
        assert manifest.condition_latent_shapes is None

    def test_load_latent_per_condition_shape(self, tmp_path: Path) -> None:
        """MR 工件 load_latent 按条件形状校验：异条件异形状各自对照，
        错位形状拒绝（工件自洽在装载端闭环）。"""
        entries = [
            PoolEntryFactory.mr("S1"),
            PoolEntryFactory.mr("S2", condition="flair/axial"),
        ]
        manifest = self.manifest(
            entries,
            condition_shapes={
                "t1w/axial": LATENT_SHAPE_A,
                "flair/axial": LATENT_SHAPE_B,
            },
        )
        manifest._path = tmp_path / "real_pool.json"
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

    def test_roundtrip_preserves_condition_contract(self, tmp_path: Path) -> None:
        """落盘 → 装载往返保持逐条件契约（工件可被 reward 管线装载，AC1）。"""
        manifest = self.manifest(
            [PoolEntryFactory.mr("S1")],
            condition_shapes={"t1w/axial": LATENT_SHAPE_A},
        )
        path = tmp_path / "real_pool.json"
        path.write_text(manifest.model_dump_json(), encoding="utf-8")
        revived = LatentManifest.load(path, "real_pool")
        assert revived.condition_latent_shapes == {"t1w/axial": LATENT_SHAPE_A}
        assert revived.modalities == {"t1w/axial": 1}


class TestCapacityGuard:
    """容量装配守卫（ADR-0008 决策 4）：条件集经注入（BraTS = 四序列常量、
    MR-RATE = 词汇表条件集），本类不设代码内副本。"""

    def test_mr_guard_per_condition(self) -> None:
        manifest = TestMrLatentManifest.manifest(
            [PoolEntryFactory.mr(f"S{i}") for i in range(3)],
            condition_shapes={"t1w/axial": LATENT_SHAPE_A},
        )
        manifest.assert_condition_capacity(3, 1, ("t1w/axial",))
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(4, 1, ("t1w/axial",))
        assert "t1w/axial×3" in str(exc_info.value)

    def test_guard_catches_zero_condition_off_table(self) -> None:
        """条件集含计数表外条件（本域语料里 0 条）→ 饿死拒绝：prepare
        装配期传 ``vocabulary.names()`` 的语义（稀疏模态小池触发口径）。"""
        manifest = TestMrLatentManifest.manifest(
            [PoolEntryFactory.mr(f"S{i}") for i in range(3)],
            condition_shapes={"t1w/axial": LATENT_SHAPE_A},
        )
        manifest.assert_condition_capacity(3, 1, ("t1w/axial",))  # 表内键全过
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(
                3, 1, ("t1w/axial", "flair/axial", "mra/all-planes"),
            )
        assert "flair/axial×0" in str(exc_info.value)

    def test_guard_brats_uses_four_series(self) -> None:
        """BraTS 线回归锚：条件集 = MODALITIES 常量（缺序 = 0 条 = 饿死）。"""
        manifest = LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=LATENT_SHAPE_A,
            split_seed=0,
            split_sizes={"train": 4, "val": 1, "test": 1},
            entries=[PoolEntryFactory.brats("C1", m) for m in MODALITIES],
        )
        manifest.assert_condition_capacity(1, 1, MODALITIES)
        with pytest.raises(ValueError) as exc_info:
            manifest.assert_condition_capacity(2, 1, MODALITIES)
        assert "t1n×1" in str(exc_info.value)


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
            heldout_quota_volumes=512,
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

    def test_non_train_volumes_default_zero(self) -> None:
        """非 train split 卷计数缺省 0（既有留痕形态可扩不可改名）。"""
        assert self.manifest().non_train_volumes == 0


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
                source_commit="deadbeef",
                intensity_clip=False,
                resize_semantics="uniform-grid",
                upstream_anchor="NVIDIA v1 clip=False",
            ),
        )
        revived = ChannelStats.model_validate_json(stats.model_dump_json())
        assert revived.provenance == stats.provenance
        assert revived.provenance.source_commit == "deadbeef"

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

    def test_mean_std_length_mismatch_rejected(self) -> None:
        """mean/std 长度与 latent 通道数不符 → 拒绝（逐通道成对性破坏）。"""
        with pytest.raises(ValidationError, match="通道数"):
            ChannelStats(
                mean=[0.0] * 4,
                std=[1.0] * 3,
                num_latents=8,
                latent_shape=LATENT_SHAPE_A,
                source_manifest="real_pool.json",
            )
