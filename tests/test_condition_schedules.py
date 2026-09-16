"""sigma 日程逐条件锚测试（#129，spec #125 实现决策 2）。

数值锚语义（ADR-0002 逐条件化，不变式：每条件锚 = 该条件空间 numel）：
MONAI ``set_timesteps(use_timestep_transform=True)`` 的 SD3 式 timestep
transform 以 numel 为输入（``ratio_space = (numel/32³)^(1/3)``）——锚
错则 sigma 日程静默错位。本测试面：

- 逐条件日程表：异锚条件产出异 timesteps（fixture 异形小词汇表，
  numel 2048 vs 1024）；同锚条件共享日程（numel 碰撞的日程等价）；
  未知条件名/缺名显式拒绝（防「按名选日程」的静默回退）。
- 单条件日程表（BraTS 特例）：任意条件名共用一份 ADR-0002 日程
  （单域语义 = 单条件词汇特例；现状 BraTS 行为的回归锚）。
- 与 netbuild 直接构建的日程逐位一致（装配面换日程表后 BraTS
  数值零漂移）。
"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.conditions import (
    BraTSConditionVocabulary,
    MrConditionVocabulary,
)
from cynosure.fixtures import (
    FIXTURE_MODALITY_MAPPING,
    FIXTURE_MR_MODALITY_TOKENS,
    Fixture,
)
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.condition import ModalityMapping
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.schedules import (
    PerConditionSchedules,
    SingleConditionSchedules,
)
from cynosure.train.rollout import MrConditionSampler

FIXTURE_STEPS = Fixture.NUM_INFERENCE_STEPS
FIXTURE_NUMEL = Fixture.INPUT_IMG_SIZE_NUMEL

# fixture 异形小词汇表（B 条件网格换薄厚轴：latent (8,8,16)，numel 1024；
# 与 A 的 (16,16,8)/2048 不同锚——逐条件日程的输入面）
HETEROGENEOUS_CONDITIONS = [
    {
        "name": "t1w/axial", "modality": "t1w", "plane": "axial",
        "fov_mm": [64.0, 64.0, 32.0], "fov_source": "fixture",
        "grid_xyz": [64, 64, 32],
    },
    {
        "name": "flair/axial", "modality": "flair", "plane": "axial",
        "fov_mm": [32.0, 32.0, 64.0], "fov_source": "fixture",
        "grid_xyz": [32, 32, 64],
    },
]


class HeterogeneousVocabularyFixture:
    """逐条件日程测试的装配面：tmp 落盘 fixture 异形小词汇表并装载
    （fixture_mode 通道）。"""

    @staticmethod
    def load(tmp_path: Path) -> MrConditionVocabulary:
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps({
            "kind": "mrrate-condition-vocabulary",
            "issue": "#129-test",
            "upstream_reference": "fixture",
            "grid_semantics": "fixture 异形小词汇表：逐条件日程测试输入",
            "modality_tokens": FIXTURE_MR_MODALITY_TOKENS,
            "conditions": HETEROGENEOUS_CONDITIONS,
        }), encoding="utf-8")
        return MrConditionVocabulary.load(path, fixture_mode=True)


class TestSingleConditionSchedules:
    """BraTS 单条件日程表：任意条件名（含缺名）共用一份日程。"""

    def test_cursor_matches_netbuild_schedule(self) -> None:
        """与 netbuild 直接构建的 TrajectoryCursor 逐位一致（装配面
        换日程表后 BraTS 数值零漂移的回归锚）。"""
        schedules = SingleConditionSchedules(
            num_inference_steps=FIXTURE_STEPS,
            input_img_size_numel=FIXTURE_NUMEL,
        )
        reference = TrajectoryCursor(NetworkAssembler.rflow_scheduler(
            num_inference_steps=FIXTURE_STEPS,
            input_img_size_numel=FIXTURE_NUMEL,
        ))
        for name in ("t1n", None, "t2f"):
            cursor = schedules.cursor(name)
            assert torch.equal(cursor.timesteps, reference.timesteps)
            assert torch.equal(cursor.next_timesteps, reference.next_timesteps)

    def test_cursor_is_stable_across_calls(self) -> None:
        """同一实例多次取 cursor 为同一快照（轨迹游标自持语义不变）。"""
        schedules = SingleConditionSchedules(
            num_inference_steps=FIXTURE_STEPS,
            input_img_size_numel=FIXTURE_NUMEL,
        )
        assert schedules.cursor("t1n") is schedules.cursor(None)


class TestPerConditionSchedules:
    """逐条件日程表：锚 = 词汇表条件的空间 numel，按条件名选日程。"""

    def test_distinct_anchors_produce_distinct_schedules(
        self, tmp_path: Path,
    ) -> None:
        """异锚条件（numel 2048 vs 1024）→ timesteps 不同：锚逐条件
        真实生效（sigma 日程随条件形状变），不是单一日程换名。"""
        vocab = HeterogeneousVocabularyFixture.load(tmp_path)
        schedules = PerConditionSchedules(
            num_inference_steps=FIXTURE_STEPS, vocabulary=vocab,
        )
        cursor_a = schedules.cursor("t1w/axial")
        cursor_b = schedules.cursor("flair/axial")
        assert vocab.latent_numel("t1w/axial") == 2048
        assert vocab.latent_numel("flair/axial") == 1024
        assert not torch.equal(cursor_a.timesteps, cursor_b.timesteps)
        # 两端日程各自的锚一致性：与 netbuild 按该条件锚直接构建逐位相同
        for name in ("t1w/axial", "flair/axial"):
            reference = TrajectoryCursor(NetworkAssembler.rflow_scheduler(
                num_inference_steps=FIXTURE_STEPS,
                input_img_size_numel=vocab.latent_numel(name),
            ))
            assert torch.equal(
                schedules.cursor(name).timesteps, reference.timesteps,
            )

    def test_same_anchor_conditions_share_schedule(self, tmp_path: Path) -> None:
        """numel 碰撞（网格不同、空间 numel 相同）的条件共享 sigma 日程
        ——MONAI transform 的输入只有 numel，日程等价是数值事实而非巧合。"""
        path = tmp_path / "vocab.json"
        conditions = [
            {
                "name": "t1w/axial", "modality": "t1w", "plane": "axial",
                "fov_mm": [64.0, 64.0, 32.0], "fov_source": "fixture",
                "grid_xyz": [64, 64, 32],
            },
            {
                "name": "t1w/sagittal", "modality": "t1w", "plane": "sagittal",
                "fov_mm": [32.0, 64.0, 64.0], "fov_source": "fixture",
                "grid_xyz": [32, 64, 64],
            },
        ]
        path.write_text(json.dumps({
            "kind": "mrrate-condition-vocabulary",
            "issue": "#129-test",
            "upstream_reference": "fixture",
            "grid_semantics": "fixture 同锚异形小词汇表",
            "modality_tokens": FIXTURE_MR_MODALITY_TOKENS,
            "conditions": conditions,
        }), encoding="utf-8")
        vocab = MrConditionVocabulary.load(path, fixture_mode=True)
        schedules = PerConditionSchedules(
            num_inference_steps=FIXTURE_STEPS, vocabulary=vocab,
        )
        assert vocab.latent_numel("t1w/axial") == vocab.latent_numel(
            "t1w/sagittal"
        )
        assert torch.equal(
            schedules.cursor("t1w/axial").timesteps,
            schedules.cursor("t1w/sagittal").timesteps,
        )

    def test_unknown_condition_rejected(self, tmp_path: Path) -> None:
        """按名选日程：未知条件名显式拒绝（不静默回退任意日程——防
        「日程错位」的最后一道运行时闸口）。"""
        vocab = HeterogeneousVocabularyFixture.load(tmp_path)
        schedules = PerConditionSchedules(
            num_inference_steps=FIXTURE_STEPS, vocabulary=vocab,
        )
        with pytest.raises(KeyError, match="not-in-vocabulary"):
            schedules.cursor("not-in-vocabulary")

    def test_missing_name_rejected(self, tmp_path: Path) -> None:
        """逐条件日程表必须携带条件名：缺名即拒绝（单域例外由
        SingleConditionSchedules 承担，不在本类静默兼容）。"""
        vocab = HeterogeneousVocabularyFixture.load(tmp_path)
        schedules = PerConditionSchedules(
            num_inference_steps=FIXTURE_STEPS, vocabulary=vocab,
        )
        with pytest.raises(ValueError, match="条件名"):
            schedules.cursor(None)


class TestConditionVocabularyBraTSStrategy:
    """BraTS 词汇策略（单域特例）：条件集 = 四序列，形状/锚全局单值。"""

    def _vocabulary(self) -> BraTSConditionVocabulary:
        return BraTSConditionVocabulary(
            latent_shape=Fixture.LATENT_SHAPE,
            mapping=ModalityMapping(FIXTURE_MODALITY_MAPPING),
        )

    def test_names_are_modalities(self) -> None:
        assert self._vocabulary().names() == ("t1n", "t1c", "t2w", "t2f")

    def test_any_condition_maps_to_single_latent_shape(self) -> None:
        vocab = self._vocabulary()
        for name in vocab.names():
            assert vocab.latent_shape(name) == (4, 16, 16, 8)
            assert vocab.latent_numel(name) == 2048

    def test_unknown_modality_rejected(self) -> None:
        """域外序列名显式拒绝（防「条件拼错静默按同形处理」——形状
        消费前的合法域守卫）。"""
        with pytest.raises(KeyError, match="not-a-modality"):
            self._vocabulary().latent_shape("not-a-modality")

    def test_token_comes_from_mapping(self) -> None:
        """token 取数自 modality mapping 装载产物（单一来源，不设
        代码内副本）：fixture 映射 t1n → 29、t2w → 30。"""
        vocab = self._vocabulary()
        assert vocab.token("t1n") == 29
        assert vocab.token("t2w") == 30
        with pytest.raises(KeyError):
            vocab.token("not-a-modality")

    def test_spacing_is_unit_x1e2(self) -> None:
        """BraTS 组1 间距 = 单位间距 ×1e2 常量（policy-modeling 章
        spacing ×1e2 恒传口径）。"""
        assert self._vocabulary().spacing_condition("t1n") == (100.0, 100.0, 100.0)


class TestMrConditionSampler:
    """MR-RATE 组1 条件分布：词汇表条件均匀轮转、条件属性贯通。"""

    def test_sample_produces_vocabulary_conditions(self, tmp_path: Path) -> None:
        vocab = HeterogeneousVocabularyFixture.load(tmp_path)
        sampler = MrConditionSampler(
            vocab, torch.Generator().manual_seed(0), torch.device("cpu"),
        )
        assert sampler.targets() == ("t1w/axial", "flair/axial")
        condition, name = sampler.sample()
        assert name in sampler.targets()
        assert condition.name == name
        spec = vocab.by_name(name)
        assert condition.label.tolist() == [spec.token]
        assert condition.spacing.tolist() == [
            [axis * 100.0 for axis in spec.spacing_mm]
        ]

    def test_sample_target_known_and_unknown(self, tmp_path: Path) -> None:
        vocab = HeterogeneousVocabularyFixture.load(tmp_path)
        sampler = MrConditionSampler(
            vocab, torch.Generator().manual_seed(0), torch.device("cpu"),
        )
        condition = sampler.sample_target("flair/axial")
        assert condition.name == "flair/axial"
        assert condition.label.tolist() == [vocab.by_name("flair/axial").token]
        with pytest.raises(KeyError, match="not-in-vocabulary"):
            sampler.sample_target("not-in-vocabulary")
