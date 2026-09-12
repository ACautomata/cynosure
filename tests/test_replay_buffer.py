"""两区 Replay buffer 行为测试（ticket #20 AC + ADR-0008-01 #84）：

base 分区内容不变、近期分区 FIFO 滚动、混采占比 50/50、回放半区跨两区
均匀；条目带目标模态标签（ADR-0008 决策 2），回放采样可按本 iteration
条件过滤——该条件候选充足则在该条件内维持两区各半、可互补语义；该
条件不足时显式拒绝（可区分「条件不足」与「总数不足」，绝不静默回退
全池混采）；base 分区每条件配额量产依据与装配期供给守卫（ADR-0008
决策 4，train 装配与预训练 driver 同口径）。

设计（reward-model 章「在线更新机制」）：容量对半切分——固定 base 分区
（初始冻结 policy 产出，填满即锁）+ FIFO 近期分区（新 fake 滚动挤出最老）；
回放采样在 base / recent 之间均匀分配，某区样本不足时由另一区补足
（训练首步 recent 为空，回放全量由 base 承担）。
"""

import pytest
import torch

from cynosure.config import MODALITIES, CynosureConfig
from cynosure.reward.buffer import (
    ReplayBuffer,
    ReplayDraw,
    ReplayEntry,
    ZoneModalities,
    ZoneSizes,
    assert_replay_supply,
    base_condition_quota,
)

SHAPE = (4, 16, 16, 8)
CAPACITY = 64


class ZoneScenario:
    """两区缓冲测试场景：唯一值标记的样本构造与常用缓冲装配。"""

    @staticmethod
    def latents(values: list[float], shape: tuple = SHAPE) -> torch.Tensor:
        """每枚 latent 用唯一常数值填充（测试识别样本来源）。"""
        return torch.stack([torch.full(shape, value) for value in values])

    @staticmethod
    def modalities(count: int, modality: str = "t2w") -> list[str]:
        return [modality] * count

    @staticmethod
    def generator(seed: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(seed)

    @staticmethod
    def full_buffer() -> tuple[ReplayBuffer, list[float], list[float]]:
        """base 满（0..31）+ recent 满（100..131）。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(
            ZoneScenario.latents([float(i) for i in range(32)]),
            ZoneScenario.modalities(32),
        )
        buffer.push(
            ZoneScenario.latents([100.0 + i for i in range(32)]),
            "t2w",
        )
        return buffer, list(range(32)), [100.0 + i for i in range(32)]

    @staticmethod
    def tagged_full_buffer() -> tuple[ReplayBuffer, dict[str, list[float]]]:
        """两区满、条目按模态分组标记（条件过滤测试的识别面）。

        base（0..31）：t1n/t1c/t2w/t2f 循环打标（每模态 8 条）；
        recent（32 容量）：t2w 16 条（100..115）+ t2f 16 条（200..215）。"""
        buffer = ReplayBuffer(CAPACITY)
        base_modalities = [MODALITIES[i % 4] for i in range(32)]
        buffer.fill_base(
            ZoneScenario.latents([float(i) for i in range(32)]),
            base_modalities,
        )
        # push 是整批单标签：分两批构造混合标签的 recent
        buffer.push(
            ZoneScenario.latents([100.0 + i for i in range(16)]), "t2w",
        )
        buffer.push(
            ZoneScenario.latents([200.0 + i for i in range(16)]), "t2f",
        )
        values = {
            "t1n": list(range(0, 32, 4)),
            "t1c": list(range(1, 32, 4)),
            "t2w": list(range(2, 32, 4)) + [100.0 + i for i in range(16)],
            "t2f": list(range(3, 32, 4)) + [200.0 + i for i in range(16)],
        }
        return buffer, values


class TestZoneLayout:
    def test_capacity_splits_into_two_zones(self) -> None:
        """容量对半：base 与 recent 各占 capacity/2。"""
        buffer = ReplayBuffer(CAPACITY)
        assert buffer.base_capacity == 32
        assert buffer.recent_capacity == 32
        assert buffer.zone_sizes() == ZoneSizes(base=0, recent=0)

    def test_odd_capacity_rounds_extra_to_recent(self) -> None:
        """奇数容量余数归近期分区（近当前分布，见模块 docstring 口径）。"""
        buffer = ReplayBuffer(7)
        assert buffer.base_capacity == 3
        assert buffer.recent_capacity == 4


class TestEntryLabels:
    """ADR-0008-01 AC 1：两区条目的存储与观测面（快照、zone 观测）带模态标签。"""

    def test_fill_base_stores_labelled_entries(self) -> None:
        buffer = ReplayBuffer(CAPACITY)
        modalities = [MODALITIES[i % 4] for i in range(32)]
        buffer.fill_base(
            ZoneScenario.latents([float(i) for i in range(32)]), modalities,
        )
        entries = buffer.base_samples()
        assert all(isinstance(entry, ReplayEntry) for entry in entries)
        assert [entry.modality for entry in entries] == modalities
        assert torch.equal(
            torch.stack([entry.latent for entry in entries]),
            ZoneScenario.latents([float(i) for i in range(32)]),
        )

    def test_push_stores_labelled_entries(self) -> None:
        buffer = ReplayBuffer(CAPACITY)
        fakes = ZoneScenario.latents([7.0, 8.0])
        buffer.push(fakes, "t1c")
        entries = buffer.recent_samples()
        assert [entry.modality for entry in entries] == ["t1c", "t1c"]
        assert torch.equal(entries[0].latent, fakes[0])

    def test_zone_modalities_reports_per_condition_occupancy(self) -> None:
        """zone 观测面带模态标签：每条件 × 两区占用（回放条件审计面）。"""
        buffer, values = ZoneScenario.tagged_full_buffer()
        profile = buffer.zone_modalities()
        assert isinstance(profile, ZoneModalities)
        assert profile.base == {
            "t1n": 8, "t1c": 8, "t2w": 8, "t2f": 8,
        }
        assert profile.recent == {"t2w": 16, "t2f": 16}

    def test_zone_modalities_empty_buffer(self) -> None:
        buffer = ReplayBuffer(CAPACITY)
        profile = buffer.zone_modalities()
        assert profile.base == {}
        assert profile.recent == {}

    def test_fill_base_label_count_mismatch_rejected(self) -> None:
        """标签清单与样本数一一对应：长度不符显式拒绝（不猜、不截断）。"""
        buffer = ReplayBuffer(CAPACITY)
        with pytest.raises(ValueError, match="标签"):
            buffer.fill_base(
                ZoneScenario.latents([float(i) for i in range(32)]),
                ["t2w"] * 31,
            )


class TestBaseZone:
    def test_fill_base_exactly_capacity(self) -> None:
        buffer = ReplayBuffer(CAPACITY)
        base = ZoneScenario.latents([float(i) for i in range(32)])
        buffer.fill_base(base, ZoneScenario.modalities(32))
        assert buffer.zone_sizes() == ZoneSizes(base=32, recent=0)
        assert torch.equal(
            torch.stack([entry.latent for entry in buffer.base_samples()]), base,
        )

    def test_fill_base_insufficient_rejected(self) -> None:
        """「base 分区由初始 policy rollout 填满」：不足容量显式拒绝。"""
        buffer = ReplayBuffer(CAPACITY)
        with pytest.raises(ValueError, match="base"):
            buffer.fill_base(
                ZoneScenario.latents([float(i) for i in range(31)]),
                ZoneScenario.modalities(31),
            )

    def test_fill_base_excess_takes_first_capacity(self) -> None:
        """超出容量取前 base_capacity 条（编排方不必预切），标签同步截取。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(
            ZoneScenario.latents([float(i) for i in range(40)]),
            [MODALITIES[i % 4] for i in range(40)],
        )
        assert buffer.zone_sizes() == ZoneSizes(base=32, recent=0)
        assert buffer.base_samples()[-1].latent[0, 0, 0, 0].item() == 31.0
        assert buffer.base_samples()[-1].modality == MODALITIES[31 % 4]

    def test_fill_base_twice_rejected(self) -> None:
        """base 分区固定语义：一次填满后拒绝再次填充。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(
            ZoneScenario.latents([float(i) for i in range(32)]),
            ZoneScenario.modalities(32),
        )
        with pytest.raises(ValueError, match="已填"):
            buffer.fill_base(
                ZoneScenario.latents([100.0] * 32),
                ZoneScenario.modalities(32),
            )


class TestRecentZone:
    def test_push_fifo_rolls_over(self) -> None:
        """AC：近期分区 FIFO 滚动——超容后最老样本被挤出、顺序保持插入序。"""
        buffer = ReplayBuffer(16)
        first = ZoneScenario.latents([float(i) for i in range(8)])  # 0..7 填满
        second = ZoneScenario.latents([100.0, 101.0, 102.0])        # 挤出 0..2
        buffer.push(first, "t2w")
        buffer.push(second, "t2w")
        assert buffer.zone_sizes() == ZoneSizes(base=0, recent=8)
        values = [t.latent[0, 0, 0, 0].item() for t in buffer.recent_samples()]
        assert values == [3.0, 4.0, 5.0, 6.0, 7.0, 100.0, 101.0, 102.0]

    def test_push_never_touches_base(self) -> None:
        """AC：base 分区内容不变——push 多批后 base 仍为初始内容。"""
        buffer = ReplayBuffer(CAPACITY)
        base = ZoneScenario.latents([float(i) for i in range(32)])
        buffer.fill_base(base, ZoneScenario.modalities(32))
        for shift in range(3):
            buffer.push(
                ZoneScenario.latents([1000.0 + shift * 10 + i for i in range(4)]),
                "t1n",
            )
        assert torch.equal(
            torch.stack([entry.latent for entry in buffer.base_samples()]), base,
        )


class TestSampleReplay:
    def test_replay_split_evenly_across_zones(self) -> None:
        """AC：回放半区跨两区均匀——base / recent 各取一半（奇数归 recent）。"""
        buffer, base_values, recent_values = ZoneScenario.full_buffer()
        draw: ReplayDraw = buffer.sample_replay(6, ZoneScenario.generator())
        assert (draw.num_base, draw.num_recent) == (3, 3)
        assert tuple(draw.samples.shape) == (6, *SHAPE)
        for sample in draw.samples[: draw.num_base]:
            assert sample[0, 0, 0, 0].item() in base_values
        for sample in draw.samples[draw.num_base :]:
            assert sample[0, 0, 0, 0].item() in recent_values

    def test_replay_odd_count_rounds_to_recent(self) -> None:
        buffer, _, _ = ZoneScenario.full_buffer()
        draw = buffer.sample_replay(5, ZoneScenario.generator())
        assert (draw.num_base, draw.num_recent) == (2, 3)

    def test_replay_backfills_from_base_when_recent_empty(self) -> None:
        """训练首步 recent 为空：回放全量由 base 补足（样本可用性语义）。"""
        buffer = ReplayBuffer(CAPACITY)
        base_values = list(range(32))
        buffer.fill_base(
            ZoneScenario.latents([float(v) for v in base_values]),
            ZoneScenario.modalities(32),
        )
        draw = buffer.sample_replay(4, ZoneScenario.generator())
        assert (draw.num_base, draw.num_recent) == (4, 0)
        for sample in draw.samples:
            assert sample[0, 0, 0, 0].item() in base_values

    def test_replay_backfills_from_recent_when_base_short(self) -> None:
        """base 不足（容量奇数时 base < 需求）由 recent 补足。"""
        buffer = ReplayBuffer(7)  # base 3 / recent 4
        buffer.fill_base(
            ZoneScenario.latents([0.0, 1.0, 2.0]), ZoneScenario.modalities(3),
        )
        buffer.push(
            ZoneScenario.latents([100.0, 101.0, 102.0, 103.0]), "t2w",
        )
        draw = buffer.sample_replay(6, ZoneScenario.generator())
        assert (draw.num_base, draw.num_recent) == (3, 3)  # 需求 3+3，base 恰 3 条全上

    def test_replay_insufficient_samples_rejected(self) -> None:
        """base 未填、recent 不足需求：显式拒绝而非静默重复采样。"""
        buffer = ReplayBuffer(CAPACITY)
        with pytest.raises(ValueError, match="回放"):
            buffer.sample_replay(2, ZoneScenario.generator())

    def test_replay_deterministic_given_generator(self) -> None:
        """同 seed 生成器 → 采样完全一致（fixture 复现前提）。"""
        first_buffer, _, _ = ZoneScenario.full_buffer()
        second_buffer, _, _ = ZoneScenario.full_buffer()
        first = first_buffer.sample_replay(6, ZoneScenario.generator(7))
        second = second_buffer.sample_replay(6, ZoneScenario.generator(7))
        assert torch.equal(first.samples, second.samples)

    def test_replay_no_duplicate_within_single_sample(self) -> None:
        """单次回放采样无放回：同批内 base/recent 来源样本互不重复。"""
        buffer, _, _ = ZoneScenario.full_buffer()
        draw = buffer.sample_replay(6, ZoneScenario.generator())
        values = [s[0, 0, 0, 0].item() for s in draw.samples]
        assert len(set(values)) == 6
        assert len(values[: draw.num_base]) == len(set(values[: draw.num_base]))
        assert len(values[draw.num_base :]) == len(set(values[draw.num_base :]))


class TestSampleReplayByCondition:
    """ADR-0008-01 AC 2：回放按本 iteration 条件过滤——充足则该条件内
    两区各半混采（可互补语义不变）；不足则显式拒绝，可区分「条件不足」
    与「总数不足」，绝不静默回退全池混采。"""

    def test_condition_draw_splits_zones_within_condition(self) -> None:
        """该条件候选充足：base/recent 各半都在该条件内（混采语义不变）。"""
        buffer, values = ZoneScenario.tagged_full_buffer()
        # t2w：base 8 + recent 16 = 24 条；需求 6 → base 3 + recent 3
        draw = buffer.sample_replay(6, ZoneScenario.generator(), "t2w")
        assert (draw.num_base, draw.num_recent) == (3, 3)
        assert draw.modalities == ["t2w"] * 6  # 采样结果的标签观测与条件一致
        for sample in draw.samples[: draw.num_base]:
            assert sample[0, 0, 0, 0].item() in values["t2w"]
        for sample in draw.samples[draw.num_base :]:
            assert sample[0, 0, 0, 0].item() in values["t2w"]

    def test_condition_draw_backfills_within_condition(self) -> None:
        """条件过滤下互补语义保持：某区该条件不足由另一区同条件补足。

        t1n 仅 base 有（8 条）、recent 无：需求 6 全由 base 承担。"""
        buffer, values = ZoneScenario.tagged_full_buffer()
        draw = buffer.sample_replay(6, ZoneScenario.generator(), "t1n")
        assert (draw.num_base, draw.num_recent) == (6, 0)
        assert all(
            sample[0, 0, 0, 0].item() in values["t1n"] for sample in draw.samples
        )

    def test_condition_shortage_rejected_without_silent_fallback(self) -> None:
        """条件不足：显式拒绝且点名条件与可用量——绝不静默混回全池。"""
        buffer, _ = ZoneScenario.tagged_full_buffer()
        # t1n 共 8 条（全在 base）：需求 9 超出该条件候选
        with pytest.raises(ValueError, match="t1n") as exc_info:
            buffer.sample_replay(9, ZoneScenario.generator(), "t1n")
        message = str(exc_info.value)
        assert "回放" in message
        assert "回退" in message  # 报错声明不静默回退全池混采的口径

    def test_total_shortage_distinguished_from_condition_shortage(self) -> None:
        """「总数不足」与「条件不足」可区分：全池都不够时不报条件短缺。"""
        buffer = ReplayBuffer(CAPACITY)
        # 全池共 0 条：无条件不足可言，报总量短缺
        with pytest.raises(ValueError, match="回放") as exc_info:
            buffer.sample_replay(2, ZoneScenario.generator(), "t2w")
        assert "t2w" not in str(exc_info.value)

    def test_condition_shortage_message_reports_available_counts(self) -> None:
        """条件不足的报错含该条件两区可用量（可读、可行动）。"""
        buffer, _ = ZoneScenario.tagged_full_buffer()
        with pytest.raises(ValueError, match=r"t1n.*base \d+.*recent \d+") as exc_info:
            buffer.sample_replay(9, ZoneScenario.generator(), "t1n")
        assert "8" in str(exc_info.value)  # t1n 的 base 可用量

    def test_condition_draw_deterministic_given_generator(self) -> None:
        first, _ = ZoneScenario.tagged_full_buffer()
        second, _ = ZoneScenario.tagged_full_buffer()
        draw_first = first.sample_replay(6, ZoneScenario.generator(7), "t2w")
        draw_second = second.sample_replay(6, ZoneScenario.generator(7), "t2w")
        assert torch.equal(draw_first.samples, draw_second.samples)

    def test_condition_draw_no_duplicate_within_single_sample(self) -> None:
        """条件过滤不改变无放回语义：同批内样本互不重复。"""
        buffer, _ = ZoneScenario.tagged_full_buffer()
        draw = buffer.sample_replay(6, ZoneScenario.generator(), "t2w")
        values = [s[0, 0, 0, 0].item() for s in draw.samples]
        assert len(set(values)) == 6


class TestBaseConditionQuota:
    """ADR-0008-01 AC 3/4：base 分区每条件配额的量产依据与装配期守卫。"""

    def test_quota_splits_base_capacity_across_modalities(self) -> None:
        """base 容量均匀分派到各目标模态（余数按 MODALITIES 顺序）。"""
        quota = base_condition_quota(CAPACITY)  # base 32 → 每模态 8
        assert quota == {
            "t1n": 8, "t1c": 8, "t2w": 8, "t2f": 8,
        }

    def test_quota_remainder_rounds_in_modality_order(self) -> None:
        """base 容量不整除时余数按 MODALITIES 顺序逐个 +1（确定性）。"""
        quota = base_condition_quota(38)  # base 19 → 5,5,5,4
        assert sum(quota.values()) == 19
        assert quota["t1n"] == 5
        assert quota["t1c"] == 5
        assert quota["t2w"] == 5
        assert quota["t2f"] == 4

    def test_quota_sums_to_base_capacity(self) -> None:
        for capacity in (2, 7, 64, 65, 100):
            quota = base_condition_quota(capacity)
            assert sum(quota.values()) == capacity // 2

    def test_quota_covers_every_modality(self) -> None:
        quota = base_condition_quota(7)  # base 3：0/1/1/1——配额可为 0（守卫兜底线）
        assert set(quota) == set(MODALITIES)


class TestAssertReplaySupply:
    """装配期回放供给守卫（train 装配与预训练 driver 同口径）。"""

    @staticmethod
    def _config(tmp_path, **reward_overrides) -> CynosureConfig:
        import json
        data = {
            "experiment": {"group": "modal-label"},
            "latent_shape": [4, 16, 16, 8],
            "artifacts": {
                "unet_ckpt": "ckpts/unet.pt",
                "vae_ckpt": "ckpts/vae.pt",
                "net_config_json": "configs/net.json",
                "modality_mapping_json": "configs/mapping.json",
                "dataset_root": "data",
            },
            "reward": {
                "disc_batch_size_k": 4,
                "replay_buffer_capacity": 64,
                "real_pool_manifest": "pool.json",
                "heldout_real_manifest": "heldout.json",
                "channel_stats_json": "stats.json",
                "pretrain_report_json": "report.json",
            },
            "policy": {"input_img_size_numel": 16 * 16 * 8},
            "schedule": {"seed": 0},
        }
        data["reward"].update(reward_overrides)
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return CynosureConfig.model_validate(data)

    def test_valid_supply_passes(self, tmp_path) -> None:
        assert_replay_supply(self._config(tmp_path).reward)  # 不抛

    def test_replay_half_zero_rejected(self, tmp_path) -> None:
        """K=1：回放半区为 0 条——更新批无回放成分，装配期显式拒绝。"""
        with pytest.raises(ValueError, match="回放半区"):
            assert_replay_supply(
                self._config(tmp_path, disc_batch_size_k=1).reward,
            )

    def test_per_condition_quota_shortage_rejected(self, tmp_path) -> None:
        """ADR-0008-01 AC 4：某条件配额 < 回放半区需求 → fail-fast 可读报错。

        capacity=8 → base 4 → 每条件配额 1 < 回放半区 2（K=4）。"""
        with pytest.raises(ValueError, match="每条件配额") as exc_info:
            assert_replay_supply(
                self._config(
                    tmp_path,
                    disc_batch_size_k=4, replay_buffer_capacity=8,
                ).reward,
            )
        message = str(exc_info.value)
        assert "1" in message and "2" in message
        assert "replay_buffer_capacity" in message  # 可行动：指明调哪个 knob

    def test_guard_counts_min_quota_not_total(self, tmp_path) -> None:
        """守卫按最弱条件配额判定（非 base 总量）：总量够、单条件不够也拒。"""
        # capacity=20 → base 10 → 配额 3/3/2/2：min 2 ≥ 回放半区 2 恰好过
        assert_replay_supply(
            self._config(
                tmp_path, disc_batch_size_k=4, replay_buffer_capacity=20,
            ).reward,
        )
        # K=6 → 回放半区 3 > min 配额 2：拒
        with pytest.raises(ValueError, match="每条件配额"):
            assert_replay_supply(
                self._config(
                    tmp_path, disc_batch_size_k=6, replay_buffer_capacity=20,
                ).reward,
            )
