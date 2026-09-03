"""两区 Replay buffer 行为测试（ticket #20 AC）：
base 分区内容不变、近期分区 FIFO 滚动、混采占比 50/50、回放半区跨两区均匀。

设计（reward-model 章「在线更新机制」）：容量对半切分——固定 base 分区
（初始冻结 policy 产出，填满即锁）+ FIFO 近期分区（新 fake 滚动挤出最老）；
回放采样在 base / recent 之间均匀分配，某区样本不足时由另一区补足
（训练首步 recent 为空，回放全量由 base 承担）。
"""

import pytest
import torch

from cynosure.reward.buffer import ReplayBuffer, ReplayDraw, ZoneSizes

SHAPE = (4, 16, 16, 8)
CAPACITY = 64


class ZoneScenario:
    """两区缓冲测试场景：唯一值标记的样本构造与常用缓冲装配。"""

    @staticmethod
    def latents(values: list[float], shape: tuple = SHAPE) -> torch.Tensor:
        """每枚 latent 用唯一常数值填充（测试识别样本来源）。"""
        return torch.stack([torch.full(shape, value) for value in values])

    @staticmethod
    def generator(seed: int = 0) -> torch.Generator:
        return torch.Generator().manual_seed(seed)

    @staticmethod
    def full_buffer() -> tuple[ReplayBuffer, list[float], list[float]]:
        """base 满（0..31）+ recent 满（100..131）。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(ZoneScenario.latents([float(i) for i in range(32)]))
        buffer.push(ZoneScenario.latents([100.0 + i for i in range(32)]))
        return buffer, list(range(32)), [100.0 + i for i in range(32)]


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


class TestBaseZone:
    def test_fill_base_exactly_capacity(self) -> None:
        buffer = ReplayBuffer(CAPACITY)
        base = ZoneScenario.latents([float(i) for i in range(32)])
        buffer.fill_base(base)
        assert buffer.zone_sizes() == ZoneSizes(base=32, recent=0)
        assert torch.equal(torch.stack(buffer.base_samples()), base)

    def test_fill_base_insufficient_rejected(self) -> None:
        """「base 分区由初始 policy rollout 填满」：不足容量显式拒绝。"""
        buffer = ReplayBuffer(CAPACITY)
        with pytest.raises(ValueError, match="base"):
            buffer.fill_base(ZoneScenario.latents([float(i) for i in range(31)]))

    def test_fill_base_excess_takes_first_capacity(self) -> None:
        """超出容量取前 base_capacity 条（编排方不必预切）。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(ZoneScenario.latents([float(i) for i in range(40)]))
        assert buffer.zone_sizes() == ZoneSizes(base=32, recent=0)
        assert buffer.base_samples()[-1][0, 0, 0, 0].item() == 31.0

    def test_fill_base_twice_rejected(self) -> None:
        """base 分区固定语义：一次填满后拒绝再次填充。"""
        buffer = ReplayBuffer(CAPACITY)
        buffer.fill_base(ZoneScenario.latents([float(i) for i in range(32)]))
        with pytest.raises(ValueError, match="已填"):
            buffer.fill_base(ZoneScenario.latents([100.0] * 32))


class TestRecentZone:
    def test_push_fifo_rolls_over(self) -> None:
        """AC：近期分区 FIFO 滚动——超容后最老样本被挤出、顺序保持插入序。"""
        buffer = ReplayBuffer(16)
        first = ZoneScenario.latents([float(i) for i in range(8)])  # 0..7 填满
        second = ZoneScenario.latents([100.0, 101.0, 102.0])        # 挤出 0..2
        buffer.push(first)
        buffer.push(second)
        assert buffer.zone_sizes() == ZoneSizes(base=0, recent=8)
        values = [t[0, 0, 0, 0].item() for t in buffer.recent_samples()]
        assert values == [3.0, 4.0, 5.0, 6.0, 7.0, 100.0, 101.0, 102.0]

    def test_push_never_touches_base(self) -> None:
        """AC：base 分区内容不变——push 多批后 base 仍为初始内容。"""
        buffer = ReplayBuffer(CAPACITY)
        base = ZoneScenario.latents([float(i) for i in range(32)])
        buffer.fill_base(base)
        for shift in range(3):
            buffer.push(
                ZoneScenario.latents([1000.0 + shift * 10 + i for i in range(4)]),
            )
        assert torch.equal(torch.stack(buffer.base_samples()), base)


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
        buffer.fill_base(ZoneScenario.latents([float(v) for v in base_values]))
        draw = buffer.sample_replay(4, ZoneScenario.generator())
        assert (draw.num_base, draw.num_recent) == (4, 0)
        for sample in draw.samples:
            assert sample[0, 0, 0, 0].item() in base_values

    def test_replay_backfills_from_recent_when_base_short(self) -> None:
        """base 不足（容量奇数时 base < 需求）由 recent 补足。"""
        buffer = ReplayBuffer(7)  # base 3 / recent 4
        buffer.fill_base(ZoneScenario.latents([0.0, 1.0, 2.0]))
        buffer.push(ZoneScenario.latents([100.0, 101.0, 102.0, 103.0]))
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
