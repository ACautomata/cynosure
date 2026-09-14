"""条件白名单动态恢复（ADR-0008 决策 8）与逐 iteration 门控的决定面。

验收面（issue #89）：

- EMA 平滑观测器：首观测置值、α = 2/(span+1) 递推、观测计数、落盘
  状态回填（restore 与 observe 的一对写入口）；
- 滞回门控：gated 条件 EMA 越 enter 阈值恢复更新、名单内条件跌破
  exit 阈值重新门控、enter/exit 之间的滞回带维持现状（防抖动）；
- 动态恢复开关关闭 = 静态白名单降级路径（observe 旁路，名单恒为
  启动名单）；
- 门控状态 state/adopt roundtrip 逐位一致（续训复原的落盘侧）；
  损坏形态显式拒绝；
- world-1（单进程）下 observe 的集体协议恒等退化——同一条执行序；
- broadcast_object 的单进程恒等；
- config knobs：滞回带形状校验（0.5 < exit < enter < 1.0）、EMA 跨度
  正整数；IterEvent 的 policy_gated 观测面默认 False。

多 rank 的集体一致性（gather → rank 0 判定 → broadcast 镜像）由
test_distributed 的 spawn world 覆盖（本文件的 world-1 恒等路径 +
判定逻辑单测共同收口）。
"""

import pytest

from cynosure.config import RewardConfig
from cynosure.distributed import DistributedContext
from cynosure.train.artifacts import IterEvent
from cynosure.train.gating import ConditionAucEma, DynamicWhitelist, Observation
from cynosure.train.whitelist import ConditionWhitelist


class GatingFixture:
    """门控单测的装配面：最小 RewardConfig 与单进程门控对象的构造
    （工厂收拢为类，不留在模块级）。"""

    @staticmethod
    def reward_config(**overrides) -> RewardConfig:
        """最小合法 RewardConfig（门控 knobs 可覆写）。"""
        fields = dict(
            disc_batch_size_k=4,
            replay_buffer_capacity=64,
            real_pool_manifest="artifacts/real_pool.json",
            heldout_real_manifest="artifacts/heldout_real.json",
            channel_stats_json="artifacts/channel_stats.json",
            pretrain_report_json="artifacts/pretrain_report.json",
        )
        fields.update(overrides)
        return RewardConfig(**fields)

    @classmethod
    def gating(cls, members, measured=None, dist=None, **reward_overrides) -> DynamicWhitelist:
        """单进程（world-1）门控对象（初始名单 + 默认 knobs）。"""
        return DynamicWhitelist(
            ConditionWhitelist(tuple(members), measured or {}),
            cls.reward_config(**reward_overrides),
            dist if dist is not None else DistributedContext(0, 1, False),
        )


class TestConditionAucEma:
    """EMA 平滑观测器的数值语义。"""

    def test_first_observation_seeds_value(self) -> None:
        tracker = ConditionAucEma(span=8)
        assert tracker.value is None
        assert tracker.count == 0
        assert tracker.observe(0.7) == 0.7
        assert tracker.value == 0.7
        assert tracker.count == 1

    def test_recursion_uses_span_implied_alpha(self) -> None:
        """span=8 → α=2/9：ema ← (1−α)·ema + α·sample 的显式对照。"""
        tracker = ConditionAucEma(span=8)
        tracker.observe(0.9)
        expected = (7.0 / 9.0) * 0.9 + (2.0 / 9.0) * 0.1
        assert tracker.observe(0.1) == pytest.approx(expected)
        assert tracker.count == 2

    def test_restore_refills_state_bitwise(self) -> None:
        tracker = ConditionAucEma(span=8)
        tracker.observe(0.3)
        tracker.observe(0.8)
        twin = ConditionAucEma(span=8)
        twin.restore(tracker.value, tracker.count)
        assert twin.value == tracker.value
        assert twin.count == tracker.count
        # 回填后的状态机继续同轨迹演化（后续观测逐位一致）
        assert twin.observe(0.55) == tracker.observe(0.55)

    def test_invalid_span_rejected(self) -> None:
        with pytest.raises(ValueError, match="跨度"):
            ConditionAucEma(span=0)

    def test_restore_rejects_zero_count(self) -> None:
        with pytest.raises(ValueError, match="计数"):
            ConditionAucEma(span=8).restore(0.5, 0)


class TestHysteresisGating:
    """滞回判定：enter 恢复 / exit 重新门控 / 滞回带维持。"""

    def test_gated_condition_recovers_above_enter(self) -> None:
        gating = GatingFixture.gating(members=("t1n",))
        assert "t2w" not in gating.whitelist
        gating.observe("t2w", 0.9)  # 首观测即 EMA，越过 enter 0.55
        assert "t2w" in gating.whitelist

    def test_member_condition_regates_below_exit(self) -> None:
        gating = GatingFixture.gating(members=("t1n", "t2w"))
        gating.observe("t2w", 0.9)
        assert "t2w" in gating.whitelist
        # 连续低观测驱动 EMA 跌破 exit 0.52（0.9 → 0.722 → 0.562 → 0.415）
        for _ in range(3):
            gating.observe("t2w", 0.1)
        assert "t2w" not in gating.whitelist

    def test_hysteresis_band_holds_state(self) -> None:
        """enter/exit 之间的 EMA 不触发进出（滞回带防名单抖动）。"""
        gating = GatingFixture.gating(members=("t1n",))
        gating.observe("t2w", 0.53)  # enter 0.55 之下、exit 0.52 之上
        assert "t2w" not in gating.whitelist
        recovered = GatingFixture.gating(members=("t1n", "t2w"))
        recovered.observe("t2w", 0.53)  # exit 之上不退出
        assert "t2w" in recovered.whitelist

    def test_strict_boundaries(self) -> None:
        """越过 enter = ≥（0.55 即恢复）；跌破 exit = <（0.52 不退出）。"""
        at_enter = GatingFixture.gating(members=("t1n",))
        at_enter.observe("t2w", 0.55)
        assert "t2w" in at_enter.whitelist
        at_exit = GatingFixture.gating(members=("t1n", "t2w"))
        at_exit.observe("t2w", 0.52)
        assert "t2w" in at_exit.whitelist

    def test_recovery_survives_subsequent_regate_cycle(self) -> None:
        """恢复后的条件可再次被门控（动态机制双向可逆）。"""
        gating = GatingFixture.gating(members=("t1n",))
        gating.observe("t2w", 0.9)
        assert "t2w" in gating.whitelist
        for _ in range(4):
            gating.observe("t2w", 0.1)
        assert "t2w" not in gating.whitelist
        gating.observe("t2w", 0.95)  # EMA = 7/9·0.337 + 2/9·0.95 ≈ 0.474 < enter
        assert "t2w" not in gating.whitelist
        gating.observe("t2w", 0.95)  # ≈ 0.588 ≥ enter → 再次恢复
        assert "t2w" in gating.whitelist

    def test_member_order_normalized_to_rotation_order(self) -> None:
        """名单成员恒为 MODALITIES 固定序（轮转序子序列）——乱序初始
        名单与乱序恢复顺序都归一。"""
        gating = GatingFixture.gating(members=("t2w", "t1n"))
        assert gating.whitelist.members == ("t1n", "t2w")
        gating.observe("t2f", 0.9)
        assert gating.whitelist.members == ("t1n", "t2w", "t2f")

    def test_per_condition_ema_independent(self) -> None:
        """各条件 EMA 独立递推（观测流按条件分流，互不串扰）。"""
        gating = GatingFixture.gating(members=("t1n",))
        gating.observe("t2w", 0.9)
        gating.observe("t2f", 0.1)
        gating.observe("t2w", 0.9)  # t2w EMA 保持 0.9
        assert gating.whitelist.members == ("t1n", "t2w")
        assert "t2f" not in gating.whitelist

    def test_measured_snapshot_preserved_across_changes(self) -> None:
        """报告实测快照（measured）跨名单变更保留（gate 拒绝报错的
        实测值来源语义不受动态恢复影响）。"""
        gating = GatingFixture.gating(
            members=("t1n",), measured={"t1n": 0.7, "t2w": 0.48},
        )
        gating.observe("t2w", 0.9)
        assert gating.whitelist.measured == {"t1n": 0.7, "t2w": 0.48}


class TestStaticFallback:
    """动态恢复关闭 = 静态白名单降级路径。"""

    def test_observe_is_a_noop_when_disabled(self) -> None:
        gating = GatingFixture.gating(
            members=("t1n",), gating_dynamic_recovery=False,
        )
        gating.observe("t2w", 0.9)  # 越 enter 也不恢复
        gating.observe("t1n", 0.1)  # 跌破 exit 也不门控
        assert gating.whitelist.members == ("t1n",)
        assert gating.state()["ema"] == {}  # EMA 未建立

    def test_default_is_dynamic(self) -> None:
        """默认开启动态恢复（静态白名单是显式降级选择）。"""
        assert GatingFixture.reward_config().gating_dynamic_recovery is True
        gating = GatingFixture.gating(members=("t1n",))
        gating.observe("t2w", 0.9)
        assert "t2w" in gating.whitelist


class TestGatingState:
    """门控状态的落盘形态与回填（续训复原的落盘侧）。"""

    def test_state_roundtrip_bitwise(self) -> None:
        source = GatingFixture.gating(
            members=("t1n", "t2w"),
            measured={"t1n": 0.7},
            gating_ema_span=4,
        )
        source.observe("t2w", 0.31)
        source.observe("t2w", 0.89)
        source.observe("t2f", 0.6)
        state = source.state()

        target = GatingFixture.gating(members=("t2f",), gating_ema_span=4)
        target.adopt(state)
        assert target.whitelist.members == source.whitelist.members
        assert target.state() == source.state()
        # 回填后的状态机继续同轨迹演化（后续观测逐位一致）
        target.observe("t2w", 0.5)
        source.observe("t2w", 0.5)
        assert target.state() == source.state()

    def test_adopt_rejects_malformed_state(self) -> None:
        gating = GatingFixture.gating(members=("t1n",))
        for bad in (
            {},
            {"members": ["t1n"]},  # 缺 ema
            {"members": ["t1n"], "ema": {}, "extra": 1},
            {"members": ["mri"], "ema": {}},  # 非法成员
            {"members": ["t1n"], "ema": {"t2w": 0.5}},  # 条目形态非法
            {"members": ["t1n"], "ema": {"t2w": {"value": 0.5, "count": 0}}},
            {"members": ["t1n"], "ema": {"mri": {"value": 0.5, "count": 1}}},
            {"members": ["t1n"], "ema": {"t2w": {"value": None, "count": 1}}},
            {"members": ["t1n"], "ema": {"t2w": {"value": 0.5, "count": True}}},
        ):
            with pytest.raises(ValueError):
                gating.adopt(bad)
        assert gating.whitelist.members == ("t1n",)  # 拒绝后名单不动

    def test_unobserved_conditions_absent_from_state(self) -> None:
        gating = GatingFixture.gating(members=("t1n", "t2w"))
        assert gating.state() == {"members": ["t1n", "t2w"], "ema": {}}


class StubWorldDist:
    """多 rank 集合原语的测试替身（鸭子类型：DynamicWhitelist 只消费
    ``all_gather``/``broadcast_object``/``rank``）——``peer_submissions``
    预设世界内各 rank 的提交物，模拟「各 rank 条件独立采样」的分布式
    观测面；``rank0_snapshot`` 为广播载荷（rank 0 判定产出的门控快照）。"""

    def __init__(
        self, rank: int, peer_submissions: list, rank0_snapshot: dict | None,
    ) -> None:
        self._rank = rank
        self._peers = peer_submissions
        self._rank0_snapshot = rank0_snapshot

    @property
    def rank(self) -> int:
        return self._rank

    def all_gather(self, items: list) -> list[list]:
        return [list(peer) for peer in self._peers]

    def broadcast_object(self, value):
        return value if self._rank == 0 else self._rank0_snapshot


class TestCollectiveGateDecision:
    """跳过决定的全 rank 集体口径（OR 归约——FSDP 集合操作错配死锁防线）。"""

    def test_any_rank_gated_skips_update_for_all(self) -> None:
        """世界 = [(t1n, 0.7, False), (t2w, 0.6, True)]：rank 1 的条件被
        门控 → 两个 rank 的 observe 都返回 True（全体跳过本 iteration
        的 policy 更新）。"""
        submissions = [[Observation("t1n", 0.7, False)], [Observation("t2w", 0.6, True)]]
        snapshot = {"members": ["t1n"], "ema": {}}
        for rank in (0, 1):
            gating = GatingFixture.gating(
                members=("t1n",),
                dist=StubWorldDist(rank, submissions, snapshot),
            )
            decision = gating.observe(
                submissions[rank][0][0], submissions[rank][0][1],
            )
            assert decision is True

    def test_all_ranks_admitted_runs_update(self) -> None:
        """全体条件在名单内 → observe 返回 False（正常更新步）。"""
        submissions = [[Observation("t1n", 0.7, False)], [Observation("t1c", 0.65, False)]]
        snapshot = {"members": ["t1n", "t1c"], "ema": {}}
        gating = GatingFixture.gating(
            members=("t1n", "t1c"),
            dist=StubWorldDist(0, submissions, snapshot),
        )
        decision = gating.observe("t1n", 0.7)
        assert decision is False

    def test_same_condition_observations_merged_by_mean(self) -> None:
        """同 iteration 同条件的各 rank 观测均值合并为一条（集体观测
        口径）——rank 0 的 EMA 收到 0.7 = mean(0.8, 0.6)。"""
        submissions = [[Observation("t2w", 0.8, True)], [Observation("t2w", 0.6, True)]]
        snapshot = {"members": ["t1n"], "ema": {"t2w": {"value": 0.7, "count": 1}}}
        gating = GatingFixture.gating(
            members=("t1n",),
            dist=StubWorldDist(0, submissions, snapshot),
        )
        gating.observe("t2w", 0.8)
        assert gating.state()["ema"]["t2w"] == {"value": 0.7, "count": 1}

    def test_static_fallback_still_collective(self) -> None:
        """静态降级路径：名单不变更（无 EMA 记录），但跳过决定仍走
        集体对账——部分 rank 被门控时全体跳过。"""
        submissions = [[False], [True]]
        gating = GatingFixture.gating(
            members=("t1n",), gating_dynamic_recovery=False,
            dist=StubWorldDist(0, submissions, None),
        )
        decision = gating.observe("t1n", 0.7)
        assert decision is True
        assert gating.state()["ema"] == {}

    def test_world1_return_matches_local_query(self) -> None:
        """world-1 恒等：OR 归约退化为本 rank 查询（同一条执行序）。"""
        gating = GatingFixture.gating(members=("t1n",))
        assert gating.observe("t2w", 0.6) is True
        assert gating.observe("t1n", 0.6) is False


class TestWorld1CollectiveIdentity:
    """world-1 恒等：单进程走同一条集体执行序（原语恒等退化）。"""

    def test_broadcast_object_identity(self) -> None:
        dist = DistributedContext(0, 1, False)
        payload = {"members": ["t1n"], "ema": {"t2w": {"value": 0.5, "count": 2}}}
        assert dist.broadcast_object(payload) is payload


class TestConfigKnobs:
    """门控 knobs 的 schema 约束（暂定值 + 滞回带形状）。"""

    def test_tentative_defaults(self) -> None:
        config = GatingFixture.reward_config()
        assert config.gating_enter_auc == 0.55
        assert config.gating_exit_auc == 0.52
        assert config.gating_ema_span == 8
        assert config.gating_dynamic_recovery is True

    def test_hysteresis_band_shape_enforced(self) -> None:
        base = dict(GatingFixture.reward_config().model_dump())
        for enter, exit_ in ((0.52, 0.55), (0.5, 0.5), (0.55, 0.55),
                             (0.48, 0.4)):
            with pytest.raises(ValueError, match="滞回带"):
                RewardConfig(**{**base, "gating_enter_auc": enter,
                                "gating_exit_auc": exit_})

    def test_calibrated_pair_below_enter_above_chance_is_legal(self) -> None:
        """exit 与 enter 均在 chance 之上且滞回带非空即合法（校准定版
        后的取值形态，如 0.6/0.51）。"""
        base = dict(GatingFixture.reward_config().model_dump())
        config = RewardConfig(**{**base, "gating_enter_auc": 0.6,
                                 "gating_exit_auc": 0.51})
        assert config.gating_enter_auc == 0.6
        assert config.gating_exit_auc == 0.51

    def test_ema_span_must_be_positive(self) -> None:
        base = dict(GatingFixture.reward_config().model_dump())
        with pytest.raises(ValueError):
            RewardConfig(**{**base, "gating_ema_span": 0})


class TestIterEventObservation:
    """iter 事件的门控观测面（事件契约可扩不可改名）。"""

    def test_policy_gated_defaults_false(self) -> None:
        event = IterEvent(
            iteration=0,
            modality="t1n",
            anchor_eval_reward=0.5,
            intra_group_reward_std=0.1,
            heldout_auc=0.6,
            loss={},
            buffer_current_fraction=0.5,
            buffer_replay_fraction=0.5,
            buffer_base_occupied=8,
            buffer_recent_occupied=4,
            lr=2e-6,
            elapsed_s=1.0,
        )
        assert event.policy_gated is False
