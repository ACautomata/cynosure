"""过拟合分叉监控（ADR-0009-β，issue #105）的决定面与观测面。

验收面（issue #105 AC）：

- 分叉 EMA：首观测置值、α = 2/(span+1) 递推（跨度与 ADR-0008 EMA(AUC)
  同换算口径）、观测计数、落盘状态回填（restore 与 observe 的一对写
  入口）；
- 报警触发边界（越线发、不越线静默）：分叉 EMA 自下而上越过报警阈值
  即发（首次观测即越线 = 出生即分叉，同样发）、线上滞留不重发、回落后
  再越线重发；阈值点本身算越线（与门控 enter 判定的 ``>=`` 同语义）；
- per-condition 独立记账：条件间 EMA 与越线判定互不可见；
- 非有限浮点观测在测量层显式拒绝（「全流拒绝」口径的源头闸口）；
- state/adopt roundtrip 逐位一致（续训复原的落盘侧），损坏形态显式
  拒绝；
- 报警不动作：越线只报警——白名单成员与噪声 σ_max 都不被分叉监控
  触碰（测试断言无副作用，ADR-0009 决策 5）。

train 侧干净域复算（更新原语 seam）的观测缝在 test_online_update；
overfit_alert 事件契约（序列化 / 非有限拒绝 / 混存 / 回退记账）在
test_pretrain 的事件契约族；多 rank 归并序由 test_distributed 覆盖。
"""

import pytest

from cynosure.config import RewardConfig
from cynosure.distributed import DistributedContext
from cynosure.reward.overfit import DivergenceEma, OverfitMonitor
from cynosure.train.gating import DynamicWhitelist
from cynosure.train.whitelist import ConditionWhitelist


class OverfitFixture:
    """分叉监控单测的装配面：最小 RewardConfig 与监控对象的构造
    （工厂收拢为类，不留在模块级；与 test_gating.GatingFixture 同款）。"""

    @staticmethod
    def reward_config(**overrides) -> RewardConfig:
        """最小合法 RewardConfig（分叉 knobs 可覆写）。"""
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
    def monitor(cls, **reward_overrides) -> OverfitMonitor:
        """默认 knobs 的单条件监控器（阈值/跨度覆写经 reward_overrides）。"""
        return OverfitMonitor(cls.reward_config(**reward_overrides))


class TestDivergenceEma:
    """分叉 EMA 的递推语义（与 gating 的 ConditionAucEma 同换算口径）。"""

    def test_first_observation_seeds_value(self) -> None:
        ema = DivergenceEma(span=8)
        assert ema.value is None
        assert ema.observe(0.3) == pytest.approx(0.3)
        assert ema.value == pytest.approx(0.3)
        assert ema.count == 1

    def test_recursion_uses_span_implied_alpha(self) -> None:
        # span=8 → α = 2/9：1.0 后观测 0.0 → (1−α)·1.0 = 7/9
        ema = DivergenceEma(span=8)
        ema.observe(1.0)
        assert ema.observe(0.0) == pytest.approx(7.0 / 9.0)
        assert ema.count == 2

    def test_restore_refills_state_bitwise(self) -> None:
        ema = DivergenceEma(span=4)
        ema.observe(0.1)
        ema.observe(0.2)
        # 回填到显式状态后，递推链从回填点续写
        ema.restore(0.9, count=3)
        assert ema.value == pytest.approx(0.9)
        assert ema.count == 3
        alpha = 2.0 / 5.0
        assert ema.observe(0.0) == pytest.approx((1 - alpha) * 0.9)

    def test_invalid_span_rejected(self) -> None:
        with pytest.raises(ValueError):
            DivergenceEma(span=0)

    def test_restore_rejects_zero_count(self) -> None:
        ema = DivergenceEma(span=4)
        with pytest.raises(ValueError):
            ema.restore(0.5, count=0)


class TestAlertBoundary:
    """报警触发边界（越线发、不越线静默；越线判定 = EMA 自下而上
    达到阈值）。"""

    def test_below_threshold_is_silent(self) -> None:
        monitor = OverfitFixture.monitor(overfit_alert_divergence=0.2)
        reading = monitor.observe("t1n", train_pairwise_acc=0.6, heldout_auc=0.55)  # 分叉 0.05 < 0.2
        assert reading.alerted is False
        assert reading.divergence == pytest.approx(0.05)

    def test_crossing_fires_exactly_at_the_crossing(self) -> None:
        # span=3 → α = 0.5（二进制精确，边界判定不吃 FP 噪声）：
        # e ← (e + x)/2。观测 0.0 → e=0（线下）；观测 0.5 → e=0.25（恰
        # 阈值，触发）；线上滞留 → e=0.375，静默
        monitor = OverfitFixture.monitor(
            overfit_ema_span=3, overfit_alert_divergence=0.25,
        )
        assert monitor.observe("t1n", train_pairwise_acc=0.5, heldout_auc=0.5).alerted is False
        assert monitor.observe("t1n", train_pairwise_acc=1.0, heldout_auc=0.5).alerted is True
        assert monitor.observe("t1n", train_pairwise_acc=1.0, heldout_auc=0.5).alerted is False

    def test_first_observation_above_threshold_fires(self) -> None:
        # 出生即分叉（如 warm-start 判别器已记住预训练池）：首观测即越线
        monitor = OverfitFixture.monitor(overfit_alert_divergence=0.2)
        reading = monitor.observe("t2w", train_pairwise_acc=0.9, heldout_auc=0.5)
        assert reading.alerted is True

    def test_threshold_point_itself_counts_as_crossed(self) -> None:
        # 阈值点算越线（与门控 enter 判定的 >= 同语义）
        monitor = OverfitFixture.monitor(
            overfit_ema_span=3, overfit_alert_divergence=0.25,
        )
        reading = monitor.observe("t1n", train_pairwise_acc=0.75, heldout_auc=0.5)
        assert reading.divergence == 0.25  # EMA 首观测 = 分叉样本本身
        assert reading.alerted is True

    def test_recrossing_refires(self) -> None:
        # span=3 → α = 0.5：0.5 → fire（e=0.5）→ 滞留静默（e=0.5）→
        # 跌回线下（e=0.0）→ 0.5 → e=0.25 恰阈值：状态机重新武装并再次报警
        monitor = OverfitFixture.monitor(
            overfit_ema_span=3, overfit_alert_divergence=0.25,
        )
        assert monitor.observe("t1n", train_pairwise_acc=1.0, heldout_auc=0.5).alerted is True
        assert monitor.observe("t1n", train_pairwise_acc=1.0, heldout_auc=0.5).alerted is False
        assert monitor.observe("t1n", train_pairwise_acc=0.0, heldout_auc=0.5).alerted is False
        assert monitor.observe("t1n", train_pairwise_acc=1.0, heldout_auc=0.5).alerted is True

    def test_per_condition_independent(self) -> None:
        # 条件间 EMA 与越线判定互不可见（per-condition 独立记账）
        monitor = OverfitFixture.monitor(overfit_alert_divergence=0.2)
        assert monitor.observe("t1n", train_pairwise_acc=0.9, heldout_auc=0.5).alerted is True
        # 另一条件同样观测：独立触警，不受 t1n 已越线影响
        assert monitor.observe("t2w", train_pairwise_acc=0.9, heldout_auc=0.5).alerted is True
        # t2w 线上滞留静默不影响 t1n 的静默（各自边沿各自记）
        assert monitor.observe("t1n", train_pairwise_acc=0.9, heldout_auc=0.5).alerted is False
        assert monitor.observe("t2w", train_pairwise_acc=0.9, heldout_auc=0.5).alerted is False
        state = monitor.state()["ema"]
        assert set(state) == {"t1n", "t2w"}
        assert state["t1n"] == state["t2w"]  # 同观测序列 → 同 EMA 终值

    def test_non_finite_observation_rejected(self) -> None:
        # 非有限浮点在测量层显式拒绝（判别器数值发散不在监控面伪装成
        # 分叉值）——「全流拒绝」口径的源头闸口
        monitor = OverfitFixture.monitor()
        with pytest.raises(ValueError):
            monitor.observe("t1n", train_pairwise_acc=float("nan"), heldout_auc=0.5)
        with pytest.raises(ValueError):
            monitor.observe("t1n", train_pairwise_acc=0.5, heldout_auc=float("inf"))
        assert monitor.state() == {"ema": {}}  # 拒绝后状态不动


class TestOverfitState:
    """监控状态的落盘形态与回填（续训复原的落盘侧，resume v6）。"""

    def test_state_roundtrip_bitwise(self) -> None:
        source = OverfitFixture.monitor(overfit_ema_span=4)
        source.observe("t2w", train_pairwise_acc=0.6, heldout_auc=0.5)
        source.observe("t2w", train_pairwise_acc=0.8, heldout_auc=0.5)
        source.observe("t2f", train_pairwise_acc=0.5, heldout_auc=0.5)
        state = source.state()

        target = OverfitFixture.monitor(overfit_ema_span=4)
        target.adopt(state)
        assert target.state() == source.state()
        # 回填后的状态机继续同轨迹演化（后续观测逐位一致）
        source.observe("t2w", train_pairwise_acc=0.7, heldout_auc=0.5)
        target.observe("t2w", train_pairwise_acc=0.7, heldout_auc=0.5)
        assert target.state() == source.state()

    def test_adopt_rejects_malformed_state(self) -> None:
        monitor = OverfitFixture.monitor()
        for bad in (
            {},
            {"ema": {}, "extra": 1},  # 形态非法（多余键）
            {"ema": {"t2w": 0.5}},  # 条目形态非法
            {"ema": {"t2w": {"value": 0.5}}},  # 缺 count
            {"ema": {"t2w": {"value": 0.5, "count": 0}}},  # 计数须 ≥ 1
            {"ema": {"mri": {"value": 0.5, "count": 1}}},  # 非法条件
            {"ema": {"t2w": {"value": None, "count": 1}}},  # value 非数值
            {"ema": {"t2w": {"value": 0.5, "count": True}}},  # count 须 int
        ):
            with pytest.raises(ValueError):
                monitor.adopt(bad)
        assert monitor.state() == {"ema": {}}  # 拒绝后状态不动

    def test_unobserved_conditions_absent_from_state(self) -> None:
        monitor = OverfitFixture.monitor()
        assert monitor.state() == {"ema": {}}


class TestAlertDoesNotAct:
    """报警不动作（ADR-0009 决策 5）：分叉越线不自动改白名单、不自动
    调 σ——监控器只产读数，动作面（白名单/σ）不被它触碰（AC 的无
    副作用断言）。"""

    def test_crossing_alert_leaves_whitelist_and_sigma_untouched(self) -> None:
        config = OverfitFixture.reward_config()
        monitor = OverfitMonitor(config)
        whitelist = DynamicWhitelist(
            ConditionWhitelist.unrestricted(),
            config,
            DistributedContext(0, 1, False),
        )
        members_before = whitelist.whitelist.members
        sigma_before = config.disc_noise_sigma_max

        fired = [
            monitor.observe(modality, train_pairwise_acc=0.9, heldout_auc=0.5).alerted
            for modality in ("t1n", "t2w", "t2f")
        ]
        assert fired == [True, True, True]  # 持续越线、报警面在响

        assert whitelist.whitelist.members == members_before  # 名单不动
        assert config.disc_noise_sigma_max == sigma_before  # σ 不动
