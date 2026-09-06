"""早停判定单测（AC：合成指标流注入三例——plateau 触发、hacking 签名
触发、不误触发）。

判定器消费训练指标流的已解析事件（dict 列表，与 metrics.jsonl 读回
同构）：主判据 FID 连续 N_plateau 个里程碑无改善 → plateau 早停；
最新 held-out AUC 近 chance 且 eval reward 斜率为正 → hacking 早停。
"""

import pytest

from cynosure.config import CynosureConfig
from cynosure.train.earlystop import EarlyStopJudge, EarlyStopVerdict
from tests.conftest import MINIMAL_CONFIG_DICT


class SyntheticStream:
    """合成指标流构造器：iter / milestone 事件的紧凑工厂。"""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def iter_event(
        self, iteration: int, reward: float, auc: float = 0.9,
        modality: str | None = None,
    ) -> "SyntheticStream":
        event = {
            "event": "iter", "iteration": iteration,
            "anchor_eval_reward": reward, "heldout_auc": auc,
        }
        if modality is not None:
            event["modality"] = modality
        self.events.append(event)
        return self

    def milestone_event(self, iteration: int, fid: float) -> "SyntheticStream":
        self.events.append({
            "event": "milestone", "iteration": iteration, "fid": fid,
        })
        return self

    @classmethod
    def plateau_free(cls) -> "SyntheticStream":
        """主判据持续改善（plateau 通道关闭）的流背景。"""
        stream = cls()
        for index, fid in enumerate((10.0, 8.0, 6.0, 4.0), start=1):
            stream.milestone_event(50 * index, fid)
        return stream

    def judge(self, current_fid: float | None = None) -> EarlyStopVerdict:
        config = CynosureConfig.model_validate({
            **MINIMAL_CONFIG_DICT,
            "schedule": {"seed": 0, "plateau_tolerance": 0.5, "n_plateau": 3},
        })
        return EarlyStopJudge(config).judge(self.events, current_fid=current_fid)


class TestPlateauTrigger:
    def test_three_non_improving_milestones_stop(self) -> None:
        """里程碑 FID 相对历史最优改善 ≤ 容差连续 N_plateau 次 → 早停。"""
        stream = SyntheticStream()
        stream.milestone_event(50, 10.0)   # 历史最优 10.0
        stream.milestone_event(100, 9.9)   # 改善 0.1 ≤ 0.5 → plateau 1
        stream.milestone_event(150, 9.8)   # plateau 2
        stream.milestone_event(200, 9.7)   # plateau 3 → 停
        verdict = stream.judge()
        assert verdict.stop is True
        assert verdict.reason == "plateau"
        assert verdict.plateau_stalled is True
        assert verdict.hacking_signature is False

    def test_improvement_resets_stall_counter(self) -> None:
        """中途一次显著改善（> 容差）清零连续计数——plateau 必须连续。"""
        stream = SyntheticStream()
        stream.milestone_event(50, 10.0)
        stream.milestone_event(100, 9.9)   # plateau 1
        stream.milestone_event(150, 5.0)   # 显著改善 → 计数清零
        stream.milestone_event(200, 4.9)   # plateau 1
        stream.milestone_event(250, 4.8)   # plateau 2（不足 3）
        verdict = stream.judge()
        assert verdict.stop is False


class TestHackingTrigger:
    def _plateau_free_stream(self) -> SyntheticStream:
        return SyntheticStream.plateau_free()

    def test_auc_near_chance_with_rising_reward_stops(self) -> None:
        stream = self._plateau_free_stream()
        for iteration in range(12):
            stream.iter_event(iteration, reward=-1.0 + 0.5 * iteration, auc=0.505)
        verdict = stream.judge()
        assert verdict.stop is True
        assert verdict.reason == "reward_hacking"
        assert verdict.hacking_signature is True

    def test_auc_near_chance_with_falling_reward_does_not_stop(self) -> None:
        """签名必须两要素齐备：AUC 近 chance 但 eval reward 在降 → 不停。"""
        stream = self._plateau_free_stream()
        for iteration in range(12):
            stream.iter_event(iteration, reward=10.0 - 0.5 * iteration, auc=0.505)
        verdict = stream.judge()
        assert verdict.stop is False
        assert verdict.hacking_signature is False

    def test_auc_away_from_chance_with_rising_reward_does_not_stop(self) -> None:
        """判别器仍保有 out-of-sample 判别力（AUC 离 chance 远）→ 不停。"""
        stream = self._plateau_free_stream()
        for iteration in range(12):
            stream.iter_event(iteration, reward=-1.0 - 0.5 * iteration, auc=0.9)
        verdict = stream.judge()
        assert verdict.stop is False
        assert verdict.hacking_signature is False

    def test_insufficient_window_does_not_stop(self) -> None:
        """iter 事件不足 reward_trend_window：reward 趋势证据不足 → 不停。"""
        stream = self._plateau_free_stream()
        for iteration in range(5):  # < 默认窗口 10
            stream.iter_event(iteration, reward=-1.0 - 0.5 * iteration, auc=0.505)
        verdict = stream.judge()
        assert verdict.stop is False


class TestPerModalityHacking:
    """hacking 签名按目标序列分层（spec：各项指标按目标序列分层出）：
    AUC 证据与 reward 趋势须**同模态配对**——跨模态基线差不制造幻影
    斜率，坍缩模态不被健康模态的最新事件遮蔽。"""

    def test_collapsed_modality_not_hidden_behind_healthy_latest(self) -> None:
        """坍缩模态（AUC 近 chance 且自身 reward 升）先于健康模态落流：
        最新事件属于健康模态（AUC 远 chance）也不得遮蔽坍缩模态的签名。"""
        stream = SyntheticStream.plateau_free()
        for iteration in range(12):  # 坍缩模态在前
            stream.iter_event(
                iteration, reward=-1.0 + 0.5 * iteration, auc=0.505,
                modality="t1n",
            )
        for iteration in range(12):  # 健康模态殿后（最新事件的模态）
            stream.iter_event(
                100 + iteration, reward=-1.0 + 0.1 * iteration, auc=0.9,
                modality="t2w",
            )
        verdict = stream.judge()
        assert verdict.stop is True
        assert verdict.reason == "reward_hacking"
        assert verdict.hacking_signature is True

    def test_cross_modality_baselines_do_not_create_phantom_slope(self) -> None:
        """reward 窗口按模态分层：不同模态的基线差在混合窗口里制造幻影
        正斜率——两模态各自 reward 都在降时必须不停。"""
        stream = SyntheticStream.plateau_free()
        for step in range(10):
            stream.iter_event(
                step, reward=10.0 - step, auc=0.9, modality="t1n",
            )
            stream.iter_event(
                step, reward=110.0 - step, auc=0.505, modality="t2w",
            )
        verdict = stream.judge()
        assert verdict.stop is False
        assert verdict.hacking_signature is False

    def test_modality_window_filled_by_its_own_events(self) -> None:
        """窗口按模态各自计满：某模态自身 iter 事件不足窗口 → 证据不足
        不停（与全流口径同一「宁缓停、不误停」语义）。"""
        stream = SyntheticStream.plateau_free()
        for iteration in range(5):  # < 默认窗口 10
            stream.iter_event(
                iteration, reward=-1.0 - 0.5 * iteration, auc=0.505,
                modality="t1n",
            )
        verdict = stream.judge()
        assert verdict.stop is False


class TestNoFalseTrigger:
    def test_healthy_training_never_stops(self) -> None:
        """FID 稳步下降 + AUC 远离 chance + reward 有升有降 → 不误触发。"""
        stream = SyntheticStream()
        for index in range(6):
            stream.milestone_event(50 * (index + 1), 10.0 - index)
        for iteration in range(20):
            stream.iter_event(
                iteration, reward=-1.0 + 0.1 * (iteration % 3), auc=0.85,
            )
        verdict = stream.judge()
        assert verdict.stop is False
        assert verdict.reason is None

    def test_single_stalled_milestone_does_not_stop(self) -> None:
        stream = SyntheticStream()
        stream.milestone_event(50, 10.0)
        stream.milestone_event(100, 9.9)
        verdict = stream.judge()
        assert verdict.stop is False

    def test_empty_stream_does_not_stop(self) -> None:
        verdict = SyntheticStream().judge()
        assert verdict.stop is False

    def test_events_after_stop_iteration_are_not_required(self) -> None:
        """判定是流的前缀函数：同一前缀两次判定结果一致（消费指标流、
        无隐藏状态）。"""
        stream = SyntheticStream()
        stream.milestone_event(50, 10.0)
        stream.milestone_event(100, 9.9)
        stream.milestone_event(150, 9.8)
        stream.milestone_event(200, 9.7)
        first = stream.judge()
        second = stream.judge()
        assert first.stop == second.stop is True

    def test_pending_current_fid_counts_toward_stall(self) -> None:
        """连续第 N 个 plateau 里程碑在事件落盘前即触发：train 循环以
        ``current_fid`` 把当前里程碑判据递给判定器（milestone 事件写入
        前调用），不滞后一个里程碑。"""
        config = CynosureConfig.model_validate({
            **MINIMAL_CONFIG_DICT,
            "schedule": {"seed": 0, "n_plateau": 3, "plateau_tolerance": 0.5},
        })
        judge = EarlyStopJudge(config)
        stream_events = [
            {"event": "milestone", "iteration": 50, "fid": 10.0},
            {"event": "milestone", "iteration": 100, "fid": 9.9},
            {"event": "milestone", "iteration": 150, "fid": 9.8},
        ]
        assert judge.judge(stream_events).stop is False  # 盘上仅 2 次连续 plateau
        assert judge.judge(stream_events, current_fid=9.7).stop is True  # 当前值入判


class TestJudgeConfigSensitivity:
    def test_n_plateau_one_stops_immediately_on_first_stall(self) -> None:
        config = CynosureConfig.model_validate({
            **MINIMAL_CONFIG_DICT,
            "schedule": {"seed": 0, "n_plateau": 1, "plateau_tolerance": 0.5},
        })
        judge = EarlyStopJudge(config)
        verdict = judge.judge([
            {"event": "milestone", "iteration": 50, "fid": 10.0},
            {"event": "milestone", "iteration": 100, "fid": 9.9},
        ])
        assert verdict.stop is True

    def test_tolerance_zero_requires_strict_improvement(self) -> None:
        config = CynosureConfig.model_validate({
            **MINIMAL_CONFIG_DICT,
            "schedule": {"seed": 0, "plateau_tolerance": 0.0, "n_plateau": 2},
        })
        events = [
            {"event": "milestone", "iteration": 50, "fid": 10.0},
            {"event": "milestone", "iteration": 100, "fid": 10.0},
            {"event": "milestone", "iteration": 150, "fid": 10.0},
        ]
        assert EarlyStopJudge(config).judge(events).stop is True
