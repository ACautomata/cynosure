"""早停判定（experiment-design「成功判据/早停」+ 早停数据流补钉）。

判定器消费**训练指标流**（metrics.jsonl 读回的事件 dict 列表）：早停
判定在 train 进程内进行、判定输入只有流本身——合成流注入即可单测
（AC），流外无隐藏状态（同一前缀必得同一判定）。

两条触发通道（命中其一即停；AC「plateau 触发、hacking 签名触发、
不误触发三例」把两通道列为并列的触发例，故读作或而非与——ticket
原文「plateau + 判据不再升 + hacking 签名」的加号读法存歧义，如有
维护者另裁按与关系收窄，只需改本类的 stop 组装）：

- **plateau**：主判据（里程碑 FID，越低越好）相对历史最优的改善
  ≤ ``plateau_tolerance`` 连续 ``n_plateau`` 个里程碑——「主判据连续
  N_plateau 个里程碑 plateau + 判据不再升」的执行化；
- **hacking 签名**（reward-model 章）：最新 held-out AUC 落在 chance
  ± ``auc_chance_epsilon`` 带内 **且** 最近 ``reward_trend_window`` 个
  iter 的 anchor eval reward 线性斜率 > 0——判别器已无 out-of-sample
  判别力而 reward 曲线仍在升 = reward hacking 的典型签名。证据不足
  （iter 事件不足窗口）按未触发处理，不猜。
"""

from dataclasses import dataclass

import torch

from cynosure.config import CynosureConfig


@dataclass(frozen=True)
class EarlyStopVerdict:
    """一次早停判定的结论（里程碑事件 criteria_summary 的来源）。"""

    stop: bool
    reason: str | None
    """触发通道："plateau" / "reward_hacking"；未停为 None。"""
    plateau_stalled: bool
    hacking_signature: bool


class EarlyStopJudge:
    """指标流的早停判定器（纯函数式：判定 = f(事件前缀)）。"""

    def __init__(self, config: CynosureConfig) -> None:
        self._n_plateau = config.schedule.n_plateau
        self._plateau_tolerance = config.schedule.plateau_tolerance
        self._auc_chance_epsilon = config.schedule.auc_chance_epsilon
        self._reward_trend_window = config.schedule.reward_trend_window

    def judge(
        self, events: list[dict], current_fid: float | None = None,
    ) -> EarlyStopVerdict:
        """消费指标流事件（iter / milestone 混合、按落盘序）给出判定。

        ``current_fid``：尚未写入流的当前里程碑主判据值（train 循环在
        ``milestone`` 事件落盘前调用本判定——连续第 N 个 plateau 里程碑
        发生时即触发，而非滞后一个里程碑）。
        """
        milestones = [
            event for event in events if event.get("event") == "milestone"
        ]
        if current_fid is not None:
            milestones = [*milestones, {"fid": current_fid}]
        plateau_stalled = self._plateau_stalled(milestones)
        hacking_signature = self._hacking_signature(
            [event for event in events if event.get("event") == "iter"],
        )
        if plateau_stalled:
            return EarlyStopVerdict(
                stop=True, reason="plateau",
                plateau_stalled=True, hacking_signature=hacking_signature,
            )
        if hacking_signature:
            return EarlyStopVerdict(
                stop=True, reason="reward_hacking",
                plateau_stalled=False, hacking_signature=True,
            )
        return EarlyStopVerdict(
            stop=False, reason=None,
            plateau_stalled=False, hacking_signature=False,
        )

    def _plateau_stalled(self, milestones: list[dict]) -> bool:
        """主判据连续 N_plateau 个里程碑无显著改善（首个里程碑只立基准）。"""
        best: float | None = None
        stalled = 0
        for event in milestones:
            fid = float(event["fid"])
            if best is None or fid < best - self._plateau_tolerance:
                best = fid
                stalled = 0
            else:
                stalled += 1
        return stalled >= self._n_plateau

    def _hacking_signature(self, iterations: list[dict]) -> bool:
        """AUC 近 chance 且 eval reward 仍升（两要素齐备才成立）。"""
        if not iterations:
            return False
        latest_auc = float(iterations[-1]["heldout_auc"])
        if abs(latest_auc - 0.5) > self._auc_chance_epsilon:
            return False
        window = iterations[-self._reward_trend_window:]
        if len(window) < self._reward_trend_window:
            return False  # 趋势证据不足：不猜（宁缓停、不误停）
        rewards = torch.tensor(
            [float(event["anchor_eval_reward"]) for event in window],
            dtype=torch.float64,
        )
        steps = torch.arange(rewards.shape[0], dtype=torch.float64)
        slope = self._least_squares_slope(steps, rewards)
        return bool(slope > 0.0)

    @staticmethod
    def _least_squares_slope(
        steps: torch.Tensor, values: torch.Tensor,
    ) -> torch.Tensor:
        centered_steps = steps - steps.mean()
        centered_values = values - values.mean()
        return (centered_steps * centered_values).sum() / (centered_steps ** 2).sum()
