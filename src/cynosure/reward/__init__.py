"""reward 模块：prepare 数据工件管线（ticket #18）与 Reward model 打分 /
在线更新 / 两区回放（ticket #20）。

- prepare（扫描 → split → 预编码 → 统计量 → 三工件落盘，幂等）：
  Real sample pool（train split，按序列分层）、Held-out real（val split，
  与 pool 病例级不相交、永不参与判别器更新）、per-channel 标准化统计量；
- Reward model（reward-model 章 + ADR-0001）：MONAI PatchDiscriminator
  封装（GroupNorm、raw real-logit patch 聚合、SpectralNorm 触发式）、
  LSGAN 在线更新一步（AdamW、50% 当前 / 50% 回放）、两区 Replay buffer
  （固定 base + FIFO recent）、held-out AUC 监控信号。

数值语义见 docs/spec/reward-model.md 与 ADR-0001。
"""

from cynosure.reward.artifacts import ChannelStats, LatentManifest, PoolEntry
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.buffer import ReplayBuffer, ReplayDraw, ReplayStore, ZoneSizes
from cynosure.reward.dataset import (
    BratsSeriesLayout,
    CaseSeries,
    CaseSplit,
    CaseSplitter,
)
from cynosure.reward.encoder import LatentEncoder, SyntheticLatentEncoder
from cynosure.reward.pipeline import PreparePipeline, PrepareReport
from cynosure.reward.preprocessing import UpstreamPreprocessChain
from cynosure.reward.sampler import RealPoolSampler, RealSampling
from cynosure.reward.scorer import (
    ChannelNormalizer,
    LatentScorer,
    LsganTerms,
    RewardScorer,
)
from cynosure.reward.update import OnlineUpdate, UpdateReport

__all__ = [
    "BratsSeriesLayout",
    "CaseSeries",
    "CaseSplit",
    "CaseSplitter",
    "ChannelNormalizer",
    "ChannelStats",
    "HeldOutAuc",
    "LatentEncoder",
    "LatentManifest",
    "LatentScorer",
    "LsganTerms",
    "OnlineUpdate",
    "PoolEntry",
    "PreparePipeline",
    "PrepareReport",
    "RealPoolSampler",
    "RealSampling",
    "ReplayBuffer",
    "ReplayDraw",
    "ReplayStore",
    "RewardScorer",
    "SyntheticLatentEncoder",
    "UpdateReport",
    "UpstreamPreprocessChain",
    "ZoneSizes",
]
