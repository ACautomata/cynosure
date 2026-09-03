"""prepare 数据工件管线（spec #15「产物工件契约」reward 侧，ticket #18）。

扫描 dataset_root（BraTS 病例四序列布局）→ 病例级 70/10/20 确定性划分 →
VAE 预编码 → per-channel 统计量 → 三工件落盘（幂等，可被 train/eval 装载）：

- Real sample pool（train split，按序列分层）；
- Held-out real（val split，与 pool 病例级不相交、永不参与判别器更新）；
- per-channel 标准化统计量（来自 pool 所用训练集）。

判别器 / 在线更新 / Replay buffer 由后续 ticket 填充；数值语义见
docs/spec/reward-model.md 与 ADR-0001。
"""

from cynosure.reward.artifacts import ChannelStats, LatentManifest, PoolEntry
from cynosure.reward.dataset import (
    BratsSeriesLayout,
    CaseSeries,
    CaseSplit,
    CaseSplitter,
)
from cynosure.reward.encoder import LatentEncoder, SyntheticLatentEncoder
from cynosure.reward.pipeline import PreparePipeline, PrepareReport

__all__ = [
    "BratsSeriesLayout",
    "CaseSeries",
    "CaseSplit",
    "CaseSplitter",
    "ChannelStats",
    "LatentEncoder",
    "LatentManifest",
    "PoolEntry",
    "PreparePipeline",
    "PrepareReport",
    "SyntheticLatentEncoder",
]
