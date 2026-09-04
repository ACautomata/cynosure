"""prepare 数据工件契约（spec「产物工件契约」的 reward 侧最小集）。

三个工件均可被 train / eval 装载消费：

- **Real sample pool manifest**（``kind="real_pool"``）：train split 全量
  VAE 预编码 latent 的索引，按序列分层；
- **Held-out real manifest**（``kind="heldout_real"``）：val split 预编码
  latent，与 pool 病例级不相交、永不参与判别器更新；
- **per-channel 标准化统计量**（``kind="channel_stats"``）：判别器输入
  标准化所用 mean/std，来自 Real sample pool 所用训练集。

latent 张量本体不经 JSON：每条目一个 ``torch.save`` 文件，manifest 以相对
路径索引（``PoolEntry.latent``），训练侧可按条目懒加载切片。
字段名为契约最小集：施工可扩不可改名。
"""

import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from cynosure.config import Modality
from cynosure.reward.dataset import SplitPart


ManifestKind = Literal["real_pool", "heldout_real"]
"""预编码 latent manifest 的两种语义（pool = 判别器「真」训练侧；held-out =
out-of-sample 监控侧），装载时以 kind 守卫互换使用。"""


class PoolEntry(BaseModel):
    """manifest 条目：一病例一序列的一枚预编码 latent。"""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    modality: Modality
    latent: str
    """latent 文件路径，相对 manifest 文件所在目录。"""


class LatentManifest(BaseModel):
    """预编码 latent 索引（Real sample pool 与 Held-out real 共用契约，
    以 ``kind`` 区分）。"""

    model_config = ConfigDict(extra="forbid")

    _path: Path | None = PrivateAttr(default=None)
    """manifest 文件自身位置：条目相对路径的解析基准（load 时记录）。"""

    kind: ManifestKind
    encoder: str
    """预编码来源标识（fixture 合成 / 生产 MONAI VAE），随工件留痕——
    消费方可从工件本身区分 latent 出处。"""
    latent_shape: tuple[int, int, int, int]
    split_seed: int
    split_sizes: dict[SplitPart, int]
    """病例级 70/10/20 的三段病例数（train/val/test），split 全貌留痕。"""
    entries: list[PoolEntry]
    modalities: dict[Modality, int] = Field(default_factory=dict)
    """序列分层计数：不传入时由 entries 派生（单一来源），传入则须一致。"""

    @classmethod
    def load(cls, path: Path, kind: ManifestKind) -> "LatentManifest":
        """装载并校验 kind：train 侧不得把 held-out manifest 当 pool 消费
        （held-out 永不参与判别器更新，在装载层守住）。"""
        path = Path(path)
        manifest = cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if manifest.kind != kind:
            raise ValueError(
                f"工件 kind 不符：期望 {kind}，{path} 实为 {manifest.kind}"
            )
        manifest._path = path
        return manifest

    def load_latent(self, entry: PoolEntry) -> torch.Tensor:
        """装载单条 latent（按条目懒加载，供判别器 real 切片采样）。"""
        if self._path is None:
            raise ValueError(
                "本 manifest 非经 load() 装载，无条目路径解析基准"
                "（落盘侧用 save()，装载侧一律走 load()）"
            )
        if entry not in self.entries:
            raise ValueError(f"条目不属于本 manifest: {entry}")
        latent = torch.load(
            self._path.parent / entry.latent,
            map_location="cpu", weights_only=True,
        )
        if tuple(latent.shape) != self.latent_shape:
            raise ValueError(
                f"latent 形状 {tuple(latent.shape)} 与契约 {self.latent_shape} 不符:"
                f" {entry.latent}"
            )
        return latent

    @model_validator(mode="after")
    def _modalities_from_entries(self) -> "LatentManifest":
        counts: dict[Modality, int] = {}
        for entry in self.entries:
            counts[entry.modality] = counts.get(entry.modality, 0) + 1
        if self.modalities and self.modalities != counts:
            raise ValueError(
                f"modalities 计数 {self.modalities} 与条目实际分布 {counts} 不符"
            )
        self.modalities = counts
        return self


class ChannelStats(BaseModel):
    """判别器输入 per-channel 标准化统计量（来自 Real sample pool 所用训练集）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: Literal["channel_stats"] = "channel_stats"
    mean: list[float]
    std: list[float]
    num_latents: int
    latent_shape: tuple[int, int, int, int]
    source_manifest: str
    """来源 pool manifest 的路径（相对本文件）。"""

    @field_validator("std")
    @classmethod
    def _std_non_negative(cls, value: list[float]) -> list[float]:
        if any(component < 0 for component in value):
            raise ValueError(f"std 不得为负: {value}")
        return value

    @model_validator(mode="after")
    def _channels_match_shape(self) -> "ChannelStats":
        num_channels = self.latent_shape[0]
        if len(self.mean) != num_channels or len(self.std) != num_channels:
            raise ValueError(
                f"mean/std 长度必须等于 latent 通道数 {num_channels}，"
                f"得到 mean={len(self.mean)} std={len(self.std)}"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        return cls.model_validate(
            json.loads(Path(path).read_text(encoding="utf-8")),
        )
