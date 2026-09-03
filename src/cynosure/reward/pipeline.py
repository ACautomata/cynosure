"""``prepare`` 子命令背后的编排：扫描 → split → 预编码 → 统计量 → 三工件落盘。

幂等契约（ticket #18）：数据扫描、病例级划分（seed 洗牌）、合成/生产预编码、
统计量归约全程确定性，重跑工件零漂移——prepare 产物是可重建的派生工件，
覆盖写是预期语义（与 train run 目录的「不静默覆盖」不同）。latent 子树与
manifest 随每次运行整体重建：先失效旧件、编码成功才落盘新件——任何时刻
盘上要么全量一致、要么明确缺失，不留「索引指向缺失 latent」的悬挂工件。
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from monai.data import MetaTensor
from monai.transforms import LoadImage
from pydantic import BaseModel, ConfigDict

import torch

from cynosure.config import CynosureConfig, MODALITIES
from cynosure.reward.artifacts import (
    ChannelStats,
    LatentManifest,
    ManifestKind,
    PoolEntry,
)
from cynosure.reward.dataset import (
    BratsSeriesLayout,
    CaseSeries,
    CaseSplit,
    CaseSplitter,
    SplitPart,
)
from cynosure.reward.encoder import LatentEncoder


@dataclass
class PrepareReport:
    """一次 prepare 的结果摘要（CLI stdout 报告的数据）。"""

    pool_manifest: Path
    heldout_manifest: Path
    channel_stats: Path
    split_sizes: dict[SplitPart, int]
    pool_entries: int
    heldout_entries: int
    mean: list[float]
    std: list[float]


class ChannelRunningStats:
    """per-channel 一二阶矩的运行期累加（float64）：全量 mean/std 的单遍归约。"""

    def __init__(self) -> None:
        self._sum = torch.zeros(0, dtype=torch.float64)
        self._sum_sq = torch.zeros(0, dtype=torch.float64)
        self._numel = 0

    def update(self, latent: torch.Tensor) -> None:
        if not self._sum.numel():  # 首个样本定通道形状
            self._sum = torch.zeros(latent.shape[0], dtype=torch.float64)
            self._sum_sq = torch.zeros(latent.shape[0], dtype=torch.float64)
        values = latent.double()
        self._sum += values.sum(dim=(1, 2, 3))
        self._sum_sq += (values ** 2).sum(dim=(1, 2, 3))
        self._numel += latent[0].numel()  # 每通道空间元素数：通道和是 per-channel 的

    def finalize(self) -> tuple[list[float], list[float]]:
        """(mean, std)：二阶矩公式 E[x²] − E[x]²；样本量须非空。"""
        if self._numel == 0:
            raise ValueError("统计量归约未收到任何 latent")
        mean = self._sum / self._numel
        variance = self._sum_sq / self._numel - mean ** 2
        std = variance.clamp_min(0.0).sqrt()  # 浮点误差可致微量负方差
        return mean.tolist(), std.tolist()


class LatentSummary(BaseModel):
    """单侧工件（pool 或 held-out）的派生落盘位置：manifest 文件 + latent 子树。"""

    model_config = ConfigDict(extra="forbid")

    kind: ManifestKind
    manifest_path: Path
    latent_root: Path


class PreparePipeline:
    """prepare 编排：依赖注入数据布局、划分策略与预编码策略（fixture 与
    生产共用同一管线，仅编码器策略不同）。"""

    def __init__(self, config: CynosureConfig, encoder: LatentEncoder) -> None:
        self._config = config
        self._encoder = encoder
        self._layout = BratsSeriesLayout(config.artifacts.dataset_root)
        self._splitter = CaseSplitter(config.schedule.seed)
        self._reader = LoadImage(image_only=True)

    def run(self) -> PrepareReport:
        cases = self._layout.scan()
        split = self._splitter.split([case.case_id for case in cases])
        series_by_case: dict[str, CaseSeries] = {
            case.case_id: case for case in cases
        }
        pool = self._build_summary("real_pool", self._config.reward.real_pool_manifest)
        heldout = self._build_summary(
            "heldout_real", self._config.reward.heldout_real_manifest,
        )
        self._invalidate(pool, heldout)  # 先失效旧件：编码成功才落新件
        Path(self._config.reward.channel_stats_json).unlink(missing_ok=True)
        stats = ChannelRunningStats()
        pool_entries = self._encode_cases(split.train, series_by_case, pool, stats)
        heldout_entries = self._encode_cases(
            split.val, series_by_case, heldout, None,
        )
        mean, std = stats.finalize()
        self._write_manifest(pool, pool_entries, split)
        self._write_manifest(heldout, heldout_entries, split)
        self._write_stats(pool, mean, std, len(pool_entries))
        return PrepareReport(
            pool_manifest=pool.manifest_path,
            heldout_manifest=heldout.manifest_path,
            channel_stats=self._config.reward.channel_stats_json,
            split_sizes=split.sizes(),
            pool_entries=len(pool_entries),
            heldout_entries=len(heldout_entries),
            mean=mean, std=std,
        )

    @staticmethod
    def _invalidate(*summaries: LatentSummary) -> None:
        """删除旧 manifest 与 latent 子树（失败重跑不留悬挂索引）。"""
        for summary in summaries:
            shutil.rmtree(summary.latent_root, ignore_errors=True)
            summary.manifest_path.unlink(missing_ok=True)
            summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _build_summary(self, kind: ManifestKind, manifest_path: Path) -> LatentSummary:
        """manifest 与 latent 子树的落盘位置（子目录名随 manifest stem 派生，
        同目录共存不撞名）。"""
        manifest_path = Path(manifest_path)
        return LatentSummary(
            kind=kind,
            manifest_path=manifest_path,
            latent_root=manifest_path.parent / f"{manifest_path.stem}_latents",
        )

    def _encode_cases(
        self,
        case_ids: list[str],
        series_by_case: dict[str, CaseSeries],
        summary: LatentSummary,
        stats: ChannelRunningStats | None,
    ) -> list[PoolEntry]:
        """按（序列、病例）双键排序编码——manifest 条目的序列分层顺序。
        stats 非 None 时同步累加统计量（train pool）。"""
        entries: list[PoolEntry] = []
        for modality in MODALITIES:
            for case_id in sorted(case_ids):
                latent = self._encode_one(series_by_case[case_id].series[modality])
                latent_dir = summary.latent_root / modality
                latent_dir.mkdir(parents=True, exist_ok=True)
                latent_path = latent_dir / f"{case_id}.pt"
                torch.save(latent, latent_path)
                entries.append(PoolEntry(
                    case_id=case_id,
                    modality=modality,
                    latent=latent_path.relative_to(
                        summary.manifest_path.parent,
                    ).as_posix(),
                ))
                if stats is not None:
                    stats.update(latent)
        return entries

    def _encode_one(self, series_path: Path) -> torch.Tensor:
        # 宽捕获有据：第三方读取栈（MONAI reader、nibabel ImageFileError、压缩层）
        # 的异常类面不可枚举，nibabel 异常又不在 import 白名单内无法按类型接；
        # 保留异常链（from exc）不吞根因，原始类名入消息供分诊。
        try:
            loaded = self._reader(series_path)
        except Exception as exc:
            raise ValueError(
                f"影像读取失败: {series_path}"
                f"（{type(exc).__name__}: {exc}）"
            ) from exc
        # 剥离 MetaTensor 的 numpy 元数据：工件落纯张量，weights_only 装载才可行
        image = torch.as_tensor(
            loaded.as_tensor() if isinstance(loaded, MetaTensor) else loaded,
        ).unsqueeze(0)
        latent = self._encoder.encode(image)
        if tuple(latent.shape) != self._config.latent_shape:
            raise ValueError(
                f"预编码输出形状 {tuple(latent.shape)} 与 latent_shape 契约"
                f" {self._config.latent_shape} 不符（输入 {tuple(image.shape)}）"
            )
        return latent

    def _write_manifest(
        self, summary: LatentSummary, entries: list[PoolEntry], split: CaseSplit,
    ) -> None:
        manifest = LatentManifest(
            kind=summary.kind,
            encoder=self._encoder.name,
            latent_shape=self._config.latent_shape,
            split_seed=self._config.schedule.seed,
            split_sizes=split.sizes(),
            entries=entries,  # modalities 由条目派生（artifacts 层单一来源）
        )
        summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        summary.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )

    def _write_stats(
        self, pool: LatentSummary, mean: list[float], std: list[float],
        num_latents: int,
    ) -> None:
        stats_path = Path(self._config.reward.channel_stats_json)
        stats = ChannelStats(
            mean=mean,
            std=std,
            num_latents=num_latents,
            latent_shape=self._config.latent_shape,
            # 与 PoolEntry.latent 同一相对化机制（relative_to）：跨树布局在此
            # 显式失败，不静默产出 ../ 逃逸路径——stats 与 manifest 同目录是布局契约
            source_manifest=pool.manifest_path.relative_to(
                stats_path.parent,
            ).as_posix(),
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
