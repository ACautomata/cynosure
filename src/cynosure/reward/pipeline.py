"""``prepare`` 子命令背后的编排：装配计划 → 预编码 → 统计量 → 工件落盘。

幂等契约（ticket #18）：装配（扫描/划分/抽样）、合成/生产预编码、统计量
归约全程确定性，重跑工件零漂移——prepare 产物是可重建的派生工件，覆盖
写是预期语义（与 train run 目录的「不静默覆盖」不同）。latent 子树与
manifest 随每次运行整体重建：**先失效旧件**、编码成功才落盘新件——失效
落在装配计划之前，计划期失败（元数据/split 错版、互斥命中、候选卷影像
缺失）的路径同样清盘：任何时刻盘上要么全量一致、要么明确缺失，不留
「索引指向缺失 latent」的悬挂工件，也不留「一次失败的运行 + 一套看起来
有效的上一轮产物」。
逐位幂等归测试口径（ADR-0011：生产 pipeline 不开确定性 kernel，生产
预编码的 VAE 前向重跑有浮点噪声级漂移；seeded 后验采样与统计量归约
本身仍逐位确定）。

两域装配（#121/#131，spec #125 实现决策 3）：数据装配语义按
``experiment.dataset`` 分派到策略（``reward.mrrate`` 的 BratsAssembly /
MrRateAssembly）——BraTS 线语义零改动（病例目录扫描 + 70/10/20），
MR-RATE 线 = 官方 split join + 评估集互斥守卫 + patient 级 held-out
二分 + 逐条件配额抽样。编排骨架（失效 → 计划 → 守卫 → 编码 →
统计量 → 落盘）两域单份；MR-RATE 域追加抽样 manifest 落盘与装配期守卫
（逐条件容量 ≥ K×world + 逐条件 held-out 覆盖），守卫消费装配计划的
计数口径、落在编码之前——稀疏条件与覆盖缺口在开工前失败。
"""

import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

from monai.data import MetaTensor
from pydantic import BaseModel, ConfigDict

import torch

from cynosure.conditions import MrConditionVocabulary
from cynosure.config import CynosureConfig
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import (
    ChannelStats,
    LatentManifest,
    ManifestKind,
    PoolEntry,
    PrepareProvenance,
    SamplingManifest,
)
from cynosure.reward.encoder import (
    LatentEncoder,
    MaisiLatentEncoder,
    SyntheticLatentEncoder,
)
from cynosure.reward.mrrate import (
    AssemblyPlan,
    AssemblyTask,
    BratsAssembly,
    MrRateAssembly,
)
from cynosure.reward.preprocessing import SpacingSidecar, UpstreamPreprocessChain


@dataclass
class PrepareReport:
    """一次 prepare 的结果摘要（CLI stdout 报告的数据）。"""

    pool_manifest: Path
    heldout_manifest: Path
    channel_stats: Path
    split_sizes: dict[str, int]
    pool_entries: int
    heldout_entries: int
    mean: list[float]
    std: list[float]
    condition_counts: dict[str, int] | None = None
    """逐条件编码计数（MR-RATE 域；BraTS 域 None——序列分层见
    manifest.modalities）。"""
    sampling_manifest: Path | None = None
    """配额抽样留痕工件（MR-RATE 域；BraTS 域 None）。"""


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
    """prepare 编排：依赖注入编码器与装配策略（fixture 与生产、BraTS 与
    MR-RATE 共用同一骨架，装配语义按域分派）。"""

    def __init__(
        self,
        config: CynosureConfig,
        encoder: LatentEncoder,
        vocabulary: MrConditionVocabulary | None = None,
    ) -> None:
        self._config = config
        self._encoder = encoder
        self._vocabulary = vocabulary
        # 读图编码走上游训练 recipe 六步链（data-preparation + ADR-0006 +
        # #130 两臂旋钮）：强度臂从 config 注入（两域锁死：BraTS clip=True /
        # MR-RATE clip=False）；resize 目标 BraTS 臂走 dim 公式（本链），
        # MR-RATE 臂按条件统一网格逐条件构造链（``_chain_for`` 缓存）
        self._preprocess = UpstreamPreprocessChain(
            resize_base=config.preprocessing.resize_base,
            clip_intensity=config.preprocessing.intensity_clip,
        )
        self._condition_chains: dict[str, UpstreamPreprocessChain] = {}
        # spacing 侧车（issue #46）：BraTS per-case raw header zooms ×1e2
        # 随条目落盘；MR-RATE 线 spacing 是条件属性，由装配任务携带
        self._spacing = SpacingSidecar()
        self._assembly = self.build_assembly(config, vocabulary)

    @staticmethod
    def build_assembly(
        config: CynosureConfig,
        vocabulary: MrConditionVocabulary | None = None,
    ) -> BratsAssembly | MrRateAssembly:
        """装配策略分派（Factory Method，与 ``build_encoder`` 同构的分派
        缝）：按 ``experiment.dataset`` 选数据域装配语义——MR-RATE 缺
        词汇表装载产物即拒绝（schema 已锁必填，此处防御直调路径）。"""
        if config.experiment.dataset == "MR-RATE":
            if vocabulary is None:
                raise ValueError(
                    "MR-RATE 装配须装载条件词汇表（MrConditionVocabulary）"
                    "——条件归属解析与统一网格/等效 spacing 的来源"
                )
            return MrRateAssembly(config, vocabulary)
        return BratsAssembly(config)

    @staticmethod
    def build_encoder(
        config: CynosureConfig, device: torch.device,
    ) -> LatentEncoder:
        """编码器策略分派（Factory Method，CLI 与单测共用的装配缝）：
        fixture 合成（device 无关）/ 生产 VAE 预编码（``vae_config_json``
        + ``vae_ckpt`` 工件对经 netbuild 严格装载）。生产 config 缺
        VAE 网络配置工件显式拒绝（工件存在性属运行时契约，schema 不拦）。"""
        if config.fixture_mode:
            return SyntheticLatentEncoder()
        if config.artifacts.vae_config_json is None:
            raise ValueError(
                "生产 prepare 须 VAE 网络配置工件（artifacts.vae_config_json，"
                "与 vae_ckpt 成对经 netbuild 装载 AutoencoderKlMaisi），"
                "当前为 None"
            )
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.vae_config_json),
            checkpoint=config.artifacts.vae_ckpt,
        )
        # 滑窗参数走既有 spec 字段通道（#143，与 decode 的 roi/overlap 同构）
        return MaisiLatentEncoder(
            artifact,
            device,
            roi_size=tuple(config.preprocessing.encode_roi_size),
            overlap=config.preprocessing.encode_overlap,
        )

    @staticmethod
    def load_vocabulary(config: CynosureConfig) -> MrConditionVocabulary | None:
        """MR-RATE 域的条件词汇表装载（工件唯一来源，fixture_mode 通道
        透传）；BraTS 域 None（四序列常量语义在装配策略内）。"""
        if config.experiment.dataset != "MR-RATE":
            return None
        return MrConditionVocabulary.load(
            config.artifacts.condition_vocabulary_json,
            fixture_mode=config.fixture_mode,
        )

    def run(self) -> PrepareReport:
        pool = self._build_summary("real_pool", self._config.reward.real_pool_manifest)
        heldout = self._build_summary(
            "heldout_real", self._config.reward.heldout_real_manifest,
        )
        self._invalidate(pool, heldout)  # 先失效旧件：编码成功才落新件
        self._invalidate_derived()  # 单文件派生件（统计量/抽样留痕）同批失效
        # 失效先于装配计划：计划本身也会失败（元数据/split 错版、互斥命中、
        # 候选卷影像缺失）——失败重跑若把上一轮工件原地留着，盘上就是
        # 「一次失败的运行 + 一套看起来有效的产物」；「要么全量一致、要么
        # 明确缺失」须覆盖计划期失败，不能只在编码期成立
        plan = self._assembly.plan()
        # 装配期守卫落在编码**之前**：逐条件计数在装配计划里已经完备，
        # 稀疏模态小池与 held-out 覆盖缺口在开工前失败（spec「开工前失败
        # 而非训练中途」），不白烧一遍全量预编码、盘上不留半个 latent
        self._guard_assembly(plan)
        stats = ChannelRunningStats()
        is_mr = plan.is_mr_rate
        pool_entries = self._encode_tasks(plan.pool, pool, stats, is_mr)
        heldout_entries = self._encode_tasks(plan.heldout, heldout, None, is_mr)
        mean, std = stats.finalize()
        pool_manifest = self._build_manifest(
            pool, pool_entries, plan, self._manifest_extras(plan, pool_entries),
        )
        heldout_manifest = self._build_manifest(
            heldout, heldout_entries, plan,
            self._manifest_extras(plan, heldout_entries),
        )
        self._persist(pool, pool_manifest)
        self._persist(heldout, heldout_manifest)
        self._write_stats(plan, mean, std, len(pool_entries))
        sampling_manifest = self._write_sampling_manifest(plan)
        condition_counts = (
            dict(self._condition_counts(pool_entries)) if plan.is_mr_rate else None
        )
        return PrepareReport(
            pool_manifest=pool.manifest_path,
            heldout_manifest=heldout.manifest_path,
            channel_stats=self._config.reward.channel_stats_json,
            split_sizes=plan.split_sizes,
            pool_entries=len(pool_entries),
            heldout_entries=len(heldout_entries),
            mean=mean, std=std,
            condition_counts=condition_counts,
            sampling_manifest=sampling_manifest,
        )

    def _manifest_extras(
        self, plan, entries: list[PoolEntry],
    ) -> dict | None:
        """MR-RATE 域 manifest 的附加字段（逐条件 latent 形状契约，按本
        manifest 条目实际出现的条件，形状权威 = 条件词汇表）；BraTS 域
        None（单条件词汇特例——全局 latent_shape 对账即完备，缺表合法）。
        条件分层计数由 manifest 从条目派生（``modalities``，单一来源）。"""
        if not plan.is_mr_rate:
            return None
        counts = self._condition_counts(entries)
        return {
            "condition_latent_shapes": {
                name: self._vocabulary.latent_shape(name)
                for name in counts
            },
        }

    def _condition_counts(
        self, entries: list[PoolEntry],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            counts[entry.modality] = counts.get(entry.modality, 0) + 1
        return counts

    def _guard_assembly(self, plan: AssemblyPlan) -> None:
        """MR-RATE 装配期守卫（编码之前、计数口径；#121 AC5）：

        - **逐条件容量**（ADR-0008-03）：逐（条件, 全量）容量 ≥
          ``disc_batch_size_k × world_size``，world_size 取 config 声明的
          部署宽度（``deployment.nproc_per_node``）——train 装配期的同一
          守卫按**真实 rank 数**判定同一份 manifest（
          ``TrainingRuntime.assemble_rewards``），prepare 侧钉死 1 等于把
          「开工前失败」推迟到烧完全量预编码之后才由 train 拒绝；条件全集
          = 词汇表 11 格——稀疏模态小池任一条件不足即装配期可读拒绝；
        - **逐条件 held-out 覆盖**：pool 侧出现的条件在 held-out 侧须有卷。
          patient 级二分是条件无关的全局洗牌（排序 + seed 洗牌 + 按
          ``heldout_fraction`` 切片），稀疏条件的患者可能整批落 pool 侧
          ——该条件在 held-out 工件缺条目，要到预训练装配期的轮转条件集
          守卫（ADR-0008-04「每条件 held-out 非空」）才炸，而根因（二分
          份额 / seed / 患者基数）在本侧。

        计数取自装配计划（``SamplingTrace``）：条目与编码任务一一对应，
        故与 manifest 侧 ``assert_condition_capacity`` 同一算术、同一报错。
        BraTS 域 no-op（同款守卫语义在 train 装配期，零改动）。
        """
        if not plan.is_mr_rate:
            return
        trace = plan.sampling_trace
        LatentManifest.assert_capacity_counts(
            trace.census_quota_taken,
            self._config.reward.disc_batch_size_k,
            self._config.deployment.nproc_per_node,
            self._vocabulary.names(),
        )
        uncovered = [
            (condition, count)
            for condition, count in sorted(trace.census_quota_taken.items())
            if trace.heldout_counts.get(condition, 0) < 1
        ]
        if uncovered:
            detail = ", ".join(
                f"{condition}（pool {count} 卷 / held-out 0 卷）"
                for condition, count in uncovered
            )
            raise ValueError(
                f"逐条件 held-out 覆盖不足：{detail}——patient 级二分是条件"
                "无关的全局洗牌，稀疏条件的患者可能整批落 pool 侧；该条件"
                "在 held-out real 工件缺条目 = per-condition AUC 归因无米下锅"
                "（预训练装配期按 ADR-0008-04 拒绝启动），而根因在本侧：增大 "
                "reward.heldout_fraction、增补该条件的患者，或调整 "
                "schedule.seed 后重跑"
            )

    def _invalidate_derived(self) -> None:
        """单文件派生件与 latent 子树同批失效（per-channel 统计量；配额
        抽样留痕按 config 声明）：失败路径（装配计划期或编码期）若把上一轮
        的抽样 manifest 留在原地，盘上就是「一次失败的运行 + 一份描述上一
        轮归属的审计工件」——「要么全量一致、要么明确缺失」须对全部产物
        成立。BraTS 域无抽样留痕（schema 的互斥绑定已钉死：MR-RATE 必填、
        BraTS 携带即拒），此处按声明判读、不重复 dataset 分派；也不依赖装配
        计划（失效落在计划之前——计划期失败的路径同样要清盘）。"""
        Path(self._config.reward.channel_stats_json).unlink(missing_ok=True)
        sampling_manifest = self._config.reward.sampling_manifest_json
        if sampling_manifest is not None:
            Path(sampling_manifest).unlink(missing_ok=True)

    def _write_sampling_manifest(self, plan) -> Path | None:
        """配额抽样留痕落盘（MR-RATE 域）：seed / 快照 / 配额 / 逐条件
        计数 / pool 与 held-out 逐卷归属 / 评估集互斥守卫读数——prepare
        幂等与 held-out 互斥「落档可查」的登记面（#131）。BraTS 域
        no-op。"""
        if not plan.is_mr_rate:
            return None
        trace = plan.sampling_trace
        manifest = SamplingManifest(
            seed=self._config.schedule.seed,
            data_snapshot=self._config.artifacts.mrrate_data_snapshot,
            quota=self._config.reward.real_pool_quota,
            heldout_fraction=self._config.reward.heldout_fraction,
            census_candidates=trace.census_candidates,
            census_quota_taken=trace.census_quota_taken,
            heldout_counts=trace.heldout_counts,
            out_of_vocabulary_volumes=trace.out_of_vocabulary_volumes,
            non_train_volumes=trace.non_train_volumes,
            eval_exclusion_keys=trace.eval_exclusion_keys,
            eval_exclusion_series_hits=trace.eval_exclusion_series_hits,
            eval_exclusion_patient_hits=trace.eval_exclusion_patient_hits,
            entries=trace.entries,
        )
        path = Path(self._config.reward.sampling_manifest_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return path

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

    @staticmethod
    def noise_seed(schedule_seed: int, case_id: str, stratification: str) -> int:
        """后验采样噪声的内容寻址种子（crc32 稳定派生，非负域内）：
        同（schedule seed, 卷键, 分层键）恒同种子——重跑零漂移；换键
        换种子——各（卷, 分层）的后验样本互异。31 位域在全语料 ~5e3 键
        下的生日碰撞概率约 0.6%，后果仅是两个样本共享同一 eps 序列
        （z_mu 基底各异，统计影响可忽略）。BraTS 键 =（病例, 序列）与
        既有工件零漂移；MR-RATE 键 =（卷, 生成条件）。"""
        digest = zlib.crc32(
            f"{schedule_seed}|{case_id}|{stratification}".encode("utf-8"),
        )
        return digest & 0x7FFFFFFF

    def _encode_tasks(
        self,
        tasks: list[AssemblyTask],
        summary: LatentSummary,
        stats: ChannelRunningStats | None,
        is_mr_rate: bool,
    ) -> list[PoolEntry]:
        """按装配计划编码（条目序 = 计划序：两域各自的确定性分层顺序）。
        stats 非 None 时同步累加统计量（train pool）。条件键两域同落
        ``modality``（#129 统一面：BraTS = 序列名、MR-RATE = 生成条件名）。"""
        entries: list[PoolEntry] = []
        for task in tasks:
            latent, spacing = self._encode_one(task, is_mr_rate)
            latent_dir = (
                summary.latent_root / task.stratification.replace("/", "_")
            )
            latent_dir.mkdir(parents=True, exist_ok=True)
            latent_path = latent_dir / f"{task.case_id.replace('/', '_')}.pt"
            torch.save(latent, latent_path)
            entries.append(PoolEntry(
                case_id=task.case_id,
                # 条件键两域同名（#129 统一面）：BraTS = 序列名、MR-RATE =
                # 生成条件名——判别器条件匹配采样与分层计数的同一归因轴
                modality=task.stratification,
                latent=latent_path.relative_to(
                    summary.manifest_path.parent,
                ).as_posix(),
                spacing=spacing,
            ))
            if stats is not None:
                stats.update(latent)
        return entries

    def _chain_for(self, task: AssemblyTask) -> UpstreamPreprocessChain:
        """任务 → 预处理链（#130 两臂装配）：BraTS 任务（resize 目标
        缺省）共用 dim 公式链；MR-RATE 任务按条件统一网格逐条件构造链
        （``target_grid`` 绑定的绝对目标），同条件复用缓存实例。"""
        if task.target_grid is None:
            return self._preprocess
        chain = self._condition_chains.get(task.stratification)
        if chain is None:
            chain = UpstreamPreprocessChain(
                resize_base=self._config.preprocessing.resize_base,
                clip_intensity=self._config.preprocessing.intensity_clip,
                target_grid=task.target_grid,
            )
            self._condition_chains[task.stratification] = chain
        return chain

    def _encode_one(
        self, task: AssemblyTask, is_mr_rate: bool,
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        """单（卷, 分层）的编码产物：(latent, spacing 侧车/条件属性)。"""
        # 宽捕获有据：第三方读取栈（MONAI reader、nibabel ImageFileError、压缩层）
        # 的异常类面不可枚举，nibabel 异常又不在 import 白名单内无法按类型接；
        # 保留异常链（from exc）不吞根因，原始类名入消息供分诊。
        try:
            image = self._chain_for(task)(task.image_path)
            spacing = (
                self._spacing.read(image) if task.spacing is None else task.spacing
            )
        except Exception as exc:
            raise ValueError(
                f"影像读取失败: {task.image_path}"
                f"（{type(exc).__name__}: {exc}）"
            ) from exc
        # 剥离 MetaTensor 的 numpy 元数据：工件落纯张量，weights_only 装载才可行
        # （链末端已是 [1, D, H, W]：EnsureChannelFirst 起通道在前）
        tensor = torch.as_tensor(
            image.as_tensor() if isinstance(image, MetaTensor) else image,
        )
        latent = self._encoder.encode(
            tensor, self.noise_seed(
                self._config.schedule.seed, task.case_id, task.stratification,
            ),
        )
        expected_shape = (
            self._vocabulary.latent_shape(task.stratification)
            if is_mr_rate else self._config.latent_shape
        )
        if tuple(latent.shape) != expected_shape:
            raise ValueError(
                f"预编码输出形状 {tuple(latent.shape)} 与 latent 形状契约"
                f" {expected_shape} 不符（输入 {tuple(tensor.shape)}）"
            )
        return latent, spacing

    def _build_manifest(
        self,
        summary: LatentSummary,
        entries: list[PoolEntry],
        plan,
        manifest_extras: dict | None,
    ) -> LatentManifest:
        """manifest 构造（内存对象）：守卫先于落盘消费它（容量守卫失败
        时盘上不留本次 manifest）。"""
        return LatentManifest(
            kind=summary.kind,
            encoder=self._encoder.name,
            # 全局 latent_shape 两域恒填（#129）：多条件域该值只作通道数
            # 对账锚——空间分量非权威（逐条目形状走 condition_latent_shapes
            # 的权威表）；BraTS 单条件词汇特例下全局对账即完备
            latent_shape=tuple(self._config.latent_shape),
            split_seed=self._config.schedule.seed,
            split_sizes=plan.split_sizes,
            entries=entries,  # 分层计数由条目派生（artifacts 层单一来源）
            **(manifest_extras or {}),
        )

    @staticmethod
    def _persist(summary: LatentSummary, manifest: LatentManifest) -> None:
        summary.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        summary.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8",
        )

    def _write_stats(
        self, plan, mean: list[float], std: list[float],
        num_latents: int,
    ) -> None:
        stats_path = Path(self._config.reward.channel_stats_json)
        provenance = None
        if plan.is_mr_rate:
            provenance = PrepareProvenance(
                dataset=self._config.experiment.dataset,
                data_snapshot=self._config.artifacts.mrrate_data_snapshot,
                source_commit=self._config.artifacts.source_commit,
                intensity_clip=self._config.preprocessing.intensity_clip,
                resize_semantics="uniform-grid",
                upstream_anchor=(
                    "NVIDIA v1 clip=False（上游 transforms.py 原文；#71 裁决）"
                ),
            )
        stats = ChannelStats(
            mean=mean,
            std=std,
            num_latents=num_latents,
            latent_shape=tuple(self._config.latent_shape),
            # 与 PoolEntry.latent 同一相对化机制（relative_to）：跨树布局在此
            # 显式失败，不静默产出 ../ 逃逸路径——stats 与 manifest 同目录是布局契约
            source_manifest=Path(
                self._config.reward.real_pool_manifest,
            ).relative_to(stats_path.parent).as_posix(),
            provenance=provenance,
        )
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
