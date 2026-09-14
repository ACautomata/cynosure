"""预训练 run 目录与产物契约（ADR-0007：判别器 warm-start 的产物面）。

- **PretrainRun**：预训练 run 目录布局（config 快照 + metrics.jsonl +
  ``checkpoints/`` + 预训练报告）——与 train run 目录同惯例（不静默覆盖、
  唯一写者；单进程执行下 rank 0 独写退化为直写），但工件集最小：无
  Baseline manifest、无采样体（预训练不产出像素体，评测相不参与）；
- **PretrainReport**：预训练报告契约（kind 标识 + per-condition held-out
  AUC + 条件白名单 + 数据口径指纹）与守卫重载入口——组别不符 / kind 不符
  / 缺报告 / 旧格式（池化口径单标量，ADR-0008 之前）即拒绝装载（守卫哲学），
  判别器形态指纹对照后经 netbuild 严格装载路径还原。

判别器 checkpoint 与训练期产物同构（``NetworkAssembler.loadable_state_dict``
的可装载 state_dict；spectral norm 启用时携带参数化状态，装载面按形态
分派**逐位还原**——见 ``NetworkAssembler.discriminator``），下游消费点
= train 新 run 的 warm-start 守卫重载（``PretrainReport.load_discriminator``
按报告内相对路径还原）；resume 占位装配不消费本产物（随机初始化占位，
分片恢复覆写）。
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr, ValidationError

from cynosure.config import CynosureConfig, Modality
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.artifacts import ChannelStats
from cynosure.reward.scorer import RewardScorer

if TYPE_CHECKING:
    # 事件模型在 train.artifacts（指标流事件类型的集中地）；仅作类型标注
    # 使用——运行时 import 会与 train.runtime 的 PretrainReport 装配依赖
    # 成环（train 侧 warm-start 装载反向消费本模块）
    from cynosure.train.artifacts import PretrainEvent


class PretrainProvenance(BaseModel):
    """预训练数据口径指纹（下游门槛检查的口径对照面）。

    路径为 config 原值（相对语义随 config 自身），sha256 为**文件内容**
    指纹——口径不匹配（换了 ChannelStats 来源、manifest 重建、判别器
    形态不同）在指纹对照层显式暴露，不给静默通过留缝。
    """

    model_config = ConfigDict(extra="forbid")

    real_pool_manifest: str
    real_pool_manifest_sha256: str
    heldout_manifest: str
    heldout_manifest_sha256: str
    channel_stats: str
    channel_stats_sha256: str
    discriminator_config: str
    discriminator_config_sha256: str
    discriminator_ckpt: str
    """判别器 checkpoint 路径（相对预训练 run 目录，与报告同款相对形）。"""
    discriminator_ckpt_sha256: str
    """checkpoint 内容指纹：报告的白名单与 per-condition 实测值只对预
    训练落盘的这份权重负责——启动期重算废止（ADR-0008 决策 5）后，
    「测量对象 = 装载对象」由装载期指纹对照把守（同形态换权重显式
    拒绝，见 ``PretrainReport.load_discriminator``）。"""

    @staticmethod
    def digest(path: Path) -> str:
        """文件内容 sha256（口径指纹的计算单点）。"""
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PretrainReport(BaseModel):
    """判别器预训练报告（run 目录 ``pretrain_report.json`` 契约）。

    per-condition 口径（ADR-0008 决策 5）：``condition_auc`` = 每条件
    最终 held-out AUC，``gate_whitelist`` = 条件白名单——池化口径的
    ``final_heldout_auc`` 单标量已成历史格式，``load()`` 对其显式拒绝
    （BraTS 线旧报告同此路径），报告值与工件可复现对照。
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    _path: Path | None = PrivateAttr(default=None)
    """报告文件自身位置：``discriminator_ckpt`` 相对路径的解析基准。"""

    kind: Literal["pretrain_report"] = "pretrain_report"
    group: str
    """预训练组别（组1 modal-label / 组2 cross-modal；fake 分布不同的
    归因轴）。装载守卫与消费 config 的组别严格等值对照（
    ``assert_data_provenance``，#113）：per-condition AUC 与条件白名单
    在本组 fake 分布上测量，跨组消费是显式拒绝的错误、无配置开关可
    绕过。"""
    latent_shape: tuple[int, int, int, int]
    condition_auc: dict[Modality, float]
    """每条件最终 held-out AUC（该条件 held-out 全量卷的池化点估计）：
    白名单内条件 = 确认时刻「首测 + 换批复测」的较小者（保守口径；
    确认后判别器继续受训，该值与最终落盘 checkpoint 不必同快照——
    它是确认时刻的测量记录，上岗判定直接信任报告值，数据口径漂移由
    装载期指纹对照把守，ADR-0008 决策 5）；未过线条件 = 步数耗尽后对
    落盘 checkpoint 权重的补测值（同快照可对照，白名单空时拒绝报错的
    实测值来源）。"""
    gate_whitelist: list[Modality]
    """条件白名单（ADR-0008 决策 5 的 gate 产物）：复测确认过线的条件，
    轮转序。空名单 = 无条件达线——报告与 checkpoint 照常落盘供诊断
    （拒跑由 train gate 把守，诊断产物不丢）。"""
    steps_completed: int
    """完成的判别器更新步数（全部条件确认过线的终止路径 = 确认前的
    更新步数；步数上限路径 = 上限值减去其中的确认步——确认步不更新）。"""
    gate_auc: float
    """本次预训练采用的门槛阈值（报告留痕：阈值可配置，跨 run 可比性
    以报告值为准）。"""
    gate_passed: bool
    """终止成功判据是否通过：全部轮转条件都经「首测 + 换批复测」两次
    独立测量确认过线（False = 步数上限耗尽——白名单可能非空，已确认
    者仍在名单内；checkpoint 仍落盘供诊断，上岗与否由 train 侧读报告
    白名单判定，ADR-0008 决策 5）。"""
    discriminator_ckpt: str
    """判别器 checkpoint 路径（相对本报告文件所在目录；可装载
    state_dict，与训练期产物 checkpoint 同构）。"""
    provenance: PretrainProvenance

    @classmethod
    def load(cls, path: Path) -> "PretrainReport":
        """装载并校验报告：缺报告 / kind 不符 / 旧格式即拒绝（守卫哲学）。

        ADR-0008 之前的池化口径（``final_heldout_auc`` 单标量）在
        ``extra=forbid`` 下本会以裸 ValidationError 拒绝——检出该字段
        名后改抛指向格式变更的可读报错（schema 断代显式可见，不静默
        误装也不留难懂的原始报错）。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"预训练报告缺失（缺报告 = 未预训练，拒绝装载）: {path}"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        try:
            report = cls.model_validate(raw)
        except ValidationError as exc:
            if isinstance(raw, dict) and "final_heldout_auc" in raw:
                raise ValueError(
                    "预训练报告为 ADR-0008 之前的池化口径格式"
                    "（final_heldout_auc 单标量）：per-condition 报告契约"
                    "（condition_auc dict + gate_whitelist 白名单）自 "
                    "ADR-0008 起生效，旧报告显式拒绝装载（BraTS 线旧报告"
                    "同此路径）——请以当前版本重新预训练产出"
                ) from exc
            raise
        report._path = path
        return report

    def assert_data_provenance(self, config: CynosureConfig) -> None:
        """当前 config 的数据口径与报告对照：不匹配即拒绝。

        组别对照先行（纯内存比较）：group 是 fake 分布的归因轴，
        per-condition AUC 与条件白名单都在预训练组别自己的 fake 分布上
        测量——跨组消费是口径错位而非可配置语义，显式拒绝、无逃生门
        （#113；组3 序贯 stage-2 的合法消费路径由 stage-2 报告路径绑定
        交付，#116）。随后 latent 形状对照（纯内存比较）：口径指纹与
        判别器形态指纹都不覆盖分辨率——全卷积 scorer 可用旧 shape 的
        real 评新 shape 的 fake 静默通过 gate 并把错位数据带进在线更新。
        real pool / held-out manifest / channel stats 任一文件内容与预训
        练时的指纹不符（manifest 重建、统计量换源）都让上岗判别力与预
        训练报告脱钩——warm-start 装载前显式拒绝，不给静默错位留缝
        （判别器形态指纹的对照在 ``load_discriminator``）。
        """
        if config.experiment.group != self.group:
            raise ValueError(
                f"预训练报告组别不符：报告 {self.group}，当前 config "
                f"{config.experiment.group}（预训练与上岗须同组别口径——"
                "per-condition AUC 与条件白名单是在预训练组别自己的 fake "
                "分布上测量的，跨组消费是显式拒绝的错误，无配置开关可"
                "绕过）"
            )
        if tuple(config.latent_shape) != self.latent_shape:
            raise ValueError(
                f"latent 形状不符：报告 {list(self.latent_shape)}，"
                f"当前 config {list(config.latent_shape)}（分辨率不在口径"
                "指纹与网络配置指纹的覆盖面内——预训练与上岗须同一 "
                "latent 口径）"
            )
        checks = (
            ("real_pool_manifest", config.reward.real_pool_manifest,
             self.provenance.real_pool_manifest_sha256),
            ("heldout_real_manifest", config.reward.heldout_real_manifest,
             self.provenance.heldout_manifest_sha256),
            ("channel_stats_json", config.reward.channel_stats_json,
             self.provenance.channel_stats_sha256),
        )
        mismatched = []
        for name, path, recorded in checks:
            current = PretrainProvenance.digest(path)  # 每工件读盘+哈希一次
            if current != recorded:
                mismatched.append(
                    f"{name}（报告 {recorded[:12]}… ≠ 当前 {current[:12]}…）",
                )
        if mismatched:
            raise ValueError(
                "预训练数据口径指纹不符（预训练与上岗须同一数据口径）: "
                + "; ".join(mismatched)
            )

    def load_discriminator(
        self, config: CynosureConfig, device: torch.device | None = None,
    ) -> RewardScorer:
        """按报告守卫重载判别器：形态指纹对照 → checkpoint 指纹对照 →
        checkpoint 严格装载。

        形态指纹不符（预训练与当前 config 的判别器网络配置不同）显式
        拒绝——strict 装载对同 shape 异配置会静默通过，指纹是那层守卫；
        checkpoint 指纹不符（盘上权重 ≠ 报告实测的那份）同此——白名单
        与实测值的绑定对象是预训练落盘的 checkpoint，不是「该路径下
        此刻的任何权重」。
        """
        if self._path is None:
            raise ValueError(
                "本报告非经 load() 装载，无 checkpoint 路径解析基准"
                "（落盘侧直写，装载侧一律走 load()）"
            )
        if config.artifacts.discriminator_config_json is None:
            raise ValueError(
                "判别器网络配置缺失（artifacts.discriminator_config_json）"
            )
        current = PretrainProvenance.digest(
            config.artifacts.discriminator_config_json,
        )
        if current != self.provenance.discriminator_config_sha256:
            raise ValueError(
                "判别器形态指纹不符：报告 "
                f"{self.provenance.discriminator_config_sha256[:12]}…，当前 "
                f"config 网络配置 {current[:12]}…（预训练与装载的网络形态"
                "须一致）"
            )
        checkpoint = self._path.parent / self.discriminator_ckpt
        observed = PretrainProvenance.digest(checkpoint)
        if observed != self.provenance.discriminator_ckpt_sha256:
            raise ValueError(
                "判别器 checkpoint 指纹不符：报告 "
                f"{self.provenance.discriminator_ckpt_sha256[:12]}…，盘上 "
                f"{observed[:12]}…（报告的白名单与 per-condition 实测值只"
                "对预训练落盘的这份权重负责——启动期重算废止后，同形态换"
                "权重无其他检查可拦，装载期显式拒绝）"
            )
        scorer = RewardScorer(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.discriminator_config_json,
                ),
                checkpoint=checkpoint,
            ),
            config.reward,
            ChannelStats.load(config.reward.channel_stats_json),
        )
        return scorer.to(device if device is not None else torch.device("cpu"))


@dataclass
class PretrainPaths:
    """预训练 run 目录内各工件文件的固定路径（契约布局）。"""

    root: Path
    config_snapshot: Path
    metrics: Path
    checkpoints: Path
    discriminator_ckpt: Path
    report: Path


class PretrainRun:
    """预训练 run 目录与产物工件契约（单进程执行 = 唯一写者）。"""

    def __init__(self, paths: PretrainPaths) -> None:
        self.paths = paths

    @classmethod
    def init(cls, config: CynosureConfig, root: Path) -> "PretrainRun":
        """创建预训练 run 目录并落盘契约最小集（config 快照 + 空指标流）；
        目录已存在则拒绝（不静默覆盖，与 train run 目录同惯例）。"""
        paths = cls.layout(root)
        if paths.config_snapshot.exists():
            raise FileExistsError(f"预训练 run 目录已存在（不静默覆盖）: {root}")
        paths.root.mkdir(parents=True)
        paths.checkpoints.mkdir()
        paths.config_snapshot.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        paths.metrics.touch()
        return cls(paths)

    @classmethod
    def layout(cls, root: Path) -> PretrainPaths:
        root = Path(root)
        return PretrainPaths(
            root=root,
            config_snapshot=root / "config.json",
            metrics=root / "metrics.jsonl",
            checkpoints=root / "checkpoints",
            discriminator_ckpt=root / "checkpoints" / "pretrain_discriminator.pt",
            report=root / "pretrain_report.json",
        )

    def append_event(self, event: "PretrainEvent") -> None:
        """向预训练指标流追加一行 JSON 事件（事件类型与 iter/milestone
        混存同一 metrics.jsonl，event 判别字段区分）。"""
        with open(self.paths.metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                event.model_dump(), ensure_ascii=False, allow_nan=False,
            ) + "\n")

    def read_events(self) -> list[dict]:
        """读回指标流全部事件（预训练曲线与阈值校准的消费面）。"""
        lines = self.paths.metrics.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
