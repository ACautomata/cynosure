"""run 目录与产物工件契约（spec「产物工件契约」）：run 目录布局、
config 快照、metrics.jsonl 指标流（事件 schema）、Baseline manifest、
checkpoint 目录。

本模块由 ticket #16（prefactor）交付、#21 起承载训练循环的消费面；
从 ``train`` 包体拆出（循环模块 artifacts/trainer/rollout 的共同依赖，
防 trainer → 包体的循环 import）。
"""

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cynosure.conditions import ConditionVocabulary
from cynosure.config import CynosureConfig

_SEQUENTIAL_STAGES: list[str] = ["modal-label", "cross-modal"]
"""组3 序贯 = 先组1 后组2（experiment-design 章），manifest conditions 按两阶段名记录。"""

POLICY_CHECKPOINT_TEMPLATE = "policy_iter{iteration}.pt"
"""policy checkpoint 文件名模板（契约布局的一部分：组3 stage-1 的复用
查找 ``SequentialTrainer._locate_stage1_product`` 按同一形态解析）。"""


class IterEvent(BaseModel):
    """训练指标流的 per-iteration 事件（契约最小集：施工可扩不可改名）。

    指标 JSONL 是契约工件：NaN/Inf 经默认 json.dumps 会写成非标准 token，
    严格消费方拒读——构造期即拒绝非有限浮点。
    """

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["iter"] = "iter"
    iteration: int
    stage: int = 1
    """组内阶段号（序贯两阶段的归因轴）：单阶段组（组1/组2）恒 1，
    组3 stage-2 事件 = 2——「每组判别器与 buffer 独立」在指标流上的
    观测面（各阶段事件互不混淆）。"""
    rank: int = 0
    """产出本事件的 rank（分布式归并的排序轴：同一 iteration 的 N 条
    事件按 rank 升序连续排列——无重复/无丢失的观测面；单进程恒 0，
    与历史布局逐字一致）。"""
    modality: str
    """本 iteration 采样的目标序列（条件分布四序列均匀采样）——reward/
    loss/AUC 按目标序列归因的轴（per-sequence 健康监控，experiment-design）。"""
    anchor_eval_reward: float
    intra_group_reward_std: float
    heldout_auc: float
    loss: dict[str, float]
    buffer_current_fraction: float
    """更新批的当前 fake 混合占比（N_d 跳过的 iteration 为 0）。"""
    buffer_replay_fraction: float
    """更新批的回放混合占比（N_d 跳过的 iteration 为 0）。"""
    buffer_replay_degraded: bool = False
    """更新步回放退化标记（ADR-0008-03）：本步目标模态的回放候选不足
    半区需求，该步退化纯 current 半区（回放 0 条、real 侧与退化后批
    同量匹配，批减半而两个占比字段仍按 K 分母记账）——正常混采步与
    N_d 跳过的 iteration 均为 False，占比 0 的两种成因靠本标记区分
    （观测面扩展：事件契约可扩不可改名）。"""
    policy_gated: bool = False
    """policy 更新门控标记（ADR-0008 决策 7）：全 rank 集体门控决定——
    任一 rank 的目标条件不在（动态）白名单，本 iteration 全体跳过
    policy 更新（loss 无 policy_step_* 项；rollout / fake 入 buffer /
    判别器更新照常）。单 rank 下退化为「本 iteration 的目标条件不在
    白名单」。语义 = 拒绝在 RM 无分辨率的样本上做策略梯度。
    False = 正常更新步。观测面扩展：事件契约可扩不可改名。"""
    train_pairwise_acc: float | None = None
    """train 侧干净域 pairwise 准确率（ADR-0009-β）：本 iteration 判别器
    更新批上的干净域 no_grad 复算（更新前快照，随单步更新报告上行）。
    N_d 跳过的 iteration（无判别器步）为 None。观测面扩展：事件契约
    可扩不可改名。"""
    overfit_divergence_ema: float | None = None
    """per-condition 分叉 EMA（ADR-0009-β）：EMA(train pairwise acc −
    held-out AUC) 的本 rank 读数——按 rank 独立计算落盘（rank 间离散
    是数据切片异质性的诊断信号，不跨 rank 平均）。N_d 跳过的 iteration
    为 None。观测面扩展：事件契约可扩不可改名。"""
    buffer_base_occupied: int
    """Replay buffer base 分区当前占用（固定分区的状态观测面）。"""
    buffer_recent_occupied: int
    """Replay buffer 近期分区当前占用（FIFO 滚动观测面）。"""
    lr: float
    elapsed_s: float
    phase_seconds: dict[str, float] = Field(default_factory=dict)
    """逐 iter 卡时分解（#123 tracer 首跑的成本读数面）：本 iteration 各
    相位的墙钟秒数，键 = 相位名（``rollout`` / ``heldout_auc`` /
    ``gating`` / ``policy_update`` / ``discriminator``，可扩）。
    覆盖区间与 ``elapsed_s`` 同界——[iteration 起点, 本事件构造]：周期
    checkpoint、里程碑解码评测与迭代末 barrier 在该区间之外（随既有
    ``elapsed_s`` 口径，不单列）。各相位之和 ≤ ``elapsed_s``（差额 =
    相位边界上的簿记开销，不追求逐字节闭合）。空 dict = 调用方未做
    相位计量（测试/替身直调场景）。观测面扩展：事件契约可扩不可改名。"""


class MilestoneEvent(BaseModel):
    """训练指标流的里程碑评测事件（解码评测只发生在里程碑）。"""

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["milestone"] = "milestone"
    iteration: int
    stage: int = 1
    """里程碑归属阶段号（组3 两阶段事件互不混淆，与 IterEvent 同轴）。"""
    fid: float
    kid: float | None = None
    ssim: float | None = None
    mae: float | None = None
    psnr: float | None = None
    criteria_summary: dict[str, float] = Field(default_factory=dict)
    early_stop: bool = False
    early_stop_reason: str | None = None
    """触发早停的签名（"plateau" / "reward_hacking"）；未停为 None。"""
    elapsed_s: float | None = None
    """里程碑评测总卡时（#124 监控成本读数：采样 + 解码 + FID 的全
    区间，监控相在 RL 主循环外的独立成本行）。None = 调用方未计量
    （替身直调场景）。"""
    phase_seconds: dict[str, float] = Field(default_factory=dict)
    """里程碑评测相位卡时分解（#124，#111 监控账的成本行取数面）：
    ``decode`` = 合成侧 VAE 分块解码；``fid`` = 参照装载 + 特征提取 +
    距离核。空 dict = 调用方未做相位计量（替身直调场景）。"""


class OverfitAlertEvent(BaseModel):
    """训练指标流的过拟合分叉报警事件（ADR-0009 决策 4/5，issue #105；
    γ 按相分轨，#106）。

    per-condition 分叉（EMA(train 干净域 pairwise acc − held-out AUC)）
    自下而上越线时由编排方产出——RL 相由 train 循环产出（随 iter 事件
    同归并序写出），预训练相由预训练 driver 产出（随 pretrain 事件之后
    直写）——按 rank 独立计算（rank 间离散是数据切片异质性的诊断信号，
    不跨 rank 平均）。**报警不动作**：事件只承载读数，白名单与噪声 σ 不
    被联动（人工裁决，ADR-0009 决策 5）。回退记账按相分轨（γ）：RL 相
    按 iteration 轴随所属 iteration 删除（``REWIND_ACCOUNTING`` 的
    ``ITERATION`` 口径——恢复点之后的告警由重执行重发）；预训练相
    （``phase="pretrain"``）与预训练事件同口径全量保留——预训练执行史
    没有对应的 checkpoint 可重放，删除即永久丢失。事件契约「可扩不可
    改名」；非有限浮点构造期拒绝（判别器数值发散不产毒事件，指标流的
    全流拒绝口径）。
    """

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["overfit_alert"] = "overfit_alert"
    iteration: int
    """报警所属轴号：RL 相 = iteration 号（回退记账轴）；预训练相 =
    预训练步号（与 pretrain 事件的 ``step`` 同轴）。"""
    stage: int = 1
    """报警归属阶段号（组3 两阶段事件互不混淆，与 IterEvent 同轴）。"""
    rank: int = 0
    """观测到越线的 rank（分叉按 rank 独立计算落盘的归因轴）。"""
    phase: Literal["pretrain", "rl"] = "rl"
    """告警所属相（γ 的回退记账分轨轴）：``"rl"`` = RL 在线循环（默认，
    既有构造点与旧事件零改动兼容）；``"pretrain"`` = warm-start 预训练
    driver（登记 EXEMPT——预训练执行史不参与续训回退重写）。"""
    modality: str
    """越线条件（目标模态，与 iter 事件同归因轴）。"""
    divergence_ema: float
    """分叉值（该条件 EMA(train pairwise acc − held-out AUC) 的越线读数）。"""
    train_pairwise_acc: float
    """本步 train 侧干净域 pairwise 准确率（两侧原始量之一）。"""
    heldout_auc: float
    """本 iteration 该条件 held-out AUC（两侧原始量之二，更新前快照）。"""


class PretrainEvent(BaseModel):
    """判别器预训练指标流的逐步事件（ADR-0007 warm-start）。

    ``event`` 判别字段与 iter/milestone 混存同一 metrics.jsonl（事件契约
    沿「可扩不可改名」口径：新类型加判别值与字段、既有字段名与语义不改）；
    预训练事件在回退记账中登记为 ``EXEMPT``（见 ``REWIND_ACCOUNTING``）
    ——续训回退只重写 RL iteration 的半截执行史，预训练执行史全量保留。
    """

    model_config = ConfigDict(allow_inf_nan=False)

    event: Literal["pretrain"] = "pretrain"
    step: int
    """预训练步号（0 起；每步 = 一批同源重构 fake + 一次判别器单步更新，
    确认步不更新不发事件）。"""
    modality: str
    """本步条件（目标模态，轮转条件集的 ``step % n`` 项）——预训练
    事件按条件归因 recon-AUC（ADR-0008-04：阈值校准数据面的条件轴；
    iter 事件同款归因字段）。"""
    loss_discriminator: float
    heldout_auc: float
    """本步更新前测得的 **recon-AUC**（ADR-0012 决策 5）：该条件 held-out
    全量卷原始 vs 其冻结基座同源重构体的池化点估计（与在线期 iter 事件
    同「更新前快照」测量时点——更新后测同一测量批会把 in-sample 拟合
    计入 AUC）。字段名沿用「可扩不可改名」契约（口径已换域，见
    ``PretrainReport.gate_criterion``）；**与 iter 事件的 rollout-AUC
    不可跨相直接比较**：预训练判据测的是判别器训练任务上的 out-of-sample
    泛化力、在线口径测的是对打分对象（rollout 终点）的分辨力。"""
    buffer_base_occupied: int
    """Replay buffer base 分区当前占用（ADR-0012 后预训练相无种植方 →
    恒 0；组件与字段按契约保留，物理删除归 #173）。"""
    buffer_recent_occupied: int
    """Replay buffer 近期分区当前占用（同上，恒 0）。"""
    reconstruction_forwards: int | None = None
    """本步测量批重构消耗的 policy 前向次数（#171 AC5 的成本口径读数：
    批量量产的 num_steps 步全 ODE 路径已退役，本读数让「每卷重构前向 =
    num_steps − 起始步位、恒严格小于全 ODE 步数」在事件流上可核对——
    与装配原语的 ``measurement_forward_count`` 同源推算，非平行复刻）。
    None = 本步未测量（当前执行路径每步都测，留 None 供旧事件与未来
    分支）。"""
    measurement_volumes: int | None = None
    """本步测量批的卷数（= 该条件 held-out 全量卷数；支撑度判定的卷数
    轴在事件流上的留痕）。None 同 ``reconstruction_forwards``。"""
    lr: float
    elapsed_s: float


class RewindAccounting(Enum):
    """事件类型在续训回退（rewind）中的记账口径（每型事件声明的保留策略）。

    回退 = 删除恢复点之外的**半截**执行史、由恢复后的重执行重写（保住
    「每 iteration 每 stage 一条事件」的流不变量）。保留边界按各型自身的
    记账轴取——同一份流里三种轴并存，故口径随事件类型声明而非全局统一：

    - ``ITERATION``：iter 事件以 0-based iteration 号记账，保留号 <
      恢复点（号 ≥ 恢复点的迭代未进 checkpoint 覆盖面，重执行重写）；
    - ``COMPLETION``：milestone 事件以完成数记账，保留完成数 ≤ 恢复点
      ——里程碑评测与 checkpoint 同批产出，按 iter 边界删会抹掉该里程碑
      的评测历史（FID 序列断点、早停 verdict 消失且不再重放）；
    - ``EXEMPT``：不参与回退记账，全量保留。warm-start 预训练事件属此类：
      它没有对应的 checkpoint 可重放，按任何边界删都是永久丢失（收敛
      曲线断点、RM readiness gate 的阈值校准数据不可复现）；表外事件
      类型（未登记 / 新增未声明）的兜底同为 ``EXEMPT``——同一条「不参与
      记账」语义在已登记与未登记两侧共用。
    """

    ITERATION = "iteration"
    COMPLETION = "completion"
    EXEMPT = "exempt"

    def covers(self, number: int, recovery: int) -> bool:
        """恢复点 ``recovery`` 是否覆盖事件号 ``number``（覆盖 = 保留）。

        ``number`` 按本口径自身的记账轴取：``ITERATION`` 传事件的
        0-based iteration 号，``COMPLETION`` 传完成数（两者都是事件的
        ``iteration`` 字段，语义随口径而异——指标流的既有字段不动）。
        """
        if self is RewindAccounting.ITERATION:
            return number < recovery
        if self is RewindAccounting.COMPLETION:
            return number <= recovery
        if self is RewindAccounting.EXEMPT:
            return True
        raise ValueError(f"未实现的记账口径（新增口径须实现保留边界）: {self}")


REWIND_ACCOUNTING: dict[str, RewindAccounting] = {
    "iter": RewindAccounting.ITERATION,
    "milestone": RewindAccounting.COMPLETION,
    "pretrain": RewindAccounting.EXEMPT,
    "overfit_alert": RewindAccounting.ITERATION,
}
"""事件判别值 → 回退记账口径的登记表（契约「可扩不可改名」的记账面）。

登记表是**删除的准入名单**：``rewind_events`` 只对表内口径为删除的轴做
判定，表外（未登记 / 新增未声明）的事件类型一律保留——宁可留痕不可误删。
新增事件类型 = 新判别值 + 登记口径 + spec 事件类型清单同步（三者同批），
既有类型的判别值与字段名不变。

``overfit_alert`` 的口径按相分轨（ADR-0009-γ）：表内登记的
``ITERATION`` 是 **RL 相**（``phase="rl"``，缺省值）的记账轴；预训练相
（``phase="pretrain"``）与 ``pretrain`` 事件同口径全量保留，分派在
``RunArtifacts._kept_by_rewind`` 的相特判——登记表按判别值索引，相是
事件级字段，一型两轨的判定不进表。
"""


class ManifestEntry(BaseModel):
    """Baseline 采样清单的单条目：一个采样位（阶段 + 序号）的种子、条件
    与样本路径（契约最小集：seed、条件、样本路径——spec「产物工件契约」）。

    条目在 run 目录创建时按 config 确定性生成；``baseline_sample`` 由训练
    启动期的冻结初始 policy 采样落盘填入（冻结只采一次），``resample_sample``
    由 RL 后同 seed 重采填入——两侧复用同一 manifest（同 seed 同条件，
    差异唯一归因于 RL）。``source_case``（组2）由 baseline 采样期记录：
    源病例锁定后重采与里程碑评测读回同一病例。
    """

    model_config = ConfigDict(extra="forbid")

    stage: int = 1
    index: int
    condition: str | list[str]
    """本采样位的条件（组1 = 目标序列名；组2/组3-stage2 = [源序列, 目标序列]）。"""
    source_case: str | None = None
    """组2 锁定的源病例 id（baseline 采样期写入；组1 为 None）。"""
    noise_seed: int
    """本采样位的初始噪声种子（确定性派生，重采与里程碑共用）。"""
    baseline_sample: str | None = None
    """Baseline 像素体文件路径（相对 run 目录）。"""
    resample_sample: str | None = None
    """RL 后重采像素体文件路径（相对 run 目录）。"""


class BaselineManifest(BaseModel):
    """Baseline 采样清单（run 目录 ``manifest.json`` 的契约）：seed、条件
    词汇表与采样条目清单。Baseline 与 RL 后重采、里程碑评测共同消费——
    同一条目 = 同一 seed 同一条件，是「差异唯一归因于 RL」的载体。

    条目生成确定性：同 config 必得同 manifest（噪声种子为
    ``(seed, stage, index)`` 的纯函数）；样本路径随 baseline 采样与重采
    先后写入对应条目（T10 契约）。
    """

    model_config = ConfigDict(extra="forbid")

    seed: int
    group: str
    conditions: list
    """本组条件的词汇表（组1 四序列 / 组2 12 有序对 / 组3 两阶段名）。"""
    entries: list[ManifestEntry] = Field(default_factory=list)

    @staticmethod
    def noise_seed(seed: int, stage: int, index: int) -> int:
        """采样位初始噪声种子的确定性派生（splitmix64 终混）：
        纯函数、与生成顺序无关——baseline / 重采 / 里程碑三侧独立重算同值。"""
        mask = 0xFFFFFFFFFFFFFFFF
        h = (seed + 0x9E3779B97F4A7C15 + (stage << 32) + index) & mask
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & mask
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & mask
        h ^= h >> 31
        return h

    @classmethod
    def build(cls, config: CynosureConfig) -> "BaselineManifest":
        """按 config 确定性生成清单：组1 = 四序列轮转，组2 = 12 有序对
        轮转，组3 = 两阶段各自的采样位（条目数均为 N_baseline）。"""
        entries = [
            ManifestEntry(
                stage=stage,
                index=index,
                condition=conditions[index % len(conditions)],
                noise_seed=cls.noise_seed(config.schedule.seed, stage, index),
            )
            for stage, conditions in cls._stage_conditions(config).items()
            for index in range(config.schedule.baseline_samples)
        ]
        return cls(
            seed=config.schedule.seed,
            group=config.experiment.group,
            conditions=cls._condition_vocabulary(config),
            entries=entries,
        )

    @staticmethod
    def _stage_conditions(config: CynosureConfig) -> dict[int, list]:
        """group → {阶段号: 条件清单} 的执行映射（词汇表单一来源：
        BraTS = ``config.stage_condition_vocabulary()``；MR-RATE =
        条件词汇表装配（``ConditionVocabulary.assemble`` 统一分派）的
        names()（#129——schema 不读文件，装载发生在本 build 消费点，
        与评测/policy 装配同口径）。本方法只叠加执行
        语义。

        组3 指定 ``stage1_run_dir``（复用既有 stage-1 产物）时只建
        stage-2 条目：stage-1 不在本 run 执行，manifest 不留无人填充的
        null 条目（stage-1 样本对住在源 run 自己的 manifest）。"""
        if config.experiment.dataset == "MR-RATE":
            stages = {1: list(ConditionVocabulary.assemble(config).names())}
        else:
            stages = config.stage_condition_vocabulary()
        if (
            config.experiment.group == "sequential"
            and config.experiment.stage1_run_dir is not None
        ):
            return {2: stages[2]}
        return stages

    @classmethod
    def _condition_vocabulary(cls, config: CynosureConfig) -> list:
        stages = cls._stage_conditions(config)
        if len(stages) == 1:
            return next(iter(stages.values()))
        return list(_SEQUENTIAL_STAGES)

    @classmethod
    def load(cls, path: Path) -> "BaselineManifest":
        """装载既有 run 目录的 manifest（重采与里程碑评测的读取入口）。"""
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def write(self, path: Path) -> None:
        """落盘（baseline 采样与重采写入样本路径后回写）。"""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def entries_for_stage(self, stage: int) -> list[ManifestEntry]:
        """本阶段的采样条目（单阶段组恒 stage=1；组3 两阶段各取各的）。"""
        entries = [entry for entry in self.entries if entry.stage == stage]
        if not entries:
            raise ValueError(f"manifest 无 stage={stage} 的采样条目")
        return entries


@dataclass
class RunPaths:
    """run 目录内各工件文件的固定路径（契约布局）。"""

    root: Path
    config_snapshot: Path
    metrics: Path
    manifest: Path
    checkpoints: Path
    samples: Path
    """Baseline 与 RL 后重采的解码像素体目录（评测材料，契约布局成员）。"""
    trajectory_diagnostic: Path
    """轨迹诊断工件（fixture 诊断开关 --dump-trajectory 产出；诊断未跑则无此文件）。"""
    training_diagnostic: Path
    """训练诊断工件（--dump-trajectory 时的训练侧 log-prob 对；未开启则无此文件）。"""


class RunArtifacts:
    """run 目录与产物工件契约：config 快照 + metrics.jsonl + manifest +
    checkpoint 目录，落 ``$HOME``（多 rank 下指标由 rank 0 归并写出）。"""

    def __init__(self, paths: RunPaths) -> None:
        self.paths = paths

    @classmethod
    def init(cls, config: CynosureConfig, root: Path) -> "RunArtifacts":
        """创建 run 目录并落盘契约最小集工件；run 目录已存在则拒绝
        （每次运行一个 run 目录的隔离契约，续训须显式复用并经续训入口）。

        只应由协调方（CLI：进程组 rendezvous 之后的 rank 0）在新 run
        启动时调用一次，成败经广播裁决同步各 rank——文件存在性无法区分
        「rank 0 本轮新建」与「上轮遗留」，历史上「非 0 rank 轮询等待
        config 快照出现」的握手在预存目录下会让非 0 rank 误判 rank 0
        成功、径自进入 rendezvous 挂死。
        """
        paths = cls.layout(root)
        if paths.config_snapshot.exists():
            raise FileExistsError(f"run 目录已存在（不静默覆盖）: {root}")
        cls._create_minimal_set(config, paths)
        return cls(paths)

    @classmethod
    def _create_minimal_set(cls, config: CynosureConfig, paths: RunPaths) -> None:
        paths.root.mkdir(parents=True)
        paths.checkpoints.mkdir()
        paths.config_snapshot.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        paths.metrics.touch()
        cls.manifest(config).write(paths.manifest)

    @classmethod
    def layout(cls, root: Path) -> RunPaths:
        return RunPaths(
            root=root,
            config_snapshot=root / "config.json",
            metrics=root / "metrics.jsonl",
            manifest=root / "manifest.json",
            checkpoints=root / "checkpoints",
            samples=root / "samples",
            trajectory_diagnostic=root / "trajectory.json",
            training_diagnostic=root / "training.json",
        )

    @classmethod
    def default_root(cls, config: CynosureConfig) -> Path:
        """默认 run 目录：``$HOME/.cynosure/runs/<UTC 微秒时间戳>-<group>``。"""
        base = Path.home() / ".cynosure" / "runs"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        return base / f"{timestamp}-{config.experiment.group}"

    @classmethod
    def manifest(cls, config: CynosureConfig) -> BaselineManifest:
        """Baseline 采样清单（契约最小集：seed、条件、样本路径）。

        条目按 config 确定性生成；样本路径由训练启动期的 baseline 采样
        与 RL 后同 seed 重采先后填入——两侧共用同一 manifest。
        """
        return BaselineManifest.build(config)

    def append_event(
        self,
        event: IterEvent | MilestoneEvent | PretrainEvent | OverfitAlertEvent,
    ) -> None:
        """向训练指标流追加一行 JSON 事件（按行追加、rank 0 归并）。"""
        with open(self.paths.metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(
                event.model_dump(), ensure_ascii=False, allow_nan=False,
            ) + "\n")

    def read_events(self) -> list[dict]:
        """读回指标流全部事件（早停判定与评测脚本共同消费）。"""
        lines = self.paths.metrics.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def rewind_events(self, iteration: int, stage: int) -> int:
        """续训回退指标流：删除恢复点之后本 stage 的**半截**事件——
        checkpoint 覆盖面之外的执行史由恢复后的重执行重写，保住「每
        iteration 每 stage 一条事件」的流不变量（重复事件会污染早停判定
        等下游消费者）。

        保留边界按事件类型登记的回退记账口径取（``REWIND_ACCOUNTING``）：
        预训练事件（pretrain）登记为 ``EXEMPT``，不参与回退记账——warm-start
        执行史没有对应的 checkpoint 重放，删除即永久丢失（spec「实现
        警点」）；未登记的事件类型同样不参与删除（宁可留痕不可误删）。
        stage 不匹配的事件（其他阶段的历史）一概不动。返回删除的事件数。"""
        events = self.read_events()
        kept = [
            event for event in events
            if self._kept_by_rewind(event, iteration, stage)
        ]
        removed = len(events) - len(kept)
        if removed:
            tmp = self.paths.metrics.with_name(
                self.paths.metrics.name + ".rewind.tmp",
            )
            with open(tmp, "w", encoding="utf-8") as fh:
                for event in kept:
                    fh.write(json.dumps(
                        event, ensure_ascii=False, allow_nan=False,
                    ) + "\n")
            # 原子替换（与续训状态同一 durability 口径）：中途崩溃不留
            # 半截重写的指标流
            os.replace(tmp, self.paths.metrics)
        return removed

    @staticmethod
    def _kept_by_rewind(event: dict, iteration: int, stage: int) -> bool:
        """单事件在回退后的去留：先按判别值查记账口径，再按该口径的轴
        取保留边界。口径外类型（含未登记类型）不进 stage 过滤——它们
        没有本 stage 的执行史语义，过滤即等于按错误的轴判删。"""
        accounting = REWIND_ACCOUNTING.get(
            event.get("event"), RewindAccounting.EXEMPT,
        )
        if accounting is RewindAccounting.EXEMPT:
            return True
        # overfit_alert 按相分轨（ADR-0009-γ）：预训练相告警与预训练
        # 事件同口径——预训练执行史没有对应的 checkpoint 可重放，按
        # iteration 轴删是永久丢失（旧事件无 phase 字段，缺省 RL 相，
        # 与既有流的删除口径逐字兼容）；stage/iteration 轴只有 RL 相
        # 告警消费
        if (
            event.get("event") == "overfit_alert"
            and event.get("phase", "rl") == "pretrain"
        ):
            return True
        if event.get("stage", 1) != stage:
            return True
        return accounting.covers(event.get("iteration", 0), iteration)
