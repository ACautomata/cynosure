"""全量 config schema（spec #15「配置项清单」的落地面）。

设计约定：

- 每个 config 字段都在 ``json_schema_extra`` 里携带结构化的状态标注
  （``status`` + ``source``），对应 spec 配置项清单的「状态」与「出处」两列；
  ``source`` 一般为单一文档名，决策同时落在 ADR 时用复合串
  （如 ``"orchestration + ADR-0005"``，对应 spec 配置项清单出处列）；
- 「定死」项用 ``Literal`` 单值类型或等值 validator 表达——改动即字段级拒绝；
- ``extra="forbid"``：拼错/多余的字段名直接被拒，配合 CLI 输出字段级错误；
- 数值锚（ADR-0002）：``input_img_size_numel`` 必须等于
  ``prod(latent_shape[1:])``，防 sigma 日程静默错位；
- 本模块只做 schema，不读环境、不查文件存在性（工件存在性校验属后续 ticket）。
"""

import json
from math import prod
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

Modality = Literal["t1n", "t1c", "t2w", "t2f"]
"""脑 MRI 四序列（BraTS2023）；CT↔MR 不在本轮实验矩阵（experiment-design 章）。"""

MODALITIES: tuple[Modality, ...] = ("t1n", "t1c", "t2w", "t2f")
"""组1 模态标签条件与组2 跨模态方向共用的四序列清单（定死，experiment-design）。"""

# 组1 采样场（ADR-0002 定死）：CFG=10 组合场，v_cfg = v_uncond + 10·(v_cond − v_uncond)
CFG_MODAL_LABEL: float = 10.0
# 组2 采样场（ADR-0002 定死）：基座代码强制 CFG=0，裸条件单前向
CFG_CROSS_MODAL: float = 0.0
# 无条件分支（policy-modeling 定死）：全零 label（label 0 实际承担「无条件」语义）
UNCONDITIONAL_LABEL: int = 0

# 组2 跨模态方向：12 个有序 src→tgt 对（每序列作 anchor、其余三序列为目标），定死集合
DEFAULT_CROSS_MODAL_PAIRS: tuple[tuple[str, str], ...] = tuple(
    (s, t) for s in MODALITIES for t in MODALITIES if s != t
)


class SpecField:
    """spec 配置项清单的字段声明：状态（status）+ 出处（source）标注。

    构造即产出对应的 pydantic ``Field``——
    ``SpecField("tunable", "reward-model", "描述", default=4, ge=1)``
    就是一个带标注的字段声明。状态标注是配置项清单落 schema 的
    机器可读形式。
    """

    def __new__(cls, status: str, source: str, description: str, **field_kwargs: Any) -> Any:
        return Field(
            description=description,
            json_schema_extra={"status": status, "source": source},
            **field_kwargs,
        )


class Artifacts(BaseModel):
    """输入工件路径（零依赖原则：唯一接口是 checkpoint 文件 + 网络配置 JSON；
    prepare 子命令另需源影像数据集根目录）。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    unet_ckpt: Path = SpecField(
        "运行时", "experiment-design",
        "base UNet checkpoint（实验基座 diff_unet_3d_rflow-mr-brain_v1.pt）",
    )
    vae_ckpt: Path = SpecField(
        "运行时", "experiment-design",
        "图像 VAE checkpoint（autoencoder_v1.pt，AutoencoderKlMaisi）",
    )
    net_config_json: Path = SpecField(
        "运行时", "policy-modeling",
        "网络配置 JSON（UNet 构建参数与 scheduler 配置）",
    )
    modality_mapping_json: Path = SpecField(
        "运行时", "policy-modeling",
        "modality token 映射（t1n/t1c/t2w/t2f → 29/34/30/31）",
    )
    controlnet_ckpt: Path | None = SpecField(
        "运行时", "experiment-design",
        "组2/组3 必需：fork P3 跨序列 ControlNet checkpoint（本地工件）",
        default=None,
    )
    controlnet_config_json: Path | None = SpecField(
        "运行时", "policy-modeling",
        "组2/组3 必需：ControlNet 网络配置 JSON（键 = MONAI ControlNetMaisi "
        "构造参数名；netbuild 与 UNet 同构的 artifact 构建契约）",
        default=None,
    )
    dataset_root: Path = SpecField(
        "运行时", "experiment-design",
        "源影像数据集根目录（BraTS 原始影像；prepare 预编码的输入，"
        "experiment-design「real 样本库」节）",
    )
    discriminator_config_json: Path | None = SpecField(
        "运行时", "reward-model",
        "判别器网络配置 JSON（键 = MONAI PatchDiscriminator 构造参数名；"
        "reward 打分与在线更新的网络装配源）",
        default=None,
    )
    discriminator_ckpt: Path | None = SpecField(
        "运行时", "reward-model",
        "判别器 checkpoint（reward 网络工件的装配源；None = 随机初始化起步"
        "的在线训练）",
        default=None,
    )


class Experiment(BaseModel):
    """三组实验矩阵（experiment-design 章）：group 三选一 + 各组定死语义。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    group: Literal["modal-label", "cross-modal", "sequential"] = SpecField(
        "定死", "experiment-design",
        "实验组：modal-label（组1 模态标签）/ cross-modal（组2 跨模态）/ sequential（组3 序贯），"
        "每次运行三选一",
    )
    base_model: Literal["rflow-mr-brain_v1"] = SpecField(
        "定死", "experiment-design",
        "实验基座 = rflow-mr-brain_v1 + BraTS2023",
        default="rflow-mr-brain_v1",
    )
    dataset: Literal["BraTS2023"] = SpecField(
        "定死", "experiment-design",
        "数据集 = BraTS2023（下游 nnUNet 仪器与跨模态 ControlNet 同域）",
        default="BraTS2023",
    )
    cross_modal_pairs: list[tuple[Modality, Modality]] = SpecField(
        "定死", "experiment-design",
        "组2 跨模态方向 = 脑 MRI 四序列 12 个有序 src→tgt 对（非 CT↔MR），均匀采样",
        default_factory=lambda: list(DEFAULT_CROSS_MODAL_PAIRS),
    )
    stage1_run_dir: Path | None = SpecField(
        "定死", "本 spec 补钉",
        "组3：既有 stage-1 产物路径（指定则跳过 stage-1 训练；None = 同一次运行内先跑 stage-1）",
        default=None,
    )

    @field_validator("cross_modal_pairs")
    @classmethod
    def _pairs_are_the_ordered_12(
        cls, value: list[tuple[Modality, Modality]],
    ) -> list[tuple[Modality, Modality]]:
        expected = set(DEFAULT_CROSS_MODAL_PAIRS)
        seen = set(value)
        if len(value) != 12 or seen != expected:
            raise ValueError(
                "cross_modal_pairs 定死为四序列 12 个有序 src→tgt 对，"
                f"期望 {sorted(expected)}，得到 {sorted(seen)}"
            )
        return value

    @field_validator("stage1_run_dir")
    @classmethod
    def _stage1_product_only_for_sequential(
        cls, value: Path | None, info: ValidationInfo,
    ) -> Path | None:
        """既有 stage-1 产物路径只对组3 有语义：其他组携带即拒绝
        （拼错组名时静默跳过 stage-1 比显式拒绝危险）。"""
        group = info.data.get("group")
        if value is not None and group != "sequential":
            raise ValueError(
                "stage1_run_dir 仅对组3（sequential）有语义：既有 stage-1 "
                f"产物路径用于跳过 stage-1 训练，得到组 {group}"
            )
        return value


class PolicyConfig(BaseModel):
    """policy 采样场与单步 SDE（policy-modeling 章 + ADR-0002）。

    定死语义（不设字段、由实现固化）：CFG 组1=10 组合场 / 组2=0 单前向；
    无条件分支 = 全零 label 且 batch=1 一次评估全组复用；sigma 日程直接取自
    MONAI ``set_timesteps`` 实际输出（config 字面 scale=1.4 是死参数，禁止照抄）。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    num_inference_steps: int = SpecField(
        "定死", "policy-modeling",
        "ODE 步数定死 30（基座行为、timestep transform、实际 scale=1.0）；"
        "缩小日程属 fixture，须经顶层 fixture_mode=true 显式声明",
        default=30, gt=1,
    )
    input_img_size_numel: int = SpecField(
        "定死", "policy-modeling",
        "数值锚 = prod(latent_shape[1:])（30 步 = 131072；fixture [4,16,16,8] = 2048）",
        default=131072, gt=0,
    )
    group_size_g: int = SpecField(
        "tunable", "policy-modeling",
        "G（Group 大小）：组内共享初始噪声的方向数"
        "（显存不够降 6–8 或逐 k 释放；fixture 亦保持 12）",
        default=12, ge=2,
    )
    sde_eta: float = SpecField(
        "扫描接口", "policy-modeling",
        "η（SDE 噪声强度）：被优化训练步的高斯核强度，0 = 精确退化为 MONAI step()",
        default=0.7, ge=0.0,
    )
    sde_s_max: float = SpecField(
        "tunable", "policy-modeling",
        "SDE 奇异点钳制 s_max：s_k > s_max 时钳到 s_max（对齐参考实现同语义，近 1）",
        default=0.999, gt=0.0,
    )
    train_step_indices_m: set[int] = SpecField(
        "配置化 + 扫描", "policy-modeling",
        "M（被优化训练步集合）：沿 timesteps 数组下标、0=最噪端；"
        "排除 0 与末下标（避 s≈1 奇异端 / 保证扰动后有 ODE 续跑空间）",
        default_factory=lambda: set(range(2, 16)),
    )
    granularity_intervals_lambda: set[int] = SpecField(
        "消融", "policy-modeling",
        "Λ（Granularity 间隔集合）：{1,2} 或 {1,2,3}，MGAI 各 λ 的 advantage 组内标准化后求和",
        default_factory=lambda: {1, 2},
    )
    ratio_clip: float = SpecField(
        "定死", "policy-modeling",
        "ratio clip = 1e-4 极窄 trust region（无 KL 时的主要稳定器）",
        default=1e-4,
    )
    optimizer: Literal["AdamW"] = SpecField(
        "定死", "policy-modeling",
        "policy 优化器类型 = AdamW",
        default="AdamW",
    )
    policy_lr: float = SpecField(
        "起步值", "policy-modeling",
        "policy 学习率（AdamW），bf16 autocast + fp32 master weights 配套",
        default=2e-6, gt=0.0,
    )
    policy_weight_decay: float = SpecField(
        "起步值", "policy-modeling",
        "policy AdamW 的 weight decay（参考实现超参总表 1e-4；PyTorch 默认"
        " 1e-2 是 100× 过正则，会淹没 2e-6 的 policy 学习步，故显式落位）",
        default=1e-4, ge=0.0,
    )
    amp_dtype: Literal["bf16"] = SpecField(
        "定死", "policy-modeling",
        "autocast dtype = bf16（gfx936 上需 profile 验证生效，见 M0 门槛）",
        default="bf16",
    )
    master_weights: Literal["fp32"] = SpecField(
        "定死", "policy-modeling",
        "fp32 master weights（bf16 autocast 配套）",
        default="fp32",
    )
    source_latent_scale_factor: float = SpecField(
        "运行时", "policy-modeling",
        "组2 双条件之一：ControlNet 条件 = 源影像 latent × scale_factor"
        "（policy-modeling 章 MDP 条件 c；生产值随基座 ControlNet 推理 "
        "config 核对，fixture 取中性 1.0）",
        default=1.0, gt=0.0,
    )

    @field_validator("ratio_clip")
    @classmethod
    def _ratio_clip_is_fixed(cls, value: float) -> float:
        if value != 1e-4:
            raise ValueError("ratio clip 定死为 1e-4（spec #15 / policy-modeling 章）")
        return value

    @field_validator("sde_s_max")
    @classmethod
    def _s_max_below_singular_point(cls, value: float) -> float:
        if value >= 1.0:
            raise ValueError("s_max 是 σ→1 奇异点钳制，必须严格小于 1")
        return value

    @field_validator("train_step_indices_m")
    @classmethod
    def _m_within_schedule(cls, value: set[int], info: ValidationInfo) -> set[int]:
        if not value:
            raise ValueError("M（被优化训练步集合）不得为空")
        if 0 in value:
            raise ValueError("M 的下标 0 是 s≈1 奇异端（最噪端），必须排除")
        if min(value) < 1:
            raise ValueError(
                f"M 的下标沿 timesteps 数组取、0=最噪端，负下标（{min(value)}）无意义"
            )
        num_steps = info.data.get("num_inference_steps")
        if num_steps is not None and max(value) > num_steps - 2:
            raise ValueError(
                f"M 的最大下标 {max(value)} 超出日程（num_inference_steps={num_steps}）："
                "扰动步之后须至少保留一步 ODE 续跑"
            )
        return value

    @field_validator("granularity_intervals_lambda")
    @classmethod
    def _lambda_within_schedule(cls, value: set[int], info: ValidationInfo) -> set[int]:
        if frozenset(value) not in ({1, 2}, {1, 2, 3}):
            raise ValueError(
                "Λ 消融取值须为完整集合 {1,2} 或 {1,2,3}"
                "（policy-modeling 章，MGAI 可比性），得到"
                f" {sorted(value)}"
            )
        num_steps = info.data.get("num_inference_steps")
        if num_steps is not None and max(value) > num_steps - 1:
            raise ValueError(
                f"Λ 的最大间隔 {max(value)} 超出日程（num_inference_steps={num_steps}）"
            )
        return value


class GrpoConfig(BaseModel):
    """GRPO 核心（policy-modeling 章 + ADR-0001）：组内标准化、MGAI、窄 clip、无 KL。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    advantage_clamp: float = SpecField(
        "定死", "policy-modeling",
        "advantage clamp = ±5（跨 λ 求和后统一截断，参考实现顺序）",
        default=5.0,
    )
    kl_beta: float = SpecField(
        "定死", "ADR-0001",
        "KL 系数 = 0：无 KL、无参考模型（省一整份 UNet 显存）",
        default=0.0,
    )
    ema_anchor_enabled: bool = SpecField(
        "升级项", "ADR-0001",
        "参数 EMA 锚（hacking 签名出现时启用的软约束，不常驻参考模型）",
        default=False,
    )

    @field_validator("advantage_clamp")
    @classmethod
    def _clamp_is_fixed(cls, value: float) -> float:
        if value != 5.0:
            raise ValueError("advantage clamp 定死为 ±5（spec #15）")
        return value

    @field_validator("kl_beta")
    @classmethod
    def _kl_is_fixed_to_zero(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("KL 定死为 0（无 KL、无参考模型，EMA 锚为升级项）")
        return value


class RewardConfig(BaseModel):
    """Reward model（reward-model 章 + ADR-0001）：latent 域在线 PatchDiscriminator。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    disc_num_layers_d: int = SpecField(
        "消融", "reward-model",
        "判别器深度 num_layers_d：2 起步（感受野 ~34³、输出 16×16×8 patch），{1,2} 消融",
        default=2,
    )
    disc_num_scales: int = SpecField(
        "消融", "reward-model",
        "判别器尺度 num_d：单尺度 1 起步，{1,2,3} 消融（多尺度臂各尺度 mean 后相加）",
        default=1,
    )
    patch_aggregation: Literal["mean", "min"] = SpecField(
        "消融", "reward-model",
        "patch logit 图聚合：mean 为主，min 作正交消融（对局部伪影更敏感）",
        default="mean",
    )
    disc_norm: Literal["group"] = SpecField(
        "定死", "ADR-0001",
        "判别器归一化 = GroupNorm（弃默认 BatchNorm：在线小 batch 不泄漏 batch 统计）",
        default="group",
    )
    spectral_norm_enabled: bool = SpecField(
        "触发式", "ADR-0001",
        "SpectralNorm 叠加：默认关闭，判别器过锐/不稳时触发式启用（消融轴）",
        default=False,
    )
    loss_type: Literal["lsgan"] = SpecField(
        "定死", "ADR-0001",
        "判别器损失 = LSGAN（least squares，非饱和梯度）",
        default="lsgan",
    )
    reward_mode: Literal["raw_real_logit"] = SpecField(
        "定死", "ADR-0001",
        "reward = raw real-logit（不过 sigmoid，保留组内分辨率）",
        default="raw_real_logit",
    )
    reward_tanh_bounding: bool = SpecField(
        "触发式", "reward-model",
        "reward 有界化：默认关闭；logit 幅度持续膨胀时 tanh 压 (-1,1) 一行保险",
        default=False,
    )
    disc_update_interval_n_d: int = SpecField(
        "tunable", "reward-model",
        "判别器更新节奏 N_d：每个 RL iteration 都更新（D:G 更新比 ≈ 1:1）",
        default=1, ge=1,
    )
    disc_batch_size_k: int = SpecField(
        "tunable", "reward-model",
        "判别器每批样本量 K（章节未定值，待 rollout 吞吐 profile 后定，执行期）",
    )
    disc_lr: float = SpecField(
        "tunable", "reward-model",
        "判别器学习率（AdamW），5e-5 = 区间 1e-5~1e-4 中点（profile 后定）",
        default=5e-5, gt=0.0,
    )
    replay_buffer_capacity: int = SpecField(
        "tunable", "reward-model",
        "Replay buffer 容量（固定 base 分区 + FIFO 近期分区；base 分区由初始 policy "
        "rollout 填满；章节未定值，待 rollout 吞吐 profile 后定，执行期）",
    )
    replay_current_fraction: float = SpecField(
        "定死", "reward-model",
        "更新判别器时当前 fake 占比 = 0.5（50% 当前 / 50% 回放，防灾难性遗忘）",
        default=0.5,
    )
    real_pool_manifest: Path = SpecField(
        "运行时", "reward-model",
        "Real sample pool manifest（train split 全量 VAE 预编码 latent，按序列分层；prepare 产出）",
    )
    heldout_real_manifest: Path = SpecField(
        "运行时", "本 spec 补钉",
        "Held-out real manifest（val split 预编码 latent；与 D 训练 real 不相交、"
        "永不参与判别器更新）",
    )
    channel_stats_json: Path = SpecField(
        "运行时", "reward-model",
        "判别器输入 per-channel 标准化统计量（来自 Real sample pool 所用训练集；prepare 产出）",
    )

    @field_validator("replay_current_fraction")
    @classmethod
    def _replay_mix_is_fixed(cls, value: float) -> float:
        if value != 0.5:
            raise ValueError("Replay buffer 混合比定死为 50% 当前 / 50% 回放（spec #15）")
        return value

    @field_validator("disc_num_layers_d")
    @classmethod
    def _depth_ablation_axis(cls, value: int) -> int:
        if value not in (1, 2):
            raise ValueError("num_layers_d 消融轴为 {1,2}（2 起步）")
        return value

    @field_validator("disc_num_scales")
    @classmethod
    def _scale_ablation_axis(cls, value: int) -> int:
        if value not in (1, 2, 3):
            raise ValueError("num_d 消融轴为 {1,2,3}（单尺度 1 起步）")
        return value

    @field_validator("disc_batch_size_k", "replay_buffer_capacity")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("必须为正整数")
        return value


class ScheduleConfig(BaseModel):
    """运行时 knobs：规模、里程碑、早停、续训周期与随机性控制。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    max_iterations: int = SpecField(
        "运行时", "experiment-design",
        "每组 RL iteration 数（目标 200–500；先 50 sanity 再扩，故下界不设限）",
        default=200, ge=1, le=500,
    )
    baseline_samples: int = SpecField(
        "运行时", "experiment-design",
        "N_baseline：Baseline 样本量（与评估集同规模、同 seed 同条件、冻结只采一次）",
        default=200, ge=200, le=500,
    )
    n_plateau: int = SpecField(
        "运行时", "experiment-design",
        "N_plateau：早停 plateau 里程碑数（默认 3）",
        default=3, ge=1,
    )
    milestone_interval: int = SpecField(
        "tunable", "本 spec 补钉",
        "里程碑评测间隔（iteration；默认每 50，解码评测只发生在里程碑）",
        default=50, ge=1,
    )
    checkpoint_interval: int = SpecField(
        "tunable", "本 spec 补钉",
        "续训 checkpoint 周期（默认每 10 iteration；每里程碑强制落盘）",
        default=10, ge=1,
    )
    seed: int = SpecField(
        "运行时", "experiment-design",
        "随机种子（随机性控制：Baseline 与 RL 后同 seed 同条件，差异唯一归因于 RL）",
    )


class ShardingConfig(BaseModel):
    """分布式分片（orchestration 章 + ADR-0003）：torchrun + FSDP 同卡交替。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    strategy: Literal["fsdp", "ddp", "zero3"] = SpecField(
        "定死 + fallback", "orchestration",
        "分片策略：FSDP full-shard 起步（降级链 DDP → ZeRO-3，fallback 非默认）",
        default="fsdp",
    )
    gradient_checkpointing: Literal[True] = SpecField(
        "定死", "orchestration",
        "FSDP 配套梯度检查点",
        default=True,
    )


class DeploymentConfig(BaseModel):
    """部署默认（orchestration 章 + ADR-0005）：SothisAI 单实例 4 卡 torchrun、产物落持久分区。

    原为 zzeshell SLURM 的 ``SlurmConfig``（partition/gres/walltime），集群访问权
    永久失去后随平台迁移整体替换；无作业调度器，sbatch 专属字段不再存在。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    nproc_per_node: int = SpecField(
        "部署默认", "orchestration + ADR-0005",
        "每实例进程数 = DCU 卡数（torchrun --nproc_per_node=4；BW/gfx936 4×64GiB/实例）",
        default=4, ge=1,
    )
    output_root: Path = SpecField(
        "部署默认", "orchestration + ADR-0005",
        "产物根目录（run 目录/checkpoint 落盘根；持久分区 /root/private_data 下，"
        "绝不落易失系统盘 /）",
        default=Path("/root/private_data/cynosure"),
    )


class CynosureConfig(BaseModel):
    """cynosure 全量运行配置：train / eval / prepare 三子命令共享同一 schema。"""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    experiment: Experiment = SpecField("定死", "experiment-design", "实验组矩阵（三选一）")
    artifacts: Artifacts = SpecField("运行时", "experiment-design", "输入工件路径")
    latent_shape: tuple[int, int, int, int] = SpecField(
        "定死（fixture 可缩小）", "reward-model",
        "latent 形状 [4,64,64,32]（256×256×128 影像体 ÷4 空间压缩；fixture 缩小为 [4,16,16,8]）",
        default=(4, 64, 64, 32),
    )
    fixture_mode: bool = SpecField(
        "运行时", "Fixture 策略",
        "fixture 诊断模式显式声明：true 才允许缩小采样日程（如 3 步 ODE）；"
        "缺省 false 时生产 config 钉 30 步",
        default=False,
    )
    policy: PolicyConfig = SpecField(
        "定死", "policy-modeling", "policy 采样场与单步 SDE", default_factory=PolicyConfig,
    )
    grpo: GrpoConfig = SpecField(
        "定死", "policy-modeling", "GRPO 核心", default_factory=GrpoConfig,
    )
    reward: RewardConfig = SpecField("定死", "reward-model", "Reward model（在线 PatchDiscriminator）")
    schedule: ScheduleConfig = SpecField("运行时", "experiment-design", "运行时 knobs（规模/里程碑/早停/续训）")
    sharding: ShardingConfig = SpecField(
        "定死", "orchestration", "分布式分片", default_factory=ShardingConfig,
    )
    deployment: DeploymentConfig = SpecField(
        "部署默认", "orchestration + ADR-0005",
        "部署默认（SothisAI 单实例 4 卡、产物落持久分区）",
        default_factory=DeploymentConfig,
    )

    @field_validator("latent_shape")
    @classmethod
    def _latent_shape_is_valid(
        cls, value: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        if len(value) != 4 or any(d < 1 for d in value):
            raise ValueError("latent_shape 必须是 4 个正整数（C, D, H, W）")
        if value[0] != 4:
            raise ValueError(
                f"latent 通道数定死为 4（VAE latent_channels=4），得到 {value[0]}"
            )
        return value

    @field_validator("policy")
    @classmethod
    def _numel_anchor_matches_latent(
        cls, policy: PolicyConfig, info: ValidationInfo,
    ) -> PolicyConfig:
        """数值锚（ADR-0002）：input_img_size_numel 与 latent 空间 numel 同语义一致。"""
        latent_shape = info.data.get("latent_shape")
        if latent_shape is None:
            return policy
        expected_numel = prod(latent_shape[1:])
        if policy.input_img_size_numel != expected_numel:
            raise ValueError(
                f"input_img_size_numel={policy.input_img_size_numel} 与 latent 空间 "
                f"numel {expected_numel}（prod{latent_shape[1:]}）不一致——"
                "sigma 日程会静默错位"
            )
        return policy

    @field_validator("artifacts")
    @classmethod
    def _controlnet_required_for_stage2_groups(
        cls, artifacts: Artifacts, info: ValidationInfo,
    ) -> Artifacts:
        """组2/组3 的训练对象含 ControlNet：checkpoint 与网络配置 JSON
        构成的 artifact 对都必需（netbuild 按 artifact 构建的同一契约）。"""
        experiment = info.data.get("experiment")
        if experiment is None:
            return artifacts
        if experiment.group in ("cross-modal", "sequential"):
            missing = [
                name for name in ("controlnet_ckpt", "controlnet_config_json")
                if getattr(artifacts, name) is None
            ]
            if missing:
                raise ValueError(
                    f"组 {experiment.group} 需要 ControlNet 工件（artifacts."
                    f"{' / artifacts.'.join(missing)}）"
                )
        return artifacts

    @model_validator(mode="after")
    def _inference_steps_match_mode(self) -> "CynosureConfig":
        """缩小采样日程的通道显式化：fixture_mode=false 时 num_inference_steps 钉 30。"""
        if not self.fixture_mode and self.policy.num_inference_steps != 30:
            raise ValueError(
                "生产 config（fixture_mode=false）下 num_inference_steps 定死 30"
                f"（policy-modeling 章），得到 {self.policy.num_inference_steps}；"
                "缩小日程属 fixture，须经顶层 fixture_mode=true 显式声明"
            )
        return self


class ConfigLoader:
    """config 文件装载：JSON 反序列化 + schema 校验（三子命令共用）。"""

    @classmethod
    def load(cls, path: str | Path) -> CynosureConfig:
        """加载并校验 config；校验失败抛 pydantic ValidationError
        （CLI 层负责把 errors() 渲染成字段级错误输出）。"""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return CynosureConfig.model_validate(data)


__all__ = [
    "Artifacts",
    "CFG_CROSS_MODAL",
    "CFG_MODAL_LABEL",
    "ConfigLoader",
    "CynosureConfig",
    "DEFAULT_CROSS_MODAL_PAIRS",
    "DeploymentConfig",
    "Experiment",
    "GrpoConfig",
    "MODALITIES",
    "Modality",
    "PolicyConfig",
    "RewardConfig",
    "ScheduleConfig",
    "ShardingConfig",
    "SpecField",
    "UNCONDITIONAL_LABEL",
]
