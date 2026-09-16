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

REGISTERED_DATASETS: tuple[str, ...] = ("BraTS2023", "MR-RATE")
"""已登记数据域（#127 泛化：dataset 从 Literal 定死放宽为 str + 登记
validator——新增数据域扩本登记处并接线其条件词汇来源，不再动类型层；
BraTS2023 的词汇 = 代码内四序列常量语义，MR-RATE 的词汇 =
条件词汇表工件 ``artifacts.condition_vocabulary_json``）。

MR-RATE 的条件词表口径（五模态集 / token 映射 9/10/11/20/16 / 11 生成
条件五元组）已整体迁出本模块：工件
``data/conditions/mrrate_conditions.json`` 是唯一来源，装载面在
``cynosure.conditions.MrConditionVocabulary``（#127；#119 的 config 内嵌
词表 ``MrRateConditioning`` 退役）。"""

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

UPSTREAM_RESIZE_BASE = 128
"""prepare transform 链的上游 resize 基数（fork create_training_data 的
round_number，data-preparation + ADR-0006）：生产定死值、schema 权威单一来源，
fixture 经 fixture_mode=true 显式声明后可注入小基数替代。"""

SPACING_CONDITION_SCALE: float = 1e2
"""header zooms / 等效 spacing（mm）→ spacing 条件张量单位的换算因子
（×1e2；policy-modeling 章「体素间距 ×1e2」的 schema 权威单一来源）。
两臂消费共享：BraTS 臂 ``SpacingSidecar``（per-case raw zooms 侧车，
issue #46）与 MR-RATE 臂 ``MrConditionVocabulary.spacing_condition``
（条件属性解析面，#130，spec #125 决策 6）。"""


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
    vae_config_json: Path | None = SpecField(
        "运行时", "reward-model",
        "VAE 网络配置 JSON（键 = MONAI AutoencoderKlMaisi 构造参数名；"
        "里程碑解码评测与生产预编码的装配源，与判别器工件对同构）",
        default=None,
    )
    radimagenet_weights: Path | None = SpecField(
        "运行时", "experiment-design",
        "RadImageNet-ResNet50 权重文件（像素域 2.5D FID/KID 特征提取器的"
        "生产装载源；公开发布权重的下载属施工，fixture 走 stub 注入）",
        default=None,
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
    condition_vocabulary_json: Path | None = SpecField(
        "运行时", "#125 + #127",
        "MR-RATE 条件词汇表工件（11 生成条件五元组：modality token / "
        "plane / 推荐 FOV / 统一网格 / 等效 spacing；唯一来源，代码内无"
        "词表副本）。仅 dataset=MR-RATE 有语义且必填（schema 守卫）；"
        "BraTS config 携带即拒绝。装载面 = cynosure.conditions."
        "MrConditionVocabulary（普查期望网格对账在装载期）",
        default=None,
    )
    mrrate_metadata_csv: Path | None = SpecField(
        "运行时", "#125 + #131",
        "MR-RATE series 级元数据 CSV（列：study_uid / series_id / "
        "patient_uid / modality / plane；官方 batchXX_metadata.csv 的键列"
        "子集）——prepare 配额抽样的候选域来源。仅 dataset=MR-RATE 有语义"
        "且必填（schema 守卫）；BraTS config 携带即拒绝",
        default=None,
    )
    mrrate_splits_csv: Path | None = SpecField(
        "运行时", "#125 + #131",
        "MR-RATE 官方 patient 级 split CSV（列：patient_uid / split；"
        "官方 splits.csv）——real 数据链只取 train split（病例级，同患者"
        "所有 study 同 split），留出集与 held-out 互斥的官方优先来源。"
        "仅 dataset=MR-RATE 有语义且必填；BraTS config 携带即拒绝",
        default=None,
    )
    eval_manifest_csv: Path | None = SpecField(
        "运行时", "#125 + #131 + #78",
        "MR-RATE 评估集 manifest（#78 工件 eval_manifest.csv）——装配期"
        "评估集互斥硬守卫的键源（study_uid + series_id 键、patient 级"
        "双守卫）。仅 dataset=MR-RATE 有语义且必填；BraTS config 携带即"
        "拒绝",
        default=None,
    )
    mrrate_data_snapshot: str | None = SpecField(
        "运行时", "#125 + #131",
        "MR-RATE 数据 release 快照标识（与评估集 #78 同一冻结快照，如 "
        "HF revision）——随 prepare 工件 provenance 留痕，real 数据链与"
        "评估集分类口径一致的凭据。仅 dataset=MR-RATE 有语义且必填；"
        "BraTS config 携带即拒绝",
        default=None,
    )
    source_commit: str | None = SpecField(
        "运行时", "#121",
        "产出 prepare 工件的代码版本标识（来源 commit）——由运行环境"
        "（实验脚本 / CI）显式填入并随工件 provenance 落档（#121 AC2："
        "provenance 的「来源 commit」承载；集群 rsync 部署无 .git，"
        "不设运行时 git 自读的隐式通道）；缺省 None = 未声明，"
        "provenance 该字段留空",
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
        "实验基座 = rflow-mr-brain_v1（BraTS2023 与 MR-RATE 两域共用冻结基座）",
        default="rflow-mr-brain_v1",
    )
    dataset: str = SpecField(
        "定死", "experiment-design + #67",
        "数据域（登记域 validator 守卫，已登记 "
        + " / ".join(REGISTERED_DATASETS) + "，未登记值字段级拒绝）："
        "BraTS2023（下游 nnUNet 仪器与跨模态 ControlNet 同域，条件语义 = "
        "四序列常量）/ MR-RATE（地图 #67 换域：同基座的组1 模态标签 RL "
        "后训练，条件词汇表经 artifacts.condition_vocabulary_json 工件装载，"
        "#127）——两套口径的互斥选择开关。#127 泛化：从 Literal 定死放宽为 "
        "str + 登记域 validator，新增数据域扩登记处与词汇接线，不动类型层",
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
    stage2_pretrain_report_json: Path | None = SpecField(
        "定死", "本 spec 补钉",
        "组3：stage-2（cross-modal config）上岗消费的 cross-modal 预训练报告路径"
        "（stage 级报告绑定，#116——序贯编排把绑定路径重写进 stage-2 计划 config 的"
        " reward.pretrain_report_json，不静默继承 stage-1 报告；序贯必填，非序贯组"
        "携带即拒绝）",
        default=None,
    )

    @field_validator("dataset")
    @classmethod
    def _dataset_is_registered(cls, value: str) -> str:
        """dataset 登记域守卫（#127 泛化形态）：类型层 Literal 退役后，
        未登记的数据域在装载期字段级拒绝——拼错域名不再静默流入
        分派逻辑。"""
        if value not in REGISTERED_DATASETS:
            raise ValueError(
                f"dataset {value!r} 未登记（已登记域：{REGISTERED_DATASETS}；"
                "新增数据域须扩登记处并接线其条件词汇来源，#127）"
            )
        return value

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

    @model_validator(mode="after")
    def _mrrate_supports_modal_label_only(self) -> "Experiment":
        """MR-RATE 线只定义组1（地图 #67：上游无 MR ControlNet，跨模态/
        序贯是 BraTS 语义）——非组1 即拒绝：组2/组3 的 cross_modal_pairs、
        ControlNet 工件等 BraTS 条件语义在 MR config 里静默错位。"""
        if self.dataset == "MR-RATE" and self.group != "modal-label":
            raise ValueError(
                "MR-RATE 线只定义组1（modal-label）：上游无 MR ControlNet，"
                f"跨模态/序贯是 BraTS 语义（地图 #67），得到组 {self.group}"
            )
        return self

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

    @field_validator("stage2_pretrain_report_json")
    @classmethod
    def _stage2_report_binding_is_sequential_only(
        cls, value: Path | None, info: ValidationInfo,
    ) -> Path | None:
        """stage-2 报告绑定只对组3 有语义，双向显式拒绝（#116）：

        - 非序贯组携带即拒绝（与 ``stage1_run_dir`` 同款——拼错组名时
          静默绑定比显式拒绝危险）；
        - 序贯缺绑定也拒绝——stage-2 若不绑定 cross-modal 报告，就会沿
          stage-1 的 ``reward.pretrain_report_json`` 装载 modal-label
          报告、到 stage-2 装配期才被组别等值守卫（#113）拒绝；装载期
          显式拒绝把指引提前到 config 面，不静默继承。"""
        group = info.data.get("group")
        if group != "sequential" and value is not None:
            raise ValueError(
                "stage2_pretrain_report_json 仅对组3（sequential）有语义："
                "stage-2 消费的 cross-modal 预训练报告路径绑定，得到组 "
                f"{group}"
            )
        if group == "sequential" and value is None:
            raise ValueError(
                "组3（sequential）须配置 stage2_pretrain_report_json"
                "（stage-2 消费的 cross-modal 预训练报告路径，#116 stage 级"
                "报告绑定）：缺省时 stage-2 会沿 stage-1 的 reward."
                "pretrain_report_json 继承 modal-label 报告，被组别等值守卫"
                "在装载期拒绝——序贯不静默继承 stage-1 报告，两份预训练产物"
                "的路径须分别显式声明"
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
    latent_scale_factor: float = SpecField(
        "运行时", "policy-modeling",
        "主流 latent 的 checkpoint 域缩放因子（基座 diffusion UNet checkpoint "
        "的 scale_factor = 1/std(z)）：官方权重在 scaled 域训练/采样，prepared "
        "latents 按存储契约是 encode 原始输出（未乘）——policy 侧装载点乘入、"
        "评测解码前除回（本字段消费点）。生产值随基座 checkpoint 核对，"
        "fixture 取中性 1.0",
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
    disc_weight_decay: float = SpecField(
        "起步值", "ADR-0007",
        "判别器 AdamW 的 weight decay：显式落位并与 policy 侧同值口径"
        "（1e-4，policy_weight_decay 同源）；此前隐式取 PyTorch 默认 0.01，"
        "与 policy 侧的 1e-4 不对称（ADR-0007 卫生项）",
        default=1e-4, ge=0.0,
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
    real_pool_quota: dict[str, int] = SpecField(
        "tunable", "#125 + #131",
        "MR-RATE real pool 逐条件配额上限（键 = 生成条件名，值 = 抽样"
        "卷数上限；词表内未登记的条件 = 全量不设限——头部模态各数千条、"
        "MRA 全量 ≈ 110 的登记形态）。固定 seed、排序后抽样、配额为上限"
        "而非硬指标（候选不足取全量，容量下限由装配期容量守卫把守）。"
        "仅 MR-RATE 线有语义，BraTS 线（train split 全量，无配额语义）"
        "不消费",
        default_factory=dict,
    )
    heldout_fraction: float = SpecField(
        "tunable", "#125 + #131 + #73",
        "MR-RATE held-out real 的 train split 内 patient 级二分配比"
        "（(0,1) 开区间）：候选 patients 排序 + seed 洗牌后按此份额切出"
        "held-out 侧——病例级不相交、永不参与判别器更新；与官方 val/test"
        "评估留出池天然不相交（官方 split 优先，#73 原则）。仅 MR-RATE "
        "线有语义，BraTS 线二分 = 70/10/20 写死（CaseSplitter）不消费",
        default=0.1, gt=0.0, lt=1.0,
    )
    sampling_manifest_json: Path | None = SpecField(
        "运行时", "#125 + #131",
        "MR-RATE 配额抽样留痕工件（#78 抽样机制同款：seed / 逐条件候选"
        "与抽取计数 / pool 与 held-out 逐卷归属 / 评估集互斥守卫读数/"
        "数据快照）——prepare 幂等与 held-out 互斥的可审计落档。仅 "
        "dataset=MR-RATE 有语义且必填；BraTS config 携带即拒绝",
        default=None,
    )
    pretrain_gate_auc: float = SpecField(
        "tunable", "ADR-0007",
        "RM readiness gate 门槛阈值：预训练 per-condition held-out AUC 的"
        "过线判定（暂定 0.65，T13 实测 chance 带 ≈ 0.5±0.02；MR-RATE "
        "预训练曲线校准后定版——ADR-0008 决策 6）。门槛数值只在预训练侧"
        "消费：train 上岗判定读报告条件白名单，不重算不复核（ADR-0008 "
        "决策 5，启动期池化重算废止）",
        default=0.65, gt=0.0, lt=1.0,
    )
    gate_support_min_volumes: int = SpecField(
        "tunable", "ADR-0008",
        "门槛支撑度界（ADR-0008 决策 6）：条件 held-out 卷数 < 此界时，该条件"
        "过线判据从池化点估计改为 bootstrap CI 下界 ≥ 门槛（卷级聚类重采样，"
        "重复数与分位固化在 cynosure.reward.support；MRA ≈ 16 卷命中走 CI "
        "口径、T2w ≈ 67 卷不命中走点估计）；≥ 界维持点估计口径（暂定 20，"
        "MR-RATE 预训练曲线校准后定版）",
        default=20, ge=1,
    )
    pretrain_max_steps: int = SpecField(
        "tunable", "ADR-0007",
        "预训练密集步进上限（远超在线期 1 step/iter；AUC 达门槛即提前终止，"
        "起步值待 DCU 预训练曲线校准）",
        default=2000, ge=1,
    )
    pretrain_fake_batch: int = SpecField(
        "tunable", "ADR-0007",
        "预训练每步量产的 fake 批量（base policy 冻结 rollout 的产出量；"
        "须覆盖判别器更新批的当前半区，装配期守卫）",
        default=16, ge=1,
    )
    pretrain_report_json: Path = SpecField(
        "运行时", "ADR-0007",
        "判别器预训练报告路径（kind 标识 + 最终 held-out AUC + 数据口径指纹；"
        "train 上岗门槛的守卫装载源，预训练 run 目录产物）。必填无默认——"
        "RL 不带 warm-start 工件在 schema 层就无法启动",
    )
    gating_dynamic_recovery: bool = SpecField(
        "tunable", "ADR-0008",
        "白名单动态恢复（ADR-0008 决策 8）：在线 per-condition AUC 流驱动 "
        "EMA 滞回判定，名单自动进出；false = 静态白名单降级路径（名单恒为"
        "预训练报告产物，gated 条件不自动恢复，判别器仍照常受训）",
        default=True,
    )
    gating_enter_auc: float = SpecField(
        "tunable", "ADR-0008",
        "动态恢复 enter 阈值（暂定 0.55，MR-RATE 预训练曲线校准后定版）："
        "gated 条件的 EMA(held-out AUC) 越过此线即恢复该条件的 policy 更新"
        "（判别力出带的自动上岗）",
        default=0.55, gt=0.0, lt=1.0,
    )
    gating_exit_auc: float = SpecField(
        "tunable", "ADR-0008",
        "动态恢复 exit 阈值（暂定 0.52，MR-RATE 预训练曲线校准后定版）："
        "名单内条件的 EMA(held-out AUC) 跌破此线即重新门控（拒绝在 RM "
        "无分辨率的样本上做策略梯度）",
        default=0.52, gt=0.0, lt=1.0,
    )
    gating_ema_span: int = SpecField(
        "tunable", "ADR-0008",
        "动态恢复的 EMA 跨度（暂定 8 iter，MR-RATE 预训练曲线校准后定版）："
        "per-condition AUC 的指数移动平均时间尺度（α = 2/(span+1)），"
        "观测流的平滑窗口——抑制单次测量的噪声进出",
        default=8, ge=1,
    )
    disc_noise_sigma_max: float = SpecField(
        "tunable", "ADR-0009",
        "判别器训练期对称噪声注入的强度上限（暂定 0.2，MR-RATE 预训练"
        "曲线校准后定版）：参数更新前向中 real/fake 两侧逐样本 "
        "σ ~ U[0, σ_max] 的归一化域加噪（σ 以相对通道 std 的比例参数化）；"
        "打分路径（reward / held-out AUC / 监控复算）恒干净域。"
        "σ_max = 0 是唯一关闭形态（回归锚：全链路与无注入逐位一致），"
        "不设独立 off 开关",
        default=0.2, ge=0.0,
    )
    overfit_ema_span: int = SpecField(
        "tunable", "ADR-0009",
        "过拟合分叉监控的 EMA 跨度（ADR-0009 决策 4，暂定 8——与 "
        "gating_ema_span 的 EMA(AUC) 跨度同值口径，MR-RATE 预训练曲线"
        "校准后定版）：per-condition 分叉 = EMA(train 干净域 pairwise "
        "acc − held-out AUC) 的平滑窗口（α = 2/(span+1)），观测流是"
        "该条件判别器步的稀疏序列（跨度语义 = 观测条数尺度）",
        default=8, ge=1,
    )
    overfit_alert_divergence: float = SpecField(
        "tunable", "ADR-0009",
        "overfit_alert 的分叉报警阈值（ADR-0009 决策 5，暂定 0.2，"
        "MR-RATE 预训练曲线校准后定版）：per-condition 分叉 EMA 自下"
        "而上越线即发 overfit_alert 事件——只报警、人工裁决，不自动移出"
        "白名单、不自动调 σ（升级项留校准后另议）。两侧同为 [0,1] 的 "
        "Mann-Whitney pairwise 占比，健康判别器的分叉贴 0；0 与 1 分属"
        "「任何正分叉即报警」的噪声区与「永不报警」的哑区，均不合法",
        default=0.2, gt=0.0, lt=1.0,
    )

    @field_validator("real_pool_quota")
    @classmethod
    def _quota_entries_positive(cls, value: dict[str, int]) -> dict[str, int]:
        """逐条件配额须为正（0 与负数 = 抽不出任何卷的死配额，属配置
        错误而非「关闭该条件」——条件缺席用不登记键表达）。"""
        starved = sorted(key for key, count in value.items() if count < 1)
        if starved:
            raise ValueError(
                f"real_pool_quota 配额须 ≥ 1（配额是抽样上限，死配额属"
                f"配置错误；关闭条件用不登记键表达）: {starved}"
            )
        return value

    @model_validator(mode="after")
    def _gating_hysteresis_band(self) -> "RewardConfig":
        """动态门控的滞回带形状：exit < enter（滞回带非空，防名单在
        阈值线上的进出抖动）且两者都在 chance（0.5）之上——AUC ≤ 0.5
        即判别器无分辨率，不存在「过线恢复」语义。"""
        enter, exit_ = self.gating_enter_auc, self.gating_exit_auc
        if not 0.5 < exit_ < enter < 1.0:
            raise ValueError(
                f"动态门控阈值须 0.5 < exit < enter < 1.0（滞回带非空、"
                f"均在 chance 之上），得到 enter={enter} exit={exit_}"
            )
        return self

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


class PreprocessingConfig(BaseModel):
    """prepare 读图编码的上游 recipe 参数（data-preparation spec + ADR-0006）。

    transform 链本体在 ``cynosure.reward.preprocessing``（MONAI 六步语义重写，
    零依赖）；config 只携带可注入参数：resize 基数与强度臂 clip。链中不存在
    spacing 重采样 / foreground crop / z-score，均为对齐结论、无参数可暴露。
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    resize_base: int = SpecField(
        "定死（fixture 可缩小）", "data-preparation + ADR-0006",
        "transform 链 resize 基数：每轴 max(round(size/base),1)×base，"
        "size 从 RAS 重定向后的空间形状读取；生产钉上游基数"
        f"（{UPSTREAM_RESIZE_BASE}，BraTS 240×240×155 → 256×256×128），"
        "fixture 注入小基数使夹具影像尺寸不变（须经 fixture_mode=true "
        "显式声明）。仅 BraTS 线有语义——MR-RATE 线 resize 目标 = 逐条件"
        "统一网格（词汇表工件携带），本字段取值偏离上游基数即拒绝"
        "（schema 守卫）",
        default=UPSTREAM_RESIZE_BASE, ge=1,
    )
    intensity_clip: bool = SpecField(
        "定死", "#130 + ADR-0006 + #71",
        "强度臂 clip 口径（#130 参数化）：BraTS 线 clip=True 是 ADR-0006"
        "裁决的 fork recipe 锚（fork issue #251 记录在案偏差）；MR-RATE "
        "线 clip=False 是 NVIDIA v1 官方口径（上游 transforms.py 原文，"
        "#71 裁决：对齐基座训练域）——两臂 embedding 不可互用，两域各自"
        "锁死裁决值（schema 守卫，显式携带错误值即拒绝）。另：MR-RATE 线"
        "的 resize_base 无语义（resize 目标 = 逐条件统一网格），取值偏离"
        "上游基数即拒绝",
        default=True,
    )
    encode_roi_size: list[int] = SpecField(
        "定死", "data-preparation + T12 复核探针",
        "VAE 编码滑动窗口的影像空间 roi（三轴；NVIDIA create_training_data 锚 "
        "[320,320,160]）——单样本空间体素数 ≤ roi 元素数时整前向豁免（上游 "
        "dynamic_infer 同语义，BraTS [1,1,256,256,128] 恒走此路）；超出时 roi "
        "逐轴 clamp 到影像尺寸后走 SlidingWindowInferer（b 语义，#143）",
        default=[320, 320, 160],
    )
    encode_overlap: float = SpecField(
        "定死", "data-preparation + T12 复核探针",
        "VAE 编码滑动窗口的重叠比（NVIDIA create_training_data 锚 0.4）。"
        "mode 定死 gaussian、sw_batch_size 定死 1（上游口径）",
        default=0.4, ge=0.0, lt=1.0,
    )


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
        "N_baseline：Baseline 样本量（与评估集同规模、同 seed 同条件、冻结只采一次；"
        "生产 200–500，fixture_mode 下放宽——Baseline manifest 条目随训练全流程走）",
        default=200, ge=1, le=500,
    )
    milestone_eval_samples: int = SpecField(
        "tunable", "本 spec 补钉",
        "里程碑解码评测样本数（取 Baseline manifest 条目的前缀，同 seed 同条件，"
        "使里程碑 FID/KID 跨里程碑可比）。小样本相对信号：K 只服务于训练期"
        "跨里程碑 plateau 比较（特征空间高维、K 小则协方差秩亏，绝对值噪声"
        "大）；验收口径 = N_baseline 全量对照（experiment-design「对照基线」）",
        default=8, ge=2,
    )
    decode_batch_size: int = SpecField(
        "tunable", "本 spec 补钉",
        "解码像素体的分块批大小（Baseline/重采与里程碑评测共用）：每批解码后"
        "即落盘/聚合，峰值显存以块为界——生产 200–500 条目的整 manifest "
        "单批解码是 OOM 级分配",
        default=8, ge=1,
    )
    decode_roi_size: list[int] = SpecField(
        "tunable", "本 spec 补钉",
        "VAE 解码滑动窗口的 latent 空间 roi（三轴；官方 NV-Generate-CTMR "
        "config_infer 的 autoencoder_sliding_window_infer_size = [48,48,48]）"
        "——生产大体积（256³ 级）整前向解码是 OOM 级分配；单样本单通道"
        "空间体素数 ≤ roi 元素数时整前向（官方 dynamic_infer 小体豁免"
        "语义，issue #142 单通道口径修正，fixture 恒走此路）",
        default=[48, 48, 48],
    )
    decode_overlap: float = SpecField(
        "tunable", "本 spec 补钉",
        "VAE 解码滑动窗口的重叠比（官方 autoencoder_sliding_window_infer_"
        "overlap 字面 0.6666 = 2/3 的四位截断；此处取满精度 2/3——MONAI "
        "要求 overlap×roi×zoom_scale 逐维为整数，VAE 4× 上采样下 "
        "48×2/3×4=128 整除成立）。mode 定死 gaussian（官方口径）",
        default=0.6666666666666666, ge=0.0, lt=1.0,
    )
    kid_bootstrap_replicates: int = SpecField(
        "tunable", "本 spec 补钉",
        "KID 置信区的无放回重采样重复数（重复越多 CI 越稳，代价线性增长）",
        default=20, ge=1,
    )
    n_plateau: int = SpecField(
        "运行时", "experiment-design",
        "N_plateau：早停 plateau 里程碑数（默认 3）",
        default=3, ge=1,
    )
    plateau_tolerance: float = SpecField(
        "tunable", "experiment-design",
        "主判据（FID）plateau 判定容差：里程碑 FID 相对历史最优的改善 ≤ 容差"
        "视为 plateau（绝对阈值为训练期经验数据，spec 只钉判据形态）",
        default=0.5, ge=0.0,
    )
    auc_chance_epsilon: float = SpecField(
        "tunable", "experiment-design",
        "hacking 签名的「AUC 近 chance」判定带半径：|AUC − 0.5| ≤ ε",
        default=0.02, gt=0.0,
    )
    reward_trend_window: int = SpecField(
        "tunable", "experiment-design",
        "「eval reward 仍升」判定的迭代窗口（最近 W 个 iter 事件的"
        " anchor eval reward 线性斜率 > 0）",
        default=10, ge=2,
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

    @field_validator("auc_chance_epsilon")
    @classmethod
    def _chance_band_within_half(cls, value: float) -> float:
        if value >= 0.5:
            raise ValueError(
                "AUC 近 chance 判定带半径必须小于 0.5（带触 0.5 即恒真，"
                "hacking 签名失去判别力）"
            )
        return value


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


_MR_ASSEMBLY_ARTIFACTS: tuple[str, ...] = (
    "mrrate_metadata_csv",
    "mrrate_splits_csv",
    "eval_manifest_csv",
    "mrrate_data_snapshot",
)
"""MR-RATE prepare 装配输入工件四件套的字段名（#121/#131）：MR-RATE
线必填、BraTS 线携带即拒的互斥绑定清单（模块级常量——类体下划线属性
会被 pydantic 当 private attr 收编，validator 里不可迭代）。"""


class CynosureConfig(BaseModel):
    """cynosure 全量运行配置：train / eval / prepare / pretrain 四子命令共享同一 schema。"""

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
        "fixture 诊断模式显式声明：true 才允许缩小采样日程（如 3 步 ODE）与"
        "缩小预处理链参数（resize_base）；缺省 false 时生产 config 钉 30 步、"
        "上游 resize 基数",
        default=False,
    )
    preprocessing: PreprocessingConfig = SpecField(
        "定死", "data-preparation + ADR-0006",
        "prepare 读图编码的上游 recipe 参数（transform 链的 resize 基数）",
        default_factory=PreprocessingConfig,
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

    @field_validator("artifacts")
    @classmethod
    def _condition_vocabulary_matches_dataset(
        cls, artifacts: Artifacts, info: ValidationInfo,
    ) -> Artifacts:
        """条件词汇表工件与 dataset 互斥绑定（#119 互斥哲学在工件形态
        下的延续）：MR-RATE 缺工件即拒绝（11 生成条件的取数域无从装配）；
        BraTS 携带即拒绝（拼错 dataset 时两套口径静默共存比显式拒绝
        危险，同 conditioning 段先例）。"""
        experiment = info.data.get("experiment")
        if experiment is None:
            return artifacts
        if experiment.dataset == "MR-RATE":
            if artifacts.condition_vocabulary_json is None:
                raise ValueError(
                    "dataset=\"MR-RATE\" 须提供 artifacts."
                    "condition_vocabulary_json（11 生成条件五元组工件，"
                    "#127；词表唯一来源，config 不内嵌词表）"
                )
        elif artifacts.condition_vocabulary_json is not None:
            raise ValueError(
                "artifacts.condition_vocabulary_json（MR-RATE 条件词汇表"
                f"工件）仅对 dataset=\"MR-RATE\" 有语义，得到 dataset="
                f"{experiment.dataset}（两套口径互斥激活：BraTS config "
                "携带 MR 词汇工件即拒绝）"
            )
        return artifacts


    @field_validator("artifacts")
    @classmethod
    def _mr_assembly_artifacts_match_dataset(
        cls, artifacts: Artifacts, info: ValidationInfo,
    ) -> Artifacts:
        """MR-RATE prepare 装配输入工件四件套与 dataset 互斥绑定（#121/
        #131，spec #125 实现决策 3）：MR-RATE 缺任一件即拒绝（配额抽样
        与评估集互斥守卫的取数域无从装配）；BraTS 携带即拒绝（MR 输入
        工件指向不存在的布局，携带即口径混乱信号）。"""
        experiment = info.data.get("experiment")
        if experiment is None:
            return artifacts
        if experiment.dataset == "MR-RATE":
            missing = [
                name for name in _MR_ASSEMBLY_ARTIFACTS
                if getattr(artifacts, name) is None
            ]
            if missing:
                raise ValueError(
                    "dataset=\"MR-RATE\" 须提供 MR prepare 装配输入工件"
                    f"（artifacts.{' / artifacts.'.join(missing)}）："
                    "配额抽样 manifest、评估集互斥守卫与 provenance 留痕"
                    "的取数域（#131）"
                )
        else:
            carried = [
                name for name in _MR_ASSEMBLY_ARTIFACTS
                if getattr(artifacts, name) is not None
            ]
            if carried:
                raise ValueError(
                    "artifacts."
                    f"{' / artifacts.'.join(carried)}（MR-RATE prepare "
                    "装配输入工件）仅对 dataset=\"MR-RATE\" 有语义，得到 "
                    f"dataset={experiment.dataset}（两套口径互斥激活："
                    "BraTS config 携带 MR 装配工件即拒绝）"
                )
        return artifacts

    @field_validator("preprocessing")
    @classmethod
    def _intensity_arm_matches_dataset(
        cls, preprocessing: PreprocessingConfig, info: ValidationInfo,
    ) -> PreprocessingConfig:
        """强度臂两域锁死（#130 参数化）+ MR 线 resize 口径互斥（#121/
        #131）：BraTS 臂 clip=True 是 ADR-0006 裁决的 fork recipe 锚
        （fork issue #251 记录在案偏差），MR-RATE 臂 clip=False 是
        NVIDIA v1 官方口径（#71 裁决：对齐基座训练域）——两臂 embedding
        不可互用，显式携带错误值即拒绝（静默换 recipe 比显式拒绝危险）。
        MR 线 resize 目标 = 逐条件统一网格（词汇表工件携带），resize 基数
        公式无语义——偏离上游基数的取值即拒绝（显式性无法跨 JSON
        roundtrip 判读：dump 会把默认值写成显式键，故守卫落值域；
        防两套 resize 口径静默共存）。"""
        experiment = info.data.get("experiment")
        if experiment is None:
            return preprocessing
        if experiment.dataset == "MR-RATE":
            if preprocessing.intensity_clip:
                raise ValueError(
                    "dataset=\"MR-RATE\" 的强度臂须 clip=False（NVIDIA v1 "
                    "官方口径，#71/#130 裁决：对齐基座训练域；clip=True "
                    "臂属 BraTS 线，两臂 embedding 不可互用），显式置 "
                    "preprocessing.intensity_clip=true 即拒绝"
                )
            if preprocessing.resize_base != UPSTREAM_RESIZE_BASE:
                raise ValueError(
                    "MR-RATE 线的 resize 目标 = 逐条件统一网格（条件词汇"
                    "表工件携带，spec #125 实现决策 3），preprocessing."
                    "resize_base 基数公式（BraTS 口径）无语义——取值偏离"
                    f"上游基数 {UPSTREAM_RESIZE_BASE} 即拒绝，防两套 "
                    "resize 口径静默共存"
                )
        elif not preprocessing.intensity_clip:
            raise ValueError(
                "dataset=\"BraTS2023\" 的强度臂须 clip=True（ADR-0006 "
                "裁决的 fork recipe 锚，fork issue #251 记录在案偏差）："
                "显式置 preprocessing.intensity_clip=false 等于换基座"
                "训练分布，须先经 ADR 层重论证"
            )
        return preprocessing

    @field_validator("reward")
    @classmethod
    def _mr_sampling_manifest_matches_dataset(
        cls, reward: RewardConfig, info: ValidationInfo,
    ) -> RewardConfig:
        """配额抽样留痕工件与 dataset 互斥绑定（#131）：MR-RATE 必填
        （prepare 幂等与 held-out 互斥的可审计落档缺位即拒绝）；BraTS
        携带即拒绝（同 MR 装配工件守卫哲学）。"""
        experiment = info.data.get("experiment")
        if experiment is None:
            return reward
        if experiment.dataset == "MR-RATE":
            if reward.sampling_manifest_json is None:
                raise ValueError(
                    "dataset=\"MR-RATE\" 须提供 reward.sampling_manifest_"
                    "json（配额抽样留痕工件路径：seed / 逐条件计数 / "
                    "pool 与 held-out 逐卷归属 / 互斥守卫读数，#131——"
                    "held-out 互斥「落档可查」的登记面）"
                )
        elif reward.sampling_manifest_json is not None:
            raise ValueError(
                "reward.sampling_manifest_json（MR-RATE 配额抽样留痕"
                f"工件）仅对 dataset=\"MR-RATE\" 有语义，得到 dataset="
                f"{experiment.dataset}（两套口径互斥激活：BraTS config "
                "携带即拒绝）"
            )
        return reward

    @model_validator(mode="after")
    def _inference_steps_match_mode(self) -> "CynosureConfig":
        """缩小采样日程的通道显式化：fixture_mode=false 时 num_inference_steps 钉 30。"""
        if not self.fixture_mode and self.policy.num_inference_steps != 30:
            raise ValueError(
                "生产 config（fixture_mode=false）下 num_inference_steps 定死 30"
                f"（policy-modeling 章），得到 {self.policy.num_inference_steps}；"
                "缩小日程属 fixture，须经顶层 fixture_mode=true 显式声明"
            )
        if not self.fixture_mode and self.schedule.baseline_samples < 200:
            raise ValueError(
                f"生产 config（fixture_mode=false）下 N_baseline 口径 200–500"
                f"（experiment-design 章），得到 {self.schedule.baseline_samples}；"
                "缩小样本量属 fixture，须经顶层 fixture_mode=true 显式声明"
            )
        return self

    def stage_condition_vocabulary(self) -> dict[int, list]:
        """组 → {阶段号: 条件词汇表} 的唯一映射（组1 四序列、组2 12 有序对、
        组3 两阶段各一份）——Baseline manifest 条目生成与本 validator 共同
        消费（词汇表单一来源，组矩阵一变只改此处）。"""
        pairs = [list(pair) for pair in self.experiment.cross_modal_pairs]
        return {
            "modal-label": {1: list(MODALITIES)},
            "cross-modal": {1: pairs},
            "sequential": {1: list(MODALITIES), 2: pairs},
        }[self.experiment.group]

    @model_validator(mode="after")
    def _resize_base_matches_mode(self) -> "CynosureConfig":
        """缩小预处理链参数的通道显式化：fixture_mode=false 时 resize_base 钉
        上游基数——生产上静默偏离上游 recipe 等于换基座训练分布（ADR-0006）。"""
        if not self.fixture_mode and self.preprocessing.resize_base != UPSTREAM_RESIZE_BASE:
            raise ValueError(
                "生产 config（fixture_mode=false）下 preprocessing.resize_base "
                f"定死上游基数 {UPSTREAM_RESIZE_BASE}（ADR-0006），得到 "
                f"{self.preprocessing.resize_base}；注入小基数属 fixture，"
                "须经顶层 fixture_mode=true 显式声明"
            )
        return self

    @model_validator(mode="after")
    def _milestone_samples_match_manifest_support(self) -> "CynosureConfig":
        """里程碑评测样本面与 manifest 支撑面一致（两个方向都显式拒绝）：

        - **下界**：manifest 条目按条件轮转，前缀取样 K < 词汇表即
          **永久**漏掉尾部方向——早停判据对那些方向失明（12 有序对只
          评前 K 个）；
        - **上界**：评测条目取 manifest 前缀，K > N_baseline 即静默
          缩水到盘上条目数——配置声明的评测样本量与实际评测面失真。

        fixture 豁免（条目数随 fixture 缩小，覆盖以盘上条目为准）。"""
        if self.fixture_mode:
            return self
        vocabulary = max(
            len(conditions)
            for conditions in self.stage_condition_vocabulary().values()
        )
        if self.schedule.milestone_eval_samples < vocabulary:
            raise ValueError(
                f"生产 config 下 schedule.milestone_eval_samples 须覆盖本组"
                f"条件词汇表（{vocabulary} 个条件；manifest 条件轮转下 K "
                f"不足即永久漏方向），得到 "
                f"{self.schedule.milestone_eval_samples}；"
                "缩小评测面属 fixture，须经顶层 fixture_mode=true 显式声明"
            )
        if self.schedule.milestone_eval_samples > self.schedule.baseline_samples:
            raise ValueError(
                f"生产 config 下 schedule.milestone_eval_samples 不得超过 "
                f"schedule.baseline_samples（评测条目取 manifest 前缀，"
                f"超出即静默缩水到盘上条目数、评测面与配置声明失真），"
                f"得到 {self.schedule.milestone_eval_samples} > "
                f"{self.schedule.baseline_samples}；全量对照评估直接用 "
                f"N_baseline 口径"
            )
        return self


class ConfigLoader:
    """config 文件装载：JSON 反序列化 + schema 校验（四子命令共用）。"""

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
    "PreprocessingConfig",
    "REGISTERED_DATASETS",
    "RewardConfig",
    "ScheduleConfig",
    "ShardingConfig",
    "SPACING_CONDITION_SCALE",
    "SpecField",
    "UNCONDITIONAL_LABEL",
    "UPSTREAM_RESIZE_BASE",
]
