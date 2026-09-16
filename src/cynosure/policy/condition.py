"""组条件 c：MDP state ``s_t = (c, t, x_t)`` 里的条件（policy-modeling 章）。

组1 = (modality label, spacing)；组2 再带源影像 latent 与源模态 label
（ControlNet 条件，乘 scale_factor 发生在组2 采样场——条件的唯一缩放
点）。双 label 各收其职（issue #115）：目标模态 token 随 UNet 前向、源
模态 token 随 ControlNet 前向（组1 无源模态位，源 label 为 ``None``）。
同批 rollout 的条件共享：label/spacing/源位允许 batch=1 广播（广播由
采样场负责），spacing（体素间距 ×1e2）恒传（基座
``include_spacing_input=true``）。
"""

from dataclasses import dataclass

import torch

from cynosure.conditions import CONDITION_SPACING_X1E2, ModalityMapping

__all__ = [
    "CONDITION_SPACING_X1E2",
    "ModalityMapping",
    "RolloutCondition",
]

CONDITION_SPACING_X1E2 = CONDITION_SPACING_X1E2
"""组1 条件的体素间距常量（re-export：常量定义位在 cynosure.conditions
——两域词汇 spacing 语义的共享概念；import 方向单向 policy → conditions
防环）。语义：1.0 × 1e2（fixture 单位间距；基座 ``include_spacing_input=
true`` 的 ×1e2 恒传口径）。组1 条件只含 label、无源影像 case 可依；
组2 源影像条件的 spacing 已接 manifest per-case 侧车（issue #46：与源
latent 同条目同源），不再消费本常量。本模块是 import 环安全位——
train/eval 两侧条件组装共同依赖。"""


@dataclass(frozen=True)
class RolloutCondition:
    """一条 rollout 的采样条件：目标模态标签 token + 体素间距（+ 组2 的
    源影像 latent 与源模态标签）+ 条件名（sigma 日程选择键，#129）。"""

    label: torch.Tensor
    """目标模态 token（int64），形状 [B]；同批共享时可为 [1]。UNet 前向
    的 class label（组1/组2 共用）。"""

    spacing: torch.Tensor
    """体素间距 ×1e2，形状 [B, 3]；同批共享时可为 [1, 3]。"""

    source_latent: torch.Tensor | None = None
    """组2 源位之一：源影像 latent（[B, C, D, H, W]，ControlNet 条件的
    缩放前形态）；组1 为 ``None``。与 label/spacing 同 batch（构造即校验）。"""

    source_label: torch.Tensor | None = None
    """组2 源位之二：源模态 token（int64），形状 [B]；组1 为 ``None``。
    ControlNet 前向的 class label（issue #115 各收其职：ControlNet 解读
    源影像，残差按源模态分化；UNet 保持目标 label）。源位一致性
    （与 source_latent 同齐同缺）由构造期 contract 保证（issue #117）。"""

    name: str | None = None
    """本条件的域键（#129）：BraTS = 目标序列名（t1n/t1c/t2w/t2f），
    MR-RATE = 生成条件名（t1w/axial 等）——sigma 日程按名选择（
    ``ConditionSchedules.cursor``）；BraTS 单域日程对任意名（含
    ``None``）恒等。组2 条目取目标端序列名。"""

    def __post_init__(self) -> None:
        if self.source_latent is not None and self.source_label is None:
            raise ValueError(
                "组2 条件构造缺源模态 label（RolloutCondition.source_label）："
                "源影像 latent 与源 label 须同源齐备（issue #117 收紧为组2 "
                "必填——漏 label 即 ControlNet 退回目标 label 的同源错位语义，"
                "构造期拒绝而非静默默认）"
            )
        if self.source_latent is None and self.source_label is not None:
            raise ValueError(
                "条件源位不齐：source_label 在场而 source_latent 缺席"
                "（源位须同齐同缺——组1 双位缺席、组2 双位齐备；"
                "组2 构造漏传 source_latent 亦落本分支）"
                "（issue #117 源位一致性 contract）"
            )
        if self.label.shape[0] != self.spacing.shape[0]:
            raise ValueError(
                f"条件 batch 不符：label {self.label.shape[0]} vs spacing "
                f"{self.spacing.shape[0]}"
            )
        if (
            self.source_latent is not None
            and self.source_latent.shape[0] != self.label.shape[0]
        ):
            raise ValueError(
                f"条件 batch 不符：label {self.label.shape[0]} vs source_latent "
                f"{self.source_latent.shape[0]}"
            )
        if (
            self.source_label is not None
            and self.source_label.shape[0] != self.label.shape[0]
        ):
            raise ValueError(
                f"条件 batch 不符：label {self.label.shape[0]} vs source_label "
                f"{self.source_label.shape[0]}"
            )

    def name_or_raise(self) -> str:
        """条件键（sigma 日程与 latent 形状解析的贯通键，#129）：缺失
        即显式拒绝——逐条件贯通后无名条件无从按条件解析形状（
        ``ConditionVocabulary.latent_shape`` 的调用前置；单域日程对无名
        的容忍是兼容面，形状解析不容忍）。"""
        if self.name is None:
            raise ValueError(
                "rollout 条件缺条件名（RolloutCondition.name）：latent 形状"
                "按条件解析（#129），无名条件无从解析——BraTS 线请传目标"
                "序列名，MR-RATE 线请传生成条件名"
            )
        return self.name

    def broadcast_to(self, batch: int) -> "RolloutCondition":
        """同批 rollout 的条件共享：batch=1 的条件广播到整批
        （G 方向并行续跑共享同一条件），batch 数不符即显式拒绝。"""
        if self.label.shape[0] == 1 and batch > 1:
            source_latent = (
                self.source_latent.expand(batch, *self.source_latent.shape[1:])
                if self.source_latent is not None else None
            )
            source_label = (
                self.source_label.expand(batch, *self.source_label.shape[1:])
                if self.source_label is not None else None
            )
            return RolloutCondition(
                label=self.label.expand(batch, *self.label.shape[1:]),
                spacing=self.spacing.expand(batch, *self.spacing.shape[1:]),
                source_latent=source_latent,
                source_label=source_label,
                name=self.name,
            )
        if self.label.shape[0] != batch:
            raise ValueError(
                f"条件 batch {self.label.shape[0]} 与样本 batch {batch} 不符"
                "（仅支持 batch=1 广播或逐元素对齐）",
            )
        return self
