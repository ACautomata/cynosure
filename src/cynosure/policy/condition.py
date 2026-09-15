"""组条件 c：MDP state ``s_t = (c, t, x_t)`` 里的条件（policy-modeling 章）。

组1 = (modality label, spacing)；组2 再带源影像 latent 与源模态 label
（ControlNet 条件，乘 scale_factor 发生在组2 采样场——条件的唯一缩放
点）。双 label 各收其职（issue #115）：目标模态 token 随 UNet 前向、源
模态 token 随 ControlNet 前向（组1 无源模态位，源 label 为 ``None``）。
同批 rollout 的条件共享：label/spacing/源位允许 batch=1 广播（广播由
采样场负责），spacing（体素间距 ×1e2）恒传（基座
``include_spacing_input=true``）。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from cynosure.config import MODALITIES

CONDITION_SPACING_X1E2: tuple[float, float, float] = (100.0, 100.0, 100.0)
"""组1 条件的体素间距常量（1.0 × 1e2，fixture 单位间距；基座
``include_spacing_input=true`` 的 ×1e2 恒传口径）。组1 条件只含 label、
无源影像 case 可依；组2 源影像条件的 spacing 已接 manifest per-case
侧车（issue #46：与源 latent 同条目同源），不再消费本常量。本模块是
import 环安全位——train/eval 两侧条件组装共同依赖。"""


@dataclass(frozen=True)
class RolloutCondition:
    """一条 rollout 的采样条件：目标模态标签 token + 体素间距（+ 组2 的
    源影像 latent 与源模态标签）。"""

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
                "组1 条件构造携带源模态 label（RolloutCondition.source_label）："
                "源 label 是组2 专属位，须与源影像 latent 同齐同缺"
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
            )
        if self.label.shape[0] != batch:
            raise ValueError(
                f"条件 batch {self.label.shape[0]} 与样本 batch {batch} 不符"
                "（仅支持 batch=1 广播或逐元素对齐）",
            )
        return self


class ModalityMapping:
    """组1 模态标签映射（spec 输入物 modality_mapping：t1n/t1c/t2w/t2f →
    modality token）。从工件装载、单一来源——不设代码内常量副本，防与
    基座映射静默漂移。"""

    def __init__(self, labels: dict[str, int]) -> None:
        missing = [modality for modality in MODALITIES if modality not in labels]
        if missing:
            raise ValueError(
                f"modality mapping 缺少序列 {missing}（须覆盖 {MODALITIES}）",
            )
        self._labels = dict(labels)

    @classmethod
    def load(cls, path: Path) -> "ModalityMapping":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"modality mapping 工件须为 JSON 对象: {path}")
        return cls({str(key): int(value) for key, value in data.items()})

    def label(self, modality: str) -> int:
        """序列的 modality token（组1 条件与组2 双 label 的共同取数点：
        目标/源模态 token 都从本映射查表，杜绝代码内副本漂移）。"""
        return self._labels[modality]
