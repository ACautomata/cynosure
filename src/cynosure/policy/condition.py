"""组条件 c：MDP state ``s_t = (c, t, x_t)`` 里的条件（policy-modeling 章）。

组1 = (modality label, spacing)；组2 将再带源影像 latent × scale_factor
（ControlNet 条件）。同批 rollout 的条件共享：label 允许 batch=1 广播
（广播由组合场负责），spacing（体素间距 ×1e2）恒传（基座
``include_spacing_input=true``）。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from cynosure.config import MODALITIES


@dataclass(frozen=True)
class RolloutCondition:
    """一条 rollout 的采样条件：模态标签 token + 体素间距。"""

    label: torch.Tensor
    """modality token（int64），形状 [B]；同批共享时可为 [1]。"""

    spacing: torch.Tensor
    """体素间距 ×1e2，形状 [B, 3]；同批共享时可为 [1, 3]。"""


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
        """序列的 modality token（组1 条件分支的 class label）。"""
        return self._labels[modality]
