"""组1 采样场：CFG=10 组合场（ADR-0002，逐字复刻基座 batch 组织）。

组合公式与 batch 组织是基座推理的字面复刻（policy-modeling 章 / 用户故事
#2）：batch=2B 单次前向按 ``chunk(2)`` 拆分、序 [cond, uncond]、无条件分支
= 全零 label（训练期 label 0 实际承担「无条件」语义）——CFG 组合的唯一
发生地（对齐基座 ``_unet_output`` 形态）。

组内效率技巧（G²RPO）：扰动步全组共享同一 ``x_k``，无条件分支 batch=1
一次评估全组复用、条件分支 batch=1 一次评估后 expand 成 G——组合场数值
不变，仅省去无条件分支随方向复制 G 倍的算力。
``spacing_tensor``（体素间距 ×1e2）恒传（基座 ``include_spacing_input=true``）。
"""

import torch
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from cynosure.config import CFG_MODAL_LABEL
from cynosure.policy.condition import RolloutCondition


class CfgCombinedField:
    """CFG 组合场（组1）：v_cfg = v_uncond + w·(v_cond − v_uncond)，w=10 定死。"""

    CFG_WEIGHT: float = CFG_MODAL_LABEL
    """组合场的引导强度 w（组1 定死 10，ADR-0002）。"""

    def __init__(self, unet: DiffusionModelUNetMaisi) -> None:
        self._unet = unet

    def velocity(
        self,
        x: torch.Tensor,
        timesteps: int,
        condition: RolloutCondition,
    ) -> torch.Tensor:
        """组合场 velocity：batch=2B 单次前向、序 [cond, uncond]、无条件
        分支全零 label（锚轨迹与 ODE 续跑走此基座组织）。"""
        batch = x.shape[0]
        cond = condition.broadcast_to(batch)
        paired_x = torch.cat((x, x), dim=0)
        paired_labels = torch.cat((cond.label, torch.zeros_like(cond.label)), dim=0)
        paired_spacing = torch.cat((cond.spacing, cond.spacing), dim=0)
        paired_velocity = self._forward(paired_x, batch * 2, timesteps, paired_labels, paired_spacing)
        v_cond, v_uncond = torch.chunk(paired_velocity, 2)
        return self._combine(v_cond, v_uncond)

    def group_velocity(
        self,
        x_shared: torch.Tensor,
        timesteps: int,
        condition: RolloutCondition,
        group_size: int,
    ) -> torch.Tensor:
        """扰动步的组内 velocity（全组共享 x_k）：无条件分支与条件分支各
        batch=1 评估一次，组合后 expand 成 G（零拷贝复用）。"""
        v_cond = self._forward(x_shared, 1, timesteps, condition.label, condition.spacing)
        v_uncond = self._forward(
            x_shared, 1, timesteps, torch.zeros_like(condition.label), condition.spacing,
        )
        combined = self._combine(v_cond, v_uncond)
        return combined.expand(group_size, *combined.shape[1:])

    @staticmethod
    def _combine(v_cond: torch.Tensor, v_uncond: torch.Tensor) -> torch.Tensor:
        """组合公式（唯一发生地）：v_cfg = v_uncond + w·(v_cond − v_uncond)。"""
        return v_uncond + CfgCombinedField.CFG_WEIGHT * (v_cond - v_uncond)

    def _forward(
        self,
        x: torch.Tensor,
        batch: int,
        timesteps: int,
        labels: torch.Tensor,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        return self._unet(
            x=x,
            timesteps=torch.full(
                (batch,), timesteps, dtype=torch.int64, device=x.device,
            ),
            class_labels=labels,
            spacing_tensor=spacing,
        )
