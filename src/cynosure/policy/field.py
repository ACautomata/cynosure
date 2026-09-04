"""按组采样场（policy-modeling 章「Policy = 按组对齐基座 CFG 的采样场」）。

组1 = CFG=10 组合场（CfgCombinedField，逐字复刻基座 batch 组织）；组2 =
CFG=0 裸条件单前向（BareConditionField，frozen base UNet + trainable
ControlNet 残差注入）。两组实现同一 VelocityField 接口——rollout 编排
与 log-prob 重算对采样场无感知，组间差异全部收敛在场的实现里。

组内效率技巧（G²RPO）对两组同构：扰动步全组共享同一 ``x_k``，组内
``group_velocity`` 以 batch=1 前向得出组合 velocity 后 expand 成 G。
``spacing_tensor``（体素间距 ×1e2）恒传（基座 ``include_spacing_input=true``）。
"""

from typing import Protocol

import torch
from monai.apps.generation.maisi.networks.controlnet_maisi import ControlNetMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)

from cynosure.config import CFG_MODAL_LABEL
from cynosure.policy.condition import RolloutCondition


class VelocityField(Protocol):
    """采样场接口：给 (x, t, c) 求 velocity 的两种组级实现（组合场 /
    裸条件单前向）。rollout 编排与 GRPO 更新只依赖本接口。"""

    def velocity(
        self, x: torch.Tensor, timesteps: int, condition: RolloutCondition,
    ) -> torch.Tensor:
        """采样场 velocity（batch 维并行；Anchor 轨迹与 ODE 续跑走此入口）。"""
        ...

    def group_velocity(
        self,
        x_shared: torch.Tensor,
        timesteps: int,
        condition: RolloutCondition,
        group_size: int,
    ) -> torch.Tensor:
        """扰动步的组内 velocity（全组共享 x_k，batch=1 评估后 expand 成 G）。"""
        ...


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


class BareConditionField:
    """组2 裸条件单前向（CFG=0，ADR-0002：基座代码强制）：frozen base
    UNet + trainable ControlNet 残差每次前向都参与——无无条件分支、无
    组合公式，velocity = UNet(x, residuals=ControlNet(x, cond))。

    双条件（policy-modeling 章 MDP 条件 c）：源影像 latent × scale_factor
    （ControlNet 的 controlnet_cond；缩放唯一发生地在构造参数
    ``source_latent_scale_factor``）+ 目标序列 modality token（ControlNet
    与 UNet 的 class label 同源）。ControlNet 的 ``conditioning_scale``
    保持 MONAI 默认 1.0（基座行为；与源 latent 缩放是两个正交旋钮）。"""

    def __init__(
        self,
        unet: DiffusionModelUNetMaisi,
        controlnet: ControlNetMaisi,
        source_latent_scale_factor: float,
    ) -> None:
        self._unet = unet
        self._controlnet = controlnet
        self._scale_factor = source_latent_scale_factor

    def velocity(
        self,
        x: torch.Tensor,
        timesteps: int,
        condition: RolloutCondition,
    ) -> torch.Tensor:
        """裸条件单前向 velocity（batch 维并行；残差按 batch 一次算出）。"""
        batch = x.shape[0]
        cond = condition.broadcast_to(batch)
        residuals = self._control_residuals(x, batch, timesteps, cond)
        return self._unet_forward(x, batch, timesteps, cond, *residuals)

    def group_velocity(
        self,
        x_shared: torch.Tensor,
        timesteps: int,
        condition: RolloutCondition,
        group_size: int,
    ) -> torch.Tensor:
        """扰动步的组内 velocity（全组共享 x_k）：ControlNet 与 UNet 各
        batch=1 一次前向（G²RPO 效率技巧，与组1 组合同构），velocity
        输出 expand 成 G（零拷贝复用）——组内 G 条方向共享同一 x_k 与
        条件，batch-G 前向是 G 倍显存/算力的纯浪费。"""
        down_residuals, mid_residual = self._control_residuals(
            x_shared, 1, timesteps, condition,
        )
        velocity = self._unet_forward(
            x_shared, 1, timesteps, condition, down_residuals, mid_residual,
        )
        return velocity.expand(group_size, *velocity.shape[1:])

    def _control_residuals(
        self,
        x: torch.Tensor,
        batch: int,
        timesteps: int,
        condition: RolloutCondition,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """ControlNet 残差：(down_block_res_samples, mid_block_res_sample)。"""
        if condition.source_latent is None:
            raise ValueError(
                "组2 采样场需要源影像 latent 条件（RolloutCondition."
                "source_latent）：缺失即跨模态对齐静默失效"
            )
        down_residuals, mid_residual = self._controlnet(
            x=x,
            timesteps=torch.full(
                (batch,), timesteps, dtype=torch.int64, device=x.device,
            ),
            controlnet_cond=condition.source_latent * self._scale_factor,
            class_labels=condition.label,
        )
        return tuple(down_residuals), mid_residual

    def _unet_forward(
        self,
        x: torch.Tensor,
        batch: int,
        timesteps: int,
        condition: RolloutCondition,
        down_residuals: tuple[torch.Tensor, ...],
        mid_residual: torch.Tensor,
    ) -> torch.Tensor:
        return self._unet(
            x=x,
            timesteps=torch.full(
                (batch,), timesteps, dtype=torch.int64, device=x.device,
            ),
            class_labels=condition.label,
            spacing_tensor=condition.spacing,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
        )
