"""每组 policy 侧装配（glossary「Policy」按组实例化的 Facade）。

Policy = 被 RL 训练的扩散模型——模态标签阶段是 base UNet（全参更新），
跨模态阶段是 ControlNet（base UNet 冻结）。本模块把「可训练网络 + 采样场
+ 条件分布 + 优化器」收敛为单点装配：**训练循环的采样/更新语义**（trainer /
rollout / updater）经本装配对组无感知；编排层的组间分流（序贯两阶段、
诊断工件门槛）仍由 CLI / SequentialTrainer 按 experiment.group 处理。

base 冻结经断言验证（issue #23 验收）：组2 装配期显式关闭 UNet 全部参数
的 ``requires_grad`` 并复检——静默未冻结会让「仅训 ControlNet」退化为
全参训练，显存与优化语义双双失真。
"""

import torch

from cynosure.config import CynosureConfig
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import ModalityMapping
from cynosure.policy.field import BareConditionField, CfgCombinedField, VelocityField
from cynosure.reward.artifacts import LatentManifest
from cynosure.train.rollout import (
    ConditionSampler,
    CrossModalConditionSampler,
    ModalLabelConditionSampler,
    SourceLatentPool,
)


class GroupPolicy:
    """一次训练的 policy 侧装配：可训练网络、base 网络、采样场、条件分布
    与优化器的单点持有（trainer 只面对本类的引用组）。"""

    def __init__(
        self,
        unet: torch.nn.Module,
        network: torch.nn.Module,
        field: VelocityField,
        conditions: ConditionSampler,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._unet = unet
        self._network = network
        self._field = field
        self._conditions = conditions
        self._optimizer = optimizer

    @classmethod
    def build(
        cls,
        config: CynosureConfig,
        generator: torch.Generator,
        device: torch.device,
    ) -> "GroupPolicy":
        """按组装配（三组实验矩阵的唯一分派点）：网络工件经 netbuild 装载、
        可训练对象与采样场/条件分布按 experiment.group 落位。"""
        group = config.experiment.group
        if group == "sequential":
            raise ValueError(
                "组3 的两阶段序贯由 SequentialTrainer 编排（单次训练循环"
                "只面对一个可训练对象）"
            )
        unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=config.artifacts.unet_ckpt,
        )).to(device)
        mapping = ModalityMapping.load(config.artifacts.modality_mapping_json)
        if group == "cross-modal":
            network = cls._assemble_cross_modal(unet, config, device)
            field: VelocityField = BareConditionField(
                unet,
                network,
                config.policy.source_latent_scale_factor,
            )
            conditions: ConditionSampler = CrossModalConditionSampler(
                mapping,
                [tuple(pair) for pair in config.experiment.cross_modal_pairs],
                SourceLatentPool(
                    LatentManifest.load(
                        config.reward.real_pool_manifest, kind="real_pool",
                    ),
                    device,
                ),
                generator,
                device,
            )
        else:
            network = unet
            field = CfgCombinedField(unet)
            conditions = ModalLabelConditionSampler(mapping, generator, device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=config.policy.policy_lr,
            weight_decay=config.policy.policy_weight_decay,
        )
        return cls(unet, network, field, conditions, optimizer)

    @staticmethod
    def _assemble_cross_modal(
        unet: torch.nn.Module, config: CynosureConfig, device: torch.device,
    ) -> torch.nn.Module:
        """组2/组3-stage2 的可训练对象 = ControlNet；base UNet 冻结
        （requires_grad 关闭 + 复检断言，AC「base 冻结经断言验证」）。"""
        controlnet = NetworkAssembler.controlnet(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.controlnet_config_json,
            ),
            checkpoint=config.artifacts.controlnet_ckpt,
        )).to(device)
        unet.requires_grad_(False)
        if any(p.requires_grad for p in unet.parameters()):
            raise ValueError(
                "组2 的 base UNet 冻结失败：仍有参数 requires_grad"
                "（仅训 ControlNet 的优化语义被破坏）"
            )
        return controlnet

    @property
    def unet(self) -> torch.nn.Module:
        """base UNet（组2/组3-stage2 为冻结 base；checkpoint 不经它落盘）。"""
        return self._unet

    @property
    def network(self) -> torch.nn.Module:
        """本组被 RL 训练的网络（glossary「Policy」）：组1 = UNet 全参、
        组2/组3-stage2 = ControlNet。checkpoint 落盘与优化器主体。"""
        return self._network

    @property
    def field(self) -> VelocityField:
        """本组采样场（policy-modeling 章采样场表的组级行）。"""
        return self._field

    @property
    def conditions(self) -> ConditionSampler:
        """本组条件分布（均匀采样的唯一入口）。"""
        return self._conditions

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """可训练网络的 AdamW（参考实现超参：lr 与 weight_decay 显式落位）。"""
        return self._optimizer

    def eval_phase(self) -> None:
        """执行序第 1 相：全部 policy 网络进 eval（rollout 与 π_old 记录
        的数值口径，与更新相的逐位重算一致——GroupNorm 无 batch 统计）。"""
        self._unet.eval()
        self._network.eval()

    def train_phase(self) -> None:
        """执行序第 2 相：可训练网络进 train；冻结 base 恒 eval
        （训练语义只对被优化的网络存在）。"""
        if self._network is not self._unet:
            self._unet.eval()
        self._network.train()
