"""policy 网络的 FSDP full-shard + 梯度检查点封装（ADR-0003：分片定死 +
fallback 链；orchestration 章「全部同构 FSDP rank」）。

单 FSDP unit 包整个可训练网络（参数 + 梯度 + 优化器状态全分片 =
FULL_SHARD 语义）；梯度检查点经 torch 官方 ``checkpoint_wrapper`` 在
FSDP 包装**之前**应用于 MONAI 的 resnet 块（DiffusionUNetResnetBlock，
fixture 与生产 UNet/ControlNet 的前向主干）——MONAI 1.6 的 UNet 无
原生 use_checkpointing 开关（ControlNet 有、不可依赖单侧），统一走
torch 的模块包装语义，网络类与权重零改动。

数值口径：use_orig_params=True 使 FULL state dict 的键与裸网络一致
（产物 checkpoint 契约：可装载裸网络）；device_id 显式钉本 rank 设备
（CPU fixture 必需——默认设备探测会被 MPS 等加速器劫持）；
sync_module_states=False 配合装配期同 seed/同 checkpoint 的跨 rank
逐位一致初始权重（CPU FSDP 的 sync 限制，语义等价——初始一致时
sync 是恒等操作）。

单进程（无 torchrun 环境）整路恒等：不包装、不检查点——FSDP 是
torchrun 多进程拓扑的配套（config.sharding 仅在分布式装配下生效）。
"""

import torch
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    ShardingStrategy,
    StateDictType,
)

from cynosure.config import CynosureConfig
from cynosure.distributed.process import DistributedContext

_FULL_STATE_CONFIG = FullStateDictConfig(
    offload_to_cpu=True, rank0_only=False,
)
"""full state dict 导出口径：每 rank 都可得完整权重（per-rank 续训状态
的保存形态）、张量回落 CPU（checkpoint 落盘契约的存储形态）。"""


class PolicySharding:
    """可训练网络的 FSDP full-shard 封装（wrap + state dict 出入口）。"""

    def __init__(
        self, context: DistributedContext, gradient_checkpointing: bool,
    ) -> None:
        self._context = context
        self._gradient_checkpointing = gradient_checkpointing

    @classmethod
    def from_config(
        cls, context: DistributedContext, config: CynosureConfig,
    ) -> "PolicySharding":
        """按 config 装配分片策略；未交付的降级链显式拒绝。

        降级链 DDP/ZeRO-3（orchestration 章）是 fallback 非默认：config
        选了未交付策略时装配期拒绝，而非静默退化成另一条语义。
        单进程装配读定 checkpointing 配置但不生效（恒等路径）。
        """
        if not context.distributed:
            return cls(context, gradient_checkpointing=False)
        if config.sharding.strategy != "fsdp":
            raise ValueError(
                f"分片策略 {config.sharding.strategy} 属降级链 fallback"
                "（ADR-0003，非默认路径），未交付：当前仅 fsdp"
            )
        return cls(
            context,
            gradient_checkpointing=config.sharding.gradient_checkpointing,
        )

    def wrap(self, network: torch.nn.Module) -> torch.nn.Module:
        """可训练网络 → FSDP full-shard（单进程恒等；梯度检查点先于包装应用）。"""
        if not self._context.distributed:
            return network
        if self._gradient_checkpointing:
            apply_activation_checkpointing(
                network,
                checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
                    module, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=self._is_resnet_block,
            )
        return FullyShardedDataParallel(
            network,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            sync_module_states=False,
            device_id=self._context.local_device(),
        )

    @staticmethod
    def _is_resnet_block(module: torch.nn.Module) -> bool:
        """梯度检查点的重算锚点：MONAI resnet 块（UNet/ControlNet 共用族）。"""
        return isinstance(module, DiffusionUNetResnetBlock)

    @staticmethod
    def full_state_dict(network: torch.nn.Module) -> dict:
        """网络权重的 full state dict（裸网络键形；单进程直取）。"""
        if not isinstance(network, FullyShardedDataParallel):
            return {key: value.detach().cpu() for key, value in network.state_dict().items()}
        with FullyShardedDataParallel.state_dict_type(
            network, StateDictType.FULL_STATE_DICT, _FULL_STATE_CONFIG,
        ):
            return network.state_dict()

    @staticmethod
    def load_full_state_dict(
        network: torch.nn.Module, state: dict,
    ) -> None:
        """full state dict 装载（FSDP 内部完成分片分发；单进程严格装载）。"""
        if not isinstance(network, FullyShardedDataParallel):
            network.load_state_dict(state, strict=True)
            return
        with FullyShardedDataParallel.state_dict_type(
            network, StateDictType.FULL_STATE_DICT, _FULL_STATE_CONFIG,
        ):
            network.load_state_dict(state)
