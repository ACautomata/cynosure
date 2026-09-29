"""梯度检查点装配策略（#217 §4「最重配套项」解耦落地，#233）。

检查点原本挂 ``PolicySharding`` 且仅分布式路径生效、单进程恒不检查点——
FSDP 退役后（async 执行模型 per-card 完整副本）激活全量驻留是 48GB 卡
真实 OOM 风险，解耦为独立必选路径：应用缝移至 ``GroupPolicy.build``
（FSDP wrap **之前**、对裸网络原地包装 MONAI resnet 块），经
``config.policy.gradient_checkpointing`` 控制（生产定死 true、
``fixture_mode=true`` 可关，spec 补钉纪律）。组2（cross-modal）注记：
包装只覆盖可训练 ControlNet，梯度回传路径上的冻结 base UNet 激活恒
驻留、不受本开关保护——组2 的显存账须将 UNet 侧另行计入。

**装配期 bitwise 一致性探针（fail-fast）**：``checkpoint_wrapper`` 的
数值透明性（重算前向与原前向逐位一致）不做声明假设——装配期以真实
梯度前向对照校验：plain 腿（裸网络）与 wrapped 腿（包装后）各跑一次
``field.group_velocity`` 前向 + 反向，梯度与输出逐位比较，失配即
``ValueError``（静默数值漂移在首个消费点暴露，而不是训练中期以难归因
的精度损失出现）；「探针不改网络状态」同为受守卫不变式——双根 buffer
键集与数值前后对账（新键/漂移即拒绝）。探针输入合成（latent 形状随
dataset 派生、条件按组
构造）、显式局部 generator（零全局 RNG 消耗，「训练路径禁碰全局 RNG」
不变式）、fp32 非混精度口径（重算路径与前向函数级透明性在 fp32 下
同样成立且确定性最强；生产更新相的 bf16 autocast 面不在本探针内，
由 on/off 端到端锚——经 updater 全程 autocast 的双档逐位比较——
承担覆盖，tests/test_checkpointing.py）。

no_grad 语义：``CheckpointImpl.NO_REENTRANT`` 包装在 grad 关闭的前向
（rollout 相、预训练测量批）下直通执行内层模块——检查点只作用于更新
相的梯度前向，rollout 数值路径零扰动。
"""

import torch
from monai.networks.nets.diffusion_model_unet import DiffusionUNetResnetBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from cynosure.conditions import CONDITION_SPACING_X1E2, ConditionVocabulary
from cynosure.config import CynosureConfig
from cynosure.policy.condition import RolloutCondition
from cynosure.policy.field import (
    BareConditionField,
    CfgCombinedField,
    VelocityField,
)


class GradientCheckpointing:
    """policy 网络的梯度检查点装配（resnet 块 activation 重算包装 +
    装配期 bitwise 一致性探针）；config 关闭时整体 no-op。"""

    PROBE_TIMESTEP: int = 1
    """探针前向的 timestep 值（plain/wrapped 两腿一致即可，embedding 对
    任意正整数连续）。"""

    PROBE_GROUP_SIZE: int = 2
    """探针 ``group_velocity`` 的组宽（更新相真实入口的 batch=1 复用形态）。"""

    PROBE_SEED: int = 0
    """探针显式局部 generator 的 seed（装配期固定；不触碰全局 RNG）。"""

    def __init__(
        self,
        config: CynosureConfig,
        device: torch.device,
        vocabulary: ConditionVocabulary | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._vocabulary = vocabulary
        self._generator = torch.Generator().manual_seed(self.PROBE_SEED)

    @property
    def _is_cross_modal(self) -> bool:
        """组别判定单点（组2 = cross-modal：条件输入与采样场均分叉）。"""
        return self._config.experiment.group == "cross-modal"

    def verify_and_apply(
        self, unet: torch.nn.Module, network: torch.nn.Module,
    ) -> None:
        """装配期单点入口：开启时 plain/wrapped 双腿逐位校验后包装 resnet
        块（``GroupPolicy.build`` 在 FSDP wrap 之前调用——包装作用于裸
        网络，根模块身份不变）；关闭时 no-op（fixture 双档测试面）。
        组1 分支两参为同一对象（unet 即可训练网络）——buffer 快照同对象
        双记，键带根序、语义无害。"""
        if not self._config.policy.gradient_checkpointing:
            return
        probe_input = self._probe_input()
        field = self._probe_field(unet, network)
        # 推理口径探针（与执行序相位显式切换对齐）：双根 eval——生产
        # base 恒 eval；可训练面经 DropoutGuard 无 dropout，eval 不改
        # 数值（GroupNorm 族无 batch 统计）。排掉上游工件残留 dropout
        # 的两腿全局 RNG 消耗错位（那会让探针误伤且归因失真）。
        training_before = (unet.training, network.training)
        unet.eval()
        network.eval()
        buffers = self._buffer_snapshot(unet, network)
        plain_grads, plain_output = self._forward_backward(field, probe_input, network)
        self._apply(network)
        wrapped_grads, wrapped_output = self._forward_backward(
            field, probe_input, network,
        )
        self._assert_bitwise(plain_grads, wrapped_grads, plain_output, wrapped_output)
        network.zero_grad(set_to_none=True)
        self._assert_buffers_untouched(buffers, unet, network)
        # training 标志同属「网络状态」：探针后恢复进位（下游相位切换
        # 本会显式覆盖，恢复使「探针零状态扰动」彻底闭环）
        unet.train(training_before[0])
        network.train(training_before[1])

    def _apply(self, network: torch.nn.Module) -> None:
        """resnet 块的 ``checkpoint_wrapper(NO_REENTRANT)`` 原地包装。"""
        apply_activation_checkpointing(
            network,
            checkpoint_wrapper_fn=lambda module: checkpoint_wrapper(
                module, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=self._is_resnet_block,
        )

    def _probe_field(
        self, unet: torch.nn.Module, network: torch.nn.Module,
    ) -> CfgCombinedField | BareConditionField:
        """探针采样场（按组构造，与生产 field 同类同参、引用裸网络——
        包装是子模块原地替换，同一 field 实例跨双腿复用）。形状面 =
        单点抽检：MR-RATE 取词汇表首条件形（异形条件不逐一遍历——
        装配期单点口径，重算透明性按前向函数级断言，不随形状枚举）。"""
        if self._is_cross_modal:
            return BareConditionField(
                unet, network, self._config.policy.source_latent_scale_factor,
            )
        return CfgCombinedField(network)

    def _probe_input(self) -> tuple[torch.Tensor, RolloutCondition]:
        """合成探针输入（x + 条件）：latent 形状随 dataset 派生（BraTS =
        config 锚形、MR-RATE = 词汇表首条件形），条件按组齐位（组2 带
        源影像 latent 与源 label）。全部张量经同一局部 generator 生成。"""
        latent_shape = self._probe_latent_shape()
        x = torch.randn(
            (1, *latent_shape), generator=self._generator, dtype=torch.float32,
        ).to(self._device)
        label = torch.zeros(1, dtype=torch.int64).to(self._device)
        spacing = torch.tensor(
            CONDITION_SPACING_X1E2, dtype=torch.float32,
        ).unsqueeze(0).to(self._device)
        if self._is_cross_modal:
            source_latent = torch.randn(
                (1, *latent_shape), generator=self._generator,
                dtype=torch.float32,
            ).to(self._device)
            source_label = torch.zeros(1, dtype=torch.int64).to(self._device)
            condition = RolloutCondition(
                label, spacing, source_latent, source_label,
            )
        else:
            condition = RolloutCondition(label, spacing)
        return x, condition

    def _probe_latent_shape(self) -> tuple[int, int, int, int]:
        if self._config.experiment.dataset != "MR-RATE":
            return tuple(self._config.latent_shape)
        # vocabulary 由 GroupPolicy.build 的 MR 分支在场（与本研究分支
        # 同键：dataset == "MR-RATE" 才走 ConditionVocabulary.assemble）
        return self._vocabulary.latent_shape(self._vocabulary.names()[0])

    def _forward_backward(
        self,
        field: VelocityField,
        probe_input: tuple[torch.Tensor, RolloutCondition],
        network: torch.nn.Module,
    ) -> tuple[list[torch.Tensor | None], torch.Tensor]:
        """真实更新相入口的一次前向 + 反向（``group_velocity`` batch=1
        G²RPO 复用形态——与 ``StepwisePolicyUpdate`` 同入口路径，fp32
        子集口径：生产更新相的 bf16 autocast 面不在本探针，见模块
        docstring）；梯度比较面 = 可训练网络参数（组2 的冻结 base UNet
        不在其列——两侧恒无梯度）。返回逐参数梯度快照与输出快照。"""
        x, condition = probe_input
        network.zero_grad(set_to_none=True)
        velocity = field.group_velocity(
            x, self.PROBE_TIMESTEP, condition, self.PROBE_GROUP_SIZE,
        )
        velocity.sum().backward()
        grads = [
            parameter.grad.detach().clone()
            if parameter.grad is not None else None
            for parameter in network.parameters()
        ]
        return grads, velocity.detach().clone()

    def _assert_bitwise(
        self,
        plain_grads: list[torch.Tensor | None],
        wrapped_grads: list[torch.Tensor | None],
        plain_output: torch.Tensor,
        wrapped_output: torch.Tensor,
    ) -> None:
        """梯度 + 输出的逐位比较（``torch.equal``）；任一失配即 fail-fast
        （带参数名与偏差量的诊断面）。``None`` 两侧须同齐同缺。"""
        if not torch.equal(plain_output, wrapped_output):
            deviation = (plain_output - wrapped_output).abs().max().item()
            raise ValueError(
                "梯度检查点 bitwise 校验失败：wrapped 腿输出与 plain 腿"
                f"不逐位一致（最大偏差 {deviation}）——activation 重算"
                "路径数值透明性被破坏，装配期拒绝"
            )
        for index, (plain, wrapped) in enumerate(
            zip(plain_grads, wrapped_grads),
        ):
            if (plain is None) != (wrapped is None):
                raise ValueError(
                    f"梯度检查点 bitwise 校验失败：第 {index} 个参数梯度"
                    f"在场性失配（plain={plain is not None}，"
                    f"wrapped={wrapped is not None}）"
                )
            if plain is not None and not torch.equal(plain, wrapped):
                deviation = (plain - wrapped).abs().max().item()
                raise ValueError(
                    "梯度检查点 bitwise 校验失败：第 "
                    f"{index} 个参数梯度不逐位一致（最大偏差 {deviation}）"
                    "——activation 重算路径数值透明性被破坏，装配期拒绝"
                )

    @staticmethod
    def _is_resnet_block(module: torch.nn.Module) -> bool:
        """重算锚点：MONAI resnet 块（UNet/ControlNet 共用族，fixture 与
        生产网络的前向主干）。"""
        return isinstance(module, DiffusionUNetResnetBlock)

    def _buffer_snapshot(
        self, *roots: torch.nn.Module,
    ) -> dict[str, torch.Tensor]:
        """双根 buffer 快照（探针前取、探针后对账；键带根序防同名碰撞）。"""
        return {
            f"{root_index}:{name}": buffer.detach().clone()
            for root_index, root in enumerate(roots)
            for name, buffer in root.named_buffers()
        }

    def _assert_buffers_untouched(
        self, before: dict[str, torch.Tensor], *roots: torch.nn.Module,
    ) -> None:
        """探针零状态扰动守卫（buffer 面）：前向/反向不得新增、删除或
        推进任何 buffer——「探针不改网络状态」是受守卫不变式而非外部
        事实（eval 口径挡住 BN 类 train-mode 推进，本守卫兜 eval 下仍
        写 buffer 的非标准模块）。参数面由 bitwise 比较覆盖（grads 经
        zero_grad 清场）；training 标志面由 verify_and_apply 进位恢复。"""
        after = self._buffer_snapshot(*roots)
        if set(before) != set(after):
            raise ValueError(
                "梯度检查点探针状态守卫失败：buffer 键集漂移（新增 "
                f"{sorted(set(after) - set(before))}，消失 "
                f"{sorted(set(before) - set(after))}）——网络状态被装配"
                "探针扰动，装配期拒绝"
            )
        for name, snapshot in before.items():
            if not torch.equal(snapshot, after[name]):
                raise ValueError(
                    f"梯度检查点探针状态守卫失败：buffer {name} 被探针"
                    "前向/反向推进——「探针不改网络状态」不变式被破坏，"
                    "装配期拒绝"
                )
