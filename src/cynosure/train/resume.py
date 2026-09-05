"""断点续训状态机（ticket T07，spec #15「训练循环执行序」的定期条目）。

续训状态全清单（spec 钦定）：policy 与判别器各自的权重 + optimizer
state、Replay buffer 两区内容、RNG 状态（torch/CUDA/numpy/python）、
iteration 计数、LR scheduler 状态、（若启用）EMA 权重——目标是跨作业
边界恢复后训练轨迹与指标可复现。

形态：单文件滚动 checkpoint（``checkpoints/<前缀>resume_state.pt``，原子
写 tmp + ``os.replace``）——崩溃恢复只消费最新状态，按周期覆写（周期 =
``schedule.checkpoint_interval``，默认每 10 iteration + 每里程碑强制 +
收尾兜底，与产物 checkpoint 同节奏、由 trainer 消费 config 契约驱动）。

与产物 checkpoint（``policy_iter*.pt`` / ``discriminator_iter*.pt``）的
分工：后者是**契约工件**（评测 / milestone / 组3 stage-1 复用消费的可
装载有效权重）；本文件是**训练机内部状态**，判定目标是恢复后逐位续跑
——判别器以原始 ``state_dict`` 落盘（spectral norm 启用时含 power
iteration buffer ``_u``/``_v``；有效权重语义的可装载形式见
``netbuild.loadable_state_dict``）。

清单各项的落地面：

- LR scheduler 状态：当前实现无独立 scheduler 对象（常数 LR），状态 =
  两 optimizer ``param_groups`` 的 lr——``lr`` 槽位与 optimizer state
  恢复后显式对账；scheduler 对象落地后此处扩展为其 ``state_dict``。
- EMA（条件项）：``ema_anchor_enabled=true`` 属升级项，trainer 装配期
  显式拒绝（静默忽略会让清单缺 EMA 权重），槽位预留、当前恒 ``None``。
- RNG：六条命名 ``torch.Generator`` 流（trainer.generators 注册表）+
  进程全局 torch/CUDA/numpy/python——全部编码为 ``weights_only`` 可安全
  反序列化的原语（张量 / int / float / None）：numpy 的 MT19937 键数组
  转 uint32 张量，python random 状态转 int 列表。

恢复语义：trainer 装配（网络构建、冷启动判别器初始化）完成后整体覆写
——权重 / optimizer / buffer / 全部 RNG 流 / 全局 RNG 逐一回到落盘时刻，
从 ``iteration`` 计数继续；恢复调用方还须回退指标流（RunArtifacts.
``rewind_events``）删除恢复点之后的半截事件。
"""

import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cynosure.config import ConfigLoader

if TYPE_CHECKING:
    from cynosure.config import CynosureConfig
    from cynosure.train.trainer import GranularGrpoTrainer

RESUME_STATE_FILENAME = "resume_state.pt"
"""续训状态文件名（checkpoints 目录内、组3 stage 前缀隔离）。"""

RESUME_STATE_FORMAT_VERSION = 1
"""payload 契约版本：字段集变更时递增，恢复入口按版本拒绝旧文件。"""

_REQUIRED_KEYS: tuple[str, ...] = (
    "format_version",
    "iteration",
    "policy_network",
    "policy_optimizer",
    "discriminator_network",
    "discriminator_optimizer",
    "replay_buffer",
    "generators",
    "rng",
    "lr",
    "ema",
)

_ALLOWED_CONFIG_DRIFT: frozenset[tuple[str, ...]] = frozenset(
    {("schedule", "max_iterations")},
)
"""续训 config 的白名单漂移字段：max_iterations 是延长/收缩训练规模的
正当地址（跨作业边界续跑的动机本身）；其余字段漂移会让恢复的 RNG 流、
buffer 内容与 optimizer 状态语义失配，一律拒绝。"""


def resume_state_path(
    checkpoints_dir: Path, prefix: str = "",
) -> Path:
    """续训状态文件路径（checkpoints 目录 + stage 前缀）。"""
    return checkpoints_dir / f"{prefix}{RESUME_STATE_FILENAME}"


def save_resume_state(trainer: "GranularGrpoTrainer", iteration: int) -> None:
    """全清单快照原子落盘（trainer 在周期/里程碑/收尾点调用）。"""
    payload = capture_state(trainer, iteration)
    path = resume_state_path(
        trainer.artifacts.paths.checkpoints,
        trainer.stage_tag.checkpoint_prefix,
    )
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 崩溃下的原子替换：恢复面只见完整旧文件或完整新文件


def resume_latest(trainer: "GranularGrpoTrainer") -> int:
    """从 run 目录最新续训状态整体恢复，返回恢复点 iteration。

    先对账 config（除白名单漂移字段外逐字段一致），再按序恢复：两模型
    权重与 optimizer → lr 槽位对账 → buffer 两区 → 命名 RNG 流 → 全局
    RNG。装配期随机性（冷启动判别器初始化）被整体覆写，恢复即落盘时刻
    的训练机状态。
    """
    path = resume_state_path(
        trainer.artifacts.paths.checkpoints,
        trainer.stage_tag.checkpoint_prefix,
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"无续训状态可恢复（期望 {path}）：周期落盘"
            "（schedule.checkpoint_interval）尚未产出，或 run 目录不是"
            "训练中断现场"
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # 损坏/半截文件 → 干净的输入契约错误（非裸 traceback）
        raise ValueError(f"续训状态文件不可读（{path}）: {exc}") from exc
    _validate_payload(state)
    assert_resumable_config(
        ConfigLoader.load(trainer.artifacts.paths.config_snapshot),
        trainer.config,
    )
    trainer.policy.network.load_state_dict(state["policy_network"], strict=True)
    trainer.policy.optimizer.load_state_dict(state["policy_optimizer"])
    trainer.rewards.discriminator.load_state_dict(
        state["discriminator_network"], strict=True,
    )
    trainer.rewards.update.optimizer.load_state_dict(
        state["discriminator_optimizer"],
    )
    _restore_lr(trainer, state["lr"])
    _restore_buffer(trainer, state["replay_buffer"])
    _restore_generators(trainer, state["generators"])
    _restore_global_rng(state["rng"])
    return state["iteration"]


def capture_state(trainer: "GranularGrpoTrainer", iteration: int) -> dict[str, Any]:
    """续训状态全清单快照（T07 验收清单的落盘形态）。"""
    policy = trainer.policy
    rewards = trainer.rewards
    base = rewards.buffer.base_samples()
    recent = rewards.buffer.recent_samples()
    return {
        "format_version": RESUME_STATE_FORMAT_VERSION,
        "iteration": int(iteration),
        "policy_network": policy.network.state_dict(),
        "policy_optimizer": policy.optimizer.state_dict(),
        # 判别器原始 state_dict（非 loadable 有效权重形式）：训练态续跑
        # 要求 spectral norm 的 power iteration buffer 逐位回归
        "discriminator_network": rewards.discriminator.state_dict(),
        "discriminator_optimizer": rewards.update.optimizer.state_dict(),
        "replay_buffer": {
            "base": torch.stack(base) if base else None,
            "recent": torch.stack(recent) if recent else None,
        },
        "generators": {
            name: generator.get_state()
            for name, generator in trainer.generators.items()
        },
        "rng": _global_rng_state(),
        "lr": {
            "policy": policy.optimizer.param_groups[0]["lr"],
            "discriminator": rewards.update.optimizer.param_groups[0]["lr"],
        },
        "ema": None,  # 条件项：EMA 锚升级项未交付（trainer 装配期拒绝启用）
    }


def assert_resumable_config(
    saved: "CynosureConfig", current: "CynosureConfig",
) -> None:
    """续训 config 与原 run 快照的一致性守卫（漂移白名单见模块常量）。"""
    if saved == current:
        return
    drift = _config_drift(saved.model_dump(), current.model_dump())
    unallowed = sorted(
        ".".join(path) for path in drift if tuple(path) not in _ALLOWED_CONFIG_DRIFT
    )
    if unallowed:
        raise ValueError(
            "续训 config 与原 run 快照不一致（除 schedule.max_iterations 外"
            f"须逐字段一致，漂移字段会让恢复状态语义失配）: {', '.join(unallowed)}"
        )


def _config_drift(
    left: Any, right: Any, prefix: tuple[str, ...] = (),
) -> list[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        drift: list[tuple[str, ...]] = []
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                drift.append((*prefix, key))
            else:
                drift.extend(_config_drift(left[key], right[key], (*prefix, key)))
        return drift
    return [] if left == right else [prefix]


def _validate_payload(state: dict) -> None:
    version = state.get("format_version")
    if version != RESUME_STATE_FORMAT_VERSION:
        raise ValueError(
            f"续训状态契约版本不符：期望 {RESUME_STATE_FORMAT_VERSION}，"
            f"得到 {version}"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in state]
    if missing:
        raise ValueError(f"续训状态缺字段: {missing}")
    iteration = state["iteration"]
    if not isinstance(iteration, int) or iteration < 0:
        raise ValueError(f"续训状态 iteration 计数非法: {iteration!r}")


def _restore_lr(trainer: "GranularGrpoTrainer", slot: dict) -> None:
    """LR scheduler 状态对账：常数 LR 实现的 scheduler 状态 = 两 optimizer
    ``param_groups`` 的 lr（load_state_dict 已随 param_groups 回归）——
    槽位与其实测值显式对账，不一致 = 文件损坏/篡改；scheduler 对象落地
    后此处扩展为其 state_dict 装载。"""
    optimizers = {
        "policy": trainer.policy.optimizer,
        "discriminator": trainer.rewards.update.optimizer,
    }
    if set(slot) != set(optimizers):
        raise ValueError(f"续训状态 lr 槽位字段不符: {sorted(slot)}")
    for name, optimizer in optimizers.items():
        saved_lr = float(slot[name])
        for group in optimizer.param_groups:
            if group["lr"] != saved_lr:
                raise ValueError(
                    f"续训状态 lr 槽位与 optimizer state 不一致（{name}: "
                    f"{saved_lr} vs {group['lr']}）"
                )


def _restore_buffer(trainer: "GranularGrpoTrainer", saved: dict) -> None:
    """buffer 两区内容恢复：base 按固定容量严格对账后整体回填，recent
    按 FIFO 插入序重放；恢复后两区占用必须与落盘一致（容量漂移在显式
    错误处暴露，不静默截断）。"""
    buffer = trainer.rewards.buffer
    base = saved["base"]
    recent = saved["recent"]
    expected_shape = tuple(trainer.config.latent_shape)
    if (
        base is None
        or base.shape[0] != buffer.base_capacity
        or tuple(base.shape[1:]) != expected_shape
    ):
        raise ValueError(
            f"续训状态 base 分区（{None if base is None else tuple(base.shape)}）"
            f"与 buffer 容量 {buffer.base_capacity} × latent {expected_shape} 不符"
        )
    buffer.fill_base(base.to(trainer.device))
    if recent is not None:
        if tuple(recent.shape[1:]) != expected_shape:
            raise ValueError(
                f"续训状态 recent 分区形状 {tuple(recent.shape)} 与 latent "
                f"{expected_shape} 不符"
            )
        buffer.push(recent.to(trainer.device))
    sizes = buffer.zone_sizes()
    if (sizes.base, sizes.recent) != (
        buffer.base_capacity,
        0 if recent is None else recent.shape[0],
    ):
        raise ValueError("续训状态 buffer 恢复后两区占用与落盘不一致")


def _restore_generators(trainer: "GranularGrpoTrainer", saved: dict) -> None:
    if set(saved) != set(trainer.generators):
        raise ValueError(
            f"续训状态 generator 清单与当前装配不一致: "
            f"{sorted(saved)} vs {sorted(trainer.generators)}"
        )
    for name, generator_state in saved.items():
        trainer.generators[name].set_state(generator_state)


def _global_rng_state() -> dict[str, Any]:
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    version, state, gauss = random.getstate()
    return {
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else None
        ),
        "numpy": {
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.uint32)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
        "python": {"version": version, "state": list(state), "gauss": gauss},
    }


def _restore_global_rng(saved: dict[str, Any]) -> None:
    torch.set_rng_state(saved["torch"])
    cuda = saved["cuda"]
    cuda_available = torch.cuda.is_available()
    if (cuda is not None) != cuda_available:
        # CUDA 可用性在落盘与恢复两侧不一致（跨设备续训）：静默丢弃
        # CUDA RNG = 恢复后轨迹静默漂移，显式拒绝
        raise ValueError(
            "续训状态的 CUDA RNG 侧与当前环境不一致"
            f"（落盘 {'含' if cuda is not None else '不含'} CUDA 状态，"
            f"当前 {'有' if cuda_available else '无'} CUDA）："
            "跨设备续训不受支持"
        )
    if cuda is not None:
        torch.cuda.set_rng_state_all(cuda)
    numpy_state = saved["numpy"]
    np.random.set_state((
        "MT19937",
        numpy_state["keys"].numpy(),
        int(numpy_state["pos"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached"]),
    ))
    python_state = saved["python"]
    random.setstate((
        int(python_state["version"]),
        tuple(int(value) for value in python_state["state"]),
        python_state["gauss"],
    ))
