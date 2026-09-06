"""断点续训状态机（ticket T07，spec #15「训练循环执行序」的定期条目）。

续训状态全清单（spec 钦定）：policy 与判别器各自的权重 + optimizer
state、Replay buffer 两区内容、RNG 状态（torch/CUDA/numpy/python）、
iteration 计数、LR scheduler 状态、（若启用）EMA 权重——目标是跨作业
边界恢复后训练轨迹与指标可复现。

形态：单文件滚动 checkpoint——**per-rank 分片**（单进程/world-1 =
``checkpoints/<前缀>resume_state.pt``；多 rank = ``..._rank{R}.pt``，
原子写 tmp + ``os.replace``）——崩溃恢复只消费各 rank 自己的最新状态，
按周期覆写（周期 = ``schedule.checkpoint_interval``，默认每 10
iteration + 每里程碑强制 + 收尾兜底，与产物 checkpoint 同节奏、由
trainer 消费 config 契约驱动）。per-rank 分片的原因：Replay buffer
（per-rank 两区内容）、六条命名 RNG 流（rank 派生 seed 下各 rank 独立
演化）与 FSDP 优化器状态（分片动量）本就是 rank 本地状态；policy 与
判别器权重在各 rank 间经梯度 allreduce 保持逐位一致，随每个分片冗余
保存（同时是「同步生效」的外部观测面）。

与产物 checkpoint（``policy_iter*.pt`` / ``discriminator_iter*.pt``，
rank 0 独写）的分工：后者是**契约工件**（评测 / milestone / 组3
stage-1 复用消费的可装载有效权重）；本文件是**训练机内部状态**，判定
目标是恢复后逐位续跑——判别器以原始 ``state_dict`` 落盘（spectral
norm 启用时含 power iteration buffer ``_u``/``_v``；有效权重语义的
可装载形式见 ``netbuild.loadable_state_dict``）。

清单各项的落地面：

- LR scheduler 状态：当前实现无独立 scheduler 对象（常数 LR），状态 =
  两 optimizer ``param_groups`` 的 lr——``lr`` 槽位与 optimizer state
  恢复后显式对账；scheduler 对象落地后此处扩展为其 ``state_dict``。
- EMA（条件项）：``ema_anchor_enabled=true`` 属升级项，trainer 装配期
  显式拒绝（静默忽略会让清单缺 EMA 权重），槽位预留、当前恒 ``None``。
- RNG：六条命名 ``torch.Generator`` 流（TrainingRngStreams 注册表）+
  进程全局 torch/CUDA/numpy/python——全部编码为 ``weights_only`` 可安全
  反序列化的原语（张量 / int / float / None）：numpy 的 MT19937 键数组
  转 uint32 张量，python random 状态转 int 列表。

恢复语义：trainer 装配（网络构建、冷启动判别器初始化）完成后
``ResumeStore.restore`` 整体覆写——权重 / optimizer / buffer / 全部 RNG
流 / 全局 RNG 逐一回到落盘时刻，从 ``iteration`` 计数继续；恢复调用方
还须回退指标流（RunArtifacts.``rewind_events``）删除恢复点之后的半截
事件。
"""

import json
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from cynosure.config import ConfigLoader
from cynosure.distributed import DistributedContext

if TYPE_CHECKING:
    from cynosure.config import CynosureConfig
    from cynosure.train.trainer import GranularGrpoTrainer

RESUME_STATE_FILENAME = "resume_state.pt"
"""续训状态文件名（checkpoints 目录内、组3 stage 前缀隔离；多 rank 下
每 rank 追加 ``_rank{R}`` 后缀形成分片文件）。"""

RESUME_GENERATION_FILENAME = "resume_generation.json"
"""续训代际标记（checkpoints 目录内、stage 前缀隔离）：全部 rank 分片
均已持久化到同一 iteration 的提交记录——save 在分片落盘后的 barrier 之
后才由 rank 0 写出。多 rank 恢复必须对齐标记代际：per-rank 分片各自
原子替换，保存中途崩溃可留下混代际现场（部分 rank 已到 N、其余还在
N-1），静默恢复会让各 rank 从不同 iteration 继续训练（集合操作错配、
指标流重复、权重分叉）。world-1 的历史 run 目录可无标记（单分片自身
原子替换已保证一致性），对账跳过。"""

RESUME_STATE_FORMAT_VERSION = 2
"""payload 契约版本：字段集变更时递增，恢复入口按版本拒绝旧文件。
v2：+ world_size（多 rank 续训的拓扑对账）。"""

_REQUIRED_KEYS: tuple[str, ...] = (
    "format_version",
    "iteration",
    "world_size",
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


class ResumeStore:
    """续训状态分片存取（断点续训状态机的落盘/恢复单点，per-rank）。

    落盘：全清单快照（``_capture``）原子写本 rank 分片文件（tmp +
    ``os.replace``）。恢复（``restore``）的校验与应用分两段集合裁决：
    本地前置（payload 契约、拓扑、代际标记）的结果作为报告数据进对账
    collective——装载失败与代际不一致全体一致拒绝；应用段（config
    一致性守卫 → 两模型权重与 optimizer → lr 槽位对账 → buffer 两区 →
    命名 RNG 流 → 全局 RNG）的失败经第二段 collective 全体拒绝——
    装配期随机性（冷启动判别器初始化）被整体覆写，恢复即落盘时刻的
    训练机状态。
    """

    def __init__(
        self, checkpoints_dir: Path, prefix: str, dist: DistributedContext,
    ) -> None:
        self._checkpoints_dir = checkpoints_dir
        self._prefix = prefix
        self._dist = dist

    def shard_path(self) -> Path:
        """本 rank 的续训状态文件路径（checkpoints 目录 + stage 前缀；
        多 rank 追加 rank 后缀形成 per-rank 分片文件）。"""
        name = RESUME_STATE_FILENAME if self._dist.world_size <= 1 else (
            f"resume_state_rank{self._dist.rank}.pt"
        )
        return self._checkpoints_dir / f"{self._prefix}{name}"

    def generation_marker_path(self) -> Path:
        """续训代际标记路径（与分片同前缀隔离；提交点语义见模块常量）。"""
        return self._checkpoints_dir / f"{self._prefix}{RESUME_GENERATION_FILENAME}"

    def save(
        self, trainer: "GranularGrpoTrainer", iteration: int,
        policy_state: dict,
    ) -> None:
        """全清单快照原子落盘（trainer 在周期/里程碑/收尾点调用；每 rank
        写自己的分片文件）。``policy_state`` 是调用方已导出的 policy full
        state（产物 checkpoint 写盘与续训分片共享同一份导出——本方法内部
        不再二次导出，那会让每 rank 在 checkpoint 期同时驻留两份完整 CPU
        权重副本）。落盘后的 barrier 是代际提交的前置：标记（rank 0 写）
        承诺**全部**分片已持久化到同一 iteration。"""
        payload = self._capture(trainer, iteration, policy_state)
        path = self.shard_path()
        tmp = path.with_name(f"{path.name}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # 崩溃下的原子替换：恢复面只见完整旧文件或完整新文件
        self._dist.barrier()
        if self._dist.rank == 0:
            self._write_generation_marker(iteration)

    def _write_generation_marker(self, iteration: int) -> None:
        marker = self.generation_marker_path()
        tmp = marker.with_name(f"{marker.name}.tmp")
        tmp.write_text(json.dumps({
            "iteration": int(iteration),
            "world_size": self._dist.world_size,
        }), encoding="utf-8")
        os.replace(tmp, marker)

    def _read_generation_marker(self) -> int | None:
        """代际标记读取（缺失返回 None：world-1 的历史 run 目录无标记，
        单分片自身原子替换已保证一致性）。"""
        marker = self.generation_marker_path()
        if not marker.is_file():
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            return int(payload["iteration"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"续训代际标记不可读（{marker}）: {exc}"
            ) from exc

    def restore(self, trainer: "GranularGrpoTrainer") -> int:
        """从 run 目录本 rank 的最新续训状态整体恢复，返回恢复点 iteration。

        恢复的失败拒绝全部集合化：本地前置（标记读取、分片装载、
        payload/拓扑校验）的结果作为报告数据进对账 collective——任何
        rank 的本地先抛都会让通过校验的邻居停在 all_gather 永等（拒绝
        方退出、通过方挂死）。第一段裁决装载失败与代际不一致，第二段
        裁决恢复应用失败；两段全绿才回到训练循环。
        """
        try:
            fields, state = self._prepare_local_state()
        except (ValueError, FileNotFoundError) as exc:
            fields, state = {"error": str(exc)}, None
        peers = [
            entry[0] for entry in self._dist.all_gather(
                [{"rank": self._dist.rank, **fields}],
            )
        ]
        failed = [peer for peer in peers if "error" in peer]
        if failed:
            peer = failed[0]
            raise ValueError(
                f"rank {peer['rank']} 续训状态装载失败: {peer['error']}"
                "——任一 rank 分片损坏/缺失都是保存中途崩溃的现场，恢复"
                "入口集体拒绝（部分 rank 单方面恢复会让其余 rank 停在"
                "集合操作）"
            )
        mismatched = [
            peer for peer in peers
            if (peer["iteration"], peer["marker"], peer["world_size"])
            != (fields["iteration"], fields["marker"], fields["world_size"])
        ]
        if mismatched:
            peer = mismatched[0]
            raise ValueError(
                "各 rank 续训分片代际不一致"
                f"（rank {peer['rank']} 分片 {peer['iteration']} / 标记 "
                f"{peer['marker']} vs 本 rank（{self._dist.rank}）分片 "
                f"{fields['iteration']} / 标记 {fields['marker']}）：保存"
                "中途崩溃留下的混代际分片不可恢复（滚动覆写不保留历史"
                "分片），请从更早的完整 checkpoint 现场恢复"
            )
        try:
            iteration = self._apply(trainer, state)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            verdict: dict[str, Any] = {"error": str(exc)}
        else:
            verdict = {"iteration": iteration}
        outcomes = [
            entry[0] for entry in self._dist.all_gather(
                [{"rank": self._dist.rank, **verdict}],
            )
        ]
        failed_apply = [peer for peer in outcomes if "error" in peer]
        if failed_apply:
            peer = failed_apply[0]
            raise ValueError(
                f"rank {peer['rank']} 续训状态回填失败: {peer['error']}"
                "——全体拒绝（部分 rank 带半恢复状态继续训练是权重分叉）"
            )
        return iteration

    def _prepare_local_state(self) -> tuple[dict[str, Any], dict]:
        """restore 的本地前置：代际标记读取、分片装载、payload 契约与
        拓扑校验。返回 (对账报告字段, 落盘 payload)；失败即抛——调用方
        把异常转为 collective 报告数据（任何 rank 的本地拒绝不得先于
        集合对账发生，否则通过校验的邻居停在 all_gather 永等）。"""
        marker_iteration = self._read_generation_marker()
        if marker_iteration is None and self._dist.world_size > 1:
            raise ValueError(
                f"多 rank 续训缺代际标记"
                f"（{self.generation_marker_path().name}）：run 目录不是"
                "完整的多 rank 训练现场（分片与标记由同一 checkpoint 节奏"
                "产出），拒绝猜测各 rank 的共同恢复点"
            )
        path = self.shard_path()
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
        self._validate_payload(state)
        if state["world_size"] != self._dist.world_size:
            raise ValueError(
                f"续训状态 world_size（{state['world_size']}）与当前拓扑"
                f"（{self._dist.world_size}）不符：FSDP 分片与 optimizer 状态是"
                "按 world 切分的，跨拓扑续训不受支持"
            )
        return {
            "iteration": int(state["iteration"]),
            "marker": marker_iteration,
            "world_size": int(state["world_size"]),
        }, state

    def _apply(self, trainer: "GranularGrpoTrainer", state: dict) -> int:
        """代际对齐通过后的恢复应用：config 守卫 → 两模型权重与
        optimizer → lr 槽位对账 → buffer 两区 → 命名 RNG 流 → 全局 RNG
        ——装配期随机性（冷启动判别器初始化）被整体覆写；返回恢复点
        iteration。失败即抛，由调用方的第二段 collective 裁决同步给
        全体（半恢复状态不进训练循环）。"""
        self._assert_resumable_config(
            ConfigLoader.load(trainer.artifacts.paths.config_snapshot),
            trainer.config,
        )
        trainer.policy.load_full_state(state["policy_network"])
        trainer.policy.optimizer.load_state_dict(state["policy_optimizer"])
        trainer.rewards.discriminator.load_state_dict(
            state["discriminator_network"], strict=True,
        )
        trainer.rewards.update.optimizer.load_state_dict(
            state["discriminator_optimizer"],
        )
        self._restore_lr(trainer, state["lr"])
        self._restore_buffer(trainer, state["replay_buffer"])
        self._restore_generators(trainer, state["generators"])
        self._restore_global_rng(state["rng"])
        return state["iteration"]

    def _capture(
        self, trainer: "GranularGrpoTrainer", iteration: int,
        policy_state: dict,
    ) -> dict[str, Any]:
        """续训状态全清单快照（T07 验收清单的落盘形态）。"""
        rewards = trainer.rewards
        base = rewards.buffer.base_samples()
        recent = rewards.buffer.recent_samples()
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "iteration": int(iteration),
            "world_size": trainer.runtime.dist.world_size,
            # full state（裸网络键形）：FSDP 装配下由 PolicySharding 导出，
            # 每 rank 冗余保存全量（同步生效的外部观测面 + 恢复入口简单）；
            # 调用方传入的同一份导出（不在此二次导出）
            "policy_network": policy_state,
            "policy_optimizer": trainer.policy.optimizer.state_dict(),
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
                for name, generator in trainer.rng.named().items()
            },
            "rng": self._capture_global_rng(),
            "lr": {
                "policy": trainer.policy.optimizer.param_groups[0]["lr"],
                "discriminator": rewards.update.optimizer.param_groups[0]["lr"],
            },
            "ema": None,  # 条件项：EMA 锚升级项未交付（trainer 装配期拒绝启用）
        }

    def _assert_resumable_config(
        self, saved: "CynosureConfig", current: "CynosureConfig",
    ) -> None:
        """续训 config 与原 run 快照的一致性守卫（漂移白名单见模块常量）。"""
        if saved == current:
            return
        drift = self._config_drift(saved.model_dump(), current.model_dump())
        unallowed = sorted(
            ".".join(path)
            for path in drift if tuple(path) not in _ALLOWED_CONFIG_DRIFT
        )
        if unallowed:
            raise ValueError(
                "续训 config 与原 run 快照不一致（除 schedule.max_iterations 外"
                f"须逐字段一致，漂移字段会让恢复状态语义失配）: {', '.join(unallowed)}"
            )

    def _config_drift(
        self, left: Any, right: Any, prefix: tuple[str, ...] = (),
    ) -> list[tuple[str, ...]]:
        if isinstance(left, dict) and isinstance(right, dict):
            drift: list[tuple[str, ...]] = []
            for key in sorted(set(left) | set(right)):
                if key not in left or key not in right:
                    drift.append((*prefix, key))
                else:
                    drift.extend(
                        self._config_drift(left[key], right[key], (*prefix, key))
                    )
            return drift
        return [] if left == right else [prefix]

    def _validate_payload(self, state: dict) -> None:
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

    def _restore_lr(self, trainer: "GranularGrpoTrainer", slot: dict) -> None:
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

    def _restore_buffer(
        self, trainer: "GranularGrpoTrainer", saved: dict,
    ) -> None:
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

    def _restore_generators(
        self, trainer: "GranularGrpoTrainer", saved: dict,
    ) -> None:
        streams = trainer.rng.named()
        if set(saved) != set(streams):
            raise ValueError(
                f"续训状态 generator 清单与当前装配不一致: "
                f"{sorted(saved)} vs {sorted(streams)}"
            )
        for name, generator_state in saved.items():
            streams[name].set_state(generator_state)

    def _capture_global_rng(self) -> dict[str, Any]:
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

    def _restore_global_rng(self, saved: dict[str, Any]) -> None:
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
