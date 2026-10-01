"""async 执行序的续训分片存取（#236 续训与事件契约期；#222/#218 结票
口径的落地面）。

**新执行序独立 store + 独立 nominal-v12 常量**（#236）：旧执行序的
``ResumeStore``（train/resume.py，v11、per-rank 分片 + 代际标记 +
两段 all_gather 集合化拒绝）继续服务 torchrun 旧路径至切换期删除，
本模块的 ``AsyncResumeStore`` 只服务 async 门面——「加厚期新 v12
分片被旧 resume 拒、旧 v11 分片被新 store 拒 = #222 版本对账承载
跨执行器拒绝」：双侧常量不同，各自拒绝不等于自己口径的版本，跨
执行器恢复不可静默发生。v11 号位已被 #228 阶段 0（门控链退役）
占用，v-next 账单一次 bump 不拆（#217 明令）→ nominal v12。

与旧执行序分片的形态差异（#222 §1 全口径）：

- **单文件** ``<stage 前缀>resume_state.pt``、主线程直写、tmp +
  ``os.replace`` 原子替换——单进程多卡下 per-rank 分片的存在理由
  （FSDP 优化器分片动量 rank 本地、独立进程无法共写）全部消失；
  写成功即一致、崩溃留旧代际完整可用，**Resume generation marker
  整体退役**（提交记录冗余），**两段 all_gather 集合化拒绝整体
  退役**（单进程内 try/except fail-fast，无 rank 间互等死锁面）。
- **拓扑守卫 = payload 只对账 ``slots``（协程数），卡数不进对账**
  （#218：四条 RNG 流全 CPU generator、权重每卡完整副本逐位一致、
  槽数不变则分配表纯函数重导出不变）——跨卡数（槽数不变）恢复
  放行，代价逐位一致断、生产统计等价口径（ADR-0011）仍成立。
- **删除面**（#218 贡献清单 + #222 汇总）：全局 RNG（``rng`` 键，
  「训练路径禁碰进程全局 RNG」不变式的机器锚在门面测试承载）、
  ``gating``（v11 已删）、``world_size``（→ ``slots``）、``ema``
  预留槽（恒 None）、分配表状态（纯函数重导出，#218 取消）。
- **``generators`` per-(槽×流) 嵌套**（``{"slot{i}": {rollout /
  real_pool / heldout_auc / recon: state}}``，#218）——流清单逐槽
  set 比较，旧执行序的扁平流名清单在此二次拒绝（双保险）。
- **``overfit`` per-condition 嵌套**（ADR-0009-β）：#222 §3 曾按
  当时的 per-rank 记账起草 per-槽嵌套，#234 判别器链期已按
  #220 决议 13 把 rank/槽轴整体退役（分叉 = 池化 per-condition
  单值，每条件一条 EMA）——分片形态随现行监控器按条件嵌套，
  ``OverfitMonitor.adopt`` 形态校验沿组件自持（ADR-0014）。
- **``seeds`` = 派生值记录性字段（不对账）**：per-槽 seed 派生值
  落盘留痕（#217「recon 流 per-槽派生值」）；恢复走状态回填
  （``set_state``）不重派生对账——防未来含卡号的派生公式把跨
  卡数恢复误拒（#222 明文）。形态仍校验（损坏分片显式拒绝），
  值不参与对账。
- **恢复序**（#218 直线）：config 守卫 → 权重/optimizer（门面下发
  各卡）→ lr 槽位对账（门面实测）→ generators 状态回填 → overfit
  adopt → 进循环。

「ResumeStore 挂运行时聚合层」的裁决（#222 §1，ADR-0014 平移）随
#231 骨架期的新执行序装配形态落地为：**门面持有 store，payload 的
攒装（收集各卡权威状态）与恢复应用（下发各卡）由门面编排**——
store 只跨「分片存取 + 输入契约校验」一道 seam，不触碰门面内部
结构（旧执行序 ResumeStore 收 trainer 的穿透形态不镜像）。
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from cynosure.config import ConfigLoader
from cynosure.train.rng import TrainingRngStreams

if TYPE_CHECKING:
    from cynosure.config import CynosureConfig
    from cynosure.train.artifacts import RunArtifacts

ASYNC_RESUME_FORMAT_VERSION = 12
"""async 执行序续训分片的 payload 契约版本（nominal-v12，#222）。

v12 = #218 贡献清单 + #222 汇总的一次 bump（不拆，#217 明令）：
per-(槽×流) generators 嵌套、``rng``（全局 RNG）键删除、``slots``
字段（协程数语义）+ 分配表状态字段取消（#218）；``seeds`` 记录性
字段、单文件形态、marker/集合化拒绝退役、``world_size``→``slots``、
``ema`` 预留槽删除（#222）。v11 号位由 #228 阶段 0（门控链退役，
``gating`` 键删除）占用——版本号以阶段 0 落地时点为准的既有裁决。

跨执行器对账：v12 分片被旧执行序 v11 口径拒（!= 11）、v11 分片被
本 store 拒（!= 12）——版本号是跨执行器拒绝的唯一承载（#222 §0）。"""

RESUME_STATE_FILENAME = "resume_state.pt"
"""续训分片文件名（checkpoints 目录内、stage 前缀隔离与旧执行序同
形态；单文件，无 per-rank 后缀）。"""

_REQUIRED_KEYS: tuple[str, ...] = (
    "version",
    "iteration",
    "slots",
    "policy_network",
    "policy_optimizer",
    "discriminator_network",
    "discriminator_optimizer",
    "lr",
    "generators",
    "overfit",
    "seeds",
)
"""v12 nominal 全键清单（#222 终稿汇总）：多一键少一键都是契约变更、
须走版本 bump。"""

_STREAM_NAMES: frozenset[str] = frozenset(
    {
        TrainingRngStreams.ROLLOUT,
        TrainingRngStreams.REAL_POOL,
        TrainingRngStreams.HELDOUT_AUC,
        TrainingRngStreams.RECON,
    },
)
"""generators 的逐槽流清单（#218 嵌套结构的校验面）：从
``TrainingRngStreams`` 流常量派生的单点清单（executor 侧导出面
同源派生，无字面量副本），键名字符串即契约。"""

_LR_SLOTS: frozenset[str] = frozenset({"policy", "discriminator"})

_ALLOWED_CONFIG_DRIFT: frozenset[tuple[str, ...]] = frozenset(
    {("schedule", "max_iterations")},
)
"""续训 config 的漂移白名单（口径与旧执行序 resume.py 同源）：
max_iterations 是延长/收缩训练规模的正当地址；其余字段漂移会让
恢复的 RNG 流与 optimizer 状态语义失配，一律拒绝。"""


class AsyncResumeStore:
    """async 执行序的续训分片存取（v12 单文件，主线程直写）。

    落盘（``save``）：调用方攒装的全清单 payload 原子写单文件
    （tmp + ``os.replace``——崩溃窗口外只见完整旧文件或完整新文件）。
    恢复（``restore``）：单进程内校验与应用分离——本类做输入契约
    （版本、键清单、slots 拓扑、generators 嵌套、字段形态、config
    漂移），返回校验后的 payload；应用（权重/optimizer 下发、lr 实测
    对账、流状态回填、overfit adopt）由门面编排。任何校验失败即抛，
    无集合化裁决面（单进程 fail-fast，#222 §1）。
    """

    def __init__(self, artifacts: "RunArtifacts") -> None:
        self._paths = artifacts.paths

    @staticmethod
    def to_cpu_snapshot(state: dict[str, Any]) -> dict[str, Any]:
        """state_dict（权重或 optimizer 状态）的深拷贝 CPU 形态——
        分片落盘、产物写盘与跨卡逐位校验的统一取数面：张量叶
        ``detach().cpu().clone()`` 克隆（快照与源存储彻底解耦——
        主线程的序列化与比较不受卡上存储别名影响，checkpoint 点
        之后的卡上计算不会暗中改到已导出快照；CPU 上 ``.cpu()``
        是零拷贝回传，须 ``.clone()`` 兑现深拷贝承诺）；容器
        （dict/list）递归、非张量叶原样保留。"""
        return AsyncResumeStore._to_cpu_value(state)

    @staticmethod
    def _to_cpu_value(value: Any) -> Any:
        """单值递归面（``to_cpu_snapshot`` 的逐叶分派）：张量叶克隆、
        dict/list 递归（**含 list 直含张量**——optimizer state 的
        per-param 张量列表等形态，#236 review 补漏）、其它叶原样。"""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {
                key: AsyncResumeStore._to_cpu_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                AsyncResumeStore._to_cpu_value(item) for item in value
            ]
        return value

    def path(self) -> Path:
        """续训分片路径（checkpoints 目录；单文件无 rank 后缀）。"""
        return self._paths.checkpoints / RESUME_STATE_FILENAME

    def save(self, payload: dict[str, Any]) -> None:
        """全清单快照原子落盘（门面在周期/收尾 checkpoint 点调用；
        ``_validate_payload`` 的镜像前置——攒装即校验，损坏 payload
        不落盘）。"""
        self._validate_payload(payload, slot_count=payload["slots"])
        path = self.path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)  # 崩溃下的原子替换：恢复面只见完整旧/新文件

    def restore(
        self, current: "CynosureConfig", *, slot_count: int,
    ) -> dict[str, Any]:
        """从 run 目录恢复：装载 → 输入契约校验 → config 漂移守卫，
        返回校验后的 payload（``iteration`` = 恢复点）。失败即抛——
        单进程内本地 fail-fast，无跨 rank 对账面（#222 §1）。"""
        path = self.path()
        if not path.is_file():
            raise FileNotFoundError(
                f"无续训状态可恢复（期望 {path}）：周期落盘"
                "（schedule.checkpoint_interval）尚未产出，或 run 目录不是"
                "训练中断现场"
            )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # 损坏/半截文件 → 干净的输入契约错误
            raise ValueError(f"续训状态文件不可读（{path}）: {exc}") from exc
        self._validate_payload(payload, slot_count=slot_count)
        self._assert_resumable_config(
            ConfigLoader.load(self._paths.config_snapshot), current,
        )
        return payload

    def _validate_payload(self, payload: Any, *, slot_count: int) -> None:
        """v12 输入契约：版本 → 键清单 → iteration/slots 形态 →
        lr/seeds 形态 → generators 嵌套 → slots 拓扑对账（协程数）。"""
        if not isinstance(payload, dict):
            raise ValueError(
                f"续训状态形态非法（须为 dict）: {type(payload)}"
            )
        version = payload.get("version")
        if version != ASYNC_RESUME_FORMAT_VERSION:
            raise ValueError(
                f"续训状态格式版本不符：本代码口径 v"
                f"{ASYNC_RESUME_FORMAT_VERSION}（async 执行序 nominal-v12："
                "per-槽嵌套 generators、slots 拓扑对账、全局 RNG/"
                "world_size/ema 预留槽删除），"
                f"得到 {version!r}——跨执行器/跨口径续训不可恢复（旧执行序"
                " v11 分片与 async v12 分片互不承认，#222 版本对账）；"
                "请从产物 checkpoint 重启新 run"
            )
        missing = [key for key in _REQUIRED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"续训状态缺字段: {missing}")
        iteration = payload["iteration"]
        if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
            raise ValueError(f"续训状态 iteration 计数非法: {iteration!r}")
        slots = payload["slots"]
        if not isinstance(slots, int) or isinstance(slots, bool) or slots < 1:
            raise ValueError(f"续训状态 slots 非法（须为正整数协程数）: {slots!r}")
        self._validate_lr(payload["lr"])
        self._validate_seeds(payload["seeds"], slots=slots)
        self._validate_generators(payload["generators"], slots=slots)
        if slots != slot_count:
            raise ValueError(
                f"续训状态协程数 slots（{slots}）与当前调度拓扑"
                f"（{slot_count}）不符：跨拓扑续训不受支持（#218：拓扑"
                "对账对象 = 协程数；卡数不进对账——跨卡数恢复放行）"
            )

    def _validate_generators(self, saved: Any, *, slots: int) -> None:
        """per-(槽×流) 嵌套清单逐槽 set 比较（#218）：旧执行序的扁平
        流名清单（键 = 流名）与槽键缺失/多余都在此显式拒绝——版本号
        之外的二次拒绝面（#222 §1 双保险）。"""
        if not isinstance(saved, dict):
            raise ValueError(
                f"续训状态 generators 形态非法（须为 per-槽嵌套 dict）: "
                f"{type(saved)}"
            )
        expected_slots = {f"slot{i}" for i in range(slots)}
        if set(saved) != expected_slots:
            raise ValueError(
                f"续训状态 generators 槽清单与协程数不符: "
                f"{sorted(saved)} vs 期望 {sorted(expected_slots)}"
                "（async 执行序的流状态按槽嵌套——旧执行序的扁平流名"
                "清单不被承认，#218 嵌套结构）"
            )
        for slot_key in sorted(expected_slots):
            streams = saved[slot_key]
            if not isinstance(streams, dict) or set(streams) != _STREAM_NAMES:
                raise ValueError(
                    f"续训状态 generators[{slot_key}] 流清单失配: "
                    f"{sorted(streams) if isinstance(streams, dict) else streams}"
                    f" vs 期望 {sorted(_STREAM_NAMES)}"
                )

    def _validate_lr(self, lr_slots: Any) -> None:
        if not isinstance(lr_slots, dict) or set(lr_slots) != _LR_SLOTS:
            raise ValueError(
                f"续训状态 lr 槽位字段不符: "
                f"{sorted(lr_slots) if isinstance(lr_slots, dict) else lr_slots}"
            )
        for name, value in lr_slots.items():
            if isinstance(value, bool):
                raise ValueError(f"续训状态 lr[{name}] 形态非法: {value!r}")
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"续训状态 lr[{name}] 形态非法（须为数值）: {value!r}"
                ) from exc

    def _validate_seeds(self, seeds: Any, *, slots: int) -> None:
        """派生值记录性字段的形态校验（值不参与对账，#222 §1：恢复走
        状态回填、不重派生比较——防未来含卡号的派生公式误拒跨卡数
        恢复）。"""
        if (
            not isinstance(seeds, dict)
            or set(seeds) != {"base", "per_slot"}
            or not isinstance(seeds["base"], int)
            or isinstance(seeds["base"], bool)
            or not isinstance(seeds["per_slot"], list)
            or len(seeds["per_slot"]) != slots
            or any(
                not isinstance(v, int) or isinstance(v, bool)
                for v in seeds["per_slot"]
            )
        ):
            raise ValueError(
                f"续训状态 seeds 形态非法（须为 {{base: int, per_slot: "
                f"[int × slots]}}）: {seeds!r}"
            )

    def _assert_resumable_config(
        self, saved: "CynosureConfig", current: "CynosureConfig",
    ) -> None:
        """续训 config 与原 run 快照的一致性守卫（漂移白名单见模块
        常量；口径与旧执行序 resume.py 同源——非白名单字段漂移会让
        恢复的 RNG 流与 optimizer 状态语义失配）。"""
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
                f"须逐字段一致，漂移字段会让恢复状态语义失配）: "
                f"{', '.join(unallowed)}"
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
