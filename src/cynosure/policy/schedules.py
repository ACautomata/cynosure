"""sigma 日程表（#129「sigma 日程逐条件锚」的装配面）。

ADR-0002 的数值锚语义逐条件化（不变式：每条件锚 = 该条件空间 numel）：
MONAI ``set_timesteps(use_timestep_transform=True)`` 的 SD3 式 timestep
transform 以 numel 为输入——锚错则 sigma 日程静默错位。两域形态：

- ``SingleConditionSchedules``：BraTS 单域特例（单条件词汇），任意
  条件名共用一份全局锚日程（现状 BraTS 行为的等价形态）；
- ``PerConditionSchedules``：MR-RATE 逐条件形态，锚 =
  ``ConditionVocabulary.latent_numel(name)``（词汇表单一派生入口，
  与 rollout 噪声形状结构性同源），按条件名惰性构建并缓存
  ``TrajectoryCursor``（numel 碰撞的条件天然共享日程——transform 的
  输入只有 numel）。

未知条件名/缺名显式拒绝：按名选日程不静默回退——全卷积 UNet 下形状
错位不抛错、静默产出错误日程的采样，本拒绝是最后一道运行时闸口。
"""

from typing import Protocol

from cynosure.conditions import ConditionVocabulary
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.cursor import TrajectoryCursor


class ConditionSchedules(Protocol):
    """条件名 → 轨迹日程的解析协议（RolloutSampler 的日程装配缝）。"""

    def cursor(self, name: str | None) -> TrajectoryCursor:
        """该条件的轨迹游标（自持 timesteps 快照）；语义依实现：
        单域任意名同一游标，逐条件缺名/未知名显式拒绝。"""
        ...


class SingleConditionSchedules:
    """单条件日程表（BraTS 单域特例）：全局锚（ADR-0002
    ``input_img_size_numel``）驱动唯一一份日程，任意条件名（含缺名）
    共用——单域语义 = 单条件词汇特例的等价形态，现状 BraTS 数值的
    零漂移回归锚。"""

    def __init__(
        self, num_inference_steps: int, input_img_size_numel: int,
    ) -> None:
        self._num_inference_steps = num_inference_steps
        self._input_img_size_numel = input_img_size_numel
        self._cursor: TrajectoryCursor | None = None

    def cursor(self, name: str | None) -> TrajectoryCursor:
        """忽略条件名返回同一游标（惰性构建一次；游标自持快照语义
        保证共享安全——共享调度器被复写不影响已开出的轨迹）。"""
        if self._cursor is None:
            self._cursor = TrajectoryCursor(NetworkAssembler.rflow_scheduler(
                num_inference_steps=self._num_inference_steps,
                input_img_size_numel=self._input_img_size_numel,
            ))
        return self._cursor


class PerConditionSchedules:
    """逐条件日程表（MR-RATE）：锚 = 词汇表条件的空间 numel（
    ``ConditionVocabulary.latent_numel`` 单一派生入口——锚与 rollout
    噪声形状同源于条件词汇表，结构性防日程静默错位）。按条件名
    惰性构建并缓存游标；同锚条件（numel 碰撞）的日程逐位等价，
    各缓存独立游标、互不复写。"""

    def __init__(
        self, num_inference_steps: int, vocabulary: ConditionVocabulary,
    ) -> None:
        self._num_inference_steps = num_inference_steps
        self._vocabulary = vocabulary
        self._cursors: dict[str, TrajectoryCursor] = {}

    def cursor(self, name: str | None) -> TrajectoryCursor:
        """按条件名取该条件的 sigma 日程游标。缺名即拒绝（逐条件
        形态必须携带条件名——单域例外由 SingleConditionSchedules
        承担）；未知名即拒绝（全卷积 UNet 下形状错位不抛错，静默
        回退任意日程 = sigma 日程静默错位的最后一道运行时闸口）。"""
        if name is None:
            raise ValueError(
                "逐条件日程表必须携带条件名（condition.name）：sigma 日程"
                "锚逐条件派生，缺名无从选日程；单域 config 请走 "
                "SingleConditionSchedules"
            )
        if name not in self._cursors:
            self._cursors[name] = TrajectoryCursor(
                NetworkAssembler.rflow_scheduler(
                    num_inference_steps=self._num_inference_steps,
                    input_img_size_numel=self._vocabulary.latent_numel(name),
                ),
            )
        return self._cursors[name]
