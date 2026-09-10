"""判别器 warm-start 预训练（ADR-0007）：driver、run 目录与产物契约。

- ``driver``：PretrainDriver——real = Real sample pool（kind 守卫装载）、
  fake = base policy 冻结 rollout 量产（复用回放缓冲 base 分区采样入口），
  以在线期同款 ``OnlineUpdate.step`` 原语密集步进至 held-out AUC 达 RM
  readiness gate 或步数上限；单进程执行（World-1 退化路径），产物全局唯一；
- ``artifacts``：PretrainRun（run 目录）与 PretrainReport（报告契约 +
  守卫重载入口：kind 不符 / 缺报告 / 形态指纹不符即拒绝装载）。
"""

from cynosure.pretrain.artifacts import (
    PretrainPaths,
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.pretrain.driver import PretrainDriver

__all__ = [
    "PretrainDriver",
    "PretrainPaths",
    "PretrainProvenance",
    "PretrainReport",
    "PretrainRun",
]
