"""判别器 warm-start 预训练（ADR-0007）：run 目录与产物契约、driver。

- ``driver``：PretrainDriver（经 ``cynosure.pretrain.driver`` 子模块
  消费——不做包级 re-export：driver 运行时装配 train 组件
  （``TrainingRuntime``），而 train 侧（runtime 的 warm-start 装载）也
  消费本包 artifacts；包级急切 re-export 会让 train → pretrain 的包
  初始化成环。子模块直进保持 runtime 依赖方向 acyclic）——real =
  Real sample pool（kind 守卫装载）、fake = 冻结基座的同源重构体
  （装配原语产出，ADR-0012），以在线期同款 ``OnlineUpdate.step``
  原语密集步进至 per-condition recon-AUC 达 RM readiness gate 或
  步数上限；单进程执行（World-1 退化路径），产物全局唯一；
- ``artifacts``：PretrainRun（run 目录）与 PretrainReport（报告契约 +
  守卫重载入口：kind 不符 / 缺报告 / 形态指纹不符即拒绝装载）。
"""

from cynosure.pretrain.artifacts import (
    PretrainPaths,
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)

__all__ = [
    "PretrainPaths",
    "PretrainProvenance",
    "PretrainReport",
    "PretrainRun",
]
