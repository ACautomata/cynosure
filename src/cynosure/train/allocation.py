"""静态分配表（glossary「静态分配表」，#217 决议 1 / #231 骨架期）：iteration
与调度槽 → 生成条件的确定性映射。

**分配表 = config 条件轴 + seed + iteration + 槽号的确定函数**（纯函数、
无落盘状态——置换可从 payload 既有 iteration 与 seed 重导出，#218 第 3
节裁「分配表状态字段取消」的机制前提）。轮内置换：一轮 = ⌈C/D⌉ 个
iteration 覆盖全条件一次，每轮 seed 派生固定置换、轮间重洗；C > D 与
C < D 用单一规则覆盖（C < D：条件 tile 到槽、每轮重洗、每条件
⌊D/C⌋ 或 ⌈D/C⌉ 槽——#217 决议 1 字面「⌈D/C⌉ 或 ⌈D/C⌉+1」与均匀
tile 的正确边际差一、疑笔误，按实现口径勘误记档、随切换期 ADR 一并
提请（ADR-0019 决策 6））——不留两态特例代码路径。弃纯轮转（D 整除 C 时
卡-条件静态绑死：BraTS 4 序列 gauss 4 卡 card0 永远 t1n）。

置换派生**不消费命名流**（#218：分配表无落盘状态、不进注册表）——
轮种子 = (seed, 轮号) 的独立混合（splitmix64 终混，BaselineManifest
noise_seed 同款机制），经一次性 ``torch.Generator`` 抽 ``randperm``；
消耗序因此与四条命名流的消耗面正交。

槽轴 = 协程（默认 = 卡数）；同条件异噪声的多样性轴 = 协程（#217 调度
单元契约）。

**算法语义定性修正（#217，记档不声称零变化）**：现行条件序列是各卡
i.i.d. 均匀采样；轮转是边际均匀 + 序列相关（一轮内恒互异 vs 现行同
iteration 条件互异概率仅 ~9%）——判别器有效批的条件构成分布性质变化，
方向良性（一轮内条件互异增强判别器有效批的覆盖）。该定性修正的 ADR
载体随执行模型切换期落地（编排归 #226 切换期；本模块先行承载机制）。

组2 装配断言（#217 决议 1）：条件轴 = 目标端时，分配表的均匀边际假设
要求 ``cross_modal_pairs`` 对每目标端的有序对数相等（非对称配置下按端
分配的 (源, 目标) 对边际分布漂移最多 3 倍）——``assert_pair_symmetry``
在装配期显式拒绝。
"""

import torch

_GOLDEN = 0x9E3779B97F4A7C15
"""轮种子混合的金色比例常数（splitmix64 惯例，BaselineManifest.noise_seed
同款——与命名流偏移域、评测 manifest noise_seed 域均无语义交互）。"""

_SEED_MASK = 0xFFFFFFFFFFFFFFFF
"""64 位混合的掩码（混合中间值全程无符号 64 位运算）。"""

POSITIVE_INT64_MASK = 0x7FFFFFFFFFFFFFFF
"""torch.Generator.manual_seed 的正 int64 域掩码（混合产物超 int64 时
截取低 63 位——同一 (seed, 轮号) 恒得同值，确定性不受影响）。分配表
轮种子与窗口 real 抽取两消费方共享（#234 评审收敛：原两处同款私有
常量合一）。"""

_MIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_MIX_MULTIPLIER_2 = 0x94D049BB133111EB
"""splitmix64 终混的两轮乘法常数。"""


class SeedMixer:
    """整数项序列的 splitmix64 终混种子派生（分配表轮种子 /
    ``WindowRealDraw`` 窗口抽取种子 / ``BaselineManifest.noise_seed``
    的同款机制单一实现，#234 评审收敛）。

    混合结构：项序列求和入状态（外加金色比例常数）→ 三轮 xor-shift
    × 乘法终混（全程 64 位无符号域）。加法交换律保证「项集合恒定则
    同值」——各消费方以不同项结构调用（轮号 / 窗口号+条件摘要 /
    stage+index），历史混合结果经本实现逐位保持。返回**全 64 位**
    值——正 int64 域截取（torch ``Generator.manual_seed`` 域）是各
    消费方的历史私有口径（分配表与窗口抽取截取、Baseline manifest
    不截取），不在本实现内强加。不进 RNG 注册表（纯函数、无落盘
    状态）。"""

    @staticmethod
    def mix(*terms: int) -> int:
        """(terms...) 的终混种子（同项集合恒同值，与项次序无关）。"""
        mixed = (_GOLDEN + sum(terms)) & _SEED_MASK
        mixed ^= mixed >> 30
        mixed = (mixed * _MIX_MULTIPLIER_1) & _SEED_MASK
        mixed ^= mixed >> 27
        mixed = (mixed * _MIX_MULTIPLIER_2) & _SEED_MASK
        mixed ^= mixed >> 31
        return mixed


class AllocationTable:
    """iteration 与调度槽 → 条件的确定性映射（轮内置换纯函数）。"""

    def __init__(
        self, conditions: tuple[str, ...], slots: int, seed: int,
    ) -> None:
        """``conditions`` = 条件轴全集（条件分布 ``targets()`` 的登记序；
        集合知识归条件分布自身，本表不从 config 复制按组分派）；
        ``slots`` = 调度槽数（= 协程数）；``seed`` = run 主 seed。"""
        if not conditions:
            raise ValueError("分配表的条件轴不得为空（条件分布 targets() 缺席）")
        if slots < 1:
            raise ValueError(f"分配表的槽数须 ≥ 1，得到 {slots}")
        self._conditions = tuple(conditions)
        self._slots = slots
        self._seed = seed
        self._round_length = -(-len(self._conditions) // slots)
        """轮长 = ⌈C/D⌉：一轮的 iteration 数（覆盖全条件一次）。"""

    @property
    def slot_count(self) -> int:
        """调度槽数（= 协程数；静态绑卡的槽轴）。"""
        return self._slots

    @property
    def round_length(self) -> int:
        """轮长（⌈C/D⌉ 个 iteration 覆盖全条件一次）。"""
        return self._round_length

    @property
    def conditions(self) -> tuple[str, ...]:
        """条件轴全集（构造注入的登记序）。"""
        return self._conditions

    def condition_for(self, iteration: int, slot: int) -> str:
        """确定函数本体：iteration × 槽 → 条件名。

        轮 ``r = iteration // L`` 派生固定置换，轮内位置 ``p`` 取 tile
        序列的连续 D 段——同 iteration 内 C ≥ D 时槽间条件恒互异（slice
        不越置换长度），一轮恰好覆盖全条件（tile 补齐的尾段条件恰为
        重复者）。"""
        if not 0 <= slot < self._slots:
            raise ValueError(
                f"槽号 {slot} 越界（分配表槽数 {self._slots}）"
            )
        if iteration < 0:
            raise ValueError(f"iteration 号须非负，得到 {iteration}")
        round_index, position = divmod(iteration, self._round_length)
        permutation = self._round_permutation(round_index)
        tiled = [
            permutation[index % len(permutation)]
            for index in range(self._round_length * self._slots)
        ]
        return self._conditions[tiled[position * self._slots + slot]]

    def _round_permutation(self, round_index: int) -> list[int]:
        """轮 ``r`` 的条件置换（seed 派生、轮内固定、轮间重洗）。

        独立一次性 generator：不消费任何命名流（#218），同 (seed, 轮号)
        跨 run / 跨进程重导出同值。"""
        generator = torch.Generator()
        generator.manual_seed(self._round_seed(round_index))
        return torch.randperm(len(self._conditions), generator=generator).tolist()

    def _round_seed(self, round_index: int) -> int:
        """轮种子 = (seed, 轮号) 的 splitmix64 终混（``SeedMixer`` 单一
        实现；与命名流偏移域异构，撞位无语义）——截取正 int64 域
        （``Generator.manual_seed`` 域，共享 ``POSITIVE_INT64_MASK``）。"""
        return SeedMixer.mix(self._seed, round_index << 32) & POSITIVE_INT64_MASK

    @staticmethod
    def assert_pair_symmetry(
        pairs: list[tuple[str, str]], targets: tuple[str, ...],
    ) -> None:
        """组2 装配断言：每目标端的有序对数相等（#217 决议 1）。

        分配表按目标端均匀覆盖（条件轴 = targets()）；目标端内「源序列
        自由度」留槽内 RNG 流均匀抽取——非对称配置（如 t1n 有 3 个源、
        t2w 只有 1 个）下 (源, 目标) 对的边际分布按端漂移最多 3 倍，
        显式拒绝而非静默接受。"""
        if not pairs:
            raise ValueError("组2 有序对清单不得为空")
        counts = {target: 0 for target in targets}
        for source, target in pairs:
            if target not in counts:
                raise ValueError(
                    f"有序对 ({source}, {target}) 的目标端 {target!r} 不在 "
                    f"条件分布 targets()（{list(targets)}）——分配表的条件轴"
                    "与配对清单不一致"
                )
            counts[target] += 1
        distinct = sorted(set(counts.values()))
        if len(distinct) > 1:
            detail = ", ".join(
                f"{target}×{count}" for target, count in counts.items()
            )
            raise ValueError(
                f"cross_modal_pairs 对目标端非对称（{detail}）：分配表按目标"
                "端均匀覆盖、源序列自由度留槽内流均匀抽取，每目标端有序对"
                "数必须相等（否则 (源, 目标) 对的边际分布按端漂移）——"
                "非对称配对属 ADR-0012 非目标的条件化重构语义，另行专项"
            )
