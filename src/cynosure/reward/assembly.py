"""判别器更新批装配原语（ADR-0012 的唯一新缝）。

输入 = 目标条件 + 专属随机流；输出 = 配对批（real 批 + 同源重构 fake 批
+ 条件标记）。real 侧经 ``RealSampling`` 无放回采样（条件匹配、容量硬
守卫照旧——``LatentManifest.assert_condition_capacity`` 装配期把门）；
fake 侧由**同批 real** 逐样本重构而来：

1. real latent（pool 存储域）乘回 policy 工作域；
2. 逐样本从该条件**被优化步**（``train_step_indices_m``，{2..15}）的
   sigma 日程点抽噪声水平 s（先抽 s 后抽 ε 的采样契约）；
3. rectified flow 插值加噪 x_s = (1−s)·x + s·ε；
4. 用当时的 policy（预训练 = 冻结基座、在线 = 当前 policy）以 η=0 确
   定性 ODE 续跑到 σ=0（与 anchor 续跑同一 ``continue_to_terminal``
   kernel）；
5. 除回 pool 存储域。

real/fake 同内容配对——判别器只能学生成伪影分界，「记真实样本库病例
共性」的内容捷径结构性失效。判别器输入恒干净域：加噪只发生在 fake 构
造的输入端，配对批两侧都不携带噪声（参数更新走干净域 ``patch_logits``
入口）。

随机性：重构构造走**专属命名随机流**（``TrainingRngStreams.RECON``
——与训练/评测/AUC 流不交叉，一条流的抽取数变化不漂移其余流）；
条件内的自由度抽取（组2 的源对/源条目，组1 实现不耗 RNG）、
``randint`` 抽日程位（= s）、``randn`` 抽 ε 依次全走本流——ε 的消耗
量与 s 的取值无关，同 seed 重放逐位一致、随续训分片落盘恢复后序列
不漂移。seeding 按 rank 无关的 shared seed 派生（runtime 装配位传原
seed）：s 抽样的调用结构是分布式 FSDP 集合序列的一部分，跨 rank 必须
一致（逐 rank 相异 = 首个判别器更新步集合错位死锁）。

按组语义（ADR-0012 决策 7）：条件构造与重构前向按组自然分派——组1
CFG 组合场、组2 裸条件单前向（condition 经 ``ConditionSampler`` 产出、
续跑经按组装配的采样场）。跨模态阶段（组2）同源重构的**语义**未裁决
（真正的「同源」需要（源影像, 目标标签）配对条件化、真实样本库现无源
影像——ADR-0012 非目标，stage-2 到来时另行设计）：机制缝本票照常对
组2 开放（机械链路可行），判别任务语义留待专项票。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from cynosure.policy.condition import RolloutCondition
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.numerics import AmpContext
from cynosure.policy.sampler import RolloutSampler
from cynosure.policy.schedules import ConditionSchedules
from cynosure.reward.sampler import RealSampling

if TYPE_CHECKING:
    from cynosure.train.rollout import ConditionSampler


@dataclass(frozen=True)
class PairBatch:
    """判别器更新批的配对批（ADR-0012 判别器更新步的输入契约）。

    ``fakes[i]`` 由 ``reals[i]`` 同源重构而来——两侧逐样本配对、同条件
    （``modality`` 标记）、同形状（同条件 latent 同形，#129）、同域
    （real pool 存储域，判别器比较两侧同域）。"""

    reals: torch.Tensor
    """real 批 [K, C, D, H, W]（pool 存储域；无放回采样，容量守卫照旧）。"""
    fakes: torch.Tensor
    """同源重构 fake 批 [K, C, D, H, W]（pool 存储域；fakes[i] ← reals[i]）。"""
    modality: str
    """本批的条件标记（目标条件键）——update 归因轴与 real 侧过滤条件
    同源（ADR-0008 决策 1）；fake 侧经同源自动匹配（ADR-0012 决策 8）。"""


class ReconstructionAssembler:
    """判别器更新批装配原语：目标条件 + 专属随机流 → 配对批。

    依赖全部协议注入（real 采样 / ODE 续跑 / sigma 日程 / 条件构造），
    预训练（冻结基座 + driver 装配）与在线（当前 policy + train 装配）
    两阶段经同一原语供批——构造同构、warm-start 权重不面临分布跳变
    （ADR-0012 决策 7）。
    """

    def __init__(
        self,
        real_sampler: RealSampling,
        sampler: RolloutSampler,
        schedules: ConditionSchedules,
        conditions: "ConditionSampler",
        train_step_indices: Sequence[int],
        batch_size_k: int,
        latent_scale_factor: float,
        generator: torch.Generator,
        amp: AmpContext,
    ) -> None:
        if not train_step_indices:
            raise ValueError("被优化步集合 M 不得为空（重构的 s 候选集）")
        if 0 in train_step_indices:
            raise ValueError(
                "被优化步集合 M 不得含日程下标 0（s≈1 最噪端是奇异点，"
                "天然排除于重构候选——ADR-0012 决策 2）"
            )
        if batch_size_k < 1:
            raise ValueError(f"batch_size_k 必须为正整数，得到 {batch_size_k}")
        self._real_sampler = real_sampler
        self._sampler = sampler
        self._schedules = schedules
        self._conditions = conditions
        self._step_indices: tuple[int, ...] = tuple(sorted(train_step_indices))
        self._batch_size_k = batch_size_k
        self._scale_factor = latent_scale_factor
        self._generator = generator
        self._amp = amp

    def assemble(self, modality: str) -> PairBatch:
        """装配该条件的判别器更新批：real 无放回采样 → 先抽 s 后抽 ε →
        同源重构 → 配对批（no_grad + autocast 口径——重构是 policy 的
        inference 前向，与 rollout 相同数值口径）。"""
        # 条件内的其余自由度（组2 的源对/源条目抽取）穿重构流：组1 实现
        # 不耗 RNG、组2 缺省会用 policy 主流（train/policy.py 把条件分布
        # 建在 rollout 流上）——不穿流则每个判别器更新步漂移 rollout 流，
        # rollout 样本序列从此依赖判别器更新节奏（流隔离契约）
        condition = self._conditions.sample_target(
            modality, generator=self._generator,
        )
        reals = self._real_sampler.sample(
            self._batch_size_k, modality=modality,
        )
        # 采样契约（先 s 后 ε）：先抽逐样本日程位（= 噪声水平 s），再抽
        # ε 张量——同 seed 重放的次序锚；ε 全量抽（消耗量与 s 取值无关）
        position = torch.randint(
            len(self._step_indices), (self._batch_size_k,),
            generator=self._generator,
        )
        steps = [self._step_indices[i] for i in position.tolist()]
        name = condition.name_or_raise()
        cursor = self._schedules.cursor(name)
        self._assert_indices_within_schedule(cursor, name)
        sigmas = [cursor.sigma_level(step) for step in steps]
        noise = torch.randn(
            reals.shape, generator=self._generator,
        ).to(reals.device)
        with torch.no_grad(), torch.autocast(
            self._amp.device_type, dtype=self._amp.dtype,
        ):
            fakes = self.reconstruct(reals, condition, sigmas, noise)
        return PairBatch(reals=reals, fakes=fakes, modality=modality)

    def candidate_sigmas(self, modality: str) -> tuple[float, ...]:
        """该条件重构候选噪声水平的导出面（被优化步的 sigma 日程点，
        按日程位升序）——s 抽样与日程同源的镜像口径（测试与诊断消费）。"""
        cursor = self._schedules.cursor(modality)
        self._assert_indices_within_schedule(cursor, modality)
        return tuple(cursor.sigma_level(step) for step in self._step_indices)

    def reconstruct(
        self,
        reals: torch.Tensor,
        condition: RolloutCondition,
        sigmas: Sequence[float],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """同源重构核：逐样本加噪到 σ 水平 → η=0 确定性 ODE 续跑到
        σ=0 → 域除回（纯数学，无 RNG——``assemble`` 持流，锚测试直接
        喂确定性输入）。

        同一 σ 水平的样本分组批量续跑（``continue_to_terminal`` 的
        batch 维并行）；s = 0 短路严格透传（σ=0 位于日程终点之后、无
        续跑空间——不乘除不积分，任意域换算系数下重构恒等，回归锚：
        s=0 重构的确定性 ODE 终点 = 起点）。"""
        if reals.shape != noise.shape:
            raise ValueError(
                f"noise 形状 {tuple(noise.shape)} 与 real 批 "
                f"{tuple(reals.shape)} 不符（逐样本 ε 须与 latent 同形）"
            )
        if len(sigmas) != reals.shape[0]:
            raise ValueError(
                f"sigma 数 {len(sigmas)} 与批大小 {reals.shape[0]} 不符"
                "（逐样本噪声水平）"
            )
        cursor = self._schedules.cursor(condition.name_or_raise())
        working = reals * self._scale_factor  # pool 存储域 → policy 工作域
        # σ 水平张量随输入设备落位（生产路径 real 批在加速器上——CPU
        # 常量的 device mismatch 在 fixture 全 CPU 口径下测不到）
        levels = torch.tensor(sigmas, device=reals.device).view(-1, 1, 1, 1, 1)
        noised = working * (1.0 - levels) + noise * levels
        groups: dict[float, list[int]] = {}
        for index, sigma in enumerate(sigmas):
            groups.setdefault(sigma, []).append(index)
        reconstructed = torch.empty_like(reals)
        for sigma, members in groups.items():
            if sigma == 0.0:
                # σ=0 短路：加噪恒等 + 零步续跑——严格透传原条目
                # （不走乘除往返，生产域换算系数下同样逐位恒等）
                reconstructed[members] = reals[members]
                continue
            terminal = self._sampler.continue_to_terminal(
                noised[members],
                self._start_index(cursor, sigma),
                condition,
            )
            reconstructed[members] = terminal / self._scale_factor
        return reconstructed

    def _start_index(
        self, cursor: "TrajectoryCursor", sigma: float,
    ) -> int:
        """σ 水平 → ``continue_to_terminal`` 的起点下标：σ = s_k 的样本
        位于第 k 步的输入位置（即第 k−1 步的输出位置），续跑从第 k 步
        开始积分。s = 0（σ=0 终点之后）走调用方短路、不进本方法；
        日程点外的 σ 与最噪端 s≈1（下标 0，M 排除）显式拒绝。"""
        for step in range(cursor.num_steps):
            if cursor.sigma_level(step) == sigma:
                if step == 0:
                    raise ValueError(
                        f"sigma={sigma} 是最噪端（日程下标 0，s≈1 奇异点）"
                        "——不在重构候选（被优化步集合 M 排除下标 0，"
                        "ADR-0012 决策 2）"
                    )
                return step - 1
        raise ValueError(
            f"sigma={sigma!r} 不是该条件的日程点（重构起点无从定位；"
            "候选 = 被优化步的 sigma 日程点）"
        )

    def _assert_indices_within_schedule(self, cursor, name: str) -> None:
        """被优化步集合在该条件日程内——小锚日程下 M 截尾在装配期显式
        暴露（可读报错点名条件与越界位），而非 ``sigma_level`` 越界的
        IndexError。"""
        overflow = [
            step for step in self._step_indices if step >= cursor.num_steps
        ]
        if overflow:
            raise ValueError(
                f"被优化步 {overflow} 超出条件 {name} 的日程"
                f"（num_steps={cursor.num_steps}）：重构候选的日程位越界"
            )
