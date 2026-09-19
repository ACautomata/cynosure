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

两条供批入口共用同一重构核（``reconstruct``），差在 s 的导出方式与随机流：

- ``assemble``（**更新批**）：s 逐样本均匀抽自被优化步日程点（ADR-0012
  决策 2）、ε 同流 —— 训练分布；
- ``measure_condition``（**gate 测量批**，ADR-0012 决策 5 的 recon-AUC
  构造面）：s 按候选步点**定序轮转**、ε 走**批次起手复位**的测量流 ——
  同输入逐位同输出。测量是上岗判据的原料（报告值与白名单都由它出），
  不许随「此前抽了多少次」漂移：逐条件读全量 held-out 卷打分，逐次测量
  与 run 配置无关地可比、可复算。
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


MEASUREMENT_STREAM_OFFSET = 10
"""测量流（gate 测量批的 ε 与条件构造）相对重构流的 seed 偏移：
``TrainingRngStreams`` 的注册表占 seed+0..+9（八条流；+5/+8 为退役中
的流位），+10 落在注册表之外——测量不参与续训状态清单，且与
``PretrainDriver`` 的 +7（SupportRule bootstrap）不撞位。"""


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
        # 测量流 + 复位模板（measure_condition 的确定性来源）：模板只取
        # 状态、永不被推进——逐次测量复位到同一个起手点，测量输入因此
        # 与「本 run 此前测量过几次」无关。与 recon 流同为 shared seed
        # 派生（+10 = 注册表 seed+9 之外），跨 rank 一致但互不交叉；
        # 不进 TrainingRngStreams 注册表——测量不参与续训状态清单。
        self._measurement_template = (
            torch.Generator()
            .manual_seed(generator.initial_seed() + MEASUREMENT_STREAM_OFFSET)
            .get_state()
        )
        self._amp = amp

    def assemble(self, modality: str) -> PairBatch:
        """装配该条件的判别器更新批：real 无放回采样 → 条件构造 → 先抽 s
        后抽 ε → 同源重构 → 配对批（no_grad + autocast 口径——重构是
        policy 的 inference 前向，与 rollout 相同数值口径）。"""
        reals = self._real_sampler.sample(
            self._batch_size_k, modality=modality,
        )
        # 条件构造在抽取序列的首位（组2 实现会消耗本流——次序即重放锚，
        # ADR-0007 交付期的抽取序逐位保持）
        condition = self._resolve_condition(modality, self._generator)
        # 采样契约（先 s 后 ε）：先抽逐样本日程位（= 噪声水平 s），再抽
        # ε 张量——同 seed 重放的次序锚；ε 全量抽（消耗量与 s 取值无关）
        position = torch.randint(
            len(self._step_indices), (self._batch_size_k,),
            generator=self._generator,
        )
        steps = [self._step_indices[i] for i in position.tolist()]
        cursor = self._schedules.cursor(condition.name_or_raise())
        self._assert_indices_within_schedule(cursor, condition.name_or_raise())
        sigmas = [cursor.sigma_level(step) for step in steps]
        noise = torch.randn(
            reals.shape, generator=self._generator,
        ).to(reals.device)
        return self._build(reals, condition, modality, sigmas, noise)

    def measure_condition(
        self, reals: torch.Tensor, modality: str,
    ) -> PairBatch:
        """该条件 **gate 测量批**的装配（ADR-0012 决策 5 的 recon-AUC
        构造面）：调用方给出的 real 卷 → 定序轮转 σ → 同源重构 →
        配对批。返回的 ``reals`` 与入参**同一个张量**——AUC 的 real 侧
        与 fake 侧因此逐样本配对，判别目标只剩重构伪影。

        ``reals`` = 该条件**全量** held-out 卷（``HeldOutAuc.
        condition_latents`` 的返回值），量由调用方持有：real 侧既作
        AUC 的 real、又作重构的源，两次各自自抽会让同源配对在测量层
        悄悄失效。

        与 ``assemble`` 的两处差别都在「测量的可复算性」上：

        - **s 定序轮转**：第 i 枚卷取候选步点的第 ``i % |M|`` 位——全员
          覆盖候选噪声带，且同输入恒同输出（逐样本抽 s 会让报告值随
          「本 run 此前抽了几次」漂移，上岗判据不可复算）；
        - **ε 走批次起手复位的测量流**：不消耗 recon 流（续训分片的流
          位置不被测量次数搅动），也不漂移 policy 主流。
        """
        if reals.shape[0] < 1:
            raise ValueError("测量批需要非空 real 卷（重构的源）")
        sigmas = self._round_robin_sigmas(modality, reals.shape[0])
        # 批次起手复位（一次，不逐卷复位）：本批的条件构造与 ε 从这里
        # 同一起手点顺序展开——同输入的逐次测量逐位同输出
        measurement = torch.Generator()
        measurement.set_state(self._measurement_template)
        condition = self._resolve_condition(modality, measurement)
        noise = torch.randn(reals.shape, generator=measurement).to(reals.device)
        return self._build(reals, condition, modality, sigmas, noise)

    def _round_robin_sigmas(self, modality: str, count: int) -> list[float]:
        """定序轮转的逐卷噪声水平：第 i 枚卷取候选步点的第 ``i % |M|``
        位（候选 = 被优化步的 sigma 日程点，按日程位升序——与
        ``candidate_sigmas`` 同源）。``measure_condition`` 与
        ``measurement_forward_count`` 共享本定序——成本读数与实际测量
        批同源推算，不是平行复刻。"""
        cursor = self._schedules.cursor(modality)
        self._assert_indices_within_schedule(cursor, modality)
        candidates = [cursor.sigma_level(step) for step in self._step_indices]
        return [candidates[index % len(candidates)] for index in range(count)]

    def measurement_forward_count(
        self, reals: torch.Tensor, modality: str,
    ) -> int:
        """该测量批（``reals`` 同上 ``measure_condition`` 的入参）重构的
        policy 前向次数（定序轮转下的确定值）——#171 AC5 成本口径的读数
        面：**无全 ODE 量产**在事件流上可核对。逐卷步数 = 日程步数 − 起点
        下标（见 ``_start_index``），恒严格小于 ``num_steps``：候选档位
        取自被优化步（{2..15} 类中段日程点），起点下标 ≥ 0。

        量纲随入参卷数（测量批的规模由调用方持有的 real 决定——装配原
        语的 real 侧采样器≠调用方的 real 来源时，按采样器规模读会得到
        与真实测量批无关的数）。"""
        cursor = self._schedules.cursor(modality)
        sigmas = self._round_robin_sigmas(modality, reals.shape[0])
        return sum(self._remaining_steps(cursor, sigma) for sigma in sigmas)

    def _remaining_steps(
        self, cursor: TrajectoryCursor, sigma: float,
    ) -> int:
        """该 σ 档位的续跑步数（= 日程步数 − 1 − 起点下标，
        ``_start_index`` 的镜像口径）。候选档位经
        ``_assert_indices_within_schedule`` 钉在中段（1..num_steps−2），
        零步档位在装配期即不可达，本方法只处理正步数。"""
        return cursor.num_steps - 1 - self._start_index(cursor, sigma)

    def _resolve_condition(
        self, modality: str, generator: torch.Generator,
    ) -> RolloutCondition:
        """条件构造（穿注入的随机流）：组1 实现不耗 RNG；组2 的源对/
        源条目抽取会消耗——不穿流则调用方流出（policy 主流）会被漂移
        （train/policy.py 把条件分布建在 rollout 流上），rollout 样本
        序列从此依赖判别器更新/测量节奏（流隔离契约）。"""
        return self._conditions.sample_target(modality, generator=generator)

    def _build(
        self,
        reals: torch.Tensor,
        condition: RolloutCondition,
        modality: str,
        sigmas: Sequence[float],
        noise: torch.Tensor,
    ) -> PairBatch:
        """两条供批入口的共用装配尾：重构 → 配对批（no_grad + autocast
        口径——重构是 policy 的 inference 前向，与 rollout 相同数值
        口径）。``condition`` 由调用方按其抽取序构造后注入。"""
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
        """被优化步集合在该条件日程的**中段**（末位亦是非法候选）——
        小锚日程下 M 截尾在装配期显式暴露（可读报错点名条件与越界位），
        而非 ``sigma_level`` 越界的 IndexError 或首步测量才炸。

        末位被排除与 config 的 ``train_step_indices_m`` 校验同源
        （``max(M) ≤ num_steps − 2``）：末位之后无续跑空间，重构会退化为
        透传（fake ≡ real），而测量面**没有** ``reconstruct`` 的 s=0 短路
        ——它是判别器要学的「生成伪影」的零内容批次，静默进入测量会把
        上岗判据污染成 chance 带上的噪声。"""
        overflow = [
            step for step in self._step_indices
            if step >= cursor.num_steps - 1
        ]
        if overflow:
            raise ValueError(
                f"被优化步 {overflow} 越界条件 {name} 的重构候选"
                f"（num_steps={cursor.num_steps}：合法候选为 1..num_steps−2"
                "，首位的 s≈1 奇异端与末位的零续跑空间都排除）"
            )
