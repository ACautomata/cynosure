"""判别器链（async 执行序，#234 / #220 结票全口径的执行组件）。

**判别器窗口（N_d window）**：两次判别器步之间的 iteration 区间（窗口
末 = 判别器步 iteration ``w``，``w % N_d == 0``；首窗口退化为
``[0]``）。每窗口每卡恰 K 个重构任务、有效批 = K×卡数与 N_d 解耦；
K 任务逐 iter 发射 ``floor(K/L)``（L = 实际窗口长，余数补窗口首
iter——K < N_d 时「均匀」名不副实，#220 决议 5 spec 注明）；任务创建
序 = 桶序（条件名排序）×对序（卡, iter, j）。

**判别器桶（Discriminator bucket）**：单条件同形 (real, fake) 同源
对集合——MR-RATE 异条件异形下混合条件批不可拼接为单一张量，判别器
前向按桶进行（同桶批前向、跨桶梯度累积），跨卡聚合只有梯度 allreduce
（配对数据不跨卡搬运）。容器不变式（单条件、两侧同形同量同 device、
桶内对同形）构造期断言（#220 决议 3）。

**real 侧窗口起点全局无放回抽取**（#220 决议 10）：逐条件抽「窗口内
该条件总对数」（分配表纯函数）→ 切片到任务；抽取调用序 = 条件名排序
（与桶序同源）；跨窗口重抽 = 现行跨步语义。per-(卡×流名) 独立流只落
recon 流（fake 侧）——real 侧是池数据不是随机流：随机性经
(seed, 窗口号, 条件名) 的 splitmix64 终混派生**一次性** generator
（与分配表轮种子同款机制），不进 RNG 注册表、无续训状态（纯函数可从
(seed, 窗口号) 重导出）。

**判别器步相位序列**（#220 决议 4）：先全桶 eval（train acc 复算、
同一参数快照、eval 相不推进 spectral norm 幂迭代）后全桶 train
（更新前向 + backward 累积，SN 推进 = 2×桶数：real/fake 各一次）→
跨卡梯度 allreduce（SUM，M1 形态与逐 k 收集-同步同构）→ optimizer
.step → 步末 spectral norm u/v broadcast（卡 0 权威；奇异向量不可
平均，broadcast 即最优）。

**逐对等权全局 mean**（#220 决议 8 显式新裁）：``loss = Σ_b
loss_b×(n_b/N_total)`` → backward → allreduce SUM；N_total = K×卡数
常量。⚠️ 与现行 patch 级 mean 分离、与逐桶 mean 等权求和在异形桶下
不等价——以条件对称意图为名立此存照，不得以「零变化」表述。

fake 产出即 CPU 暂存（#220 决议 16）：任务 fake 构造（policy 前向）
后立即落 CPU，判别器步前 join 后逐桶按桶序回迁装配；GPU 峰值 = 最大
桶 + 累积梯度。
"""

import zlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.auc import HeldOutAuc
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import ChunkedScorer, LatentScorer
from cynosure.reward.update import ConditionUpdateDetail, UpdateReport
from cynosure.train.allocation import AllocationTable, SeedMixer

if TYPE_CHECKING:
    from cynosure.train.executor import SlotExample

_POSITIVE_INT64_MASK = 0x7FFFFFFFFFFFFFFF
"""torch.Generator.manual_seed 的正 int64 域掩码（窗口抽取种子的
私有截取口径——分配表轮种子同款；Baseline manifest 不截取的历史
分歧见 ``SeedMixer`` docstring）。"""


@dataclass(frozen=True)
class WindowPairRecord:
    """单对任务产出（fake 产出即 CPU 暂存的记账单元，#220 决议 16）。

    不变式构造期断言：条件非空、real/fake 同形同 device、批维恰 1
    （单对容器——桶由对聚合、不由批切分）；``fake`` 由 ``real`` 同源
    重构而来是装配链的结构性质，同形同量是产出点的可断言面。"""

    condition: str
    real: torch.Tensor
    fake: torch.Tensor

    def __post_init__(self) -> None:
        if not self.condition:
            raise ValueError("单对记录的条件名不得为空")
        if self.real.dim() != 5 or self.real.shape[0] != 1:
            raise ValueError(
                f"单对记录的 real 须为批维恰 1 的 [1,C,D,H,W]，得到 "
                f"{tuple(self.real.shape)}（桶由对聚合、不由批切分）"
            )
        if self.fake.shape != self.real.shape:
            raise ValueError(
                f"单对记录两侧须同形（fakes[i]←reals[i] 同源配对），得到 "
                f"real {tuple(self.real.shape)} / fake {tuple(self.fake.shape)}"
            )
        if self.fake.device != self.real.device:
            raise ValueError(
                f"单对记录两侧须同 device（暂存域一致），得到 "
                f"real {self.real.device} / fake {self.fake.device}"
            )


@dataclass(frozen=True)
class DiscriminatorBucket:
    """判别器桶：单条件同形 (real, fake) 对集合（#220 决议 3/4）。

    桶序恒条件名排序（确定性遍历锚）；不变式（单条件、两侧同形同量、
    5 维批）构造期断言——``PairBatch`` 零校验现状不延续。BraTS 单条件
    域退化为单桶。
    """

    condition: str
    reals: torch.Tensor
    fakes: torch.Tensor

    def __post_init__(self) -> None:
        if not self.condition:
            raise ValueError("判别器桶的条件名不得为空")
        if self.reals.dim() != 5 or self.reals.shape[0] < 1:
            raise ValueError(
                f"判别器桶须为非空 5 维批 [n,C,D,H,W]，得到 "
                f"{tuple(self.reals.shape)}"
            )
        if self.fakes.shape != self.reals.shape:
            raise ValueError(
                f"判别器桶两侧须同形同量（同源配对），real "
                f"{tuple(self.reals.shape)} / fake {tuple(self.fakes.shape)}"
            )
        if self.fakes.device != self.reals.device:
            raise ValueError(
                "判别器桶两侧须同 device（同存储域比较前提），real "
                f"{self.reals.device} / fake {self.fakes.device}"
            )

    @property
    def pair_count(self) -> int:
        """本桶对数——逐对等权全局 mean 的加权因子（#220 决议 8）。"""
        return self.reals.shape[0]

    @classmethod
    def assemble(
        cls,
        condition: str,
        pairs: tuple[WindowPairRecord, ...] | list[WindowPairRecord],
        device: torch.device,
    ) -> "DiscriminatorBucket":
        """桶装配：同条件对序列 → 批（步前 join 后按桶序回迁的落点）。

        逐对断言条件匹配（单条件不变式）与形状一致（同形桶前提），
        cat 后一次性回迁 ``device``（CPU 暂存 → 判别器前向域）。"""
        if not pairs:
            raise ValueError(f"判别器桶 {condition!r} 须至少一对（非空）")
        for pair in pairs:
            if pair.condition != condition:
                raise ValueError(
                    f"判别器桶 {condition!r} 收到异条件对 {pair.condition!r}"
                    "——桶单条件不变式（#220 决议 3）"
                )
        shape = pairs[0].real.shape
        for pair in pairs:
            if pair.real.shape != shape:
                raise ValueError(
                    f"判别器桶 {condition!r} 内对须同形，得到 "
                    f"{tuple(shape)} 与 {tuple(pair.real.shape)}"
                    "（同条件同统一网格的结构前提，同名异形装配期拒绝）"
                )
        reals = torch.cat([pair.real for pair in pairs]).to(device)
        fakes = torch.cat([pair.fake for pair in pairs]).to(device)
        return cls(condition=condition, reals=reals, fakes=fakes)


@dataclass(frozen=True)
class WindowTask:
    """窗口内单重构任务：桶序×对序全局排序的调度单元（#220 决议 5）。

    条件 = 分配表在该 (iteration, slot) 的派发值（纯函数）；``index``
    = 本 (卡, iter) 内任务序（卡内槽轮转 j % n_slots 的 j）。"""

    card: int
    iteration: int
    slot: int
    index: int
    condition: str


class DiscriminatorWindow:
    """判别器窗口计划（分配表 + N_d + K + 卡槽绑定的确定函数）。

    全组件无状态、可重导出（同分配表先例）：窗口构成、发射节奏、桶
    计数都是纯函数——续训重放时从 (seed, 已完成的 iteration) 重导出。
    """

    def __init__(
        self,
        allocation: AllocationTable,
        n_d: int,
        batch_size_k: int,
        card_slots: dict[int, tuple[int, ...]],
    ) -> None:
        if n_d < 1:
            raise ValueError(f"判别器窗口的 N_d 须 ≥ 1，得到 {n_d}")
        if batch_size_k < 1:
            raise ValueError(f"判别器窗口的 K 须 ≥ 1，得到 {batch_size_k}")
        if not card_slots:
            raise ValueError("判别器窗口须至少绑定一张卡（卡槽绑定为空）")
        bound = [slot for slots in card_slots.values() for slot in slots]
        if sorted(bound) != list(range(allocation.slot_count)):
            raise ValueError(
                f"卡槽绑定 {sorted(bound)} 须恰好覆盖全槽集 "
                f"0..{allocation.slot_count - 1}（静态绑定的纯函数镜像）"
            )
        self._allocation = allocation
        self._n_d = n_d
        self._batch_size_k = batch_size_k
        self._card_slots = {card: tuple(slots) for card, slots in card_slots.items()}

    @property
    def n_d(self) -> int:
        return self._n_d

    @property
    def batch_size_k(self) -> int:
        return self._batch_size_k

    def is_step(self, iteration: int) -> bool:
        """本 iteration 是否判别器步（``iteration % N_d == 0``）。"""
        return iteration >= 0 and iteration % self._n_d == 0

    def _length(self, step_iteration: int) -> int:
        """实际窗口长 = min(N_d, 步号+1)——首窗口（步 0）退化为 [0]，
        run 从判别器步起步时 K 任务全落窗口首 iter（θ_0 权重，fake
        代际零漂移）。"""
        if not self.is_step(step_iteration):
            raise ValueError(
                f"iteration {step_iteration} 不是判别器步（% {self._n_d} "
                "≠ 0）——窗口只锚判别器步"
            )
        return min(self._n_d, step_iteration + 1)

    def window_start(self, step_iteration: int) -> int:
        """窗口首 iter（窗口起点 real 抽取的发生位，#220 决议 10）。"""
        return step_iteration - self._length(step_iteration) + 1

    def window_iterations(self, step_iteration: int) -> range:
        """窗口的 iter 区间 [start, step]（含判别器步 iter 本身）。"""
        return range(self.window_start(step_iteration), step_iteration + 1)

    def step_for_launch(self, iteration: int) -> int | None:
        """该 iter 若为某窗口的首 iter，返回其判别器步号；否则 None。

        门面在窗口首 iter 的 rollout 相前做 real 抽取的消费口（判定
        纯函数：iter 0 → 步 0；iter ≥ 1 → 步 iter+N_d−1，校验其为判
        别器步且窗口首恰为本 iter）。"""
        if iteration < 0:
            return None
        step = 0 if iteration == 0 else iteration + self._n_d - 1
        if self.is_step(step) and self.window_start(step) == iteration:
            return step
        return None

    def launch_count(self, step_iteration: int, offset: int) -> int:
        """窗口内第 ``offset`` 个 iter 的发射数：``K // L``（余数补窗
        口首 iter），L = 实际窗口长。"""
        length = self._length(step_iteration)
        if not 0 <= offset < length:
            raise ValueError(
                f"窗口内偏移 {offset} 越界（窗口长 {length}）"
            )
        base, remainder = divmod(self._batch_size_k, length)
        return base + (remainder if offset == 0 else 0)

    def tasks(self, step_iteration: int) -> tuple[WindowTask, ...]:
        """全窗口任务清单，全局排序 = 桶序（条件名）×对序（卡, iter,
        j）——任务创建序锚（real 条目分配序的基准；recon 流消耗序见下）。

        **recon 流消耗序的裁决字面展开**（#220 决议 5「recon 流按创建
        序消耗」×「逐 iter 发射」的组合语义）：fake 构造只能发生在任务
        被发射的 iteration rollout 相——跨 iteration 消耗序 = 发射序
        （iteration 升序，物理必然）；**iteration 内**的消耗序 = 创建序
        的 iter 内相对序（桶序×对序）。per-槽流因此按「(iteration,
        本槽任务的创建序相对位)」消耗——同槽任务在 _draw 插入序
        （= 全局创建序）中的相对位即执行序，门面 ``tasks_for`` 的筛选
        保序保证这一点；执行完成序（协程并发）与流消耗无关。"""
        start = self.window_start(step_iteration)
        tasks: list[WindowTask] = []
        for card in sorted(self._card_slots):
            slots = self._card_slots[card]
            for iteration in range(start, step_iteration + 1):
                count = self.launch_count(step_iteration, iteration - start)
                for index in range(count):
                    slot = slots[index % len(slots)]
                    tasks.append(WindowTask(
                        card=card,
                        iteration=iteration,
                        slot=slot,
                        index=index,
                        condition=self._allocation.condition_for(
                            iteration, slot,
                        ),
                    ))
        tasks.sort(key=lambda t: (t.condition, t.card, t.iteration, t.index))
        return tuple(tasks)

    def bucket_counts(self, step_iteration: int) -> tuple[tuple[str, int], ...]:
        """桶构成：(条件, 对数) 清单，条件名排序（桶序遍历锚）。

        Σ 对数 = 卡数 × K = N_total/1（每窗口每卡恰 K 对的不变量）。"""
        counts: dict[str, int] = {}
        for task in self.tasks(step_iteration):
            counts[task.condition] = counts.get(task.condition, 0) + 1
        return tuple(sorted(counts.items()))


class WindowRealDraw:
    """real 侧窗口起点全局无放回抽取（#220 决议 10）。

    逐条件（条件名排序）以 (seed, 窗口号, 条件名) splitmix64 终混派生
    的一次性 generator 抽全量排列、取窗口对数；条目按任务创建序
    （桶序×对序的组内投影）切片到任务。不进 RNG 注册表（纯函数重导出、
    无续训状态）；同窗口同条件条目互异（全局无放回），跨窗口重抽可
    重复（现行跨步语义的窗口化）。
    """

    def __init__(self, manifest: LatentManifest) -> None:
        if manifest.kind != "real_pool":
            raise ValueError(
                f"窗口 real 抽取须 real_pool 工件，得到 {manifest.kind}"
                "（held-out real 冒充训练 real 失去池语义）"
            )
        self._manifest = manifest

    @staticmethod
    def _derive_seed(seed: int, step_iteration: int, condition: str) -> int:
        """(seed, 窗口号, 条件名) 的 splitmix64 终混（``SeedMixer`` 单一
        实现；条件名经 crc32 内容寻址混入——与 PreparePipeline.noise_seed
        同款机制）——截取正 int64 域（``Generator.manual_seed`` 域，本
        消费方的历史私有口径）。"""
        digest = zlib.crc32(condition.encode("utf-8"))
        return (
            SeedMixer.mix(seed, step_iteration << 32, digest)
            & _POSITIVE_INT64_MASK
        )

    def assign(
        self,
        seed: int,
        step_iteration: int,
        window: DiscriminatorWindow,
    ) -> dict[WindowTask, PoolEntry]:
        """窗口 real 抽取：任务 → 池条目的全局无放回分配。

        抽取调用序 = 条件名排序（sorted 的显式遍历锚）；抽取量超过条件
        池即拒绝（容量守卫的抽取期兜底——装配期逐条件 ≥ K×卡数把门，
        正常路径不可达）。"""
        tasks = window.tasks(step_iteration)
        by_condition: dict[str, list[WindowTask]] = {}
        for task in tasks:
            by_condition.setdefault(task.condition, []).append(task)
        assigned: dict[WindowTask, PoolEntry] = {}
        for condition in sorted(by_condition):
            group = by_condition[condition]
            candidates = [
                entry for entry in self._manifest.entries
                if entry.modality == condition
            ]
            if len(group) > len(candidates):
                raise ValueError(
                    f"窗口 real 抽取超出条件 {condition!r} 的池容量：需 "
                    f"{len(group)} 条、池 {len(candidates)} 条（全局无放回；"
                    "装配期容量守卫 = 逐条件全池 ≥ K×卡数应已把门——"
                    "直调路径或工件缺条的防御兜底）"
                )
            generator = torch.Generator().manual_seed(
                self._derive_seed(seed, step_iteration, condition),
            )
            permutation = torch.randperm(
                len(candidates), generator=generator,
            ).tolist()
            chosen = [candidates[i] for i in permutation[:len(group)]]
            for task, entry in zip(group, chosen):
                assigned[task] = entry
        return assigned


class PooledHeldOutAuc:
    """per-condition 池化 AUC 的**本卡测量段**（#220 决议 12/15）。

    同 iter 同条件各卡 fakes 汇一个测量批（逻辑汇集——配对数据不跨
    卡搬运，fake 打分在 fakes 所在卡：判别器副本同值、步末 u/v 同步
    保证）；real 侧每卡经 heldout 流按条件名排序采样打分（卡内
    consume 序恒条件名排序——#220 决议 15 的确定性锚）。分数由门面
    收集（future 回传序 = 卡号升序，单进程多卡无进程组 gather 的
    结构性替代）在单点做 float64 midrank Mann-Whitney——每活跃条件
    恰一次读数。打分前向 no_grad（AUC 非可微、判别器恒 eval 相）。
    """

    def __init__(
        self,
        heldout_manifest: LatentManifest,
        scorer: LatentScorer,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        if heldout_manifest.kind != "heldout_real":
            raise ValueError(
                f"池化 AUC 需 heldout_real 工件，得到 {heldout_manifest.kind}"
                "（train real 冒充 held-out 会失去 out-of-sample 语义）"
            )
        self._manifest = heldout_manifest
        self._chunked = ChunkedScorer(scorer)
        self._real_sampler = RealPoolSampler(heldout_manifest, generator, device)
        self._device = device

    def score_conditions(
        self, conditions: tuple[str, ...], count: int,
    ) -> dict[str, torch.Tensor]:
        """条件名排序逐条件：无放回采 ``count`` 条 held-out real（上限
        = 该条件池）+ 定块打分 → CPU 展平分数（门面的池化原料）。

        采样数钳到 min(count, 该条件池)——与 per-例口径同构（Mann-
        Whitney 对非对称 n×m 有效）；消耗本卡 heldout 流（流终态锚的
        比较面）。``conditions`` 传入即消耗序（调用方按条件名排序）。
        """
        with torch.no_grad():
            scores: dict[str, torch.Tensor] = {}
            for condition in conditions:
                pool_size = self._pool_size(condition)
                sample_count = max(1, min(count, pool_size))
                latents = self._real_sampler.sample(
                    sample_count, modality=condition,
                )
                scores[condition] = self._chunked_scores(latents).cpu()
            return scores

    def _pool_size(self, condition: str) -> int:
        """该条件的 held-out 条目数（采样上界钳制面）。"""
        return sum(
            1 for entry in self._manifest.entries
            if entry.modality == condition
        )

    def _chunked_scores(self, latents: torch.Tensor) -> torch.Tensor:
        """定块打分前向（``ChunkedScorer`` 单一实现：GroupNorm 分块与
        全批逐位等价、分数级拼接不改 rank 口径）。"""
        return self._chunked.scores(latents)


class DiscriminatorWindowRun:
    """判别器窗口的**运行期账簿**（窗口起点抽取 → 逐 iter 任务摊派 →
    rollout 相产出累计 → 判别器步消费清零的单写者状态机）。

    与 ``DiscriminatorWindow``（计划纯函数）分工：本类持有跨 iteration
    的运行态（当前窗口的任务→real 条目切片、逐卡已 join 的产出、逐
    条件池化 AUC 的最后一次有效测量），门面单写者驱动——``launch``（
    窗口首 iter 判定 + 抽取）、``collect``（rollout 相后按任务累计）、
    ``take`` 系（判别器步的装配/分叉/告警消费）与 ``clear``（步末
    清任务账）。线程亲和：门面主线程唯一写者；任务产出经 SlotExample
    跨线程回传后累计（future 完成序 = 主线程可见）。

    任务账（``_draw``/``_records``）随判别器步消费清零；AUC 账
    （``_auc``）**跨窗口持久**——步末快照的覆盖式入账即「窗口内最后
    一次有效测量」（#220 决议 14），非判别器步 iter 的事件桥接与
    run 后的观测面都读这本持久账（其逐条件来源 iter = 该条件最后被
    分配的判别器步，存在性由分配表纯函数钉死）。"""

    def __init__(
        self,
        window: DiscriminatorWindow,
        real_draw: WindowRealDraw,
        seed: int,
    ) -> None:
        self._window = window
        self._real_draw = real_draw
        self._seed = seed
        self._draw: dict[WindowTask, PoolEntry] = {}
        """当前窗口任务 → real 池条目切片（窗口起点抽取结果）。"""
        self._records: dict[int, list[tuple[WindowTask, WindowPairRecord]]] = {}
        """卡 → 本窗口已 join 的任务产出（rollout 相逐 iter 累计）。"""
        self._auc: dict[str, float] = {}
        """逐条件最后一次有效池化读数（**跨窗口持久**、覆盖式更新）——
        判别器步末快照入账；「窗口内最后一次有效测量」（#220 决议 14）
        的窗口延拓：本窗口活跃条件被新读数覆盖，本窗口未活跃条件保留
        上次读数作非判别器步 iter 事件的桥接回退。"""

    @property
    def window(self) -> DiscriminatorWindow:
        """窗口计划纯函数（门面事件面/测试锚的取数口）。"""
        return self._window

    def launch(self, iteration: int) -> None:
        """窗口首 iter 判定（``step_for_launch``）+ 起点抽取——非首
        iter 为 no-op（当前窗口切片继续有效）。"""
        step = self._window.step_for_launch(iteration)
        if step is not None:
            self._draw = self._real_draw.assign(self._seed, step, self._window)

    def tasks_for(
        self, iteration: int, slot: int,
    ) -> tuple[tuple[WindowTask, PoolEntry], ...]:
        """本 (iteration, 槽) 摊派的窗口任务（real 条目随任务下发）。"""
        return tuple(
            (task, entry)
            for task, entry in self._draw.items()
            if task.iteration == iteration and task.slot == slot
        )

    def collect(self, example: "SlotExample") -> None:
        """rollout 相产出的窗口任务记账（按任务归属卡累计——rollout
        相 gather 完成即「判别器步前 join」的语义落点）。"""
        for task, record in example.window_pairs:
            self._records.setdefault(task.card, []).append((task, record))

    def planned_count(self, card: int) -> int:
        """窗口计划的本卡任务数（判别器步 join 对账的「不静默降批」
        基准，#220 决议 6）。"""
        return sum(1 for task in self._draw if task.card == card)

    def records(
        self, card: int,
    ) -> list[tuple[WindowTask, WindowPairRecord]]:
        """本卡已 join 的任务产出（判别器步桶装配的原料）。"""
        return self._records.get(card, [])

    def record_auc(self, pooled: dict[str, float]) -> None:
        """池化读数入账（覆盖式：本窗口活跃条件的「最后一次有效测量」
        刷新，#220 决议 12/14；本窗口未活跃条件的既有读数原样保留）。"""
        self._auc.update(pooled)

    def window_auc(self, condition: str) -> float | None:
        """该条件的最后一次有效池化读数（分叉配对与事件桥接的取数口；
        本 run 尚无读数 = None——首现于非判别器步 iter 的条件，其所在
        窗口的测量尚未发生）。"""
        return self._auc.get(condition)

    def clear(self) -> None:
        """判别器步末清**任务账**（``_draw``/``_records``——下一窗口
        首 iter 重新抽取）；AUC 账（``_auc``）跨窗口持久，不在此清。"""
        self._draw = {}
        self._records = {}


class DiscriminatorPhase:
    """判别器步的 per-卡相位编排（#220 决议 4/8）。

    段一 ``accumulate``（卡线程）：先全桶 eval（train acc 复算、同一
    参数快照、eval 相不推进 spectral norm 幂迭代）后全桶 train（更新
    前向 ×2×桶数 + 加权 backward 累积，SN 推进 = 2×桶数）——逐桶交替
    结构性否决（跨桶快照分叉破坏 ADR-0009 同刻可比）。段二门面跨卡
    梯度 allreduce SUM（M1 形态，与逐 k 收集-同步同构）。段三
    ``step``：optimizer.step + eval 相恢复。段四门面步末 u/v broadcast
    （卡 0 权威）——多卡面；单卡/CPU fixture 全部结构性退化。
    """

    def __init__(
        self,
        scorer: LatentScorer,
        optimizer: torch.optim.Optimizer,
        total_pairs: int,
    ) -> None:
        if total_pairs < 1:
            raise ValueError(
                f"逐对等权分母 N_total 须 ≥ 1，得到 {total_pairs}"
                "（= K×卡数，判别器步的有效批常量）"
            )
        self._scorer = scorer
        self._optimizer = optimizer
        self._total = total_pairs

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """本卡判别器优化器（编排方 step 段消费）。"""
        return self._optimizer

    def accumulate(self, buckets: list[DiscriminatorBucket]) -> UpdateReport:
        """eval 全桶 + train 全桶 + 加权 backward（梯度落本卡 .grad）。

        ``buckets`` 须非空且按桶序（条件名排序）传入——每卡每窗口恰
        K ≥ 1 任务的窗口不变量保证非空，直调空清单显式拒绝。"""
        if not buckets:
            raise ValueError(
                "判别器步须至少一桶（每窗口每卡恰 K ≥ 1 对的不变量）"
            )
        discriminator = self._scorer.discriminator
        discriminator.eval()
        try:
            with torch.no_grad():
                accuracies = [
                    self._bucket_accuracy(bucket) for bucket in buckets
                ]
        finally:
            discriminator.train()
        self._optimizer.zero_grad()
        weighted: torch.Tensor | None = None
        details: list[ConditionUpdateDetail] = []
        for bucket, accuracy in zip(buckets, accuracies):
            logits_real = self._scorer.patch_logits(bucket.reals)
            logits_fake = self._scorer.patch_logits(bucket.fakes)
            terms = self._scorer.discriminator_terms(
                logits_real, logits_fake,
            )
            # 逐对等权全局 mean（#220 决议 8 显式新裁）：
            # loss_b × (n_b/N_total) 求和 → 单次 backward（跨桶梯度
            # 累积在统一计算图内，allreduce 在 backward 之后）
            scaled = terms.total * (bucket.pair_count / self._total)
            weighted = scaled if weighted is None else weighted + scaled
            details.append(ConditionUpdateDetail(
                condition=bucket.condition,
                loss_discriminator=terms.total.item(),
                loss_real_term=terms.real_term.item(),
                loss_fake_term=terms.fake_term.item(),
                pair_count=bucket.pair_count,
                train_pairwise_acc=accuracy,
            ))
        assert weighted is not None
        weighted.backward()
        return UpdateReport(
            conditions=tuple(details),
            batch_size=sum(bucket.pair_count for bucket in buckets),
            global_batch_size=self._total,
            loss_discriminator=float(weighted.detach()),
        )

    def step(self) -> None:
        """optimizer.step + eval 相恢复（步末快照归位打分/监控面）。"""
        self._optimizer.step()
        self._scorer.discriminator.eval()

    def spectral_buffers(self) -> dict[str, torch.Tensor]:
        """本卡判别器 spectral norm 的 ``_u``/``_v`` buffer 集（步末
        broadcast 的取数面；无 spectral norm 时为空 dict——broadcast
        面结构性跳过）。"""
        return {
            name: buffer
            for name, buffer in self._scorer.discriminator.named_buffers()
            if name.endswith("._u") or name.endswith("._v")
        }

    def _bucket_accuracy(self, bucket: DiscriminatorBucket) -> float:
        """单桶 train 侧干净域 pairwise 准确率（eval 相 no_grad 复算，
        ADR-0009-β 估计量 = Mann-Whitney pairwise 占比，与 held-out
        AUC 同尺）。"""
        real_scores = self._scorer.patch_logits(bucket.reals).flatten()
        fake_scores = self._scorer.patch_logits(bucket.fakes).flatten()
        return HeldOutAuc.auc_from_scores(real_scores, fake_scores)

    @staticmethod
    def _require_multicard(phases: list["DiscriminatorPhase"]) -> None:
        """多卡集合面的判定单点（与 ``PerKCollectReduce`` 同构）：
        单副本恒 False（无跨卡面）；多副本须全 CUDA（M1 前提）。"""
        if len(phases) <= 1:
            return
        devices = {
            phase._scorer.discriminator.parameters().__next__().device
            for phase in phases
        }
        if any(device.type != "cuda" for device in devices):
            raise ValueError(
                "多副本判别器归约要求全部副本落 CUDA 设备，得到 "
                f"{sorted(map(str, devices))}"
            )

    @staticmethod
    def reduce_gradients(phases: list["DiscriminatorPhase"]) -> None:
        """跨卡判别器梯度 SUM allreduce（M1：逐 tensor、串行、固定全
        卡组——与逐 k 收集-同步同 communicator 语义的先后两次之一，
        #217 §3「判别器与 policy allreduce 共用 communicator 先后两
        次」）。无梯度参数跨卡结构性一致（同前向图），恒跳过。单卡
        结构性退化（本卡梯度即全局）。"""
        if len(phases) <= 1:
            return
        DiscriminatorPhase._require_multicard(phases)
        parameter_lists = [
            list(phase._scorer.discriminator.parameters()) for phase in phases
        ]
        for grads in zip(*parameter_lists):
            if grads[0].grad is None:
                continue
            torch.cuda.nccl.all_reduce(
                [grad for grad in (p.grad for p in grads)],
                op=torch.cuda.nccl.SUM,
            )

    @staticmethod
    def synchronize_spectral(phases: list["DiscriminatorPhase"]) -> None:
        """步末 spectral norm u/v broadcast（卡 0 权威，#220 决议 2）。

        奇异向量不可平均，broadcast 即最优；步末而非步首——窗口内
        AUC 打分须读同步后值。多卡 CUDA 面生效；单卡/无 spectral
        norm（buffer 集为空）结构性跳过，不空发集合调用。"""
        if len(phases) <= 1:
            return
        DiscriminatorPhase._require_multicard(phases)
        reference = phases[0].spectral_buffers()
        if not reference:
            return
        buffer_maps = [
            dict(phase._scorer.discriminator.named_buffers())
            for phase in phases
        ]
        for name in reference:
            missing = [
                index for index, mapping in enumerate(buffer_maps)
                if name not in mapping
            ]
            if missing:
                raise ValueError(
                    f"spectral buffer {name!r} 跨卡不同构（卡 {missing} "
                    "缺失）——broadcast 前提破坏，拒绝静默部分同步"
                )
            torch.cuda.nccl.broadcast(
                [mapping[name] for mapping in buffer_maps], root=0,
            )
