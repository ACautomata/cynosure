"""判别器链期单元锚（#220 断言面：桶构造 / 窗口计划 / 逐桶等权 / 相位序列
/ real 窗口抽取 / UpdateReport 升格）——纯 CPU、无标记，与门面接线档
（test_async_executor 的判别器链节）分层。

锚面与裁决的对应：

- 桶构造（#220 决议 3/4）：单条件同形 (real,fake) 同源对集合、桶序恒
  条件名排序、容器不变式构造期断言（混条件 / 异形 / 空桶显式拒绝）；
- 窗口计划（决议 5/10）：每窗口每卡恰 K 对、K 任务逐 iter 发射
  ``floor(K/N_d)``（余数补窗口首 iter）、任务创建序 = 桶序×对序、桶
  构成与 real 抽取量 = 分配表纯函数；
- 逐对等权全局 mean（决议 8）：``Σ_b loss_b×(n_b/N_total)``——
  显式新裁（现行 patch 级 mean 的分离），不得表述为零变化；
- 相位序列（决议 4）：先全桶 eval（acc 复算、同一参数快照）后全桶
  train（更新前向 + backward 累积）→ allreduce → optimizer.step；
- real 抽取（决议 10）：窗口起点全局无放回、调用序 = 条件名排序、
  逐条件抽窗口对数、切片到任务、同 (seed, 窗口, 条件) 可复算、不进
  命名流（一次性派生 generator）；
- 任务双型（#277 / ADR-0020 决策 2）：任务型 = 创建序位置上的确定性
  交错纯函数（每 ⌈1/ρ⌉ 个任务一个 rollout 型，零随机流）——ρ=0 时
  任务集与现状逐位一致、指派确定性、比例与边界情形（ρ 不整除 /
  K×卡数 < ⌈1/ρ⌉）、在途窗口重导出的类型一致；
- UpdateReport 升格（决议 11）：per-condition 明细 + batch_size = 本卡
  K + 全局量 N_total；单值字段 property 化（单桶派生、多桶拒绝）。
"""

import math
from fractions import Fraction

import pytest
import torch

from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.scorer import LsganTerms
from cynosure.train.allocation import AllocationTable
from cynosure.train.discriminator import (
    DiscriminatorBucket,
    DiscriminatorPhase,
    DiscriminatorWindow,
    DiscriminatorWindowRun,
    WindowPairRecord,
    WindowRealDraw,
    WindowTask,
)


def _record(condition: str, shape=(1, 4, 4, 4, 4), fill: float = 0.5) -> WindowPairRecord:
    real = torch.full(shape, fill)
    return WindowPairRecord(
        condition=condition,
        real=real,
        fake=real.clone() + 0.1,
    )


class TestWindowPairRecord:
    """单对产出容器的不变式断言（#220 决议 3/16：fake 产出即 CPU 暂存
    的记账单元，同源同形在产出点钉死）。"""

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="同形"):
            WindowPairRecord(
                condition="t1n",
                real=torch.zeros(1, 4, 4, 4, 4),
                fake=torch.zeros(2, 4, 4, 4, 4),
            )

    def test_rejects_multi_sample_pair(self) -> None:
        """单对容器：批维 ≠ 1 显式拒绝（桶由对聚合、不由批切分）。"""
        with pytest.raises(ValueError, match="单对"):
            WindowPairRecord(
                condition="t1n",
                real=torch.zeros(2, 4, 4, 4, 4),
                fake=torch.zeros(2, 4, 4, 4, 4),
            )

    def test_rejects_empty_condition(self) -> None:
        with pytest.raises(ValueError, match="条件"):
            WindowPairRecord(
                condition="",
                real=torch.zeros(1, 4, 4, 4, 4),
                fake=torch.zeros(1, 4, 4, 4, 4),
            )


class TestDiscriminatorBucket:
    """判别器桶（#220 决议 3/4）：单条件同形对集合 + 构造期断言。"""

    def test_assembles_pairs_in_order(self) -> None:
        pairs = [_record("t1n", fill=0.1), _record("t1n", fill=0.9)]
        bucket = DiscriminatorBucket.assemble("t1n", pairs, torch.device("cpu"))
        assert bucket.condition == "t1n"
        assert bucket.pair_count == 2
        assert bucket.reals.shape == (2, 4, 4, 4, 4)  # cat 单对 [1,C,D,H,W]
        assert torch.equal(bucket.reals[1], pairs[1].real[0])  # 对序保持

    def test_rejects_mixed_conditions(self) -> None:
        with pytest.raises(ValueError, match="单条件"):
            DiscriminatorBucket.assemble(
                "t1n", [_record("t1n"), _record("t1c")], torch.device("cpu"),
            )

    def test_rejects_mismatched_pair_shapes(self) -> None:
        with pytest.raises(ValueError, match="同形"):
            DiscriminatorBucket.assemble(
                "t1n",
                [_record("t1n"), _record("t1n", shape=(1, 4, 8, 4, 4))],
                torch.device("cpu"),
            )

    def test_rejects_empty_bucket(self) -> None:
        with pytest.raises(ValueError, match="非空"):
            DiscriminatorBucket.assemble("t1n", [], torch.device("cpu"))

    def test_direct_construction_asserts_invariants(self) -> None:
        with pytest.raises(ValueError, match="同形"):
            DiscriminatorBucket(
                condition="t1n",
                reals=torch.zeros(2, 4, 4, 4, 4),
                fakes=torch.zeros(3, 4, 4, 4, 4),
            )


class TestDiscriminatorWindow:
    """窗口计划纯函数（#220 决议 5/10）：发射节奏、任务创建序、桶构成。"""

    def _table(self, conditions=("a", "b", "c", "d"), slots=2, seed=0):
        return AllocationTable(conditions, slots, seed)

    def test_launch_schedule_floor_with_remainder_at_first(self) -> None:
        """K=5、N_d=3 的完整窗口（步 3 = [1,2,3]）：每卡发射
        [3, 1, 1]（floor=1，余 2 补窗口首 iter——首 iter 恰
        floor+余数）。"""
        window = DiscriminatorWindow(self._table(), n_d=3, batch_size_k=5, card_slots={0: (0, 1)})
        assert [window.launch_count(3, i) for i in range(3)] == [3, 1, 1]
        assert sum(window.launch_count(3, i) for i in range(3)) == 5

    def test_launch_schedule_uniform_when_divides(self) -> None:
        window = DiscriminatorWindow(self._table(), n_d=2, batch_size_k=4, card_slots={0: (0, 1)})
        assert [window.launch_count(2, i) for i in range(2)] == [2, 2]

    def test_k_smaller_than_n_d_uneven_by_design(self) -> None:
        """K<N_d 时「均匀」名不副实（#220 决议 5 spec 注明）：K=2、N_d=4
        → 发射 [2, 0, 0, 0]——任务全集在窗口首 iter，不静默摊伪均匀。"""
        window = DiscriminatorWindow(self._table(), n_d=4, batch_size_k=2, card_slots={0: (0, 1)})
        assert [window.launch_count(4, i) for i in range(4)] == [2, 0, 0, 0]

    def test_every_card_gets_exactly_k_per_window(self) -> None:
        window = DiscriminatorWindow(
            self._table(slots=4), n_d=3, batch_size_k=5,
            card_slots={0: (0, 1), 1: (2, 3)},
        )
        tasks = window.tasks(3)  # 窗口 [1,2,3]，判别器步在 iter 3
        assert len(tasks) == 2 * 5
        per_card = {0: 0, 1: 0}
        for task in tasks:
            per_card[task.card] += 1
        assert per_card == {0: 5, 1: 5}

    def test_window_iterations_end_at_step(self) -> None:
        window = DiscriminatorWindow(self._table(), n_d=3, batch_size_k=2, card_slots={0: (0, 1)})
        assert list(window.window_iterations(6)) == [4, 5, 6]
        assert list(window.window_iterations(0)) == [0]  # 首窗口退化
        assert window.is_step(6) and not window.is_step(7)

    def test_task_creation_order_is_bucket_then_pair(self) -> None:
        """任务创建序 = 桶序（条件名排序）× 对序（卡, iter, j）——
        recon 流消耗序与 real 条目分配序的共同锚。"""
        table = AllocationTable(("b", "a"), 2, seed=1)
        window = DiscriminatorWindow(table, n_d=1, batch_size_k=2, card_slots={0: (0, 1)})
        tasks = window.tasks(0)
        ordered = sorted(tasks, key=lambda t: (t.condition, t.card, t.iteration, t.index))
        assert [(t.condition, t.card, t.iteration, t.index) for t in tasks] == [
            (t.condition, t.card, t.iteration, t.index) for t in ordered
        ]
        assert tasks[0].condition <= tasks[-1].condition

    def test_task_condition_comes_from_allocation(self) -> None:
        table = self._table(seed=3)
        window = DiscriminatorWindow(table, n_d=1, batch_size_k=3, card_slots={0: (0, 1)})
        for task in window.tasks(0):
            assert task.condition == table.condition_for(task.iteration, task.slot)

    def test_multislot_card_spreads_tasks_over_slots(self) -> None:
        """一卡多槽（CPU fixture 拓扑）：卡的任务摊到卡内槽（j % n_slots），
        每卡每窗口合计仍恰 K——「每窗口每卡恰 K 对」的槽域投影。"""
        window = DiscriminatorWindow(
            self._table(), n_d=2, batch_size_k=3, card_slots={0: (0, 1)},
        )
        by_slot = {0: 0, 1: 0}
        for task in window.tasks(2):  # 窗口 [1,2] 发射 [2,1]
            by_slot[task.slot] += 1
        assert by_slot == {0: 2, 1: 1}  # 3 = 2 + 1（j%2 轮转）
        assert sum(by_slot.values()) == 3

    def test_bucket_counts_group_by_condition_sorted(self) -> None:
        table = AllocationTable(("d", "c", "b", "a"), 2, seed=2)
        window = DiscriminatorWindow(table, n_d=2, batch_size_k=2, card_slots={0: (0, 1)})
        buckets = window.bucket_counts(2)  # 窗口 [1,2]，单卡 K=2
        names = [name for name, _ in buckets]
        assert names == sorted(names)
        assert sum(count for _, count in buckets) == 2  # 卡数(1) × K

    def test_rejects_zero_k_and_n_d(self) -> None:
        with pytest.raises(ValueError):
            DiscriminatorWindow(self._table(), n_d=1, batch_size_k=0, card_slots={0: (0,)})
        with pytest.raises(ValueError):
            DiscriminatorWindow(self._table(), n_d=0, batch_size_k=1, card_slots={0: (0,)})

    def test_next_step_is_first_discriminator_step_at_or_after(self) -> None:
        """恢复点在途窗口定位的纯函数（#265）：≥ iteration 的下一个判别
        器步号（iteration 自身是步即它自己）。"""
        window = DiscriminatorWindow(
            self._table(), n_d=4, batch_size_k=2, card_slots={0: (0, 1)},
        )
        assert window.next_step(0) == 0
        assert window.next_step(1) == 4
        assert window.next_step(3) == 4
        assert window.next_step(4) == 4
        assert window.next_step(6) == 8
        pair = DiscriminatorWindow(
            self._table(), n_d=2, batch_size_k=2, card_slots={0: (0, 1)},
        )
        assert pair.next_step(1) == 2
        assert pair.next_step(2) == 2
        assert pair.next_step(3) == 4

    def test_next_step_rejects_negative_iteration(self) -> None:
        window = DiscriminatorWindow(
            self._table(), n_d=4, batch_size_k=2, card_slots={0: (0, 1)},
        )
        with pytest.raises(ValueError):
            window.next_step(-1)

    def test_zero_ratio_all_recon_task_set_bitwise_current(self) -> None:
        """ρ=0（默认）：全 recon 型，且任务集与现状逐位一致——默认构造
        与显式 ρ=0.0 同 seed 重放逐位相等（类型字段默认值不参与结构
        差异），任务五元组锚（条件, 卡, iter, 序, 槽）与分配表纯函数
        派生值逐一相同（#277 验收 1）。"""
        table = AllocationTable(("b", "a"), 2, seed=1)
        default = DiscriminatorWindow(table, n_d=1, batch_size_k=2, card_slots={0: (0, 1)})
        explicit = DiscriminatorWindow(
            table, n_d=1, batch_size_k=2, card_slots={0: (0, 1)},
            rollout_fake_ratio=0.0,
        )
        tasks = default.tasks(0)
        assert tasks == explicit.tasks(0)  # 同 seed 重放逐位锚
        assert all(task.kind == "recon" for task in tasks)
        assert [
            (t.condition, t.card, t.iteration, t.index, t.slot)
            for t in tasks
        ] == sorted(
            (table.condition_for(0, slot), 0, 0, index, slot)
            for index, slot in ((0, 0), (1, 1))
        )

    def test_kind_assignment_same_seed_window_is_bitwise_reproducible(self) -> None:
        """指派确定性（#277 验收 2）：同 (seed, 窗口号) 重导出的任务型
        序列逐位一致——两次导出全等（含类型字段的 dataclass 相等）。"""
        window = DiscriminatorWindow(
            self._table(), n_d=2, batch_size_k=4, card_slots={0: (0, 1)},
            rollout_fake_ratio=0.25,  # 4 任务、interval=4 → 位置序 3 为 rollout
        )
        first = [task.kind for task in window.tasks(2)]
        second = [task.kind for task in window.tasks(2)]
        assert first == second
        assert first == ["recon", "recon", "recon", "rollout"]

    def test_kind_does_not_change_task_creation_order(self) -> None:
        """类型字段不改变任务创建序（#277 验收 2）：ρ>0 的排序键序列
        与 ρ=0 逐位相同——类型不参与排序，real 条目分配序与 recon 流
        消耗序的锚不漂移。"""
        table = self._table()
        base = DiscriminatorWindow(table, n_d=2, batch_size_k=3, card_slots={0: (0, 1)})
        typed = DiscriminatorWindow(
            table, n_d=2, batch_size_k=3, card_slots={0: (0, 1)},
            rollout_fake_ratio=0.5,
        )
        def order_key(t: WindowTask) -> tuple:
            return (t.condition, t.card, t.iteration, t.index)

        assert [order_key(t) for t in typed.tasks(2)] == [
            order_key(t) for t in base.tasks(2)
        ]

    def test_rollout_ratio_interleaving_hits_one_per_interval(self) -> None:
        """指派比例（#277 验收 3）：ρ=0.25 → ⌈1/ρ⌉=4，创建序位置序 i
        （0 基）满足 (i+1)%4==0 的任务为 rollout 型——两卡 6 任务恰 1
        个 rollout，落位 = 第 4 个（逐位锚，非只数个数）。"""
        window = DiscriminatorWindow(
            self._table(slots=4), n_d=2, batch_size_k=3,
            card_slots={0: (0, 1), 1: (2, 3)},
            rollout_fake_ratio=0.25,
        )
        tasks = window.tasks(2)  # 窗口 [1,2]，2 卡 × K=3 = 6 任务
        assert len(tasks) == 6
        assert [task.kind for task in tasks] == [
            "recon", "recon", "recon", "rollout", "recon", "recon",
        ]

    def test_ratio_non_integer_reciprocal_uses_ceiling(self) -> None:
        """ρ 不整除的边界（#277 验收 3）：⌈1/ρ⌉ 规则——ρ=0.3（1/ρ=3.33）
        与 ρ=0.25 同为 interval 4，指派序列逐位相同；ρ=0.6（1/ρ=1.67）
        → interval 2，每 2 个任务一个 rollout。"""
        table = self._table(slots=4)
        common = dict(n_d=2, batch_size_k=3, card_slots={0: (0, 1), 1: (2, 3)})
        quarter = [t.kind for t in DiscriminatorWindow(
            table, rollout_fake_ratio=0.25, **common,
        ).tasks(2)]
        point_three = [t.kind for t in DiscriminatorWindow(
            table, rollout_fake_ratio=0.3, **common,
        ).tasks(2)]
        assert point_three == quarter  # ⌈1/0.3⌉=⌈1/0.25⌉=4
        six = [t.kind for t in DiscriminatorWindow(
            table, rollout_fake_ratio=0.6, **common,
        ).tasks(2)]
        assert six == [
            "recon", "rollout", "recon", "rollout", "recon", "rollout",
        ]

    def test_window_smaller_than_interval_has_no_rollout(self) -> None:
        """K×卡数 < ⌈1/ρ⌉ 的边界（#277 验收 3）：窗口任务数凑不满一个
        交错周期 → 零 rollout 型（不静默摊、不越界指派）。"""
        window = DiscriminatorWindow(
            self._table(), n_d=1, batch_size_k=2, card_slots={0: (0, 1)},
            rollout_fake_ratio=0.25,  # interval=4 > 2 任务
        )
        assert all(task.kind == "recon" for task in window.tasks(0))

    def test_interval_snaps_unit_fraction_neighbors_to_intent(self) -> None:
        """interval 的单位分数邻域锚：ρ 取配置浮点 snapping 后的简单有
        理倒数取顶——double(1/3) 低于 1/3（精确算术取顶得 4，非语义），
        double(1/49) 的浮点除法 1.0/ρ = 49.00000000000001（取顶得 50，
        亦非语义）；用户写 1/n 即每 n 个任务一个 rollout（#277 验收 3
        的数值保真面）。"""
        rho_third = 1 / 3  # double 略低于 1/3
        assert math.ceil(1.0 / rho_third) == 3  # 浮点除法本例碰巧合语义
        assert Fraction(1) / Fraction(rho_third) > 3  # 精确算术反而失守
        third = DiscriminatorWindow(
            self._table(), n_d=1, batch_size_k=4, card_slots={0: (0, 1)},
            rollout_fake_ratio=rho_third,
        )
        assert [task.kind for task in third.tasks(0)] == [
            "recon", "recon", "rollout", "recon",
        ]
        rho_49 = 1 / 49  # double 略低于 1/49
        assert math.ceil(1.0 / rho_49) == 50  # 浮点除法本例失守
        forty_ninth = DiscriminatorWindow(
            self._table(), n_d=1, batch_size_k=50, card_slots={0: (0, 1)},
            rollout_fake_ratio=rho_49,
        )
        assert [task.kind for task in forty_ninth.tasks(0)] == (
            ["recon"] * 48 + ["rollout", "recon"]
        )

    def test_tiny_ratio_below_snap_resolution_falls_back_without_crash(self) -> None:
        """snapping 分辨率下界锚（评审复审新增）：ρ=1e-7 < 5e-7 时
        limit_denominator snap 到 0（裸 snapping 路径 1/0 崩溃）——
        声明域 (0,1) 内不得崩，退回浮点除法取顶（interval=1e7 > 窗口
        任务数 → 全 recon）；小到倒数溢出则可读拒绝。"""
        window = DiscriminatorWindow(
            self._table(), n_d=1, batch_size_k=2, card_slots={0: (0, 1)},
            rollout_fake_ratio=1e-7,
        )
        assert all(task.kind == "recon" for task in window.tasks(0))
        with pytest.raises(ValueError, match="ρ"):
            DiscriminatorWindow(
                self._table(slots=1), n_d=1, batch_size_k=2,
                card_slots={0: (0,)},
                rollout_fake_ratio=1e-320,  # 倒数溢出 = 无可行窗口尺度
            )

    def test_rejects_rollout_ratio_out_of_domain(self) -> None:
        """ρ 物理域 [0,1) 的直调防御（schema 域之外的兜底，ADR-0020：
        ρ=1 = 纯 rollout 终点退化面为开区间）。"""
        for bad in (1.0, -0.1, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="ρ"):
                DiscriminatorWindow(
                    self._table(), n_d=1, batch_size_k=1,
                    card_slots={0: (0,)}, rollout_fake_ratio=bad,
                )


class TestWindowRealDraw:
    """real 侧窗口起点全局无放回抽取（#220 决议 10）：条件名排序调用序、
    逐条件抽窗口对数、切片到任务、一次性派生 generator（不进命名流）。"""

    def _manifest(self, entries_per_condition: dict[str, int]) -> LatentManifest:
        entries = [
            PoolEntry(
                case_id=f"{condition}-{index}",
                modality=condition,
                latent=f"{condition}/{index}.pt",
                spacing=(1.0, 1.0, 1.0),
            )
            for condition, count in sorted(entries_per_condition.items())
            for index in range(count)
        ]
        return LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=(1, 4, 4, 4),
            split_seed=0,
            split_sizes={"train": len(entries)},
            entries=entries,
        )

    def _window(self, n_d=1, k=2, slots=2):
        return DiscriminatorWindow(
            AllocationTable(("a", "b"), slots, seed=0),
            n_d=n_d, batch_size_k=k, card_slots={0: tuple(range(slots))},
        )

    def test_draw_count_matches_window_tasks(self) -> None:
        window = self._window(n_d=2, k=3, slots=2)
        manifest = self._manifest({"a": 8, "b": 8})
        draw = WindowRealDraw(manifest)
        assigned = draw.assign(seed=7, step_iteration=2, window=window)
        assert len(assigned) == len(window.tasks(2))

    def test_draw_is_without_replacement_within_condition(self) -> None:
        window = self._window(n_d=1, k=2, slots=2)
        manifest = self._manifest({"a": 6, "b": 6})
        draw = WindowRealDraw(manifest)
        assigned = draw.assign(seed=7, step_iteration=0, window=window)
        per_condition: dict[str, list[str]] = {}
        for task, entry in assigned.items():
            per_condition.setdefault(task.condition, []).append(entry.case_id)
        for condition, ids in per_condition.items():
            assert len(set(ids)) == len(ids), (
                f"同窗口同条件 {condition} 的 real 条目必须互异（全局无放回）"
            )

    def test_same_seed_same_window_reproducible(self) -> None:
        window = self._window(n_d=2, k=3, slots=2)
        manifest = self._manifest({"a": 8, "b": 8})
        draw = WindowRealDraw(manifest)
        first = draw.assign(seed=7, step_iteration=2, window=window)
        second = draw.assign(seed=7, step_iteration=2, window=window)
        assert {k: v.case_id for k, v in first.items()} == {
            k: v.case_id for k, v in second.items()
        }

    def test_different_windows_redraw(self) -> None:
        """跨窗口重抽（= 现行跨步语义）：不同窗口号的抽取分道。"""
        manifest = self._manifest({"a": 12, "b": 12})
        draw = WindowRealDraw(manifest)
        window = self._window(n_d=2, k=2, slots=2)
        first = draw.assign(seed=7, step_iteration=2, window=window)
        second = draw.assign(seed=7, step_iteration=4, window=window)
        ids_first = sorted(entry.case_id for entry in first.values())
        ids_second = sorted(entry.case_id for entry in second.values())
        assert ids_first != ids_second

    def test_capacity_violation_rejected_at_draw(self) -> None:
        """抽取量 > 条件池：构造期断言（容量守卫的装配期把门之外的
        抽取期兜底——正常装配不可达，防御直调路径）。单条件表 + 一槽：
        K=4 任务全落该条件。"""
        window = DiscriminatorWindow(
            AllocationTable(("a",), 1, seed=0),
            n_d=1, batch_size_k=4, card_slots={0: (0,)},
        )
        manifest = self._manifest({"a": 3})  # a 池 3 < 4
        draw = WindowRealDraw(manifest)
        with pytest.raises(ValueError, match="容量|无放回"):
            draw.assign(seed=7, step_iteration=0, window=window)

    def test_condition_call_order_is_name_sorted(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """抽取调用序 = 条件名排序（#220 决议 10：与桶序同源的确定性
        遍历锚）——观测派生种子的调用记录。"""
        window = self._window(n_d=1, k=2, slots=2)
        manifest = self._manifest({"b": 8, "a": 8})
        draw = WindowRealDraw(manifest)
        calls: list[str] = []
        real_derive = WindowRealDraw._derive_seed

        def spy(seed, step_iteration, condition):
            calls.append(condition)
            return real_derive(seed, step_iteration, condition)

        monkeypatch.setattr(
            WindowRealDraw, "_derive_seed", staticmethod(spy),
        )
        draw.assign(seed=7, step_iteration=0, window=window)
        assert calls == ["a", "b"]


class TestDiscriminatorWindowRun:
    """运行期账簿的恢复面（#265）：在途窗口纯函数重导出 + AUC 桥接账
    自持（state/adopt——ADR-0014 协作者自持先例，overfit EMA 同款）。"""

    def _manifest(self, entries_per_condition: dict[str, int]) -> LatentManifest:
        entries = [
            PoolEntry(
                case_id=f"{condition}-{index}",
                modality=condition,
                latent=f"{condition}/{index}.pt",
                spacing=(1.0, 1.0, 1.0),
            )
            for condition, count in sorted(entries_per_condition.items())
            for index in range(count)
        ]
        return LatentManifest(
            kind="real_pool",
            encoder="synthetic",
            latent_shape=(1, 4, 4, 4),
            split_seed=0,
            split_sizes={"train": len(entries)},
            entries=entries,
        )

    def _run(self, *, n_d=4, k=4, slots=2, seed=7, conditions=("a", "b"), ratio=0.0):
        window = DiscriminatorWindow(
            AllocationTable(conditions, slots, seed=0),
            n_d=n_d, batch_size_k=k, card_slots={0: tuple(range(slots))},
            rollout_fake_ratio=ratio,
        )
        return DiscriminatorWindowRun(
            window,
            WindowRealDraw(self._manifest({c: 12 for c in conditions})),
            seed,
        )

    def test_reexport_drops_tasks_before_resume_iteration(self) -> None:
        """恢复点 6、N_d=4、K=4（窗口 [5,8]、每 iter 每卡 1 任务）：
        重导出后仅 iteration 6/7/8 的任务在账、5 已弃——已过 iter 不
        补跑（#265：跨快照 fake 的 accepted drift 进记账面）。"""
        run = self._run()
        run.reexport(step_iteration=8, resume_iteration=6)
        assert run.planned_count(0) == 3  # 1 任务/iter × [6,7,8]
        assert len(run.tasks_for(5, 0)) == 0
        assert len(run.tasks_for(6, 0)) == 1
        assert len(run.tasks_for(8, 0)) == 1

    def test_reexport_keeps_original_entries_for_remaining_tasks(self) -> None:
        """重导出 = 同 (seed, 步号) 的抽取纯函数：剩余任务的 real 条目
        与全量抽取逐一相同（切片序不变、不重洗）。"""
        run = self._run()
        run.reexport(step_iteration=8, resume_iteration=6)
        full = WindowRealDraw(self._manifest({"a": 12, "b": 12})).assign(
            seed=7, step_iteration=8, window=run.window,
        )
        for iteration in (6, 7, 8):
            for slot in (0, 1):
                expected = tuple(
                    (task, entry) for task, entry in full.items()
                    if task.iteration == iteration and task.slot == slot
                )
                assert run.tasks_for(iteration, slot) == expected

    def test_reexport_at_launch_iteration_is_full_window(self) -> None:
        """恢复点即窗口首 iter：重导出 = 全量抽取（无丢弃——与 run
        循环 launch 的抽取同款）。"""
        run = self._run()
        run.reexport(step_iteration=8, resume_iteration=5)
        assert run.planned_count(0) == 4  # K 任务全集

    def test_reexport_preserves_task_kinds(self) -> None:
        """在途窗口重导出的类型字段覆盖（#277 验收 4）：ρ>0 时恢复后
        剩余任务的类型与全量重导同窗口任务按 iteration ≥ 恢复点的
        切片逐位一致——类型是创建序位置的纯函数，重导出不重指派、
        丢弃任务不改变幸存任务的创建序位置。N_d=4、K=4：窗口 [5,8]
        每 iter 1 任务、共 4 任务，ρ=0.25 → 位置序 3（0 基）为
        rollout 型（全窗型序列的交错锚，与分配表无关）。"""
        run = self._run(ratio=0.25)
        full = run.window.tasks(8)
        assert [task.kind for task in full] == [
            "recon", "recon", "recon", "rollout",
        ]
        run.reexport(step_iteration=8, resume_iteration=6)
        full_kinds = {
            (t.condition, t.card, t.iteration, t.index): t.kind
            for t in full
        }
        remaining = [
            task
            for iteration in (6, 7, 8)
            for slot in (0, 1)
            for task, _ in run.tasks_for(iteration, slot)
        ]
        assert len(remaining) == 3  # 1 任务/iter × [6,7,8]（#265 丢弃语义）
        for task in remaining:
            assert task.kind == full_kinds[
                (task.condition, task.card, task.iteration, task.index)
            ]

    def test_reexport_rejects_unrebuildable_window_with_start_criterion(self) -> None:
        """恢复守卫（#265）：K=2 < N_d=4 时窗口任务全落窗口首 iter、
        恢复点在其后 → 重导出后无剩余任务 → 可读报错点名窗口起点
        口径（N_d、窗口起点 iteration 5、K<N_d 判据）——取代循环内
        「判别器步须至少一桶」的误导性空桶报错。"""
        run = self._run(n_d=4, k=2)
        with pytest.raises(ValueError, match="窗口起点") as exc_info:
            run.reexport(step_iteration=8, resume_iteration=6)
        message = str(exc_info.value)
        assert "N_d=4" in message
        assert "K=2" in message
        assert "5" in message  # 窗口起点 = iteration 5

    def test_reexport_rejects_non_step_and_out_of_window(self) -> None:
        run = self._run()
        with pytest.raises(ValueError, match="判别器步"):
            run.reexport(step_iteration=7, resume_iteration=6)
        with pytest.raises(ValueError, match="窗口"):
            run.reexport(step_iteration=8, resume_iteration=9)

    def test_state_adopt_roundtrip_restores_auc_bridge(self) -> None:
        """AUC 桥接账自持（#265 / ADR-0014）：record_auc 入账 → state()
        落盘形态 → 新账簿 adopt 回填 → window_auc 逐条件复原（跨窗口
        持久账的恢复面）。"""
        run = self._run()
        run.record_auc({"a": 0.8, "b": 0.3})
        fresh = self._run()
        fresh.adopt(run.state())
        assert fresh.window_auc("a") == 0.8
        assert fresh.window_auc("b") == 0.3
        assert run.state() == {"a": 0.8, "b": 0.3}

    def test_adopt_empty_state_is_empty_bridge(self) -> None:
        """旧分片缺 window_auc 键的恢复面缺省 = 空账（#265 payload 裁决：
        v12 追加键 optional-additive——缺席分片按空账复原，N_d=1 语义
        逐位不变）。"""
        run = self._run()
        run.adopt({})
        assert run.window_auc("a") is None

    def test_adopt_rejects_malformed_state(self) -> None:
        run = self._run()
        for malformed in (
            ["a"],                    # 非 dict
            {"a": True},              # bool 值（isinstance(int) 的漏网形态）
            {"a": float("nan")},      # 非有限浮点
            {1: 0.5},                 # 非 str 条件名
        ):
            with pytest.raises(ValueError):
                run.adopt(malformed)

    def test_clear_keeps_auc_bridge(self) -> None:
        """判别器步末清任务账不清 AUC 账（#220 决议 14 跨窗口持久的
        恢复前提——桥接账的存续面）。"""
        run = self._run()
        run.record_auc({"a": 0.7})
        run.clear()
        assert run.window_auc("a") == 0.7


class _RecorderScorer:
    """patch_logits 调用序记录器（相位序列锚的消费面）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, bool]] = []
        self._disc = torch.nn.Conv3d(1, 1, 1)

    @property
    def discriminator(self):
        return self._disc

    def patch_logits(self, latents):
        self.calls.append(("fwd", latents.shape[0], torch.is_grad_enabled()))
        # 输出卷入判别器参数（backward 需要 grad_fn——真实 scorer 的
        # 前向经网络权重，替身临界行为保持一致）
        return (
            latents.sum(dim=(2, 3, 4), keepdim=True)
            * self._disc.weight
            + self._disc.bias.reshape(1, -1, 1, 1, 1)
        )

    def discriminator_terms(self, logits_real, logits_fake):
        real_term = ((logits_real - 1.0) ** 2).mean()
        fake_term = (logits_fake ** 2).mean()
        return LsganTerms(real_term + fake_term, real_term, fake_term)


class TestDiscriminatorPhase:
    """判别器步相位编排（#220 决议 4/8）：先全桶 eval 后全桶 train、
    逐桶等权全局 mean（loss×n_b/N_total → backward → allreduce SUM）、
    步末 u/v broadcast 卡 0 权威（多卡面；单卡结构性退化）。"""

    def _buckets(self):
        first = DiscriminatorBucket.assemble(
            "a", [_record("a", fill=0.2), _record("a", fill=0.4)], torch.device("cpu"),
        )
        second = DiscriminatorBucket.assemble(
            "b", [_record("b", fill=0.6)], torch.device("cpu"),
        )
        return [first, second]

    def test_eval_phase_precedes_all_train_forwards(self) -> None:
        """相位序列锚：全部 eval 前向（no_grad）先于全部 train 前向
        （grad）——同一参数快照的 acc 复算不被更新前向污染。"""
        scorer = _RecorderScorer()
        phase = DiscriminatorPhase(scorer, torch.optim.SGD(scorer.discriminator.parameters(), lr=0.0), total_pairs=3)
        phase.accumulate(self._buckets())
        grad_flags = [flag for _, _, flag in scorer.calls]
        assert False in grad_flags and True in grad_flags
        first_true = grad_flags.index(True)
        assert all(flag is False for flag in grad_flags[:first_true]), (
            "eval 段（no_grad）必须完整先于 train 段（grad）——逐桶交替否决"
        )
        # 每桶 2 次 eval 前向 + 2 次 train 前向（real/fake 各一）
        assert len(scorer.calls) == 8

    def test_per_pair_equal_weighting(self) -> None:
        """逐对等权全局 mean（#220 决议 8 显式新裁）：N_total=3、
        桶 a 2 对 / 桶 b 1 对——backward 的 loss = loss_a×(2/3) +
        loss_b×(1/3)。以 lr=1 的 SGD 单参更新反推实际 backward 的
        加权 loss（加权系数错即分道）。"""
        torch.manual_seed(0)
        scorer = _RecorderScorer()
        optimizer = torch.optim.SGD(scorer.discriminator.parameters(), lr=1.0)
        phase = DiscriminatorPhase(scorer, optimizer, total_pairs=3)
        report = phase.accumulate(self._buckets())
        # 手工复算：每桶未缩放 loss（复用 scorer 前向，参数未 step）
        expected = 0.0
        for bucket, detail in zip(self._buckets(), report.conditions):
            logits_real = scorer.patch_logits(bucket.reals)
            logits_fake = scorer.patch_logits(bucket.fakes)
            real_term = ((logits_real - 1.0) ** 2).mean()
            fake_term = (logits_fake ** 2).mean()
            assert detail.loss_discriminator == pytest.approx(
                float(real_term + fake_term), abs=1e-6,
            )
            expected += float(real_term + fake_term) * (detail.pair_count / 3)
        assert report.loss_discriminator == pytest.approx(expected, abs=1e-6)

    def test_report_is_per_condition_with_global_fields(self):
        """UpdateReport 升格（#220 决议 11）：per-condition 明细（桶序
        恒条件名排序）、batch_size = 本卡 Σ桶对数、全局量 N_total。"""
        scorer = _RecorderScorer()
        phase = DiscriminatorPhase(
            scorer, torch.optim.SGD(scorer.discriminator.parameters(), lr=0.0),
            total_pairs=3,
        )
        report = phase.accumulate(self._buckets())
        assert [d.condition for d in report.conditions] == ["a", "b"]
        assert report.batch_size == 3  # 本卡 Σ桶对数
        assert report.global_batch_size == 3  # N_total（单卡：本卡即全局）
        assert all(0.0 <= d.train_pairwise_acc <= 1.0 for d in report.conditions)
        assert all(math.isfinite(d.loss_discriminator) for d in report.conditions)

    def test_single_card_gradients_are_global(self):
        """单卡结构性退化：无跨卡归约面，本卡加权 backward 的梯度
        即全局 mean 的梯度（backward 后 grad 非空）；step 段后判别器
        回 eval 相。"""
        scorer = _RecorderScorer()
        phase = DiscriminatorPhase(
            scorer, torch.optim.SGD(scorer.discriminator.parameters(), lr=0.0),
            total_pairs=3,
        )
        phase.accumulate(self._buckets())
        assert all(
            parameter.grad is not None
            for parameter in scorer.discriminator.parameters()
        )
        assert scorer.discriminator.training is True  # accumulate 止于 train 相
        phase.step()
        assert scorer.discriminator.training is False  # step 末 eval 归位

    def test_graceful_no_spectral_buffers_single_card(self):
        """无 spectral norm 的判别器：u/v broadcast 面无 buffer 可同步
        ——结构性跳过，不空发集合调用。"""
        scorer = _RecorderScorer()
        phase = DiscriminatorPhase(
            scorer, torch.optim.SGD(scorer.discriminator.parameters(), lr=0.0),
            total_pairs=3,
        )
        phase.accumulate(self._buckets())
        assert phase.spectral_buffers() == {}
        DiscriminatorPhase.reduce_gradients([phase])  # 单卡：跳过不抛
        DiscriminatorPhase.synchronize_spectral([phase])
