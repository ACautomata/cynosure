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
- UpdateReport 升格（决议 11）：per-condition 明细 + batch_size = 本卡
  K + 全局量 N_total；单值字段 property 化（单桶派生、多桶拒绝）。
"""

import math

import pytest
import torch

from cynosure.reward.artifacts import LatentManifest, PoolEntry
from cynosure.reward.scorer import LsganTerms
from cynosure.train.allocation import AllocationTable
from cynosure.train.discriminator import (
    DiscriminatorBucket,
    DiscriminatorPhase,
    DiscriminatorWindow,
    WindowPairRecord,
    WindowRealDraw,
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
