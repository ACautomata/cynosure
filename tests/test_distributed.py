"""分布式执行层残余的单进程单元面（#238 第三批删除面收口后）。

torchrun 多进程 spawn 档测试已随归并器（EventMerger）与判别器副本
（ReplicatedDiscriminator）本体退役退役——旧执行序的分布式语义（FSDP
归并/allreduce/多 rank 续训 roundtrip）随切换期整删，CPU fixture 档的
新执行序覆盖见 test_async_executor。本文件保留**本体保留面**的单元
语义：进程组 Facade 的单进程退化与 seed 派生、Real sample pool 条带
切片（RankSlicedPool——旧执行序衰减窗口内仍在役）。
"""

import json
import os
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES
from cynosure.distributed import DistributedContext, RankSlicedPool
from cynosure.reward.artifacts import LatentManifest


class TestDistributedContextUnit:
    """进程组 Facade 的单进程退化与 seed 派生语义。"""

    def teardown_method(self) -> None:
        os.environ.pop("RANK", None)
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("LOCAL_RANK", None)

    def test_bootstrap_without_env_is_single_process(self) -> None:
        context = DistributedContext.bootstrap()
        assert context.rank == 0
        assert context.world_size == 1
        assert not context.distributed
        # gather 恒等（单进程退化：本体保留面在 world-1 的行为）
        assert context.gather([{"rank": 0}]) == [[{"rank": 0}]]
        context.destroy()

    def test_derived_seed_keeps_rank0_identity(self) -> None:
        context = DistributedContext.bootstrap()
        assert context.derive_seed(7) == 7  # 等价性前提：rank 0 恒等偏移
        context.destroy()

    def test_broadcast_flag_and_local_device_degenerate_on_single_process(self) -> None:
        """world-1 恒等：broadcast_flag 原样返回传入值（不构造张量）、
        local_device 回落 CPU。CUDA 下的 cuda:LOCAL_RANK 绑定与 NCCL 的
        广播张量设备正确性属集群 torchrun 冒烟门槛（本机 CPU fixture 只
        锁退化语义与代码路径收敛）。"""
        context = DistributedContext.bootstrap()
        assert context.broadcast_flag(True) is True
        assert context.broadcast_flag(False) is False
        assert context.local_device().type == (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        context.destroy()

    def test_pg_timeout_env_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """watchdog 超时环境变量：未设置 = None（不传参、torch 默认）、
        正整数 = timedelta 分钟、非法值显式拒绝（部署侧拼写错误静默回落
        默认会让 SothisAI 平台的长 watchdog 预期失效）。"""
        from datetime import timedelta

        monkeypatch.delenv("CYNOSURE_PG_TIMEOUT_MIN", raising=False)
        assert DistributedContext._pg_timeout() is None
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "40")
        assert DistributedContext._pg_timeout() == timedelta(minutes=40)
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "0")
        with pytest.raises(ValueError, match="CYNOSURE_PG_TIMEOUT_MIN"):
            DistributedContext._pg_timeout()
        monkeypatch.setenv("CYNOSURE_PG_TIMEOUT_MIN", "abc")
        with pytest.raises(ValueError):
            DistributedContext._pg_timeout()


class TestPoolSliceUnit:
    """Real sample pool 的 rank 切片语义（条带切片 + 分层保持）。"""

    @pytest.fixture
    def manifest(self, tmp_path: Path) -> LatentManifest:
        """8 条目 × 四序列交替的池（load 装载形态：条目路径存在性不作要求）。"""
        path = tmp_path / "real_pool.json"
        entries = [
            {
                "case_id": f"case-{index}",
                "modality": ["t1n", "t1c", "t2w", "t2f"][index % 4],
                "latent": f"latent-{index}.pt",
                # spacing 侧车（issue #46 契约必填）：BraTS 1mm iso 的
                # header zooms ×1e2（切片语义不消费取值，仅须通过装载校验）
                "spacing": [100.0, 100.0, 100.0],
            }
            for index in range(8)
        ]
        path.write_text(json.dumps({
            "kind": "real_pool",
            "encoder": "fixture",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"train": 8, "val": 0, "test": 0},
            "entries": entries,
        }), encoding="utf-8")
        return LatentManifest.load(path, kind="real_pool")

    def test_stripe_slice_keeps_alternating_layers(self, manifest: LatentManifest) -> None:
        context = DistributedContext(0, 1, False)
        sliced = RankSlicedPool(manifest, context, MODALITIES).view()
        assert sliced.kind == manifest.kind
        assert sliced.entries == manifest.entries  # world=1 恒等

    def test_stripe_slice_distributes_entries_by_rank(self, manifest: LatentManifest) -> None:
        """分层条带切片：每序列内部 entries[rank::world]，各片覆盖全部序列。"""
        context = DistributedContext(1, 2, True)  # 不 init 进程组的纯切片语义
        sliced = RankSlicedPool(manifest, context, MODALITIES).view()
        # t1n=[0,4] t1c=[1,5] t2w=[2,6] t2f=[3,7] → rank 1 取各序列第 2 条
        assert [entry.case_id for entry in sliced.entries] == [
            "case-4", "case-5", "case-6", "case-7",
        ]
        assert set(entry.modality for entry in sliced.entries) == {"t1n", "t1c", "t2w", "t2f"}
        assert sliced.modalities == {m: 1 for m in ("t1n", "t1c", "t2w", "t2f")}

    def test_slice_rejects_empty_modality_band(self, manifest: LatentManifest) -> None:
        """切片后某序列条目归零 = 判别器 real 侧断供，装配期显式拒绝。"""
        context = DistributedContext(0, 4, True)  # 4 路切片 × 每序列仅 2 条
        starved = manifest.model_copy(deep=True)
        starved.entries = starved.entries[:2]  # 只剩 t1n/t1c 两序列
        with pytest.raises(ValueError, match="t2w|t2f|序列"):
            RankSlicedPool(starved, context, MODALITIES).view()

    def test_insufficient_pool_rejected_consistently_on_every_rank(
        self, manifest: LatentManifest,
    ) -> None:
        """pool 不足的拒绝必须全 rank 一致：校验消费**切片前**的 full
        manifest（每 rank 对同一全量判定同一结果）。按切片后本地视图校验
        时 rank 间可见性不同（序列仅 1 条时 rank 0 满额通过、高 rank 条带
        为空才拒绝）——失败方单方面退出装配、其余 rank 进入集合操作互等
        （连接错误/挂死），而非全 rank 一致的输入拒绝。"""
        starved = manifest.model_copy(deep=True)
        starved.entries = starved.entries[:4]  # 每序列恰好 1 条
        for rank in range(2):
            with pytest.raises(ValueError, match="不足"):
                RankSlicedPool(
                    starved, DistributedContext(rank, 2, True), MODALITIES,
                ).view()


