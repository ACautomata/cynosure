"""async 执行序的续训存取锚（#236 续训与事件契约期；#222/#218 结票口径）。

单元锚（无标记，纯 CPU、合成 payload、不经 fixture 库）：v12 单文件
分片的契约校验矩阵——版本拒旧（旧执行序 v11 分片被新 store 拒）、
nominal-v12 全键清单锚、拓扑守卫只对账 slots（卡数不进对账）、
per-(槽×流) generators 嵌套（旧扁平形态二次拒绝面）、lr/seeds 形态、
config 漂移白名单仅 max_iterations、单文件 tmp + ``os.replace`` 原子写。
跨执行器拒绝的另一半（新 v12 分片被旧 resume 拒）在
test_resume_roundtrip 的 trainer 场景承载。

`AsyncResumeStore` 是存取与校验单点：payload 的攒装（门面收集各卡）
与应用（下发各卡）在门面侧——本文件只锚 store 的输入输出契约。
"""

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.train.artifacts import RunArtifacts
from cynosure.train.async_resume import AsyncResumeStore
from cynosure.train.rng import SLOT_SEED_STRIDE, TrainingRngStreams

STREAMS = (
    TrainingRngStreams.ROLLOUT,
    TrainingRngStreams.REAL_POOL,
    TrainingRngStreams.HELDOUT_AUC,
    TrainingRngStreams.RECON,
)

SLOTS = 2


def slot_generators() -> dict:
    return {
        f"slot{slot}": {
            stream: torch.randint(0, 255, (16,), dtype=torch.uint8)
            for stream in STREAMS
        }
        for slot in range(SLOTS)
    }


def make_payload(**overrides) -> dict:
    """合成 v12 分片（键集 = #222 终稿清单；小随机张量代替权重）。"""
    base = {
        "version": 12,
        "iteration": 3,
        "slots": SLOTS,
        "policy_network": {"policy.weight": torch.randn(2, 2)},
        "policy_optimizer": {
            "state": {0: {"step": 3, "exp_avg": torch.randn(2, 2)}},
            "param_groups": [{"lr": 1e-5}],
        },
        "discriminator_network": {"disc.weight": torch.randn(3)},
        "discriminator_optimizer": {
            "state": {0: {"step": 2}},
            "param_groups": [{"lr": 2e-4}],
        },
        "lr": {"policy": 1e-5, "discriminator": 2e-4},
        "generators": slot_generators(),
        "overfit": {"ema": {"t1n": {"value": 0.125, "count": 4}}},
        "seeds": {
            "base": 0,
            "per_slot": [slot * SLOT_SEED_STRIDE for slot in range(SLOTS)],
        },
    }
    base.update(overrides)
    return base


@pytest.fixture
def run_artifacts(tmp_path, valid_config_dict) -> RunArtifacts:
    config = CynosureConfig.model_validate(valid_config_dict)
    return RunArtifacts.init(config, tmp_path / "run")


@pytest.fixture
def store(run_artifacts):
    return AsyncResumeStore(run_artifacts)


@pytest.fixture
def config(valid_config_dict) -> CynosureConfig:
    return CynosureConfig.model_validate(valid_config_dict)


class TestPayloadContract:
    """v12 nominal 全键清单 + 形态校验矩阵。"""

    def test_required_keys_match_v12_bill(self, store, config) -> None:
        """#222 终稿汇总清单一次成文：多一键少一键都须显式决策——
        键集是 payload 契约的锚。"""
        store.save(make_payload())
        restored = store.restore(config, slot_count=SLOTS)
        assert set(restored) == {
            "version", "iteration", "slots",
            "policy_network", "policy_optimizer",
            "discriminator_network", "discriminator_optimizer",
            "lr", "generators", "overfit", "seeds",
        }

    def test_v12_bill_excludes_retired_keys(self, store, config) -> None:
        """删除面锚（#218 贡献清单 + #222 汇总）：全局 RNG（rng）/
        门控（gating，v11 已删）/ world_size（→ slots）/ ema 预留槽
        随 v12 整体消失。"""
        store.save(make_payload())
        restored = store.restore(config, slot_count=SLOTS)
        for retired in ("rng", "gating", "world_size", "ema"):
            assert retired not in restored

    def test_roundtrip_self_read_write(self, store, config) -> None:
        """v12 自写自读（AC 1）：分片逐位回读——张量逐位、字段全等。"""
        payload = make_payload()
        store.save(payload)
        restored = store.restore(config, slot_count=SLOTS)
        assert restored["version"] == 12
        assert restored["iteration"] == 3
        assert restored["slots"] == SLOTS
        for key in ("policy_network", "discriminator_network"):
            for name in payload[key]:
                assert torch.equal(restored[key][name], payload[key][name])
        assert restored["generators"].keys() == payload["generators"].keys()
        for slot_key in payload["generators"]:
            for stream in STREAMS:
                assert torch.equal(
                    restored["generators"][slot_key][stream],
                    payload["generators"][slot_key][stream],
                )
        assert restored["overfit"] == payload["overfit"]
        assert restored["seeds"] == payload["seeds"]
        assert restored["lr"] == payload["lr"]

    def test_atomic_write_leaves_no_tmp(self, store, config, tmp_path) -> None:
        """单文件原子替换（tmp + os.replace，#222）：落盘完成即无
        tmp 残留（崩溃窗口外只见完整分片）。"""
        store.save(make_payload())
        checkpoints = tmp_path / "run" / "checkpoints"
        assert store.path() == checkpoints / "resume_state.pt"
        assert store.path().is_file()
        assert not (checkpoints / "resume_state.pt.tmp").exists()

    def test_missing_shard_fails_cleanly(self, store, config) -> None:
        with pytest.raises(FileNotFoundError, match="续训状态"):
            store.restore(config, slot_count=SLOTS)

    def test_corrupt_shard_rejected(self, store, config) -> None:
        store.path().write_bytes(b"not a torch payload")
        with pytest.raises(ValueError, match="不可读"):
            store.restore(config, slot_count=SLOTS)

    def test_to_cpu_snapshot_handles_optimizer_state(self) -> None:
        """optimizer state_dict 的嵌套形态深拷贝（#236 实证修补：
        param_groups 的 ``params`` 是 int 索引列表、state 的 ``step``
        是 int——容器递归只下钻 dict/list，标量叶原样保留）。"""
        nested = {
            "state": {0: {"step": 3, "exp_avg": torch.randn(2, 2)}},
            "param_groups": [{"lr": 1e-5, "params": [0, 1]}],
        }
        snapshot = AsyncResumeStore.to_cpu_snapshot(nested)
        assert snapshot["state"][0]["step"] == 3
        assert snapshot["param_groups"][0]["params"] == [0, 1]
        assert torch.equal(snapshot["state"][0]["exp_avg"], nested["state"][0]["exp_avg"])
        assert snapshot["state"][0]["exp_avg"] is not nested["state"][0]["exp_avg"]

    def test_to_cpu_snapshot_clones_tensor_leaves_in_lists(self) -> None:
        """list 直含张量的形态也兑现深拷贝（#236 review 补漏）：快照
        后原地改源 tensor，快照不受影响（``.cpu()`` 零拷贝回传，须
        真 ``.clone()``）。"""
        original = {
            "params": [torch.arange(4)],
            "state": {"momentum": torch.zeros(2)},
        }
        snapshot = AsyncResumeStore.to_cpu_snapshot(original)
        original["params"][0].add_(1)
        original["state"]["momentum"].add_(1)
        assert torch.equal(snapshot["params"][0], torch.arange(4))
        assert torch.equal(snapshot["state"]["momentum"], torch.zeros(2))


class TestVersionAccounting:
    """版本对账：跨执行器拒绝由版本号承载（#222 §0/§1）。"""

    def test_legacy_v11_shard_rejected(self, store, config) -> None:
        """旧执行序（torchrun）v11 分片被新 store 拒：版本号 ≠ 12 即
        拒（「加厚期新 v12 分片被旧 resume 拒、旧 v11 分片被新 store
        拒」的双向拒绝之一）。分片经 ``torch.save`` 直写（模拟外部
        已存在的旧分片）——不经 store.save（攒装前置校验同源拒绝）。"""
        torch.save(make_payload(version=11), store.path())
        with pytest.raises(ValueError, match="v12"):
            store.restore(config, slot_count=SLOTS)

    def test_unknown_higher_version_rejected(self, store, config) -> None:
        """更高版本（未来代码写出）的分片同样被版本对账拒绝——经
        ``torch.save`` 直写（模拟外部已存在的分片），恢复入口拒。"""
        torch.save(make_payload(version=13), store.path())
        with pytest.raises(ValueError, match="格式版本"):
            store.restore(config, slot_count=SLOTS)

    def test_legacy_executor_key_shape_rejected(self, store, config) -> None:
        """旧执行序分片形态（``format_version`` 键、无 ``version``）在
        版本校验即拒——缺失键与版本不符同属「跨口径续训不可恢复」。"""
        store.save(make_payload())
        shard = torch.load(store.path(), weights_only=True)
        legacy = {k: v for k, v in shard.items() if k != "version"}
        legacy["format_version"] = 11
        torch.save(legacy, store.path())
        with pytest.raises(ValueError, match="格式版本"):
            store.restore(config, slot_count=SLOTS)


class TestTopologyGuard:
    """拓扑守卫 = payload 只对账 slots（协程数；#218「跨拓扑 resume
    拒绝的校验对象 = 协程数」）——卡数不进对账（四条 RNG 流全 CPU
    generator、权重每卡完整副本、分配表纯函数重导出，#222 §1）。"""

    def test_slot_count_mismatch_rejected(self, store, config) -> None:
        store.save(make_payload())
        with pytest.raises(ValueError, match="协程数"):
            store.restore(config, slot_count=SLOTS + 1)

    def test_same_slots_any_card_count_accepted(self, store, config) -> None:
        """slots 一致即放行、无卡数字段可对账（分片里根本没有
        world_size 键）——跨卡数恢复放行是决议明文，不是缺口。本测
        只承载结构面（对账面无卡数键可拒）；跨卡数变化的端到端行为
        锚在 TestResumeAcrossCardCount（slow+gpu 多卡档）。"""
        store.save(make_payload())
        assert store.restore(config, slot_count=SLOTS)["slots"] == SLOTS


class TestGeneratorsNesting:
    """per-(槽×流) 嵌套 generators（#218）——旧扁平形态二次拒绝面。"""

    def test_flat_generator_manifest_rejected(self, store, config) -> None:
        """旧执行序的扁平流名清单（无槽嵌套）被嵌套校验显式拒绝——
        跨执行器拒绝的版本号双保险（#222 §1）。"""
        torch.save(make_payload(generators={
            stream: torch.zeros(16, dtype=torch.uint8) for stream in STREAMS
        }), store.path())
        with pytest.raises(ValueError, match="嵌套|slot"):
            store.restore(config, slot_count=SLOTS)

    def test_unknown_slot_key_rejected(self, store, config) -> None:
        generators = slot_generators()
        generators["slot9"] = generators.pop("slot1")
        torch.save(make_payload(generators=generators), store.path())
        with pytest.raises(ValueError, match="slot"):
            store.restore(config, slot_count=SLOTS)

    def test_stream_manifest_mismatch_rejected(self, store, config) -> None:
        generators = slot_generators()
        del generators["slot0"][TrainingRngStreams.RECON]
        torch.save(make_payload(generators=generators), store.path())
        with pytest.raises(ValueError, match="流"):
            store.restore(config, slot_count=SLOTS)


class TestFieldValidation:
    """lr/seeds/iteration 形态：损坏分片显式拒绝，不静默漂移。"""

    def test_lr_slot_keys_rejected_when_mismatched(
        self, store, config,
    ) -> None:
        torch.save(
            make_payload(lr={"policy": 1e-5, "extra": 1.0}), store.path(),
        )
        with pytest.raises(ValueError, match="lr"):
            store.restore(config, slot_count=SLOTS)

    def test_seeds_shape_rejected_when_mismatched(
        self, store, config,
    ) -> None:
        torch.save(
            make_payload(seeds={"base": 0, "per_slot": [0]}), store.path(),
        )
        with pytest.raises(ValueError, match="seeds"):
            store.restore(config, slot_count=SLOTS)

    def test_negative_iteration_rejected(self, store, config) -> None:
        torch.save(make_payload(iteration=-1), store.path())
        with pytest.raises(ValueError, match="iteration"):
            store.restore(config, slot_count=SLOTS)


class TestConfigDriftGuard:
    """续训 config 漂移白名单仅 max_iterations（#222 清单沿袭）。"""

    def test_seed_drift_rejected(self, store, config) -> None:
        store.save(make_payload())
        drifted = config.model_copy(update={
            "schedule": config.schedule.model_copy(update={"seed": 99}),
        })
        with pytest.raises(ValueError, match="schedule.seed"):
            store.restore(drifted, slot_count=SLOTS)

    def test_max_iterations_drift_allowed(self, store, config) -> None:
        store.save(make_payload())
        extended = config.model_copy(update={
            "schedule": config.schedule.model_copy(
                update={"max_iterations": config.schedule.max_iterations + 10},
            ),
        })
        assert store.restore(extended, slot_count=SLOTS)["iteration"] == 3
