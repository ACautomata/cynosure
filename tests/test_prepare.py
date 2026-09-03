"""prepare 数据工件管线测试：Real sample pool / Held-out real / per-channel
统计量三工件（ticket #18）。

测试原则（spec「Testing Decisions」）：经唯一 CLI seam 驱动 fixture 合成数据
端到端，只断言工件契约的外部行为——序列分层、病例级不相交、可装载、幂等。"""

from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES, ConfigLoader
from cynosure.fixtures import Fixture
from cynosure.reward import ChannelStats, LatentManifest
from tests.conftest import CliSession, CliResult, SyntheticBratsDataset

# fixture 影像体：fixture latent [4,16,16,8] 的 4× 空间上采样
FIXTURE_SERIES_SHAPE = (64, 64, 32)
# 病例级 70/10/20（experiment-design）：20 病例 → 14 train / 2 val / 4 test
NUM_CASES = 20
TRAIN_CASES = 14
VAL_CASES = 2
TEST_CASES = 4


class PrepareScenario:
    """一次 prepare 端到端场景：合成 BraTS 数据集 + fixture config + CLI 调用。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.config_path = tmp_path / "config.json"
        self._tmp_path = tmp_path

    def dataset_case_ids(self, count: int) -> list[str]:
        return [f"BraTS-GLI-{index:05d}-000" for index in range(count)]

    def run(self, *, num_cases: int = NUM_CASES, seed: int = 0) -> CliResult:
        """重建场景（数据集 + config）并经 CLI 执行 prepare。"""
        config = Fixture().config(self._tmp_path / "fixtures")
        config.schedule.seed = seed  # fixture 样板钉 seed=0；本场景显式覆写
        SyntheticBratsDataset(
            config.artifacts.dataset_root,
            self.dataset_case_ids(num_cases),
            FIXTURE_SERIES_SHAPE,
            seed,
        ).write()
        self._write_config(config)
        return self.cli.run("prepare", "--config", str(self.config_path))

    def run_without_dataset(self) -> CliResult:
        """只写 config、不建数据集（dataset_root 缺失场景）。"""
        self._write_config(Fixture().config(self._tmp_path / "fixtures"))
        return self.cli.run("prepare", "--config", str(self.config_path))

    def _write_config(self, config) -> None:
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )

    def config(self):
        return ConfigLoader.load(self.config_path)

    def load_pool(self) -> LatentManifest:
        return LatentManifest.load(
            self.config().reward.real_pool_manifest, kind="real_pool",
        )

    def load_heldout(self) -> LatentManifest:
        return LatentManifest.load(
            self.config().reward.heldout_real_manifest, kind="heldout_real",
        )

    def load_stats(self) -> ChannelStats:
        return ChannelStats.load(self.config().reward.channel_stats_json)


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> PrepareScenario:
    return PrepareScenario(cli, tmp_path)


class TestPrepareArtifacts:
    def test_writes_three_artifacts(self, scenario: PrepareScenario) -> None:
        result = scenario.run()
        assert result.code == 0
        config = scenario.config()
        assert config.reward.real_pool_manifest.is_file()
        assert config.reward.heldout_real_manifest.is_file()
        assert config.reward.channel_stats_json.is_file()

    def test_stdout_reports_artifacts_and_split(
        self, scenario: PrepareScenario,
    ) -> None:
        result = scenario.run()
        assert result.code == 0
        assert "real_pool.json" in result.stdout
        assert "heldout_real.json" in result.stdout
        assert "channel_stats.json" in result.stdout
        # 病例级 split 报告：20 病例 → 14 / 2 / 4
        assert "train 14 / val 2 / test 4" in result.stdout

    def test_pool_is_stratified_by_modality(
        self, scenario: PrepareScenario,
    ) -> None:
        """按序列分层（AC）：每序列条目数一致、条目按（序列、病例）双键排序。"""
        assert scenario.run().code == 0
        pool = scenario.load_pool()
        assert set(pool.modalities) == set(MODALITIES)
        assert all(count == TRAIN_CASES for count in pool.modalities.values())
        order = {m: i for i, m in enumerate(MODALITIES)}
        keys = [(order[e.modality], e.case_id) for e in pool.entries]
        assert keys == sorted(keys)
        assert len(pool.entries) == TRAIN_CASES * 4

    def test_heldout_is_stratified_by_modality(
        self, scenario: PrepareScenario,
    ) -> None:
        """Held-out real 同契约按序列分层（spec：held-out 集「按序列分层」）。"""
        assert scenario.run().code == 0
        heldout = scenario.load_heldout()
        assert set(heldout.modalities) == set(MODALITIES)
        assert all(count == VAL_CASES for count in heldout.modalities.values())
        assert len(heldout.entries) == VAL_CASES * 4


class TestPrepareSplit:
    def test_pool_and_heldout_are_case_disjoint(
        self, scenario: PrepareScenario,
    ) -> None:
        """Held-out real 与 Real sample pool 病例级不相交（AC，spec：
        held-out 永不参与判别器更新、保证 AUC 是 out-of-sample 信号）。"""
        assert scenario.run().code == 0
        pool_cases = {e.case_id for e in scenario.load_pool().entries}
        heldout_cases = {e.case_id for e in scenario.load_heldout().entries}
        assert pool_cases & heldout_cases == set()

    def test_case_split_proportions(self, scenario: PrepareScenario) -> None:
        """病例级 70/10/20：20 病例 → 14 train / 2 val / 4 test；
        test split 不进任何工件。"""
        assert scenario.run().code == 0
        pool = scenario.load_pool()
        heldout = scenario.load_heldout()
        expected_sizes = {"train": TRAIN_CASES, "val": VAL_CASES, "test": TEST_CASES}
        assert pool.split_sizes == expected_sizes
        assert heldout.split_sizes == expected_sizes
        pool_cases = {e.case_id for e in pool.entries}
        heldout_cases = {e.case_id for e in heldout.entries}
        all_cases = set(scenario.dataset_case_ids(NUM_CASES))
        assert len(all_cases - pool_cases - heldout_cases) == TEST_CASES

    def test_split_is_seed_deterministic(self, scenario: PrepareScenario) -> None:
        """同 seed 重跑（全新数据集同内容）：split 结果一致（manifest 记录 seed）。"""
        assert scenario.run(seed=3).code == 0
        first = scenario.load_pool()
        assert scenario.run(seed=3).code == 0
        second = scenario.load_pool()
        assert first.split_seed == second.split_seed == 3
        assert [e.case_id for e in first.entries] == [
            e.case_id for e in second.entries
        ]


class TestPrepareContract:
    """产物工件契约（spec「产物工件契约」）：字段最小集，train/eval 可装载。"""

    def test_manifests_load_with_declared_kind(
        self, scenario: PrepareScenario,
    ) -> None:
        """kind 契约：train 侧拿 held-out manifest 当 pool 用会被拒绝——
        held-out 永不参与判别器更新在装载层即守住。"""
        assert scenario.run().code == 0
        config = scenario.config()
        with pytest.raises(ValueError, match="heldout_real"):
            LatentManifest.load(config.reward.real_pool_manifest, kind="heldout_real")
        with pytest.raises(ValueError, match="real_pool"):
            LatentManifest.load(
                config.reward.heldout_real_manifest, kind="real_pool",
            )

    def test_entries_load_as_configured_latent_shape(
        self, scenario: PrepareScenario,
    ) -> None:
        assert scenario.run().code == 0
        config = scenario.config()
        pool = scenario.load_pool()
        assert pool.latent_shape == config.latent_shape
        for entry in pool.entries[:4]:  # 抽样装载（train/eval 消费面）
            latent = pool.load_latent(entry)
            assert tuple(latent.shape) == config.latent_shape
            assert latent.dtype == torch.float32

    def test_manifest_records_encoder_provenance(
        self, scenario: PrepareScenario,
    ) -> None:
        """工件自证编码来源：消费方可区分 fixture 合成与（将来的）生产预编码。"""
        assert scenario.run().code == 0
        assert scenario.load_pool().encoder == "synthetic"
        assert scenario.load_heldout().encoder == "synthetic"

    def test_channel_stats_match_pool_numerics(
        self, scenario: PrepareScenario,
    ) -> None:
        """统计量 = Real sample pool 所用训练集的全量 per-channel mean/std。"""
        assert scenario.run().code == 0
        pool = scenario.load_pool()
        stats = scenario.load_stats()
        latents = torch.stack([pool.load_latent(e) for e in pool.entries])
        expected_mean = latents.mean(dim=(0, 2, 3, 4))
        expected_std = latents.std(dim=(0, 2, 3, 4), correction=0)
        assert len(stats.mean) == len(stats.std) == 4
        for channel in range(4):
            assert stats.mean[channel] == pytest.approx(
                expected_mean[channel].item(), abs=1e-6,
            )
            assert stats.std[channel] == pytest.approx(
                expected_std[channel].item(), abs=1e-6,
            )
        assert stats.num_latents == len(pool.entries)
        assert stats.latent_shape == pool.latent_shape


class TestPrepareIdempotency:
    def test_rerun_is_drift_free(self, scenario: PrepareScenario) -> None:
        """prepare 幂等（AC）：重跑不产生工件漂移——JSON 字节相等、latent 内容相等。"""
        assert scenario.run().code == 0
        config = scenario.config()
        artifact_paths = (
            config.reward.real_pool_manifest,
            config.reward.heldout_real_manifest,
            config.reward.channel_stats_json,
        )
        first_json = {p.name: p.read_bytes() for p in artifact_paths}
        first_pool = scenario.load_pool()
        first_latents = [first_pool.load_latent(e).clone() for e in first_pool.entries]
        assert scenario.run().code == 0  # 重跑
        second_json = {p.name: p.read_bytes() for p in artifact_paths}
        assert first_json == second_json
        second_pool = scenario.load_pool()
        for first, entry in zip(first_latents, second_pool.entries):
            assert torch.equal(first, second_pool.load_latent(entry))

    def test_dataset_change_leaves_no_orphan_entries(
        self, scenario: PrepareScenario, cli: CliSession,
    ) -> None:
        """latent 子树随每次运行整体重建：病例删除后重跑，盘上不留孤儿条目。"""
        assert scenario.run().code == 0
        config = scenario.config()
        dataset_root = config.artifacts.dataset_root
        removed_case = scenario.dataset_case_ids(NUM_CASES)[0]
        removed = dataset_root / removed_case
        for series_file in removed.iterdir():
            series_file.unlink()
        removed.rmdir()
        result = cli.run("prepare", "--config", str(scenario.config_path))
        assert result.code == 0
        pool = scenario.load_pool()
        assert all(e.case_id != removed_case for e in pool.entries)
        latent_root = config.reward.real_pool_manifest.parent / "real_pool_latents"
        on_disk = {
            p.relative_to(latent_root.parent).as_posix()
            for p in latent_root.rglob("*.pt")
        }
        assert on_disk == {e.latent for e in pool.entries}

    def test_failed_rerun_leaves_no_stale_manifest(
        self, scenario: PrepareScenario, cli: CliSession,
    ) -> None:
        """失败重跑显式失效旧工件：盘上要么全量一致、要么明确缺失，
        不留「manifest 索引指向已删 latent」的悬挂状态。"""
        assert scenario.run().code == 0
        config = scenario.config()
        case_dir = config.artifacts.dataset_root / scenario.dataset_case_ids(1)[0]
        nifti = next(case_dir.glob("*-t1n.nii.gz"))
        nifti.write_bytes(b"not a nifti")  # 损坏一序列：重跑编码失败
        result = cli.run("prepare", "--config", str(scenario.config_path))
        assert result.code == 2
        assert "影像读取失败" in result.stderr
        assert not config.reward.real_pool_manifest.exists()
        assert not config.reward.heldout_real_manifest.exists()
        assert not config.reward.channel_stats_json.exists()


class TestPrepareInputGuard:
    """prepare 输入契约：坏数据显式失败，不静默产出残缺工件。"""

    def test_missing_series_case_rejected(self, scenario: PrepareScenario) -> None:
        assert scenario.run().code == 0
        config = scenario.config()
        # 病例缺 t2f 序列后重跑：整条管线显式失败
        orphan = config.artifacts.dataset_root / "BraTS-GLI-00003-000"
        (orphan / "BraTS-GLI-00003-000-t2f.nii.gz").unlink()
        result = scenario.cli.run("prepare", "--config", str(scenario.config_path))
        assert result.code == 2
        assert "BraTS-GLI-00003-000" in result.stderr

    def test_dataset_root_missing_rejected(self, scenario: PrepareScenario) -> None:
        result = scenario.run_without_dataset()
        assert result.code == 2
        assert "dataset" in result.stderr

    def test_too_few_cases_rejected(self, scenario: PrepareScenario) -> None:
        """病例切不出非空 val split 时显式失败（held-out 为空即失去
        out-of-sample 信号语义）。"""
        result = scenario.run(num_cases=5)
        assert result.code == 2
        assert "val" in result.stderr
