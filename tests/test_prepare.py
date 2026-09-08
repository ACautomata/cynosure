"""prepare 数据工件管线测试：Real sample pool / Held-out real / per-channel
统计量三工件（ticket #18）；生产 VAE 预编码策略与其分派点（T12）。

测试原则（spec「Testing Decisions」）：经唯一 CLI seam 驱动 fixture 合成数据
端到端，只断言工件契约的外部行为——序列分层、病例级不相交、可装载、幂等；
生产编码器单测用 fixture 微型 VAE 走同一 netbuild 严格装载契约。"""

import copy
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
import torch

from cynosure.config import CynosureConfig, MODALITIES, ConfigLoader
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward import (
    ChannelStats,
    LatentManifest,
    MaisiLatentEncoder,
    PreparePipeline,
    SyntheticLatentEncoder,
)
from cynosure.reward.preprocessing import SPACING_CONDITION_SCALE
from tests.conftest import (
    ANISOTROPIC_AFFINE,
    CliSession,
    CliResult,
    MINIMAL_CONFIG_DICT,
    SyntheticBratsDataset,
)

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


class TestPrepareSpacingSidecar:
    """spacing 侧车全链（issue #46）：prepare 逐 case 读 header zooms ×1e2
    写入 manifest 条目——夹具 NIfTI 的 header zooms 与 manifest 侧车值一致
    （×1e2 后），值是读出来的、不是写死常量。"""

    @staticmethod
    def header_zooms_x1e2(dataset_root: Path, entry) -> tuple[float, ...]:
        """条目对应 NIfTI 的 raw header zooms ×1e2（审计基准，独立于 prepare）。"""
        nifti = (
            dataset_root / entry.case_id
            / f"{entry.case_id}-{entry.modality}.nii.gz"
        )
        zooms = nib.load(nifti).header.get_zooms()[:3]
        return tuple(float(zoom) * SPACING_CONDITION_SCALE for zoom in zooms)

    def test_entries_carry_header_zooms_x1e2(
        self, scenario: PrepareScenario,
    ) -> None:
        """pool 与 held-out 的每条 manifest 条目携带该（病例, 序列）的
        header zooms ×1e2（可装载、可审计）。"""
        assert scenario.run().code == 0
        dataset_root = scenario.config().artifacts.dataset_root
        for manifest in (scenario.load_pool(), scenario.load_heldout()):
            assert len(manifest.entries) > 0
            for entry in manifest.entries:
                assert entry.spacing == self.header_zooms_x1e2(dataset_root, entry)

    def test_spacing_is_per_case_data_not_constant(
        self, scenario: PrepareScenario, cli: CliSession,
    ) -> None:
        """「来自数据」的判别性断言：把一个 t1n 换成各向异性 zooms 后重跑，
        该条目的 spacing 随数据变化、其余条目不受影响（写死常量必假绿）。"""
        assert scenario.run().code == 0
        config = scenario.config()
        target = next(
            entry for entry in scenario.load_pool().entries
            if entry.modality == "t1n"
        )
        nifti = (
            config.artifacts.dataset_root / target.case_id
            / f"{target.case_id}-t1n.nii.gz"
        )
        volume = np.random.default_rng(0).standard_normal(
            FIXTURE_SERIES_SHAPE,
        ).astype(np.float32)
        nib.save(nib.Nifti1Image(volume, ANISOTROPIC_AFFINE), nifti)
        assert cli.run("prepare", "--config", str(scenario.config_path)).code == 0
        for manifest in (scenario.load_pool(), scenario.load_heldout()):
            for entry in manifest.entries:
                expected = self.header_zooms_x1e2(
                    config.artifacts.dataset_root, entry,
                )
                assert entry.spacing == expected
                if (entry.case_id, entry.modality) == (target.case_id, "t1n"):
                    assert entry.spacing == (50.0, 100.0, 200.0)


class TestPrepareChainSemantics:
    """端到端走 transform 链（issue #45）：夹具影像携带真实方向语义
    （非单位 affine，LPS 为主、混入 RAS），方向重定向在端到端中真实发生；
    fixture 经 config 注入小 resize 基数，与生产共用同一条链逻辑。"""

    def test_dataset_carries_lps_dominant_affines(
        self, scenario: PrepareScenario,
    ) -> None:
        """夹具 affine 非单位：LPS 为主、确定性混入 RAS（方向步被真实驱动）。"""
        scenario.run()
        dataset_root = scenario.config().artifacts.dataset_root
        axcodes = [
            tuple(nib.aff2axcodes(
                nib.load(case_dir / f"{case_dir.name}-t1n.nii.gz").affine,
            ))
            for case_dir in sorted(dataset_root.iterdir())
        ]
        assert set(axcodes) == {("L", "P", "S"), ("R", "A", "S")}
        assert axcodes.count(("L", "P", "S")) > axcodes.count(("R", "A", "S"))

    def test_chain_encoded_latents_satisfy_contract(
        self, scenario: PrepareScenario,
    ) -> None:
        """全链（方向/强度/dtype/resize）真实执行后 latent 形状契约满足。"""
        assert scenario.run().code == 0
        pool = scenario.load_pool()
        config = scenario.config()
        assert len(pool.entries) == TRAIN_CASES * 4
        for entry in pool.entries:
            latent = pool.load_latent(entry)
            assert tuple(latent.shape) == config.latent_shape
            assert latent.dtype == torch.float32


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


class MeanPoolEncoderTwin:
    """AutoencoderKlMaisi.encode 的均值池化替身（4× 空间压缩语义）：记录每次
    encode 调用的输入形状（整前向豁免的运行时观测面），返回 (z_mu, z_sigma)
    二元组与真实 encode 同构。"""

    def __init__(self) -> None:
        self.encode_shapes: list[tuple[int, ...]] = []

    def to(self, device: torch.device) -> "MeanPoolEncoderTwin":
        return self

    def eval(self) -> "MeanPoolEncoderTwin":
        return self

    def encode(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.encode_shapes.append(tuple(batch.shape))
        latent = torch.nn.functional.avg_pool3d(
            batch.repeat(1, 4, 1, 1, 1), kernel_size=4, stride=4,
        )  # 单通道 → 4 latent 通道（VAE latent_channels=4 同构）
        return latent, torch.zeros_like(latent)


class TestMaisiLatentEncoder:
    """生产预编码器（fixture 微型 VAE 走同一 netbuild 严格装载契约）：
    z_mu 确定性与 raw 存储域是 prepare 幂等/缩放契约的根基。"""

    ENCODE_INPUT_SHAPE = (1, 64, 64, 32)
    LATENT_SHAPE = (4, 16, 16, 8)

    def artifact(self, tmp_path: Path) -> NetworkArtifact:
        directory = tmp_path / "fixture_artifacts"
        Fixture().write_artifacts(directory)
        return NetworkArtifact(
            config=NetworkAssembler.load_json(directory / "vae_config.json"),
            checkpoint=directory / "vae.pt",
        )

    def encoder(self, artifact: NetworkArtifact, **overrides) -> MaisiLatentEncoder:
        return MaisiLatentEncoder(artifact, torch.device("cpu"), **overrides)

    def test_shape_dtype_device_contract(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        encoder = self.encoder(self.artifact(tmp_path))
        latent = encoder.encode(torch.randn(self.ENCODE_INPUT_SHAPE))
        assert tuple(latent.shape) == self.LATENT_SHAPE
        assert latent.dtype == torch.float32
        assert latent.device == torch.device("cpu")

    def test_encode_is_bitwise_deterministic(self, tmp_path: Path) -> None:
        """seeded sampling z（内容寻址噪声种子）是 prepare 幂等的前提：
        同 checkpoint、同 noise_seed 三次编码逐位相等——无种子 RNG 或
        GPU 流的采样都会破坏重跑零漂移。"""
        torch.manual_seed(0)
        artifact = self.artifact(tmp_path)  # 工件只写一次：权重不随测试内 RNG 漂移
        image = torch.randn(self.ENCODE_INPUT_SHAPE)
        first = self.encoder(artifact).encode(image, noise_seed=42)
        second = self.encoder(artifact).encode(image, noise_seed=42)
        assert torch.equal(first, second)
        third = self.encoder(artifact).encode(image, noise_seed=42)
        assert torch.equal(first, third)  # 新实例、同一 checkpoint 重载

    def test_encode_stores_seeded_posterior_sample(self, tmp_path: Path) -> None:
        """存储域语义 = seeded sampling z（上游 create_training_data 同
        语义）：encoder 输出与裸网络 z_mu + eps(noise_seed)·z_sigma 逐位
        相等——z_mu 与 policy rollout 域是分布级错配（探针实测 raw z_mu
        std≈0.48 vs rollout≈0.94），判别器比较要求 real/fake 同为后验
        采样分布。"""
        torch.manual_seed(0)
        artifact = self.artifact(tmp_path)
        image = torch.randn(self.ENCODE_INPUT_SHAPE)
        latent = self.encoder(artifact).encode(image, noise_seed=7)
        # 对照面与被测面同 eval 相（廉价保险：未来 fixture 配置若引入
        # 相位敏感层，两侧仍同口径）
        raw_vae = NetworkAssembler.vae(artifact).eval()
        with torch.no_grad():
            z_mu, z_sigma = raw_vae.encode(image.unsqueeze(0))
            eps = torch.randn(
                z_mu.shape, generator=torch.Generator().manual_seed(7),
            )
            expected = (z_mu + eps * z_sigma).squeeze(0)
        assert torch.equal(latent, expected)
        # 内容寻址：不同噪声种子给不同后验样本（同 z_mu 基底）
        other = self.encoder(artifact).encode(image, noise_seed=8)
        assert not torch.equal(latent, other)

    def test_name_records_provenance(self, tmp_path: Path) -> None:
        encoder = self.encoder(self.artifact(tmp_path))
        assert encoder.name == "autoencoderkl_maisi"

    def test_output_detached_no_grad(self, tmp_path: Path) -> None:
        torch.manual_seed(0)
        encoder = self.encoder(self.artifact(tmp_path))
        latent = encoder.encode(torch.randn(self.ENCODE_INPUT_SHAPE))
        assert latent.requires_grad is False

    def test_input_contract_rejected(self, tmp_path: Path) -> None:
        encoder = self.encoder(self.artifact(tmp_path))
        with pytest.raises(ValueError, match="影像体须为"):
            encoder.encode(torch.randn(64, 64, 32))  # 缺通道维
        with pytest.raises(ValueError, match="影像体须为"):
            encoder.encode(torch.randn(2, 1, 64, 64, 32))  # 带 batch 维

    def test_whole_forward_within_threshold_large_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """官方 dynamic_infer 小体豁免语义：单样本元素数 ≤ 阈值元素数
        → 整前向（BraTS [1,1,256,256,128]=8.39M ≤ 影像空间阈值
        [320,320,160]，生产恒走此路）；超过 → 显式拒绝而非滑窗
        （MONAI SlidingWindowInferer 对下采样 encoder 拼合通道错乱，
        静默错误比显式失败危险）。"""
        artifact = self.artifact(tmp_path)
        image = torch.randn(self.ENCODE_INPUT_SHAPE)

        twin = MeanPoolEncoderTwin()
        monkeypatch.setattr(
            "cynosure.reward.encoder.NetworkAssembler",
            SimpleNamespace(vae=lambda unused_artifact: twin),
        )
        latent = self.encoder(artifact).encode(image)
        assert twin.encode_shapes == [(1, 1, 64, 64, 32)]  # 豁免：一次整前向
        assert tuple(latent.shape) == self.LATENT_SHAPE

        small_threshold = self.encoder(artifact, roi_size=(32, 32, 32))
        with pytest.raises(ValueError, match="整前向豁免阈值"):
            small_threshold.encode(image)
        assert len(twin.encode_shapes) == 1  # 拒绝路径不产生前向

    def test_invalid_roi_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="roi"):
            self.encoder(self.artifact(tmp_path), roi_size=(0, 320, 160))

    def test_checkpoint_mismatch_rejected(self, tmp_path: Path) -> None:
        """工件错配（VAE 网络配置对 UNet checkpoint）：严格装载契约在
        构造期显式失败，不静默产出错误编码。"""
        directory = tmp_path / "fixture_artifacts"
        Fixture().write_artifacts(directory)
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(directory / "vae_config.json"),
            checkpoint=directory / "unet.pt",
        )
        with pytest.raises(RuntimeError):
            MaisiLatentEncoder(artifact, torch.device("cpu"))


class TestPrepareEncoderDispatch:
    """编码器策略分派（PreparePipeline.build_encoder，Factory Method）：
    fixture 合成 vs 生产 VAE 预编码——CLI 保持薄，分派点直接可单测。"""

    def test_fixture_config_builds_synthetic(self, tmp_path: Path) -> None:
        config = Fixture().config(tmp_path / "fixtures")
        encoder = PreparePipeline.build_encoder(config, torch.device("cpu"))
        assert isinstance(encoder, SyntheticLatentEncoder)

    def test_production_config_builds_maisi(self, tmp_path: Path) -> None:
        """合法生产 config（fixture_mode=false）+ VAE 工件对 → 生产编码器
        （只装配验证分派与装载契约，不跑全量生产编码）。"""
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures")
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        data["artifacts"]["vae_ckpt"] = str(artifacts.vae_ckpt)
        data["artifacts"]["vae_config_json"] = str(artifacts.vae_config_json)
        config = CynosureConfig.model_validate(data)
        encoder = PreparePipeline.build_encoder(config, torch.device("cpu"))
        assert isinstance(encoder, MaisiLatentEncoder)
        assert encoder.name == "autoencoderkl_maisi"

    def test_production_without_vae_config_rejected(self, tmp_path: Path) -> None:
        """vae_config_json 缺失（None）的生产 config：装配源不存在的
        预编码显式拒绝（schema 不拦、分派点拦——工件存在性属运行时契约）。"""
        config = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
        with pytest.raises(ValueError, match="vae_config_json"):
            PreparePipeline.build_encoder(config, torch.device("cpu"))

    def test_noise_seed_is_content_addressed(self) -> None:
        """后验采样噪声种子按（schedule seed, 病例, 序列）稳定派生：
        同键同种子（重跑幂等的前提）、换键换种子（后验样本互异）。"""
        base = PreparePipeline.noise_seed(0, "BraTS-GLI-00000-000", "t1n")
        assert base == PreparePipeline.noise_seed(0, "BraTS-GLI-00000-000", "t1n")
        assert base != PreparePipeline.noise_seed(0, "BraTS-GLI-00001-000", "t1n")
        assert base != PreparePipeline.noise_seed(1, "BraTS-GLI-00000-000", "t1n")
        assert base != PreparePipeline.noise_seed(0, "BraTS-GLI-00000-000", "t2f")
        assert 0 <= base < 2 ** 31  # torch.Generator().manual_seed 的非负域
