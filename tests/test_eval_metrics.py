"""eval 度量核单测：Frechet 距离 / 无偏 MMD² + bootstrap CI / 三正交面
切片 / 特征提取器策略（stub 确定性、RadImageNet 装载契约）。"""

import pytest
import torch

from cynosure.eval.frechet import BootstrapKernelMmd, FrechetDistance, KernelMmd
from cynosure.eval.features import (
    RadImageNetFeatureExtractor,
    StubSliceFeatureExtractor,
)
from cynosure.eval.volumes import OrthoPlane, RealVolumeStore, VolumePairFidelity
from tests.conftest import SyntheticBratsDataset


class TestFrechetDistance:
    def test_identical_feature_sets_score_zero(self) -> None:
        features = torch.randn(16, 8, dtype=torch.float64)
        assert FrechetDistance().score(features, features) == pytest.approx(0.0, abs=1e-9)

    def test_shifted_distribution_scores_positive(self) -> None:
        torch.manual_seed(0)
        base = torch.randn(64, 8, dtype=torch.float64)
        shifted = base + 5.0
        score = FrechetDistance().score(base, shifted)
        assert score > 0.0
        # 同分布两份独立样本的距离显著小于跨分布
        close = FrechetDistance().score(base, torch.randn(64, 8, dtype=torch.float64))
        assert score > close

    def test_too_few_samples_rejected(self) -> None:
        one = torch.randn(1, 8, dtype=torch.float64)
        with pytest.raises(ValueError, match="2"):
            FrechetDistance().score(one, torch.randn(8, 8, dtype=torch.float64))

    def test_scale_invariance_under_shared_scaling(self) -> None:
        """FID 对两侧同乘常数不敏感（协方差与均值平方项同阶缩放）：
        同一分布形状、整体放大 c 倍后距离按 c² 缩放——单调性而非不变性。"""
        torch.manual_seed(1)
        a = torch.randn(64, 8, dtype=torch.float64)
        b = torch.randn(64, 8, dtype=torch.float64) + 0.1
        small = FrechetDistance().score(a, b)
        large = FrechetDistance().score(a * 3.0, b * 3.0)
        assert large == pytest.approx(small * 9.0, rel=1e-6)


class TestKernelMmd:
    def test_same_distribution_near_zero_and_ordered_against_shifted(self) -> None:
        """无偏 MMD² 对同分布独立样本 ≈ 0（允许微负），跨分布显著为正且
        随偏移单调增大。"""
        torch.manual_seed(0)
        base = torch.randn(64, 8, dtype=torch.float64)
        mmd = KernelMmd()
        same = mmd.score(base, torch.randn(64, 8, dtype=torch.float64))
        shifted = mmd.score(base, torch.randn(64, 8, dtype=torch.float64) + 8.0)
        assert abs(same) < shifted
        assert shifted > 0.0

    def test_too_few_samples_rejected(self) -> None:
        one = torch.randn(1, 8, dtype=torch.float64)
        with pytest.raises(ValueError, match="2"):
            KernelMmd().score(one, one)


class TestBootstrapKernelMmd:
    @staticmethod
    def _interval(estimator: BootstrapKernelMmd, a, b) -> tuple:
        """(点估计, CI 下界, CI 上界)（经公共 score_and_replicates 口径）。"""
        point, replicates = estimator.score_and_replicates(a, b)
        return (
            point,
            float(replicates.quantile(estimator.LOWER_QUANTILE)),
            float(replicates.quantile(estimator.UPPER_QUANTILE)),
        )

    def test_ci_brackets_point_estimate(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(32, 8, dtype=torch.float64)
        b = torch.randn(32, 8, dtype=torch.float64)
        estimator = BootstrapKernelMmd(
            replicates=20, generator=torch.Generator().manual_seed(7),
        )
        point, low, high = self._interval(estimator, a, b)
        assert low <= point <= high

    def test_deterministic_under_fixed_generator(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(24, 8, dtype=torch.float64)
        b = torch.randn(24, 8, dtype=torch.float64)
        first = self._interval(
            BootstrapKernelMmd(replicates=10, generator=torch.Generator().manual_seed(3)),
            a, b,
        )
        second = self._interval(
            BootstrapKernelMmd(replicates=10, generator=torch.Generator().manual_seed(3)),
            a, b,
        )
        assert first == second

    def test_score_and_replicates_yields_point_and_distribution(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(24, 8, dtype=torch.float64)
        b = torch.randn(24, 8, dtype=torch.float64)
        estimator = BootstrapKernelMmd(
            replicates=10, generator=torch.Generator().manual_seed(5),
        )
        point, replicates = estimator.score_and_replicates(a, b)
        assert replicates.shape == (10,)
        assert torch.isfinite(replicates).all()
        full = KernelMmd().score(a, b)
        assert point == pytest.approx(full)  # 点估计 = 全量特征集口径

    def test_replicates_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            BootstrapKernelMmd(replicates=0, generator=torch.Generator())


class TestStubSliceFeatureExtractor:
    def test_deterministic_and_conforms_to_contract(self) -> None:
        extractor = StubSliceFeatureExtractor()
        slices = torch.randn(6, 1, 16, 16)
        first = extractor.extract(slices)
        second = extractor.extract(slices)
        assert torch.equal(first, second)
        assert first.shape == (6, extractor.feature_dim)
        assert torch.isfinite(first).all()

    def test_feature_dim_is_stable_constant(self) -> None:
        """特征维度是契约：FID/KID 两侧特征空间必须同维，维度漂移即静默错配。"""
        assert StubSliceFeatureExtractor().feature_dim == (
            StubSliceFeatureExtractor.FEATURE_DIM
        )


class TestRadImageNetFeatureExtractor:
    def test_missing_weights_rejected_with_config_hint(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="radimagenet_weights"):
            RadImageNetFeatureExtractor(tmp_path / "absent.pth")

    def test_loads_monai_resnet_state_dict(self, tmp_path) -> None:
        """生产特征器装载契约：MONAI resnet50 同构 state_dict 可装载，
        装载后对同批切片特征确定。"""
        torch.manual_seed(0)
        backbone = RadImageNetFeatureExtractor.build_backbone()
        weights = tmp_path / "radimagenet_resnet50.pth"
        torch.save(backbone.state_dict(), weights)
        extractor = RadImageNetFeatureExtractor(weights)
        slices = torch.randn(2, 1, 64, 64)
        features = extractor.extract(slices)
        assert features.shape == (2, extractor.feature_dim)
        assert torch.isfinite(features).all()

    def test_extraction_streams_in_bounded_forward_batches(self, tmp_path) -> None:
        """生产规模有界前向：一个里程碑上千切片分块进 ResNet50（单批
        224×224×3 激活分配是 OOM 级），特征按序拼接完整。"""
        torch.manual_seed(0)
        backbone = RadImageNetFeatureExtractor.build_backbone()
        weights = tmp_path / "radimagenet_resnet50.pth"
        torch.save(backbone.state_dict(), weights)
        extractor = RadImageNetFeatureExtractor(weights)
        batches: list[int] = []

        class RecordingBackbone:
            """前向批大小记录替身（特征维契约同 backbone 出口）。"""

            def __call__(self, batch: torch.Tensor) -> torch.Tensor:
                batches.append(batch.shape[0])
                return torch.zeros(batch.shape[0], extractor.feature_dim)

        extractor._backbone = RecordingBackbone()
        features = extractor.extract(torch.randn(100, 1, 64, 64))
        assert batches == [32, 32, 32, 4]  # EXTRACT_BATCH 分块
        assert features.shape == (100, extractor.feature_dim)


class TestOrthoPlane:
    VOLUMES_SHAPE = (3, 8, 6, 4)  # [K, X, Y, Z]

    def test_each_plane_slices_along_its_normal_axis(self) -> None:
        volumes = torch.arange(torch.Size(self.VOLUMES_SHAPE).numel()).reshape(
            self.VOLUMES_SHAPE,
        ).float()
        assert OrthoPlane.XY.slice(volumes).shape == (3 * 4, 1, 8, 6)
        assert OrthoPlane.YZ.slice(volumes).shape == (3 * 8, 1, 6, 4)
        assert OrthoPlane.ZX.slice(volumes).shape == (3 * 6, 1, 4, 8)

    def test_plane_slice_content_matches_axis_convention(self) -> None:
        volumes = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        xy = OrthoPlane.XY.slice(volumes)
        assert torch.equal(xy[0, 0], volumes[0, :, :, 0])  # 法轴 Z
        yz = OrthoPlane.YZ.slice(volumes)
        assert torch.equal(yz[0, 0], volumes[0, 0, :, :])  # 法轴 X
        zx = OrthoPlane.ZX.slice(volumes)
        assert torch.equal(zx[0, 0], volumes[0, :, 0, :].T.contiguous())  # 法轴 Y，面内 [Z, X]

    def test_plane_names_are_the_spec_three(self) -> None:
        assert [plane.value for plane in OrthoPlane] == ["XY", "YZ", "ZX"]


class TestRealVolumeStore:
    @staticmethod
    def _single_case_dataset(root) -> object:
        """一病例合成数据集（四序列 NIfTI，8×8×4 体）。"""
        return SyntheticBratsDataset(
            root / "dataset", ["BraTS-GLI-00000-000"], (8, 8, 4), seed=0,
        ).write()

    def test_loads_reference_volume_by_case_and_modality(self, tmp_path) -> None:
        dataset = self._single_case_dataset(tmp_path)
        store = RealVolumeStore(dataset)
        volume = store.volume("BraTS-GLI-00000-000", "t1n")
        assert volume.shape == (8, 8, 4)
        assert torch.isfinite(volume).all()

    def test_unknown_case_rejected(self, tmp_path) -> None:
        dataset = self._single_case_dataset(tmp_path)
        store = RealVolumeStore(dataset)
        with pytest.raises(ValueError, match="no-such-case"):
            store.volume("no-such-case", "t1n")

    def test_pool_allowlist_restricts_reference_cases(self, tmp_path) -> None:
        """病例白名单 = real pool train split：参照分布不越过病例级
        分割（dataset_root 上的 val/test 病例不进 FID 参照）。"""
        dataset = SyntheticBratsDataset(
            tmp_path / "dataset",
            ["BraTS-GLI-00000-000", "BraTS-GLI-00001-000"],
            (8, 8, 4), seed=0,
        ).write()
        store = RealVolumeStore(
            dataset, case_ids={"BraTS-GLI-00000-000"},
        )
        assert store.case_ids() == ["BraTS-GLI-00000-000"]
        with pytest.raises(ValueError, match="BraTS-GLI-00001-000"):
            store.volume("BraTS-GLI-00001-000", "t1n")

    def test_unknown_allowlisted_case_rejected(self, tmp_path) -> None:
        """白名单含 dataset_root 不存在的病例（typo）显式拒绝。"""
        dataset = self._single_case_dataset(tmp_path)
        with pytest.raises(ValueError, match="no-such-case"):
            RealVolumeStore(dataset, case_ids={"no-such-case"})


class TestVolumePairFidelity:
    def test_identical_volumes_score_perfect(self) -> None:
        volumes = torch.rand(2, 1, 16, 16, 8)
        fidelity = VolumePairFidelity()
        ssim, mae, psnr = fidelity.score(volumes, volumes)
        assert ssim == pytest.approx(1.0, abs=1e-3)
        assert mae == pytest.approx(0.0, abs=1e-9)
        assert psnr == pytest.approx(VolumePairFidelity.PSNR_CAP)  # MSE=0 封顶

    def test_divergent_volumes_score_worse_than_close_ones(self) -> None:
        torch.manual_seed(0)
        base = torch.rand(2, 1, 16, 16, 8)
        fidelity = VolumePairFidelity()
        close_ssim, close_mae, close_psnr = fidelity.score(base, base + 0.01)
        far_ssim, far_mae, far_psnr = fidelity.score(base, 1.0 - base)
        assert close_ssim > far_ssim
        assert close_mae < far_mae
        assert close_psnr > far_psnr  # PSNR 随失真增大单调下降
        assert 0.0 < far_psnr < VolumePairFidelity.PSNR_CAP
