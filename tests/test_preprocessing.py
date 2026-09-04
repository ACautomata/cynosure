"""transform 链构建单测（Seam ①，issue #45；data-preparation spec + ADR-0006）。

只断言链的外部行为（spec「Testing Decisions」）：方向码 RAS（flip-only
无轴置换）、dtype float32、强度 clip 域、dim 公式、fixture 基数注入。
工件契约（序列分层 / 病例级不相交 / 幂等）由 prepare 端到端覆盖（Seam ②，
tests/test_prepare.py）。"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from cynosure.reward.preprocessing import UpstreamPreprocessChain
from tests.conftest import LPS_AFFINE, RAS_AFFINE

# 各轴不等且全为 128 倍数：基数 128 的 resize 不变形状，末端形状即 RAS 后
# 形状——若方向步发生轴置换，形状会重排，flip-only 断言由此可观测
AXIS_DISCRIMINATING_SHAPE = (128, 256, 384)


class ChainInput:
    """链输入构造器：按（体数据、affine 方向语义）落临时 NIfTI。"""

    def __init__(self, root: Path) -> None:
        self._root = root

    def volume(self, shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
        """确定性标准正态体数据（float32）；强度测试可在此基础上注入离群值。"""
        return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    def write(self, name: str, volume: np.ndarray, affine: np.ndarray) -> Path:
        path = self._root / name
        nib.save(nib.Nifti1Image(volume, affine), path)
        return path


@pytest.fixture
def chain_input(tmp_path: Path) -> ChainInput:
    return ChainInput(tmp_path)


class TestOrientationRas:
    """方向重定向（上游 recipe 第 3 步）：任一方向语义进链，轴码出 RAS。"""

    def test_lps_input_redirects_to_ras(self, chain_input: ChainInput) -> None:
        """LPS-affine 输入 → 输出轴码 RAS（BraTS 原生 ~89% LPS 的对齐验收）。"""
        path = chain_input.write(
            "lps.nii.gz", chain_input.volume(AXIS_DISCRIMINATING_SHAPE), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain()(path)
        assert tuple(nib.aff2axcodes(image.meta["affine"])) == ("R", "A", "S")

    def test_redirection_is_flip_only(self, chain_input: ChainInput) -> None:
        """flip-only 无轴置换：全 128 倍数、各轴不等的形状 resize 前后不变，
        轴置换会让形状重排而失败（BraTS LPS→RAS 走纯翻转）。"""
        path = chain_input.write(
            "lps.nii.gz", chain_input.volume(AXIS_DISCRIMINATING_SHAPE), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain()(path)
        assert tuple(image.shape) == (1, *AXIS_DISCRIMINATING_SHAPE)

    def test_ras_input_stays_ras(self, chain_input: ChainInput) -> None:
        """混入的 RAS 夹具同样合法：方向已合规时链不改变轴码。"""
        path = chain_input.write(
            "ras.nii.gz", chain_input.volume(AXIS_DISCRIMINATING_SHAPE), RAS_AFFINE,
        )
        image = UpstreamPreprocessChain()(path)
        assert tuple(nib.aff2axcodes(image.meta["affine"])) == ("R", "A", "S")


class TestDtypeAndIntensity:
    def test_output_is_float32(self, chain_input: ChainInput) -> None:
        path = chain_input.write(
            "lps.nii.gz", chain_input.volume((64, 64, 64)), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain(resize_base=64)(path)
        assert image.dtype == torch.float32

    def test_intensity_clips_to_unit_range(self, chain_input: ChainInput) -> None:
        """强度 recipe（第 5 步）：0–99.5 百分位仿射映射到 [0,1] 且 clip=True
        （fork issue #251 的有意偏差）。0.1% 体素 = 100 使 p99.5 落在正态尾部：
        clip=True 时离群值封顶 1.0；clip=False（MONAI 上游默认）它们线性外推
        到 ~14——输出 max ≤ 1 因此是判别性断言，trilinear 稀释不掩蔽。"""
        rng = np.random.default_rng(7)
        volume = rng.standard_normal((32, 32, 32)).astype(np.float32)
        volume[rng.random((32, 32, 32)) < 0.001] = 100.0
        path = chain_input.write("outliers.nii.gz", volume, LPS_AFFINE)
        image = UpstreamPreprocessChain()(path)
        assert image.min() >= 0.0
        assert image.max() > 0.5  # 高强度信号存活到末端（不因插值稀释而假绿）
        assert image.max() <= 1.0


class TestResizeTarget:
    """dim 公式（上游 recipe 第 6 步，fork issue #312 语义）。

    每轴 max(round(size/base), 1)×base，size 从 RAS 重定向后的空间形状读取
    （链内第 6 步取第 3 步之后的形状，测试以端到端形状断言驱动）。"""

    def test_brats_upstream_formula(self) -> None:
        """BraTS 240×240×155 → 256×256×128（latent [4,64,64,32] 契约的 4× 前像）。"""
        assert UpstreamPreprocessChain.resize_target((240, 240, 155), 128) == (
            256, 256, 128,
        )

    def test_each_axis_rounds_to_nearest_multiple_with_floor(self) -> None:
        """逐轴独立取整；小于半基数的轴受下界保护（max(..., 1)）不塌到 0。"""
        assert UpstreamPreprocessChain.resize_target((129, 65, 10), 128) == (
            128, 128, 128,
        )

    def test_injected_fixture_base_keeps_size(self) -> None:
        """fixture config 注入小基数：夹具影像尺寸不变（64×64×32 原样进出）。"""
        assert UpstreamPreprocessChain.resize_target((64, 64, 32), 16) == (64, 64, 32)


class TestChainEndToEnd:
    def test_brats_shape_yields_contract_preimage(self, chain_input: ChainInput) -> None:
        """真实 BraTS 尺寸的 LPS NIfTI 过整链 → [1,256,256,128]。"""
        path = chain_input.write(
            "brats.nii.gz", chain_input.volume((240, 240, 155)), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain()(path)
        assert tuple(image.shape) == (1, 256, 256, 128)

    def test_injected_base_keeps_fixture_volume_through_chain(
        self, chain_input: ChainInput,
    ) -> None:
        """fixture 基数注入端到端：夹具体 64×64×32 过链尺寸不变
        （方向/强度/dtype 步仍全走——fixture 不是对齐对象、走同一条链逻辑）。"""
        path = chain_input.write(
            "fixture.nii.gz", chain_input.volume((64, 64, 32)), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain(resize_base=16)(path)
        assert tuple(image.shape) == (1, 64, 64, 32)
        assert image.dtype == torch.float32
