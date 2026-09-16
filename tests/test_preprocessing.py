"""transform 链构建单测（Seam ①，issue #45；data-preparation spec + ADR-0006）。

只断言链的外部行为（spec「Testing Decisions」）：方向码 RAS（flip-only
无轴置换）、dtype float32、强度 clip 域、dim 公式、fixture 基数注入。
工件契约（序列分层 / 病例级不相交 / 幂等）由 prepare 端到端覆盖（Seam ②，
tests/test_prepare.py）。spacing 侧车读取同属读图环节（issue #46）。

强度臂与 resize 目标的域臂参数化（issue #130，#71 裁决）：BraTS 臂
clip=True + dim 公式（默认构造，语义零改动）；MR-RATE 臂 clip=False +
词汇表统一网格绝对目标——出链数值与形状唯一性逐条可验。"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.data import MetaTensor

from cynosure.conditions import MrConditionSpec, MrConditionVocabulary
from cynosure.reward.preprocessing import SpacingSidecar, UpstreamPreprocessChain
from tests.conftest import ANISOTROPIC_AFFINE, LPS_AFFINE, RAS_AFFINE
from tests.test_condition_vocabulary import PRODUCTION_VOCAB_PATH

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


class TestSpacingSidecar:
    """spacing 侧车读取（issue #46）：链末端 MetaTensor 的 raw header
    zooms ×1e2——读取不受 RAS 重定向影响（flip-only 下 zooms 顺序不变）。"""

    def test_reads_anisotropic_zooms_scaled_x1e2(
        self, chain_input: ChainInput,
    ) -> None:
        """各向异性 zooms 按存储轴顺序读出 ×1e2——值来自 header，非常量。"""
        path = chain_input.write(
            "anisotropic.nii.gz",
            chain_input.volume((64, 64, 32)),
            ANISOTROPIC_AFFINE,
        )
        image = UpstreamPreprocessChain(resize_base=16)(path)
        assert SpacingSidecar().read(image) == (50.0, 100.0, 200.0)

    def test_flip_only_redirect_keeps_zooms(
        self, chain_input: ChainInput,
    ) -> None:
        """LPS→RAS 翻转后读值仍 = 原始 header zooms ×1e2：方向步只改
        affine、不动 raw zooms（AC「不受方向重定向影响」）。"""
        path = chain_input.write(
            "lps.nii.gz", chain_input.volume((64, 64, 32)), LPS_AFFINE,
        )
        raw_zooms = nib.load(path).header.get_zooms()[:3]
        image = UpstreamPreprocessChain(resize_base=16)(path)
        assert tuple(nib.aff2axcodes(image.meta["affine"])) == ("R", "A", "S")
        assert SpacingSidecar().read(image) == (
            tuple(float(zoom) * 1e2 for zoom in raw_zooms)
        )

    def test_missing_zooms_metadata_rejected(self) -> None:
        """meta 无 raw header zooms（非 NIfTI 输入、读图契约破坏）显式失败。"""
        with pytest.raises(ValueError, match="zooms"):
            SpacingSidecar().read(MetaTensor(torch.zeros(1, 4, 4, 4)))


# MR-RATE 臂测试用的生产条件（词汇表最小网格之一，整链 CPU 可负担）；
# 期望值（网格 / spacing）在断言处独立登记——代码改词汇表登记即测出
UNIFORM_GRID_CONDITION = "t1w/sagittal"
UNIFORM_GRID: tuple[int, int, int] = (128, 256, 256)
# = FOV (176, 250, 250) / 网格 (128, 256, 256) ×1e2（全部二进制精确）
CONDITION_SPACING_X1E2: tuple[float, float, float] = (137.5, 97.65625, 97.65625)


@pytest.fixture(scope="module")
def mr_condition() -> MrConditionSpec:
    """生产词汇表登记的统一网格条件（module 级装载：词汇表不可变视图）。"""
    return MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH).by_name(
        UNIFORM_GRID_CONDITION,
    )


class TestIntensityArmParametrization:
    """强度臂按域参数化（issue #130，#71 裁决）：BraTS 臂 clip=True
    （fork 有意偏差，默认构造）与 MR-RATE 臂 clip=False（官方口径）——
    同一离群值输入，两臂出链数值互为反证。"""

    @staticmethod
    def _outlier_volume_path(chain_input: ChainInput) -> Path:
        """0.1% 体素 = 100 的正态体：p99.5 落在正态尾部，离群值在百分位
        窗口之外——窗口外高值是两臂分歧点（与既有 clip=True 测试同夹具）。"""
        rng = np.random.default_rng(7)
        volume = rng.standard_normal((32, 32, 32)).astype(np.float32)
        volume[rng.random((32, 32, 32)) < 0.001] = 100.0
        return chain_input.write("outliers.nii.gz", volume, LPS_AFFINE)

    def test_mrrate_arm_keeps_beyond_window_values(
        self, chain_input: ChainInput,
    ) -> None:
        """MR-RATE 臂（clip=False）：超越百分位窗口的高值线性外推 > 1.0，
        保留不被截断（对齐真上游 scripts/transforms.py 语义）。"""
        path = self._outlier_volume_path(chain_input)
        image = UpstreamPreprocessChain(clip_intensity=False)(path)
        assert image.min() >= 0.0
        assert image.max() > 1.0

    def test_brats_arm_clips_same_input(self, chain_input: ChainInput) -> None:
        """BraTS 臂（clip=True，默认）：同一输入反证——窗口外高值封顶
        1.0（两臂 embedding 不可互用的数值面）。"""
        path = self._outlier_volume_path(chain_input)
        image = UpstreamPreprocessChain()(path)
        assert image.min() >= 0.0
        assert image.max() <= 1.0


class TestConditionUniformGridResample:
    """逐条件统一网格 resample（issue #130）：MR-RATE 臂 resize 目标 =
    词汇表登记的统一网格（RAS 轴序绝对目标），同条件任意原生形状出链
    形状唯一——「条件 → latent 形状」契约的预处理半边。"""

    @pytest.fixture
    def mr_arm(
        self, mr_condition: MrConditionSpec,
    ) -> UpstreamPreprocessChain:
        return UpstreamPreprocessChain(
            clip_intensity=False, target_grid=mr_condition.grid_xyz,
        )

    def test_registered_grid_constant_matches_vocabulary(
        self, mr_condition: MrConditionSpec,
    ) -> None:
        """对账守卫：测试独立登记的统一网格常量与词汇表登记值一致——
        词汇表工件改登记时本文件期望值须同步复核（链目标本身的行为
        断言见下方两个形状唯一性测试）。"""
        assert mr_condition.grid_xyz == UNIFORM_GRID

    def test_same_condition_different_native_shapes_yield_one_shape(
        self, chain_input: ChainInput, mr_arm: UpstreamPreprocessChain,
    ) -> None:
        """同条件三个不同原生形状（LPS 翻转 / RAS 合规混入）出链形状
        唯一 = 该条件统一网格。"""
        for name, shape, affine in [
            ("shape-a.nii.gz", (100, 180, 160), LPS_AFFINE),
            ("shape-b.nii.gz", (96, 128, 240), RAS_AFFINE),
            ("shape-c.nii.gz", (130, 130, 200), LPS_AFFINE),
        ]:
            path = chain_input.write(
                name, chain_input.volume(shape, seed=1), affine,
            )
            image = mr_arm(path)
            assert tuple(image.shape) == (1, *UNIFORM_GRID), shape

    def test_axis_permuting_direction_lands_on_ras_grid(
        self, chain_input: ChainInput, mr_arm: UpstreamPreprocessChain,
    ) -> None:
        """轴置换方向（MR-RATE 常见，fork issue #312 审计的坏形状来源）：
        RAS 重定向置换存储轴后统一网格仍按 RAS 轴序落位——出链形状唯一
        且轴码 RAS（统一网格是 RAS 轴序，绝对目标不依赖取轴口径）。"""
        # 存储轴 x/y 指向互换（方向码 A/R/S → RAS 重定向置换前两存储轴）
        swapped = np.array([
            [0.0, 1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        path = chain_input.write(
            "swapped.nii.gz", chain_input.volume((210, 96, 160), seed=2), swapped,
        )
        image = mr_arm(path)
        assert tuple(nib.aff2axcodes(image.meta["affine"])) == ("R", "A", "S")
        assert tuple(image.shape) == (1, *UNIFORM_GRID)

    def test_brats_arm_dim_formula_semantics_untouched(
        self, chain_input: ChainInput, mr_condition: MrConditionSpec,
    ) -> None:
        """BraTS 臂（target_grid 缺省）dim 公式语义不回归：BraTS 尺寸
        输入照旧 256×256×128，而非词汇表统一网格。"""
        path = chain_input.write(
            "brats.nii.gz", chain_input.volume((240, 240, 155)), LPS_AFFINE,
        )
        image = UpstreamPreprocessChain()(path)
        assert tuple(image.shape) == (1, 256, 256, 128)
        assert mr_condition.grid_xyz != (256, 256, 128)  # 两臂目标确实不同


class TestSpacingConditionAgainstPerCaseZooms:
    """spacing 条件属性 vs per-case 侧车的对照（issue #130）：同条件两卷
    携带不同 native zooms——差异真实存在于卷间（侧车可证），但 MR-RATE
    臂的 spacing 语义值来自条件属性（FOV / 网格 ×1e2），两卷严格同值、
    与 header zooms 无关——「不逐卷侧车」堵死 spacing 判别捷径。"""

    def test_condition_spacing_identical_regardless_of_volume_zooms(
        self, chain_input: ChainInput, mr_condition: MrConditionSpec,
    ) -> None:
        chain = UpstreamPreprocessChain(
            clip_intensity=False, target_grid=mr_condition.grid_xyz,
        )
        per_case_sidecar_values = []
        for name, zoom in [("volume-1.nii.gz", 0.5), ("volume-2.nii.gz", 3.0)]:
            path = chain_input.write(
                name,
                chain_input.volume((128, 64, 64), seed=3),
                np.diag([zoom, 1.0, 1.0, 1.0]),
            )
            # 侧车（BraTS 臂机制）读出的 per-case 值随卷而变——卷间差异
            # 确实存在于 header
            per_case_sidecar_values.append(SpacingSidecar().read(chain(path)))
        assert per_case_sidecar_values[0] != per_case_sidecar_values[1]
        # MR-RATE 臂的 spacing 语义值 = 条件属性：独立登记的 FOV/网格 ×1e2
        # 期望值，与两卷的 header zooms 均不同（值不来自逐卷读取）
        assert CONDITION_SPACING_X1E2 == pytest.approx(tuple(
            fov / grid * 100.0
            for fov, grid in zip(mr_condition.fov_mm, mr_condition.grid_xyz)
        ))
        assert CONDITION_SPACING_X1E2 not in per_case_sidecar_values
