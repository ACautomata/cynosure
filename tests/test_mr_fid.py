"""MR FID 仪器单测（wayfinder #79 的验证面）。

覆盖：9 项冻结变量的 schema 校验、切片轴语义与归一链的 fork 口径
（数值级断言）、变换链顺序（crop→强度→pad 的 fixture 固化，
upstream-eval-protocol §4.3/§7-7）、特征缓存 fingerprint 防护
（§1.7 陷阱结构化兜底）、仪器端到端（清单截断/provenance 落盘/
双轨隔离）与 CLI fid 子命令。
"""

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch
from monai.transforms import ScaleIntensityRangePercentiles

from cynosure.eval.features import RadImageNetBackbone
from cynosure.eval.mr_fid import (
    MODALITY_PREPROCESSING,
    FidResult,
    FeatureCacheGuard,
    MrFidConfig,
    MrFidInstrument,
    VolumePlaneSlicer,
)

_RAS_AFFINE = np.eye(4)


def _write_volume(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), _RAS_AFFINE), path)
    return path


def _volume_tree(root: Path, names: list[str], shape: tuple[int, int, int]) -> Path:
    """确定性非负内容的小体树（MR 百分位臂的合法输入域）。"""
    for index, name in enumerate(names):
        rng = np.random.default_rng((0, index))
        array = rng.uniform(0.0, 100.0, shape).astype(np.float32)
        _write_volume(root / name, array)
    return root


class _ChannelMeanBackbone:
    """[N,3,H,W] → [N,3] 逐切片通道均值：归一链的数值断言载体（确定性）。"""

    def __init__(self) -> None:
        self.forward_calls = 0

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return slices.mean(dim=(2, 3))


class _AcceleratorLikeBackbone:
    """模拟加速器上的骨干：特征以 mps 设备张量返回（本机可用；无 mps
    的环境跳过——meta 设备不可作此模拟，``.cpu()`` 拷出会被拒绝）。"""

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            slices.shape[0], 3, dtype=torch.float32, device="mps",
        )


def _valid_config_dict(tmp_path: Path, **overrides) -> dict:
    real_root = _volume_tree(tmp_path / "real", ["a.nii.gz", "b.nii.gz", "c.nii.gz"], (8, 8, 8))
    synth_root = _volume_tree(tmp_path / "synth", ["x.nii.gz", "y.nii.gz"], (8, 8, 8))
    (tmp_path / "real.txt").write_text("a.nii.gz\nb.nii.gz\nc.nii.gz\n")
    (tmp_path / "synth.txt").write_text("x.nii.gz\ny.nii.gz\n")
    config = {
        "real_dataset_root": str(real_root),
        "real_filelist": str(tmp_path / "real.txt"),
        "real_features_dir": "realfeat",
        "synth_dataset_root": str(synth_root),
        "synth_filelist": str(tmp_path / "synth.txt"),
        "synth_features_dir": "synthfeat",
        "num_images": 2,
        "target_shape": [16, 16, 16],
        "center_slices_ratio": 0.5,
        "radimagenet_weights": str(tmp_path / "radimagenet.pt"),
        "output_root": str(tmp_path / "features"),
        "result_json": str(tmp_path / "result.json"),
        "num_workers": 0,
    }
    config.update(overrides)
    return config


class TestMrFidConfig:
    def test_minimal_config_valid(self, tmp_path):
        assert MrFidConfig.model_validate(
            _valid_config_dict(tmp_path),
        ).modality == "mr"

    @pytest.mark.parametrize("field", [
        "real_dataset_root", "real_filelist", "real_features_dir",
        "synth_dataset_root", "synth_filelist", "synth_features_dir",
        "num_images", "target_shape", "center_slices_ratio",
        "radimagenet_weights", "output_root", "result_json",
    ])
    def test_required_pins_absent_by_default(self, tmp_path, field):
        """9 项冻结变量的钉死形态 = 必填无默认：缺省即 schema 拒绝。"""
        data = _valid_config_dict(tmp_path)
        del data[field]
        with pytest.raises(Exception, match=field):
            MrFidConfig.model_validate(data)

    @pytest.mark.parametrize("ratio", [0.0, -0.1, 1.01])
    def test_center_slices_ratio_bounds(self, tmp_path, ratio):
        with pytest.raises(Exception):
            MrFidConfig.model_validate(
                _valid_config_dict(tmp_path, center_slices_ratio=ratio),
            )

    def test_num_images_at_least_two(self, tmp_path):
        """FID 双侧基数同时影响有偏程度；单卷对比无意义。"""
        with pytest.raises(Exception):
            MrFidConfig.model_validate(_valid_config_dict(tmp_path, num_images=1))

    @pytest.mark.parametrize("field,value", [
        ("modality", "xmri"),
        ("model_name", "squeezenet1_1"),
        ("dtype", "float16"),
    ])
    def test_literal_pins(self, tmp_path, field, value):
        """squeezenet fallback 臂（torchvision 白名单外）与 dtype 在类型层拒绝。"""
        with pytest.raises(Exception):
            MrFidConfig.model_validate(_valid_config_dict(tmp_path, **{field: value}))

    def test_extra_field_forbidden(self, tmp_path):
        with pytest.raises(Exception):
            MrFidConfig.model_validate(
                _valid_config_dict(tmp_path, drop_empty=True),
            )

    def test_bad_geometry_rejected(self, tmp_path):
        with pytest.raises(Exception):
            MrFidConfig.model_validate(
                _valid_config_dict(tmp_path, target_shape=[16, 0, 16]),
            )
        with pytest.raises(Exception):
            MrFidConfig.model_validate(
                _valid_config_dict(tmp_path, resample_spacing=[1.0, 0.0, 1.0]),
            )


class TestVolumePlaneSlicer:
    def test_slicing_axis_semantics_fork_verbatim(self):
        """切片轴语义数值级断言：XY=沿 D、YZ=沿 H、ZX=沿 W（不按名字
        推断轴）；中心窗口 int((1±r)/2·N) 截断。"""
        backbone = _ChannelMeanBackbone()
        slicer = VolumePlaneSlicer(backbone, center_slices_ratio=0.5)
        volume = torch.rand(1, 8, 12, 6)  # [C, H, W, D]
        xy, yz, zx = slicer.plane_features(volume)
        # D 轴窗: int(0.25·6)=1 → int(0.75·6)=4 → 3 张 H×W 面
        assert xy.shape == (3, 3)
        # H 轴窗: int(0.25·8)=2 → int(0.75·8)=6 → 4 张 W×D 面
        assert yz.shape == (4, 3)
        # W 轴窗: int(0.25·12)=3 → int(0.75·12)=9 → 6 张 H×D 面
        assert zx.shape == (6, 3)

    def test_full_ratio_keeps_all_slices(self):
        backbone = _ChannelMeanBackbone()
        slicer = VolumePlaneSlicer(backbone, center_slices_ratio=1.0)
        volume = torch.rand(1, 4, 5, 6)
        xy, yz, zx = slicer.plane_features(volume)
        assert xy.shape == (6, 3)
        assert yz.shape == (4, 3)
        assert zx.shape == (5, 3)

    def test_normalisation_is_per_volume_plane_not_per_slice(self):
        """归一链口径：全局 min-max 作用在「每卷×每平面」整批切片上
        （norm2d=False 4D 分支），不是逐切片；随后减翻转后通道均值。"""
        backbone = _ChannelMeanBackbone()
        slicer = VolumePlaneSlicer(backbone, center_slices_ratio=1.0)
        volume = torch.zeros(1, 4, 4, 2)
        volume[..., 0] = torch.linspace(0.0, 1.0, 16).reshape(4, 4)
        volume[..., 1] = torch.linspace(100.0, 101.0, 16).reshape(4, 4)
        xy, _, _ = slicer.plane_features(volume)
        # 平面全局 min=0/max=101：暗片 ch0 均值 ≈ 0.5/101 − 0.406，
        # 亮片 ≈ 100.5/101 − 0.406（min-max 后减通道均值）
        assert xy[0, 0] == pytest.approx(0.5 / 101.0 - 0.406, abs=1e-5)
        assert xy[1, 0] == pytest.approx(100.5 / 101.0 - 0.406, abs=1e-5)

    def test_channel_flip_and_mean_subtraction(self):
        """通道翻转 [2,1,0] + 减翻转后 ImageNet 均值 [0.406,0.456,0.485]：
        常量体的三通道分别剩下负均值（翻转对灰度复制体不可见，均值
        减除使三通道可区分——两语义联合断言）。"""
        backbone = _ChannelMeanBackbone()
        slicer = VolumePlaneSlicer(backbone, center_slices_ratio=1.0)
        volume = torch.ones(1, 3, 3, 3)
        xy, yz, zx = slicer.plane_features(volume)
        expected = torch.tensor([-0.406, -0.456, -0.485])
        for plane in (xy, yz, zx):
            # 常量体 min-max 后全 0（0/(0+1e-10)），减均值即期望
            assert torch.allclose(plane[0], expected, atol=1e-6)

    def test_ratio_out_of_bounds_rejected(self):
        with pytest.raises(ValueError, match="中心切片比例"):
            VolumePlaneSlicer(_ChannelMeanBackbone(), center_slices_ratio=1.5)


class TestModalityPreprocessingChain:
    def test_mr_chain_order_crop_intensity_pad(self, tmp_path):
        """fixture 固化（upstream-eval-protocol §4.3/§7-7）：crop→强度→pad
        下百分位统计只看真实内容——padding 的 0 不稀释百分位；且强度
        先于 padding，pad 区=padding_value(MR=0)。"""
        content = np.linspace(10.0, 20.0, 64, dtype=np.float32).reshape(4, 4, 4)
        volume_path = _write_volume(tmp_path / "mr.nii.gz", content)
        chain = MODALITY_PREPROCESSING["mr"].compose(
            target_shape=(16, 16, 16),
        )
        result = chain({"image": str(volume_path)})["image"]
        assert tuple(result.shape) == (1, 16, 16, 16)
        array = result[0].numpy()
        # 内容映射 == 对未 padding 内容直接做同一百分位变换（零稀释）
        expected = ScaleIntensityRangePercentiles(
            lower=0.0, upper=99.5, b_min=0.0, b_max=1000.0, clip=False,
        )(content.copy())
        centre = array[6:10, 6:10, 6:10]
        assert np.allclose(centre, expected, atol=1e-4)
        # clip=False：顶端超 p99.5 的内容外推到 1000 之上（与生成侧
        # MR 只 clip 下界同语义）；pad 区 = 0（强度在 pad 前，MR 的
        # padding_value 即输出域下界）
        assert array.max() > 1000.0
        assert array[0, 0, 0] == 0.0

    def test_ct_chain_fixed_window_and_pad_value(self, tmp_path):
        """CT 臂（表结构移植）：固定 HU 窗 clip、padding=-1000。"""
        content = np.full((4, 4, 4), 500.0, dtype=np.float32)
        volume_path = _write_volume(tmp_path / "ct.nii.gz", content)
        chain = MODALITY_PREPROCESSING["ct"].compose(
            target_shape=(16, 16, 16),
        )
        array = chain({"image": str(volume_path)})["image"][0].numpy()
        assert array[6, 6, 6] == pytest.approx(500.0)
        assert array[0, 0, 0] == -1000.0

    def test_single_compose_instance_shared_contract(self):
        """FID 可比性要求：同一 Compose 实例服务两侧（fork docstring
        契约）——compose 是纯构建，每次调用产独立实例、链步骤一致。"""
        mr = MODALITY_PREPROCESSING["mr"]
        first = mr.compose(target_shape=(8, 8, 8))
        second = mr.compose(target_shape=(8, 8, 8))
        assert first is not second
        assert len(first.transforms) == len(second.transforms)


class TestFeatureCacheGuard:
    def _fingerprint(self, ratio: float = 0.5) -> dict:
        return {"center_slices_ratio": ratio, "target_shape": [16, 16, 16]}

    def test_first_run_writes_fingerprint_then_reuses(self, tmp_path):
        guard = FeatureCacheGuard(tmp_path, self._fingerprint(), False)
        cache = guard.cache_path(Path("/data"), Path("/data/sub/vol.nii.gz"))
        assert cache == tmp_path / "sub" / "vol.pt"
        extract_calls = []

        def extract():
            extract_calls.append(1)
            return (torch.zeros(2, 4), torch.zeros(3, 4), torch.zeros(5, 4))

        feats = guard.load_or_extract(cache, extract)
        assert feats[2].shape == (5, 4)
        assert extract_calls == [1]
        # 命中缓存：不再提取（§1.7 默认复用语义保留）
        guard.load_or_extract(cache, extract)
        assert extract_calls == [1]

    def test_geometry_change_hard_fails_without_ignore_existing(self, tmp_path):
        FeatureCacheGuard(tmp_path, self._fingerprint(0.5), False)
        with pytest.raises(ValueError, match="指纹不匹配"):
            FeatureCacheGuard(tmp_path, self._fingerprint(0.4), False)

    def test_ignore_existing_recomputes_and_rewrites_fingerprint(self, tmp_path):
        FeatureCacheGuard(tmp_path, self._fingerprint(0.5), False)
        guard = FeatureCacheGuard(tmp_path, self._fingerprint(0.4), True)
        cache = tmp_path / "vol.pt"
        calls = []

        def extract():
            calls.append(1)
            return (torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))

        guard.load_or_extract(cache, extract)
        assert calls == [1]
        # 换血后指纹已覆写：新口径下普通复用放行
        FeatureCacheGuard(tmp_path, self._fingerprint(0.4), False)

    def test_convention_refresh_purges_unselected_stale_entries(self, tmp_path):
        """ignore_existing 换血必须先清残留条目再发布新指纹：缩小的
        清单换血后，未选中卷的旧口径 ``.pt`` 若存活，放大清单的后续
        运行会看到新指纹而静默装载旧口径特征（§1.7 陷阱的换血变体，
        中途被打断的换血同理）。"""
        guard_a = FeatureCacheGuard(tmp_path, self._fingerprint(0.5), False)
        stale = (torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 4))
        guard_a.load_or_extract(tmp_path / "v1.pt", lambda: stale)
        guard_a.load_or_extract(tmp_path / "v2.pt", lambda: stale)
        # 换口径换血：本次只重提 v1
        guard_b = FeatureCacheGuard(tmp_path, self._fingerprint(0.4), True)
        refresh_calls = []

        def refresh_v1():
            refresh_calls.append(1)
            fresh = (torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 4))
            return fresh

        guard_b.load_or_extract(tmp_path / "v1.pt", refresh_v1)
        assert refresh_calls == [1]
        # 新指纹下的普通运行：v2 的旧口径条目必须已被清除、重新提取
        guard_c = FeatureCacheGuard(tmp_path, self._fingerprint(0.4), False)
        recompute_calls = []

        def recompute_v2():
            recompute_calls.append(1)
            return stale

        guard_c.load_or_extract(tmp_path / "v2.pt", recompute_v2)
        assert recompute_calls == [1]


class TestMrFidInstrument:
    def _instrument(
        self, tmp_path, backbone=None, **overrides,
    ) -> MrFidInstrument:
        config = MrFidConfig.model_validate(_valid_config_dict(tmp_path, **overrides))
        return MrFidInstrument(config, backbone=backbone)

    def test_end_to_end_result_and_provenance(self, tmp_path):
        backbone = _ChannelMeanBackbone()
        instrument = self._instrument(tmp_path, backbone=backbone)
        result = instrument.run(comparison_tag="cell_T1w_AXIAL")
        # 三面 FID + 算术平均（fork 口径）
        assert result.fid_avg == pytest.approx(
            (result.fid_xy + result.fid_yz + result.fid_zx) / 3.0,
        )
        assert result.comparison_tag == "cell_T1w_AXIAL"
        # 落盘纪律：provenance 全字段（冻结变量逐项可溯源）
        loaded = FidResult.load(tmp_path / "result.json")
        assert loaded.modality == "mr"
        assert loaded.model_name == "radimagenet_resnet50"
        assert loaded.num_images == 2
        assert loaded.center_slices_ratio == 0.5
        assert loaded.target_shape == "16x16x16"
        assert loaded.enable_resampling_spacing is None
        assert loaded.enable_padding is True
        assert loaded.enable_center_cropping is True

    def test_cache_lives_under_modality_namespace(self, tmp_path):
        """fork 改造 #5：output_root/<modality>/<features_dir>/。"""
        self._instrument(tmp_path, backbone=_ChannelMeanBackbone()).run()
        assert (tmp_path / "features" / "mr" / "realfeat").is_dir()
        assert (tmp_path / "features" / "mr" / "synthfeat").is_dir()
        cached = sorted((tmp_path / "features" / "mr" / "realfeat").rglob("*.pt"))
        assert len(cached) == 2  # num_images=2 截断
        # 指纹随缓存落盘
        assert (tmp_path / "features" / "mr" / "realfeat" / "fingerprint.json").is_file()

    def test_filelist_sorted_and_truncated(self, tmp_path):
        backbone = _ChannelMeanBackbone()
        self._instrument(tmp_path, backbone=backbone).run()
        # 2 侧 × 2 卷（3 卷截断到 num_images=2）× 3 面 = 12 次前向
        assert backbone.forward_calls == 12
        cached = list((tmp_path / "features" / "mr" / "realfeat").rglob("*.pt"))
        assert len(cached) == 2

    def test_second_run_reuses_cache(self, tmp_path):
        first = _ChannelMeanBackbone()
        self._instrument(tmp_path, backbone=first).run()
        calls_after_first = first.forward_calls
        second = _ChannelMeanBackbone()
        result = self._instrument(tmp_path, backbone=second).run()
        assert second.forward_calls == 0
        assert result.fid_avg == pytest.approx(
            FidResult.load(tmp_path / "result.json").fid_avg,
        )
        assert calls_after_first > 0

    def test_geometry_change_blocks_cache_reuse(self, tmp_path):
        self._instrument(tmp_path, backbone=_ChannelMeanBackbone()).run()
        with pytest.raises(ValueError, match="指纹不匹配"):
            self._instrument(
                tmp_path, backbone=_ChannelMeanBackbone(),
                center_slices_ratio=0.4,
            ).run()

    def test_empty_filelist_rejected(self, tmp_path):
        data = _valid_config_dict(tmp_path)
        Path(data["real_filelist"]).write_text("\n")
        config = MrFidConfig.model_validate(data)
        with pytest.raises(ValueError, match="清单为空"):
            MrFidInstrument(config, backbone=_ChannelMeanBackbone()).run()

    def test_missing_volume_rejected(self, tmp_path):
        data = _valid_config_dict(tmp_path)
        Path(data["synth_filelist"]).write_text("ghost.nii.gz\ny.nii.gz\n")
        config = MrFidConfig.model_validate(data)
        with pytest.raises(FileNotFoundError, match="不存在"):
            MrFidInstrument(config, backbone=_ChannelMeanBackbone()).run()

    def test_missing_weights_rejected_on_default_backbone(self, tmp_path):
        """默认装配（生产路径）按装载契约在缺权重时显式失败。"""
        with pytest.raises(FileNotFoundError, match="RadImageNet"):
            self._instrument(tmp_path).run()

    def test_identical_sides_zero_fid(self, tmp_path):
        """接线自检：同分布两侧（real vs real 同清单）FID ≈ 0。"""
        config = _valid_config_dict(tmp_path)
        config["synth_filelist"] = config["real_filelist"]
        config["synth_dataset_root"] = config["real_dataset_root"]
        config["synth_features_dir"] = "realfeat"  # 同缓存命名空间亦无妨
        config["num_images"] = 3
        instrument = MrFidConfig.model_validate(config)
        result = MrFidInstrument(instrument, backbone=_ChannelMeanBackbone()).run()
        assert result.fid_avg == pytest.approx(0.0, abs=1e-9)

    def test_shared_features_dir_across_distinct_roots_rejected(self, tmp_path):
        """缓存条目绑定数据源：同一 features_dir + 两个数据根时（镜像
        布局），real 侧先写缓存后 synth 侧若能装载会得到人为趋零的
        FID——fingerprint 必须含 dataset_root，令这种碰撞显式硬错误。"""
        config = MrFidConfig.model_validate(_valid_config_dict(
            tmp_path, real_features_dir="shared", synth_features_dir="shared",
        ))
        with pytest.raises(ValueError, match="dataset_root"):
            MrFidInstrument(config, backbone=_ChannelMeanBackbone()).run()

    def test_result_provenance_records_extraction_freeze_variables(
        self, tmp_path,
    ):
        """provenance 自足性：判定两份读数可否互比的全部冻结输入
        （权重本体 sha256 / 设备 / dtype / 双侧缓存目录 / ignore_existing）
        直接随 FidResult 落盘——缓存指纹可能住在别的根下、也会被
        ignore_existing 换血覆写，单独的结果文件不能依赖它。"""
        weights = tmp_path / "w.pt"
        weights.write_bytes(b"fixture-weights-bytes")
        config = MrFidConfig.model_validate(_valid_config_dict(
            tmp_path, radimagenet_weights=str(weights),
        ))
        result = MrFidInstrument(config, backbone=_ChannelMeanBackbone()).run()
        assert result.radimagenet_weights_sha256 == hashlib.sha256(
            b"fixture-weights-bytes"
        ).hexdigest()
        assert result.device == "cpu"
        assert result.dtype == "float32"
        assert result.real_features_dir == "realfeat"
        assert result.synth_features_dir == "synthfeat"
        assert result.ignore_existing is False

    @pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="mps 不可用的环境无法模拟加速器出口的设备驻留",
    )
    def test_features_parked_on_cpu_per_volume(self, tmp_path):
        """逐卷特征须在缓存/累积前回 CPU：device=cuda 的生产读数若把
        双侧整栈特征留到距离计算才搬运，数百卷 × 三面的 2048 维特征
        是 GiB 级加速器驻留（OOM 级）；缓存 ``.pt`` 也必须落 CPU 形态
        （跨设备可装载）。替身以 mps 设备模拟加速器出口。"""
        config = MrFidConfig.model_validate(_valid_config_dict(tmp_path))
        MrFidInstrument(config, backbone=_AcceleratorLikeBackbone()).run()
        cached = sorted((tmp_path / "features" / "mr").rglob("*.pt"))
        assert cached
        for path in cached:
            feats = torch.load(path, weights_only=True)
            assert all(t.device.type == "cpu" for t in feats)


class TestRadImageNetBackboneContract:
    """装载契约经重构后由 RadImageNetBackbone 承载（原
    RadImageNetFeatureExtractor 测试同源，本处只冒烟委托关系与
    仪器装配的 2048 维出口）。"""

    def test_instrument_with_real_monai_state_dict(self, tmp_path):
        torch.manual_seed(0)
        backbone = RadImageNetBackbone.build_backbone()
        weights = tmp_path / "radimagenet.pt"
        torch.save(backbone.state_dict(), weights)
        instrument = self._instrument_with_weights(tmp_path, weights)
        result = instrument.run()
        assert np.isfinite(result.fid_avg)

    def _instrument_with_weights(self, tmp_path, weights) -> MrFidInstrument:
        config = MrFidConfig.model_validate(
            _valid_config_dict(tmp_path, radimagenet_weights=str(weights)),
        )
        return MrFidInstrument(config)


class TestFidCli:
    def _cli_config(self, tmp_path: Path, **overrides) -> Path:
        torch.manual_seed(0)
        backbone = RadImageNetBackbone.build_backbone()
        weights = tmp_path / "radimagenet.pt"
        torch.save(backbone.state_dict(), weights)
        data = _valid_config_dict(tmp_path, radimagenet_weights=str(weights))
        data.update(overrides)
        path = tmp_path / "fid_config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_fid_subcommand_end_to_end(self, tmp_path, cli):
        config_path = self._cli_config(tmp_path)
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 0, result.stderr
        assert "FID XY:" in result.stdout
        assert "FID Avg:" in result.stdout
        assert "FID 结果已落盘" in result.stdout
        assert (tmp_path / "result.json").is_file()

    def test_fid_config_validation_failure_exit_2(self, tmp_path, cli):
        config_path = self._cli_config(tmp_path, center_slices_ratio=1.5)
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 2
        assert "config 校验失败" in result.stderr
        assert "center_slices_ratio" in result.stderr

    def test_fid_missing_config_exit_2(self, tmp_path, cli):
        result = cli.run("fid", "--config", str(tmp_path / "absent.json"))
        assert result.code == 2
        assert "不存在" in result.stderr

    def test_fid_input_contract_violation_exit_2(self, tmp_path, cli):
        """运行时输入契约（清单/权重/缓存指纹）= exit 2，同 train 族口径。"""
        config_path = self._cli_config(tmp_path, radimagenet_weights=str(tmp_path / "no.pt"))
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 2
        assert "fid 输入契约违反" in result.stderr

    def test_fid_rejects_torchrun_launch(self, tmp_path, cli, monkeypatch):
        """fid 是单进程仪器：多 rank 各自全量提取会并发覆写同一缓存
        指纹/特征与结果工件——torchrun 启动显式拒绝（pretrain 同先例）。"""
        monkeypatch.setenv("RANK", "1")
        config_path = self._cli_config(tmp_path)
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 2
        assert "torchrun" in result.stderr

    def test_fid_corrupt_weights_exit_2(self, tmp_path, cli):
        """损坏/不兼容的权重工件 = 输入契约违反（exit 2）而非裸
        traceback——prepare/pretrain 装载期同口径。"""
        weights = tmp_path / "corrupt.pt"
        weights.write_bytes(b"not-a-torch-archive")
        config_path = self._cli_config(tmp_path, radimagenet_weights=str(weights))
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 2
        assert "fid 输入契约违反" in result.stderr

    def test_fid_corrupt_cache_pt_exit_2(self, tmp_path, cli):
        """执行期的缓存 ``.pt`` 反序列化失败同属输入契约违反：fid 是
        只读评测，执行无训练态可破坏，损坏缓存的 RuntimeError 面归入
        契约违反而非裸 traceback。"""
        config_path = self._cli_config(tmp_path)
        assert cli.run("fid", "--config", str(config_path)).code == 0
        cache = next((tmp_path / "features" / "mr").rglob("*.pt"))
        cache.write_bytes(b"corrupt-payload")
        result = cli.run("fid", "--config", str(config_path))
        assert result.code == 2
        assert "fid 输入契约违反" in result.stderr
