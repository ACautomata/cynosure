"""条件词汇表工件装载测试（#127，spec #125 实现决策 1 的 prefactor）。

好测试只测外部行为（spec 测试决策）：工件装载产出什么五元组、对账在
什么输入下以可读错误拒绝——不测内部函数编排。上游权威数值（token 9/
10/11/20/16、官方推荐 FOV 表）与本仓导出的普查众数网格在测试内独立
登记，交叉复核工件内容——代码内无常量副本，数值的唯一运行时来源是
工件本身（映射单一来源验收）。
"""

import csv
import json
import statistics
from pathlib import Path

import pytest
from pydantic import ValidationError

from cynosure.conditions import (
    PRODUCTION_CONDITION_COUNT,
    ModalityMapping,
    MrConditionVocabulary,
)
from cynosure.fixtures import Fixture

# 仓库登记工件（#127 交付物本体；锚仓库根，不依赖 pytest 的 cwd）
_REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_VOCAB_PATH = _REPO_ROOT / "data/conditions/mrrate_conditions.json"
PRODUCTION_CENSUS_PATH = (
    _REPO_ROOT
    / "data/eval/mrrate-baseline/census_latent_grid_distribution.csv"
)
EVAL_MANIFEST_PATH = (
    _REPO_ROOT / "data/eval/mrrate-baseline/eval_manifest.csv"
)
MR_MODALITY_MAPPING_PATH = (
    _REPO_ROOT / "configs/mrrate-base/modality_mapping.json"
)

# 上游权威（NV-Generate-CTMR configs/modality_mapping.json 的 MR-RATE
# whole-brain 条目；skull-stripped 29–33 不进本轮词汇）——测试守卫登记处
UPSTREAM_TOKENS: dict[str, int] = {
    "t1w": 9, "t2w": 10, "flair": 11, "swi": 20, "mra": 16,
}

# #81 白名单全量：11 生成条件（10 头部 (modality, plane) 格 + MRA）
WHITELIST: frozenset[str] = frozenset({
    "t1w/axial", "t1w/sagittal", "t1w/coronal",
    "t2w/axial", "t2w/sagittal", "t2w/coronal",
    "flair/axial", "flair/sagittal", "flair/coronal",
    "swi/axial", "mra/all-planes",
})

# 五元组期望值表（FOV = 官方 NV-Generate-CTMR docs/inference.md「Recommended
# FOV for MR rflow-mr-brain」表逐行抄录（RAS 轴序），与工件写入同源但独立
# 登记：240×240×174 / 176×250×250 / 240×200×240（T1 ax/sag/cor）、
# 240×240×158 / 162×240×240 / 200×180×200（T2）、250×250×175 /
# 176×250×250 / 250×200×250（FLAIR）、230×230×145（SWI）；MRA 无
# all-planes 官方行，取 #78 评估 manifest 中位——该格另经
# test_mra_fov_matches_eval_manifest_median 程序化对账仓库内独立工件）；
# 网格 = #78 普查工件逐条件众数 latent 网格 ×4（RAS 轴序，薄轴按 FOV
# 归位）；spacing = FOV / 网格）。
EXPECTED: dict[str, dict] = {
    "t1w/axial": {
        "fov": (240.0, 240.0, 174.0), "grid": (256, 256, 128),
    },
    "t1w/sagittal": {
        "fov": (176.0, 250.0, 250.0), "grid": (128, 256, 256),
    },
    "t1w/coronal": {
        "fov": (240.0, 200.0, 240.0), "grid": (512, 256, 512),
    },
    "t2w/axial": {
        "fov": (240.0, 240.0, 158.0), "grid": (512, 512, 128),
    },
    "t2w/sagittal": {
        "fov": (162.0, 240.0, 240.0), "grid": (128, 256, 256),
    },
    "t2w/coronal": {
        "fov": (200.0, 180.0, 200.0), "grid": (256, 128, 256),
    },
    "flair/axial": {
        "fov": (250.0, 250.0, 175.0), "grid": (256, 256, 128),
    },
    "flair/sagittal": {
        "fov": (176.0, 250.0, 250.0), "grid": (128, 512, 512),
    },
    "flair/coronal": {
        "fov": (250.0, 200.0, 250.0), "grid": (384, 256, 384),
    },
    "swi/axial": {
        "fov": (230.0, 230.0, 145.0), "grid": (256, 256, 128),
    },
    "mra/all-planes": {
        "fov": (158.4, 220.0, 220.0), "grid": (128, 384, 384),
    },
}


def _mirror_repo(tmp_path: Path) -> None:
    """tmp 内镜像仓库 ``data/`` 布局：工件放 ``data/conditions/``，普查
    CSV 以 symlink 接入 ``data/eval/mrrate-baseline/``——工件的相对
    census 引用在新位置照常解析（对账路径与仓库同构）。"""
    census_dir = tmp_path / "data" / "eval" / "mrrate-baseline"
    census_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "conditions").mkdir(parents=True, exist_ok=True)
    target = census_dir / PRODUCTION_CENSUS_PATH.name
    if not target.exists():
        target.symlink_to(PRODUCTION_CENSUS_PATH.resolve())


def _edited(tmp_path: Path, mutate, name: str = "vocab.json") -> Path:
    """读生产工件 → 篡改 → 落盘（tmp 镜像布局），返回路径。"""
    _mirror_repo(tmp_path)
    data = json.loads(PRODUCTION_VOCAB_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "data" / "conditions" / name
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TestProductionVocabularyLoads:
    """仓库登记工件装载成功，五元组逐位与独立登记的期望值一致。"""

    @classmethod
    def _vocab(cls) -> MrConditionVocabulary:
        return MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)

    def test_modality_mapping_artifact_loads(self) -> None:
        """MR 线 modality mapping 装载工件（configs/mrrate-base/）经
        ``ModalityMapping.load`` 走通：四序列键满足构造校验（BraTS
        口径值）；MR 五模态条目在册留档（本线的条件 token 实际取数走
        词表工件的 ``modality_tokens``，#127 单一来源——本工件是装载面
        满足 + 跨域映射留档，非 MR token 的消费来源）。"""
        mapping = ModalityMapping.load(MR_MODALITY_MAPPING_PATH)
        assert mapping.label("t1n") == 29
        assert mapping.label("t1c") == 34
        assert mapping.label("t2f") == 31
        # 同键两域不同 token：本工件取 MR 值（本线口径优先，README 登记）
        assert mapping.label("t2w") == 10

    def test_loads_eleven_whitelist_conditions(self) -> None:
        vocab = self._vocab()
        assert set(vocab.names()) == WHITELIST
        assert len(vocab.conditions) == 11

    def test_token_mapping_is_upstream_authoritative(self) -> None:
        """token 映射 = 上游 modality_mapping.json 的 MR-RATE whole-brain
        条目；skull-stripped 29–33 不在本轮词汇。"""
        assert dict(self._vocab().tokens) == UPSTREAM_TOKENS

    def test_condition_tokens_derive_from_mapping(self) -> None:
        """条件 token 恒取该模态 whole-brain 条目（#81 swap 生成口径）：
        文件内不重复登记 token，装载派生——单一来源。"""
        for condition in self._vocab().conditions:
            assert condition.token == UPSTREAM_TOKENS[condition.modality]

    def test_quintuples_match_expected_table(self) -> None:
        """逐条件五元组：FOV / 统一网格与独立登记表逐位一致；等效
        spacing = FOV / 网格（往返浮点一致）。"""
        for condition in self._vocab().conditions:
            expected = EXPECTED[condition.name]
            assert condition.fov_mm == expected["fov"], condition.name
            assert condition.grid_xyz == expected["grid"], condition.name
            for fov, grid, spacing in zip(
                condition.fov_mm, condition.grid_xyz, condition.spacing_mm,
            ):
                assert spacing == pytest.approx(fov / grid), condition.name

    def test_latent_grid_is_grid_over_four(self) -> None:
        """latent 网格 = 统一网格 / 4（VAE 空间压缩），rollout 初始噪声
        形状的解析来源（#129 消费面）。"""
        vocab = self._vocab()
        assert vocab.latent_shape("t1w/axial") == (4, 64, 64, 32)
        assert vocab.latent_shape("t1w/coronal") == (4, 128, 64, 128)
        assert vocab.latent_shape("mra/all-planes") == (4, 32, 96, 96)

    def test_grids_match_census_modes(self) -> None:
        """逐条件网格与 #78 普查工件的期望网格（众数）一致：装载期对账
        通过——「条件 → latent 形状」的权威对照在案。"""
        vocab = self._vocab()
        # 独立从普查 CSV 导出众数，与装载结果对照（不经过装载器代码）
        modes: dict[str, tuple[int, ...]] = {}
        rows: dict[str, dict[tuple[int, ...], int]] = {}
        with open(PRODUCTION_CENSUS_PATH, encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
            for line in fh:
                fields = line.strip().split(",")
                row = dict(zip(header, fields))
                stratum = row["stratum"]
                grid = tuple(
                    int(part) for part in row["latent_grid_xyz_ch4"].split("x")[1:]
                )
                rows.setdefault(stratum, {})[grid] = int(row["n_volumes"])
        for stratum, counts in rows.items():
            modes[stratum.casefold()] = max(counts, key=lambda grid: counts[grid])
        assert set(modes) == {name.casefold() for name in vocab.names()}
        for condition in vocab.conditions:
            mode = modes[condition.name.casefold()]
            assert sorted(condition.grid_xyz) == sorted(
                axis * 4 for axis in mode
            ), condition.name

    def test_reload_returns_independent_instance(self) -> None:
        """同进程先后装载互不污染（#119 验收语义在工件形态下延续）：
        词表容器不可变，篡改一方 tokens 视图不影响另一方。"""
        first = self._vocab()
        second = self._vocab()
        mutated = dict(first.tokens)
        mutated["t1w"] = 99
        assert second.tokens["t1w"] == 9
        assert first.tokens["t1w"] == 9
        assert dict(second.tokens) == UPSTREAM_TOKENS

    def test_by_name_unknown_condition_is_readable(self) -> None:
        with pytest.raises(KeyError, match="pd/axial"):
            self._vocab().by_name("pd/axial")

    def test_mra_fov_matches_eval_manifest_median(self) -> None:
        """MRA/all-planes 的 FOV 独立对账锚：官方表无 all-planes 行，工件
        数值取 #78 评估 manifest（仓库内独立工件）的 MRA 逐轴中位——此处
        从 manifest 程序化重算，FOV 抄录错误测得出（其余 10 格的 FOV 锚
        是 EXPECTED 表内登记的官方表数值）。"""
        fovs: list[tuple[float, float, float]] = []
        with open(EVAL_MANIFEST_PATH, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["stratum"] != "MRA/ALL-PLANES":
                    continue
                raw = row["array_fov_mm"].strip().strip("[]")
                fovs.append(tuple(float(x) for x in raw.split(",")))
        assert len(fovs) == 14  # MRA 全取（留出池仅 14 卷，#78 分层口径）
        median_fov = tuple(
            statistics.median(axis) for axis in zip(*fovs)
        )
        mra = self._vocab().by_name("mra/all-planes")
        assert mra.fov_mm == pytest.approx(median_fov)


class TestProductionRejections:
    """生产模式（fixture_mode=False）装载的字段级拒绝面：缺格、网格
    不符、字段缺失、token 缺模态、几何自洽破坏。"""

    def test_missing_condition_rejected(self, tmp_path: Path) -> None:
        """缺格拒绝：删掉 swi/axial 一条 → 装载拒绝且点名缺的条件。"""
        def remove_swi(data: dict) -> None:
            data["conditions"] = [
                entry for entry in data["conditions"]
                if entry["name"] != "swi/axial"
            ]
        path = _edited(tmp_path, remove_swi)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "swi/axial" in str(exc_info.value)

    def test_extra_condition_rejected(self, tmp_path: Path) -> None:
        """多格拒绝：mra/axial 类型合法但不在 #81 白名单（MRA 只
        all-planes 一格）——普查对账双向等值。"""
        def add_mra_axial(data: dict) -> None:
            data["conditions"].append({
                "name": "mra/axial", "modality": "mra", "plane": "axial",
                "fov_mm": [220.0, 220.0, 158.0], "fov_source": "official",
                "grid_xyz": [256, 256, 128],
            })
        path = _edited(tmp_path, add_mra_axial)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "mra/axial" in str(exc_info.value)

    def test_grid_mismatch_rejected(self, tmp_path: Path) -> None:
        """网格不符拒绝：t2w/axial 网格偏离普查众数（512×512×128 →
        256×256×128）→ 装载拒绝且点名条件与两侧网格。"""
        def shift_grid(data: dict) -> None:
            for entry in data["conditions"]:
                if entry["name"] == "t2w/axial":
                    entry["grid_xyz"] = [256, 256, 128]
        path = _edited(tmp_path, shift_grid)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        message = str(exc_info.value)
        assert "t2w/axial" in message
        assert "512" in message and "256" in message

    def test_missing_field_rejected_field_level(self, tmp_path: Path) -> None:
        """字段缺失字段级拒绝：删一条件的 fov_mm → pydantic 字段级
        错误（loc 指向 conditions 条目的 fov_mm）。"""
        def drop_fov(data: dict) -> None:
            data["conditions"][0].pop("fov_mm")
        path = _edited(tmp_path, drop_fov)
        with pytest.raises(ValidationError) as exc_info:
            MrConditionVocabulary.load(path)
        locations = [tuple(err["loc"]) for err in exc_info.value.errors()]
        assert any(
            loc[0] == "conditions" and loc[-1] == "fov_mm" for loc in locations
        ), locations

    def test_unknown_field_rejected_field_level(self, tmp_path: Path) -> None:
        """多余字段拒绝（extra=forbid）：拼错字段名直接字段级拒绝。"""
        def typo_field(data: dict) -> None:
            data["conditions"][0]["fov_nmm"] = [1.0, 1.0, 1.0]
        path = _edited(tmp_path, typo_field)
        with pytest.raises(ValidationError) as exc_info:
            MrConditionVocabulary.load(path)
        locations = [tuple(err["loc"]) for err in exc_info.value.errors()]
        assert any("fov_nmm" in str(loc) for loc in locations)

    def test_token_mapping_missing_modality_rejected(self, tmp_path: Path) -> None:
        """token 映射缺模态可读拒绝：错误信息点名缺的模态与五模态集。"""
        def drop_mra(data: dict) -> None:
            data["modality_tokens"].pop("mra")
        path = _edited(tmp_path, drop_mra)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "mra" in str(exc_info.value)

    def test_condition_modality_without_token_rejected(
        self, tmp_path: Path,
    ) -> None:
        """条件模态在映射中无 token → 可读拒绝（映射取数键守卫）。"""
        def orphan_modality(data: dict) -> None:
            data["modality_tokens"].pop("swi")
        path = _edited(tmp_path, orphan_modality)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "swi" in str(exc_info.value)

    def test_grid_not_multiple_of_32_rejected(self, tmp_path: Path) -> None:
        """统一网格非 32 倍数拒绝（latent 逐轴 8 倍数的 UNet 跳连约束，
        #80 实测口径：latent 44 硬错）。fixture 模式隔离验证——无普查
        对账，结构校验独立拒绝。"""
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures-grid")
        path = artifacts.condition_vocabulary_json
        data = json.loads(path.read_text(encoding="utf-8"))
        data["conditions"][0]["grid_xyz"] = [64, 64, 24]
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path, fixture_mode=True)
        assert "32" in str(exc_info.value)

    def test_thin_axis_mismatch_rejected(self, tmp_path: Path) -> None:
        """FOV 薄轴与网格薄轴错位拒绝（轴序登记错误守卫）：t1w/axial
        FOV 薄轴挪到 x、网格薄轴仍在 z。FOV 单改、网格不动——普查对账
        通过，拒绝来自结构校验本身。"""
        def shift_thin_fov(data: dict) -> None:
            for entry in data["conditions"]:
                if entry["name"] == "t1w/axial":
                    entry["fov_mm"] = [174.0, 240.0, 240.0]
        path = _edited(tmp_path, shift_thin_fov, name="axis.json")
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "t1w/axial" in str(exc_info.value)

    def test_spacing_out_of_upstream_domain_rejected(
        self, tmp_path: Path,
    ) -> None:
        """等效 spacing 越出上游允许域 [0.4, 5.0] mm 拒绝（上游
        check_input_mr 的 spacing 约束）。FOV 单改 ×10——网格不动，
        普查对账通过，拒绝来自 spacing 域校验。"""
        def wild_fov(data: dict) -> None:
            for entry in data["conditions"]:
                if entry["name"] == "swi/axial":
                    entry["fov_mm"] = [2300.0, 2300.0, 1450.0]
        path = _edited(tmp_path, wild_fov, name="spacing.json")
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "swi/axial" in str(exc_info.value)

    def test_name_modality_plane_inconsistency_rejected(
        self, tmp_path: Path,
    ) -> None:
        """分组名与 (modality, plane) 漂移拒绝：name 是 per-condition
        统计与轮转的消费键（ADR-0008），与字段错位即拒。"""
        def rename(data: dict) -> None:
            for entry in data["conditions"]:
                if entry["name"] == "swi/axial":
                    entry["name"] = "swi/coronal"
        path = _edited(tmp_path, rename, name="name.json")
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "swi" in str(exc_info.value)

    def test_missing_census_reference_rejected(self, tmp_path: Path) -> None:
        """生产工件缺普查引用 → 拒绝（装载期对账是生产模式的必然步骤，
        引用字段缺失 = 字段缺失拒绝的一种）。"""
        def drop_census(data: dict) -> None:
            data.pop("census_grid_csv")
        path = _edited(tmp_path, drop_census)
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(path)
        assert "census" in str(exc_info.value)

    def test_missing_census_file_rejected(self, tmp_path: Path) -> None:
        """普查工件路径不可读 → 拒绝：census 引用相对工件目录解析，
        无普查镜像布局的工件即暴露（引用悬空 = 生产对账无法执行）。"""
        path = tmp_path / "data" / "conditions" / "vocab.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            PRODUCTION_VOCAB_PATH.read_text(encoding="utf-8"), encoding="utf-8",
        )
        with pytest.raises((ValidationError, ValueError, OSError)):
            MrConditionVocabulary.load(path)


class TestFixtureVocabularyChannel:
    """fixture 模式的小词汇表注入通道（沿用 fixture_mode 显式声明
    纪律）：fixture 工件只在 fixture_mode=True 可装载；生产模式装载
    小词汇表即拒绝（缩小词汇属 fixture，须经显式声明）。"""

    def test_fixture_artifact_loads_in_fixture_mode(self, tmp_path: Path) -> None:
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures")
        vocab = MrConditionVocabulary.load(
            artifacts.condition_vocabulary_json, fixture_mode=True,
        )
        assert len(vocab.conditions) == 2
        assert {c.plane for c in vocab.conditions} == {"axial"}
        # fixture 网格异形状（#129 全链验收的输入面）：t1w = [64,64,32]
        # （latent (4,16,16,8) = fixture 全局形状）；flair = [32,32,64]
        # （薄厚轴互换，latent (4,8,8,16)、锚 1024 ≠ 2048）
        t1w = vocab.by_name("t1w/axial")
        flair = vocab.by_name("flair/axial")
        assert t1w.grid_xyz == (64, 64, 32)
        assert vocab.latent_shape("t1w/axial") == (4, 16, 16, 8)
        assert flair.grid_xyz == (32, 32, 64)
        assert vocab.latent_shape("flair/axial") == (4, 8, 8, 16)
        assert vocab.latent_numel("flair/axial") == 1024

    def test_fixture_tokens_match_production(self, tmp_path: Path) -> None:
        """fixture 文档值与生产工件交叉对账（两处登记不漂移）。"""
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures")
        vocab = MrConditionVocabulary.load(
            artifacts.condition_vocabulary_json, fixture_mode=True,
        )
        production = MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)
        assert dict(vocab.tokens) == dict(production.tokens)

    def test_fixture_artifact_rejected_in_production_mode(
        self, tmp_path: Path,
    ) -> None:
        """生产模式装载 fixture 小词汇表 → 拒绝（可读错误点名两缺口：
        条件非 11 全量、缺普查对照）。"""
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures")
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            MrConditionVocabulary.load(artifacts.condition_vocabulary_json)
        message = str(exc_info.value)
        assert "fixture" in message

    def test_production_artifact_loads_in_fixture_mode(self) -> None:
        """fixture 模式不强制小词汇表：生产 11 条件工件照常装载
        （fixture 纪律放宽的是「可缩小」，不是「必须缩小」）。"""
        vocab = MrConditionVocabulary.load(
            PRODUCTION_VOCAB_PATH, fixture_mode=True,
        )
        assert len(vocab.conditions) == 11

    def test_fixture_mr_config_carries_vocabulary_binding(
        self, tmp_path: Path,
    ) -> None:
        """Fixture.config(dataset="MR-RATE") 的词表绑定分支回归锚（#127
        验收 4 的 config 面）：MR fixture config 通过 schema 且绑定 fixture
        词汇工件；BraTS 默认 config 不携带绑定（互斥语义）。"""
        fixture = Fixture()
        fixture.write_artifacts(tmp_path / "fixtures")
        mr_config = fixture.config(tmp_path / "fixtures", dataset="MR-RATE")
        assert mr_config.experiment.dataset == "MR-RATE"
        assert mr_config.artifacts.condition_vocabulary_json == Path(
            tmp_path / "fixtures" / "condition_vocabulary.json"
        )
        # 绑定指向的 fixture 词汇工件可经 fixture 通道装载回环
        vocab = MrConditionVocabulary.load(
            mr_config.artifacts.condition_vocabulary_json, fixture_mode=True,
        )
        assert 0 < len(vocab.conditions) < PRODUCTION_CONDITION_COUNT
        brats_config = fixture.config(tmp_path / "fixtures")
        assert brats_config.experiment.dataset == "BraTS2023"
        assert brats_config.artifacts.condition_vocabulary_json is None


class TestLatentNumelPerCondition:
    """数值锚逐条件派生面（#129）：sigma 日程锚 = 该条件空间 numel，
    从条件词汇表单一派生入口消费（装载期与运行时同一锚语义，防日程
    静默错位）。"""

    def test_latent_numel_matches_spatial_numel(self) -> None:
        vocab = MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)
        for condition in vocab.conditions:
            shape = vocab.latent_shape(condition.name)
            assert vocab.latent_numel(condition.name) == (
                shape[1] * shape[2] * shape[3]
            )

    def test_conditions_have_multiple_distinct_anchors(self) -> None:
        """生产词表的条件锚不止一种值（11 期望网格下 numel 有碰撞——
        如 t1w/axial 与 t1w/sagittal 同为 131072——但绝不止一个值）：
        逐条件锚校验的输入面真实存在，不是「单一锚换名」。"""
        vocab = MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)
        anchors = {
            vocab.latent_numel(condition.name)
            for condition in vocab.conditions
        }
        assert len(anchors) > 1

    def test_numel_unknown_condition_rejected(self) -> None:
        vocab = MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)
        with pytest.raises(KeyError, match="not-in-vocabulary"):
            vocab.latent_numel("not-in-vocabulary")


class TestSpacingConditionFace:
    """spacing 条件属性解析面（issue #130，spec #125 决策 6）：条件 →
    ×1e2 条件张量值——real 侧 manifest 条目与 fake 侧 rollout 条件的
    同值来源；换算因子 ×1e2 在测试内独立登记（换算漂移即测出）。"""

    @classmethod
    def _vocab(cls) -> MrConditionVocabulary:
        return MrConditionVocabulary.load(PRODUCTION_VOCAB_PATH)

    def test_values_are_fov_over_grid_times_1e2(self) -> None:
        """逐条件：spacing_condition = FOV / 网格 ×1e2（与 spacing_mm
        同一定义式、与 per-case 侧车同一条件单位）。"""
        for condition in self._vocab().conditions:
            expected = tuple(
                fov / grid * 100.0
                for fov, grid in zip(condition.fov_mm, condition.grid_xyz)
            )
            assert (
                self._vocab().spacing_condition(condition.name)
                == pytest.approx(expected)
            ), condition.name

    def test_anchor_value_thin_axis_has_largest_spacing(self) -> None:
        """锚值抽查：t1w/axial = (240/256, 240/256, 174/128)×1e2——薄轴
        （FOV 174 mm、网格 128）等效 spacing 最大（物理分辨率最粗）。"""
        assert self._vocab().spacing_condition("t1w/axial") == pytest.approx(
            (93.75, 93.75, 135.9375)
        )

    def test_fixture_vocabulary_same_face(self, tmp_path: Path) -> None:
        """fixture 词汇表同一解析面：FOV = 网格 = (64, 64, 32) → 单位
        spacing ×1e2 = (100, 100, 100)。"""
        artifacts = Fixture().write_artifacts(tmp_path / "fixtures")
        vocab = MrConditionVocabulary.load(
            artifacts.condition_vocabulary_json, fixture_mode=True,
        )
        assert vocab.spacing_condition("t1w/axial") == (100.0, 100.0, 100.0)

    def test_unknown_condition_rejected(self) -> None:
        with pytest.raises(KeyError, match="pd/axial"):
            self._vocab().spacing_condition("pd/axial")
