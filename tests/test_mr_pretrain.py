"""MR-RATE 线判别器预训练端到端（#122 首跑票的 fixture 层）。

MR 线的 prepare → pretrain 两段在既有测试里各自单测过
（``test_mr_prepare.py`` 装配/工件、``test_pretrain.py`` 的 driver 与
BraTS fixture 端到端），但两段在 MR 线的**贯通**从未验证——prepare
产出的逐条件异形状工件（``condition_latent_shapes`` 契约）流进 pretrain
装配（词表装载、轮转条件集守卫、条件匹配采样、报告条件域与词表指纹）
的执行序是本票首跑的前置。四个测试面（spec「Testing Decisions」：只测
外部行为，主 seam = CLI 命令级与工件契约级）：

1. **恒达标快速路径**（过线阈值 0.01）：prepare → pretrain 贯通，报告按
   多条件线口径产出（per-condition AUC + 过线条件清单 + 词表工件指纹、
   不记单域 latent_shape）且守卫重载走通；
2. **密集步进路径**（过线阈值 0.99 不可达）：per-condition 轮转落事件流、
   同源重构测量批的逐位重放（同 seed 双 run 判别器权重一致，ADR-0012）；
3. **守卫重载拒绝**：词表工件内容漂移的报告守卫被指纹对照拒绝。

BraTS 线与既有 MR prepare 测试零改动。（原第 4 面「白名单空 → 拒跑」
随 ADR-0017 门控链退役删除——train 侧无上岗门槛。）"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.pretrain.artifacts import PretrainProvenance, PretrainReport
from tests.conftest import (
    CliSession,
    SPARSE_POOL_DISC_K,
    SyntheticMrRateDataset,
)


CONDITIONS = ["t1w/axial", "flair/axial"]
"""夹具词表的两条件（t1w/axial [4,16,16,8]、flair/axial [4,8,8,16]）。"""


class MrPretrainScenario:
    """一次 MR pretrain 端到端场景：合成 MR-RATE 数据集 → CLI prepare →
    CLI pretrain（reward 覆写可注入）→ run 目录工件（访问器风格与
    ``test_pretrain.PretrainScenario`` 一致）。

    ``fixtures_dir`` 可注入共享（噪声对比的两 run 必须同一份网络工件：
    判别器初始化来自同一 checkpoint 文件，σ_max 才是唯一差异变量）。"""

    def __init__(
        self, cli: CliSession, tmp_path: Path,
        *, fixtures_dir: Path | None = None,
    ) -> None:
        self._cli = cli
        self._work_dir = Path(tmp_path)
        self._shared_fixtures_dir = fixtures_dir
        self._config: CynosureConfig | None = None
        self._vocabulary: Path | None = None
        self._run_dir: Path | None = None

    def run(
        self,
        *,
        reward_overrides: dict | None = None,
    ) -> CynosureConfig:
        """跑 prepare + pretrain 两段 CLI，返回驱动 pretrain 的 config。"""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        fixtures_dir = (
            self._shared_fixtures_dir or self._work_dir / "fixtures"
        )
        if self._shared_fixtures_dir is None:
            torch.manual_seed(7)  # fixture 网络「固定 seed」机制（库场景先例）
            Fixture().write_artifacts(fixtures_dir)
        # 共享工件（噪声对比的控制变量面）：调用方已写好、不重写
        # （重写会重掷网络权重，σ_max 不再是唯一差异）
        self._vocabulary = fixtures_dir / "condition_vocabulary.json"
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        if reward_overrides:
            config.reward = config.reward.model_copy(update=reward_overrides)
        SyntheticMrRateDataset(config.artifacts.dataset_root).write()
        prepare_config_path = self._work_dir / "prepare_config.json"
        prepare_config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        prepare = self._cli.run("prepare", "--config", str(prepare_config_path))
        assert prepare.code == 0, prepare.stderr
        pretrain_config = config.model_copy(deep=True)
        self._run_dir = self._work_dir / "pretrain_run"
        pretrain_config.reward.pretrain_report_json = str(
            self._run_dir / "pretrain_report.json",
        )
        # 容量守卫按 K×卡数把门（#238）：MR 稀疏条件池 4 条/条件，
        # 口径单点见 conftest.SPARSE_POOL_DISC_K
        pretrain_config.reward.disc_batch_size_k = SPARSE_POOL_DISC_K
        pretrain_config_path = self._work_dir / "pretrain_config.json"
        pretrain_config_path.write_text(
            pretrain_config.model_dump_json(indent=2), encoding="utf-8",
        )
        result = self._cli.run(
            "pretrain", "--config", str(pretrain_config_path),
        )
        assert result.code == 0, result.stderr
        self._config = pretrain_config
        return pretrain_config

    def config(self) -> CynosureConfig:
        assert self._config is not None
        return self._config

    def vocabulary_path(self) -> Path:
        assert self._vocabulary is not None
        return self._vocabulary

    def run_dir_path(self) -> Path:
        assert self._run_dir is not None
        return self._run_dir

    def report(self) -> PretrainReport:
        return PretrainReport.load(
            Path(self.config().reward.pretrain_report_json),
        )

    def events(self) -> list[dict]:
        lines = (self.run_dir_path() / "metrics.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        return [json.loads(line) for line in lines if line.strip()]


class TestMrPretrainEndToEnd:
    def test_prepare_then_pretrain_produces_condition_report(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：prepare → pretrain 贯通，报告按多条件线口径产出——
        per-condition recon-AUC（#171：held-out 原始 vs 冻结基座同源
        重构体）+ 过线条件清单（词表条件域）+ 判据口径标识 + 词表工件
        指纹（承载形状口径、不记单域 latent_shape，#129）；过线条件经
        支撑度规则确认（fixture held-out 每条件 2 卷 < 支撑度界 20 →
        bootstrap CI 下界口径，阈值 0.01 恒达标 → 首测 + 复测确认后零
        更新步终止）；守卫重载（load_discriminator）在 MR 线工件上走通。"""
        scenario = MrPretrainScenario(cli, tmp_path)
        config = scenario.run(reward_overrides={"pretrain_pass_threshold": 0.01})
        report = scenario.report()
        assert report.kind == "pretrain_report"
        assert report.group == "modal-label"
        # 多条件线：不落单域全局 latent_shape（口径由词表工件指纹承载）
        assert report.latent_shape is None
        # 报告条件域 = 词表条件集（夹具两条件），过线条件全过
        assert set(report.condition_auc) == set(CONDITIONS)
        assert all(0.0 <= auc <= 1.0 for auc in report.condition_auc.values())
        assert report.conditions_passed == CONDITIONS
        assert report.all_conditions_passed is True
        assert report.steps_completed == 0
        # 判据口径与支撑度卷数（#171 AC2/AC3）：recon-AUC 标识 + 逐条件
        # held-out 全量卷数（MR 线每条件 2 卷 < 支撑度界 20 → CI 下界口径）
        assert report.auc_criterion == "recon_auc"
        assert report.condition_volumes == {
            condition: 2 for condition in CONDITIONS
        }
        # 全条件确认即停：零更新步 → 零事件
        assert scenario.events() == []
        # 词表工件指纹（#129）：MR 线 fake 形状/token/spacing/sigma 锚的
        # 派生来源绑定在 provenance，与盘上工件内容一致
        provenance = report.provenance
        assert provenance.condition_vocabulary == str(
            config.artifacts.condition_vocabulary_json,
        )
        assert provenance.condition_vocabulary_sha256 == (
            PretrainProvenance.digest(scenario.vocabulary_path())
        )
        # checkpoint 照常落盘，且守卫重载路径走通（词表指纹对照生效）
        scorer = report.load_discriminator(config)
        assert scorer is not None

    def test_dense_steps_rotate_conditions_and_replay(
        self, cli: CliSession, tmp_path: Path,
        deterministic_cpu_replay: None,
        checkpoint_parity: Callable[[dict, dict], None],
    ) -> None:
        """AC：密集步进路径——per-condition 轮转落事件流（modality 字段
        轮转、字段齐备）；更新批 = 装配原语的配对批（重构构造走专属
        recon 流、先抽 s 后抽 ε，ADR-0012）在 MR 线 pretrain 路径确定性
        重放：同 seed 同 config 双 run 判别器 checkpoint 容差一致
        （``checkpoint_parity``，ADR-0018 决策 9；σ_max 对照锚随注入
        退役——更新前向恒干净域，σ_max 不再有更新链路消费面）。"""
        common = {
            "pretrain_pass_threshold": 0.99,  # 不可达：走满步数上限（真训练态）
            "pretrain_max_steps": 4,
            "disc_lr": 2e-4,
        }
        # 两 run 共享同一份网络工件（判别器初始化 = 同一 checkpoint 文件、
        # 同 seed 同数据同词表）——重放逐位一致
        shared_fixtures = tmp_path / "shared_fixtures"
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（库场景先例）
        Fixture().write_artifacts(shared_fixtures)
        first = MrPretrainScenario(
            cli, tmp_path / "first", fixtures_dir=shared_fixtures,
        )
        # 全局 RNG 每 run 重置（spectral u 惰性初始化消耗全局流——同
        # 进程双 run 间状态漂移会破坏重放逐位锚）
        torch.manual_seed(0)
        first.run(reward_overrides=common)
        second = MrPretrainScenario(
            cli, tmp_path / "second", fixtures_dir=shared_fixtures,
        )
        torch.manual_seed(0)
        second.run(reward_overrides=common)
        events = first.events()
        # 轮转条件序（目标模态均匀轮转，确定性不耗 RNG）
        assert [event["modality"] for event in events] == [
            CONDITIONS[step % len(CONDITIONS)] for step in range(4)
        ]
        assert all(event["event"] == "pretrain" for event in events)
        assert all(
            {"step", "loss_discriminator", "heldout_auc",
             "lr", "elapsed_s"} <= set(event)
            for event in events
        )
        # 配对批重构链路的确定性证据：同 seed 双 run → 权重逐位一致
        first_state = torch.load(
            first.run_dir_path() / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        second_state = torch.load(
            second.run_dir_path() / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        # checkpoint 对照口径单点见 conftest.checkpoint_parity
        checkpoint_parity(first_state, second_state)

    def test_report_guard_rejects_vocabulary_drift(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：词表工件内容漂移的守卫被指纹对照拒绝（#129：工件改动而
        real 侧未变时，报告的过线判定与 AUC 对另一份 fake 分布负责——
        装载期显式拒绝，无逃生门）。"""
        scenario = MrPretrainScenario(cli, tmp_path)
        scenario.run(reward_overrides={"pretrain_pass_threshold": 0.01})
        report = scenario.report()
        # 报告落盘后改动词表工件内容（语义等价的空白差异也算漂移——
        # 指纹对照按字节内容，不是语义 diff）
        drifted = json.loads(
            scenario.vocabulary_path().read_text(encoding="utf-8"),
        )
        drifted["grid_semantics"] = drifted["grid_semantics"] + "（漂移）"
        scenario.vocabulary_path().write_text(
            json.dumps(drifted, indent=2), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="条件词汇表指纹不符"):
            report.assert_data_provenance(scenario.config())

