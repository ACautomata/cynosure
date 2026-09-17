"""MR-RATE 线判别器预训练端到端（#122 首跑票的 fixture 层）。

MR 线的 prepare → pretrain 两段在既有测试里各自单测过
（``test_mr_prepare.py`` 装配/工件、``test_pretrain.py`` 的 driver 与
BraTS fixture 端到端），但两段在 MR 线的**贯通**从未验证——prepare
产出的逐条件异形状工件（``condition_latent_shapes`` 契约）流进 pretrain
装配（词表装载、轮转条件集守卫、条件匹配采样、报告条件域与词表指纹）
的执行序是本票首跑的前置。三个测试面（spec「Testing Decisions」：只测
外部行为，主 seam = CLI 命令级与工件契约级）：

1. **恒达标快速路径**（gate 0.01）：prepare → pretrain 贯通，报告按
   多条件线口径产出（per-condition AUC + 白名单 + 词表工件指纹、不记
   单域 latent_shape）且守卫重载走通；
2. **密集步进路径**（gate 0.99 不可达）：per-condition 轮转落事件流、
   噪声注入 knobs（ADR-0009-α）在 MR 线 pretrain 路径消费；
3. **守卫重载拒绝**：词表工件内容漂移的报告守卫被指纹对照拒绝。

BraTS 线与既有 MR prepare 测试零改动。"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.pretrain.artifacts import PretrainProvenance, PretrainReport
from tests.conftest import CliSession, SyntheticMrRateDataset


CONDITIONS = ["t1w/axial", "flair/axial"]
"""夹具词表的两条件（t1w/axial [4,16,16,8]、flair/axial [4,8,8,16]）。"""


class MrPretrainScenario:
    """一次 MR pretrain 端到端场景：合成 MR-RATE 数据集 → CLI prepare →
    CLI pretrain（reward 覆写可注入）→ run 目录工件。

    ``fixtures_dir`` 可注入共享（噪声对比的两 run 必须同一份网络工件：
    判别器初始化来自同一 checkpoint 文件，σ_max 才是唯一差异变量）。"""

    def __init__(
        self, cli: CliSession, tmp_path: Path,
        *, fixtures_dir: Path | None = None,
    ) -> None:
        self._cli = cli
        self.work_dir = Path(tmp_path)
        self._shared_fixtures_dir = fixtures_dir

    def run(
        self,
        *,
        train_patients: int = 6,
        reward_overrides: dict | None = None,
    ) -> CynosureConfig:
        """跑 prepare + pretrain 两段 CLI，返回驱动 pretrain 的 config。"""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        fixtures_dir = self._shared_fixtures_dir or (
            self.work_dir / "fixtures"
        )
        if self._shared_fixtures_dir is None:
            self.vocabulary_path = Fixture().write_artifacts(
                fixtures_dir,
            ).condition_vocabulary_json
        else:
            # 共享工件（噪声对比的控制变量面）：调用方已写好、不重写
            #（重写会重掷网络权重，σ_max 不再是唯一差异）
            self.vocabulary_path = (
                fixtures_dir / "condition_vocabulary.json"
            )
        config = Fixture().config(fixtures_dir, dataset="MR-RATE")
        if reward_overrides:
            config.reward = config.reward.model_copy(update=reward_overrides)
        SyntheticMrRateDataset(config.artifacts.dataset_root).write()
        prepare_config_path = self.work_dir / "prepare_config.json"
        prepare_config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        prepare = self._cli.run("prepare", "--config", str(prepare_config_path))
        assert prepare.code == 0, prepare.stderr
        pretrain_config = config.model_copy(deep=True)
        self.run_dir = self.work_dir / "pretrain_run"
        pretrain_config.reward.pretrain_report_json = str(
            self.run_dir / "pretrain_report.json",
        )
        pretrain_config_path = self.work_dir / "pretrain_config.json"
        pretrain_config_path.write_text(
            pretrain_config.model_dump_json(indent=2), encoding="utf-8",
        )
        self.result = self._cli.run(
            "pretrain", "--config", str(pretrain_config_path),
        )
        assert self.result.code == 0, self.result.stderr
        self.config = pretrain_config
        return pretrain_config

    def report(self) -> PretrainReport:
        return PretrainReport.load(Path(self.config.reward.pretrain_report_json))

    def events(self) -> list[dict]:
        lines = (self.run_dir / "metrics.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        return [json.loads(line) for line in lines if line.strip()]


class TestMrPretrainEndToEnd:
    def test_prepare_then_pretrain_produces_condition_report(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：prepare → pretrain 贯通，报告按多条件线口径产出——
        per-condition AUC + 白名单（词表条件域）+ 词表工件指纹，不记
        单域 latent_shape（#129 口径）；白名单内条件过支撑度规则确认
        （fixture held-out 每条件 1 卷 < 支撑度界 20 → bootstrap CI
        下界口径，gate 0.01 恒达标 → 首测 + 复测确认后零更新步终止）；
        守卫重载（load_discriminator）在 MR 线工件上走通。"""
        scenario = MrPretrainScenario(cli, tmp_path)
        config = scenario.run(reward_overrides={"pretrain_gate_auc": 0.01})
        report = scenario.report()
        assert report.kind == "pretrain_report"
        assert report.group == "modal-label"
        # 多条件线：不落单域全局 latent_shape（口径由词表工件指纹承载）
        assert report.latent_shape is None
        # 报告条件域 = 词表条件集（夹具两条件），白名单全过
        assert set(report.condition_auc) == set(CONDITIONS)
        assert all(0.0 <= auc <= 1.0 for auc in report.condition_auc.values())
        assert report.gate_whitelist == CONDITIONS
        assert report.gate_passed is True
        assert report.steps_completed == 0
        # 全条件确认即停：零更新步 → 零事件
        assert scenario.events() == []
        # 词表工件指纹（#129）：MR 线 fake 形状/token/spacing/sigma 锚的
        # 派生来源绑定在 provenance，与盘上工件内容一致
        provenance = report.provenance
        assert provenance.condition_vocabulary == str(
            config.artifacts.condition_vocabulary_json,
        )
        assert provenance.condition_vocabulary_sha256 == (
            PretrainProvenance.digest(scenario.vocabulary_path)
        )
        # checkpoint 照常落盘，且守卫重载路径走通（词表指纹对照生效）
        scorer = report.load_discriminator(config)
        assert scorer is not None

    def test_dense_steps_rotate_conditions_and_consume_noise(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：密集步进路径——per-condition 轮转落事件流（modality 字段
        轮转、字段齐备），ADR-0009-α 噪声注入 knobs（σ_max > 0）在 MR
        线 pretrain 路径真实消费（σ_max = 0 的同 seed 对照 run 与之权重
        分叉 = 带噪更新前向生效；disc_noise 专属流 σ_max = 0 时零消耗，
        对照 run 即无注入回归锚）。"""
        common = {
            "pretrain_gate_auc": 0.99,  # 不可达：走满步数上限（真训练态）
            "pretrain_max_steps": 4,
            "disc_lr": 2e-4,
        }
        # 两 run 共享同一份网络工件（判别器初始化 = 同一 checkpoint 文件、
        # 同 seed 同数据同词表）——σ_max 是唯一差异变量
        shared_fixtures = tmp_path / "shared_fixtures"
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（库场景先例）
        Fixture().write_artifacts(shared_fixtures)
        noisy = MrPretrainScenario(
            cli, tmp_path / "noisy", fixtures_dir=shared_fixtures,
        )
        noisy.run(reward_overrides={
            **common, "disc_noise_sigma_max": 0.2,
        })
        clean = MrPretrainScenario(
            cli, tmp_path / "clean", fixtures_dir=shared_fixtures,
        )
        clean.run(reward_overrides={**common, "disc_noise_sigma_max": 0.0})
        events = noisy.events()
        # 轮转条件序（目标模态均匀轮转，确定性不耗 RNG）
        assert [event["modality"] for event in events] == [
            CONDITIONS[step % len(CONDITIONS)] for step in range(4)
        ]
        assert all(event["event"] == "pretrain" for event in events)
        assert all(
            {"step", "loss_discriminator", "heldout_auc",
             "buffer_base_occupied", "lr", "elapsed_s"} <= set(event)
            for event in events
        )
        # 噪声注入生效的直接证据：同 seed 下 σ_max 唯一差异 → 权重分叉
        noisy_state = torch.load(
            noisy.run_dir / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        clean_state = torch.load(
            clean.run_dir / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        assert noisy_state.keys() == clean_state.keys()
        assert any(
            not torch.equal(noisy_state[key], clean_state[key])
            for key in noisy_state
        )

    def test_report_guard_rejects_vocabulary_drift(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：词表工件内容漂移的守卫被指纹对照拒绝（#129：工件改动而
        real 侧未变时，报告的白名单与 AUC 对另一份 fake 分布负责——
        装载期显式拒绝，无逃生门）。"""
        scenario = MrPretrainScenario(cli, tmp_path)
        scenario.run(reward_overrides={"pretrain_gate_auc": 0.01})
        report = scenario.report()
        # 报告落盘后改动词表工件内容（语义等价的空白差异也算漂移——
        # 指纹对照按字节内容，不是语义 diff）
        drifted = json.loads(
            scenario.vocabulary_path.read_text(encoding="utf-8"),
        )
        drifted["grid_semantics"] = drifted["grid_semantics"] + "（漂移）"
        scenario.vocabulary_path.write_text(
            json.dumps(drifted, indent=2), encoding="utf-8",
        )
        with pytest.raises(ValueError, match="条件词汇表指纹不符"):
            report.assert_data_provenance(scenario.config)
