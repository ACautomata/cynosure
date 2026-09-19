"""pretrain 子命令端到端与产物契约测试（ticket #58/#87 验收标准聚合）。

fixture 合成流：小 pool + 小 fake 集 → per-condition 密集步进 → 产物落盘
→ 重载 → 守卫生效；AUC 随步数上升可观测（先例量级）。终止语义（全部
轮转条件过线 / 步数上限）、组1/组2 同路径、kind 守卫、旧格式报告拒绝、
预训练事件的回退记账口径由专属用例覆盖；真实收敛动力学留给 DCU 实跑
（spec「Testing Decisions」：fixture 只验判定逻辑与数据流）。

per-condition 步进（ADR-0008-04）分两层锁：**轮转与归因**在端到端用例
（事件流 modality 逐轮转序断言）；**终止状态机**在替身注入用例（测量 /
判定 / 更新三 seam 换脚本替身，确认棘轮、部分白名单、复测掉线、耗尽
补测的语义逐项驱动）。过线判定的支撑度规则本身（CI 下界 / 点估计
分派）由 test_support_rule 锁，此处锁「判定经 SupportRule 消费」的 seam。

事件类型 × 回退记账口径（ticket #59）分两层锁：本文件锁**口径表**本身
（三型事件的登记与各自的保留边界、未登记类型的不删语义）；resume seam 上
的真回退（真 `--resume` 不误删预训练事件）见 test_resume_roundtrip。
"""

import copy
import json
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from cynosure.config import ConfigLoader, CynosureConfig, MODALITIES
from cynosure.distributed import DistributedContext
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkAssembler
from cynosure.policy.numerics import AMP_DTYPES
from cynosure.pretrain import (
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.pretrain.driver import PretrainDriver
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.assembly import PairBatch
from cynosure.reward.overfit import OverfitMonitor
from cynosure.reward.update import OnlineUpdate
from cynosure.train import (
    REWIND_ACCOUNTING,
    AmpContext,
    CrossModalConditionSampler,
    IterEvent,
    MilestoneEvent,
    OverfitAlertEvent,
    PretrainEvent,
    RewindAccounting,
    RunArtifacts,
    TrainingRuntime,
)
from cynosure.train.policy import GroupPolicy
from cynosure.train.rng import TrainingRngStreams
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    MINIMAL_CONFIG_DICT,
    RecordingUpdate,
)


def iter_event(iteration: int, stage: int = 1) -> IterEvent:
    """最小合法 iter 事件（rewind 记账口径的对照事件）。"""
    return IterEvent(
        iteration=iteration,
        stage=stage,
        modality="t1n",
        anchor_eval_reward=0.0,
        intra_group_reward_std=1.0,
        heldout_auc=0.5,
        loss={"discriminator": 1.0},
        buffer_current_fraction=0.5,
        buffer_replay_fraction=0.5,
        buffer_base_occupied=32,
        buffer_recent_occupied=0,
        lr=5e-5,
        elapsed_s=0.0,
    )


def pretrain_event(step: int) -> PretrainEvent:
    """最小合法预训练事件（混存同一 metrics.jsonl 的第二步事件类型）。"""
    return PretrainEvent(
        step=step,
        modality="t1n",
        loss_discriminator=1.0,
        heldout_auc=0.5,
        buffer_base_occupied=32,
        buffer_recent_occupied=4,
        lr=5e-5,
        elapsed_s=0.1,
    )


def milestone_event(iteration: int) -> MilestoneEvent:
    """最小合法里程碑事件（回退记账的完成数口径对照事件）。"""
    return MilestoneEvent(iteration=iteration, fid=1.0)


def alert_event(
    iteration: int, stage: int = 1, phase: str = "rl",
) -> OverfitAlertEvent:
    """最小合法 overfit_alert 事件（ADR-0009-β 的分叉报警事件；γ 起带
    相判别字段，默认 RL 相与既有构造点逐字一致）。"""
    return OverfitAlertEvent(
        iteration=iteration,
        stage=stage,
        phase=phase,
        modality="t1n",
        divergence_ema=0.3,
        train_pairwise_acc=0.8,
        heldout_auc=0.5,
    )


def event_type_vocabulary() -> set[str]:
    """指标流事件类型的判别值词汇表（由事件模型实例的 ``event`` 默认值取
    真值——判别字段的 Literal 是那一处的单一来源，测试不另抄字面量）。"""
    return {
        event.event
        for event in (
            iter_event(0), milestone_event(0), pretrain_event(0),
            alert_event(0),
        )
    }


def fresh_run_artifacts(tmp_path: Path) -> RunArtifacts:
    """本文件各用例的最小落盘面：全新 run 目录（config 快照 + 空指标流）。"""
    config = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
    return RunArtifacts.init(config, tmp_path / "run")


class TestPretrainEventContract:
    def test_event_type_discriminant_and_roundtrip(self, tmp_path: Path) -> None:
        """事件判别字段区分于 iter/milestone；同流混存读取无损。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(iter_event(0))
        artifacts.append_event(pretrain_event(0))
        events = artifacts.read_events()
        assert [event["event"] for event in events] == ["iter", "pretrain"]
        assert events[1]["step"] == 0
        assert events[1]["modality"] == "t1n"  # ADR-0008-04：事件按条件归因
        assert events[1]["heldout_auc"] == pytest.approx(0.5)

    def test_rewind_preserves_pretrain_events(self, tmp_path: Path) -> None:
        """预训练事件不参与续训回退记账（spec「实现警点」）：rewind 只删
        iter/milestone 的半截执行史，混存的 pretrain 事件全量保留。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(iter_event(0))
        artifacts.append_event(pretrain_event(0))
        artifacts.append_event(pretrain_event(1))
        artifacts.append_event(iter_event(9))
        removed = artifacts.rewind_events(5, stage=1)
        assert removed == 1  # 只删 iteration 9 的半截 iter 事件
        survivors = artifacts.read_events()
        assert [event["event"] for event in survivors] == [
            "iter", "pretrain", "pretrain",
        ]
        assert [event.get("step") for event in survivors if event["event"] == "pretrain"] == [0, 1]


class TestEventRewindAccounting:
    """事件类型 × 回退记账口径（ticket #59：「可扩不可改名」的记账面）。

    口径表本身在契约层锁：三型事件各登记一个口径、保留边界按各自记账轴
    取、未登记类型退化为不删（新增事件类型必须显式声明口径，不得静默
    继承删除口径）。resume seam 上的真回退见 test_resume_roundtrip。"""

    def test_registry_covers_event_type_vocabulary(self) -> None:
        """记账口径登记表与事件类型词汇表逐项对齐：每型事件的判别值都有
        显式登记的口径（新增事件类型必须声明口径，不得静默继承删除口径）。"""
        assert set(REWIND_ACCOUNTING) == event_type_vocabulary()
        assert REWIND_ACCOUNTING["iter"] is RewindAccounting.ITERATION
        assert REWIND_ACCOUNTING["milestone"] is RewindAccounting.COMPLETION
        assert REWIND_ACCOUNTING["pretrain"] is RewindAccounting.EXEMPT
        assert REWIND_ACCOUNTING["overfit_alert"] is RewindAccounting.ITERATION

    def test_recovery_point_covers_by_event_own_accounting(self) -> None:
        """保留边界按各型自身口径取（恢复点 = 最近 checkpoint 的计数）：
        iter 以 0-based iteration 号记账（号 < 恢复点 = 已在 checkpoint
        覆盖面内）；milestone 以完成数记账（完成数 ≤ 恢复点 = 与恢复点
        checkpoint 同批产出，删即抹掉 FID 历史）；pretrain 不参与记账。"""
        assert RewindAccounting.ITERATION.covers(1, 2) is True
        assert RewindAccounting.ITERATION.covers(2, 2) is False  # 半截：待重写
        assert RewindAccounting.COMPLETION.covers(2, 2) is True  # 同批产出：保留
        assert RewindAccounting.COMPLETION.covers(3, 2) is False
        assert RewindAccounting.EXEMPT.covers(0, 0) is True
        assert RewindAccounting.EXEMPT.covers(999, 0) is True

    def test_rewind_keeps_and_drops_by_event_type(self, tmp_path: Path) -> None:
        """三型事件混存同一流的回退现场（恢复点 2）：iter 删恢复点及之后的
        半截执行史（号 ≥ 2）、milestone 按完成数保留同批（≤ 2）、pretrain
        全量保留且逐字不动；其他 stage 的历史一概不动。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(pretrain_event(0))
        artifacts.append_event(iter_event(0))
        artifacts.append_event(pretrain_event(1))
        artifacts.append_event(iter_event(1))
        artifacts.append_event(milestone_event(2))
        artifacts.append_event(iter_event(2))
        artifacts.append_event(iter_event(0, stage=2))  # 组3 stage-1 的历史
        removed = artifacts.rewind_events(2, stage=1)
        assert removed == 1  # 只有 stage-1 的 iteration 2 半截事件
        survivors = artifacts.read_events()
        assert [(event["event"], event.get("step", event.get("iteration")))
                for event in survivors] == [
            ("pretrain", 0), ("iter", 0), ("pretrain", 1), ("iter", 1),
            ("milestone", 2), ("iter", 0),
        ]

    def test_rewind_to_origin_keeps_pretrain_events(self, tmp_path: Path) -> None:
        """恢复点 0（stage 的首个 iteration 都未进 checkpoint 覆盖面）：
        预训练事件仍须全量保留——按 iter 口径记账时它们的 ``iteration``
        缺省为 0，恰落在删除边界内，口径隔离是唯一挡得住这次误删的机制。
        恢复点 0 在 CLI seam 上难以构造（无 checkpoint 即无恢复入口），
        故在契约层直接锁。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(pretrain_event(0))
        artifacts.append_event(pretrain_event(1))
        artifacts.append_event(iter_event(0))
        assert artifacts.rewind_events(0, stage=1) == 1  # 只删 stage-1 的 iter@0
        assert [event["event"] for event in artifacts.read_events()] == [
            "pretrain", "pretrain",
        ]

    def test_unregistered_event_type_is_never_deleted(self, tmp_path: Path) -> None:
        """未登记的事件类型一律不参与删除：口径表是删除的准入名单，表外
        类型退化为全量保留，而非静默继承 iter 口径被当作半截执行史抹掉。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(iter_event(9))  # 恢复点之外的半截事件：该删
        future = {"event": "future-metric", "iteration": 99}
        with open(artifacts.paths.metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(future) + "\n")
        assert artifacts.rewind_events(1, stage=1) == 1
        assert artifacts.read_events() == [future]


class PretrainReportScenario:
    """报告契约场景：fixture 判别器工件 + channel stats + 可定制报告。"""

    def __init__(self, tmp_path: Path) -> None:
        torch.manual_seed(0)  # fixture 网络工件确定性
        self.tmp_path = tmp_path
        self.fixture_dir = tmp_path / "fixtures"
        Fixture().write_artifacts(self.fixture_dir)
        self.config = Fixture().config(self.fixture_dir)
        self.stats_path = Path(self.config.reward.channel_stats_json)
        self.stats_path.write_text(
            ChannelStats(
                mean=[0.0, 0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0, 1.0],
                num_latents=1,
                latent_shape=Fixture.LATENT_SHAPE,
                source_manifest="real_pool.json",
            ).model_dump_json(indent=2),
            encoding="utf-8",
        )
        self.report_dir = tmp_path / "pretrain_run"
        (self.report_dir / "checkpoints").mkdir(parents=True)
        # 判别器 checkpoint 契约与训练期产物同构：可装载 state_dict
        self.ckpt_path = self.report_dir / "checkpoints" / "pretrain_discriminator.pt"
        torch.save(
            torch.load(
                self.config.artifacts.discriminator_ckpt,
                map_location="cpu", weights_only=True,
            ),
            self.ckpt_path,
        )
        self.report_path = self.report_dir / "pretrain_report.json"

    def loadable_state(self) -> dict:
        return torch.load(self.ckpt_path, map_location="cpu", weights_only=True)

    def report(self, **overrides) -> PretrainReport:
        provenance = PretrainProvenance(
            real_pool_manifest="real_pool.json",
            real_pool_manifest_sha256="0" * 64,
            heldout_manifest="heldout_real.json",
            heldout_manifest_sha256="0" * 64,
            channel_stats=str(self.stats_path),
            channel_stats_sha256=PretrainProvenance.digest(self.stats_path),
            discriminator_config=str(
                self.config.artifacts.discriminator_config_json
            ),
            discriminator_config_sha256=PretrainProvenance.digest(
                self.config.artifacts.discriminator_config_json
            ),
            discriminator_ckpt="checkpoints/pretrain_discriminator.pt",
            discriminator_ckpt_sha256=PretrainProvenance.digest(
                self.ckpt_path,
            ),
        )
        fields = {
            "group": "modal-label",
            "gate_criterion": "recon_auc",
            "latent_shape": tuple(Fixture.LATENT_SHAPE),
            "condition_auc": {"t1n": 0.72, "t1c": 0.68, "t2w": 0.70, "t2f": 0.66},
            "gate_whitelist": ["t1n", "t1c", "t2w", "t2f"],
            "steps_completed": 40,
            "gate_auc": 0.65,
            "gate_passed": True,
            "discriminator_ckpt": "checkpoints/pretrain_discriminator.pt",
            "provenance": provenance,
        }
        fields.update(overrides)
        return PretrainReport(**fields)

    def write(self, report: PretrainReport) -> Path:
        self.report_path.write_text(
            report.model_dump_json(indent=2), encoding="utf-8",
        )
        return self.report_path


@pytest.fixture
def report_scenario(tmp_path: Path) -> PretrainReportScenario:
    return PretrainReportScenario(tmp_path)


class TestPretrainReportConditionDomain:
    """报告的条件域 = 本域词汇表条件集（#129）：MR-RATE 的条件名
    （``t1w/axial`` 等）此前被 BraTS 四序列 ``Modality`` 字面量挡在
    schema 之外——每次 MR 预训练都在 ``_finalize`` 处以 ValidationError
    收场、``pretrain_report.json`` 落不了盘（无报告 = train 侧拿不到
    warm-start 产物，整条换域线在收尾处断）。"""

    def test_mr_condition_names_roundtrip(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """MR 条件名报告落盘 → 装载无损；多条件线不记单域全局形状
        （形状逐条件派生自词表工件，口径由 provenance 承载）。"""
        report = report_scenario.report(
            latent_shape=None,
            condition_auc={"t1w/axial": 0.72, "flair/axial": 0.61},
            gate_whitelist=["t1w/axial"],
        )
        loaded = PretrainReport.load(report_scenario.write(report))
        assert set(loaded.condition_auc) == {"t1w/axial", "flair/axial"}
        assert loaded.gate_whitelist == ["t1w/axial"]
        assert loaded.latent_shape is None

    def test_brats_report_keeps_global_shape(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """单域（BraTS）口径不动：四序列条件名 + 全局 latent 形状。"""
        loaded = PretrainReport.load(
            report_scenario.write(report_scenario.report()),
        )
        assert loaded.latent_shape == tuple(Fixture.LATENT_SHAPE)
        assert loaded.gate_whitelist == list(MODALITIES)


class TestPretrainReportGuard:
    def test_load_rejects_missing_report(self, tmp_path: Path) -> None:
        """缺报告 = 未预训练：拒绝装载（守卫哲学）。"""
        with pytest.raises(FileNotFoundError, match="预训练报告"):
            PretrainReport.load(tmp_path / "pretrain_report.json")

    def test_load_rejects_wrong_kind(self, report_scenario: PretrainReportScenario) -> None:
        """kind 不符的 JSON 冒充预训练报告：装载期拒绝。"""
        data = json.loads(report_scenario.report().model_dump_json())
        data["kind"] = "real_pool"
        path = report_scenario.report_dir / "impostor.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError):
            PretrainReport.load(path)

    def test_load_rejects_legacy_pooled_format(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """ADR-0008 之前的池化口径报告（final_heldout_auc 单标量）在新
        schema 下显式拒绝：可读报错点名格式变更与 ADR-0008，而非裸
        ValidationError——BraTS 线旧报告同此路径。"""
        data = json.loads(report_scenario.report().model_dump_json())
        del data["condition_auc"]
        del data["gate_whitelist"]
        data["final_heldout_auc"] = 0.72
        path = report_scenario.report_dir / "legacy.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="ADR-0008") as exc_info:
            PretrainReport.load(path)
        assert "final_heldout_auc" in str(exc_info.value)

    def _guard_aligned_report(self, report_scenario: PretrainReportScenario):
        """数据口径指纹全部对齐（real pool / held-out manifest 落盘 + 真
        digest）的报告——守卫测试中唯一的拒绝来源即被测对照本身。"""
        Path(report_scenario.config.reward.real_pool_manifest).write_text("[]")
        Path(report_scenario.config.reward.heldout_real_manifest).write_text("[]")
        report = report_scenario.report()
        return report.model_copy(update={
            "provenance": report.provenance.model_copy(update={
                "real_pool_manifest_sha256": PretrainProvenance.digest(
                    Path(report_scenario.config.reward.real_pool_manifest),
                ),
                "heldout_manifest_sha256": PretrainProvenance.digest(
                    Path(report_scenario.config.reward.heldout_real_manifest),
                ),
            }),
        })

    def test_group_mismatch_rejected(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """报告组别 ≠ 消费 config 组别（组1/组2 独立 run 的跨组消费）：
        装载期显式拒绝（#113）——per-condition AUC 与条件白名单是在预
        训练组别自己的 fake 分布上测量的，跨组上岗是口径错位而非可配置
        语义（无逃生门）。判别性构造：数据口径指纹全部对齐，拒绝只能
        来自组别对照；报错含两侧组别值、「同组别口径」与 stage-2 报告
        绑定配置面（#116）的可读指引。"""
        report = self._guard_aligned_report(report_scenario)
        report_scenario.config.experiment.group = "cross-modal"
        with pytest.raises(ValueError) as exc_info:
            report.assert_data_provenance(report_scenario.config)
        message = str(exc_info.value)
        assert "组别" in message
        assert "modal-label" in message  # 报告侧组别值
        assert "cross-modal" in message  # config 侧组别值
        assert "同组别" in message  # 预训练与上岗须同组别口径的指引
        assert "stage2_pretrain_report_json" in message  # 组3 stage-2 的绑定配置面
        assert "stage-1" in message  # 不继承 stage-1 报告的指引

    def test_group_match_passes(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """同组别消费：守卫全链（组别 → latent → 三工件指纹）照常通过，
        行为与守卫落地前完全一致（#113 AC：同组消费逐字节不变）。"""
        report = self._guard_aligned_report(report_scenario)
        report.assert_data_provenance(report_scenario.config)

    def test_report_roundtrip(self, report_scenario: PretrainReportScenario) -> None:
        """报告落盘 → 装载无损（含 per-condition 口径与条件白名单）。"""
        report = report_scenario.report()
        path = report_scenario.write(report)
        loaded = PretrainReport.load(path)
        assert loaded.model_dump() == report.model_dump()
        assert loaded.kind == "pretrain_report"
        assert loaded.gate_passed is True
        assert loaded.condition_auc == report.condition_auc
        assert loaded.gate_whitelist == report.gate_whitelist
        assert loaded.provenance.channel_stats_sha256 == report.provenance.channel_stats_sha256

    def test_load_discriminator_restores_saved_weights(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """工件可重载进判别器：装载后权重与落盘 checkpoint 逐位一致。"""
        report_scenario.write(report_scenario.report())
        report = PretrainReport.load(report_scenario.report_path)
        scorer = report.load_discriminator(report_scenario.config)
        saved = report_scenario.loadable_state()
        restored = NetworkAssembler.loadable_state_dict(scorer.discriminator)
        assert restored.keys() == saved.keys()
        assert all(torch.equal(restored[key], saved[key]) for key in saved)

    def test_load_discriminator_rejects_network_config_mismatch(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """判别器形态指纹不符（预训练与当前 config 的网络配置不同）：
        装载期拒绝，不给静默错位的可乘之机。"""
        report_scenario.write(report_scenario.report())
        other = report_scenario.tmp_path / "other_discriminator_config.json"
        other.write_text('{"spatial_dims": 3, "channels": 8}', encoding="utf-8")
        report_scenario.config.artifacts.discriminator_config_json = other
        report = PretrainReport.load(report_scenario.report_path)
        with pytest.raises(ValueError, match="指纹"):
            report.load_discriminator(report_scenario.config)

    def test_load_discriminator_rejects_checkpoint_substitution(
        self, report_scenario: PretrainReportScenario,
    ) -> None:
        """盘上 checkpoint 与报告实测的那份不符（同形态换权重）：装载期
        拒绝——报告的白名单与 per-condition 实测值只对预训练落盘的这份
        权重负责；启动期重算废止后（ADR-0008 决策 5），「测量对象 =
        装载对象」由 checkpoint 内容指纹对照把守。"""
        report_scenario.write(report_scenario.report())
        state = report_scenario.loadable_state()
        key = next(k for k, v in state.items() if v.is_floating_point())
        state[key] = state[key] + 0.5
        torch.save(state, report_scenario.ckpt_path)
        report = PretrainReport.load(report_scenario.report_path)
        with pytest.raises(ValueError, match="指纹"):
            report.load_discriminator(report_scenario.config)


DISCRIMINATOR_SEED = 7
"""fixture 判别器初始化 seed（与 test_reward_fixture 同款：网络工件确定性）。"""


@pytest.fixture(scope="module")
def pretrain_inputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """module 级共享输入：prepare 三工件 + fixture 网络工件。

    prepare 幂等且与门槛/批量配置无关，构造一次供各用例复用；各用例只
    改 reward 字段与 run 目录（预训练 run 目录本就每次运行独立）。"""
    base = tmp_path_factory.mktemp("pretrain_inputs")
    fixture_dir = base / "fixtures"
    config = Fixture().config(fixture_dir)
    FixturePrepareScenario(CliSession(), config, base).run(base / "config.json")
    # fixture 网络「固定 seed」机制：随机初始化随 seed 定死（轨迹可复现）
    torch.manual_seed(DISCRIMINATOR_SEED)
    Fixture().write_artifacts(fixture_dir)
    return base


class PretrainScenario:
    """预训练端到端场景：共享 prepare 工件 + 每用例独立 config 与 run 目录。"""

    def __init__(self, inputs: Path, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.tmp_path = tmp_path
        self.config_dict = json.loads(
            (inputs / "config.json").read_text(encoding="utf-8"),
        )
        self.config_path = tmp_path / "config.json"
        self.run_dir = tmp_path / "pretrain_run"
        # 报告路径 = run 目录内的契约名（run 目录缺省随它派生）；buffer
        # 缩小到 16（base 分区 8、每条件配额 2 = 回放半区 2——ADR-0008
        # 决策 4 装配守卫的下限）：量产启动成本是每用例固定开销，容量
        # 与判定逻辑无关（生产 64）
        self.config_dict["reward"].update({
            "pretrain_report_json": str(self.run_dir / "pretrain_report.json"),
            "replay_buffer_capacity": 16,
        })

    def write_config(
        self, group: str = "modal-label", reward: dict | None = None,
    ) -> None:
        self.config_dict["experiment"]["group"] = group
        self.config_dict["reward"].update(reward or {})
        self.config_path.write_text(
            json.dumps(self.config_dict), encoding="utf-8",
        )

    def config(self) -> CynosureConfig:
        return ConfigLoader.load(self.config_path)

    def run_dir_path(self) -> Path:
        """预训练 run 目录 = config 声明的报告路径所在目录。"""
        return Path(self.config().reward.pretrain_report_json).parent

    def pretrain(self, *args: str) -> CliResult:
        return self.cli.run(
            "pretrain", "--config", str(self.config_path), *args,
        )

    def events(self) -> list[dict]:
        lines = (self.run_dir_path() / "metrics.jsonl").read_text(
            encoding="utf-8",
        ).splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def report(self) -> PretrainReport:
        return PretrainReport.load(self.run_dir_path() / "pretrain_report.json")


@pytest.fixture
def scenario(
    pretrain_inputs: Path, cli: CliSession, tmp_path: Path,
) -> PretrainScenario:
    return PretrainScenario(pretrain_inputs, cli, tmp_path)


class TestPretrainEndToEnd:
    def test_fixture_pipeline_produces_guardable_artifacts(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：fixture 合成流上 pretrain 端到端跑通并产出工件（checkpoint
        + 报告），报告含 per-condition AUC 与条件白名单、完整口径指纹。"""
        # 门槛 0.01 恒达标：全部条件首测即过线、换批复测确认 → 4 个轮转
        # 确认步零更新终止（判定分支的专属用例；「密集步进」路径见
        # dense-steps 用例）。fixture held-out 每条件 4 卷 < 支撑度界 20
        # → 判定走 bootstrap CI 下界口径（CI 下界 ≥ 0.01 恒真）
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        run_dir = scenario.run_dir_path()
        assert (run_dir / "config.json").is_file()
        assert (run_dir / "checkpoints" / "pretrain_discriminator.pt").is_file()
        report = scenario.report()
        assert report.kind == "pretrain_report"
        assert report.group == "modal-label"
        assert report.latent_shape == tuple(Fixture.LATENT_SHAPE)
        assert report.gate_auc == pytest.approx(0.01)
        assert report.gate_passed is True
        assert report.steps_completed == 0
        assert report.gate_whitelist == list(MODALITIES)
        assert set(report.condition_auc) == set(MODALITIES)
        assert all(0.0 <= auc <= 1.0 for auc in report.condition_auc.values())
        # 确认步无更新 → 无事件（「事件数 == 完成步数」不变量的零步退化）
        assert scenario.events() == []
        provenance = report.provenance
        assert Path(provenance.real_pool_manifest) == Path(
            scenario.config().reward.real_pool_manifest
        )
        assert provenance.real_pool_manifest_sha256 == PretrainProvenance.digest(
            Path(provenance.real_pool_manifest)
        )
        assert provenance.channel_stats_sha256 == PretrainProvenance.digest(
            Path(provenance.channel_stats)
        )

    def test_artifacts_reload_into_discriminator(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：工件可重载进判别器（报告 → load_discriminator 与落盘
        checkpoint 逐位一致）。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        assert scenario.pretrain().code == 0
        report = scenario.report()
        scorer = report.load_discriminator(scenario.config())
        saved = torch.load(
            scenario.run_dir_path() / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        restored = NetworkAssembler.loadable_state_dict(scorer.discriminator)
        assert all(torch.equal(restored[key], saved[key]) for key in saved)

    def test_spectral_norm_warm_start_is_bitwise_faithful(
        self, scenario: PretrainScenario,
    ) -> None:
        """SN 启用时的 warm-start 保真（ADR-0007 上岗语义）：落盘 checkpoint
        携带**参数化状态**（``*.parametrizations.weight.*`` = original 权重
        + 幂迭代 buffer ``_u``/``_v``）；``load_discriminator`` 按形态先叠
        谱归一化再严格装载——重建的判别器状态与落盘时**逐位一致**，且
        结果与 ambient RNG 无关（两个不同 seed 下装配结果全等）。

        对照（固化有效权重的形态）在装载时要**再归一化一次**（随机 u/v
        起步 + 15 次幂迭代）→ 上岗的判别器不是预训练认证的那一份（实测
        fixture 端到端 logits 偏离 4.5e-6、held-out AUC 0.5386 vs 0.5389）。"""
        scenario.write_config(reward={
            "spectral_norm_enabled": True,
            "pretrain_gate_auc": 0.99,  # 不可达：走满步数上限（真训练态）
            "pretrain_max_steps": 3,
        })
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        report = scenario.report()
        assert report.steps_completed == 3
        saved = torch.load(
            scenario.run_dir_path() / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        # 落盘面 = 参数化状态（原始权重 + u/v 同盘），不是物化有效权重
        assert any(".parametrizations." in key for key in saved)
        config = scenario.config()
        torch.manual_seed(11)
        first = report.load_discriminator(config, device=torch.device("cpu"))
        torch.manual_seed(20260910)  # ambient seed 不同：装载结果不得依赖它
        second = report.load_discriminator(config, device=torch.device("cpu"))
        for scorer in (first, second):
            restored = scorer.discriminator.state_dict()
            assert restored.keys() == saved.keys()
            assert all(torch.equal(restored[key], saved[key]) for key in saved)
        # 判别函数层：两次装配的前向一致（形态还原的语义本体）。容差而非
        # 逐位：同权重、不同实例的前向可差 1 ulp（2^-23 ≈ 1.2e-7，浮点
        # 执行路径的分配/分块选择，实测偶发）——判别力由上面的 state
        # 逐位对比承担；1e-6 在噪声底之上、修复前形态的偏离之下。
        torch.manual_seed(3)
        probe = torch.randn(2, *config.latent_shape)
        first.discriminator.eval()
        second.discriminator.eval()
        assert torch.allclose(
            first.patch_logits(probe), second.patch_logits(probe),
            rtol=0.0, atol=1e-6,
        )

    def test_dense_steps_terminate_at_gate_or_cap(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：终止语义 = 全部条件最近一次 per-condition recon-AUC 过线或
        步数上限——每步落盘预训练事件（判别字段 + 条件 + loss/AUC/buffer
        占用 + 重构成本读数），事件流 AUC 随密集步进按条件可归因。

        阈值 0.99 不可达：走满步数上限分支（白名单空仍落盘报告 +
        checkpoint 供诊断——拒跑由 train gate 把守，诊断产物不丢；
        「全部条件过线」分支由 0.01 恒达标用例覆盖）。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.99,
            "pretrain_max_steps": 12,
            "disc_lr": 2e-4,
        })
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        report = scenario.report()
        events = scenario.events()
        # 判定逻辑不变量：确认过线步不发事件 → 事件数 == 完成步数；步号连续
        assert len(events) == report.steps_completed == 12
        assert [event["step"] for event in events] == list(range(12))
        assert all(event["event"] == "pretrain" for event in events)
        assert all("loss_discriminator" in event for event in events)
        # per-condition 轮转：每步条件 = 轮转条件集的 step % n（目标模态
        # 均匀轮转，ADR-0008 决策 3）——事件流按条件可归因
        assert [event["modality"] for event in events] == [
            MODALITIES[step % len(MODALITIES)] for step in range(12)
        ]
        # 曲线的操作者可见面：终点报出指标流路径与事件数（离线查看收敛
        # 曲线、校准门槛阈值的数据源）
        assert f"{len(events)} 条 pretrain 事件" in result.stdout
        # 白名单空仍落盘报告 + checkpoint（诊断产物不丢）
        assert report.gate_whitelist == []
        assert report.gate_passed is False
        assert (scenario.run_dir_path() / "checkpoints" /
                "pretrain_discriminator.pt").is_file()
        # 耗尽路径：未过线条件逐个补测（与落盘 checkpoint 同快照）→ 报告
        # dict 覆盖全部轮转条件，值 < 门槛
        assert set(report.condition_auc) == set(MODALITIES)
        assert all(auc < report.gate_auc for auc in report.condition_auc.values())
        # 密集步进拉动 AUC（条件化口径）：判别力建立按条件分化——fixture
        # 判别器对不同序列的判别力基线不同，池化口径的「整体上升」断言
        # 被取代为「至少一个条件在其自身事件子序列上呈上升趋势」（单步
        # patch 级 AUC 在 fixture 小批量下噪声大，阈值 0.05 隔离噪声）
        rises = 0
        for modality in MODALITIES:
            series = [
                event["heldout_auc"] for event in events
                if event["modality"] == modality
            ]
            if series[-1] > series[0] + 0.05:
                rises += 1
        assert rises >= 1

    def test_cross_modal_group_uses_same_path(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：组1/组2 走同一条路径，仅 config 不同（组2 fake 走
        ControlNet 条件分布）；轮转条件集 = 有序对清单的目标端去重
        （12 全组合对 → 四目标端全部在集）。"""
        scenario.write_config(
            group="cross-modal", reward={"pretrain_gate_auc": 0.01},
        )
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        report = scenario.report()
        assert report.group == "cross-modal"
        assert sorted(report.gate_whitelist) == sorted(MODALITIES)


class TestPretrainCliGuards:
    def test_rejects_existing_run_directory(
        self, scenario: PretrainScenario,
    ) -> None:
        """run 目录已存在：拒绝（不静默覆盖）。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        scenario.run_dir_path().mkdir(parents=True)
        result = scenario.pretrain()
        assert result.code == 2
        assert "已存在" in result.stderr

    def test_rejects_torchrun_launch(
        self, scenario: PretrainScenario, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """预训练单进程执行（World-1 退化路径）：torchrun（RANK env）
        下启动显式拒绝——多 rank 各自预训练会分叉判别器。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "2")
        result = scenario.pretrain()
        assert result.code == 2
        assert "单进程" in result.stderr

    def test_explicit_run_dir_override(
        self, scenario: PretrainScenario, tmp_path: Path,
    ) -> None:
        """--run-dir 显式覆盖默认（config 报告路径所在目录）：产物路径
        以 config 声明为准——覆盖目录与声明分叉时拒绝（train 按 config
        声明装载，分叉即 missing-report 或静默装旧）。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        override = tmp_path / "override_run"
        result = scenario.pretrain("--run-dir", str(override))
        assert result.code == 2
        assert "声明" in result.stderr
        assert not override.exists()

    def test_report_path_divergence_rejected(
        self, scenario: PretrainScenario,
    ) -> None:
        """config 声明 basename ≠ 报告契约名：pretrain 固定写 run 目录
        内的 ``pretrain_report.json``，train 按声明路径装载——分叉即
        missing-report 或静默装旧报告，入口显式拒绝。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.01,
            "pretrain_report_json": str(
                scenario.run_dir / "warm_start_v2.json"
            ),
        })
        result = scenario.pretrain()
        assert result.code == 2
        assert "声明" in result.stderr
        assert not scenario.run_dir.exists()

    def test_run_dir_matching_declared_path_passes(
        self, scenario: PretrainScenario, tmp_path: Path,
    ) -> None:
        """--run-dir 与 config 声明路径一致（目录与契约名都对上）：
        显式覆盖放行——一致性不变式只拒绝分叉，不拒绝显式声明。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        result = scenario.pretrain("--run-dir", str(scenario.run_dir))
        assert result.code == 0, result.stderr
        assert scenario.report().gate_passed is True

    def test_same_location_different_spelling_passes(
        self, scenario: PretrainScenario, tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """同一位置的两种写法（config 声明相对路径、--run-dir 给绝对
        路径）是同一份工件：一致性不变式比对**归一化后的**路径，不比对
        字面拼写——字面比较会把「相对 vs 绝对」「``.`` 分量」「符号链接
        祖先」这些同址写法误判成分叉，拒绝合法调用。"""
        monkeypatch.chdir(tmp_path)
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.01,
            "pretrain_report_json": "pretrain_run/pretrain_report.json",
        })
        result = scenario.pretrain("--run-dir", str(scenario.run_dir))
        assert result.code == 0, result.stderr
        assert scenario.report().gate_passed is True


class TestPretrainDriverAssembly:
    def test_reuses_online_update_with_explicit_weight_decay(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：driver 复用在线期同款单步更新原语（OnlineUpdate 实例，
        无第二套判别器训练逻辑）；weight_decay 显式配置且与 policy 同值。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "assembly_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        assert isinstance(driver.rewards.update, OnlineUpdate)
        decay = driver.rewards.update.optimizer.param_groups[0]["weight_decay"]
        assert decay == pytest.approx(config.reward.disc_weight_decay)
        assert decay == pytest.approx(config.policy.policy_weight_decay)

    def test_consumes_same_overfit_monitor_via_shared_assembly(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC（ADR-0009-γ，issue #106 验收 1）：预训练 driver 每步消费与
        在线同一监控组件、同一 config knobs——分叉监控器经共享装配缝
        （``TrainingRuntime.assemble_rewards``）挂进 ``RewardCoordinator``，
        两阶段同一组件类、knobs 同源于 ``config.reward.overfit_*``；
        与在线装配逐位一致（同一调用、同一入参，产物 knobs 无分歧）。
        """
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "overfit_assembly_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        assert isinstance(driver.rewards.overfit, OverfitMonitor)
        assert driver.rewards.overfit._threshold == pytest.approx(
            config.reward.overfit_alert_divergence
        )
        assert driver.rewards.overfit._span == config.reward.overfit_ema_span
        # 在线装配同缝重放：产物监控器 knobs 与预训练侧逐位一致
        dist = DistributedContext.bootstrap()
        amp = AmpContext(
            device=torch.device("cpu"),
            dtype=AMP_DTYPES[config.policy.amp_dtype],
        )
        generators = TrainingRngStreams(
            dist.derive_seed(config.schedule.seed),
            shared_seed=config.schedule.seed,  # 生产装配位同款（数据侧
            # 逐 rank 派生、recon 用 shared）——同缝重放保持镜像逐字
        ).named()
        policy = GroupPolicy.build(config, generators["rollout"], amp.device)
        sampler = TrainingRuntime.assemble_sampler(
            config, policy.field, device=amp.device,
        )
        online = TrainingRuntime.assemble_rewards(
            config, amp, generators, dist,
            sampler=sampler, conditions=policy.conditions,
        )
        assert isinstance(online.overfit, OverfitMonitor)
        assert online.overfit._threshold == driver.rewards.overfit._threshold
        assert online.overfit._span == driver.rewards.overfit._span

    def test_cross_modal_conditions_from_controlnet_path(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：组1/组2 走同一条路径、仅 config 不同——组2 的条件分布装配
        为 ControlNet 交叉模态采样器（GroupPolicy 按组分派）；配对批装
        配原语对组2 同样组装（ADR-0012 决策 7：条件构造与重构前向按组
        自然分派——组2 判别任务的语义裁决属 stage-2 专项票）。"""
        scenario.write_config(
            group="cross-modal", reward={"pretrain_gate_auc": 0.01},
        )
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "assembly_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        assert isinstance(driver.policy.conditions, CrossModalConditionSampler)
        assert driver.rewards.assembler is not None

    @pytest.mark.slow  # 满步数轮转 × 每步真实装配原语重构（fixture 网络 ODE 续跑）
    def test_rotation_steps_round_robin(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：per-condition 均匀轮转——每步条件 = 轮转条件集的
        ``step % n``，update_step 穿同一步条件（确定性轮转不耗 RNG；
        ADR-0008 决策 3 的调度形态）——替身按步记录条件，步数上限内
        每步一步更新、条件按轮转序循环。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.99,  # 不可达：跑满上限，逐步观测
            "pretrain_max_steps": 6,
        })
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "assembly_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        recording = RecordingUpdate(
            driver.rewards.discriminator,
            buffer_capacity=config.reward.replay_buffer_capacity,
        )
        driver.rewards.update = recording
        report = driver.run()
        assert report.steps_completed == 6
        assert recording.modalities == [
            MODALITIES[step % len(MODALITIES)] for step in range(6)
        ]

    def test_rejects_real_pool_below_batch_capacity(
        self, scenario: PretrainScenario,
    ) -> None:
        """ADR-0008-03 装配守卫：逐 (全池, 模态) real 容量 < K → fail-fast
        可读报错（driver 经 assemble_rewards 与 train 同一条装配缝——
        守卫先于任何 rollout/更新执行）。"""
        small_pool = scenario.tmp_path / "starved_pool.json"
        small_pool.write_text(json.dumps({
            "kind": "real_pool",
            "encoder": "starved-fixture",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"train": 12},
            "entries": [
                {
                    "case_id": f"case-{modality}-{index}",
                    "modality": modality,
                    "latent": f"latents/{modality}-{index}.pt",
                    "spacing": [100.0, 100.0, 100.0],
                }
                for modality in ("t1n", "t1c", "t2w", "t2f")
                for index in range(3)  # 每模态 3 条 < K=4
            ],
        }), encoding="utf-8")
        scenario.write_config(reward={"real_pool_manifest": str(small_pool)})
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "capacity_run")
        with pytest.raises(ValueError, match="容量不足") as exc_info:
            PretrainDriver(config, run, device=torch.device("cpu"))
        message = str(exc_info.value)
        assert "disc_batch_size_k" in message  # 可读：点名条件、可用量与 knob

    def test_rejects_rotation_target_without_heldout(
        self, scenario: PretrainScenario,
    ) -> None:
        """ADR-0008-04 装配守卫：轮转条件集某条件的 held-out 无条目 →
        fail-fast（per-condition AUC 归因无米下锅；首步 rollout 之前
        拒绝，报错点名缺条目的条件）。"""
        starved_heldout = scenario.tmp_path / "starved_heldout.json"
        starved_heldout.write_text(json.dumps({
            "kind": "heldout_real",
            "encoder": "starved-fixture",
            "latent_shape": [4, 16, 16, 8],
            "split_seed": 0,
            "split_sizes": {"val": 9},
            "entries": [
                {
                    "case_id": f"case-{modality}-{index}",
                    "modality": modality,
                    "latent": f"latents/{modality}-{index}.pt",
                    "spacing": [100.0, 100.0, 100.0],
                }
                for modality in ("t1n", "t1c", "t2w")  # t2f 缺条目
                for index in range(3)
            ],
        }), encoding="utf-8")
        scenario.write_config(reward={
            "heldout_real_manifest": str(starved_heldout),
        })
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "starved_run")
        with pytest.raises(ValueError, match="t2f") as exc_info:
            PretrainDriver(config, run, device=torch.device("cpu"))
        assert "held-out" in str(exc_info.value)


class TestPretrainReconstructionFakeSupply:
    """#171 AC1：「预训练全程 fake 均来自装配原语（冻结基座重构），
    量产 rollout 不再被预训练路径调用」——执行路径面与事件读数面两侧
    钉住。

    量产退役是**范围收窄**（RolloutPhase 从 driver 装配面消失、成本
    读数改口径），不是「换个名字继续跑」：本类断言 driver 上不存在
    任何量产入口，且事件流自带「重构前向次数 < 量产步数」的成本证据。
    """

    def test_driver_has_no_rollout_phase_seam(
        self, scenario: PretrainScenario,
    ) -> None:
        """装配面：driver 不再持有 RolloutPhase（``rollout`` 公开面、
        ``_rollout`` 私有位与旧量产入口 `_measurement_batch` 一律不在）
        ——量产路径在预训练相结构性不可达，而非「约定不调用」。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "supply_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        for attribute in ("rollout", "_rollout", "_measurement_batch"):
            assert not hasattr(driver, attribute), attribute
        # 配送面在场：测量与更新同源于装配原语（两阶段构造同构的前提）
        assert driver.rewards.assembler is not None
        assert callable(driver.rewards.assembler.measure_condition)

    def test_measurement_batch_is_full_heldout_reconstruction(
        self, scenario: PretrainScenario,
    ) -> None:
        """测量面：每步测量批 = 该条件**全量 held-out 卷**的冻结基座
        重构（非批次量产）——卷数 = 该条件 held-out 条目数、real 与
        fake 逐样本同形同源；``pretrain_fake_batch`` 不再是测量批的
        量纲（配错也不改变测量批规模）。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.01,
            # 与 held-out 每条件 2 卷刻意配错：量产口径的量纲不再消费
            "pretrain_fake_batch": 7,
        })
        config = scenario.config()
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        assert heldout.modalities["t1n"] == 2
        run = PretrainRun.init(config, scenario.tmp_path / "measure_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        target = driver.policy.conditions.targets()[0]
        reals = driver.rewards.auc.condition_latents(target)
        batch = driver.rewards.assembler.measure_condition(reals, target)
        assert batch.reals.shape[0] == 2  # 该条件全量卷，非 pretrain_fake_batch
        assert batch.fakes.shape == batch.reals.shape
        assert batch.reals is reals  # 逐样本配对的同一批张量
        assert not torch.equal(batch.fakes, batch.reals)  # 冻结基座重构在场
        # 成本读数与测量批的规模同源（同入参、同定序轮转）：逐卷余量
        # 之和（MONAI 实际日程步数），严格小于量产的「每卷全 ODE」上界
        cursor = TrainingRuntime.assemble_schedules(config).cursor(target)
        forwards = driver.rewards.assembler.measurement_forward_count(
            reals, target,
        )
        assert 0 < forwards < batch.reals.shape[0] * cursor.num_steps

    def test_report_carries_recon_criterion_and_support_volumes(
        self, scenario: PretrainScenario,
    ) -> None:
        """报告面（#171 AC2/AC3）：``gate_criterion="recon_auc"`` 明示判据
        口径（两阶段读数不可横向比较的审计锚），``condition_volumes``
        给出支撑度判定的卷数轴（逐条件 = 该条件 held-out 全量卷数）。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        assert scenario.pretrain().code == 0
        report = scenario.report()
        assert report.gate_criterion == "recon_auc"
        config = scenario.config()
        heldout = LatentManifest.load(
            config.reward.heldout_real_manifest, kind="heldout_real",
        )
        assert report.condition_volumes == {
            modality: heldout.modalities[modality] for modality in MODALITIES
        }
        # 判据口径与支撑度卷数进 JSON 工件（旧监控的解析面无损扩展）
        payload = json.loads(
            (scenario.run_dir_path() / "pretrain_report.json").read_text(
                encoding="utf-8",
            ),
        )
        assert payload["gate_criterion"] == "recon_auc"
        assert payload["condition_volumes"] == report.condition_volumes

    def test_pretrain_events_carry_reconstruction_cost_readout(
        self, scenario: PretrainScenario,
    ) -> None:
        """事件面（#171 AC5）：每步 pretrain 事件带重构成本读数——
        前向次数 = 测量批逐卷余量之和（严格小于「每卷全 ODE」的
        ``num_steps`` 上界）、卷数 = 该条件 held-out 全量卷；量产
        rollout 退出执行路径后再无该量级的固定开销。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.99,  # 不可达：走满步数上限
            "pretrain_max_steps": 4,
        })
        assert scenario.pretrain().code == 0
        events = [
            event for event in scenario.events()
            if event["event"] == "pretrain"
        ]
        assert len(events) == 4
        config = scenario.config()
        cursor = TrainingRuntime.assemble_schedules(config).cursor(MODALITIES[0])
        for event in events:
            assert event["measurement_volumes"] == 2  # 每条件 held-out 2 卷
            assert 0 < event["reconstruction_forwards"] < (
                event["measurement_volumes"] * cursor.num_steps
            )


SCRIPTED_SHAPE = (4, 4, 16, 16, 8)
"""替身批形状共此一源：ScriptedAuc 测量批与 ScriptedAssembler 更新批
同形（driver 的测量批与更新批同经装配 seam，形状分叉只会在运行断言
炸——常量共享让两处不可能漂移）。"""


class ScriptedClusters:
    """卷级聚类替身：携带条件名、固定点估计与卷数的哑观测（ScriptedSupport
    按 ``modality`` 查判定脚本；``pooled_auc`` 原样透传，``volume_count``
    = real 侧行数——报告 ``condition_volumes`` 留痕的报告面）。"""

    def __init__(
        self, modality: str, point_estimate: float, volume_count: int,
    ) -> None:
        self.modality = modality
        self._point_estimate = point_estimate
        self.volume_count = volume_count

    def pooled_auc(self) -> float:
        return self._point_estimate


class ScriptedAuc:
    """HeldOutAuc 替身：按条件脚本返回点估计（记录测量次序供断言）；
    容量查询恒充足（守卫路径由真实 manifest 用例覆盖）。

    ``condition_latents`` 与 ``compute_volume_clusters`` 的配对语义照搬
    真实实现（real 侧由调用方给出、与 fake 同量同形）——条件归因不在
    聚类 seam 的入参里（生产签名无 modality 位），由测量流「先取
    real 侧」的 ``condition_latents`` 调用带出。状态机用例只换测量面的
    **数值来源**，不绕开配对契约。"""

    def __init__(self, values: dict[str, float]) -> None:
        self.values = values
        self.measurements: list[str] = []
        self.paired: list[bool] = []
        self._measuring: str | None = None

    def condition_latents(self, modality: str) -> torch.Tensor:
        self._measuring = modality  # 测量流先取 real 侧：条件归因随之而来
        return torch.zeros(SCRIPTED_SHAPE)

    def compute_volume_clusters(self, latents, fake_latents):
        assert latents.shape == fake_latents.shape  # 同源配对的逐样本对齐
        assert self._measuring is not None  # 生产 seam 无 modality 位
        modality = self._measuring
        self.measurements.append(modality)
        self.paired.append(True)
        return ScriptedClusters(
            modality, self.values[modality], latents.shape[0],
        )

    def condition_volume_count(self, modality) -> int:
        return 4


class ScriptedAssembler:
    """ReconstructionAssembler 替身（旋转状态机用例的装配 seam）：测量批
    real/fake 同量同形、值可辨识（fake = −real，同源配对的可观测面）；
    成本读数恒 1。更新批同经本替身（ADR-0012 后 driver 的测量与更新是
    同一装配原语的两条入口，替换必须同时覆盖）——RecordingUpdate 不做
    真实前向，形状仅走 ``UpdateReport`` 的 batch_size 记账（4 卷 ×
    BraTS latent 形状，与 ScriptedAuc 替身批同约定）。"""

    def assemble(self, modality: str) -> PairBatch:
        return PairBatch(
            reals=torch.zeros(SCRIPTED_SHAPE),
            fakes=torch.zeros(SCRIPTED_SHAPE),
            modality=modality,
        )

    def measure_condition(self, reals: torch.Tensor, modality: str):
        return PairBatch(reals=reals, fakes=-reals, modality=modality)

    def measurement_forward_count(
        self, reals: torch.Tensor, modality: str,
    ) -> int:
        return 1


class ScriptedSupport:
    """SupportRule 替身：每条件一个判定脚本队列（逐次弹出，耗尽后重复
    末值——「首测过、复测掉」的序列 = [True, False]），记录判定次序。"""

    def __init__(self, plan: dict[str, list[bool]]) -> None:
        self.plan = {modality: list(seq) for modality, seq in plan.items()}
        self.verdicts: list[tuple[str, bool]] = []

    def passes(self, point_estimate: float, clusters) -> bool:
        seq = self.plan[clusters.modality]
        verdict = seq.pop(0) if len(seq) > 1 else seq[0]
        self.verdicts.append((clusters.modality, verdict))
        return verdict


class TestPretrainRotationStateMachine:
    """终止状态机替身用例（ADR-0008-04 决策 3 的判定语义）：测量
    （ScriptedAuc）/ 支撑度判定（ScriptedSupport）/ 更新（RecordingUpdate）
    四 seam 换脚本替身（测量 / 配对批装配 / 支撑度 / 更新），确认棘轮、
    部分白名单、复测掉线、耗尽补测的语义逐项驱动。事件落盘走真实路径。"""

    @pytest.fixture
    def scripted(self, scenario: PretrainScenario):
        """替身驱动的 driver 工厂：config 定死不可达门槛，三 seam 注入；
        ``reward_overrides`` 供 γ 用例覆写分叉 knobs（不牵动其余用例）。"""
        def factory(
            values: dict[str, float], plan: dict[str, list[bool]],
            max_steps: int = 8, reward_overrides: dict | None = None,
        ) -> tuple[PretrainDriver, ScriptedAuc, ScriptedSupport, RecordingUpdate]:
            scenario.write_config(reward={
                "pretrain_gate_auc": 0.99,
                "pretrain_max_steps": max_steps,
                **(reward_overrides or {}),
            })
            config = scenario.config()
            run = PretrainRun.init(config, scenario.tmp_path / "state_run")
            driver = PretrainDriver(config, run, device=torch.device("cpu"))
            recording = RecordingUpdate(
                driver.rewards.discriminator,
                buffer_capacity=config.reward.replay_buffer_capacity,
            )
            auc = ScriptedAuc(values)
            support = ScriptedSupport(plan)
            driver.rewards.update = recording
            driver.rewards.auc = auc
            driver.rewards.assembler = ScriptedAssembler()
            driver._support = support
            return driver, auc, support, recording
        return factory

    def test_all_conditions_confirmed_stops_early(
        self, scripted, scenario: PretrainScenario,
    ) -> None:
        """全部条件确认过线即停（决策 3 终止语义）：每条件首测过线 →
        换批复测确认 → 棘轮入白名单；全部入列终止，零更新零事件。"""
        values = {modality: 0.8 for modality in MODALITIES}
        plan = {modality: [True] for modality in MODALITIES}
        driver, auc, support, recording = scripted(values, plan)
        report = driver.run()
        assert report.gate_passed is True
        assert report.steps_completed == 0
        assert report.gate_whitelist == list(MODALITIES)
        assert report.condition_auc == {modality: 0.8 for modality in MODALITIES}
        assert recording.received == []  # 确认步不更新
        assert driver._run.read_events() == []  # 无更新即无事件
        # 测量次序：每条件「首测 + 复测」成对、按轮转序推进
        assert auc.measurements == [
            modality
            for modality in MODALITIES
            for _ in range(2)
        ]
        assert len(support.verdicts) == 8  # 每条件两次判定

    def test_partial_whitelist_on_step_exhaustion(
        self, scripted,
    ) -> None:
        """步数耗尽 → 白名单 = 已确认者：唯一过线条件的确认发生在首步，
        其余条件轮转测量不过线照常更新；已确认条件后续轮转仍测量 +
        更新（棘轮不撤销、报告值保留确认时的两次较小者）；未确认条件
        耗尽后逐个补测（与 checkpoint 同快照）。"""
        values = {
            "t1n": 0.8, "t1c": 0.4, "t2w": 0.4, "t2f": 0.4,
        }
        plan = {
            "t1n": [True], "t1c": [False], "t2w": [False], "t2f": [False],
        }
        driver, auc, support, recording = scripted(values, plan, max_steps=8)
        report = driver.run()
        assert report.gate_passed is False
        assert report.gate_whitelist == ["t1n"]
        assert report.steps_completed == 7  # t1n 确认步无更新，其余 7 步更新
        assert report.condition_auc["t1n"] == pytest.approx(0.8)
        assert report.condition_auc["t1c"] == pytest.approx(0.4)
        assert report.condition_auc["t2w"] == pytest.approx(0.4)
        assert report.condition_auc["t2f"] == pytest.approx(0.4)
        # 测量：step0 测 t1n（首测+复测）；steps 1-7 更新步各一次首测
        # （t1n 已确认的轮转步只测不判）；耗尽补测 t1c/t2w/t2f 各一次
        assert auc.measurements == [
            "t1n", "t1n",                       # step0：首测 + 复测
            "t1c", "t2w", "t2f", "t1n",         # 轮转 steps 1-4
            "t1c", "t2w", "t2f",                # 轮转 steps 5-7
            "t1c", "t2w", "t2f",                # 耗尽补测（t1n 已确认不补）
        ]
        # 已确认条件照常参与轮转更新
        assert recording.modalities == [
            MODALITIES[step % len(MODALITIES)] for step in range(1, 8)
        ]

    def test_confirm_failure_continues_updating(
        self, scripted,
    ) -> None:
        """复测掉线不确认：首测过线换批复测不过 → 该条件不入白名单、
        本步照常更新落事件（事件 AUC = 首测值）——单批贴线越过被
        非确定性拒绝的池化语义按条件化保留。"""
        values = {"t1n": 0.9, "t1c": 0.4, "t2w": 0.4, "t2f": 0.4}
        plan = {
            "t1n": [True, False],  # 首测过、复测掉
            "t1c": [False], "t2w": [False], "t2f": [False],
        }
        driver, auc, support, recording = scripted(values, plan, max_steps=4)
        report = driver.run()
        assert report.gate_passed is False
        assert report.gate_whitelist == []
        assert report.steps_completed == 4
        events = driver._run.read_events()
        assert [event["modality"] for event in events] == [
            "t1n", "t1c", "t2w", "t2f",
        ]
        assert events[0]["heldout_auc"] == pytest.approx(0.9)  # 首测值落事件

    def test_pretrain_phase_alert_events_carry_modality_attribution(
        self, scripted,
    ) -> None:
        """AC（ADR-0009-γ，issue #106 验收 2）：预训练更新步消费与在线
        同一分叉监控（train 侧 0.5 − held-out 0.4 = 分叉 0.1 ≥ 阈值
        0.01，每条件首观测即越线）——告警落 ``overfit_alert`` 事件：
        ``phase="pretrain"``、``iteration`` = 本步步号、modality 归因 =
        本步轮转条件（与 pretrain 事件同轴），排在同 step 的 pretrain
        事件之后（与在线侧「iter 后随告警」同构的归并序）。"""
        values = {modality: 0.4 for modality in MODALITIES}
        plan = {modality: [False] for modality in MODALITIES}
        driver, auc, support, recording = scripted(
            values, plan, max_steps=4,
            reward_overrides={"overfit_alert_divergence": 0.01},
        )
        report = driver.run()
        assert report.steps_completed == 4
        events = driver._run.read_events()
        assert [
            (event["event"], event["modality"]) for event in events
        ] == [
            pair
            for modality in MODALITIES
            for pair in (("pretrain", modality), ("overfit_alert", modality))
        ]
        for step, alert in enumerate(
            event for event in events if event["event"] == "overfit_alert"
        ):
            assert alert["phase"] == "pretrain"  # γ：相判别字段
            assert alert["iteration"] == step  # 预训练相 = 步号轴
            assert alert["modality"] == MODALITIES[step % len(MODALITIES)]
            assert alert["divergence_ema"] == pytest.approx(0.1)
            assert alert["train_pairwise_acc"] == pytest.approx(0.5)
            assert alert["heldout_auc"] == pytest.approx(0.4)
        # pretrain 事件带分叉 EMA 读数之外的既有字段不受 γ 影响
        assert all(
            event["heldout_auc"] == pytest.approx(0.4)
            for event in events if event["event"] == "pretrain"
        )


class TestOverfitAlertEventContract:
    """overfit_alert 事件契约（ADR-0009-β，issue #105；γ 按相分轨，#106）：
    判别字段区分于既有三型、要素齐备、非有限浮点构造期拒绝（「可扩不
    可改名」与「全流拒绝」两口径的事件面）；回退记账按相分轨——RL 相
    按 iteration 轴随所属 iteration 删除（回退重执行重发），预训练相
    （``phase="pretrain"``）登记 EXEMPT 全量保留（预训练执行史不参与
    回退重写），口径表本体锁在 TestEventRewindAccounting。"""

    def test_event_type_discriminant_and_roundtrip(self, tmp_path: Path) -> None:
        """混存同一 metrics.jsonl 读取无损；要素齐备（modality、分叉值、
        train acc、held-out AUC、rank + iteration/stage 记账轴 + γ 的相
        判别字段）。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(iter_event(0))
        artifacts.append_event(alert_event(3))
        artifacts.append_event(pretrain_event(0))
        events = artifacts.read_events()
        assert [event["event"] for event in events] == [
            "iter", "overfit_alert", "pretrain",
        ]
        alert = events[1]
        assert alert["iteration"] == 3
        assert alert["stage"] == 1
        assert alert["rank"] == 0
        assert alert["modality"] == "t1n"
        assert alert["divergence_ema"] == pytest.approx(0.3)
        assert alert["train_pairwise_acc"] == pytest.approx(0.8)
        assert alert["heldout_auc"] == pytest.approx(0.5)
        assert alert["phase"] == "rl"  # 默认 RL 相：既有构造点零改动

    def test_phase_field_pretrain_roundtrip(self, tmp_path: Path) -> None:
        """预训练相告警的相判别字段混存读取无损；其 ``iteration`` 记
        预训练步号（与 pretrain 事件的 ``step`` 同轴）。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(pretrain_event(0))
        artifacts.append_event(alert_event(0, phase="pretrain"))
        events = artifacts.read_events()
        assert events[1]["phase"] == "pretrain"
        assert events[1]["iteration"] == 0  # 预训练相 = 步号轴

    def test_non_finite_fields_rejected(self) -> None:
        """非有限浮点在事件构造期即拒绝（判别器数值发散不产毒事件——
        指标 JSONL 的 NaN/Inf 非标准 token，严格消费方拒读）。"""
        for field in ("divergence_ema", "train_pairwise_acc", "heldout_auc"):
            for bad in (float("nan"), float("inf")):
                fields = {
                    "iteration": 0,
                    "modality": "t1n",
                    "divergence_ema": 0.3,
                    "train_pairwise_acc": 0.8,
                    "heldout_auc": 0.5,
                }
                fields[field] = bad
                with pytest.raises(ValidationError):
                    OverfitAlertEvent(**fields)

    def test_rewind_keeps_and_drops_alerts_by_iteration(self, tmp_path: Path) -> None:
        """RL 相告警按 iteration 轴参与回退记账：号 < 恢复点的告警保留
        （已进 checkpoint 覆盖面的执行史）、号 ≥ 恢复点的半截告警删除
        （重执行重发）；其他 stage 的告警一概不动。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(alert_event(0))
        artifacts.append_event(iter_event(0))
        artifacts.append_event(alert_event(1))
        artifacts.append_event(alert_event(2))
        artifacts.append_event(iter_event(2))
        artifacts.append_event(alert_event(2, stage=2))  # 组3 stage-2 历史
        removed = artifacts.rewind_events(2, stage=1)
        assert removed == 2  # stage-1 的 iter@2 与 alert@2（同轴半截执行史）
        survivors = artifacts.read_events()
        assert [
            (event["event"], event["iteration"], event["stage"])
            for event in survivors
        ] == [
            ("overfit_alert", 0, 1),
            ("iter", 0, 1),
            ("overfit_alert", 1, 1),
            ("overfit_alert", 2, 2),  # 其他 stage 的历史不动
        ]

    def test_rewind_preserves_pretrain_phase_alerts(self, tmp_path: Path) -> None:
        """预训练相告警 EXEMPT 全量保留（ADR-0009-γ，issue #106 验收 3）：
        预训练执行史（pretrain 事件 + 预训练相告警）不参与回退重写——
        即便步号落在恢复点的删除边界内（号 ≥ 恢复点的「半截」预训练
        告警没有重执行可重发），预训练收敛曲线的报警读数删除即永久
        丢失。RL 相同流告警的记账口径不受影响（照旧按 iteration 轴）。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(pretrain_event(0))
        artifacts.append_event(alert_event(0, phase="pretrain"))
        artifacts.append_event(iter_event(0))
        artifacts.append_event(pretrain_event(1))
        artifacts.append_event(alert_event(1, phase="pretrain"))
        artifacts.append_event(alert_event(1))  # RL 相：恢复点内，保留
        artifacts.append_event(iter_event(9))
        artifacts.append_event(alert_event(9))  # RL 相：恢复点外，删除
        removed = artifacts.rewind_events(5, stage=1)
        assert removed == 2  # 只删 stage-1 的 iter@9 与 RL 相 alert@9
        survivors = artifacts.read_events()
        assert [
            (
                event["event"],
                event.get("phase"),
                event["step"] if event["event"] == "pretrain"
                else event["iteration"],
            )
            for event in survivors
        ] == [
            ("pretrain", None, 0),
            ("overfit_alert", "pretrain", 0),
            ("iter", None, 0),
            ("pretrain", None, 1),
            ("overfit_alert", "pretrain", 1),
            ("overfit_alert", "rl", 1),
        ]

    def test_rewind_to_origin_keeps_pretrain_phase_alerts(
        self, tmp_path: Path,
    ) -> None:
        """恢复点 0 的误删边界（ADR-0009-γ）：预训练相告警的 ``iteration``
        = 步号，按 RL 相口径记账时号 0 恰落在删除边界外（``0 < 0`` 为
        假）——相特判是唯一挡得住这次误删的机制；RL 相 alert@0 仍删。"""
        artifacts = fresh_run_artifacts(tmp_path)
        artifacts.append_event(alert_event(0, phase="pretrain"))
        artifacts.append_event(alert_event(0))
        artifacts.append_event(iter_event(0))
        assert artifacts.rewind_events(0, stage=1) == 2  # RL 相 alert@0 + iter@0
        assert [
            (event["event"], event.get("phase"))
            for event in artifacts.read_events()
        ] == [("overfit_alert", "pretrain")]
