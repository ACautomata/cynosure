"""pretrain 子命令端到端与产物契约测试（ticket #58 验收标准聚合）。

fixture 合成流：小 pool + 小 fake 集 → 密集步进 → 产物落盘 → 重载 →
守卫生效；AUC 随步数上升可观测（先例量级）。终止条件（门槛/步数上限）
双分支、组1/组2 同路径、kind 守卫、预训练事件的回退记账口径由专属
用例覆盖；真实收敛动力学留给 DCU 实跑（spec「Testing Decisions」：
fixture 只验判定逻辑与数据流）。

事件类型 × 回退记账口径（ticket #59）分两层锁：本文件锁**口径表**本身
（三型事件的登记与各自的保留边界、未登记类型的不删语义）；resume seam 上
的真回退（真 `--resume` 不误删预训练事件）见 test_resume_roundtrip。
"""

import copy
import json
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkAssembler
from cynosure.pretrain import (
    PretrainDriver,
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.reward.artifacts import ChannelStats
from cynosure.reward.update import OnlineUpdate
from cynosure.train import (
    REWIND_ACCOUNTING,
    CrossModalConditionSampler,
    IterEvent,
    MilestoneEvent,
    PretrainEvent,
    RewindAccounting,
    RunArtifacts,
)
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    MINIMAL_CONFIG_DICT,
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


def event_type_vocabulary() -> set[str]:
    """指标流事件类型的判别值词汇表（由事件模型实例的 ``event`` 默认值取
    真值——判别字段的 Literal 是那一处的单一来源，测试不另抄字面量）。"""
    return {
        event.event
        for event in (
            iter_event(0), milestone_event(0), pretrain_event(0),
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
        )
        fields = {
            "group": "modal-label",
            "latent_shape": tuple(Fixture.LATENT_SHAPE),
            "final_heldout_auc": 0.72,
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

    def test_report_roundtrip(self, report_scenario: PretrainReportScenario) -> None:
        """报告落盘 → 装载无损（含口径指纹全字段）。"""
        report = report_scenario.report()
        path = report_scenario.write(report)
        loaded = PretrainReport.load(path)
        assert loaded.model_dump() == report.model_dump()
        assert loaded.kind == "pretrain_report"
        assert loaded.gate_passed is True
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
        # 缩小到 8（base 分区 4 ≥ 回放半区 2）：量产启动成本是每用例
        # 固定开销，容量与判定逻辑无关（生产 64）
        self.config_dict["reward"].update({
            "pretrain_report_json": str(self.run_dir / "pretrain_report.json"),
            "replay_buffer_capacity": 8,
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
        + 报告），报告含最终 held-out AUC 与完整口径指纹。"""
        # 门槛 0.01 恒达标：判定逻辑走「AUC 达阈值」分支且零步终止（判定
        # 分支的专属用例；「密集步进」路径见 dense-steps 用例）
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
        assert 0.0 <= report.final_heldout_auc <= 1.0
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
        """AC：终止条件 = AUC 达阈值或步数上限（两者皆配置化）——每步
        落盘预训练事件（判别字段 + loss/AUC/buffer 占用），事件流 AUC
        随密集步进上升（实测轨迹：head5 ≈ 0.53 → tail5 ≈ 0.61，12 步；
        disc_lr 提高是 fixture 加速，判定逻辑与生产同一条）。

        阈值 0.99 不可达：走满步数上限分支（「AUC 达阈值」分支由 0.01
        恒达标用例覆盖——两者共用同一个判定点）。"""
        scenario.write_config(reward={
            "pretrain_gate_auc": 0.99,
            "pretrain_max_steps": 12,
            "pretrain_fake_batch": 4,
            "disc_lr": 2e-4,
        })
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        report = scenario.report()
        events = scenario.events()
        # 判定逻辑不变量：达标步不发事件 → 事件数 == 完成步数；步号连续
        assert len(events) == report.steps_completed == 12
        assert [event["step"] for event in events] == list(range(12))
        assert all(event["event"] == "pretrain" for event in events)
        assert all("loss_discriminator" in event for event in events)
        # 曲线的操作者可见面：终点报出指标流路径与事件数（离线查看收敛
        # 曲线、校准门槛阈值的数据源）
        assert f"{len(events)} 条 pretrain 事件" in result.stdout
        assert report.gate_passed is False
        # 判定与报告值的一致性（走满路径：落盘 checkpoint 的补测值 < 门槛）
        assert report.final_heldout_auc < report.gate_auc
        # 密集步进拉动 AUC（平滑口径：首尾各 3 步均值；单步 patch 级 AUC
        # 在 fixture 小批量下噪声大）
        aucs = [event["heldout_auc"] for event in events]
        assert sum(aucs[-3:]) / 3 > sum(aucs[:3]) / 3

    def test_cross_modal_group_uses_same_path(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：组1/组2 走同一条路径，仅 config 不同（组2 fake 走
        ControlNet 条件分布）。"""
        scenario.write_config(
            group="cross-modal", reward={"pretrain_gate_auc": 0.01},
        )
        result = scenario.pretrain()
        assert result.code == 0, result.stderr
        report = scenario.report()
        assert report.group == "cross-modal"


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

    def test_cross_modal_conditions_from_controlnet_path(
        self, scenario: PretrainScenario,
    ) -> None:
        """AC：组1/组2 走同一条路径、仅 config 不同——组2 的条件分布装配
        为 ControlNet 交叉模态采样器（GroupPolicy 按组分派）。"""
        scenario.write_config(
            group="cross-modal", reward={"pretrain_gate_auc": 0.01},
        )
        config = scenario.config()
        run = PretrainRun.init(config, scenario.tmp_path / "assembly_run")
        driver = PretrainDriver(config, run, device=torch.device("cpu"))
        assert isinstance(driver.policy.conditions, CrossModalConditionSampler)

    def test_rejects_fake_batch_below_current_zone(
        self, scenario: PretrainScenario,
    ) -> None:
        """装配守卫：每步量产 fake 不足判别器更新批的当前半区 → 拒绝。"""
        scenario.write_config(reward={"pretrain_fake_batch": 1})
        config = scenario.config()  # K=4 → 当前半区需要 2 条
        run = PretrainRun.init(config, scenario.tmp_path / "assembly_run")
        with pytest.raises(ValueError, match="fake"):
            PretrainDriver(config, run, device=torch.device("cpu"))
