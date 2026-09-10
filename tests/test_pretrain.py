"""pretrain 子命令端到端与产物契约测试（ticket #58 验收标准聚合）。

fixture 合成流：小 pool + 小 fake 集 → 密集步进 → 产物落盘 → 重载 →
守卫生效；AUC 随步数上升可观测（先例量级）。终止条件（门槛/步数上限）
双分支、组1/组2 同路径、kind 守卫、预训练事件的回退记账口径由专属
用例覆盖；真实收敛动力学留给 DCU 实跑（spec「Testing Decisions」：
fixture 只验判定逻辑与数据流）。
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
    CrossModalConditionSampler,
    IterEvent,
    PretrainEvent,
    RunArtifacts,
)
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    MINIMAL_CONFIG_DICT,
)


def iter_event(iteration: int) -> IterEvent:
    """最小合法 iter 事件（rewind 记账口径的对照事件）。"""
    return IterEvent(
        iteration=iteration,
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


class TestPretrainEventContract:
    def test_event_type_discriminant_and_roundtrip(self, tmp_path: Path) -> None:
        """事件判别字段区分于 iter/milestone；同流混存读取无损。"""
        config = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
        artifacts = RunArtifacts.init(config, tmp_path / "run")
        artifacts.append_event(iter_event(0))
        artifacts.append_event(pretrain_event(0))
        events = artifacts.read_events()
        assert [event["event"] for event in events] == ["iter", "pretrain"]
        assert events[1]["step"] == 0
        assert events[1]["heldout_auc"] == pytest.approx(0.5)

    def test_rewind_preserves_pretrain_events(self, tmp_path: Path) -> None:
        """预训练事件不参与续训回退记账（spec「实现警点」）：rewind 只删
        iter/milestone 的半截执行史，混存的 pretrain 事件全量保留。"""
        config = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
        artifacts = RunArtifacts.init(config, tmp_path / "run")
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
        """--run-dir 显式覆盖默认（config 报告路径所在目录）。"""
        scenario.write_config(reward={"pretrain_gate_auc": 0.01})
        override = tmp_path / "override_run"
        result = scenario.pretrain("--run-dir", str(override))
        assert result.code == 0, result.stderr
        assert (override / "pretrain_report.json").is_file()


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
