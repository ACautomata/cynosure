"""RM readiness gate（ADR-0008 决策 5）：train 入口的白名单上岗检查。

验收面（issue #88）：

- gate 读 per-condition 报告 + 条件白名单：白名单空 → 拒绝启动（报错
  含各条件 per-condition 实测值，沿 preflight 失败语义由 CLI 干净报错
  + 回滚 run 目录）；非空 → 放行，未过线条件不阻塞 run；
- 启动期池化重算语义废止：gate 不再逐 rank 重算 AUC（ADR-0007 重算
  代码路径清理）；数据口径漂移由 warm-start 装载的指纹对照把守；
- 运行时白名单接线：train 循环的逐 iteration 查询面（RewardCoordinator.
  whitelist）与 gate 判定同源（同一报告产出）；resume 占位全放行；
- resume 跳过 gate（续训状态已含判别器全量状态，恢复点不重查白名单）；
- warm-start 接入与守卫拒绝（缺报告 / kind 不符 / 工件损坏 / 口径
  指纹 / latent 形状）沿既有语义不变；
- 端到端：fixture CLI train 以 warm-start 起跑，iter 事件 heldout_auc
  从报告门槛之上起步。
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from cynosure.config import MODALITIES, ConfigLoader, CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkAssembler
from cynosure.pretrain import (
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.pretrain.driver import PretrainDriver
from cynosure.reward.artifacts import ChannelStats
from cynosure.train import (
    AmpContext,
    GranularGrpoTrainer,
    RunArtifacts,
    TrainingRuntime,
)
from cynosure.train.gate import ReadinessGate
from cynosure.train.rng import TrainingRngStreams
from cynosure.train.whitelist import ConditionWhitelist
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    PretrainLightweightReward,
)

FIXTURE_GATE = 0.51
"""fixture 低阈值（Fixture.config 声明）：chance 带上沿之上、fixture
预训练小产物可达——预训练过线判定的专属取值（生产默认 0.65 不变）。"""


class GateScenario:
    """gate 端到端场景：prepare → pretrain（自产小产物）→ train 全链。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.tmp_path = tmp_path
        self.fixture_dir = tmp_path / "fixtures"
        self.run_dir = tmp_path / "run"
        self.config_path = tmp_path / "config.json"

    def write_inputs(self, *, seed: int = 0, group: str = "modal-label") -> CynosureConfig:
        """落盘 fixture 网络工件 + prepare 三工件 + 训练 config。"""
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制
        fixture = Fixture()
        fixture.write_artifacts(self.fixture_dir)
        prepared = FixturePrepareScenario(
            self.cli, fixture.config(self.fixture_dir, group=group), self.tmp_path,
        ).run(self.tmp_path / "prepare_config.json")
        config = fixture.config(self.fixture_dir, group=group)
        config.schedule.seed = seed
        config.schedule.max_iterations = 1
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )
        return config

    def pretrain(self, **reward_overrides) -> CliResult:
        """以 config 的预训练轻量变体跑 pretrain（报告路径 = config 声明）。

        轻量五元组与 gate 0.60 留 margin 的 rationale 集中在
        ``PretrainLightweightReward``（conftest）；显式
        ``reward_overrides`` 可覆盖（如白名单空用例的 gate=0.99）。"""
        config = PretrainLightweightReward.apply(
            ConfigLoader.load(self.config_path),
        )
        for key, value in reward_overrides.items():
            setattr(config.reward, key, value)
        path = self.tmp_path / "pretrain_config.json"
        path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        return self.cli.run("pretrain", "--config", str(path))

    def train(self, run_dir: Path | None = None) -> CliResult:
        return self.cli.run(
            "train", "--config", str(self.config_path),
            "--run-dir", str(run_dir if run_dir is not None else self.run_dir),
        )

    def resume(self) -> CliResult:
        return self.cli.run(
            "train", "--config", str(self.config_path),
            "--run-dir", str(self.run_dir), "--resume",
        )

    def patch_reward(self, **values) -> None:
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        data["reward"].update(values)
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

    def report(self) -> PretrainReport:
        config = ConfigLoader.load(self.config_path)
        return PretrainReport.load(Path(config.reward.pretrain_report_json))

    def rewrite_report(self, **fields) -> None:
        """篡改报告字段（白名单/实测值的语义构造用）：报告经 load()
        之外的路径改写——测试直写 JSON 后由下一次装载校验。"""
        config = ConfigLoader.load(self.config_path)
        path = Path(config.reward.pretrain_report_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        data.update(fields)
        path.write_text(json.dumps(data), encoding="utf-8")

    def events(self) -> list[dict]:
        artifacts = RunArtifacts(RunArtifacts.layout(self.run_dir))
        return artifacts.read_events()


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> GateScenario:
    return GateScenario(cli, tmp_path)


@pytest.fixture
def pretrained(scenario: GateScenario) -> GateScenario:
    """已完成预训练的场景（fixture 低阈值，白名单非空）。"""
    scenario.write_inputs()
    result = scenario.pretrain()
    assert result.code == 0, result.stderr
    return scenario


class TestWarmStartAssembly:
    """warm-start 接入：train 判别器装配从预训练报告守卫重载。"""

    def test_discriminator_loads_warm_start_weights(
        self, pretrained: GateScenario,
    ) -> None:
        """装配后判别器权重与预训练产物 checkpoint 逐位一致——非随机
        初始化的冷启动形态（ADR-0007 的 warm-start 语义本体）。"""
        config = ConfigLoader.load(pretrained.config_path)
        artifacts = RunArtifacts.init(config, pretrained.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts, device=torch.device("cpu"))
        saved = torch.load(
            Path(config.reward.pretrain_report_json).parent
            / "checkpoints" / "pretrain_discriminator.pt",
            map_location="cpu", weights_only=True,
        )
        restored = NetworkAssembler.loadable_state_dict(
            trainer.rewards.discriminator,
        )
        assert restored.keys() == saved.keys()
        assert all(torch.equal(restored[key], saved[key]) for key in saved)

    def test_missing_report_rejected_before_run_dir_pollution(
        self, scenario: GateScenario,
    ) -> None:
        """缺报告 = 未预训练：train 入口拒绝（退出码 2）且回滚预占的
        run 目录——RL 不带 warm-start 工件无法启动。"""
        scenario.write_inputs()
        result = scenario.train()
        assert result.code == 2
        assert "预训练报告" in result.stderr
        assert not scenario.run_dir.exists()

    def test_wrong_kind_report_rejected(
        self, pretrained: GateScenario,
    ) -> None:
        """kind 不符的 JSON 冒充预训练报告：装载期拒绝。"""
        config = ConfigLoader.load(pretrained.config_path)
        path = Path(config.reward.pretrain_report_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["kind"] = "real_pool"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = pretrained.train()
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr
        assert not pretrained.run_dir.exists()

    def test_corrupted_checkpoint_rejected(
        self, pretrained: GateScenario,
    ) -> None:
        """预训练产物 checkpoint 损坏（键形不符的 state_dict）：严格装载
        失败 = 输入契约违反，构造期干净报错 + run 目录回滚。"""
        config = ConfigLoader.load(pretrained.config_path)
        checkpoint = (
            Path(config.reward.pretrain_report_json).parent
            / "checkpoints" / "pretrain_discriminator.pt"
        )
        torch.save({"bogus": torch.zeros(1)}, checkpoint)
        result = pretrained.train()
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr
        assert not pretrained.run_dir.exists()

    def test_data_provenance_mismatch_rejected(
        self, pretrained: GateScenario,
    ) -> None:
        """口径指纹不匹配（预训练后换了 channel stats 来源）：装载期
        拒绝——判别器的输入标准化口径与预训练不同一，静默放行会让
        上岗判别力与预训练报告值脱钩。gate 直接信任报告值后，指纹
        对照是数据口径漂移的唯一把守（ADR-0008-05：重算复核废止）。"""
        config = ConfigLoader.load(pretrained.config_path)
        stats_path = Path(config.reward.channel_stats_json)
        stats = ChannelStats.model_validate(
            json.loads(stats_path.read_text(encoding="utf-8")),
        )
        stats.std = [component + 0.5 for component in stats.std]
        stats_path.write_text(stats.model_dump_json(), encoding="utf-8")
        result = pretrained.train()
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr
        assert not pretrained.run_dir.exists()

    def test_latent_shape_mismatch_rejected(
        self, pretrained: GateScenario,
    ) -> None:
        """报告 latent_shape 与当前 run 不符（换分辨率）：口径指纹与
        判别器形态指纹都不覆盖它，全卷积 scorer 可用旧 shape 的 real
        评新 shape 的 fake 静默通过 gate 并带错位数据进在线更新——
        装载前显式对照拒绝。"""
        data = json.loads(pretrained.config_path.read_text(encoding="utf-8"))
        data["latent_shape"] = [4, 16, 16, 4]
        data["policy"]["input_img_size_numel"] = 16 * 16 * 4  # numel 锚随行
        pretrained.config_path.write_text(
            json.dumps(data), encoding="utf-8",
        )
        result = pretrained.train()
        assert result.code == 2
        assert "latent 形状不符" in result.stderr
        assert not pretrained.run_dir.exists()


class TestGateVerdict:
    """白名单上岗判定的端到端双分支（CLI 全链）。"""

    def test_whitelist_nonempty_passes_and_train_starts_above_gate(
        self, pretrained: GateScenario,
    ) -> None:
        """非空放行：白名单非空（fixture 低阈值预训练全条件过线）→
        train 成功，iter 事件 heldout_auc 从报告门槛之上起步（warm-start
        的观测面：冷启动形态徘徊 chance 带 ~0.5±0.02，预训练产物显著
        出带）。"""
        assert pretrained.report().gate_whitelist  # 前置：名单非空
        result = pretrained.train()
        assert result.code == 0, result.stderr
        events = pretrained.events()
        iter_events = [event for event in events if event["event"] == "iter"]
        assert iter_events, result.stderr
        first_auc = iter_events[0]["heldout_auc"]
        assert first_auc >= FIXTURE_GATE
        report = pretrained.report()
        assert first_auc >= report.gate_auc

    def test_empty_whitelist_rejects_with_per_condition_readings(
        self, scenario: GateScenario,
    ) -> None:
        """白名单空 → 拒绝启动：报错含各条件 per-condition 实测值与
        报告路径，沿 preflight 失败语义（CLI 干净报错 + 回滚未产出
        工件的 run 目录）。预训练侧白名单空照常落盘（诊断产物不丢，
        ADR-0008-04），拒跑由本 gate 把守。"""
        scenario.write_inputs()
        # gate=0.99 不可达：走满步数上限，白名单空（报告 + checkpoint
        # 落盘供诊断，pretrain 本身 exit 0——#87 语义）
        assert scenario.pretrain(pretrain_gate_auc=0.99).code == 0
        report = scenario.report()
        assert report.gate_whitelist == []
        result = scenario.train()
        assert result.code == 2
        assert "RM readiness gate" in result.stderr
        assert "条件白名单为空" in result.stderr
        for modality, value in report.condition_auc.items():
            assert f"held-out AUC[{modality}]: {value:.4f}" in result.stderr
        assert not scenario.run_dir.exists()  # 未产出工件 → 已回滚

    def test_conditions_below_gate_do_not_block_run(
        self, pretrained: GateScenario,
    ) -> None:
        """非空放行不看实测值：名单只含一个条件、名单外条件实测值
        低于门槛——run 照常启动（未过线条件不阻塞，逐 iteration 门控
        兜底由门控消费票接管）。"""
        pretrained.rewrite_report(
            gate_whitelist=["t1n"],
            condition_auc={"t1n": 0.70, "t1c": 0.30, "t2w": 0.30, "t2f": 0.30},
        )
        result = pretrained.train()
        assert result.code == 0, result.stderr


class TestReportReproduction:
    """报告值的测量可复现性：gate 直接信任报告值（重算复核废止），
    报告 ``condition_auc`` 与同 seed 重演测量逐位一致是信任的测量学
    依据。"""

    def test_reported_auc_reproduces_replayed_measurement(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """重演预训练的测量（同 seed 同装配 → 同测量批 → 同读数）：
        报告 checkpoint 装载的 scorer 与预训练时权重逐位一致（同 scorer
        快照），其对同一批**测量批**的测量与报告 ``condition_auc``
        逐位一致。

        gate=0.01 恒 0 步达标：报告值 = 轮转条件集首条件的首测与复测
        （两次测量取小——producer 侧成功判据对单批测量噪声鲁棒）中较小
        者。测量批口径（ADR-0012 决策 5 / #171）：该条件**全量 held-out
        卷**经装配原语定序轮转重构（``measure_condition``——σ 定序、
        ε 走批次起手复位的测量流）。同条件的首测与复测是**独立样本**
        （held-out 抽取序走 heldout_auc 流、测量流复位再走一遍，两次
        各拿新随机数，「首测 + 换批复测」的独立性由此而来）——本测试
        的「逐位一致」来自**跨 run 重演**：重演驱动同 seed 同 config
        从同一 RNG 起点按同一调用序走（首测、复测两跳都重演），故重演
        读数与报告值逐位吻合，而非同 run 内两批相同。"""
        scenario = GateScenario(cli, tmp_path)
        scenario.write_inputs()
        result = scenario.pretrain(pretrain_gate_auc=0.01)
        assert result.code == 0, result.stderr
        report = scenario.report()
        assert report.steps_completed == 0
        config = ConfigLoader.load(scenario.config_path)
        # 与 scenario.pretrain 同套轻量覆写（重演的前提：测量批构造所依赖
        # 的配置逐字一致；单点定义避免两处漂移）
        pretrain_config = PretrainLightweightReward.apply(config)
        run = PretrainRun.init(
            pretrain_config, tmp_path / "replay_run",
        )
        # 重演与预训练同一执行设备口径（CLI _prepare_device 同款平台
        # 检测：集群 = 加速器）——逐位复现要求两侧 driver / 装载同设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        driver = PretrainDriver(pretrain_config, run, device=device)
        # 报告 checkpoint 重载的 scorer == 预训练时的权重（0 步路径 =
        # 冷启动初始权重，seed+6 fork 随同 config seed 确定）
        scorer = report.load_discriminator(
            config, device=device,
        )
        restored = NetworkAssembler.loadable_state_dict(scorer.discriminator)
        live = NetworkAssembler.loadable_state_dict(driver.rewards.discriminator)
        assert all(torch.equal(restored[key], live[key]) for key in restored)
        # 测量重演（per-condition 口径，ADR-0008-04 × ADR-0012）：轮转
        # 条件集首条件（确定性轮转不耗 RNG）的全量 held-out 卷测量批 →
        # 两次同口径测量（driver._measurement 的等价调用序）→ 报告值 =
        # 两次较小者（全量卷池化点估计口径）
        target = driver.policy.conditions.targets()[0]
        first = self._measurement_auc(driver, target)
        second = self._measurement_auc(driver, target)
        assert report.condition_auc[target] == pytest.approx(
            min(first, second), rel=0.0, abs=0.0,
        )

    @staticmethod
    def _measurement_auc(driver: PretrainDriver, target: str) -> float:
        """driver 测量路径的等价调用（同 assembly 入口、同判别器快照）：
        held-out 全量卷 → 定序轮转重构 → 卷级聚类池化点估计。"""
        reals = driver.rewards.auc.condition_latents(target)
        batch = driver.rewards.assembler.measure_condition(reals, target)
        return driver.rewards.auc.compute_volume_clusters(
            batch.reals, batch.fakes, target,
        ).pooled_auc()


class TestGateVerdictUnit:
    """判定单测（真实值对象注入）：白名单空/非空的分支与报错内容。"""

    def test_empty_whitelist_rejects_with_readings_and_report_path(
        self,
    ) -> None:
        """白名单空 → ValueError：含各条件实测值（:4f 格式）与报告
        路径（诊断入口）。"""
        whitelist = ConditionWhitelist.from_report(_report(
            whitelist=[], condition_auc={"t1n": 0.4321, "t2w": 0.5123},
        ))
        gate = ReadinessGate(_minimal_gate_config(), whitelist)
        with pytest.raises(ValueError) as exc_info:
            gate.check()
        message = str(exc_info.value)
        assert "RM readiness gate" in message
        assert "条件白名单为空" in message
        assert "held-out AUC[t1n]: 0.4321" in message
        assert "held-out AUC[t2w]: 0.5123" in message
        assert "report.json" in message

    def test_nonempty_whitelist_passes_ignoring_readings(self) -> None:
        """非空放行：判定只看名单成员——名单外条件的实测值再低也
        不阻塞（未过线条件不阻塞 run）。"""
        whitelist = ConditionWhitelist.from_report(_report(
            whitelist=["t2w"],
            condition_auc={"t1n": 0.30, "t2w": 0.70},
        ))
        gate = ReadinessGate(_minimal_gate_config(), whitelist)
        gate.check()  # 不抛


class TestConditionWhitelist:
    """条件白名单值对象：来源收口与查询面。"""

    def test_from_report_carries_members_and_readings(self) -> None:
        report = _report(
            whitelist=["t2w", "t1n"],
            condition_auc={"t1n": 0.72, "t2w": 0.66, "t2f": 0.40},
        )
        whitelist = ConditionWhitelist.from_report(report)
        assert whitelist.members == ("t2w", "t1n")  # 报告产出序
        assert whitelist.measured == {
            "t1n": 0.72, "t2w": 0.66, "t2f": 0.40,
        }

    def test_contains_is_membership_query(self) -> None:
        whitelist = ConditionWhitelist.from_report(
            _report(whitelist=["t1n"], condition_auc={"t1n": 0.72}),
        )
        assert "t1n" in whitelist
        assert "t2w" not in whitelist

    def test_unrestricted_covers_every_modality(self) -> None:
        """resume 占位 = 全条件放行（恢复点不重查白名单）。"""
        whitelist = ConditionWhitelist.unrestricted(("t1n", "t1c", "t2w", "t2f"))
        assert len(whitelist) == len(MODALITIES)
        assert all(modality in whitelist for modality in MODALITIES)
        assert whitelist.measured == {}

    def test_empty_report_whitelist_is_empty_object(self) -> None:
        whitelist = ConditionWhitelist.from_report(
            _report(whitelist=[], condition_auc={"t1n": 0.40}),
        )
        assert len(whitelist) == 0
        assert "t1n" not in whitelist


class TestRuntimeWhitelist:
    """运行时白名单接线：train 循环的逐 iteration 查询面与 gate 判定
    消费同一来源（门控消费票的消费面在本票交付）。"""

    def test_train_whitelist_wired_from_report(
        self, pretrained: GateScenario,
    ) -> None:
        config = ConfigLoader.load(pretrained.config_path)
        artifacts = RunArtifacts.init(config, pretrained.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts, device=torch.device("cpu"))
        whitelist = trainer.rewards.whitelist
        report = pretrained.report()
        assert set(whitelist.members) == set(report.gate_whitelist)
        assert whitelist.measured == report.condition_auc

    def test_gated_conditions_queryable_before_run(
        self, pretrained: GateScenario,
    ) -> None:
        """名单外条件（人为移出名单的低实测值条件）经查询面可见——
        门控票逐 iteration 查询的形态预演。"""
        pretrained.rewrite_report(
            gate_whitelist=["t1n"],
            condition_auc={"t1n": 0.70, "t1c": 0.30, "t2w": 0.30, "t2f": 0.30},
        )
        config = ConfigLoader.load(pretrained.config_path)
        artifacts = RunArtifacts.init(config, pretrained.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts, device=torch.device("cpu"))
        whitelist = trainer.rewards.whitelist
        assert "t1n" in whitelist
        for modality in ("t1c", "t2w", "t2f"):
            assert modality not in whitelist

    def test_resume_whitelist_is_unrestricted(
        self, pretrained: GateScenario,
    ) -> None:
        """resume 装配（报告不装载）的白名单占位全放行：恢复点不重查
        白名单，逐 iteration 查询面照常在位（名单的跨 run 持久化随
        门控消费票交付）。"""
        config = ConfigLoader.load(pretrained.config_path)
        artifacts = RunArtifacts.init(config, pretrained.run_dir)
        trainer = GranularGrpoTrainer(
            config, artifacts, device=torch.device("cpu"), resume=True,
        )
        whitelist = trainer.rewards.whitelist
        assert all(modality in whitelist for modality in MODALITIES)


class TestResumeSkipsGate:
    """resume 跳过 gate：续训状态已含判别器全量状态，恢复点不重查
    白名单。"""

    def test_resume_skips_gate_that_new_run_would_fail(
        self, pretrained: GateScenario,
    ) -> None:
        """门槛与续训的语义分叉：报告白名单清空（新 run 若走 gate 必
        拒）——resume 放行（恢复点不重查白名单），同报告的新 run
        拒绝启动。"""
        assert pretrained.train().code == 0
        pretrained.rewrite_report(gate_whitelist=[])
        rejected = pretrained.train(pretrained.run_dir.parent / "run2")
        assert rejected.code == 2
        assert "条件白名单为空" in rejected.stderr
        assert pretrained.resume().code == 0

    def test_legacy_run_snapshot_rejected_for_resume(
        self, pretrained: GateScenario,
    ) -> None:
        """字段引入前的旧 run 快照（无 pretrain_report_json 字段）续训：
        schema 必填校验在快照装载期拒绝——旧口径 run 不再支持续训。"""
        assert pretrained.train().code == 0
        snapshot = pretrained.run_dir / "config.json"
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        del data["reward"]["pretrain_report_json"]
        snapshot.write_text(json.dumps(data), encoding="utf-8")
        result = pretrained.resume()
        assert result.code == 2
        assert "pretrain_report_json" in result.stderr

    def test_resume_survives_pretrain_artifact_cleanup(
        self, pretrained: GateScenario,
    ) -> None:
        """resume 不消费 warm-start 报告：续训状态已含判别器全量状态
        （分片恢复整体覆写判别器权重与 optimizer），预训练 run 目录被
        清理（报告 + checkpoint 删除）后仍可续训——resume 路径强制
        装载会让中断 run 永不可恢复。"""
        assert pretrained.train().code == 0
        config = ConfigLoader.load(pretrained.config_path)
        shutil.rmtree(Path(config.reward.pretrain_report_json).parent)
        assert pretrained.resume().code == 0

    def test_resume_survives_discriminator_ckpt_cleanup(
        self, pretrained: GateScenario,
    ) -> None:
        """resume 占位装配不消费 discriminator_ckpt：旧惯例（config 的
        判别器 checkpoint 工件指向预训练产物——ADR-0007 之前的消费
        形态）下清理整个预训练目录（报告 + checkpoint 一并消失），
        resume 仍可续训；train 语境同不消费该字段（warm-start 权重
        只经报告装载）。"""
        config = ConfigLoader.load(pretrained.config_path)
        report_path = Path(config.reward.pretrain_report_json)
        ckpt = (
            report_path.parent / "checkpoints" / "pretrain_discriminator.pt"
        )
        assert ckpt.is_file()
        data = json.loads(
            pretrained.config_path.read_text(encoding="utf-8"),
        )
        data["artifacts"]["discriminator_ckpt"] = str(ckpt)
        pretrained.config_path.write_text(json.dumps(data), encoding="utf-8")
        assert pretrained.train().code == 0  # train 不消费 discriminator_ckpt
        shutil.rmtree(report_path.parent)
        assert pretrained.resume().code == 0  # 占位装配不读任何工件


class TestAssemblyCombinationGuard:
    """assemble_rewards 的 report/resume 组合态收口（API 层守卫）。"""

    def test_report_with_resume_rejected(self) -> None:
        """report 给定 + resume=True 的矛盾组合显式拒绝：warm-start 守卫
        重载（新 run）与占位装配（续训恢复）是互斥语境，两来源权重同时
        声明时以谁为准的歧义不许静默消解。build 层恒传 report=None +
        resume=True（实际调用图不可达本组合），本守卫收口 API 层的直接
        调用——拒绝发生在消费报告内容与装配链之前（占位 sha256 不经
        校验）。"""
        report = _report(
            whitelist=["t1n"], condition_auc={"t1n": 0.72},
        )
        with pytest.raises(ValueError, match="互斥"):
            TrainingRuntime.assemble_rewards(
                _minimal_gate_config(),
                AmpContext(device=torch.device("cpu"), dtype=torch.float32),
                TrainingRngStreams(0).named(),
                DistributedContext.bootstrap(),
                report=report,
                resume=True,
            )


def _report(
    whitelist: list[str], condition_auc: dict[str, float],
) -> PretrainReport:
    """单测轻量报告（schema 合法即可，不触盘上工件）。"""
    return PretrainReport(
        group="modal-label",
        latent_shape=(4, 16, 16, 8),
        condition_auc=condition_auc,
        gate_whitelist=whitelist,
        steps_completed=40,
        gate_auc=0.65,
        gate_passed=bool(whitelist),
        discriminator_ckpt="checkpoints/pretrain_discriminator.pt",
        provenance=PretrainProvenance(
            real_pool_manifest="real_pool.json",
            real_pool_manifest_sha256="0" * 64,
            heldout_manifest="heldout_real.json",
            heldout_manifest_sha256="0" * 64,
            channel_stats="stats.json",
            channel_stats_sha256="0" * 64,
            discriminator_config="disc.json",
            discriminator_config_sha256="0" * 64,
            discriminator_ckpt="checkpoints/pretrain_discriminator.pt",
            discriminator_ckpt_sha256="0" * 64,
        ),
    )


def _minimal_gate_config() -> CynosureConfig:
    """ReadinessGate 单测的最小 config（只消费 pretrain_report_json 的
    报错路径展示）。"""
    data = {
        "experiment": {"group": "modal-label"},
        "latent_shape": [4, 16, 16, 8],
        "fixture_mode": True,
        "artifacts": {
            "unet_ckpt": "unet.pt",
            "vae_ckpt": "vae.pt",
            "net_config_json": "net.json",
            "modality_mapping_json": "mapping.json",
            "dataset_root": "data",
        },
        "policy": {
            "input_img_size_numel": 2048,
        },
        "reward": {
            "disc_batch_size_k": 4,
            "replay_buffer_capacity": 8,
            "real_pool_manifest": "pool.json",
            "heldout_real_manifest": "heldout.json",
            "channel_stats_json": "stats.json",
            "pretrain_report_json": "report.json",
            "pretrain_gate_auc": 0.65,
        },
        "schedule": {"seed": 0},
    }
    return CynosureConfig.model_validate(data)
