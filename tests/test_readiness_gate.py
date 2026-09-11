"""RM readiness gate（ADR-0007）：train 入口的上岗硬检查。

验收面（issue #60）：

- warm-start 接入：train 判别器装配从预训练报告守卫重载（权重 =
  预训练产物，非随机初始化）；
- preflight 重算 AUC 双分支：达标放行（fixture 低阈值 + 自产小产物）、
  人为低于阈值拒绝（可读报错含实测值与阈值 + run 目录回滚）；
- AUC 重算一致性：同 scorer 快照、同 fake 批的重算值与报告记录值逐位
  一致（重算与预训练测量是同一份 HeldOutAuc.compute 口径）；
- 守卫拒绝：缺报告 / kind 不符 / 工件损坏 / 口径指纹不匹配；
- resume 跳过门槛（续训状态已含判别器全量状态）；字段引入前的旧 run
  快照续训被 schema 拒绝；
- 端到端：fixture CLI train 以 warm-start 起跑，iter 事件 heldout_auc
  从门槛之上起步。
"""

import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkAssembler
from cynosure.pretrain import (
    PretrainDriver,
    PretrainProvenance,
    PretrainReport,
    PretrainRun,
)
from cynosure.reward.artifacts import ChannelStats
from cynosure.train import (
    AmpContext,
    GranularGrpoTrainer,
    RunArtifacts,
    TrainingRuntime,
)
from cynosure.train.gate import ReadinessGate
from cynosure.train.rng import TrainingRngStreams
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    PretrainLightweightReward,
)

FIXTURE_GATE = 0.51
"""fixture 低阈值（Fixture.config 声明）：chance 带上沿之上、fixture
预训练小产物可达——门槛判定逻辑的专属取值（生产默认 0.65 不变）。"""

REJECT_GATE = 0.99
"""人为不可达阈值：把重算值压在阈值之下的拒绝分支驱动值。"""


class StubAuc:
    """HeldOutAuc 的判定替身：固定实测值、记录调用（重算口径的观测面）。"""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[int, str | None]] = []

    def compute(self, fake_latents: torch.Tensor, modality=None) -> float:
        self.calls.append((fake_latents.shape[0], modality))
        return self.value


class FailingAuc:
    """HeldOutAuc 的失败替身：以非 ValueError 的工件读盘异常失败
    （held-out manifest 条目缺失/损坏的真实异常面）。"""

    def compute(self, fake_latents: torch.Tensor, modality=None) -> float:
        raise FileNotFoundError("held-out latent 缺失: heldout_latents/003.pt")


class SplitVerdictDist:
    """DistributedContext 的集合裁决替身：本地判定结果取自提交物，
    ``peer_errors`` 预置对端（rank 1..N）的本地判定结果。"""

    rank = 0

    def __init__(self, peer_errors: list[str | None]) -> None:
        self._peer_errors = peer_errors

    def all_gather(self, items: list) -> list[list]:
        received = [items[0]]
        for source, error in enumerate(self._peer_errors, start=1):
            received.append({"rank": source, "error": error})
        return [[entry] for entry in received]


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
        ``reward_overrides`` 可覆盖（如重演用例的 gate=0.01）。"""
        config = PretrainLightweightReward.apply(
            ConfigLoader.load(self.config_path),
        )
        for key, value in reward_overrides.items():
            setattr(config.reward, key, value)
        path = self.tmp_path / "pretrain_config.json"
        path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        return self.cli.run("pretrain", "--config", str(path))

    def train(self) -> CliResult:
        return self.cli.run(
            "train", "--config", str(self.config_path),
            "--run-dir", str(self.run_dir),
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

    def events(self) -> list[dict]:
        artifacts = RunArtifacts(RunArtifacts.layout(self.run_dir))
        return artifacts.read_events()


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> GateScenario:
    return GateScenario(cli, tmp_path)


@pytest.fixture
def pretrained(scenario: GateScenario) -> GateScenario:
    """已完成预训练的场景（fixture 低阈值，重算达标）。"""
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
        上岗判别力与预训练报告值脱钩。"""
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
    """preflight 重算 AUC 双分支（fixture 双分支判定专属测试）。"""

    def test_fixture_low_threshold_passes_and_train_starts_above_gate(
        self, pretrained: GateScenario,
    ) -> None:
        """达标放行：fixture 低阈值 + 自产小产物 → train 成功，iter 事件
        heldout_auc 从门槛之上起步（T14 的「一眼确认非冷启动形态」）。"""
        result = pretrained.train()
        assert result.code == 0, result.stderr
        events = pretrained.events()
        iter_events = [event for event in events if event["event"] == "iter"]
        assert iter_events, result.stderr
        first_auc = iter_events[0]["heldout_auc"]
        assert first_auc >= FIXTURE_GATE
        # 门槛之上的起步是 warm-start 的观测面：冷启动形态徘徊 chance 带
        # （~0.5±0.02），预训练产物显著出带
        report = pretrained.report()
        assert first_auc >= report.gate_auc

    def test_gate_below_threshold_rejects_with_readable_error(
        self, pretrained: GateScenario,
    ) -> None:
        """拒绝分支：人为抬阈值到不可达 → 可读报错含实测值与阈值 +
        run 目录回滚（沿用 preflight 失败语义；拒绝发生在 Baseline
        采样等昂贵启动动作之前，未产出工件的目录整体删除）。"""
        pretrained.patch_reward(pretrain_gate_auc=REJECT_GATE)
        result = pretrained.train()
        assert result.code == 2
        assert "RM readiness gate" in result.stderr
        # 实测值与阈值都在报错里（AC「含实测值与阈值」）：从消息提取
        # 重算值，断言其低于阈值且在 AUC 值域内
        match = re.search(
            r"held-out AUC (\d\.\d{4}) < 门槛 0\.9900", result.stderr,
        )
        assert match, result.stderr
        measured = float(match.group(1))
        assert 0.0 < measured < REJECT_GATE
        assert not pretrained.run_dir.exists()  # 未产出工件 → 已回滚


class TestRecomputeConsistency:
    """AUC 重算一致性：同 scorer 快照 + 同 fake 批的重算值 = 报告记录值。"""

    def test_gate_recomputation_reproduces_reported_auc(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """重演预训练的测量（同 seed 同 RNG 流 → 同 fake 批）：报告
        checkpoint 装载的 scorer 与预训练时权重逐位一致（同 scorer
        快照），其对同一批 fake 的重算值与报告 ``final_heldout_auc``
        逐位一致——门槛不信任报告旧值，但重算必须能复现报告值
        （重算与预训练测量是同一份 HeldOutAuc.compute 口径）。

        gate=0.01 恒 0 步达标：报告值 = 达标测量与复测（两批独立
        测量取小——producer 侧成功判据对单批测量噪声鲁棒）中较小者；
        消耗序确定（seed_base 的 base 分区抽取 → 达标测量批 → 复测
        批）。"""
        scenario = GateScenario(cli, tmp_path)
        scenario.write_inputs()
        result = scenario.pretrain(pretrain_gate_auc=0.01)
        assert result.code == 0, result.stderr
        report = scenario.report()
        assert report.steps_completed == 0
        config = ConfigLoader.load(scenario.config_path)
        # 与 scenario.pretrain 同套轻量五元组（重演消耗序的前提：fake 批
        # /缓冲容量/LR 一致；单点定义避免两处漂移）
        pretrain_config = PretrainLightweightReward.apply(config)
        run = PretrainRun.init(
            pretrain_config, tmp_path / "replay_run",
        )
        driver = PretrainDriver(pretrain_config, run, device=torch.device("cpu"))
        # 报告 checkpoint 重载的 scorer == 预训练时的权重（0 步路径 =
        # 冷启动初始权重，seed+6 fork 随同 config seed 确定）
        scorer = report.load_discriminator(
            config, device=torch.device("cpu"),
        )
        restored = NetworkAssembler.loadable_state_dict(scorer.discriminator)
        live = NetworkAssembler.loadable_state_dict(driver.rewards.discriminator)
        assert all(torch.equal(restored[key], live[key]) for key in restored)
        # 消耗序重演：先 base 分区（seed_base）、再达标测量批、再复测批
        # → 同流同批；报告值 = 两次独立测量的较小者
        driver.rollout.base_partition_samples(
            driver.rewards.buffer.base_capacity,
        )
        first = driver.rewards.auc.compute(
            driver.rollout.base_partition_samples(4),
        )
        second = driver.rewards.auc.compute(
            driver.rollout.base_partition_samples(4),
        )
        assert report.final_heldout_auc == pytest.approx(
            min(first, second), rel=0.0, abs=0.0,
        )

    def test_readiness_gate_uses_full_pool_modality_free_recompute(self) -> None:
        """重算口径 = 全池混采（modality=None，与预训练 gate 同口径）：
        fixture 预训练 fake 批跨条件混合、无单一目标序列可归因。"""
        auc = StubAuc(0.7)
        config = _minimal_gate_config()
        gate = ReadinessGate(config, auc, SplitVerdictDist([]))
        fakes = torch.zeros(3, 4, 16, 16, 8)
        assert gate.check(fakes) == pytest.approx(0.7)
        assert auc.calls == [(3, None)]


class TestGateVerdictUnit:
    """判定单测（注入替身）：阈值边界与集合裁决。"""

    def test_measured_at_threshold_passes(self) -> None:
        config = _minimal_gate_config()
        gate = ReadinessGate(config, StubAuc(0.65), SplitVerdictDist([]))
        assert gate.check(torch.zeros(1, 4, 16, 16, 8)) == pytest.approx(0.65)

    def test_measured_below_threshold_raises_with_values(self) -> None:
        config = _minimal_gate_config()
        gate = ReadinessGate(config, StubAuc(0.437), SplitVerdictDist([]))
        with pytest.raises(ValueError, match=r"0\.437.*0\.65") as exc_info:
            gate.check(torch.zeros(1, 4, 16, 16, 8))
        assert "RM readiness gate" in str(exc_info.value)

    def test_peer_rank_failure_rejects_collectively(self) -> None:
        """任一 rank 的重算失败 = 全体一致拒绝（分布式下各 rank base fake
        独立演化——本 rank 达标不代表全体达标，分歧退出会让其余 rank 停在
        集合操作）。"""
        config = _minimal_gate_config()
        gate = ReadinessGate(
            config, StubAuc(0.9), SplitVerdictDist(["rank 1 失败详情"]),
        )
        with pytest.raises(ValueError, match="rank 1"):
            gate.check(torch.zeros(1, 4, 16, 16, 8))

    def test_artifact_read_failure_converges_to_collective_rejection(
        self,
    ) -> None:
        """本地重算的非 ValueError 异常（held-out manifest 条目缺失 =
        FileNotFoundError；损坏 = 反序列化异常）也收敛为 local_error 进
        集体裁决——捕窄会让失败 rank 先于 all_gather 退出、其余 rank
        永等在集合操作（拒绝方退出、通过方挂死）。"""
        config = _minimal_gate_config()
        gate = ReadinessGate(config, FailingAuc(), SplitVerdictDist([]))
        with pytest.raises(ValueError, match="held-out latent 缺失"):
            gate.check(torch.zeros(1, 4, 16, 16, 8))


class TestResumeSkipsGate:
    """resume 跳过门槛：续训状态已含判别器全量状态。"""

    def test_resume_passes_gate_that_new_run_would_fail(
        self, pretrained: GateScenario,
    ) -> None:
        """门槛与续训的语义分叉：阈值抬到不可达并同步改写原 run 的
        config 快照（续训对账一致）——resume 放行（续训状态已含判别器
        全量状态，门槛若在 resume 路径执行，重算值 < 0.99 必拒）。"""
        assert pretrained.train().code == 0
        pretrained.patch_reward(pretrain_gate_auc=REJECT_GATE)
        snapshot = pretrained.run_dir / "config.json"
        data = json.loads(snapshot.read_text(encoding="utf-8"))
        data["reward"]["pretrain_gate_auc"] = REJECT_GATE
        snapshot.write_text(json.dumps(data), encoding="utf-8")
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
        report = PretrainReport(
            group="modal-label",
            latent_shape=(4, 16, 16, 8),
            final_heldout_auc=0.72,
            steps_completed=40,
            gate_auc=0.65,
            gate_passed=True,
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
            ),
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


def _minimal_gate_config() -> CynosureConfig:
    """ReadinessGate 单测的最小 config（只消费 pretrain_gate_auc）。"""
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
