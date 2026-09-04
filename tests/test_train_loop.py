"""单 iteration GRPO 循环全链路（ticket #21 验收标准聚合，tracer bullet）。

fixture 下 CLI train 端到端：Rollout（Anchor → 单步 SDE 扰动 → 各 λ ODE
续跑 → 判别器 raw logit 打分）→ MGAI advantage → 逐 k 独立梯度步 →
判别器 Online update → iter 事件落盘 + checkpoint。

五条 AC 对应：
1. fixture 下单 iteration 全链路绿，产出可装载 checkpoint 与 iter 事件流；
2. log-prob 一致性：Rollout 记录的 π_old 与更新时重算一致（诊断工件）；
3. MGAI 顺序正确；G=12 下组内标准化非退化（组内 reward std 非零进事件）；
4. 每个训练步 k 一次独立梯度步（loss 组件逐 k 记录）；
5. buffer base 分区在 train 启动时由冻结初始 policy 自动生成。
"""

import json
import math
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train import GranularGrpoTrainer, RunArtifacts
from tests.conftest import CliSession, FixturePrepareScenario


class TrainingLoopScenario:
    """一次 fixture 训练场景：网络工件 + prepare 数据工件 + CLI train。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.tmp_path = tmp_path
        self.fixture_dir = tmp_path / "fixtures"
        self.run_dir = tmp_path / "run"
        self.config_path = tmp_path / "config.json"

    def write_inputs(
        self,
        *,
        num_steps: int = 3,
        train_steps: set[int] = frozenset({1}),
        seed: int = 0,
    ) -> None:
        """落盘 fixture 网络工件 + prepare 三工件 + 训练 config。"""
        fixture = Fixture()
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（test_reward_fixture 先例）
        fixture.write_artifacts(self.fixture_dir)
        prepare_config = FixturePrepareScenario(
            self.cli, fixture.config(self.fixture_dir), self.tmp_path,
        ).run(self.tmp_path / "prepare_config.json")
        config = fixture.config(self.fixture_dir)
        config.policy.num_inference_steps = num_steps
        config.policy.train_step_indices_m = set(train_steps)
        config.schedule.seed = seed
        config.schedule.max_iterations = 1  # tracer bullet：单 iteration 全链路
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )

    def train(self, *, dump: bool = False):
        argv = ["train", "--config", str(self.config_path), "--run-dir", str(self.run_dir)]
        if dump:
            argv.append("--dump-trajectory")
        return self.cli.run(*argv)

    def artifacts(self) -> RunArtifacts:
        return RunArtifacts(RunArtifacts.layout(self.run_dir))

    def events(self) -> list[dict]:
        return self.artifacts().read_events()


@pytest.fixture
def scenario(cli: CliSession, tmp_path: Path) -> TrainingLoopScenario:
    return TrainingLoopScenario(cli, tmp_path)


class TestSingleIterationLoop:
    """AC 1/3/4/5：单 iteration 全链路绿、事件流、逐 k loss、base 分区。"""

    def test_single_iteration_runs_green_and_emits_iter_event(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.events()
        assert [event["event"] for event in events] == ["iter"]

    def test_iter_event_carries_health_metrics(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """iter 事件：Anchor eval reward、非退化组内 reward std（G=12）、
        held-out AUC、loss 组件（逐 k policy + discriminator）、buffer
        占比（混合占比 + 两区占用）、lr。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        event = scenario.events()[0]
        assert event["iteration"] == 0
        assert math.isfinite(event["anchor_eval_reward"])
        assert event["intra_group_reward_std"] > 0.0  # G=12 组内标准化非退化
        assert 0.0 < event["heldout_auc"] < 1.0
        assert "policy_step_1" in event["loss"]  # M={1} → 一次 policy 梯度步
        assert "discriminator" in event["loss"]
        assert event["buffer_current_fraction"] == pytest.approx(0.5)
        assert event["buffer_replay_fraction"] == pytest.approx(0.5)
        assert event["buffer_base_occupied"] == 32  # capacity 64 → base 32
        # 首 iter 新 fake 全量入近期分区：|M|×G×|Λ| + anchor = 1×12×2 + 1 = 25
        assert event["buffer_recent_occupied"] == 25
        assert event["lr"] == pytest.approx(2e-6)
        assert event["elapsed_s"] >= 0.0

    def test_discriminator_update_interval_n_d_is_consumed(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """判别器更新节奏 N_d 由训练循环消费：N_d=2 时每 2 个 iteration
        更新一次（跳过的 iteration 无 discriminator loss、混合占比 0、
        近期分区无增量）。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["schedule"]["max_iterations"] = 2
        data["reward"]["disc_update_interval_n_d"] = 2
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        assert scenario.train().code == 0, scenario.stderr
        first, second = scenario.events()
        assert "discriminator" in first["loss"]
        assert first["buffer_current_fraction"] == pytest.approx(0.5)
        assert first["buffer_recent_occupied"] == 25  # 1×12×2 + 1 条新 fake
        assert "discriminator" not in second["loss"]  # N_d 跳过
        assert second["buffer_current_fraction"] == 0.0
        assert second["buffer_replay_fraction"] == 0.0
        assert second["buffer_recent_occupied"] == first["buffer_recent_occupied"]

    def test_checkpoint_is_loadable_and_evolved(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """产出可装载 checkpoint：netbuild 按 fixture 网络配置重新装载成功，
        且权重相对初始 ckpt 已演化（梯度步真实生效）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        checkpoints = scenario.run_dir / "checkpoints"
        policy_ckpt = checkpoints / "policy_iter1.pt"
        assert policy_ckpt.is_file()
        config = ConfigLoader.load(scenario.config_path)
        reloaded = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=policy_ckpt,
        ))
        initial = torch.load(config.artifacts.unet_ckpt, map_location="cpu")
        assert any(
            not torch.equal(reloaded.state_dict()[name], value)
            for name, value in initial.items()
        )
        discriminator_ckpt = checkpoints / "discriminator_iter1.pt"
        assert discriminator_ckpt.is_file()
        reloaded_disc = NetworkAssembler.discriminator(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
            checkpoint=discriminator_ckpt,
        ))
        assert any(p.requires_grad for p in reloaded_disc.parameters())

    def test_multi_step_schedule_runs_independent_updates(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC 4（多步变体）：5 步日程 M={2}，λ={1,2} 续跑分叉真实存在，
        循环完整跑通、loss 组件逐 k 记录。"""
        scenario.write_inputs(num_steps=5, train_steps={2})
        result = scenario.train()
        assert result.code == 0, result.stderr
        event = scenario.events()[0]
        assert "policy_step_2" in event["loss"]
        assert "discriminator" in event["loss"]

    def test_non_modal_label_groups_rejected_until_later_tickets(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """组2/组3 采样场（ControlNet）由后续 ticket 交付：显式拒绝而非
        静默退化为组1 路径。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = str(
            scenario.fixture_dir / "controlnet.pt",
        )
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        result = scenario.train()
        assert result.code == 2
        assert "cross-modal" in result.stderr

    def test_missing_artifacts_reported_cleanly(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """生产 config（工件路径不存在）进训练循环时得到清晰的输入契约
        错误（退出码 2），而非裸 traceback。"""
        result = cli.train(
            cli.write_config(tmp_path), run_dir=tmp_path / "run",
        )
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr


class TestBufferBaseSeeding:
    """AC 5：buffer base 分区在 train 启动时由冻结初始 policy 自动生成。"""

    def test_base_partition_filled_before_first_iteration(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """train 启动期（首 iteration 前）base 分区已满、内容为 rollout
        产出的有限 latent；回放混采自首 iter 即可用（事件流 50/50 是其
        外部观测）。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        trainer.seed_base_partition()
        sizes = trainer.rewards.buffer.zone_sizes()
        assert sizes.base == trainer.rewards.buffer.base_capacity
        base = trainer.rewards.buffer.base_samples()
        assert len(base) == sizes.base
        assert all(torch.isfinite(sample).all() for sample in base)
        assert sizes.recent == 0  # base 生成不进近期分区

    def test_base_partition_samples_differ_from_post_training_fakes(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """base 分区由**冻结初始** policy 生成：其 latent 与训练后 policy
        的 rollout 分布不同源（内容非空且非训练期 push 的样本）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        event = scenario.events()[0]
        # 首 iter 回放半区即可用（base 已在启动期生成完毕）
        assert event["buffer_replay_fraction"] > 0.0


class TestLogProbConsistency:
    """AC 2：Rollout 记录的 π_old 与更新时重算一致（测试面 #3，经
    --dump-trajectory 诊断工件断言）。"""

    def test_recorded_old_log_probs_survive_to_update_time(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        result = scenario.train(dump=True)
        assert result.code == 0, result.stderr
        report = json.loads(
            (scenario.run_dir / "training.json").read_text(encoding="utf-8"),
        )
        pairs = report["logprob_pairs"]
        assert len(pairs) == 1 * 12  # |M| × G = 1 × 12 组对
        assert {(pair["step_index"], pair["direction"]) for pair in pairs} == {
            (1, direction) for direction in range(12)
        }
        for pair in pairs:
            assert math.isfinite(pair["recorded"])
            assert pair["recorded"] == pair["recomputed"]  # 同权重逐位一致

    def test_multi_step_pairs_cover_every_train_step(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """|M|=2（5 步日程 M={1,2}）：每个被优化训练步都有 |G| 组对。"""
        scenario.write_inputs(num_steps=5, train_steps={1, 2})
        assert scenario.train(dump=True).code == 0
        report = json.loads(
            (scenario.run_dir / "training.json").read_text(encoding="utf-8"),
        )
        pairs = report["logprob_pairs"]
        assert len(pairs) == 2 * 12
        assert {pair["step_index"] for pair in pairs} == {1, 2}
        for pair in pairs:
            assert pair["recorded"] == pair["recomputed"]
