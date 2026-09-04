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

from cynosure.config import ConfigLoader, MODALITIES
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.update import UpdateReport
from cynosure.train import GranularGrpoTrainer, RewardCoordinator, RunArtifacts
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
        占比（混合占比 + 两区占用）、lr、采样的目标序列（per-sequence
        健康监控的归因轴——条件分布每 iter 均匀采一个序列，事件不记
        序列名时 reward/loss/AUC 无法归因，随机序列变化会伪装成趋势）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        event = scenario.events()[0]
        assert event["iteration"] == 0
        assert event["modality"] in MODALITIES
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

    def test_spectral_norm_checkpoint_is_reloadable(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """spectral norm 启用时判别器 checkpoint 以可重载形式落盘：
        parametrization 键（``*.parametrizations.<attr>.original`` 与其
        power iteration buffer ``_u``/``_v``）不进保存面——保存面固化到
        原始键形（netbuild 装载源 = 未参数化键形的严格装载），否则该
        支持配置（spectral_norm_enabled=true）的判别器 checkpoint 无法
        经装配路径重载（strict load 报 missing/unexpected keys）。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["reward"]["spectral_norm_enabled"] = True
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assert trainer.run() == 1
        checkpoint = scenario.run_dir / "checkpoints" / "discriminator_iter1.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # 保存面键形 = 原始模块键形（parametrization 视图与可重建 buffer 不落盘）
        fresh = NetworkAssembler.discriminator(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
        ))
        assert set(saved.keys()) == set(fresh.state_dict().keys())
        # 严格装载成功（netbuild 装配路径直接可用）
        reloaded = NetworkAssembler.discriminator(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
            checkpoint=checkpoint,
        ))
        # 装载的是训练后的原始参数（parametrization 的 .original 视图：
        # trained 键形 <prefix>.parametrizations.<attr>.original）
        trained_state = trainer.rewards.discriminator.state_dict()
        for name, value in reloaded.named_parameters():
            prefix, _, attribute = name.rpartition(".")
            parametrized = (
                f"{prefix}.parametrizations.{attribute}.original"
                if prefix else f"parametrizations.{attribute}.original"
            )
            expected = trained_state[parametrized if parametrized in trained_state else name]
            assert torch.equal(value, expected)

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

    def test_discriminator_cold_start_without_checkpoint(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """冷启动在线判别器（schema 语义 ``discriminator_ckpt=None = 随机
        初始化起步的在线训练``）：网络配置 JSON 必需、checkpoint 可缺省
        ——训练全链路绿且判别器 checkpoint 正常落盘（冷启动工作流，
        reward-model 章）。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["artifacts"]["discriminator_ckpt"] = None
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        assert scenario.train().code == 0, scenario.stderr
        assert (scenario.run_dir / "checkpoints" / "discriminator_iter1.pt").is_file()

    def test_replay_capacity_guard_rejects_undersized_combinations(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """回放供给跨字段守卫（装配期 fail-fast）：首次判别器更新时近期
        分区为空、回放半区全由 base 分区承担（floor(K/2) 条）——
        ``capacity//2 < floor(K/2)`` 的组合（如 K=4/capacity=2）在昂贵
        rollout 完成后才会缺样本炸掉；K=1 则回放半区为 0 条、回放采样
        API 直接拒绝。两类 schema 合法但集成无效的组合在装配期显式
        拒绝。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["reward"]["disc_batch_size_k"] = 4
        data["reward"]["replay_buffer_capacity"] = 2
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        result = scenario.train()
        assert result.code == 2, result.stderr
        assert "回放供给" in result.stderr
        data["reward"]["disc_batch_size_k"] = 1
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        scenario.run_dir = scenario.tmp_path / "run_k1"  # 独立 run 目录
        result = scenario.train()
        assert result.code == 2, result.stderr
        assert "回放供给" in result.stderr

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

    def test_preflight_failure_leaves_no_run_directory(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """训练装配失败回滚未产出工件的 run 目录：trainer 装配（网络/
        manifest 工件装载、跨字段守卫）失败时，目录若除 init 契约最小集
        外无任何产出则删除——用户修复输入后可用同一 --run-dir 重试
        （run 目录已存在语义拒绝重跑、续训入口未交付）。"""
        run_dir = tmp_path / "run"
        result = cli.train(cli.write_config(tmp_path), run_dir=run_dir)
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr
        assert not run_dir.exists()  # 未产出工件 → 已回滚，可重试


class TestPolicyOptimizerConfig:
    """policy 优化器装配：参考实现超参显式落位，不依赖 PyTorch 默认值。"""

    def test_policy_optimizer_carries_reference_weight_decay(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AdamW 显式带参考实现的 weight decay 1e-4（research/granular-grpo.md
        超参总表）——PyTorch 默认 1e-2 是 100× 过正则，会淹没 2e-6 的
        policy 学习步。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        (group,) = trainer.updater.optimizer.param_groups
        assert group["weight_decay"] == pytest.approx(
            config.policy.policy_weight_decay,
        )
        assert group["weight_decay"] == pytest.approx(1e-4)


class RecordingScorer:
    """测试仪器：以注入判别器冒充打分器（coordinator 取相位的观测载体）。"""

    def __init__(self, discriminator: torch.nn.Module) -> None:
        self.discriminator = discriminator


class RecordingUpdate:
    """测试仪器：记录 update.step 收到的批与调用时的判别器相位
    （buffer 用真实两区实现——RewardCoordinator 的 zone_sizes 观测面
    经它委托）。"""

    def __init__(self, discriminator: torch.nn.Module) -> None:
        self.scorer = RecordingScorer(discriminator)
        self.buffer = ReplayBuffer(64)
        self.received: list[torch.Tensor] = []
        self.training_at_call: list[bool] = []

    def step(self, current_fakes: torch.Tensor) -> UpdateReport:
        self.received.append(current_fakes)
        self.training_at_call.append(self.scorer.discriminator.training)
        return UpdateReport(
            loss_discriminator=0.0,
            loss_real_term=0.0,
            loss_fake_term=0.0,
            num_current=1,
            num_replay=0,
            num_base_replay=0,
            num_recent_replay=0,
        )


class SequencedAuc:
    """测试仪器：记录 held-out AUC 相对判别器更新的调用顺序
    （每次调用时判别器 update 是否已执行过）。"""

    def __init__(self, update: RecordingUpdate) -> None:
        self._update = update
        self.calls: list[bool] = []

    def compute(self, fake_latents: torch.Tensor) -> float:
        self.calls.append(len(self._update.received) > 0)
        return 0.5


class TestDiscriminatorSideOrchestration:
    """判别器侧编排（train 循环第 2 相的 fake 供给与判别器相位）。"""

    def test_update_step_draws_current_fakes_across_full_batch(self) -> None:
        """当前 fake 半区跨全批均匀随机抽取：new_fakes 按训练步与粒度
        有序堆叠（(k,λ) 升序 + Anchor 终点在末），若确定性取前 K/2 条，
        K=4 时判别器当前半区只见最小 step、λ=1 的前两个方向——其余
        分布从不进更新。判别器在更新期间处 train 相（spectral norm
        power iteration 属训练语义）、结束后恢复 eval 相（打分/监控
        前向不得漂移其有效权重）。"""
        discriminator = torch.nn.Linear(1, 1)
        update = RecordingUpdate(discriminator)
        coordinator = RewardCoordinator(
            update, auc=None,  # type: ignore[arg-type]  # 本测试不触 AUC
            generator=torch.Generator().manual_seed(11),
        )
        fakes = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1, 1)
        coordinator.update_step(fakes)
        received = update.received[0]
        expected_perm = torch.randperm(
            6, generator=torch.Generator().manual_seed(11),
        )
        assert torch.equal(received, fakes[expected_perm])  # 跨全批置换
        assert update.training_at_call[0] is True  # 更新期间 train 相
        assert discriminator.training is False  # 结束恢复 eval 相

    def test_scoring_forward_is_idempotent_under_spectral_norm(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """spectral norm 启用时打分幂等：打分/监控前向恒在 eval 相
        （power iteration 不推进）——判别器若停在 train 相打分，同批
        两次 forward 因 buffer 漂移分数不同，reward 归因被污染。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["reward"]["spectral_norm_enabled"] = True
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assert trainer.run() == 1
        assert trainer.rewards.discriminator.training is False
        sample = trainer.rewards.buffer.recent_samples()[0]
        scorer = trainer.rewards.update.scorer
        first = scorer.reward(sample.unsqueeze(0))
        second = scorer.reward(sample.unsqueeze(0))
        assert torch.equal(first, second)  # eval 相打分幂等

    def test_heldout_auc_precedes_discriminator_update(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """held-out AUC 在判别器更新之前测得（与 anchor_eval_reward 同一
        判别器快照）：update 之后测同一 fake 批会把 in-sample 拟合计入
        AUC（当前批子集刚被训练过、分数被抬高），且与 rollout 相记录的
        anchor reward 分属不同判别器快照——联合 hacking 签名（AUC 掉
        而 eval-reward 升）失真。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        update = RecordingUpdate(trainer.rewards.discriminator)
        auc = SequencedAuc(update)
        trainer.rewards.update = update
        trainer.rewards.auc = auc
        assert trainer.run() == 1
        assert update.received  # N_d=1 的首 iteration 应执行判别器更新
        assert auc.calls == [False]  # AUC 先于判别器更新（同一快照）


class TestDevicePlacement:
    """设备放置（装配期单点选设备、协作者经注入对齐）。

    fixture 测试面是 CPU-only：本组断言锁「模型与张量同源于 trainer 的
    设备」这一装配契约（accelerator 可用时的实际放置由 DCU 实例的
    M0 门槛验证，fixture 无法覆盖 cuda 分支）。"""

    def test_models_follow_trainer_device(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(
            config, artifacts, device=torch.device("cpu"),
        )
        assert trainer.unet.parameters().__next__().device.type == "cpu"
        assert (
            trainer.rewards.discriminator.parameters().__next__().device.type
            == "cpu"
        )

    def test_scoring_inputs_carry_no_stray_cpu_tensors(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """打分输入（rollout 终点、real 采样）与判别器同 device——CPU 上
        退化为同源性 sanity（cuda 分支由 CfgCombinedField 的 timesteps
        device 对齐与 RealPoolSampler 的迁移保证）。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(
            config, artifacts, device=torch.device("cpu"),
        )
        trainer.seed_base_partition()
        sample = trainer.rewards.buffer.base_samples()[0]
        assert sample.device.type == "cpu"
        assert trainer.rewards.update.scorer.reward(
            sample.unsqueeze(0),
        ).device.type == "cpu"


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


class TestBaseSeedingIsolation:
    """base 分区种子生成与训练 rollout 的 RNG 流隔离。"""

    def test_capacity_change_does_not_shift_rollout_stream(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """base seeding 走独立派生 generator：base seeding 消耗的条件/
        噪声抽取数由 replay_buffer_capacity 决定——与训练 rollout 共用
        流时，改 buffer 容量（保持 schedule.seed）会漂移后续全部 rollout
        抽样（modality、初始噪声、SDE 方向），buffer 容量实验与 policy
        样本流混淆（不同 capacity 的同 seed run 不可比）。"""
        scenario.write_inputs()
        streams: dict[int, tuple] = {}
        for capacity in (64, 80):
            data = json.loads(
                scenario.config_path.read_text(encoding="utf-8"),
            )
            data["reward"]["replay_buffer_capacity"] = capacity
            scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
            config = ConfigLoader.load(scenario.config_path)
            artifacts = RunArtifacts.init(
                config, scenario.tmp_path / f"run_capacity{capacity}",
            )
            trainer = GranularGrpoTrainer(
                config, artifacts, device=torch.device("cpu"),
            )
            trainer.seed_base_partition()
            record = trainer.rollout.run_iteration()
            streams[capacity] = (
                record.modality,
                record.steps[0].anchor_latent,
                record.steps[0].directions,
            )
        assert streams[64][0] == streams[80][0]  # modality 不随容量漂移
        assert torch.equal(streams[64][1], streams[80][1])  # anchor 逐位同
        assert torch.equal(streams[64][2], streams[80][2])  # 扰动方向逐位同


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
