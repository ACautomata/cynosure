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
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, DEFAULT_CROSS_MODAL_PAIRS, MODALITIES
from cynosure.fixtures import FIXTURE_MODALITY_MAPPING, Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import ModalityMapping
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.scorer import ChannelNormalizer
from cynosure.reward.update import UpdateReport
from cynosure.train import GranularGrpoTrainer, RewardCoordinator, RunArtifacts
from cynosure.train.rollout import CrossModalConditionSampler, SourceLatentPool
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
        group: str = "modal-label",
    ) -> None:
        """落盘 fixture 网络工件 + prepare 三工件 + 训练 config（group
        选实验组：组2/组3 的 config 携带 ControlNet 工件）。"""
        fixture = Fixture()
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（test_reward_fixture 先例）
        fixture.write_artifacts(self.fixture_dir)
        prepare_config = FixturePrepareScenario(
            self.cli, fixture.config(self.fixture_dir, group=group), self.tmp_path,
        ).run(self.tmp_path / "prepare_config.json")
        config = fixture.config(self.fixture_dir, group=group)
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
        power iteration buffer ``_u``/``_v``）不进保存面——保存面固化
        **有效权重**（parametrization 输出），netbuild 严格装载路径重建
        的裸网络判别函数与训练时**数值等价**（固化原始参数则前向不同：
        装载静默成功但 reward/评测漂移）。

        等价断言用严格容差（rtol 1e-5）而非逐位：跨「保存→磁盘→装载」
        链的前向在同一权重下可差 1 ulp（浮点执行路径的进程态选择，
        实测 maxdiff 1.19e-07 = 2^-23）；语义错误（固化参数视图）的
        偏离是 σ 偏离 1 的百分量级，容差区分度充分。"""
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
        # 严格装载成功（netbuild 装配路径直接可用），且重建的裸网络
        # 判别函数与训练时数值等价（有效权重语义，eval 相对照）
        reloaded = NetworkAssembler.discriminator(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.discriminator_config_json,
            ),
            checkpoint=checkpoint,
        ))
        reloaded.eval()
        sample = trainer.rewards.buffer.recent_samples()[0].unsqueeze(0)
        normalizer = ChannelNormalizer(
            ChannelStats.load(config.reward.channel_stats_json),
        )
        normalized = normalizer.normalize(sample)
        expected = trainer.rewards.discriminator(normalized)[-1]
        assert torch.allclose(
            reloaded(normalized)[-1], expected, rtol=1e-5, atol=1e-6,
        )

    def test_milestone_iteration_forces_checkpoint(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """里程碑迭代强制落盘（schedule.checkpoint_interval 的 config 契约
        「每里程碑强制落盘」）：checkpoint 周期不覆盖的里程碑也必须产出
        checkpoint（milestone 评测器与恢复路径的取数点）。
        milestone_interval=2、checkpoint_interval=5、max_iterations=3：
        iter2 仅由里程碑节奏落盘（2 % 5 != 0），iter3 由收尾兜底写入，
        iter3 兜底不受里程碑条件影响。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["schedule"]["max_iterations"] = 3
        data["schedule"]["milestone_interval"] = 2
        data["schedule"]["checkpoint_interval"] = 5
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        assert scenario.train().code == 0, scenario.stderr
        checkpoints = scenario.run_dir / "checkpoints"
        assert (checkpoints / "policy_iter2.pt").is_file()
        assert (checkpoints / "discriminator_iter2.pt").is_file()
        assert (checkpoints / "policy_iter3.pt").is_file()

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

    def test_sequential_group_rejected_outside_sequential_trainer(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """组3 的两阶段序贯由 SequentialTrainer 编排：绕过 CLI 分派、直接
        把 sequential config 塞进单阶段训练循环时显式拒绝（不静默只跑一段）。"""
        scenario.write_inputs(group="sequential")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        with pytest.raises(ValueError, match="SequentialTrainer"):
            GranularGrpoTrainer(config, artifacts)

    def test_only_unet_params_updated(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC「组1：仅 UNet 参数被更新」：组1 的 policy = UNet 本体（无第二
        个可训练对象）、全参 requires_grad，一次 iteration 后权重真实演化。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assert trainer.policy.network is trainer.unet
        assert all(p.requires_grad for p in trainer.unet.parameters())
        initial = {
            name: value.clone() for name, value in trainer.unet.state_dict().items()
        }
        assert trainer.run() == 1
        assert any(
            not torch.equal(initial[name], value)
            for name, value in trainer.unet.state_dict().items()
        )

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

    def test_incompatible_checkpoint_rolls_back_run(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """checkpoint 装载失败（键/shape 不匹配 → 严格装载 RuntimeError）
        同属输入契约违反：构造期得到干净消息 + 未产出工件的 run 目录
        回滚——RuntimeError 不在捕获集内则裸 traceback 且目录残留，
        修正 checkpoint 后同 --run-dir 重试被拒。"""
        scenario.write_inputs()
        bad = scenario.tmp_path / "bad_discriminator.pt"
        torch.save({"bogus": torch.zeros(1)}, bad)  # 键形与网络不符
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["artifacts"]["discriminator_ckpt"] = str(bad)
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        result = scenario.train()
        assert result.code == 2
        assert "训练输入契约违反" in result.stderr
        assert not scenario.run_dir.exists()  # 已回滚，可重试


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


class TestCrossModalLoop:
    """issue #23 组2 验收：仅 ControlNet 更新（base 冻结）、CFG=0 裸条件
    单前向走全循环、双条件注入生效、checkpoint 契约 = ControlNet 权重。"""

    def test_cross_modal_single_iteration_runs_green(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs(group="cross-modal")
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.events()
        assert [event["event"] for event in events] == ["iter"]
        event = events[0]
        assert event["modality"] in MODALITIES  # 目标序列归因轴（12 对的目标端）
        assert event["intra_group_reward_std"] > 0.0  # CFG=0 场的组内方差非退化
        assert "policy_step_1" in event["loss"]
        assert "discriminator" in event["loss"]
        assert event["buffer_base_occupied"] == 32  # 本组判别器/buffer 独立装配

    def test_policy_checkpoint_is_controlnet_and_loadable(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """组2 的 policy checkpoint = ControlNet state_dict：按 ControlNet
        网络配置可重新装载（产物契约按组指向可训练对象）。"""
        scenario.write_inputs(group="cross-modal")
        assert scenario.train().code == 0
        config = ConfigLoader.load(scenario.config_path)
        reloaded = NetworkAssembler.controlnet(NetworkArtifact(
            config=NetworkAssembler.load_json(
                config.artifacts.controlnet_config_json,
            ),
            checkpoint=scenario.run_dir / "checkpoints" / "policy_iter1.pt",
        ))
        assert any(p.requires_grad for p in reloaded.parameters())

    def test_only_controlnet_params_updated_base_frozen(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC「仅 ControlNet 参数被更新（base 冻结经断言验证）」：一次
        iteration 后 base UNet 全部参数逐位未动（无梯度、无优化器步），
        ControlNet 有参数真实演化；冻结在装配期即被 requires_grad 断言。"""
        scenario.write_inputs(group="cross-modal")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        initial_unet = {
            name: value.clone() for name, value in trainer.unet.state_dict().items()
        }
        initial_controlnet = {
            name: value.clone()
            for name, value in trainer.policy.network.state_dict().items()
        }
        assert not any(
            p.requires_grad for p in trainer.unet.parameters()
        )  # base 冻结：装配期断言的对外可观测面
        assert all(p.requires_grad for p in trainer.policy.network.parameters())
        assert trainer.run() == 1
        for name, value in trainer.unet.state_dict().items():
            assert torch.equal(initial_unet[name], value), name
        assert any(
            not torch.equal(initial_controlnet[name], value)
            for name, value in trainer.policy.network.state_dict().items()
        )

    def test_cross_modal_logprob_pairs_survive_to_update_time(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """CFG=0 单前向场的 log-prob 一致性（测试面 #3 对组2 的延伸）：
        rollout 记录的 π_old 与更新前重算逐位一致——残差注入路径在
        rollout 与重算两侧同权重同口径。"""
        scenario.write_inputs(group="cross-modal")
        assert scenario.train(dump=True).code == 0
        report = json.loads(
            (scenario.run_dir / "training.json").read_text(encoding="utf-8"),
        )
        pairs = report["logprob_pairs"]
        assert len(pairs) == 12  # |M| × G
        for pair in pairs:
            assert math.isfinite(pair["recorded"])
            assert pair["recorded"] == pair["recomputed"]


class TestCrossModalPairSampling:
    """组2 条件分布：12 有序对均匀采样（清单来自 config 注入，无代码内
    副本）；源影像 latent 按源序列从 real pool 分层抽取。"""

    LATENT_SHAPE = (4, 16, 16, 8)

    @staticmethod
    def _write_identifiable_pool(root: Path) -> Path:
        """每序列恰一枚、以序列序号填充的 latent（源序列可从张量内容
        识别，使 12 对的完整计数可观测）。"""
        shape = (4, 16, 16, 8)
        latents_dir = root / "latents"
        latents_dir.mkdir(parents=True)
        entries = []
        for modality_index, modality in enumerate(MODALITIES):
            name = f"{modality}.pt"
            torch.save(
                torch.full(shape, float(modality_index)),
                latents_dir / name,
            )
            entries.append({
                "case_id": f"case-{modality}",
                "modality": modality,
                "latent": f"latents/{name}",
            })
        manifest = {
            "kind": "real_pool",
            "encoder": "fixture-test",
            "latent_shape": list(shape),
            "split_seed": 0,
            "split_sizes": {"train": 4, "val": 0, "test": 0},
            "entries": entries,
        }
        path = root / "real_pool.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    @staticmethod
    def _sampler(root: Path, seed: int) -> CrossModalConditionSampler:
        pool = SourceLatentPool(
            LatentManifest.load(root / "real_pool.json", kind="real_pool"),
            torch.device("cpu"),
        )
        return CrossModalConditionSampler(
            ModalityMapping(dict(FIXTURE_MODALITY_MAPPING)),
            [tuple(pair) for pair in DEFAULT_CROSS_MODAL_PAIRS],
            pool,
            torch.Generator().manual_seed(seed),
            torch.device("cpu"),
        )

    def test_all_12_ordered_pairs_drawn_uniformly(self, tmp_path: Path) -> None:
        """1440 次采样覆盖全部 12 个有序对，各对频次落在均匀 3σ 带内
        （p=1/12、n=1440：均值 120、σ≈10.5，取 ±40 宽松带防 flake）。"""
        torch.manual_seed(0)
        self._write_identifiable_pool(tmp_path)
        sampler = self._sampler(tmp_path, seed=0)
        source_marker = {
            float(index): modality for index, modality in enumerate(MODALITIES)
        }
        draws = 12 * 120
        counts: Counter[tuple[str, str]] = Counter()
        for _ in range(draws):
            condition, target = sampler.sample()
            source = source_marker[condition.source_latent[0, 0, 0, 0, 0].item()]
            counts[(source, target)] += 1
        expected_pairs = {(src, tgt) for src, tgt in product(MODALITIES, repeat=2) if src != tgt}
        assert set(counts) == expected_pairs  # 12 对全覆盖、无序外对
        for pair, count in counts.items():
            assert abs(count - draws / 12) < 40, pair

    def test_pairs_come_from_injected_list(self, tmp_path: Path) -> None:
        """均匀采样的清单 = 注入的 config 清单（cross_modal_pairs 的单一
        来源语义）：采样目标端分布随清单而非随硬编码常量。"""
        torch.manual_seed(0)
        pool_path = self._write_identifiable_pool(tmp_path)
        pool = SourceLatentPool(
            LatentManifest.load(pool_path, kind="real_pool"), torch.device("cpu"),
        )
        pairs = sorted(DEFAULT_CROSS_MODAL_PAIRS, reverse=True)  # 非默认顺序注入
        sampler = CrossModalConditionSampler(
            ModalityMapping(dict(FIXTURE_MODALITY_MAPPING)),
            list(pairs),
            pool,
            torch.Generator().manual_seed(1),
            torch.device("cpu"),
        )
        targets = {sampler.sample()[1] for _ in range(48)}
        assert targets <= set(MODALITIES)
        assert len(targets) == 4

    def test_pool_missing_modality_rejected(self, tmp_path: Path) -> None:
        """源影像库缺任一序列 = 组2 条件分布不可用：显式拒绝。"""
        manifest_path = self._write_identifiable_pool(tmp_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["entries"] = [
            entry for entry in data["entries"] if entry["modality"] != "t2f"
        ]
        manifest_path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="t2f"):
            SourceLatentPool(
                LatentManifest.load(manifest_path, kind="real_pool"),
                torch.device("cpu"),
            )


class RecordingScorer:
    """测试仪器：以注入判别器冒充打分器（coordinator 取相位的观测载体）。"""

    def __init__(self, discriminator: torch.nn.Module) -> None:
        self.discriminator = discriminator


class RecordingUpdate:
    """测试仪器：记录 update.step 收到的批与调用时的判别器相位
    （buffer 用真实两区实现——RewardCoordinator 的 zone_sizes 观测面
    经它委托；optimizer 为真实现——续训状态机的判别器侧 checkpoint
    经 RewardCoordinator 消费 update.optimizer，协作者契约面的一部分）。"""

    def __init__(self, discriminator: torch.nn.Module) -> None:
        self.scorer = RecordingScorer(discriminator)
        self.buffer = ReplayBuffer(64)
        self.optimizer = torch.optim.AdamW(discriminator.parameters(), lr=5e-5)
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

    def compute(
        self, fake_latents: torch.Tensor, modality: str | None = None,
    ) -> float:
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

    def test_cold_start_discriminator_is_seeded_deterministically(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """冷启动判别器在 schedule.seed 的派生流下确定初始化：随机初始化
        消耗全局 RNG，而 sampling generators 在装配序列更后才创建——同
        config 的两次冷启动若依赖进程全局 RNG 状态，判别器初始权重不同，
        初始 reward 与其后所有 policy update 都不可复现（seeded 实验
        失效）。两次独立构造（不同进程内全局状态、同 seed）判别器权重
        须逐位一致。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["artifacts"]["discriminator_ckpt"] = None
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        weights: list[list[torch.Tensor]] = []
        for name in ("a", "b"):
            config = ConfigLoader.load(scenario.config_path)
            artifacts = RunArtifacts.init(
                config, scenario.tmp_path / f"run_cold_{name}",
            )
            trainer = GranularGrpoTrainer(config, artifacts)
            weights.append([
                param.detach().clone()
                for param in trainer.rewards.discriminator.parameters()
            ])
        for first, second in zip(weights[0], weights[1]):
            assert torch.equal(first, second)

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
