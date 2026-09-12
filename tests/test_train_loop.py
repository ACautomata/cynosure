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
import shutil
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, DEFAULT_CROSS_MODAL_PAIRS, MODALITIES
from cynosure.fixtures import FIXTURE_MODALITY_MAPPING, Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import (
    CONDITION_SPACING_X1E2,
    ModalityMapping,
)
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.sampler import RolloutSampler
from cynosure.reward.artifacts import ChannelStats, LatentManifest
from cynosure.reward.buffer import ReplayBuffer, ReplayEntry, base_condition_quota
from cynosure.reward.scorer import ChannelNormalizer
from cynosure.reward.update import UpdateReport
from cynosure.train import GranularGrpoTrainer, RewardCoordinator, RunArtifacts
from cynosure.train.rollout import (
    CrossModalConditionSampler,
    ModalLabelConditionSampler,
    SourceLatentPool,
)
from tests.conftest import (
    CliResult,
    CliSession,
    FixturePrepareScenario,
    PretrainLightweightReward,
)


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
        reward: dict | None = None,
    ) -> None:
        """落盘 fixture 网络工件 + prepare 三工件 + 训练 config（group
        选实验组：组2/组3 的 config 携带 ControlNet 工件）。

        warm-start 前置（ADR-0007）：RM readiness gate 是 train 入口的
        硬检查、消费预训练产物——场景先以同一 config 的预训练轻量变体
        跑出报告与 checkpoint（fixture 低阈值 gate，Fixture.config），
        再落训练 config。``reward`` 覆写在预训练前置**之前**生效——
        预训练与训练同一 reward regime（如 SN 启用时预训练产物即
        谱归一化形态，warm-start 装载走形态分派的逐位还原路径）。"""
        fixture = Fixture()
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（test_reward_fixture 先例）
        fixture.write_artifacts(self.fixture_dir)
        # 场景工具的幂等重建（write_inputs 可重复调用——预训练 run 目录
        # 由本步重建，不静默覆盖语义是 CLI 的、工具面先清后建）
        shutil.rmtree(self.fixture_dir / "pretrain_run", ignore_errors=True)
        prepare_config = FixturePrepareScenario(
            self.cli, fixture.config(self.fixture_dir, group=group), self.tmp_path,
        ).run(self.tmp_path / "prepare_config.json")
        config = fixture.config(self.fixture_dir, group=group)
        config.policy.num_inference_steps = num_steps
        config.policy.train_step_indices_m = set(train_steps)
        config.schedule.seed = seed
        config.schedule.max_iterations = 1  # tracer bullet：单 iteration 全链路
        if reward:
            config.reward = config.reward.model_copy(update=reward)
        self._pretrain_warm_start(config, group)
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )

    def _pretrain_warm_start(self, config, group: str) -> None:
        """场景的预训练前置：报告落 config 声明的产物路径（train 装配
        与门槛检查的装载源）。轻量五元组（``PretrainLightweightReward``，
        含 gate 0.60 留 margin 的 rationale）只降低本步执行成本、不进训练
        config；组3 的预训练走 stage-1 的组1 形态（GroupPolicy 拒绝
        sequential 组的单次装配）。"""
        pretrain_config = PretrainLightweightReward.apply(config)
        pretrain_config.experiment.group = (
            "modal-label" if group == "sequential" else group
        )
        path = self.tmp_path / "pretrain_config.json"
        path.write_text(
            pretrain_config.model_dump_json(indent=2), encoding="utf-8",
        )
        result = self.cli.run("pretrain", "--config", str(path))
        assert result.code == 0, result.stderr

    def train(self, *, dump: bool = False):
        argv = ["train", "--config", str(self.config_path), "--run-dir", str(self.run_dir)]
        if dump:
            argv.append("--dump-trajectory")
        return self.cli.run(*argv)

    def set_schedule(self, **values) -> None:
        """改写 config 的 schedule 字段并重落盘（里程碑/早停参数变体）。"""
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        data["schedule"].update(values)
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

    def standalone_sampler(
        self, config, device: torch.device | None = None,
    ) -> RolloutSampler:
        """评测相注入测试用的独立采样封装（与 TrainingRuntime.assemble_sampler
        同一组合方式；独立于 trainer 内部装配）。网络落 ``device``
        （缺省 CPU；设备归一测试传加速器设备）。"""
        unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=config.artifacts.unet_ckpt,
        )).to(device if device is not None else torch.device("cpu"))
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=config.policy.num_inference_steps,
            input_img_size_numel=config.policy.input_img_size_numel,
        )
        return RolloutSampler(
            CfgCombinedField(unet),
            SdeKernel(eta=config.policy.sde_eta, s_max=config.policy.sde_s_max),
            TrajectoryCursor(scheduler),
        )

    def artifacts(self) -> RunArtifacts:
        return RunArtifacts(RunArtifacts.layout(self.run_dir))

    def events(self) -> list[dict]:
        return self.artifacts().read_events()

    def patch_config(self, **sections: dict) -> None:
        """按 section 覆写已写出的训练 config（JSON 补丁，写回原路径）。"""
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        for section, values in sections.items():
            data[section].update(values)
        self.config_path.write_text(json.dumps(data), encoding="utf-8")

    def resume(self, *, dump: bool = False) -> CliResult:
        """--resume 入口：同 run 目录的续训提交（跨作业边界的恢复场景）。"""
        argv = [
            "train", "--config", str(self.config_path),
            "--run-dir", str(self.run_dir), "--resume",
        ]
        if dump:
            argv.append("--dump-trajectory")
        return self.cli.run(*argv)

    def resume_state(self) -> dict:
        """单进程续训状态分片的外部读取面（契约文件名字面：world-1 无
        rank 后缀；多 rank 分片对账见 test_distributed.RankResumeShards）。"""
        return torch.load(
            self.run_dir / "checkpoints" / "resume_state.pt",
            map_location="cpu", weights_only=True,
        )

    def checkpoints_identical(self, other_run_dir: Path, names: list[str]) -> None:
        """收官 checkpoint 工件与另一 run 的同名 state_dict 逐位对账
        （同路径重放的逐位语义）。"""
        for name in names:
            first = torch.load(
                self.run_dir / "checkpoints" / name,
                map_location="cpu", weights_only=True,
            )
            second = torch.load(
                other_run_dir / "checkpoints" / name,
                map_location="cpu", weights_only=True,
            )
            assert set(first) == set(second)
            for key in first:
                assert torch.equal(first[key], second[key]), f"{name}:{key}"


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
        result = scenario.train()
        assert result.code == 0, result.stderr
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
        """spectral norm 启用时判别器 checkpoint 以**参数化状态**落盘：
        parametrization 键（``*.parametrizations.<attr>.original`` 与其
        power iteration buffer ``_u``/``_v``）整份在内。装载面按形态分派
        （先叠谱归一化 → 严格装载整份还原）：重建的判别器状态与训练时
        **逐位一致**，且结果与 ambient RNG 无关。

        对照（固化有效权重的形态）：装载时重新叠谱归一化会**再归一化
        一次**（随机 u/v 起步 + 15 次幂迭代）——前向随 ambient RNG 漂移，
        消费面拿到的不再是训练时那一份判别函数。

        执行 device 跟随训练装配（GPU 可见即加速器、CPU 强制即 CPU）：
        手动构造的 normalizer 与装载实例按被测 latent 的 device 落位
        （生产装配由 trainer 单点 ``.to()`` 接管的同款语义，#95）；
        state_dict 逐位对比在 CPU 侧先做，前向对比在迁移后做。"""
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
        live = trainer.rewards.discriminator
        # 保存面键形 = 训练态参数化状态（不是裸网络的物化有效权重）
        assert set(saved.keys()) == set(live.state_dict().keys())
        assert any(".parametrizations." in key for key in saved)
        live.eval()
        sample = trainer.rewards.buffer.recent_samples()[0].latent.unsqueeze(0)
        device = sample.device  # 训练装配设备（GPU 可见即加速器、CPU 强制即 cpu）
        # 测试构造按被测 latent 的 device 落位（#95）：normalize 的 device
        # fail-fast 契约要求统计量 buffer 与输入同源——手动构造的
        # normalizer 与生产装配一样迁移到 latent 所在 device，GPU 可见环境
        # 不再被契约拦截在真正的断言层之前。
        normalizer = ChannelNormalizer(
            ChannelStats.load(config.reward.channel_stats_json),
        ).to(device)
        normalized = normalizer.normalize(sample)
        expected = live(normalized)[-1]
        for seed in (11, 20260910):  # ambient seed 不同：装载结果不得依赖它
            torch.manual_seed(seed)
            reloaded = NetworkAssembler.discriminator(
                NetworkArtifact(
                    config=NetworkAssembler.load_json(
                        config.artifacts.discriminator_config_json,
                    ),
                    checkpoint=checkpoint,
                ),
                spectral_norm=True,
            )
            restored = reloaded.state_dict()
            assert restored.keys() == saved.keys()
            # 逐位对比在 CPU 侧先做（saved 以 map_location="cpu" 读入，
            # torch.equal 跨 device 拒绝；逐位还原语义与执行 device 无关）
            assert all(torch.equal(restored[key], saved[key]) for key in saved)
            reloaded = reloaded.to(device).eval()  # 前向对比与被测 latent 同源
            # 前向层用绝对容差而非逐位：判别力由上面的 state 逐位对比承担
            # （形态语义所在），前向在**同权重、不同实例**间可差 1 ulp
            # （2^-23 ≈ 1.2e-7，浮点执行路径的分配/分块选择，实测偶发），
            # 逐位断言会假红；1e-6 在噪声底之上、修复前形态的偏离之下。
            assert torch.allclose(
                reloaded(normalized)[-1], expected, rtol=0.0, atol=1e-6,
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
        result = scenario.train()
        assert result.code == 0, result.stderr
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
        """discriminator_ckpt=None 的工件面（随机初始化起步）：预训练
        侧冷启动（ADR-0007 的 warm-start 产物生产方）不受影响，train
        侧判别器一律从预训练报告守卫重载——``discriminator_ckpt`` 在
        train 装配中不再是消费点（warm-start 接入后废弃冷启动训练）。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["artifacts"]["discriminator_ckpt"] = None
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        result = scenario.train()
        assert result.code == 0, result.stderr
        assert (scenario.run_dir / "checkpoints" / "discriminator_iter1.pt").is_file()

    def test_replay_capacity_guard_rejects_undersized_combinations(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """回放供给装配期守卫（ADR-0008 决策 4，fail-fast）：首次判别器
        更新时近期分区为空、回放半区全由 base 分区按条件承担——
        ``每条件配额 < 回放半区需求`` 的组合（如 K=4/capacity=2，配额
        0 < 2）在昂贵 rollout 完成后才会缺样本炸掉；K=1 则回放半区为
        0 条、回放采样 API 直接拒绝。两类 schema 合法但集成无效的组合
        在装配期显式拒绝。"""
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
        """warm-start 产物 checkpoint 装载失败（键/shape 不匹配 → 严格
        装载 RuntimeError）同属输入契约违反：构造期得到干净消息 + 未
        产出工件的 run 目录回滚——修正预训练产物后同 --run-dir 重试
        不被残留目录拒绝。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        checkpoint = (
            Path(data["reward"]["pretrain_report_json"]).parent
            / "checkpoints" / "pretrain_discriminator.pt"
        )
        torch.save({"bogus": torch.zeros(1)}, checkpoint)  # 键形与网络不符
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
        (group,) = trainer.loop.updater.optimizer.param_groups
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

    # 各序列条目的 spacing 侧车值各不相同（issue #46）：消费端接线可从
    # 条件 spacing 反查源条目（float32 精确值，断言可精确相等）
    SPACING_BY_MODALITY: dict[str, tuple[float, float, float]] = {
        "t1n": (50.0, 100.0, 200.0),
        "t1c": (110.0, 120.0, 130.0),
        "t2w": (140.0, 150.0, 160.0),
        "t2f": (70.0, 80.0, 90.0),
    }

    @staticmethod
    def _write_identifiable_pool(root: Path) -> Path:
        """每序列恰一枚、以序列序号填充的 latent（源序列可从张量内容
        识别，使 12 对的完整计数可观测）；spacing 侧车按序列取不同值
        （issue #46 消费端接线的观测面）。"""
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
                "spacing": list(
                    TestCrossModalPairSampling.SPACING_BY_MODALITY[modality]
                ),
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

    def test_condition_spacing_comes_from_source_entry(
        self, tmp_path: Path,
    ) -> None:
        """组2 消费端接线（issue #46）：条件的 spacing tensor = 源影像条目
        的 manifest 侧车值（per-case 来自数据，替换写死常量）。"""
        torch.manual_seed(0)
        self._write_identifiable_pool(tmp_path)
        sampler = self._sampler(tmp_path, seed=0)
        source_marker = {
            float(index): modality for index, modality in enumerate(MODALITIES)
        }
        seen: set[str] = set()
        for _ in range(48):
            condition, _ = sampler.sample()
            source = source_marker[condition.source_latent[0, 0, 0, 0, 0].item()]
            seen.add(source)
            assert tuple(condition.spacing[0].tolist()) == (
                self.SPACING_BY_MODALITY[source]
            )
        assert seen == set(MODALITIES)  # 各源序列的接线都被真实走到

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

    def test_sample_target_draws_source_for_fixed_target(self, tmp_path: Path) -> None:
        """ADR-0008-01：base 配额量产的条件源——目标端固定为指定模态
        （label 恒为 target），源序列自由度仍按组2 分布在合法源上
        均匀抽取（12 对中目标端为 target 的 3 个源全覆盖）。"""
        torch.manual_seed(0)
        self._write_identifiable_pool(tmp_path)
        sampler = self._sampler(tmp_path, seed=0)
        source_marker = {
            float(index): modality for index, modality in enumerate(MODALITIES)
        }
        targets = {modality: set() for modality in MODALITIES}
        for target in MODALITIES:
            for _ in range(60):
                condition = sampler.sample_target(target)
                label = int(condition.label[0].item())
                mapped = {
                    label_value: name
                    for name, label_value in FIXTURE_MODALITY_MAPPING.items()
                }[label]
                source = source_marker[condition.source_latent[0, 0, 0, 0, 0].item()]
                assert mapped == target  # 目标端恒为指定模态
                assert source != target  # 有序对不自配对
                targets[target].add(source)
        for target in MODALITIES:
            # 目标端为 target 的 3 个源序列都被真实走到
            assert targets[target] == set(MODALITIES) - {target}

    def test_sample_target_rejects_unknown_target(self, tmp_path: Path) -> None:
        """注入清单无目标端为 target 的有序对（cross_modal_pairs 可配置）：
        显式拒绝而非静默回退全目标采样。"""
        torch.manual_seed(0)
        pool_path = self._write_identifiable_pool(tmp_path)
        pool = SourceLatentPool(
            LatentManifest.load(pool_path, kind="real_pool"), torch.device("cpu"),
        )
        pairs = [("t1n", "t1c"), ("t1c", "t1n")]  # 仅两对，t2w/t2f 不作目标
        sampler = CrossModalConditionSampler(
            ModalityMapping(dict(FIXTURE_MODALITY_MAPPING)),
            pairs,
            pool,
            torch.Generator().manual_seed(1),
            torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="t2w"):
            sampler.sample_target("t2w")


class TestModalLabelTargetSampling:
    """组1 条件分布的 sample_target（ADR-0008-01：base 配额量产的条件源）。"""

    def test_sample_target_binds_label_to_target(self) -> None:
        """label 条件不耗 RNG（label 由 target 决定）：给定目标序列
        构造的条件 label 恒为映射值、spacing 为单位间距常量。"""
        mapping = ModalityMapping(dict(FIXTURE_MODALITY_MAPPING))
        device = torch.device("cpu")
        sampler = ModalLabelConditionSampler(
            mapping, torch.Generator().manual_seed(0), device,
        )
        for target in MODALITIES:
            condition = sampler.sample_target(target)
            assert int(condition.label[0].item()) == FIXTURE_MODALITY_MAPPING[target]
            assert tuple(condition.spacing[0].tolist()) == CONDITION_SPACING_X1E2

    def test_sample_target_ignores_stream(self) -> None:
        """组1 的 sample_target 不消耗传入流：base 量产的 RNG 消耗
        全部来自初始噪声（条件无随机自由度）。"""
        mapping = ModalityMapping(dict(FIXTURE_MODALITY_MAPPING))
        sampler = ModalLabelConditionSampler(
            mapping, torch.Generator().manual_seed(0), torch.device("cpu"),
        )
        stream = torch.Generator().manual_seed(3)
        before = stream.get_state()
        sampler.sample_target("t2w", stream)
        assert torch.equal(before, stream.get_state())


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

    def step(
        self, current_fakes: torch.Tensor, modality: str,
    ) -> UpdateReport:
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
        coordinator.update_step(fakes, "t2w")
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
        两次 forward 因 buffer 漂移分数不同，reward 归因被污染。

        断言用绝对容差而非逐位（#95，与 checkpoint 测试同款 rationale）：
        幂等防的是 power iteration 推进，偏离量级远大于 ulp；而同权重
        两次前向可差 1 ulp（浮点执行路径固有噪声底，加速器上实测偶发），
        逐位断言假红；1e-6 在噪声底之上、真实泄漏的偏离之下。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["reward"]["spectral_norm_enabled"] = True
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assert trainer.run() == 1
        assert trainer.rewards.discriminator.training is False
        sample = trainer.rewards.buffer.recent_samples()[0].latent
        scorer = trainer.rewards.update.scorer
        first = scorer.reward(sample.unsqueeze(0))
        second = scorer.reward(sample.unsqueeze(0))
        assert torch.allclose(first, second, rtol=0.0, atol=1e-6)  # eval 相打分幂等

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
        sample = trainer.rewards.buffer.base_samples()[0].latent
        assert sample.device.type == "cpu"
        assert trainer.rewards.update.scorer.reward(
            sample.unsqueeze(0),
        ).device.type == "cpu"


class TestBufferBaseSeeding:
    """AC 5 + ADR-0008-01：buffer base 分区在 train 启动时由冻结初始
    policy 按每条件配额自动生成，条目带目标模态标签。"""

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
        assert all(torch.isfinite(entry.latent).all() for entry in base)
        assert sizes.recent == 0  # base 生成不进近期分区

    def test_base_partition_seeded_by_per_condition_quota(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """ADR-0008-01 AC 3：base 分区按每条件配额量产——每目标模态
        的条目数恰为配额（回放条件过滤后每条件候选的供给根基）。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        trainer.seed_base_partition()
        quota = base_condition_quota(config.reward.replay_buffer_capacity)
        assert trainer.rewards.buffer.zone_modalities().base == quota

    def test_base_partition_entries_carry_target_labels(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """ADR-0008-01 AC 1：base 分区观测面（快照）带目标模态标签——
        每条目为 ReplayEntry，标签 ∈ MODALITIES、按配额分布。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        trainer.seed_base_partition()
        entries = trainer.rewards.buffer.base_samples()
        assert all(isinstance(entry, ReplayEntry) for entry in entries)
        assert all(entry.modality in MODALITIES for entry in entries)
        assert {
            modality for entry in entries for modality in [entry.modality]
        } == set(MODALITIES)

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
        噪声抽取数随 replay_buffer_capacity 决定——与训练 rollout 共用
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
            record = trainer.loop.run_iteration()
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


class TestRewardDomainNormalization:
    """rollout（policy 域）→ reward/replay（real pool 存储域）的归位
    （T12 探针定谳）：fake 在打分与入 buffer 前除 latent_scale_factor——
    real 按 data-preparation 契约存 encode 原始输出、policy 输出在
    checkpoint scaled 域，判别器比较要求两侧同域。"""

    @staticmethod
    def _rollout_with_scale(
        scenario: TrainingLoopScenario, scale: float, tag: str,
    ) -> tuple:
        # fixture 权重由 write_inputs 内的 manual_seed(7) 固定；两次构造的
        # rollout 流可比性来自 TrainingRngStreams 全显式 CPU generator——
        # scale 不进任何 RNG 消耗路径（reward 打分后置、无反馈分支）
        scenario.write_inputs()
        data = json.loads(
            scenario.config_path.read_text(encoding="utf-8"),
        )
        data["policy"]["latent_scale_factor"] = scale
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.tmp_path / f"run_{tag}")
        trainer = GranularGrpoTrainer(config, artifacts, device=torch.device("cpu"))
        scored: list[torch.Tensor] = []
        scorer = trainer.rewards.update.scorer
        original_reward = scorer.reward

        def recording_reward(latents: torch.Tensor) -> torch.Tensor:
            scored.append(latents)
            return original_reward(latents)

        scorer.reward = recording_reward  # 打分输入记录（实例属性遮蔽 bound method）
        trainer.seed_base_partition()
        record = trainer.loop.run_iteration()
        return record, scored

    def test_scored_fakes_match_replay_domain(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """打分输入与 new_fakes 同值同序（同一归一域）：打分、回放入区、
        AUC fake 侧消费同一批归一后 latent，不出现「打分用 A 域、
        回放用 B 域」的错配。"""
        record, scored = self._rollout_with_scale(scenario, 2.0, "replay")
        assert scored, "打分记录为空"
        assert torch.equal(torch.cat(scored), record.new_fakes)

    def test_scale_only_affects_reward_side_not_rollout(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """scale 只作用在 reward 侧归一：rollout 采样流（条件/噪声/方向）
        与 scale 无关——scale=2 的 new_fakes 恰为 scale=1 的 ÷2
        （policy 域产出同、归一除法真实发生在 fake 上）。"""
        neutral_record, _ = self._rollout_with_scale(scenario, 1.0, "neutral")
        scaled_record, _ = self._rollout_with_scale(scenario, 2.0, "scaled")
        assert neutral_record.modality == scaled_record.modality
        assert torch.equal(
            neutral_record.new_fakes, scaled_record.new_fakes * 2.0,
        )
