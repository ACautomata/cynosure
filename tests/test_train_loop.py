"""单 iteration GRPO 循环全链路（ticket #21 验收标准聚合，tracer bullet）。

fixture 下 CLI train 端到端：Rollout（Anchor → 单步 SDE 扰动 → 各 λ ODE
续跑 → 判别器 raw logit 打分）→ MGAI advantage → 逐 k 独立梯度步 →
判别器 Online update → iter 事件落盘 + checkpoint。

四条 AC 对应（原 AC 5「buffer base 分区自动生成」随 ADR-0012 量产退役
移除，#173）：
1. fixture 下单 iteration 全链路绿，产出可装载 checkpoint 与 iter 事件流；
2. log-prob 一致性：Rollout 记录的 π_old 与更新时重算一致（诊断工件）；
3. MGAI 顺序正确；G=12 下组内标准化非退化（组内 reward std 非零进事件）；
4. 每个训练步 k 一次独立梯度步（loss 组件逐 k 记录）；
"""

import copy
import json
import math
import shutil
from collections import Counter
from itertools import product
from pathlib import Path

import pytest
import torch

from cynosure.policy.sampler import (
    DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES,
)
from cynosure.policy.schedules import SingleConditionSchedules
from cynosure.config import (
    ConfigLoader,
    CynosureConfig,
    DEFAULT_CROSS_MODAL_PAIRS,
    MODALITIES,
    RewardConfig,
)
from cynosure.distributed import DistributedContext
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
from cynosure.reward.assembly import PairBatch
from cynosure.reward.overfit import OverfitMonitor
from cynosure.reward.scorer import ChannelNormalizer
from cynosure.train import (
    DynamicWhitelist,
    GranularGrpoTrainer,
    RewardCoordinator,
    RunArtifacts,
)
from cynosure.train.runtime import TrainingRuntime
from cynosure.train.whitelist import ConditionWhitelist
from cynosure.train.resume import RESUME_STATE_FORMAT_VERSION
from cynosure.train.rollout import (
    CrossModalConditionSampler,
    ModalLabelConditionSampler,
    SourceLatentPool,
)
from tests.conftest import (
    CliResult,
    CliSession,
    FixtureArtifactLibrary,
    FixturePrepareScenario,
    PretrainLightweightReward,
    RecordingScorer,
    RecordingUpdate,
)


class TestForwardActivationBudgetResolution:
    """前向激活预算的装配期解析（#123 首跑 OOM 修复的单一解析点）：
    config 显式值优先，缺省按设备总显存自动探测。"""

    def test_pinned_value_wins(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"forward_activation_budget_gib": 36}
        config = CynosureConfig.model_validate(data)
        assert TrainingRuntime.forward_activation_budget(config) == 36 * 2**30

    def test_default_falls_back_without_cuda_device(
        self, valid_config_dict: dict,
    ) -> None:
        """无 CUDA 设备（CPU fixture 口径）：回落默认常量，分块保护恒在。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        budget = TrainingRuntime.forward_activation_budget(config)
        assert budget == DEFAULT_FORWARD_ACTIVATION_BUDGET_BYTES


class TrainingLoopScenario:
    """一次 fixture 训练场景：网络工件 + prepare 数据工件 + CLI train。"""

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self.cli = cli
        self.tmp_path = tmp_path
        # 场景根目录自建：调用方可能传尚不存在的子目录（跨 run 对比场景的
        # tmp_path / f"run{N}"）。旧内联工件构建经 Fixture.write_artifacts
        # 的 mkdir(parents=True) 隐式保证；#99 工件库化后由本类显式接管。
        self.tmp_path.mkdir(parents=True, exist_ok=True)
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
        """落盘训练 config（group 选实验组：组2/组3 的 config 携带
        ControlNet 工件）。

        场景工件（fixture 网络工件 + prepare 三工件 + 预训练产物）由
        ``FixtureArtifactLibrary`` 按 (group, 日程, seed, reward 覆写)
        变体构建一次、进程内只读共享——本方法只剩 config 落盘（场景
        搭建成本从每测试一次降为每变体一次）。warm-start 前置（ADR-0007）
        由库承担：RM readiness gate 是 train 入口的硬检查、消费预训练
        产物；``reward`` 覆写进库键（在预训练之前生效）——预训练与训练
        同一 reward regime（如 SN 启用时预训练产物即谱归一化形态，
        warm-start 装载走形态分派的逐位还原路径）。工件对本场景只读；
        要篡改预训练产物的测试先 ``fork_pretrained_artifacts``。"""
        self.fixture_dir = FixtureArtifactLibrary.artifacts_dir(
            self.cli, group,
            num_steps=num_steps, train_steps=frozenset(train_steps),
            seed=seed, reward=reward,
        )
        config = Fixture().config(self.fixture_dir, group=group)
        config.policy.num_inference_steps = num_steps
        config.policy.train_step_indices_m = set(train_steps)
        config.schedule.seed = seed
        config.schedule.max_iterations = 1  # tracer bullet：单 iteration 全链路
        if reward:
            config.reward = config.reward.model_copy(update=reward)
        self.config_path.write_text(
            config.model_dump_json(indent=2), encoding="utf-8",
        )

    def fork_pretrained_artifacts(self) -> None:
        """把共享库的预训练产物目录拷贝为本场景私有，并把 config 的
        ``pretrain_report_json`` 改指私有副本（篡改预训练产物的测试的
        写前隔离：共享工件只读，直接写共享目录即跨测试污染）。
        checkpoint 在 report 内以相对路径引用（解析基准 = 报告自身
        位置），整目录拷贝后引用随之闭合。"""
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        shared_report = Path(data["reward"]["pretrain_report_json"])
        private_dir = self.tmp_path / "pretrain_run"
        shutil.copytree(shared_report.parent, private_dir)
        data["reward"]["pretrain_report_json"] = str(
            private_dir / shared_report.name,
        )
        self.config_path.write_text(
            json.dumps(data, indent=2), encoding="utf-8",
        )

    def narrow_whitelist(self, members: list) -> None:
        """fork 私有预训练产物并把白名单收窄为 ``members``（门控场景
        的条件构造，test_train_loop.TestGradientGating 与
        test_distributed.TestTwoRankGating 共用；报告的实测值与
        checkpoint 不动——warm-start 装载与守卫链不受影响）。"""
        self.fork_pretrained_artifacts()
        data = json.loads(self.config_path.read_text(encoding="utf-8"))
        report_path = Path(data["reward"]["pretrain_report_json"])
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["gate_whitelist"] = members
        report_path.write_text(json.dumps(report), encoding="utf-8")

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
        return RolloutSampler(
            CfgCombinedField(unet),
            SdeKernel(eta=config.policy.sde_eta, s_max=config.policy.sde_s_max),
            SingleConditionSchedules(
                num_inference_steps=config.policy.num_inference_steps,
                input_img_size_numel=config.policy.input_img_size_numel,
            ),
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
    """AC 1/3/4：单 iteration 全链路绿、事件流、逐 k loss。"""

    def test_single_iteration_runs_green_and_emits_iter_event(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        scenario.write_inputs()
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.events()
        # overfit_alert 合法插入流中（小 real 池上判别器记忆化、分叉越线
        # 即告警）：只对照 iter 事件
        assert [
            event["event"] for event in events if event["event"] == "iter"
        ] == ["iter"]

    def test_iter_event_carries_health_metrics(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """iter 事件：Anchor eval reward、非退化组内 reward std（G=12）、
        held-out AUC、loss 组件（逐 k policy + discriminator）、lr、采样的
        目标序列（per-sequence 健康监控的归因轴——条件分布每 iter 均匀采
        一个序列，事件不记序列名时 reward/loss/AUC 无法归因，随机序列
        变化会伪装成趋势）。"""
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
        assert event["lr"] == pytest.approx(2e-6)
        assert event["elapsed_s"] >= 0.0

    def test_iter_event_carries_overfit_divergence_observations(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """ADR-0009-β：iter 事件观测面扩展——train 侧干净域 pairwise 准确率
        与 per-condition 分叉 EMA 随判别器步落盘（随单步更新报告上行；
        rank 本地读数、随 iter 事件同归并序，可扩不可改名）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        event = scenario.events()[0]
        acc = event["train_pairwise_acc"]
        divergence = event["overfit_divergence_ema"]
        assert acc is not None and 0.0 <= acc <= 1.0  # 干净域 pairwise 占比
        assert divergence is not None and -1.0 < divergence < 1.0
        # 分叉首观测 = train acc − held-out AUC（EMA 首观测置值）
        assert divergence == pytest.approx(acc - event["heldout_auc"])

    def test_discriminator_skip_leaves_divergence_fields_absent(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """N_d 跳过的 iteration（无判别器步、无复算）分叉观测字段为
        None——「无观测」与「观测为 0」不靠对方推断。"""
        scenario.write_inputs()
        scenario.set_schedule(max_iterations=2)
        scenario.patch_config(reward={"disc_update_interval_n_d": 2})
        result = scenario.train()
        assert result.code == 0, result.stderr
        first, second = scenario.events()
        assert first["train_pairwise_acc"] is not None
        assert first["overfit_divergence_ema"] is not None
        assert second["train_pairwise_acc"] is None  # N_d 跳过：无判别器步
        assert second["overfit_divergence_ema"] is None

    def test_default_threshold_emits_no_overfit_alert(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """默认报警阈值下 fixture 单 iteration 无告警（健康判别器两侧同
        估计量、分叉贴 0）——报警面的静默侧在循环层贯通（越线触发路径
        由 test_overfit 的边界单测与事件契约测试收口）。"""
        scenario.write_inputs()
        assert scenario.train().code == 0
        assert [
            event for event in scenario.events()
            if event["event"] == "overfit_alert"
        ] == []

    def test_discriminator_update_interval_n_d_is_consumed(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """判别器更新节奏 N_d 由训练循环消费：N_d=2 时每 2 个 iteration
        更新一次（跳过的 iteration 无 discriminator loss）。"""
        scenario.write_inputs()
        data = json.loads(scenario.config_path.read_text(encoding="utf-8"))
        data["schedule"]["max_iterations"] = 2
        data["reward"]["disc_update_interval_n_d"] = 2
        scenario.config_path.write_text(json.dumps(data), encoding="utf-8")
        result = scenario.train()
        assert result.code == 0, result.stderr
        first, second = scenario.events()
        assert "discriminator" in first["loss"]
        assert "discriminator" not in second["loss"]  # N_d 跳过

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
        手动构造的 normalizer 按被测 latent 的 device 落位（生产装配由
        trainer 单点 ``.to()`` 接管的同款语义，#95），GPU 可见环境不再被
        normalize 的 device fail-fast 拦在断言层之前；state_dict 逐位对比
        与前向对比都在 CPU 侧做——加速器上同权重不同实例的前向偏离实测
        可超 1e-6 容差（conv kernel 的分块/归约随实例内存布局漂移，#95
        集群复测非偶发），CPU 路径的 1 ulp 噪声底下这层容差才站得住。"""
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
        sample = torch.zeros(1, *config.latent_shape)
        device = trainer.device  # 训练装配设备（GPU 可见即加速器、CPU 强制即 cpu）
        # 测试构造按被测 latent 的 device 落位（#95）：normalize 的 device
        # fail-fast 契约要求统计量 buffer 与输入同源——手动构造的
        # normalizer 与生产装配一样迁移到 latent 所在 device，GPU 可见环境
        # 不再被契约拦截在真正的断言层之前。
        normalizer = ChannelNormalizer(
            ChannelStats.load(config.reward.channel_stats_json),
        ).to(device)
        normalized = normalizer.normalize(sample)
        # 前向对比固定 CPU 执行路径（#95 集群实测）：加速器上同权重不同
        # 实例的前向偏离可超 1e-6 容差且非偶发，装载实例不迁移——断言层
        # 与 CPU 强制执行完全同噪声特性；打分语义由 state_dict 逐位对比
        # 承担（见下），前向只作同权重可复现性的复核。
        expected = copy.deepcopy(live).cpu().eval()(normalized.cpu())[-1]
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
            # 逐位对比承担形态判别力（saved 以 map_location="cpu" 读入，
            # torch.equal 跨 device 拒绝；逐位还原语义与执行 device 无关）
            assert all(torch.equal(restored[key], saved[key]) for key in saved)
            # 前向层用绝对容差而非逐位：前向在**同权重、不同实例**间可差
            # 1 ulp（2^-23 ≈ 1.2e-7，浮点执行路径的分配/分块选择，实测
            # 偶发），逐位断言会假红；1e-6 在噪声底之上、修复前形态的
            # 偏离之下。
            assert torch.allclose(
                reloaded.eval()(normalized.cpu())[-1], expected,
                rtol=0.0, atol=1e-6,
            )

    @pytest.mark.gpu  # 里程碑节奏需 3 iteration 训练
    @pytest.mark.slow  # 默认跳过，--run-slow 全量时运行
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

    @pytest.mark.gpu  # 5 步 rollout 日程（本机实测 72s，大轮次）
    @pytest.mark.slow  # 默认跳过，--run-slow 全量时运行
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
        scenario.fork_pretrained_artifacts()  # 篡改面私有化（共享工件只读）
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
        # 断言面取 iter 事件（本测试的契约 = 组2 全链路绿 + iter 事件内容）：
        # 流里可能顺带有 overfit_alert——fixture 判别器在小 real 池上
        # 天然记忆化（ADR-0009-β 的监控面），其触发不属本测试锁定范围
        iter_events = [event for event in events if event["event"] == "iter"]
        assert len(iter_events) == 1
        event = iter_events[0]
        assert event["modality"] in MODALITIES  # 目标序列归因轴（12 对的目标端）
        assert event["intra_group_reward_std"] > 0.0  # CFG=0 场的组内方差非退化
        assert "policy_step_1" in event["loss"]
        assert "discriminator" in event["loss"]

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

    def test_condition_source_label_matches_source_modality(
        self, tmp_path: Path,
    ) -> None:
        """组2 条件采样产出双 label（issue #115）：source_label = 源序列
        token、label = 目标序列 token，各自与 latent 的源/目标模态对齐
        （源 latent 可从张量内容反查序列，label 映射互为印证）。"""
        torch.manual_seed(0)
        self._write_identifiable_pool(tmp_path)
        sampler = self._sampler(tmp_path, seed=0)
        source_marker = {
            float(index): modality for index, modality in enumerate(MODALITIES)
        }
        seen: set[tuple[str, str]] = set()
        for _ in range(48):
            condition, target = sampler.sample()
            source = source_marker[condition.source_latent[0, 0, 0, 0, 0].item()]
            assert condition.source_label is not None
            assert int(condition.source_label[0].item()) == (
                FIXTURE_MODALITY_MAPPING[source]
            )
            assert int(condition.label[0].item()) == FIXTURE_MODALITY_MAPPING[target]
            seen.add((source, target))
        assert seen <= {
            (src, tgt) for src, tgt in product(MODALITIES, repeat=2) if src != tgt
        }
        assert any(source != target for source, target in seen)

    def test_sample_target_source_label_matches_fixed_target(self, tmp_path: Path) -> None:
        """配额量产（ADR-0008-01）同样产双 label：source_label 随采中的
        源序列走、label 恒为指定目标端 token（与 sample 同一构造点）。"""
        torch.manual_seed(0)
        self._write_identifiable_pool(tmp_path)
        sampler = self._sampler(tmp_path, seed=0)
        source_marker = {
            float(index): modality for index, modality in enumerate(MODALITIES)
        }
        for target in MODALITIES:
            for _ in range(12):
                condition = sampler.sample_target(target)
                source = source_marker[condition.source_latent[0, 0, 0, 0, 0].item()]
                assert condition.source_label is not None
                assert int(condition.source_label[0].item()) == (
                    FIXTURE_MODALITY_MAPPING[source]
                )
                assert int(condition.label[0].item()) == (
                    FIXTURE_MODALITY_MAPPING[target]
                )

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


def _reward_config_for_gating() -> RewardConfig:
    """门控对象装配的最小 RewardConfig（默认 knobs；纯单测用途）。"""
    return RewardConfig(
        disc_batch_size_k=4,
        real_pool_manifest="artifacts/real_pool.json",
        heldout_real_manifest="artifacts/heldout_real.json",
        channel_stats_json="artifacts/channel_stats.json",
        pretrain_report_json="artifacts/pretrain_report.json",
    )


class TestDiscriminatorSideOrchestration:
    """判别器侧编排（train 循环第 2 相的配对批供给与判别器相位）。"""

    def test_update_step_consumes_pair_batch_and_restores_phase(self) -> None:
        """配对批原样透传给 update（ADR-0012：混采置换退役——批就是
        更新批），判别器在更新期间处 train 相（spectral norm
        power iteration 属训练语义）、结束后恢复 eval 相（打分/监控
        前向不得漂移其有效权重）。"""
        discriminator = torch.nn.Linear(1, 1)
        update = RecordingUpdate(discriminator)
        coordinator = RewardCoordinator(
            update, auc=None,  # type: ignore[arg-type]  # 本测试不触 AUC
            gating=DynamicWhitelist(
                ConditionWhitelist.unrestricted(tuple(MODALITIES)),  # 本测试不触白名单
                _reward_config_for_gating(),
                DistributedContext(0, 1, False),
                conditions=tuple(MODALITIES),
            ),
            overfit=OverfitMonitor(
                _reward_config_for_gating(), conditions=tuple(MODALITIES),
            ),
            assembler=None,  # type: ignore[arg-type]  # 替身场景不经装配原语
        )
        pair = PairBatch(
            reals=torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1, 1),
            fakes=torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1, 1) + 10,
            modality="t2w",
        )
        coordinator.update_step(pair)
        assert torch.equal(update.received[0].reals, pair.reals)
        assert torch.equal(update.received[0].fakes, pair.fakes)
        assert update.received[0].modality == "t2w"
        assert update.training_at_call[0] is True  # 更新期间 train 相
        assert discriminator.training is False  # 结束恢复 eval 相

    def test_scoring_forward_is_idempotent_under_spectral_norm(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """spectral norm 启用时打分幂等：打分/监控前向恒在 eval 相
        （power iteration 不推进）——判别器若停在 train 相打分，同批
        两次 forward 因谱归一化漂移分数不同，reward 归因被污染。

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
        sample = torch.zeros(1, *config.latent_shape)
        scorer = trainer.rewards.update.scorer
        first = scorer.reward(sample)
        second = scorer.reward(sample)
        assert torch.allclose(first, second, rtol=0.0, atol=1e-6)  # eval 相打分幂等

    def test_update_step_receives_iteration_condition(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """AC4：train 循环的 update_step 调用点穿本 iteration 条件——
        替身记录到的条件与 iter 事件的 modality 归因轴一致（本 iteration
        的 fake 批、回放过滤与 real 采样三侧条件的同源观测锁）。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        update = RecordingUpdate(trainer.rewards.discriminator)
        trainer.rewards.update = update
        assert trainer.run() == 1
        event = scenario.events()[0]
        assert update.modalities == [event["modality"]]

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

    @pytest.mark.slow  # 集群实测 ~109s：同权重两次完整前向的 CPU 对照
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
        sample = torch.zeros(1, *config.latent_shape, device=trainer.device)
        assert sample.device.type == "cpu"
        assert trainer.rewards.update.scorer.reward(
            sample,
        ).device.type == "cpu"


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

    @pytest.mark.gpu  # |M|=2 的 5 步日程 × dump 重放（本机实测 78s）
    @pytest.mark.slow  # 默认跳过，--run-slow 全量时运行
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
    """rollout（policy 域）→ reward（real pool 存储域）的归位
    （T12 探针定谳）：fake 在打分前除 latent_scale_factor——
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
        record = trainer.loop.run_iteration()
        return record, scored

    @pytest.mark.gpu  # 两次完整场景训练（scale 对比）
    @pytest.mark.slow  # 默认跳过，--run-slow 全量时运行
    def test_scored_fakes_match_new_fakes_domain(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """打分输入与 new_fakes 同值同序（同一归一域）：打分与
        AUC fake 侧消费同一批归一后 latent，不出现跨域错配。"""
        record, scored = self._rollout_with_scale(scenario, 2.0, "domain")
        assert scored, "打分记录为空"
        assert torch.equal(torch.cat(scored), record.new_fakes)

    @pytest.mark.gpu  # 两次完整场景训练（scale 对比）
    @pytest.mark.slow  # 默认跳过，--run-slow 全量时运行
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


class TestGradientGating:
    """逐 iteration 梯度门控与动态恢复（ADR-0008 决策 7/8，issue #89）。

    端到端口径：``TrainingLoopScenario.narrow_whitelist`` 收窄白名单
    （库场景默认全条件放行），名单外条件的 policy 更新被跳过、判别器
    侧照常——门控状态随续训分片落盘。滞回判定的数值语义（enter/exit/
    EMA 递推）由 test_gating 的决定面单测收口；多 rank 集体一致性由
    test_distributed 的 spawn world 覆盖。
    """

    @pytest.mark.gpu  # 4 iteration 训练（大轮次）
    def test_gated_iteration_skips_policy_update_only(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """名单外条件的 iteration：policy 更新被跳过（loss 无
        policy_step_* 项、事件带 policy_gated 标记）；rollout、判别器
        更新、iter 事件照常——被门控条件的判别器持续受训（fake 批由
        重构装配原语现场供批）。静态白名单（动态恢复关闭）下名单逐位
        恒定——恢复的数值语义由 test_gating 决定面单测收口，此处不叠加
        测量噪声。"""
        scenario.write_inputs()
        scenario.set_schedule(max_iterations=4)
        scenario.narrow_whitelist(["t1n"])
        scenario.patch_config(reward={"gating_dynamic_recovery": False})
        result = scenario.train()
        assert result.code == 0, result.stderr
        iter_events = [
            event for event in scenario.events() if event["event"] == "iter"
        ]
        assert len(iter_events) == 4
        gated = [event for event in iter_events if event["policy_gated"]]
        assert gated  # 名单外条件 3/4：4 iteration 至少一个 gated
        for event in gated:
            assert event["modality"] != "t1n"
            policy_terms = [
                key for key in event["loss"] if key.startswith("policy_step")
            ]
            assert not policy_terms  # policy 更新被跳过
            assert "discriminator" in event["loss"]  # 判别器更新照常（N_d=1）
            assert 0.0 <= event["heldout_auc"] <= 1.0  # AUC 观测照常
        # 门控状态随续训分片落盘：静态名单逐位恒定、无观测记录
        # （版本常量对账——分片格式随功能演进，断言不硬编码版本号）
        state = scenario.resume_state()
        assert state["format_version"] == RESUME_STATE_FORMAT_VERSION
        assert state["gating"]["members"] == ["t1n"]
        assert state["gating"]["ema"] == {}

    @pytest.mark.gpu  # 单 iteration 训练（大轮次口径与既有全链一致）
    def test_dynamic_recovery_streams_observations(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """动态恢复开启（默认）：AUC 观测流落进续训分片的门控状态
        （per-condition EMA 有记录）——恢复评估的数据面贯通；名单是否
        因实测值进出由测量决定（该语义的决定论覆盖在 test_gating）。"""
        scenario.write_inputs()
        scenario.narrow_whitelist(["t1n"])
        result = scenario.train()
        assert result.code == 0, result.stderr
        state = scenario.resume_state()
        assert len(state["gating"]["ema"]) == 1  # 单 iteration 单条件观测
        observed = next(iter(state["gating"]["ema"].values()))
        assert 0.0 < observed["value"] <= 1.0
        assert observed["count"] == 1
        assert "t1n" in state["gating"]["members"]  # 名单内条件不被门控


class TestPairedBatchSupplyGuards:
    """配对批配置的装配接受性（ADR-0012）：更新批换配对批后，real 侧
    容量守卫（逐 (全池, 模态) 容量 ≥ K）按配对批 K 校验，任意正 K 的
    合法配置不再被回放供给守卫误拒（该守卫已随 ADR-0012 退役删除，
    #173）。"""


    def test_k1_paired_batch_is_accepted_by_assembly(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """K=1（最小配对批）：装配原语支持任意正 K、real 侧容量守卫按
        K 校验，完整训练照常跑通。"""
        scenario.write_inputs(reward={"disc_batch_size_k": 1})
        scenario.set_schedule(max_iterations=1)
        result = scenario.train()
        assert result.code == 0, result.stderr


class TestOnPolicyReconstructionSupply:
    """在线判别器更新的 on-policy 供批（ADR-0012，issue #172 AC）。

    主循环口径：判别器更新批 = 装配原语用**当前 policy** 现做的同源
    重构配对批——fake 随 policy 权重演化（同一 real 在 policy 更新前后
    重构不同），rollout latent 不再进判别器更新批（此后只承担打分、
    advantage 与在线 AUC——AUC 的 fake 侧 = rollout 终点，链路零改动由
    既有测试回归）。重构前向是 policy 的推理前向：no_grad + autocast
    口径、不产生评估相副作用（判别器的 spectral norm 幂迭代等推进属
    更新步训练语义，装配前向不触）。"""

    K = 2
    """重构核直喂的批大小（``reconstruct`` 纯数学无 real 采样，任意正
    K 合法——取小批压测试成本）。"""

    def test_update_batch_comes_from_assembler(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """判别器更新批 = 装配原语的配对批：update 原样消费
        ``assembler.assemble`` 的产出（同对象透传），条件标记与 iter
        事件同源，fake ≠ real（σ>0 候选下重构非透传）。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        update = RecordingUpdate(trainer.rewards.discriminator)
        trainer.rewards.update = update
        assembler = trainer.rewards.assembler
        assembled: list[PairBatch] = []
        original_assemble = assembler.assemble

        def recording_assemble(modality: str) -> PairBatch:
            pair = original_assemble(modality)
            assembled.append(pair)
            return pair

        assembler.assemble = recording_assemble  # type: ignore[method-assign]
        assert trainer.run() == 1
        assert len(assembled) == 1  # N_d=1：单 iteration 恰一步判别器更新
        assert update.received[0] is assembled[0]
        assert assembled[0].modality == scenario.events()[0]["modality"]
        assert not torch.equal(assembled[0].fakes, assembled[0].reals)

    def test_reconstruction_follows_policy_weights(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """fake 随 policy 权重演化（on-policy 的定义性观测）：同一批
        real、同一 condition/s/ε 确定性输入（不经 recon 流抽签），一个
        iteration 的逐 k policy 更新前后重构不同——装配原语的重构前向
        持 policy 网络引用，权重演化即时反映进更新批的 fake 侧。"""
        scenario.write_inputs()
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assembler = trainer.rewards.assembler
        modality = MODALITIES[0]
        condition = trainer.policy.conditions.sample_target(modality)
        sigmas = [assembler.candidate_sigmas(modality)[0]] * self.K
        stream = torch.Generator().manual_seed(41)
        # 确定性输入随 trainer 设备落位（生产路径 real 批在加速器上，
        # CPU 常量的 device mismatch 在本机 CPU fixture 口径测不到）
        reals = torch.randn(
            self.K, *Fixture.LATENT_SHAPE, generator=stream,
        ).to(trainer.device)
        noise = torch.randn(
            self.K, *Fixture.LATENT_SHAPE, generator=stream,
        ).to(trainer.device)
        # 对比唯一变量 = policy 权重：两次重构同处 eval 相（与 rollout/
        # 重构同 inference 口径），s 与 ε 都是固定输入、不走 recon 流
        trainer.policy.eval_phase()
        before = assembler.reconstruct(reals, condition, sigmas, noise)
        assert not torch.equal(before, reals)  # σ>0 重构非透传
        assert trainer.run() == 1  # 逐 k policy 更新 + 判别器更新照常
        after = assembler.reconstruct(reals, condition, sigmas, noise)
        assert not torch.equal(after, before)

    def test_reconstruction_forward_is_inference_only(
        self, scenario: TrainingLoopScenario,
    ) -> None:
        """额外前向的数值口径与副作用面（AC：no_grad + bf16、评估相零
        副作用）：装配原语的重构前向在 no_grad + autocast(bf16) 下进行
        （前向 hook 在 assemble 窗口内采样 grad/autocast 状态）；判别器
        不进重构链路，装配前后 state_dict（参数 + spectral norm 幂迭代
        的 parametrization buffer）逐位不动——相位敏感推进只属于更新步
        的 train 相（RewardCoordinator.update_step）。SN 经 ``write_inputs``
        的 reward 覆写进库键（预训练与训练同一 reward regime）——warm-start
        产物即谱归一化形态，形态指纹守卫可过；幂迭代的 u/v buffer 存在，
        零触碰断言才有对象。"""
        scenario.write_inputs(reward={"spectral_norm_enabled": True})
        config = ConfigLoader.load(scenario.config_path)
        artifacts = RunArtifacts.init(config, scenario.run_dir)
        trainer = GranularGrpoTrainer(config, artifacts)
        assembler = trainer.rewards.assembler
        device_type = trainer.amp.device_type
        readings: list[tuple[bool, bool, torch.dtype]] = []
        recording = False

        def probe(
            module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if recording:
                readings.append((
                    torch.is_grad_enabled(),
                    torch.is_autocast_enabled(device_type),
                    torch.get_autocast_dtype(device_type),
                ))

        hook = trainer.unet.register_forward_hook(probe)
        original_assemble = assembler.assemble

        def recording_assemble(modality: str) -> PairBatch:
            nonlocal recording
            recording = True
            try:
                return original_assemble(modality)
            finally:
                recording = False

        assembler.assemble = recording_assemble  # type: ignore[method-assign]
        # state_dict 面 = 参数 + buffer（spectral norm 幂迭代的 u/v 活在
        # parametrization buffer——只快照 parameters() 会漏掉它）
        snapshot = {
            name: tensor.detach().clone()
            for name, tensor in trainer.rewards.discriminator.state_dict(
            ).items()
        }
        assembler.assemble(MODALITIES[0])
        hook.remove()
        assert readings  # 重构前向经过了 policy 网络
        assert all(not grad for grad, _, _ in readings)  # no_grad 推理前向
        assert all(autocast for _, autocast, _ in readings)  # autocast 开启
        assert all(
            dtype == torch.bfloat16 for _, _, dtype in readings
        )  # amp_dtype 定死 bf16 的口径
        after = trainer.rewards.discriminator.state_dict()
        assert set(after) == set(snapshot)
        for name, before in snapshot.items():
            assert torch.equal(before, after[name]), name  # 判别器零触碰
