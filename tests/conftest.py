"""共享测试夹具：CLI 会话（唯一 seam 驱动器）、合法最小 config 样板、
合成 BraTS 数据集（fixture 策略）、预训练轻量 reward 变体、HeldOutAuc
失败替身。"""

import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from cynosure.cli import CynosureCli
from cynosure.config import CynosureConfig, DEFAULT_CROSS_MODAL_PAIRS, MODALITIES
from cynosure.fixtures import Fixture
from cynosure.reward.buffer import ReplayBuffer
from cynosure.reward.update import UpdateReport

# 测试进程的 torch CPU 线程池固定为 4 线程。缺省值 = 全部物理核
# （sugon 上 ~56）：集群被他人训练任务占满时，小张量算子（如 MR FID
# 骨干在 CPU 上的 resnet50 forward，16×16 切片）拆给几十个线程后
# 层间同步开销远超计算量，OMP 线程自旋空转——实测单测试烧 557 分钟
# CPU 仍算不完（600 秒墙钟超时）；限 4 线程后 15 秒通过。测试数据
# 全是小张量，多线程无收益；CPU 算子按输出元素划分、归约顺序固定，
# 线程数不影响逐位结果。spawn 出的训练 worker 是独立进程，不继承
# 此限制。
torch.set_num_threads(4)


class SceneCache:
    """训练场景包的跨进程缓存（``FixtureArtifactLibrary`` miss 路径的
    持久层）：网络工件 + 合成数据集 + prepare 三工件 + pretrain 产物
    全部由固定 seed 决定——同场景参数下跨进程重建是纯冗余（``pytest
    -n`` 的每个 worker 都是独立进程，进程内缓存互不可见；单进程重跑
    pytest 同样吃不到上一进程的产物）。命中即整包 ``copytree`` 出库：
    库目录的消费语义（只读 + fork 篡改）不受影响。

    并发（pytest-xdist 多 worker 共享缓存目录）：``store`` 以唯一临时
    目录构建后原子 ``rename`` 提交，最终目录存在即完整，读侧无锁。
    """

    ROOT = Path(tempfile.gettempdir()) / "cynosure-scene-cache"

    _namespace: str | None = None

    @classmethod
    def key(cls, *parts: object) -> str:
        """场景参数 → 缓存键（入参须 JSON 可序列化；集合型先排序归一）。

        键掺 src 树内容指纹：场景产物依赖整条 prepare/pretrain 代码
        路径，任何源码改动都使旧缓存失效（否则修 bug 的测试会吃到旧
        代码的产物）。指纹每进程只算一次。"""
        if cls._namespace is None:
            digest = hashlib.sha256()
            root = Path(__file__).resolve().parent.parent / "src" / "cynosure"
            for source in sorted(root.rglob("*.py")):
                digest.update(str(source.relative_to(root)).encode())
                digest.update(source.read_bytes())
            cls._namespace = digest.hexdigest()[:12]
        return hashlib.sha256(
            json.dumps([cls._namespace, *parts], sort_keys=True, default=list).encode(),
        ).hexdigest()[:16]

    @classmethod
    def load(cls, key: str) -> Path | None:
        cached = cls.ROOT / key
        return cached if cached.is_dir() else None

    @classmethod
    def store(cls, key: str, source: Path) -> None:
        """构建缓存包（原子提交；已存在则放弃自己的副本用现成的）。"""
        final = cls.ROOT / key
        if final.exists():
            return
        cls.ROOT.mkdir(parents=True, exist_ok=True)
        staging = cls.ROOT / f".staging-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            shutil.copytree(source, staging)
            os.replace(staging, final)  # 同盘 rename 原子
        except OSError:
            shutil.rmtree(staging, ignore_errors=True)


@dataclass
class CliResult:
    """一次进程内 CLI 调用的外部可观测结果。"""

    code: int
    stdout: str
    stderr: str


class RunTrajectory:
    """iter 事件流的轨迹可比面（值对象）：wall-clock ``elapsed_s`` 不参与
    相等性——跨作业对比的语义轴是事件序下的其余字段。同**进程**重放
    （单进程续训 roundtrip）在此逐位断言；跨进程世界对的对比（分布式
    续训 roundtrip 等）的观测前向存在 1-2 ulp 重算噪声，走
    test_distributed.CrossPathEquivalence 的容差判定。"""

    def __init__(self, events: list[dict]) -> None:
        self._events = [
            {key: value for key, value in event.items() if key != "elapsed_s"}
            for event in events
        ]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunTrajectory) and self._events == other._events

    def __repr__(self) -> str:
        return (
            "RunTrajectory(iterations="
            f"{[event.get('iteration') for event in self._events]})"
        )


class CliSession:
    """CLI 会话：向 cynosure 命令行提交 argv 并捕获输出。"""

    def run(self, *args: str) -> CliResult:
        stdout, stderr = io.StringIO(), io.StringIO()
        code = CynosureCli(list(args), stdout, stderr).run()
        return CliResult(code, stdout.getvalue(), stderr.getvalue())

    def write_config(self, directory: Path, overrides: dict | None = None) -> Path:
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        for key, value in (overrides or {}).items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
        path = directory / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def train(self, config_path: Path, run_dir: Path | None = None) -> CliResult:
        argv = ["train", "--config", str(config_path)]
        if run_dir is not None:
            argv += ["--run-dir", str(run_dir)]
        return self.run(*argv)

    def sole_run_directory(self, home: Path) -> Path:
        """本会话（$HOME 下）唯一一次 train 产出的 run 目录。"""
        return next((home / ".cynosure" / "runs").iterdir())


@pytest.fixture
def cli() -> CliSession:
    return CliSession()


def pytest_addoption(parser: pytest.Parser) -> None:
    """``--run-slow`` 全量开关：slow 标记（特别耗时大轮次）默认跳过，
    日常开发只跑轻量子集；显式要求时才整包运行。"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行 slow 标记的特别耗时测试（默认跳过；全量验证用）",
    )


def pytest_collection_modifyitems(config, items) -> None:
    """标记的两条正交分派轴：

    ``gpu``（环境轴）——无 CUDA 环境自动跳过大轮次测试：CPU 口径（本机 /
    集群 ``CUDA_VISIBLE_DEVICES=""``）只跑轻量子集，验证职责由集群 GPU
    口径全量承担（仓库纪律：测试一律上集群）。

    ``slow``（成本轴）——默认跳过特别耗时测试（多 iteration 完整训练 /
    torchrun 多进程 / 像素域解码评测等小时级轮次），``--run-slow`` 显式
    全量才运行；日常开发不为一轮完整训练买单。"""
    skip_gpu = pytest.mark.skip(
        reason="gpu 标记（大轮次测试）：CPU 环境跳过，由集群 GPU 口径全量覆盖",
    )
    skip_slow = (
        None if config.getoption("--run-slow")
        else pytest.mark.skip(
            reason="slow 标记（特别耗时）：默认跳过，--run-slow 显式全量时运行",
        )
    )
    for item in items:
        if not torch.cuda.is_available() and "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if skip_slow is not None and "slow" in item.keywords:
            item.add_marker(skip_slow)


# 12 个有序 src→tgt 对（脑 MRI 四序列，每序列作 anchor、其余三序列为目标）
CROSS_MODAL_PAIRS = [[src, tgt] for src, tgt in DEFAULT_CROSS_MODAL_PAIRS]

# 合成夹具的方向语义（issue #45）：非单位 affine（1mm iso、带平移）——
# LPS 为主（BraTS 原生 ~89% LPS，flip-only 可达 RAS）、少量 RAS 覆盖
# 「方向已合规」分支；方向重定向因此在 prepare 端到端中真实发生
LPS_AFFINE = np.array(
    [[-1.0, 0.0, 0.0, 90.0],
     [0.0, -1.0, 0.0, 120.0],
     [0.0, 0.0, 1.0, 80.0],
     [0.0, 0.0, 0.0, 1.0]],
)
RAS_AFFINE = np.array(
    [[1.0, 0.0, 0.0, 10.0],
     [0.0, 1.0, 0.0, 20.0],
     [0.0, 0.0, 1.0, 30.0],
     [0.0, 0.0, 0.0, 1.0]],
)

# 各向异性 zooms（issue #46，float32 精确值 (0.5, 1.0, 2.0)）：per-case spacing
# 变化的观测载体——写死常量必假绿的判别性断言用它驱动数据变化
ANISOTROPIC_AFFINE = np.diag([-0.5, -1.0, 2.0, 1.0])

# 组1、生产尺寸的最小合法 config（必填字段全部显式给出）
MINIMAL_CONFIG_DICT: dict = {
    "experiment": {"group": "modal-label"},
    "latent_shape": [4, 64, 64, 32],
    "artifacts": {
        "unet_ckpt": "ckpts/unet.pt",
        "vae_ckpt": "ckpts/vae.pt",
        "net_config_json": "configs/net.json",
        "modality_mapping_json": "configs/modality_mapping.json",
        "dataset_root": "data/brats2023",
    },
    "reward": {
        "disc_batch_size_k": 4,
        "replay_buffer_capacity": 64,
        "real_pool_manifest": "artifacts/real_pool.json",
        "heldout_real_manifest": "artifacts/heldout_real.json",
        "channel_stats_json": "artifacts/channel_stats.json",
        # 预训练产物契约（ADR-0007）：RM readiness gate 的守卫装载源（必填无默认）
        "pretrain_report_json": "artifacts/pretrain_report.json",
    },
    "schedule": {"seed": 0},
}


@pytest.fixture
def valid_config_dict() -> dict:
    return copy.deepcopy(MINIMAL_CONFIG_DICT)


@pytest.fixture
def valid_config_json(tmp_path: Path, valid_config_dict: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config_dict), encoding="utf-8")
    return path


class SyntheticBratsDataset:
    """合成 BraTS2023 病例数据集（fixture 策略）：每病例一目录、四序列 NIfTI，
    与生产 dataset_root 同一目录布局，供 prepare 全循环在本地 CPU 跑。

    夹具影像携带真实方向语义（issue #45，affine 常量见模块级 LPS_AFFINE /
    RAS_AFFINE）：非单位 affine，LPS 为主、确定性混入 RAS（每第 5 个病例
    RAS）——方向重定向在 prepare 端到端中真实发生，而不是被单位 affine 架空。"""

    def __init__(self, root: Path, case_ids: list[str], shape: tuple[int, int, int],
                 seed: int) -> None:
        self._root = root
        self._case_ids = case_ids
        self._shape = shape
        self._seed = seed

    def write(self) -> Path:
        """按病例目录布局落盘四序列 NIfTI（确定性：seed + 病例/序列派生子种子
        决定体数据；病例下标决定方向语义）。"""
        for case_index, case_id in enumerate(self._case_ids):
            case_dir = self._root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            affine = RAS_AFFINE if case_index % 5 == 0 else LPS_AFFINE
            for modality_index, modality in enumerate(MODALITIES):
                rng = np.random.default_rng(
                    (self._seed, case_index, modality_index),
                )
                volume = rng.standard_normal(self._shape).astype(np.float32)
                self._write_nifti(
                    case_dir / f"{case_id}-{modality}.nii.gz", volume, affine,
                )
        return self._root

    @staticmethod
    def _write_nifti(path: Path, volume: np.ndarray, affine: np.ndarray) -> None:
        nib.save(nib.Nifti1Image(volume, affine), path)


class FixturePrepareScenario:
    """prepare 子命令的 fixture 端到端场景（reward 侧测试共用）：合成
    BraTS 数据集 → prepare 跑通 → pool / held-out / channel stats 三工件。"""

    NUM_CASES: int = 20
    SERIES_SHAPE: tuple[int, int, int] = (64, 64, 32)

    def __init__(
        self, cli: CliSession, config: CynosureConfig, work_dir: Path,
    ) -> None:
        self._cli = cli
        self._config = config
        self._work_dir = work_dir

    def run(self, config_path: Path) -> CynosureConfig:
        """写合成数据集并经 CLI prepare 落三工件（断言退出码 0）；
        返回驱动 prepare 的 config（seed 定死 0，供后续训练复用工件）。"""
        self._config.schedule.seed = 0
        SyntheticBratsDataset(
            self._config.artifacts.dataset_root,
            [f"BraTS-GLI-{index:05d}-000" for index in range(self.NUM_CASES)],
            self.SERIES_SHAPE,
            seed=0,
        ).write()
        config_path.write_text(
            self._config.model_dump_json(indent=2), encoding="utf-8",
        )
        result = self._cli.run("prepare", "--config", str(config_path))
        assert result.code == 0, result.stderr
        return self._config


class PretrainLightweightReward:
    """预训练轻量 reward 变体：warm-start 前置（ADR-0007）的消费方
    （readiness gate / train loop / trajectory diagnostic）共用的成本
    压低取值集——fake 批 4 / 回放容量 16 / 判别器 LR 2e-4 / gate 0.60 /
    步数上限 24。

    容量取下限 16：ADR-0008 决策 4 的装配守卫要求 base 分区每条件配额
    ≥ 回放半区（K=4 → 2 条）——capacity=16 → base 8 → 每条件配额 2
    恰好过线；再小（如 8 → 配额 1）装配期即被拒，预训练无法启动。

    轻量参数只降低预训练本步执行成本、不进训练 config；预训练 gate
    抬到 0.60——达标即停让重算值贴着停止阈值，对 train gate（0.51）
    留出测量噪声的安全 margin。"""

    @classmethod
    def apply(cls, config: CynosureConfig) -> CynosureConfig:
        """返回 reward 侧设为轻量取值的 config 副本（``model_copy``，
        不改入参）。"""
        pretrain = config.model_copy(deep=True)
        pretrain.reward.pretrain_fake_batch = 4
        pretrain.reward.replay_buffer_capacity = 16
        pretrain.reward.disc_lr = 2e-4
        pretrain.reward.pretrain_gate_auc = 0.60
        pretrain.reward.pretrain_max_steps = 24
        return pretrain


class FixtureArtifactLibrary:
    """fixture 场景工件库（pytest 进程内共享缓存）：网络工件 + prepare
    三工件 + 预训练产物按场景变体只构建一次，全进程的 train 场景只读
    复用——场景搭建的 prepare/pretrain 重建成本从每测试一次降为每变体
    一次（集群全量按小时计的执行成本由此收敛）。

    构建结果同时落 ``SceneCache``（跨进程盘上缓存）：进程内缓存随进程
    消亡——``pytest -n`` 的每个 worker 独立冷启动，单进程重跑同样吃
    不到上一进程的产物；盘上层让首个构建者建、其余 worker 与后续 run
    整包 copytree 恢复（键掺 src 树内容指纹，任何源码改动自动失效）。

    缓存键 = 场景变体全签名（组 × 采样日程 × seed × reward 覆写）：键内
    变体的产物与逐测试重建**逐位一致**（构建流程与原 write_inputs 相同）。
    ``sde_eta`` 例外——η 不影响预训练的 anchor 确定性 rollout，不进键。

    库目录放系统临时目录下的**进程私有**位置（PID 后缀），有意游离于
    pytest 的编号 tmp 体系之外：并行 pytest 进程创建更高编号目录时的
    GC（保留最近 3 个编号）会 rm-rf 掉低编号 run 的整个 tmp 树（集群
    实录：并行的 ``pytest -n 16`` 删掉了正在跑的全量的库，缓存命中的
    读取撞上一批截断/缺失文件）。命中时仍校验锚文件存在——外部删除
    不可感知时退化为重建而非带病返回。

    工件目录对消费方**只读**；要篡改/删除预训练产物的测试先 fork 成
    私有副本（场景的 ``fork_pretrained_artifacts``），写共享目录即跨
    测试污染。"""

    _cache: dict[tuple, Path] = {}

    @classmethod
    def _library_root(cls) -> Path:
        root = Path(tempfile.gettempdir()) / (
            f"cynosure-fixture-library-{os.getpid()}"
        )
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def artifacts_dir(
        cls,
        cli: CliSession,
        group: str,
        *,
        num_steps: int = 3,
        train_steps: frozenset[int] = frozenset({1}),
        seed: int = 0,
        reward: dict | None = None,
    ) -> Path:
        """返回该场景变体的工件目录（缺则构建：网络工件 → prepare →
        预训练 warm-start，流程与原逐测试 write_inputs 逐行一致）；
        进程内 miss 先查 ``SceneCache`` 盘上缓存，命中整包恢复、免重建。"""
        signature = (
            group, num_steps, tuple(sorted(train_steps)), seed,
            tuple(sorted((reward or {}).items())),
        )
        cached = cls._cache.get(signature)
        if cached is not None and (cached / "unet.pt").is_file():
            return cached
        token = hashlib.md5(repr(signature).encode()).hexdigest()[:8]
        fixture_dir = cls._library_root() / f"shared_fixtures_{token}"
        disk_key = SceneCache.key("fixture-library", *signature)
        disk_cached = SceneCache.load(disk_key)
        if disk_cached is not None:
            shutil.rmtree(fixture_dir, ignore_errors=True)
            shutil.copytree(disk_cached, fixture_dir)
            cls._normalize_whitelist(fixture_dir)  # 旧盘缓存的白名单同样归一
            cls._cache[signature] = fixture_dir
            return fixture_dir
        fixture = Fixture()
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（test_reward_fixture 先例）
        fixture.write_artifacts(fixture_dir)
        FixturePrepareScenario(
            cli, fixture.config(fixture_dir, group=group), cls._library_root(),
        ).run(cls._library_root() / f"prepare_config_{token}.json")
        # 预训练 warm-start 前置（ADR-0007）：reward 覆写在预训练**之前**
        # 生效——如 SN 启用时预训练产物即谱归一化形态；轻量五元组只降
        # 低本步执行成本（rationale 集中在 PretrainLightweightReward）
        config = fixture.config(fixture_dir, group=group)
        config.policy.num_inference_steps = num_steps
        config.policy.train_step_indices_m = set(train_steps)
        config.schedule.seed = seed
        if reward:
            config.reward = config.reward.model_copy(update=reward)
        pretrain_config = PretrainLightweightReward.apply(config)
        # 序贯降级（#113 注释）：组3 两阶段共享此单份报告——stage-1
        # （modal-label config）同组消费通过等值守卫；stage-2（cross-modal
        # config）消费它在装载期被显式拒绝（跨组消费无逃生门）。stage 级
        # 报告映射（stage-2 消费 cross-modal 报告）随 #116 交付，届时本
        # 降级与 test_sequential 的 xfail 哨兵一并拆除。
        pretrain_config.experiment.group = (
            "modal-label" if group == "sequential" else group
        )
        pretrain_path = cls._library_root() / f"pretrain_config_{token}.json"
        pretrain_path.write_text(
            pretrain_config.model_dump_json(indent=2), encoding="utf-8",
        )
        result = cli.run("pretrain", "--config", str(pretrain_path))
        assert result.code == 0, result.stderr
        cls._normalize_whitelist(fixture_dir)
        SceneCache.store(disk_key, fixture_dir)
        cls._cache[signature] = fixture_dir
        return fixture_dir

    @staticmethod
    def _normalize_whitelist(fixture_dir: Path) -> None:
        """库场景的条件白名单归一为全条件放行（幂等，构建与缓存命中的
        恢复副本上都执行）：轻量预训练的真实白名单是概率性的部分名单
        （如 22 步只确认 2/4 条件）——逐 iteration 梯度门控（ADR-0008
        决策 7，issue #89）落码后，名单外条件的 policy 更新被跳过，
        既有循环测试的 policy loss 断言会随条件采样摇。归一 = 门控不
        触发的场景前置；门控/白名单语义的专项测试 fork 私有 report
        （``fork_pretrained_artifacts``）后显式收窄名单。盘上缓存
        （SceneCache）不回写——归一只发生在进程私有的恢复副本上。"""
        report_path = fixture_dir / "pretrain_run" / "pretrain_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["gate_whitelist"] != list(MODALITIES):
            report["gate_whitelist"] = list(MODALITIES)
            report_path.write_text(json.dumps(report), encoding="utf-8")


class RecordingScorer:
    """测试仪器：以注入判别器冒充打分器（coordinator 取相位的观测载体）。"""

    def __init__(self, discriminator: torch.nn.Module) -> None:
        self.discriminator = discriminator


class RecordingUpdate:
    """测试仪器：记录 update.step 收到的批、条件与调用时的判别器相位
    （buffer 用真实两区实现——RewardCoordinator 的 zone_sizes 观测面
    经它委托；optimizer 为真实现——续训状态机的判别器侧 checkpoint
    经 RewardCoordinator 消费 update.optimizer，协作者契约面的一部分）。
    train 循环与预训练 driver 的 update_step 穿参观测共用同一替身。"""

    def __init__(
        self, discriminator: torch.nn.Module, *,
        buffer_capacity: int = 64, replay_degraded: bool = False,
    ) -> None:
        self.scorer = RecordingScorer(discriminator)
        # 容量须与被替换的装配一致（base 分区填充量随容量配额量产）
        self.buffer = ReplayBuffer(buffer_capacity)
        self.optimizer = torch.optim.AdamW(discriminator.parameters(), lr=5e-5)
        self.received: list[torch.Tensor] = []
        self.modalities: list[str] = []
        self.training_at_call: list[bool] = []
        self._replay_degraded = replay_degraded

    def step(
        self, current_fakes: torch.Tensor, modality: str,
    ) -> UpdateReport:
        self.received.append(current_fakes)
        self.modalities.append(modality)
        self.training_at_call.append(self.scorer.discriminator.training)
        return UpdateReport(
            loss_discriminator=0.0,
            loss_real_term=0.0,
            loss_fake_term=0.0,
            num_current=1,
            num_replay=0 if self._replay_degraded else 1,
            num_base_replay=0,
            num_recent_replay=0,
            modality=modality,
            replay_degraded=self._replay_degraded,
            train_pairwise_acc=0.5,
        )


def pytest_configure(config: pytest.Config) -> None:
    """pytest-xdist worker 线程限额：多 worker 合计贴着物理核数，各自
    拉满 OpenMP 只会超订阅互抢（集群 128 核跑 4 个全量的实测 load
    307）。取核数的 1/16（128 核 = 8 线程；核少机器取 1）；单进程跑
    不受影响。``test_distributed`` 的 spawn rank 从本值继承——跨路径
    数值等价断言要求两路径同线程数（torch 卷积求和顺序随线程数变）。
    """
    if hasattr(config, "workerinput"):
        torch.set_num_threads(max(1, (os.cpu_count() or 16) // 16))
        torch.set_num_interop_threads(1)
