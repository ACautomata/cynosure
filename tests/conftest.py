"""共享测试夹具：CLI 会话（唯一 seam 驱动器）、合法最小 config 样板、
合成 BraTS 数据集（fixture 策略）、预训练轻量 reward 变体。"""

import copy
import io
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from cynosure.cli import CynosureCli
from cynosure.config import CynosureConfig, DEFAULT_CROSS_MODAL_PAIRS, MODALITIES


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
    压低取值集——fake 批 4 / 回放容量 8 / 判别器 LR 2e-4 / gate 0.60 /
    步数上限 24。

    轻量参数只降低预训练本步执行成本、不进训练 config；预训练 gate
    抬到 0.60——达标即停让重算值贴着停止阈值，对 train gate（0.51）
    留出测量噪声的安全 margin。"""

    @classmethod
    def apply(cls, config: CynosureConfig) -> CynosureConfig:
        """返回 reward 侧设为轻量取值的 config 副本（``model_copy``，
        不改入参）。"""
        pretrain = config.model_copy(deep=True)
        pretrain.reward.pretrain_fake_batch = 4
        pretrain.reward.replay_buffer_capacity = 8
        pretrain.reward.disc_lr = 2e-4
        pretrain.reward.pretrain_gate_auc = 0.60
        pretrain.reward.pretrain_max_steps = 24
        return pretrain
