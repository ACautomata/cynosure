"""MR-RATE 线监控链路接线（#124 验收面）。

AC 覆盖：

1. **里程碑 FID 双轨**：里程碑可达的 MR run 产出训练内里程碑 FID 读数
   （``milestone`` 事件、按目标条件分层），里程碑链与裁决性仪器
   （``mr_fid``）两轨不混用（静态结构断言）；
2. **监控子样本 decode**：计数解码器注入的 MR run——decode 只发生在
   Baseline / 里程碑 / 重采三条评测路径的监控子样本（前缀 K 条），
   逐 iteration 主循环零解码；iter 事件账无 decode 行项（五相位恒定）；
3. **监控成本读数**：``milestone`` 事件携带 ``elapsed_s`` + ``phase_seconds``
   （decode / fid 两相分解，#111 监控账的成本行取数面）；
4. **overfit_alert 可触发可订阅**：低阈值下 MR 线告警落流、按事件类型
   可过滤订阅、要素齐备（modality = MR 生成条件名、phase="rl"、rank）。

MR 线的 prepare → pretrain 两段基建沿用 ``test_mr_train.MrTrainScenario``。
"""

import ast
import math
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader
from cynosure.eval import ManifestEvaluation
from cynosure.eval.decode import LatentDecoder
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.numerics import AmpContext
from cynosure.reward.overfit import DivergenceReading
from cynosure.train import BaselineManifest, GranularGrpoTrainer, RunArtifacts
from tests.conftest import CliSession
from tests.test_mr_train import CONDITIONS, PHASES, MrTrainScenario

pytestmark = pytest.mark.gpu  # MR fixture 全链路（prepare 预编码 + 训练轮次）

SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "cynosure"

MILESTONE_CHAIN_MODULES = (
    "__init__.py",
    "milestone.py",
    "volumes.py",
    "sampling.py",
    "decode.py",
    "features.py",
    "frechet.py",
    "condition.py",
)
"""里程碑评测链的 eval 模块（训练循环经 ``EvaluationPhase`` 触达的全部
文件）——双轨不混用的静态断言面。"""

ADJUDICATION_INSTRUMENTS = ("mr_fid", "real_real_floor")
"""裁决性读数仪器的模块名（fork 口径仪器 + real-vs-real 地板）：只经
``cynosure fid`` / ``cynosure fid-floor`` 子命令驱动，训练内里程碑链
不得引用（#73 双轨：里程碑 224 设施口径看趋势，两侧数字不可互比）。"""


class CountingDecoder:
    """测试仪器：计数解码器（decode 只发生在评测路径的运行时观测面，
    与 test_milestone_eval 同款）。"""

    def __init__(self, inner: LatentDecoder) -> None:
        self._inner = inner
        self.calls: list[tuple[int, ...]] = []

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(latents.shape))
        return self._inner.decode(latents)


class TestMilestoneDualTrack:
    """AC：里程碑可达的 MR run 按频率产出训练内里程碑 FID 读数。"""

    def test_milestone_readings_are_condition_stratified(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """里程碑间隔触发的 MR run：``milestone`` 事件入同一指标流，
        FID/KID 有限非负、按目标条件分层的读数随 ``criteria_summary``
        落盘（条件坍缩观测面在 MR 生成条件名上贯通）；监控子样本
        decode 前缀 K 条（= 词汇表条件数），跨条件异形状分组解码。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared))
        scenario.set_schedule(
            max_iterations=2,
            milestone_interval=2,
            milestone_eval_samples=len(CONDITIONS),
        )
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.events()
        assert [
            event["event"] for event in events
            if event["event"] != "overfit_alert"
        ] == ["iter", "iter", "milestone"]
        milestone = next(
            event for event in events if event["event"] == "milestone"
        )
        assert milestone["iteration"] == 2
        assert math.isfinite(milestone["fid"]) and milestone["fid"] >= 0.0
        assert math.isfinite(milestone["kid"])
        summary = milestone["criteria_summary"]
        for condition in CONDITIONS:
            assert math.isfinite(summary[f"fid_target_{condition}"])
            assert math.isfinite(summary[f"kid_target_{condition}"])
        assert milestone["ssim"] is None  # 跨模态组才带 SSIM/MAE/PSNR


class TestDecodeOnlyInMonitoringSamples:
    """AC：decode 只发生在监控子样本，主循环账无 decode 行项。"""

    def test_decode_confined_to_evaluation_paths_with_conditional_shapes(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """计数解码器注入的 MR run（3 iteration、里程碑间隔 2）：decode
        恰好发生在三条评测路径——baseline（4 条目 2 条件 → 组序两批）+
        里程碑（前缀 K=2 条、2 条件 → 两批）+ 重采（同 baseline）；逐
        iteration 循环零解码。异条件异形状批序（t1w/axial (4,16,16,8) →
        flair/axial (4,8,8,16)，#129 条件分组）是 MR 线的独立观测面。
        iter 事件的 ``phase_seconds`` 恒五相位（主循环账无 decode 行项）。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared))
        scenario.set_schedule(
            max_iterations=3,
            milestone_interval=2,
            milestone_eval_samples=len(CONDITIONS),
        )
        config = ConfigLoader.load(scenario.config_path())
        artifacts = RunArtifacts.init(config, scenario.run_dir())
        inner = LatentDecoder(
            NetworkArtifact(
                config=NetworkAssembler.load_json(
                    config.artifacts.vae_config_json,
                ),
                checkpoint=config.artifacts.vae_ckpt,
            ),
            torch.device("cpu"),
            1.0,
            (48, 48, 48),
            0.5,
        )
        counter = CountingDecoder(inner)
        evaluation = ManifestEvaluation.build(
            config,
            artifacts,
            scenario.standalone_sampler(config),
            stage=1,
            manifest=BaselineManifest.load(artifacts.paths.manifest),
            amp=AmpContext(torch.device("cpu"), torch.bfloat16),
            decoder=counter,
        )
        trainer = GranularGrpoTrainer(config, artifacts, evaluation=evaluation)
        assert trainer.run() == 3
        # baseline 4 条目（条件轮转 t1w→flair→t1w→flair）单块分两组，
        # 组序 = 条目首次出现序（ManifestVolumeSampler 的 dict 插入序，
        # 每条件 2 条 → 批维 2）；里程碑评测的组序 = 条件名字典序
        # （MilestoneEvaluator 的 sorted——flair < t1w，各 1 条 → 批维 1）；
        # 重采同 baseline。decode 输入 latent 形状 = 条件 latent 契约
        assert counter.calls == [
            (2, 4, 16, 16, 8), (2, 4, 8, 8, 16),   # baseline：t1w → flair
            (1, 4, 8, 8, 16), (1, 4, 16, 16, 8),   # 里程碑：flair → t1w（名字典序）
            (2, 4, 16, 16, 8), (2, 4, 8, 8, 16),   # 重采：t1w → flair
        ]
        # 主循环账：五相位齐备、无 decode 行项
        for event in scenario.iter_events():
            assert set(event["phase_seconds"]) == set(PHASES)
        # 监控成本读数（#111 监控账成本行取数面）：里程碑事件携带总卡时
        # 与 decode/fid 两相分解（均非负；decode 相 = 合成侧 VAE 解码，
        # fid 相 = 参照装载 + 特征提取 + 距离核）
        milestone = next(
            event for event in scenario.events()
            if event["event"] == "milestone"
        )
        assert milestone["elapsed_s"] > 0.0
        assert set(milestone["phase_seconds"]) == {"decode", "fid"}
        assert all(
            value >= 0.0 for value in milestone["phase_seconds"].values()
        )


class AlertingMonitor:
    """测试仪器：恒越线分叉监控替身（``OverfitMonitor`` 的观测面契约：
    ``observe`` 返回越线读数、``state``/``adopt`` 与续训分片对接）。

    越线判定的数值语义（上升沿、阈值对照、per-condition EMA 递推）已由
    ``test_overfit`` 单元与 #105 在线侧打穿覆盖；本替身的存在理由是
    **确定性**——fixture 判别器的分号方向不受控（随机初始化下 train 侧
    与 held-out 的干净域差在 ±0.1 内波动），自然越线不能作为 MR 线接线
    测试的前置。"""

    def observe(
        self, modality: str, *, train_pairwise_acc: float, heldout_auc: float,
    ) -> DivergenceReading:
        return DivergenceReading(divergence=0.9, alerted=True)

    def state(self) -> dict:
        return {"ema": {}}

    def adopt(self, state: dict) -> None:
        self._adopted = state


class TestOverfitAlertSubscribable:
    """AC：overfit_alert 事件在 MR 线可触发、可订阅。"""

    def test_alert_fires_into_stream_with_full_elements(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """越线观测（替身注入）经 trainer 落成 overfit_alert 事件：随
        iter 事件同归并序写出（告警排本 iter 之后）、按事件类型可从
        指标流过滤订阅，要素齐备——modality 为 MR 生成条件名（per-condition
        归因轴）、phase="rl"（RL 相按 iteration 轴记账）、rank、分叉值与
        两侧原始量均有限。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared), max_iterations=2)
        config = ConfigLoader.load(scenario.config_path())
        artifacts = RunArtifacts.init(config, scenario.run_dir())
        trainer = GranularGrpoTrainer(config, artifacts)
        trainer.rewards.overfit = AlertingMonitor()
        assert trainer.run() == 2
        events = scenario.events()
        alerts = [
            event for event in events if event["event"] == "overfit_alert"
        ]
        # 每 iteration 一次判别器步（N_d=1）→ 逐条件观测恒越线：
        # 2 iter 两条件各一条告警
        assert len(alerts) == 2
        assert [
            (event["iteration"], event["modality"])
            for event in alerts
        ] == [(0, CONDITIONS[0]), (1, CONDITIONS[1])]
        # 归并序：告警排本 rank 对应 iter 事件之后（同 rank 单流交错）
        assert [event["event"] for event in events] == [
            "iter", "overfit_alert", "iter", "overfit_alert",
        ]
        for alert in alerts:
            assert alert["phase"] == "rl"
            assert alert["rank"] == 0
            assert alert["stage"] == 1
            for key in ("divergence_ema", "train_pairwise_acc", "heldout_auc"):
                assert math.isfinite(alert[key])
            assert 0.0 <= alert["heldout_auc"] <= 1.0
            assert 0.0 <= alert["train_pairwise_acc"] <= 1.0
        # 报警不动作（ADR-0009 决策 5）：iter 事件的白名单状态不受告警
        # 联动——全条件在名单内（fixture 预训练过线），无 gated 迭代
        assert all(
            event["policy_gated"] is False
            for event in scenario.iter_events()
        )


class TestDualTrackIsolation:
    """AC：里程碑链与裁决性仪器两轨不混用（静态结构断言）。"""

    def test_milestone_chain_never_references_adjudication_instruments(
        self,
    ) -> None:
        """里程碑评测链（训练循环经 ``EvaluationPhase`` 触达的 eval 模块）
        不 import、不引用裁决性仪器（``mr_fid`` / ``real_real_floor``）
        ——#73 双轨的模块级隔离：里程碑 224 设施口径只看趋势，裁决性
        读数唯一入口是 ``cynosure fid`` / ``cynosure fid-floor`` 子命令，
        两侧数字不可互比也不可互调。"""
        offenders: list[str] = []
        instrument_names = {"MrFidConfig", "MrFidInstrument", "FidResult"}
        for name in MILESTONE_CHAIN_MODULES:
            source = SRC_ROOT / "eval" / name
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(
                            part in alias.name
                            for part in ADJUDICATION_INSTRUMENTS
                        ):
                            offenders.append(f"{name}:{node.lineno} {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and any(
                        part in node.module
                        for part in ADJUDICATION_INSTRUMENTS
                    ):
                        offenders.append(f"{name}:{node.lineno} {node.module}")
                elif isinstance(node, ast.Name) and node.id in instrument_names:
                    offenders.append(f"{name}:{node.lineno} {node.id}")
        assert offenders == [], f"里程碑链引用裁决性仪器: {offenders}"
