"""MR-RATE 线 RL 主循环端到端（#123 tracer 首跑票的 fixture 层）。

MR 线的 prepare → pretrain 两段已在 ``test_mr_pretrain.py`` 贯通；本票
补的是**主循环第三段**：G 轨迹 rollout → PatchDiscriminator latent 域打分
→ GRPO update → checkpoint + metrics 流落盘，在 MR 线的条件词汇表 / 逐
条件形状 / 逐条件 sigma 日程上跑通，并覆盖本票新交付的两个面：

1. **条件闸开关**（``reward.condition_gate_enabled``）：关闭时 held-out
   AUC 不作任何更新开关（空白名单也开跑、每 iteration 全条件更新），
   开启（既定口径）时空白名单仍硬拒绝、名单外条件仍跳过 policy 更新；
2. **逐 iter 卡时分解**（``iter`` 事件的 ``phase_seconds``）：rollout /
   held-out AUC / 门控回合 / policy 更新 / 判别器更新五相位。

监控相（里程碑解码评测）不在本票范围（MR 参照影像库归 #124）：无里程碑
触发点的 run 不装配监控相，本文件的场景一律
``max_iterations < milestone_interval``——装配分界本身也有独立测试面。
"""

import json
import shutil
from pathlib import Path

import pytest
import torch

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.train.trainer import PhaseTimer
from tests.conftest import CliSession, SyntheticMrRateDataset

CONDITIONS = ["t1w/axial", "flair/axial"]
"""夹具词表的两条件（t1w/axial [4,16,16,8]、flair/axial [4,8,8,16]）——
异形状是本文件「逐条件形状贯通」的输入面。"""

PHASES = (
    "rollout", "heldout_auc", "gating", "policy_update", "discriminator",
)
"""逐 iter 卡时分解的相位键（#123：iter 事件 ``phase_seconds``）。"""


class MrTrainScenario:
    """一次 MR-RATE 主循环端到端场景：合成数据集 → CLI prepare → CLI
    pretrain → CLI train（reward / schedule 覆写可注入）。

    访问器风格与 ``test_mr_pretrain.MrPretrainScenario`` 一致：场景只
    暴露 run 目录工件与事件流，断言在测试侧。``use`` 是训练 config 的
    唯一写入口（prepare/pretrain 的产物路径在它之前已定，train 只在其上
    调 schedule / reward）。
    """

    def __init__(self, cli: CliSession, tmp_path: Path) -> None:
        self._cli = cli
        self._work_dir = Path(tmp_path)
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._fixtures_dir = self._work_dir / "fixtures"
        # 预训练与训练的 run 目录分开：预训练产物落自己那棵树（预训练
        # 报告 = PretrainRun 的 run 目录成员），train 的 --run-dir 另起
        # ——同一目录会被 train 的「不静默覆盖」契约拒绝
        self._pretrain_dir = self._work_dir / "pretrain_run"
        self._run_dir = self._work_dir / "run"
        self._config: CynosureConfig | None = None

    def prepare(self, *, reward_overrides: dict | None = None) -> CynosureConfig:
        """网络工件 + 合成 MR-RATE 数据集 + prepare；返回基线 config。"""
        torch.manual_seed(7)  # fixture 网络「固定 seed」机制（库场景先例）
        Fixture().write_artifacts(self._fixtures_dir)
        config = Fixture().config(self._fixtures_dir, dataset="MR-RATE")
        if reward_overrides:
            config.reward = config.reward.model_copy(update=reward_overrides)
        SyntheticMrRateDataset(config.artifacts.dataset_root).write()
        path = self._work_dir / "prepare_config.json"
        path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        result = self._cli.run("prepare", "--config", str(path))
        assert result.code == 0, result.stderr
        return config

    def pretrain(self, config: CynosureConfig) -> CynosureConfig:
        """判别器 warm-start 预训练（报告落本场景独立的预训练目录）。"""
        prepared = config.model_copy(deep=True)
        prepared.reward.pretrain_report_json = str(
            self._pretrain_dir / "pretrain_report.json",
        )
        path = self._work_dir / "pretrain_config.json"
        path.write_text(prepared.model_dump_json(indent=2), encoding="utf-8")
        result = self._cli.run("pretrain", "--config", str(path))
        assert result.code == 0, result.stderr
        return prepared

    def use(self, config: CynosureConfig, **schedule: object) -> None:
        """把 config 定为本场景的训练 config 并落盘（schedule 覆写随此
        生效）；``max_iterations`` 默认压到里程碑间隔之下——本票的 run
        无监控相（MR 参照影像库归 #124）。"""
        trained = config.model_copy(deep=True)
        trained.schedule.max_iterations = min(
            trained.schedule.milestone_interval - 1, 3,
        )
        for key, value in schedule.items():
            setattr(trained.schedule, key, value)
        self._config = trained
        self.config_path().write_text(
            trained.model_dump_json(indent=2), encoding="utf-8",
        )

    def patch_config(self, **sections: dict) -> None:
        """按 section 覆写训练 config（条件闸开关、门控退化等）。"""
        data = json.loads(self.config_path().read_text(encoding="utf-8"))
        for section, values in sections.items():
            data[section].update(values)
        self.config_path().write_text(json.dumps(data), encoding="utf-8")

    def narrow_whitelist(self, members: list[str]) -> None:
        """把预训练报告的条件白名单收窄为 ``members``（门控场景的条件
        构造，与 ``TrainingLoopScenario.narrow_whitelist`` 同款）：报告
        fork 到场景私有目录后改写，实测值与判别器 checkpoint 不动——
        warm-start 守卫链不受影响。"""
        data = json.loads(self.config_path().read_text(encoding="utf-8"))
        report_path = Path(data["reward"]["pretrain_report_json"])
        private = self._work_dir / "narrowed_pretrain"
        shutil.copytree(report_path.parent, private)
        report = json.loads(
            (private / report_path.name).read_text(encoding="utf-8"),
        )
        report["gate_whitelist"] = members
        (private / report_path.name).write_text(
            json.dumps(report), encoding="utf-8",
        )
        self.patch_config(reward={
            "pretrain_report_json": str(private / report_path.name),
        })

    def set_schedule(self, **values: object) -> None:
        """改 schedule 并重落盘（续训扩迭代数的场景）。"""
        data = json.loads(self.config_path().read_text(encoding="utf-8"))
        data["schedule"].update(values)
        self.config_path().write_text(json.dumps(data), encoding="utf-8")

    def train(self, *, resume: bool = False):
        argv = [
            "train", "--config", str(self.config_path()),
            "--run-dir", str(self._run_dir),
        ]
        if resume:
            argv.append("--resume")
        return self._cli.run(*argv)

    def config_path(self) -> Path:
        return self._work_dir / "train_config.json"

    def config(self) -> CynosureConfig:
        assert self._config is not None
        return self._config

    def run_dir(self) -> Path:
        return self._run_dir

    def whitelist(self) -> list[str]:
        report = json.loads(
            (self._pretrain_dir / "pretrain_report.json").read_text(
                encoding="utf-8",
            ),
        )
        return report["gate_whitelist"]

    def events(self) -> list[dict]:
        return [
            json.loads(line)
            for line in (self._run_dir / "metrics.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]

    def iter_events(self) -> list[dict]:
        return [event for event in self.events() if event["event"] == "iter"]

    def manifest_entries(self) -> list[dict]:
        return json.loads(
            (self._run_dir / "manifest.json").read_text(encoding="utf-8"),
        )["entries"]

    def resume_state(self) -> dict:
        return torch.load(
            self._run_dir / "checkpoints" / "resume_state.pt",
            map_location="cpu", weights_only=True,
        )


class TestMrMainLoop:
    """主循环在 MR-RATE 条件词汇表上的执行序（#123 AC1/AC4）。"""

    @pytest.mark.gpu  # 3 iteration 训练（大轮次口径与既有全链一致）
    def test_main_loop_emits_complete_iter_events(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：N iter 全绿（无 NaN、无异常终止），iter 事件字段齐全——
        reward（anchor_eval_reward / intra_group_reward_std）、门控状态
        （policy_gated）、per-condition 记账（modality）、逐 iter 卡时
        分解（phase_seconds 五相位）；条件按词汇表轮转（异条件异形状在
        同一 run 内贯通）。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared))
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.iter_events()
        assert len(events) == 3
        assert [event["modality"] for event in events] == [
            CONDITIONS[index % len(CONDITIONS)] for index in range(3)
        ]
        for event in events:
            assert event["policy_gated"] is False  # 全条件在名单内
            assert set(event["loss"]) == {"policy_step_1", "discriminator"}
            for key in ("anchor_eval_reward", "intra_group_reward_std",
                        "heldout_auc"):
                value = event[key]
                assert value == value  # 非 NaN
                assert abs(value) <= 1e6
            # 逐 iter 卡时分解：五相位齐备、非负、和不超过总耗时
            assert set(event["phase_seconds"]) == set(PHASES)
            assert all(
                value >= 0.0 for value in event["phase_seconds"].values()
            )
            assert sum(event["phase_seconds"].values()) <= event["elapsed_s"]
            assert event["elapsed_s"] > 0.0
            # per-condition 记账：buffer 两区占用与混采占比
            assert 0.0 <= event["buffer_current_fraction"] <= 1.0
            assert 0.0 <= event["buffer_replay_fraction"] <= 1.0

    @pytest.mark.gpu  # 2 + 2 iteration 训练（大轮次口径）
    def test_checkpoints_are_loadable_and_resume_advances(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：checkpoint 可装载、可续训——前半跑 2 iteration 落盘，续训
        跑到 4 iteration：事件流回退半截后重执行（无重复/无丢失），
        收官 checkpoint 经 netbuild 重新装载成功且权重已演化。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared), max_iterations=2)
        first = scenario.train()
        assert first.code == 0, first.stderr
        assert [event["iteration"] for event in scenario.iter_events()] == [0, 1]
        checkpoints = scenario.run_dir() / "checkpoints"
        policy_ckpt = checkpoints / "policy_iter2.pt"
        assert policy_ckpt.is_file()
        assert (checkpoints / "discriminator_iter2.pt").is_file()
        config = ConfigLoader.load(scenario.config_path())
        reloaded = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=policy_ckpt,
        ))
        initial = torch.load(
            config.artifacts.unet_ckpt, map_location="cpu",
        )
        assert any(
            not torch.equal(reloaded.state_dict()[name], value)
            for name, value in initial.items()
        )

        scenario.set_schedule(max_iterations=4)
        second = scenario.train(resume=True)
        assert second.code == 0, second.stderr
        assert [event["iteration"] for event in scenario.iter_events()] == [
            0, 1, 2, 3,
        ]
        # 续训后条件轮转沿同一序继续（恢复点 = iteration 2 → 条件索引 0）
        assert scenario.iter_events()[2]["modality"] == CONDITIONS[0]

    @pytest.mark.gpu  # 1 iteration 训练（大轮次口径）
    def test_resume_state_carries_condition_labelled_buffer(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """AC：续训分片含条件标记（回放缓冲的带标签 FIFO 契约在 MR 线
        贯通）——base 分区按每条件配额量产、条目带目标条件标签。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared), max_iterations=1)
        result = scenario.train()
        assert result.code == 0, result.stderr
        base = scenario.resume_state()["replay_buffer"]["base"]
        modalities = set(base["modalities"])
        assert modalities
        assert modalities <= set(CONDITIONS)

    @pytest.mark.gpu  # 1 iteration 训练（大轮次口径）
    def test_baseline_sampling_runs_without_monitoring_phase(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """监控相缺席不影响评测相的另两条路径：Baseline 采样与 RL 后重采
        （二者不消费参照影像库）照常在 MR 线落盘。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared), max_iterations=1,
                     baseline_samples=2)
        result = scenario.train()
        assert result.code == 0, result.stderr
        entries = scenario.manifest_entries()
        assert len(entries) == 2
        assert all(entry["baseline_sample"] for entry in entries)
        assert all(entry["resample_sample"] for entry in entries)


class TestConditionGateSwitch:
    """条件闸总开关（``reward.condition_gate_enabled``）。"""

    def _empty_whitelist_scenario(
        self, cli: CliSession, tmp_path: Path,
    ) -> MrTrainScenario:
        """门槛不可达 → 空白名单（复刻 MR-RATE 真实首跑的报告形态）。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={
            "pretrain_gate_auc": 0.99, "pretrain_max_steps": 1,
        })
        scenario.use(scenario.pretrain(prepared), max_iterations=2)
        assert scenario.whitelist() == []
        return scenario

    def test_empty_whitelist_is_refused_by_default(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """既定口径不回归：空白名单 → readiness gate 硬拒绝（报错含
        per-condition 实测值），run 目录不留半成品。"""
        scenario = self._empty_whitelist_scenario(cli, tmp_path)
        result = scenario.train()
        assert result.code != 0
        assert "白名单为空" in result.stderr
        assert f"AUC[{CONDITIONS[0]}]" in result.stderr
        assert not (scenario.run_dir() / "metrics.jsonl").exists()

    @pytest.mark.gpu  # 2 iteration 训练（大轮次口径）
    def test_disabled_gate_runs_every_iteration(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """关闸语义：空白名单也开跑，且 AUC 不作任何更新开关——每
        iteration 都做 policy 更新（policy_gated 全 False、loss 带
        policy_step），同时 AUC 照常测量落盘（观测面不退化）。"""
        scenario = self._empty_whitelist_scenario(cli, tmp_path)
        scenario.patch_config(reward={"condition_gate_enabled": False})
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.iter_events()
        assert len(events) == 2
        assert [event["policy_gated"] for event in events] == [False, False]
        for event in events:
            assert "policy_step_1" in event["loss"]
            assert "discriminator" in event["loss"]
            assert 0.0 <= event["heldout_auc"] <= 1.0

    @pytest.mark.gpu  # 4 iteration 训练（大轮次口径与既有门控测试一致）
    def test_narrow_whitelist_skips_policy_update_only(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """既定口径的门控语义在 MR 线贯通：名单外条件的 iteration 跳过
        policy 更新（loss 无 policy_step_*、事件带 policy_gated），
        rollout / fake 入 buffer / 判别器更新 / AUC 观测照常；静态白名单
        （动态恢复关闭）下名单逐位恒定并随续训分片落盘。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        scenario.use(scenario.pretrain(prepared), max_iterations=4)
        scenario.narrow_whitelist([CONDITIONS[0]])
        scenario.patch_config(reward={"gating_dynamic_recovery": False})
        result = scenario.train()
        assert result.code == 0, result.stderr
        events = scenario.iter_events()
        assert len(events) == 4
        gated = [event for event in events if event["policy_gated"]]
        assert gated  # 名单外条件 2/4：4 iteration 至少一个 gated
        for event in gated:
            assert event["modality"] != CONDITIONS[0]
            assert not [
                key for key in event["loss"] if key.startswith("policy_step")
            ]
            assert "discriminator" in event["loss"]
            assert 0.0 <= event["heldout_auc"] <= 1.0
            assert event["buffer_current_fraction"] > 0
        # fake 入近期分区不受门控影响（滚动照常）
        assert (
            events[-1]["buffer_recent_occupied"]
            > events[0]["buffer_recent_occupied"]
        )
        assert scenario.resume_state()["gating"]["members"] == [CONDITIONS[0]]


class TestMonitoringPhaseAbsence:
    """监控相缺席与在场的装配分界（#123 的结构面）。"""

    def test_milestone_reachable_run_still_requires_reference_library(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """里程碑可达的 MR run 仍显式拒绝（MR 参照影像库属 #124）：无
        触发点的 run 不装配监控相，有触发点的 run 保持原守卫——拒绝面
        不因本票放宽。"""
        scenario = MrTrainScenario(cli, tmp_path)
        prepared = scenario.prepare(reward_overrides={"pretrain_gate_auc": 0.01})
        pretrained = scenario.pretrain(prepared)
        scenario.use(pretrained)
        scenario.set_schedule(
            max_iterations=pretrained.schedule.milestone_interval,
        )
        result = scenario.train()
        assert result.code != 0
        assert "里程碑参照影像库尚未交付" in result.stderr


class TestCostReadingShape:
    """逐 iter 卡时分解的相位命名契约（#123 AC3 的取数面）。"""

    def test_phase_timer_marks_named_phases(self) -> None:
        """``PhaseTimer`` 的边界语义：逐次 mark 产出对应相位、值为非负
        秒数，未打点的相位不出现。"""
        timer = PhaseTimer()
        timer.mark("rollout")
        timer.mark("policy_update")
        assert set(timer.marks) == {"rollout", "policy_update"}
        assert all(value >= 0.0 for value in timer.marks.values())
