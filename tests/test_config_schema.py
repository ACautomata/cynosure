"""config schema 校验测试：配置项清单全量字段落 schema、状态标注完备、
定死值不可改、跨字段定死语义（数值锚、奇异端、组2/组3 ControlNet）。"""

import copy
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from cynosure.config import (
    ConfigLoader,
    CynosureConfig,
)
from tests.conftest import CROSS_MODAL_PAIRS, MINIMAL_CONFIG_DICT


class TestValidConfigs:
    def test_minimal_config_passes(self, valid_config_dict: dict) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        assert config.experiment.group == "modal-label"

    def test_spec_default_values(self, valid_config_dict: dict) -> None:
        """配置项清单的默认值钉进 schema。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        assert config.latent_shape == (4, 64, 64, 32)
        assert config.policy.num_inference_steps == 30
        assert config.policy.input_img_size_numel == 131072
        assert config.policy.group_size_g == 12
        assert config.policy.sde_eta == pytest.approx(0.7)
        assert config.policy.sde_s_max == pytest.approx(0.999)
        assert config.policy.train_step_indices_m == set(range(2, 16))
        assert config.policy.granularity_intervals_lambda == {1, 2}
        assert config.policy.ratio_clip == pytest.approx(1e-4)
        assert config.policy.policy_lr == pytest.approx(2e-6)
        assert config.policy.policy_weight_decay == pytest.approx(1e-4)
        assert config.grpo.advantage_clamp == pytest.approx(5.0)
        assert config.grpo.kl_beta == 0.0
        assert config.reward.disc_num_layers_d == 2
        assert config.reward.patch_aggregation == "mean"
        assert config.reward.disc_update_interval_n_d == 1
        assert config.reward.disc_lr == pytest.approx(5e-5)
        assert config.reward.replay_current_fraction == pytest.approx(0.5)
        # 预训练与 RM readiness gate（ADR-0007）：阈值暂定 0.65、卫生项同
        # policy 侧口径（1e-4）
        assert config.reward.disc_weight_decay == pytest.approx(1e-4)
        assert config.reward.disc_weight_decay == config.policy.policy_weight_decay
        assert config.reward.pretrain_gate_auc == pytest.approx(0.65)
        # 支撑度界（ADR-0008 决策 6）：条件 held-out 卷数 < 此界走 bootstrap
        # CI 下界口径，≥ 界点估计口径——暂定 20 待 MR-RATE 曲线校准
        assert config.reward.gate_support_min_volumes == 20
        assert config.reward.pretrain_max_steps >= 1
        assert config.reward.pretrain_fake_batch >= 1
        # 动态门控（ADR-0008 决策 8，issue #89）：默认开启，enter/exit/
        # EMA 跨度三 knob 暂定值——MR-RATE 预训练曲线校准后定版
        assert config.reward.gating_dynamic_recovery is True
        assert config.reward.gating_enter_auc == pytest.approx(0.55)
        assert config.reward.gating_exit_auc == pytest.approx(0.52)
        assert config.reward.gating_ema_span == 8
        # 训练期噪声注入（ADR-0009 决策 3，issue #104）：σ_max 暂定 0.2——
        # MR-RATE 预训练曲线校准后定版；σ_max = 0 是唯一关闭形态
        assert config.reward.disc_noise_sigma_max == pytest.approx(0.2)
        # 过拟合分叉监控（ADR-0009 决策 4/5，issue #105）：EMA 跨度与
        # ADR-0008 的 EMA(AUC) 跨度同值口径（8）、报警阈值暂定 0.2——
        # MR-RATE 预训练曲线校准后定版
        assert config.reward.overfit_ema_span == 8
        assert config.reward.overfit_alert_divergence == pytest.approx(0.2)
        assert config.schedule.n_plateau == 3
        assert config.schedule.milestone_interval == 50
        assert config.schedule.checkpoint_interval == 10
        # 日程默认面与生产守卫自洽（纯默认即合法）：N_baseline /
        # max_iterations 缺省 = 生产口径下界，milestone_eval_samples
        # 缺省 12 = 组2/组3 词汇表宽（K ≥ 词汇表守卫全组生效）
        assert config.schedule.baseline_samples == 200
        assert config.schedule.max_iterations == 200
        assert config.schedule.milestone_eval_samples == 12
        assert config.sharding.strategy == "fsdp"
        # 部署行（orchestration + ADR-0005）：单实例 4 卡、产物根在持久分区下
        assert config.deployment.nproc_per_node == 4
        assert config.deployment.output_root == Path("/root/private_data/cynosure")

    def test_default_schedule_validates_for_every_group(self) -> None:
        """纯默认 schedule（只填 seed）在三组全合法——默认面与生产守卫
        自洽：milestone_eval_samples 缺省 12 覆盖组1 四序列与组2/组3
        12 有序对（亦覆盖 MR-RATE 11 条件的装配期同款下界），N_baseline /
        max_iterations 缺省即生产口径。默认 config 不应撞上自己的 schema
        守卫（#123 首跑曾以缩小值踩中同类拒绝、train 集体 exit 2）。"""
        group_artifacts = {
            "modal-label": {},
            "cross-modal": {
                "controlnet_ckpt": "ckpts/controlnet.pt",
                "controlnet_config_json": "configs/controlnet.json",
            },
            "sequential": {
                "controlnet_ckpt": "ckpts/controlnet.pt",
                "controlnet_config_json": "configs/controlnet.json",
            },
        }
        for group, extra_artifacts in group_artifacts.items():
            data = copy.deepcopy(MINIMAL_CONFIG_DICT)
            data["experiment"]["group"] = group
            data["artifacts"].update(extra_artifacts)
            if group == "sequential":  # stage-2 报告绑定（#116 schema 必填）
                data["experiment"]["stage2_pretrain_report_json"] = (
                    "pretrain_run_stage2/pretrain_report.json"
                )
            config = CynosureConfig.model_validate(data)
            assert config.schedule.milestone_eval_samples == 12
            assert config.schedule.baseline_samples == 200
            assert config.schedule.max_iterations == 200

    def test_cross_modal_pairs_default_is_ordered_12(self) -> None:
        config = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
        pairs = {(src, tgt) for src, tgt in config.experiment.cross_modal_pairs}
        expected = {(s, t) for s, t in map(tuple, CROSS_MODAL_PAIRS)}
        assert pairs == expected
        assert len(config.experiment.cross_modal_pairs) == 12

    def test_json_roundtrip_preserves_sets(self, valid_config_dict: dict) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        revived = CynosureConfig.model_validate_json(config.model_dump_json())
        assert revived.policy.train_step_indices_m == config.policy.train_step_indices_m
        assert revived == config

    def test_load_config_from_file(self, valid_config_json: Path) -> None:
        config = ConfigLoader.load(valid_config_json)
        assert config.experiment.group == "modal-label"

    def test_fixture_mode_allows_reduced_schedule(self, valid_config_dict: dict) -> None:
        """fixture_mode=true 显式声明后才允许缩小采样日程（spec「Fixture 策略」）。"""
        data = copy.deepcopy(valid_config_dict)
        data["fixture_mode"] = True
        data["latent_shape"] = [4, 16, 16, 8]
        data["policy"] = {
            "num_inference_steps": 3,
            "input_img_size_numel": math.prod((16, 16, 8)),
            "train_step_indices_m": [1],
        }
        config = CynosureConfig.model_validate(data)
        assert config.fixture_mode is True
        assert config.policy.num_inference_steps == 3


class TestRejection:
    @staticmethod
    def _locations(exc: ValidationError) -> list[tuple]:
        return [err["loc"] for err in exc.errors()]

    @staticmethod
    def _locations_with_messages(exc: ValidationError) -> list[tuple[tuple, str]]:
        return [(err["loc"], err["msg"]) for err in exc.errors()]

    def test_overfit_alert_threshold_outside_unit_interval_rejected(
        self, valid_config_dict: dict,
    ) -> None:
        """分叉报警阈值（ADR-0009 决策 5）须在开区间 (0,1)：0 = 任何正
        分叉即报警（测量噪声淹没报警面）、1 = 永不越线（哑区），均不合法。"""
        for bad in (0.0, -0.2, 1.0, 1.5):
            data = copy.deepcopy(valid_config_dict)
            data["reward"]["overfit_alert_divergence"] = bad
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("reward", "overfit_alert_divergence") in self._locations(
                exc_info.value,
            )

    def test_overfit_ema_span_rejects_non_positive(
        self, valid_config_dict: dict,
    ) -> None:
        """分叉 EMA 跨度（ADR-0009 决策 4）须为正整数（0 会除零、负数
        无平滑语义）。"""
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["overfit_ema_span"] = 0
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward", "overfit_ema_span") in self._locations(exc_info.value)

    def test_missing_required_field_is_field_level_error(self) -> None:
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        del data["experiment"]["group"]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "group") in self._locations(exc_info.value)

    def test_missing_pretrain_report_path_is_field_level_error(self) -> None:
        """预训练产物路径必填无默认（ADR-0007：RL 不带 warm-start 工件在
        schema 层就无法启动）。"""
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        del data["reward"]["pretrain_report_json"]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward", "pretrain_report_json") in self._locations(exc_info.value)

    def test_unknown_field_rejected(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["polic"] = {}  # 拼错段名
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "polic" in str(exc_info.value.errors()[0]["loc"])

    def test_support_bound_rejects_non_positive(self, valid_config_dict: dict) -> None:
        """支撑度界（ADR-0008 决策 6）：非正整数显式字段级拒绝
        （0 或负界会让全部条件无条件走点估计口径，规则形同虚设）。"""
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["gate_support_min_volumes"] = 0
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward", "gate_support_min_volumes") in self._locations(
            exc_info.value,
        )

    def test_support_bound_override_allowed(self, valid_config_dict: dict) -> None:
        """支撑度界可配置（tunable）：校准期改值合法（暂定值非定死）。"""
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["gate_support_min_volumes"] = 8
        config = CynosureConfig.model_validate(data)
        assert config.reward.gate_support_min_volumes == 8

    def test_group_enum_rejects_unknown(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "mask-conditioned"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "group") in self._locations(exc_info.value)

    def test_fixed_value_cannot_be_changed(self, valid_config_dict: dict) -> None:
        """定死项（ratio clip 1e-4）改值 → 字段级拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"ratio_clip": 1e-3}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "ratio_clip") in self._locations(exc_info.value)

    def test_fixed_reward_mode_cannot_be_changed(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["reward_mode"] = "sigmoid_prob"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward", "reward_mode") in self._locations(exc_info.value)

    def test_latent_channel_is_fixed_at_4(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["latent_shape"] = [8, 64, 64, 32]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert any("latent" in str(loc) for loc in self._locations(exc_info.value))

    def test_input_img_size_numel_must_match_latent_shape(self, valid_config_dict: dict) -> None:
        """数值锚：input_img_size_numel 必须 == prod(latent_shape[1:])，防日程静默错位。"""
        data = copy.deepcopy(valid_config_dict)
        data["latent_shape"] = [4, 16, 16, 8]
        # 默认 131072 与新 latent 不符 → 拒绝
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert any(
            ("policy",) == loc and "input_img_size_numel" in msg
            for loc, msg in self._locations_with_messages(exc_info.value)
        )
        # 同语义值 2048 → 通过
        data["policy"] = {"input_img_size_numel": math.prod((16, 16, 8))}
        config = CynosureConfig.model_validate(data)
        assert config.policy.input_img_size_numel == 2048

    def test_train_steps_m_exclude_singular_end(self, valid_config_dict: dict) -> None:
        """M 沿 timesteps 数组下标、0=最噪端：0 与末步必须排除（s≈1 奇异端 / 无续跑空间）。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"train_step_indices_m": [0, 2, 3]}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "train_step_indices_m") in self._locations(exc_info.value)

        data["policy"] = {"train_step_indices_m": [29]}  # 30 步日程的末下标
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "train_step_indices_m") in self._locations(exc_info.value)

    def test_train_steps_m_reject_negative_index(self, valid_config_dict: dict) -> None:
        """M 沿 timesteps 数组取下标（0=最噪端）：负下标无意义，必须拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"train_step_indices_m": [-1, 2]}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "train_step_indices_m") in self._locations(exc_info.value)

    def test_train_steps_m_out_of_schedule(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"num_inference_steps": 3, "input_img_size_numel": 131072,
                          "train_step_indices_m": [7]}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "train_step_indices_m") in self._locations(exc_info.value)

    def test_granularity_lambda_positive(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"granularity_intervals_lambda": [0, 1]}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "granularity_intervals_lambda") in self._locations(exc_info.value)

    def test_granularity_lambda_ablation_axis(self, valid_config_dict: dict) -> None:
        """Λ 消融取值 = {1,2} 或 {1,2,3} 完整集合（policy-modeling 章），其余拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"granularity_intervals_lambda": [5]}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("policy", "granularity_intervals_lambda") in self._locations(exc_info.value)
        for partial in ([1], [2], [1, 3]):  # 子集不是完整消融集（MGAI 可比性）
            data["policy"] = {"granularity_intervals_lambda": partial}
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert (
                ("policy", "granularity_intervals_lambda")
                in self._locations(exc_info.value)
            )
        data["policy"] = {"granularity_intervals_lambda": [1, 2, 3]}
        CynosureConfig.model_validate(data)  # 消融轴另一端合法

    def test_production_steps_fixed_without_fixture_mode(
        self, valid_config_dict: dict,
    ) -> None:
        """生产 config（fixture_mode 缺省 = false）下 num_inference_steps 定死 30：typo 静默改采样场必须拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"num_inference_steps": 29}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "num_inference_steps" in str(exc_info.value.errors())

    def test_artifacts_declare_source_dataset_root(self, valid_config_dict: dict) -> None:
        """prepare 的输入 = 原始影像 + VAE：源数据集根目录是必填工件路径（experiment-design「real 样本库」）。"""
        data = copy.deepcopy(valid_config_dict)
        data["artifacts"]["dataset_root"] = "data/brats2023"
        config = CynosureConfig.model_validate(data)
        assert config.artifacts.dataset_root == Path("data/brats2023")

    def test_missing_dataset_root_is_field_level_error(self) -> None:
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        del data["artifacts"]["dataset_root"]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("artifacts", "dataset_root") in self._locations(exc_info.value)

    def test_baseline_samples_in_spec_range(self, valid_config_dict: dict) -> None:
        """N_baseline 生产口径 200–500（experiment-design 章）；fixture_mode
        显式声明后放宽（Baseline manifest 条目随 fixture 全流程走）。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"].update({"baseline_samples": 100})
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "baseline_samples" in str(exc_info.value.errors())
        data["fixture_mode"] = True
        config = CynosureConfig.model_validate(data)  # fixture 显式声明后合法
        assert config.schedule.baseline_samples == 100
        data["fixture_mode"] = False
        data["schedule"].update({"baseline_samples": 500})
        CynosureConfig.model_validate(data)

    def test_early_stop_params_schema(self, valid_config_dict: dict) -> None:
        """早停参数落 config schema（AC：里程碑间隔/N_plateau/早停参数）。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"].update({
            "milestone_interval": 25,
            "milestone_eval_samples": 4,
            "n_plateau": 2,
            "plateau_tolerance": 0.1,
            "auc_chance_epsilon": 0.05,
            "reward_trend_window": 20,
        })
        config = CynosureConfig.model_validate(data)
        assert config.schedule.milestone_interval == 25
        assert config.schedule.milestone_eval_samples == 4
        assert config.schedule.n_plateau == 2
        assert config.schedule.plateau_tolerance == pytest.approx(0.1)
        assert config.schedule.auc_chance_epsilon == pytest.approx(0.05)
        assert config.schedule.reward_trend_window == 20

    def test_milestone_eval_samples_needs_at_least_two(
        self, valid_config_dict: dict,
    ) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["schedule"] = {"milestone_eval_samples": 1}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("schedule", "milestone_eval_samples") in self._locations(exc_info.value)

    def test_milestone_samples_cover_condition_support(
        self, valid_config_dict: dict,
    ) -> None:
        """生产 config 下里程碑评测样本数须覆盖本组条件词汇表：manifest
        条目按条件轮转，K < 词汇表即永久漏掉尾部方向（12 有序对只评
        前 K 个，早停判据对其余方向失明）。fixture 豁免（条目数随
        fixture 缩小）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 8
        with pytest.raises(ValidationError) as exc_info:  # 显式 K=8 < 12 对
            CynosureConfig.model_validate(data)
        assert "milestone_eval_samples" in str(exc_info.value.errors())
        data["schedule"]["milestone_eval_samples"] = 12
        CynosureConfig.model_validate(data)  # 覆盖 12 有序对 → 合法
        data["fixture_mode"] = True
        data["schedule"]["milestone_eval_samples"] = 4
        CynosureConfig.model_validate(data)  # fixture 豁免

    def test_milestone_samples_cover_modal_label_vocabulary(
        self, valid_config_dict: dict,
    ) -> None:
        """组1 词汇表 = 四序列：生产 K < 4 拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"]["milestone_eval_samples"] = 3
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "milestone_eval_samples" in str(exc_info.value.errors())

    def test_milestone_samples_cover_sequential_stage2_vocabulary(
        self, valid_config_dict: dict,
    ) -> None:
        """组3 stage-2 词汇表 = 12 有序对（取两阶段词汇表最大值）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "sequential"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        # 先补齐 stage-2 报告绑定（#116 schema 必填）：缺绑定的错误在
        # 字段层先行短路，会掩盖本用例针对的 milestone 词汇表错误
        data["experiment"]["stage2_pretrain_report_json"] = (
            "pretrain_run_stage2/pretrain_report.json"
        )
        data["schedule"]["milestone_eval_samples"] = 8
        with pytest.raises(ValidationError) as exc_info:  # 显式 K=8 < 12 对
            CynosureConfig.model_validate(data)
        assert "milestone_eval_samples" in str(exc_info.value.errors())
        data["schedule"]["milestone_eval_samples"] = 12
        CynosureConfig.model_validate(data)

    def test_milestone_samples_within_baseline_manifest(
        self, valid_config_dict: dict,
    ) -> None:
        """生产 config 下 K 不得超过 N_baseline：评测条目取 manifest 前缀，
        超出即静默缩水到盘上条目数、评测面与配置声明失真。fixture 豁免。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"].update({"milestone_eval_samples": 201})
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "milestone_eval_samples" in str(exc_info.value.errors())
        data["schedule"]["milestone_eval_samples"] = 200
        CynosureConfig.model_validate(data)  # = N_baseline → 合法
        data["fixture_mode"] = True
        data["schedule"]["milestone_eval_samples"] = 500
        CynosureConfig.model_validate(data)  # fixture 豁免

    def test_auc_chance_epsilon_below_half(self, valid_config_dict: dict) -> None:
        """AUC 近 chance 判定带半径 ≥0.5 即恒真：hacking 签名失去判别力，拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"] = {"auc_chance_epsilon": 0.5}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("schedule", "auc_chance_epsilon") in self._locations(exc_info.value)

    def test_vae_and_radimagenet_artifact_fields(self, valid_config_dict: dict) -> None:
        """评测路径的新工件位：VAE 网络配置 JSON 与 RadImageNet 权重路径
        （可缺省——装配时按用途显式拒绝），schema 可承载。"""
        data = copy.deepcopy(valid_config_dict)
        data["artifacts"]["vae_config_json"] = "configs/vae.json"
        data["artifacts"]["radimagenet_weights"] = "weights/radimagenet_resnet50.pth"
        config = CynosureConfig.model_validate(data)
        assert config.artifacts.vae_config_json == Path("configs/vae.json")
        assert config.artifacts.radimagenet_weights == Path(
            "weights/radimagenet_resnet50.pth",
        )

    def test_max_iterations_capped_at_spec_upper(self, valid_config_dict: dict) -> None:
        """每组规模目标 200–500；50 sanity 合法、超出 500 拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"].update({"max_iterations": 50})
        CynosureConfig.model_validate(data)  # sanity 运行合法
        data["schedule"].update({"max_iterations": 501})
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("schedule", "max_iterations") in self._locations(exc_info.value)

    def test_cross_modal_requires_controlnet_ckpt(self, valid_config_dict: dict) -> None:
        """组2/组3 的训练对象含 ControlNet：无 checkpoint 即拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert any("controlnet" in msg for loc, msg in self._locations_with_messages(exc_info.value))

        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 12  # 覆盖 12 有序对
        config = CynosureConfig.model_validate(data)
        assert config.experiment.group == "cross-modal"

    def test_stage2_groups_require_controlnet_network_config(
        self, valid_config_dict: dict,
    ) -> None:
        """组2/组3 的 ControlNet 装配源 = checkpoint + 网络配置 JSON 两者
        （netbuild 按 artifact 构建契约，与 UNet 同构）：缺网络配置即拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert any(
            "controlnet_config_json" in msg
            for loc, msg in self._locations_with_messages(exc_info.value)
        )

    def test_sequential_group_minimal_config(self, valid_config_dict: dict) -> None:
        """组3 最小合法 config：stage1_run_dir 缺省 = 同一次运行内先跑 stage-1
        （spec 配置项清单「组3 衔接」行的两个分支之一）；stage-2 报告绑定
        必填（#116）——缺省即装载期拒绝，不静默继承 stage-1 报告。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "sequential"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 12  # 覆盖 stage-2 有序对
        with pytest.raises(ValidationError) as exc_info:  # 未绑定 stage-2 报告
            CynosureConfig.model_validate(data)
        assert ("experiment", "stage2_pretrain_report_json") in self._locations(
            exc_info.value,
        )
        assert "stage2_pretrain_report_json" in str(
            exc_info.value.errors()[0]["msg"],
        )  # 指引配置面
        assert "stage-1" in str(exc_info.value.errors()[0]["msg"])  # 指引继承面
        data["experiment"]["stage2_pretrain_report_json"] = (
            "pretrain_run_stage2/pretrain_report.json"
        )
        config = CynosureConfig.model_validate(data)
        assert config.experiment.group == "sequential"
        assert config.experiment.stage1_run_dir is None
        assert config.experiment.stage2_pretrain_report_json == Path(
            "pretrain_run_stage2/pretrain_report.json",
        )

    def test_stage2_report_binding_only_valid_for_sequential(
        self, valid_config_dict: dict,
    ) -> None:
        """stage-2 报告绑定只对组3 有语义：非序贯组携带即拒绝（与
        stage1_run_dir 同款——拼错组名时静默绑定比显式拒绝危险）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 12  # 覆盖 12 有序对
        data["experiment"]["stage2_pretrain_report_json"] = (
            "pretrain_run_stage2/pretrain_report.json"
        )
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "stage2_pretrain_report_json") in self._locations(
            exc_info.value,
        )

    def test_stage1_run_dir_only_valid_for_sequential(
        self, valid_config_dict: dict,
    ) -> None:
        """既有 stage-1 产物路径只对组3 有语义：组1/组2 config 携带即拒绝
        （拼错组名时静默跳过 stage-1 比显式拒绝危险）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "modal-label"
        data["experiment"]["stage1_run_dir"] = "runs/stage1"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "stage1_run_dir") in self._locations(exc_info.value)

        data["experiment"]["group"] = "sequential"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 12  # 覆盖 stage-2 有序对
        # 组3 config 须绑定 stage-2 报告（#116 schema 必填，本用例补齐后
        # 验证 stage1_run_dir 的组3 合法形态）
        data["experiment"]["stage2_pretrain_report_json"] = (
            "pretrain_run_stage2/pretrain_report.json"
        )
        config = CynosureConfig.model_validate(data)
        assert config.experiment.stage1_run_dir == Path("runs/stage1")

    def test_source_latent_scale_factor_default(self, valid_config_dict: dict) -> None:
        """组2 双条件之一：ControlNet 条件 = 源影像 latent × scale_factor；
        fixture 中性默认 1.0（生产随基座 ControlNet 推理 config 核对）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["schedule"]["milestone_eval_samples"] = 12  # 覆盖 12 有序对
        config = CynosureConfig.model_validate(data)
        assert config.policy.source_latent_scale_factor == pytest.approx(1.0)

    def test_cross_modal_pairs_must_be_the_ordered_12(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        data["experiment"]["cross_modal_pairs"] = [["t1n", "t1c"]] * 12  # 重复对
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "cross_modal_pairs") in self._locations(exc_info.value)

    def test_discriminator_depth_ablation_axis(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["disc_num_layers_d"] = 3
        with pytest.raises(ValidationError):
            CynosureConfig.model_validate(data)
        data["reward"]["disc_num_layers_d"] = 1
        CynosureConfig.model_validate(data)  # 消融轴 {1,2} 两端皆合法

    def test_group_size_g_at_least_two(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"group_size_g": 1}
        with pytest.raises(ValidationError):
            CynosureConfig.model_validate(data)

    def test_eta_non_negative(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"sde_eta": -0.1}
        with pytest.raises(ValidationError):
            CynosureConfig.model_validate(data)

    def test_s_max_below_one(self, valid_config_dict: dict) -> None:
        """s_max 钳制 σ→1 奇异点：必须严格小于 1。"""
        data = copy.deepcopy(valid_config_dict)
        data["policy"] = {"sde_s_max": 1.0}
        with pytest.raises(ValidationError):
            CynosureConfig.model_validate(data)

    def test_noise_sigma_non_negative(self, valid_config_dict: dict) -> None:
        """disc_noise_sigma_max 非负（ADR-0009）：σ_max = 0 是唯一关闭
        形态（回归锚），负值无语义、拒绝而非静默钳零。"""
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["disc_noise_sigma_max"] = -0.1
        with pytest.raises(ValidationError):
            CynosureConfig.model_validate(data)
        data["reward"]["disc_noise_sigma_max"] = 0.0
        CynosureConfig.model_validate(data)  # 零强度合法：唯一关闭形态

    def test_sbatch_fields_are_gone_after_platform_migration(
        self, valid_config_dict: dict,
    ) -> None:
        """ADR-0005：sbatch 专属字段（partition/gres 等）随平台迁移删除，拼入即拒。"""
        data = copy.deepcopy(valid_config_dict)
        data["deployment"] = {"partition": "hx1hdnormal", "gres": "dcu:4"}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert any(
            ("deployment",) == loc[:1] and "partition" in str(loc)
            for loc in self._locations(exc_info.value)
        )

    def test_old_slurm_section_is_rejected(self, valid_config_dict: dict) -> None:
        """旧平台的 slurm 段必须显式拒绝：旧 config 不能静默通过迁移。"""
        data = copy.deepcopy(valid_config_dict)
        data["slurm"] = {"partition": "hx1hdnormal"}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "slurm" in str(exc_info.value.errors()[0]["loc"])
        assert "slurm" not in CynosureConfig.model_fields


class TestPreprocessingSchema:
    """prepare 读图编码的 recipe 参数（issue #45；data-preparation + ADR-0006）：
    resize 基数生产钉上游 128，fixture 经 fixture_mode 声明后可注入小基数。"""

    def test_resize_base_defaults_to_upstream(self, valid_config_dict: dict) -> None:
        config = CynosureConfig.model_validate(valid_config_dict)
        assert config.preprocessing.resize_base == 128

    def test_production_resize_base_fixed_without_fixture_mode(
        self, valid_config_dict: dict,
    ) -> None:
        """生产 config（fixture_mode 缺省 = false）下 resize_base 钉上游 128：
        静默偏离上游 recipe 等于换基座训练分布（ADR-0006），必须拒绝。"""
        data = copy.deepcopy(valid_config_dict)
        data["preprocessing"] = {"resize_base": 64}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "resize_base" in str(exc_info.value.errors())

    def test_fixture_mode_allows_injected_small_base(
        self, valid_config_dict: dict,
    ) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["fixture_mode"] = True
        data["preprocessing"] = {"resize_base": 16}
        config = CynosureConfig.model_validate(data)
        assert config.preprocessing.resize_base == 16

    def test_resize_base_must_be_positive(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["fixture_mode"] = True
        data["preprocessing"] = {"resize_base": 0}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("preprocessing", "resize_base") in self._locations(exc_info.value)

    def test_encode_sliding_window_defaults_anchor_nvidia(
        self, valid_config_dict: dict,
    ) -> None:
        """encode 滑窗参数锚 NVIDIA（#143）：roi [320,320,160] 影像空间、
        overlap 0.4（create_training_data 锚，T12 复核探针复核后交付）。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        assert config.preprocessing.encode_roi_size == [320, 320, 160]
        assert config.preprocessing.encode_overlap == 0.4

    def test_encode_overlap_out_of_range_rejected(
        self, valid_config_dict: dict,
    ) -> None:
        """encode overlap 域 [0, 1)：越界（≥1 或负）在 schema 拒绝。"""
        for bad in (1.0, -0.1, 1.5):
            data = copy.deepcopy(valid_config_dict)
            data["preprocessing"] = {"encode_overlap": bad}
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("preprocessing", "encode_overlap") in self._locations(
                exc_info.value,
            )

    @staticmethod
    def _locations(exc: ValidationError) -> list[tuple]:
        return [err["loc"] for err in exc.errors()]


class TestMrRateConditionVocabularyBinding:
    """MR-RATE 条件词汇表工件绑定（issue #127，#119 换域口径的工件化
    形态）：词表本体（五模态 token 映射 / 11 生成条件五元组）已迁出
    config——唯一来源是工件文件（``data/conditions/mrrate_conditions.json``，
    装载面 ``cynosure.conditions``）；config 只携带路径
    （``artifacts.condition_vocabulary_json``）并经 ``dataset`` 互斥绑定：
    MR-RATE 必填、BraTS 携带即拒（两套口径互斥激活的守卫面不变）。

    词表数值的定死对账（token 上游权威、11 条件与普查期望网格一致）由
    ``tests/test_condition_vocabulary.py`` 在工件装载面守卫。
    """

    @staticmethod
    def _locations(exc: ValidationError) -> list[tuple]:
        return [err["loc"] for err in exc.errors()]

    @staticmethod
    def _mr_config_dict(valid_config_dict: dict) -> dict:
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["dataset"] = "MR-RATE"
        # 单域锚字段不随 MR config 携带（#129 互斥携带守卫：latent 形状
        # 与数值锚逐条件派生自词汇表工件）
        data.pop("latent_shape", None)
        # 词汇表工件绑定（schema 强制）：路径在 config 层不可缺席——
        # 词表内容的普查对账在 MrConditionVocabulary 装载期
        data["artifacts"]["condition_vocabulary_json"] = (
            "data/conditions/mrrate_conditions.json"
        )
        # MR prepare 装配输入工件四件套 + 抽样留痕 + 强度臂（#121/#131
        # schema 必填面；装配绑定语义由 TestMrRateDataAssemblyBinding 守卫）
        data["artifacts"]["mrrate_metadata_csv"] = "mrrate/metadata.csv"
        data["artifacts"]["mrrate_splits_csv"] = "mrrate/splits.csv"
        data["artifacts"]["eval_manifest_csv"] = (
            "data/eval/mrrate-baseline/eval_manifest.csv"
        )
        data["artifacts"]["mrrate_data_snapshot"] = "MR-RATE@v1.0"
        data["reward"]["sampling_manifest_json"] = "prepare/sampling_manifest.json"
        data.setdefault("preprocessing", {})["intensity_clip"] = False
        return data

    def test_mrrate_config_passes_with_vocabulary_binding(
        self, valid_config_dict: dict,
    ) -> None:
        config = CynosureConfig.model_validate(
            self._mr_config_dict(valid_config_dict),
        )
        assert config.experiment.dataset == "MR-RATE"
        assert config.artifacts.condition_vocabulary_json == Path(
            "data/conditions/mrrate_conditions.json",
        )

    def test_mrrate_config_loads_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(self._mr_config_dict(MINIMAL_CONFIG_DICT)))
        config = ConfigLoader.load(path)
        assert config.experiment.dataset == "MR-RATE"
        assert config.artifacts.condition_vocabulary_json is not None

    def test_mrrate_requires_condition_vocabulary(
        self, valid_config_dict: dict,
    ) -> None:
        """MR config 缺词汇表工件绑定 → 字段级拒绝：11 生成条件的取数
        域无从装配，静默缺省比显式拒绝危险（条件词表是域定义本体）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["dataset"] = "MR-RATE"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("artifacts",) in self._locations(exc_info.value)
        assert "condition_vocabulary_json" in str(exc_info.value)

    def test_brats_rejects_condition_vocabulary(
        self, valid_config_dict: dict,
    ) -> None:
        """BraTS config 携带 MR 词汇工件即拒绝：拼错 dataset 时两套口径
        静默共存比显式拒绝危险（同 stage1_run_dir 守卫哲学）。"""
        data = copy.deepcopy(valid_config_dict)
        data["artifacts"]["condition_vocabulary_json"] = (
            "data/conditions/mrrate_conditions.json"
        )
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("artifacts",) in self._locations(exc_info.value)
        assert "MR-RATE" in str(exc_info.value)

    def test_unknown_dataset_rejected(self, valid_config_dict: dict) -> None:
        """dataset 放宽为 str + 登记域守卫（#127 泛化形态）：未登记域
        字段级拒绝并列出已登记集——类型层 Literal 退役后拼错域名仍不
        会静默流入分派逻辑。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["dataset"] = "MR-RATE-V2"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "dataset") in self._locations(exc_info.value)
        assert "BraTS2023" in str(exc_info.value)

    def test_mrrate_supports_modal_label_only(self, valid_config_dict: dict) -> None:
        """MR-RATE 线只定义组1（地图 #67：上游无 MR ControlNet，跨模态/
        序贯是 BraTS 语义）——非组1 携带即拒绝，防 BraTS 条件语义静默错位。
        序贯分支补齐 stage-2 报告绑定（#116 字段层先行短路的同款处理），
        让 dataset 守卫成为显式拒绝面。"""
        for group in ("cross-modal", "sequential"):
            data = self._mr_config_dict(valid_config_dict)
            data["experiment"]["group"] = group
            if group == "sequential":
                data["experiment"]["stage2_pretrain_report_json"] = (
                    "pretrain_run_stage2/pretrain_report.json"
                )
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("experiment",) in self._locations(exc_info.value)
            assert "modal-label" in str(exc_info.value)

    def test_mrrate_json_roundtrip(self, valid_config_dict: dict) -> None:
        """工件绑定路径 dump → reload 逐位一致；BraTS dump 不含词汇
        绑定（行为不变锚；config 不内嵌词表——词表唯一来源是工件）。"""
        config = CynosureConfig.model_validate(
            self._mr_config_dict(valid_config_dict),
        )
        revived = CynosureConfig.model_validate_json(config.model_dump_json())
        assert revived == config
        brats = CynosureConfig.model_validate(copy.deepcopy(MINIMAL_CONFIG_DICT))
        dumped = json.loads(brats.model_dump_json())
        assert dumped["artifacts"]["condition_vocabulary_json"] is None
        assert dumped["experiment"]["dataset"] == "BraTS2023"


class TestMrRateDataAssemblyBinding:
    """MR-RATE prepare 数据装配的 config 绑定（#121/#131，spec #125 实现决策 3）。

    MR-RATE prepare 的输入工件四件套（series 级元数据 CSV / 官方 patient
    级 splits CSV / #78 评估清单 / 数据 release 快照标识）与装配 knobs
    （逐条件配额、held-out 二分配比、抽样留痕路径）经 dataset 互斥绑定：
    MR-RATE 必填、BraTS 携带即拒——同 condition_vocabulary_json 的两套
    口径互斥激活哲学。强度臂 clip 参数化（#130）：BraTS 臂 clip=True 是
    ADR-0006 裁决（fork recipe 偏差），MR-RATE 臂 clip=False 是 NVIDIA
    v1 官方口径（#71 裁决：对齐基座训练域）——两臂各自锁死裁决值，
    显式携带错误值即拒绝（静默换 recipe 比显式拒绝危险）。
    """

    MR_ARTIFACT_FIELDS = (
        "mrrate_metadata_csv",
        "mrrate_splits_csv",
        "eval_manifest_csv",
        "mrrate_data_snapshot",
    )

    @staticmethod
    def _locations(exc: ValidationError) -> list[tuple]:
        return [err["loc"] for err in exc.errors()]

    @classmethod
    def _mr_config_dict(cls, valid_config_dict: dict) -> dict:
        """MR-RATE prepare 全字段齐备的合法 config 样板。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["dataset"] = "MR-RATE"
        data["artifacts"]["condition_vocabulary_json"] = (
            "data/conditions/mrrate_conditions.json"
        )
        data["artifacts"]["mrrate_metadata_csv"] = "mrrate/metadata.csv"
        data["artifacts"]["mrrate_splits_csv"] = "mrrate/splits.csv"
        data["artifacts"]["eval_manifest_csv"] = (
            "data/eval/mrrate-baseline/eval_manifest.csv"
        )
        data["artifacts"]["mrrate_data_snapshot"] = "MR-RATE@v1.0"
        data["reward"]["sampling_manifest_json"] = "prepare/sampling_manifest.json"
        data.setdefault("preprocessing", {})["intensity_clip"] = False
        # 单域锚字段属 BraTS 语义（#129 互斥携带即拒）：形状/锚逐条件
        # 派生自条件词汇表工件
        data.pop("latent_shape", None)
        data.get("policy", {}).pop("input_img_size_numel", None)
        return data

    def test_mrrate_assembly_config_passes(
        self, valid_config_dict: dict,
    ) -> None:
        config = CynosureConfig.model_validate(
            self._mr_config_dict(valid_config_dict),
        )
        assert config.experiment.dataset == "MR-RATE"
        assert config.preprocessing.intensity_clip is False
        assert config.reward.heldout_fraction == pytest.approx(0.1)
        assert config.reward.real_pool_quota == {}

    def test_blank_data_snapshot_rejected(
        self, valid_config_dict: dict,
    ) -> None:
        """快照标识为空白串 → 拒绝：该字段的用途是「real 数据链与评估集
        同一冻结快照」的凭据，空白串满足 `is not None` 的必填检查后照常
        随 provenance 与抽样留痕落档——工件自称登记了数据 release、实际
        什么都没登记，凭据面静默失效（快照标识是标识，不是自由文本）。"""
        data = self._mr_config_dict(valid_config_dict)
        data["artifacts"]["mrrate_data_snapshot"] = "   "
        with pytest.raises(ValidationError) as exc:
            CynosureConfig.model_validate(data)
        assert (
            "artifacts", "mrrate_data_snapshot",
        ) in self._locations(exc.value)

    def test_data_snapshot_whitespace_normalized(
        self, valid_config_dict: dict,
    ) -> None:
        """快照标识的首尾空白归一后入 config：该值原样落进 provenance 与
        抽样留痕，是跨工件比对（prepare 与 #78 评估集同一快照）的键——
        带空白的 " f6e39794" 与 "f6e39794" 是两个不相等的字符串，
        比对静默判否。"""
        data = self._mr_config_dict(valid_config_dict)
        data["artifacts"]["mrrate_data_snapshot"] = "  f6e39794\n"
        config = CynosureConfig.model_validate(data)
        assert config.artifacts.mrrate_data_snapshot == "f6e39794"

    def test_mrrate_requires_assembly_artifacts(
        self, valid_config_dict: dict,
    ) -> None:
        """MR config 缺任一装配输入工件 → 字段级拒绝：配额抽样与互斥
        守卫的取数域无从装配（同 condition_vocabulary_json 守卫哲学）。"""
        for field in self.MR_ARTIFACT_FIELDS:
            data = self._mr_config_dict(valid_config_dict)
            data["artifacts"][field] = None
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("artifacts",) in self._locations(exc_info.value)
            assert field in str(exc_info.value)

    def test_mrrate_requires_sampling_manifest_path(
        self, valid_config_dict: dict,
    ) -> None:
        data = self._mr_config_dict(valid_config_dict)
        data["reward"]["sampling_manifest_json"] = None
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward",) in self._locations(exc_info.value)
        assert "sampling_manifest_json" in str(exc_info.value)

    def test_brats_rejects_assembly_artifacts(
        self, valid_config_dict: dict,
    ) -> None:
        """BraTS config 携带 MR 装配工件即拒绝：MR 输入工件指向不存在的
        布局，携带即口径混乱信号（两套口径互斥激活）。"""
        for field in self.MR_ARTIFACT_FIELDS:
            data = copy.deepcopy(valid_config_dict)
            data["artifacts"][field] = "mrrate/stub.csv"
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("artifacts",) in self._locations(exc_info.value)
            assert "MR-RATE" in str(exc_info.value)

    def test_brats_rejects_sampling_manifest_path(
        self, valid_config_dict: dict,
    ) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["reward"]["sampling_manifest_json"] = "prepare/sampling.json"
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward",) in self._locations(exc_info.value)
        assert "MR-RATE" in str(exc_info.value)

    def test_mrrate_rejects_clip_true(
        self, valid_config_dict: dict,
    ) -> None:
        """MR 臂 clip=True 拒绝：clip=False 是基座 v1 训练口径（#71/
        #130 裁决），clip=True 臂属 BraTS 线（fork 偏差）——两臂 embedding
        不可互用，混臂等于换基座训练分布。"""
        data = self._mr_config_dict(valid_config_dict)
        data["preprocessing"]["intensity_clip"] = True
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("preprocessing",) in self._locations(exc_info.value)
        assert "clip" in str(exc_info.value)

    def test_brats_rejects_clip_false(
        self, valid_config_dict: dict,
    ) -> None:
        """BraTS 臂 clip=False 拒绝：clip=True 是 ADR-0006 裁决的 fork
        recipe 锚——显式换 recipe 须重论证（ADR 层），schema 层先拒。"""
        data = copy.deepcopy(valid_config_dict)
        data.setdefault("preprocessing", {})["intensity_clip"] = False
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("preprocessing",) in self._locations(exc_info.value)
        assert "ADR-0006" in str(exc_info.value)

    def test_mrrate_rejects_resize_base(
        self, valid_config_dict: dict,
    ) -> None:
        """MR 线 resize 目标 = 逐条件统一网格（词汇表携带），resize 基数
        公式（BraTS 口径）无语义——显式携带即拒绝，防两套 resize 口径
        静默共存（哪一套在链里生效不可从工件判读）。"""
        data = self._mr_config_dict(valid_config_dict)
        data.setdefault("preprocessing", {})["resize_base"] = 16
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("preprocessing",) in self._locations(exc_info.value)
        assert "统一网格" in str(exc_info.value)

    def test_heldout_fraction_open_unit_interval(
        self, valid_config_dict: dict,
    ) -> None:
        """held-out 二分配比须在 (0,1) 开区间：0 = held-out 池恒空
        （out-of-sample 信号语义消失，同 CaseSplitter 空拒绝哲学）、
        ≥1 = real pool 恒空。"""
        for bad in (0.0, 1.0, -0.1, 1.5):
            data = self._mr_config_dict(valid_config_dict)
            data["reward"]["heldout_fraction"] = bad
            with pytest.raises(ValidationError) as exc_info:
                CynosureConfig.model_validate(data)
            assert ("reward", "heldout_fraction") in self._locations(
                exc_info.value,
            )

    def test_real_pool_quota_must_be_positive(
        self, valid_config_dict: dict,
    ) -> None:
        data = self._mr_config_dict(valid_config_dict)
        data["reward"]["real_pool_quota"] = {"t1w/axial": 0}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("reward", "real_pool_quota") in self._locations(exc_info.value)

    def test_mrrate_assembly_json_roundtrip(
        self, valid_config_dict: dict,
    ) -> None:
        config = CynosureConfig.model_validate(
            self._mr_config_dict(valid_config_dict),
        )
        revived = CynosureConfig.model_validate_json(config.model_dump_json())
        assert revived == config


class TestStatusAnnotations:
    ALL_MODELS = [
        CynosureConfig,
        *[
            f.annotation
            for f in CynosureConfig.model_fields.values()
            if hasattr(f.annotation, "model_fields")
        ],
        # Optional 嵌套（artifacts.condition_vocabulary_json 是 Path | None，
        # 无嵌套 model）不在上面的自动收集面，无需补齐（#127：MR 词表
        # 模型已迁出 config，其字段标注由 cynosure.conditions 自带
        # pydantic Field 描述）
    ]

    def test_every_field_has_status_and_source(self) -> None:
        """配置项清单全量字段 + 状态标注落 schema：任何字段都不得缺 status/source。"""
        missing = []
        for model in self.ALL_MODELS:
            for name, field in model.model_fields.items():
                extra = field.json_schema_extra or {}
                if "status" not in extra or "source" not in extra:
                    missing.append(f"{model.__name__}.{name}")
        assert missing == [], f"缺状态标注的字段: {missing}"

    def test_status_vocabulary_is_bounded(self) -> None:
        allowed = {"定死", "定死（fixture 可缩小）", "定死 + fallback", "tunable",
                   "消融", "运行时", "起步值", "扫描接口", "配置化 + 扫描", "触发式",
                   "升级项", "部署默认"}
        for model in self.ALL_MODELS:
            for field in model.model_fields.values():
                extra = field.json_schema_extra or {}
                assert extra["status"] in allowed, (
                    f"{model.__name__} 未知状态标注: {extra['status']}"
                )


class TestMrRateAnchorsPerCondition:
    """数值锚逐条件口径（#129）：MR-RATE 线 latent 形状与 sigma 日程
    数值锚逐条件派生自条件词汇表工件，config 单域锚字段
    （``latent_shape`` / ``policy.input_img_size_numel``）属 BraTS 语义——
    显式声明即字段级拒绝（防「以为全局单值锚仍生效」的日程静默错位；
    与 condition_vocabulary_json 互斥携带同款哲学）。BraTS 线显式声明
    现状放行（单域语义 = 单条件词汇特例）。
    """

    def _mr_data(self, tmp_path: Path) -> dict:
        """MR-RATE 合法 config 原始 dict（不携带单域锚字段；schema 不查
        工件存在性，路径占位即可）。"""
        return {
            "experiment": {"group": "modal-label", "dataset": "MR-RATE"},
            "fixture_mode": True,
            # 强度臂两域锁死（#130/#71）：MR 线须显式 clip=False
            "preprocessing": {"intensity_clip": False},
            "artifacts": {
                "unet_ckpt": str(tmp_path / "unet.pt"),
                "vae_ckpt": str(tmp_path / "vae.pt"),
                "net_config_json": str(tmp_path / "unet_config.json"),
                "modality_mapping_json": str(tmp_path / "modality_mapping.json"),
                "dataset_root": str(tmp_path / "dataset"),
                "condition_vocabulary_json": str(
                    tmp_path / "condition_vocabulary.json"
                ),
                # prepare 装配输入四件套（#121/#131 schema 必填面）
                "mrrate_metadata_csv": str(tmp_path / "metadata.csv"),
                "mrrate_splits_csv": str(tmp_path / "splits.csv"),
                "eval_manifest_csv": str(tmp_path / "eval_manifest.csv"),
                "mrrate_data_snapshot": "MR-RATE@v1.0",
            },
            "reward": {
                "disc_batch_size_k": 4,
                "replay_buffer_capacity": 64,
                "real_pool_manifest": str(tmp_path / "real_pool.json"),
                "heldout_real_manifest": str(tmp_path / "heldout_real.json"),
                "channel_stats_json": str(tmp_path / "channel_stats.json"),
                "pretrain_report_json": str(tmp_path / "report.json"),
                "pretrain_gate_auc": 0.51,
                "sampling_manifest_json": str(
                    tmp_path / "sampling_manifest.json"
                ),
            },
            "schedule": {"seed": 0, "baseline_samples": 4},
        }

    def test_mr_rate_rejects_explicit_latent_shape(
        self, tmp_path: Path,
    ) -> None:
        """MR 线显式携带 latent_shape → 装载拒绝（错误点名该字段与
        逐条件口径；守卫在模型层、先例同 fixture_mode 通道守卫）。"""
        data = self._mr_data(tmp_path)
        data["latent_shape"] = [4, 16, 16, 8]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        message = str(exc_info.value.errors())
        assert "latent_shape" in message
        assert "词汇表" in message

    def test_mr_rate_rejects_explicit_numel_anchor(
        self, tmp_path: Path,
    ) -> None:
        """MR 线显式携带 policy.input_img_size_numel → 装载拒绝。"""
        data = self._mr_data(tmp_path)
        data["policy"] = {"input_img_size_numel": 2048}
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        message = str(exc_info.value.errors())
        assert "input_img_size_numel" in message
        assert "词汇表" in message

    def test_mr_rate_rejects_default_valued_explicit_declaration(
        self, tmp_path: Path,
    ) -> None:
        """显式传默认值（[4,64,64,32]）同样是单域锚意图的表达——同样
        拒绝（fields_set 语义：显式携带即意图，不区分值是否为默认）。"""
        data = self._mr_data(tmp_path)
        data["latent_shape"] = [4, 64, 64, 32]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "latent_shape" in str(exc_info.value.errors())

    def test_mr_rate_default_anchors_pass(self, tmp_path: Path) -> None:
        """未显式携带（默认单域值在 MR 线无消费）→ 装载通过。"""
        config = CynosureConfig.model_validate(self._mr_data(tmp_path))
        # 默认值仍在 schema 上（字段保留），但 MR 线运行时零消费
        assert config.latent_shape == (4, 64, 64, 32)
        assert config.policy.input_img_size_numel == 131072

    def test_brats_explicit_anchors_pass(self, valid_config_dict: dict) -> None:
        """BraTS 线显式携带单域锚字段照旧合法（单条件词汇特例语义不动）。"""
        config = CynosureConfig.model_validate(valid_config_dict)
        assert config.latent_shape == (4, 64, 64, 32)
        assert config.policy.input_img_size_numel == 131072

    def test_mr_rate_dump_excludes_single_domain_anchors(
        self, tmp_path: Path,
    ) -> None:
        """MR config 的序列化工件形态不携带单域锚字段（model_dump_json
        全量展开会把未显式声明的默认值落成显式键——不经排除则 JSON
        往返被互斥守卫误拒）。"""
        data = self._mr_data(tmp_path)
        config = CynosureConfig.model_validate(data)
        payload = json.loads(config.model_dump_json())
        assert "latent_shape" not in payload
        assert "input_img_size_numel" not in payload["policy"]
        # 往返：序列化产物可原样再装载
        revived = CynosureConfig.model_validate(payload)
        assert revived.experiment.dataset == "MR-RATE"
