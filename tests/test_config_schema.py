"""config schema 校验测试：配置项清单全量字段落 schema、状态标注完备、
定死值不可改、跨字段定死语义（数值锚、奇异端、组2/组3 ControlNet）。"""

import copy
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from cynosure.config import ConfigLoader, CynosureConfig
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
        assert config.schedule.n_plateau == 3
        assert config.schedule.milestone_interval == 50
        assert config.schedule.checkpoint_interval == 10
        assert config.sharding.strategy == "fsdp"
        # 部署行（orchestration + ADR-0005）：单实例 4 卡、产物根在持久分区下
        assert config.deployment.nproc_per_node == 4
        assert config.deployment.output_root == Path("/root/private_data/cynosure")

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

    def test_missing_required_field_is_field_level_error(self) -> None:
        data = copy.deepcopy(MINIMAL_CONFIG_DICT)
        del data["experiment"]["group"]
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("experiment", "group") in self._locations(exc_info.value)

    def test_unknown_field_rejected(self, valid_config_dict: dict) -> None:
        data = copy.deepcopy(valid_config_dict)
        data["polic"] = {}  # 拼错段名
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert "polic" in str(exc_info.value.errors()[0]["loc"])

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
        """N_baseline 口径 200–500（experiment-design 章）。"""
        data = copy.deepcopy(valid_config_dict)
        data["schedule"].update({"baseline_samples": 100})
        with pytest.raises(ValidationError) as exc_info:
            CynosureConfig.model_validate(data)
        assert ("schedule", "baseline_samples") in self._locations(exc_info.value)
        data["schedule"].update({"baseline_samples": 500})
        CynosureConfig.model_validate(data)

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
        （spec 配置项清单「组3 衔接」行的两个分支之一）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "sequential"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
        config = CynosureConfig.model_validate(data)
        assert config.experiment.group == "sequential"
        assert config.experiment.stage1_run_dir is None

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
        config = CynosureConfig.model_validate(data)
        assert config.experiment.stage1_run_dir == Path("runs/stage1")

    def test_source_latent_scale_factor_default(self, valid_config_dict: dict) -> None:
        """组2 双条件之一：ControlNet 条件 = 源影像 latent × scale_factor；
        fixture 中性默认 1.0（生产随基座 ControlNet 推理 config 核对）。"""
        data = copy.deepcopy(valid_config_dict)
        data["experiment"]["group"] = "cross-modal"
        data["artifacts"]["controlnet_ckpt"] = "ckpts/controlnet.pt"
        data["artifacts"]["controlnet_config_json"] = "configs/controlnet.json"
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


class TestStatusAnnotations:
    ALL_MODELS = [
        CynosureConfig,
        *[
            f.annotation
            for f in CynosureConfig.model_fields.values()
            if hasattr(f.annotation, "model_fields")
        ],
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
