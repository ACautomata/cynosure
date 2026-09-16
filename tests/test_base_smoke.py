"""基座 checkpoint 装载与前向自检的单测（wayfinder #120 的验证面）。

覆盖：``configs/mrrate-base`` 转写工件的键完整性与语义锚（誊抄错误的
fail-fast 面）、checkpoint 容器形态分派（上游训练容器 / 裸 state_dict /
MetaTensor 元数据的严格反序列化）、``BaseSmokeRunner`` 的读数契约
（定点前向逐位复现、参数量对账、scale factor 来源分派）与 base-smoke
子命令（exit 码 / 报告落盘 / torchrun 拒绝）。真实基座发布件（HF
``nvidia/NV-Generate-MR-Brain``）的装载自检由 gpu+slow 标记的测试承担
——smoke config 路径经环境变量注入，缺省跳过（集群侧产出）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.apps.generation.maisi.networks.diffusion_model_unet_maisi import (
    DiffusionModelUNetMaisi,
)
from monai.data.meta_tensor import MetaTensor
from monai.utils.enums import TraceKeys
from pydantic import ValidationError

from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.smoke import BaseLoadReadout, BaseSmokeConfig, BaseSmokeRunner
from tests.conftest import CliSession

REPO_CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "mrrate-base"

# 上游发布件的实测参数量锚（sugon DCU 实测；HF 模型卡的「240M」与发布
# 权重不可复算对上——差异登记在 configs/mrrate-base/README.md，本锚
# 只对账「装载出的模型 == 权重文件」这一工件内自洽关系）
EXPECTED_UNET_PARAMETERS = 180_500_868
EXPECTED_VAE_PARAMETERS = 20_944_897

_REAL_SMOKE_CONFIG_ENV = "CYNOSURE_BASE_SMOKE_CONFIG"

# 上游冻结转写的完整键集锚（Codex review 5212797632：漏转键若不影响
# state_dict 形状——如 use_flash_attention、norm_float16——strict 装载与
# unconsumed_keys 都检不出，网络会以 MONAI 缺省值静默执行）。键集被
# 逐键钉死：任何键集变化都必须是有意的转写更新，本锚显式红。
FROZEN_UNET_TRANSCRIPTION_KEYS = frozenset({
    "spatial_dims", "in_channels", "out_channels", "num_channels",
    "attention_levels", "num_head_channels", "num_res_blocks",
    "use_flash_attention", "include_top_region_index_input",
    "include_bottom_region_index_input", "include_spacing_input",
    "num_class_embeds", "resblock_updown", "include_fc",
})
FROZEN_VAE_TRANSCRIPTION_KEYS = frozenset({
    "spatial_dims", "in_channels", "out_channels", "latent_channels",
    "num_channels", "num_res_blocks", "norm_num_groups", "norm_eps",
    "attention_levels", "use_convtranspose", "norm_float16",
    "num_splits", "dim_split", "use_checkpointing",
    "with_encoder_nonlocal_attn", "with_decoder_nonlocal_attn",
})


def _write_fixture_artifacts(directory: Path):
    """fixture 网络工件（固定 seed 机制，test_reward_fixture 先例）。"""
    torch.manual_seed(7)
    return Fixture().write_artifacts(directory)


def _fixture_smoke_config(
    tmp_path: Path, artifacts, *, output_name: str = "report.json", **overrides,
) -> BaseSmokeConfig:
    """fixture 尺寸的合法 smoke config（latent [4,16,16,8]，CPU，中性
    scale factor——fixture 工件是裸 state_dict、无容器元数据）。"""
    data = {
        "unet_ckpt": str(artifacts.unet_ckpt),
        "unet_config_json": str(artifacts.unet_config_json),
        "vae_ckpt": str(artifacts.vae_ckpt),
        "vae_config_json": str(artifacts.vae_config_json),
        "output_json": str(tmp_path / output_name),
        "device": "cpu",
        "latent_shape": [4, 16, 16, 8],
        "latent_scale_factor": 1.0,
    }
    data.update(overrides)
    return BaseSmokeConfig.model_validate(data)


class TestTranscriptionArtifacts:
    """configs/mrrate-base 转写工件（零依赖只读转写的落地面）。"""

    def test_keys_are_fully_consumed_by_monai_constructors(self) -> None:
        """转写键逐键被 MONAI 构造器消费：装载面对未知键静默过滤，
        誊抄错误（键名拼错/字段漏转）必须在本层可检出而非静默丢架构参数。"""
        unet_config = NetworkAssembler.load_json(REPO_CONFIGS / "unet_config.json")
        vae_config = NetworkAssembler.load_json(REPO_CONFIGS / "vae_config.json")
        assert NetworkAssembler.unconsumed_keys(DiffusionModelUNetMaisi, unet_config) == set()
        assert NetworkAssembler.unconsumed_keys(AutoencoderKlMaisi, vae_config) == set()

    def test_transcription_key_sets_are_pinned_to_frozen_anchor(self) -> None:
        """转写键集与冻结锚逐键相等：unconsumed_keys 只拦**多余**键，
        拦不住**漏转**键（不影响 state_dict 形状的键——use_flash_attention、
        norm_float16——漏转后网络以 MONAI 缺省值静默执行）。键集方向
        的完整性只能对照冻结锚断言：漏键/多键/拼错都在此显式红。"""
        unet_config = NetworkAssembler.load_json(REPO_CONFIGS / "unet_config.json")
        vae_config = NetworkAssembler.load_json(REPO_CONFIGS / "vae_config.json")
        assert set(unet_config) == FROZEN_UNET_TRANSCRIPTION_KEYS
        assert set(vae_config) == FROZEN_VAE_TRANSCRIPTION_KEYS

    def test_upstream_semantics_are_preserved(self) -> None:
        """转写不是照抄上游 JSON（插值/死参数/其它网络段落不进来）：
        逐字段语义锚钉死（来源见该目录 README）。"""
        unet_config = NetworkAssembler.load_json(REPO_CONFIGS / "unet_config.json")
        assert unet_config["in_channels"] == unet_config["out_channels"] == 4
        assert unet_config["num_channels"] == [64, 128, 256, 512]
        assert unet_config["attention_levels"] == [False, False, True, True]
        assert unet_config["num_head_channels"] == [0, 0, 32, 32]
        assert unet_config["num_res_blocks"] == 2
        assert unet_config["num_class_embeds"] == 128
        assert unet_config["include_spacing_input"] is True
        assert unet_config["include_top_region_index_input"] is False
        assert unet_config["include_bottom_region_index_input"] is False
        vae_config = NetworkAssembler.load_json(REPO_CONFIGS / "vae_config.json")
        assert vae_config["latent_channels"] == 4
        assert vae_config["num_channels"] == [64, 128, 256]
        assert vae_config["norm_float16"] is True
        assert vae_config["use_convtranspose"] is False
        assert vae_config["num_splits"] == 4


class TestCheckpointContainerDispatch:
    """checkpoint 容器形态分派（装载面的上游发布件承载点）。"""

    @staticmethod
    def _fixture_unet_state(directory: Path) -> dict:
        artifacts = _write_fixture_artifacts(directory)
        return NetworkAssembler.read_state_dict(artifacts.unet_ckpt)

    def test_container_form_loads_nested_state(self, tmp_path: Path) -> None:
        """上游训练容器形态（unet_state_dict 包装键）解包装载：装载出的
        权重与内层 state_dict 逐位一致。"""
        state = self._fixture_unet_state(tmp_path / "state")
        container = tmp_path / "container.pt"
        torch.save(
            {
                "epoch": 3, "loss": 0.5, "num_train_timesteps": 1000,
                "scale_factor": 2.0, "unet_state_dict": state,
            },
            container,
        )
        loaded = NetworkAssembler.read_state_dict(container)
        assert set(loaded) == set(state)
        for key in state:
            assert torch.equal(loaded[key], state[key])

    def test_container_rejects_ambiguous_keys(self, tmp_path: Path) -> None:
        state = self._fixture_unet_state(tmp_path / "state")
        ambiguous = tmp_path / "ambiguous.pt"
        torch.save(
            {"unet_state_dict": state, "controlnet_state_dict": {}}, ambiguous,
        )
        with pytest.raises(ValueError, match="不唯一"):
            NetworkAssembler.read_state_dict(ambiguous)

    def test_container_value_must_be_state_dict(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.pt"
        torch.save({"unet_state_dict": [1, 2, 3]}, bad)
        with pytest.raises(ValueError, match="state_dict"):
            NetworkAssembler.read_state_dict(bad)

    def test_bare_state_dict_passthrough(self, tmp_path: Path) -> None:
        """裸 state_dict 形态（仓库自产工件）原样直通：既有装载契约
        不因容器分派改变。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        state = NetworkAssembler.read_state_dict(artifacts.unet_ckpt)
        original = torch.load(
            artifacts.unet_ckpt, map_location="cpu", weights_only=True,
        )
        assert set(state) == set(original)
        for key in original:
            assert torch.equal(state[key], original[key])

    def test_meta_tensor_metadata_loads_under_strict_deserialization(
        self, tmp_path: Path,
    ) -> None:
        """上游容器标量的 MetaTensor 形态（meta 含 TraceKeys）在
        weights_only=True 严格反序列化下可装载——safe globals 白名单
        登记的承载面（不开 weights_only=False 逃生门）。"""
        state = self._fixture_unet_state(tmp_path / "state")
        container = tmp_path / "meta.pt"
        torch.save(
            {
                "scale_factor": MetaTensor(
                    torch.tensor(0.97), meta={"origin": TraceKeys.NONE},
                ),
                "unet_state_dict": state,
            },
            container,
        )
        loaded = NetworkAssembler.read_state_dict(container)
        assert set(loaded) == set(state)
        assert NetworkAssembler.checkpoint_scale_factor(
            NetworkAssembler.read_checkpoint(container),
        ) == pytest.approx(0.97)

    def test_checkpoint_scale_factor_absent_on_bare_state_dict(
        self, tmp_path: Path,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        container = NetworkAssembler.read_checkpoint(artifacts.unet_ckpt)
        assert NetworkAssembler.checkpoint_scale_factor(container) is None


class TestBaseSmokeRunner:
    """读数契约（fixture 尺寸，CPU 全路径）。"""

    def test_report_contract_on_fixture_scale(self, tmp_path: Path) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        report = BaseSmokeRunner(config).run()
        assert report.velocity_shape == (1, 4, 16, 16, 8)
        assert report.velocity_repeat_identical is True
        assert report.latent_shape == (4, 16, 16, 8)
        assert report.modality_token == 9
        assert report.timestep == 500
        assert report.cfg_weight == pytest.approx(10.0)
        assert report.image_shape == (1, 64, 64, 32)
        assert report.encoded_shape == (4, 16, 16, 8)
        assert report.decoded_shape == (1, 1, 64, 64, 32)
        assert report.decode_autocast_dtype == "float16"
        assert report.loading.latent_scale_factor == pytest.approx(1.0)
        assert report.loading.latent_scale_factor_source == "config"
        assert (
            report.loading.unet_parameters
            == report.loading.unet_checkpoint_parameters
        )
        assert report.velocity_sha256 != report.encoded_sha256

    def test_report_is_written_and_round_trips(self, tmp_path: Path) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        report = BaseSmokeRunner(config).run()
        written = json.loads(config.output_json.read_text(encoding="utf-8"))
        assert written["velocity_sha256"] == report.velocity_sha256
        assert (
            written["loading"]["unet_parameters"]
            == report.loading.unet_parameters
        )

    def test_forward_is_deterministic_across_runner_instances(
        self, tmp_path: Path,
    ) -> None:
        """两次独立 Runner 构造（同 seed、同工件）→ 同一指纹：跨进程
        逐位复现语义的进程内投影（固定 seed 的定点输入 + 无随机前向）。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        first = BaseSmokeRunner(config).run()
        second = BaseSmokeRunner(config).run()
        assert first.velocity_sha256 == second.velocity_sha256
        assert first.encoded_sha256 == second.encoded_sha256

    def test_modality_token_is_an_active_forward_input(
        self, tmp_path: Path,
    ) -> None:
        """定点 token 是前向的活跃输入（9 vs 10 → 不同 velocity）：
        模态条件通道真实在场，不是被静默旁路。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        first = BaseSmokeRunner(
            _fixture_smoke_config(tmp_path, artifacts, modality_token=9),
        ).run()
        second = BaseSmokeRunner(
            _fixture_smoke_config(
                tmp_path, artifacts, output_name="report_t2.json",
                modality_token=10,
            ),
        ).run()
        assert first.velocity_sha256 != second.velocity_sha256

    def test_nondeterministic_forward_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        monkeypatch.setattr(
            runner, "_velocity",
            lambda: torch.randn(1, 4, 16, 16, 8),
        )
        with pytest.raises(ValueError, match="逐位不可复现"):
            runner.run()
        assert not config.output_json.exists()

    def test_scale_factor_resolved_from_container_metadata(
        self, tmp_path: Path,
    ) -> None:
        """缺省 scale factor 从基座 checkpoint 容器元数据读（生产单一
        来源）；解码按它除回 encoder 域。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        state = NetworkAssembler.read_state_dict(artifacts.unet_ckpt)
        container = tmp_path / "unet_container.pt"
        torch.save(
            {"epoch": 1, "scale_factor": 2.0, "unet_state_dict": state},
            container,
        )
        config = _fixture_smoke_config(
            tmp_path, artifacts, unet_ckpt=str(container),
            latent_scale_factor=None,
        )
        report = BaseSmokeRunner(config).run()
        assert report.loading.latent_scale_factor == pytest.approx(2.0)
        assert report.loading.latent_scale_factor_source == "checkpoint"

    def test_missing_scale_factor_is_rejected(self, tmp_path: Path) -> None:
        """无容器元数据又无显式值即显式拒绝：随手取 1.0 会让解码落在
        错误量级（读数看着「跑通」实则口径失真）。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(
            tmp_path, artifacts, latent_scale_factor=None,
        )
        with pytest.raises(ValueError, match="scale factor 缺席"):
            BaseSmokeRunner(config)

    def test_decode_input_is_policy_domain_scaled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """解码器的输入契约是 policy 域（checkpoint scaled 域）：encoder
        原始输出（存储域、未乘 scale_factor）必须先乘回再进 decode——
        decode 内部除回后恰好归位 encoder 域，VAE 往返才闭环（Codex
        review 5210746815：直送原始输出会把 encoded/scale_factor 送进
        解码器，不是同一张量的往返）。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        state = NetworkAssembler.read_state_dict(artifacts.unet_ckpt)
        container = tmp_path / "unet_container.pt"
        torch.save(
            {"epoch": 1, "scale_factor": 2.0, "unet_state_dict": state},
            container,
        )
        config = _fixture_smoke_config(
            tmp_path, artifacts, unet_ckpt=str(container),
            latent_scale_factor=None,
        )
        runner = BaseSmokeRunner(config)
        assert runner._readout.latent_scale_factor == pytest.approx(2.0)
        observed: dict[str, torch.Tensor] = {}
        original_encode = runner._encoder.encode
        original_decode = runner._decoder.decode

        def spy_encode(image: torch.Tensor, noise_seed: int = 0) -> torch.Tensor:
            observed["encoded"] = original_encode(image, noise_seed).detach()
            return observed["encoded"]

        def spy_decode(latents: torch.Tensor) -> torch.Tensor:
            observed["decode_input"] = latents.detach().cpu()
            return original_decode(latents)

        monkeypatch.setattr(runner._encoder, "encode", spy_encode)
        monkeypatch.setattr(runner._decoder, "decode", spy_decode)
        runner.run()
        assert torch.equal(
            observed["decode_input"].squeeze(0),
            observed["encoded"] * 2.0,
        )

    def test_nonfinite_encode_output_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """编码输出含 NaN/Inf 时显式拒绝而非落一份「成功」报告：fp16
        kernel 或权重损坏产出的非有限值不该通过哈希与形状读数蒙混。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        monkeypatch.setattr(
            runner._encoder, "encode",
            lambda image, noise_seed=0: torch.full((4, 16, 16, 8), float("nan")),
        )
        with pytest.raises(ValueError, match="非有限"):
            runner.run()
        assert not config.output_json.exists()

    def test_nonfinite_decode_output_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        monkeypatch.setattr(
            runner._decoder, "decode",
            lambda latents: torch.full((1, 1, 64, 64, 32), float("inf")),
        )
        with pytest.raises(ValueError, match="非有限"):
            runner.run()
        assert not config.output_json.exists()

    def test_velocity_shape_must_match_latent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """velocity 与 latent 同形是 ODE 更新的前提（Codex review
        5210746815）：out_channels 错配的工件对产出确定性有限张量也不得
        通过——形状守卫在逐位复现判定之前。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        wrong = torch.zeros(1, 8, 16, 16, 8)  # 同一对象两次返回：逐位一致
        monkeypatch.setattr(runner, "_velocity", lambda: wrong)
        with pytest.raises(ValueError, match="latent_shape"):
            runner.run()
        assert not config.output_json.exists()

    def test_encode_shape_must_match_policy_grid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """VAE 工件对自洽但 latent 通道/压缩比与 policy 网格不同时，
        装载与解码都「成功」——形状契约是唯一可判的面（Codex review
        5210746815）：不得只记录 encoded_shape 而放行成功报告。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        monkeypatch.setattr(
            runner._encoder, "encode",
            lambda image, noise_seed=0: torch.zeros(3, 16, 16, 8),
        )
        with pytest.raises(ValueError, match="latent_shape"):
            runner.run()
        assert not config.output_json.exists()

    def test_decode_shape_must_match_image_grid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        monkeypatch.setattr(
            runner._decoder, "decode",
            lambda latents: torch.zeros(1, 1, 32, 32, 16),
        )
        with pytest.raises(ValueError, match="image_shape"):
            runner.run()
        assert not config.output_json.exists()

    def test_modality_token_beyond_embedding_table_is_rejected(
        self, tmp_path: Path,
    ) -> None:
        """token 须落在装载出的类别名嵌入表内（fixture 表 128）：越界
        token 在 schema 层合法（ge=0），到 nn.Embedding 才 IndexError——
        构造期以 ValueError 拒绝（CLI 侧收敛为可读的 exit 2），而非
        traceback（Codex review 5210746815）。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(
            tmp_path, artifacts, modality_token=128,
        )
        with pytest.raises(ValueError, match="modality_token"):
            BaseSmokeRunner(config)

    def test_nonfinite_checkpoint_scale_factor_is_rejected(
        self, tmp_path: Path,
    ) -> None:
        """容器元数据 scale_factor = inf 显式拒绝：inf 通过「>0」检查，
        解码除 inf 得全零 latent——形状正确、数值有限的「成功」读数
        （Codex review 5212797632）。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        state = NetworkAssembler.read_state_dict(artifacts.unet_ckpt)
        container = tmp_path / "unet_container_inf.pt"
        torch.save(
            {"scale_factor": float("inf"), "unet_state_dict": state},
            container,
        )
        config = _fixture_smoke_config(
            tmp_path, artifacts, unet_ckpt=str(container),
            latent_scale_factor=None,
        )
        with pytest.raises(ValueError, match="非有限"):
            BaseSmokeRunner(config)

    def test_deterministic_scope_restores_warn_only(
        self, tmp_path: Path,
    ) -> None:
        """确定性作用域退出须还原 warn_only 位（Codex review
        5212797632）：环境以 (True, warn_only=True) 运行时，退出后
        warn_only 被静默降级为 False，后续非确定算子从告警变异常。"""
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(tmp_path, artifacts)
        runner = BaseSmokeRunner(config)
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            runner.run()
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.is_deterministic_algorithms_warn_only_enabled()
        finally:
            # 还原 conftest 导入期建立的基线口径 (True, warn_only=False)
            torch.use_deterministic_algorithms(True, warn_only=False)

    def test_transcription_typo_is_rejected_at_construction(
        self, tmp_path: Path,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        typo_config = tmp_path / "unet_config_typo.json"
        data = json.loads(artifacts.unet_config_json.read_text(encoding="utf-8"))
        data["num_channles"] = data.pop("num_channels")  # 誊抄键名拼错
        typo_config.write_text(json.dumps(data), encoding="utf-8")
        config = _fixture_smoke_config(
            tmp_path, artifacts, unet_config_json=str(typo_config),
        )
        with pytest.raises(ValueError, match="num_channles"):
            BaseSmokeRunner(config)

    def test_param_count_mismatch_is_rejected(self) -> None:
        """参数量对账守卫（``BaseLoadReadout`` 不变式）：模型侧与权重
        文件侧计数不一致即构造期拒绝。strict 装载下该分支实际不可达
        （键/形状相等即总量相等），直接锁定不变式——防未来装载面放松
        strict 时静默失去对账。"""
        with pytest.raises(ValueError, match="参数量对账不符"):
            BaseLoadReadout(
                unet_parameters=1,
                unet_checkpoint_parameters=2,
                vae_checkpoint_parameters=3,
                latent_scale_factor=1.0,
                latent_scale_factor_source="config",
            )


class TestBaseSmokeConfigSchema:
    def test_latent_spatial_dims_must_be_compressible(self) -> None:
        with pytest.raises(ValidationError, match="整除"):
            BaseSmokeConfig.model_validate({
                "unet_ckpt": "u.pt", "unet_config_json": "u.json",
                "vae_ckpt": "v.pt", "vae_config_json": "v.json",
                "output_json": "r.json", "device": "cpu",
                "latent_shape": [4, 15, 16, 8],
            })

    def test_decode_window_must_integrate_with_vae_upsampling(self) -> None:
        """overlap×roi×4 须逐维为整数（MONAI 滑窗缩放约束在配置期
        fail-fast，而非解码期炸出脱节错误）。满精度 2/3 的二进制浮点
        表示（127.999…）在容差下数学整除、放行；官方四位截断字面
        0.6666（=127.9872）显式拒绝——MONAI 按 ``int(roi·(1−overlap))``
        截断滑窗步长，0.6666 与 2/3 的步长并不相同（16 vs 15），放行
        即静默改变解码口径。"""
        base = {
            "unet_ckpt": "u.pt", "unet_config_json": "u.json",
            "vae_ckpt": "v.pt", "vae_config_json": "v.json",
            "output_json": "r.json", "device": "cpu",
        }
        assert BaseSmokeConfig.model_validate({
            **base, "decode_roi_size": [45, 45, 45], "decode_overlap": 0.5,
        }).decode_roi_size == (45, 45, 45)
        assert BaseSmokeConfig.model_validate(base).decode_overlap == pytest.approx(2 / 3)
        with pytest.raises(ValidationError, match="整数"):
            BaseSmokeConfig.model_validate({**base, "decode_overlap": 0.7})

    def test_upstream_overlap_literal_is_rejected(self) -> None:
        """官方 config 的四位截断字面 0.6666 不放行（锁定守卫意图）：
        操作者照抄上游字面值会得到指向满精度 2/3 的明确报错，而非
        静默落在步长不同的滑窗口径上。"""
        base = {
            "unet_ckpt": "u.pt", "unet_config_json": "u.json",
            "vae_ckpt": "v.pt", "vae_config_json": "v.json",
            "output_json": "r.json", "device": "cpu",
        }
        with pytest.raises(ValidationError, match="整数"):
            BaseSmokeConfig.model_validate({**base, "decode_overlap": 0.6666})

    def test_output_json_must_differ_from_input_artifacts(self) -> None:
        """报告落盘路径不得与任何输入工件重合（Codex review
        5212797632）：输入在报告写出前已全部装载，重合路径会被 JSON
        报告静默覆盖掉模型权重/转写配置——自检「成功」即工件损毁。"""
        base = {
            "unet_ckpt": "weights.pt", "unet_config_json": "u.json",
            "vae_ckpt": "v.pt", "vae_config_json": "v.json", "device": "cpu",
        }
        for colliding_field, value in (
            ("unet_ckpt", "weights.pt"), ("unet_config_json", "u.json"),
            ("vae_ckpt", "v.pt"), ("vae_config_json", "v.json"),
        ):
            with pytest.raises(ValidationError, match="输入工件"):
                BaseSmokeConfig.model_validate({
                    **base, colliding_field: value, "output_json": value,
                })

    def test_timestep_confined_to_rflow_domain(self) -> None:
        base = {
            "unet_ckpt": "u.pt", "unet_config_json": "u.json",
            "vae_ckpt": "v.pt", "vae_config_json": "v.json",
            "output_json": "r.json", "device": "cpu",
        }
        with pytest.raises(ValidationError, match="timestep"):
            BaseSmokeConfig.model_validate({**base, "timestep": 1001})


class TestBaseSmokeCli:
    """base-smoke 子命令（单进程、独立 schema、报告落盘）。"""

    def _write_config_json(self, tmp_path: Path, artifacts) -> Path:
        config = _fixture_smoke_config(tmp_path, artifacts)
        path = tmp_path / "base_smoke.json"
        path.write_text(config.model_dump_json(), encoding="utf-8")
        return path

    def test_end_to_end(self, cli: CliSession, tmp_path: Path) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        result = cli.run("base-smoke", "--config", str(self._write_config_json(tmp_path, artifacts)))
        assert result.code == 0, result.stderr
        assert "装载：UNet" in result.stdout
        assert "定点前向" in result.stdout
        assert "VAE 往返" in result.stdout
        assert "基座装载自检报告已落盘" in result.stdout
        assert (tmp_path / "report.json").is_file()

    def test_input_contract_violation_exits_2(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        config = _fixture_smoke_config(
            tmp_path, artifacts,
            unet_ckpt=str(tmp_path / "absent.pt"),
        )
        path = tmp_path / "base_smoke.json"
        path.write_text(config.model_dump_json(), encoding="utf-8")
        result = cli.run("base-smoke", "--config", str(path))
        assert result.code == 2
        assert "base-smoke 未通过" in result.stderr

    def test_invalid_config_exits_2(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        artifacts = _write_fixture_artifacts(tmp_path / "artifacts")
        path = self._write_config_json(tmp_path, artifacts)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["unknown_field"] = True
        path.write_text(json.dumps(data), encoding="utf-8")
        result = cli.run("base-smoke", "--config", str(path))
        assert result.code == 2
        assert "config 校验失败" in result.stderr
        assert "unknown_field" in result.stderr

    def test_missing_config_exits_2(self, cli: CliSession, tmp_path: Path) -> None:
        result = cli.run("base-smoke", "--config", str(tmp_path / "absent.json"))
        assert result.code == 2
        assert "不存在" in result.stderr

    def test_torchrun_is_rejected(
        self, cli: CliSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RANK", "3")
        result = cli.run(
            "base-smoke", "--config", str(tmp_path / "whatever.json"),
        )
        assert result.code == 2
        assert "拒绝 torchrun" in result.stderr


@pytest.mark.gpu  # 生产尺寸 encode/decode 的网络大计算：GPU 口径
@pytest.mark.slow  # 真实发布件装载 + 滑窗解码：分钟级轮次，默认跳过
class TestRealBaseArtifacts:
    """真实基座发布件（HF nvidia/NV-Generate-MR-Brain）的装载自检。

    smoke config 指向真实工件路径（集群侧产出），经
    ``_REAL_SMOKE_CONFIG_ENV`` 注入；工件缺席的环境跳过——参数量锚与
    逐位复现契约由真实权重承载，fixture 尺寸测试只覆盖语义。"""

    @pytest.mark.skipif(
        os.environ.get(_REAL_SMOKE_CONFIG_ENV) is None,
        reason=f"需要 {_REAL_SMOKE_CONFIG_ENV} 指向真实基座工件的 "
               "base-smoke config（集群侧产出）",
    )
    def test_hf_release_smoke_end_to_end(self, cli: CliSession) -> None:
        config_path = Path(os.environ[_REAL_SMOKE_CONFIG_ENV])
        result = cli.run("base-smoke", "--config", str(config_path))
        assert result.code == 0, result.stderr
        report_path = Path(
            json.loads(config_path.read_text(encoding="utf-8"))["output_json"],
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["loading"]["unet_parameters"] == EXPECTED_UNET_PARAMETERS
        assert (
            report["loading"]["unet_checkpoint_parameters"]
            == EXPECTED_UNET_PARAMETERS
        )
        assert (
            report["loading"]["vae_checkpoint_parameters"]
            == EXPECTED_VAE_PARAMETERS
        )
        assert report["loading"]["latent_scale_factor_source"] == "checkpoint"
        assert report["velocity_repeat_identical"] is True
        assert report["latent_shape"] == [4, 64, 64, 32]
        assert report["image_shape"] == [1, 256, 256, 128]
        assert report["encoded_shape"] == [4, 64, 64, 32]
        assert report["decoded_shape"] == [1, 1, 256, 256, 128]
        assert report["decode_autocast_dtype"] == "float16"

    @pytest.mark.skipif(
        os.environ.get(_REAL_SMOKE_CONFIG_ENV) is None,
        reason=f"需要 {_REAL_SMOKE_CONFIG_ENV} 指向真实基座工件的 "
               "base-smoke config（集群侧产出）",
    )
    def test_hf_release_forward_is_reproducible_across_processes(
        self, tmp_path: Path,
    ) -> None:
        """跨进程逐位复现（AC「固定 seed 与确定性 kernels」的完整形态）：
        两个**独立解释器进程**（``python -c`` 子进程，pytest 进程之外
        各自全新初始化 torch/CUDA）的同一定点前向 → 同一 velocity 指纹。

        CliSession 与 pytest 同进程，进程内循环共享 CUDA 上下文与环境
        状态，不构成跨进程证据（Codex review 5212797632）；确定性口径
        随测试 seam 显式随行——子进程导入 ``tests.conftest``（其导入期
        收口，ADR-0011 的「测试进程属性」含测试派生的子进程）。"""
        source = json.loads(
            Path(os.environ[_REAL_SMOKE_CONFIG_ENV]).read_text(encoding="utf-8"),
        )
        repo_root = Path(__file__).resolve().parent.parent
        src_path = repo_root / "src"
        shas = set()
        for index in range(2):
            config = dict(source)
            config["output_json"] = str(tmp_path / f"report_{index}.json")
            config_path = tmp_path / f"config_{index}.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable, "-c",
                    "import tests.conftest; from cynosure.cli import main; main()",
                    "base-smoke", "--config", str(config_path),
                ],
                capture_output=True, text=True, timeout=1800, check=False,
                cwd=str(repo_root),
                env={
                    **os.environ,
                    "PYTHONPATH": os.pathsep.join(
                        [str(src_path), os.environ.get("PYTHONPATH", "")],
                    ).rstrip(os.pathsep),
                },
            )
            assert result.returncode == 0, result.stderr
            report = json.loads(
                Path(config["output_json"]).read_text(encoding="utf-8"),
            )
            shas.add(report["velocity_sha256"])
        assert len(shas) == 1
