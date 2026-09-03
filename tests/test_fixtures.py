"""fixture 生成器 + netbuild 测试：MONAI 微型网络的可前向配置（CPU 可跑）、
fixture 数值锚（3 步 ODE、G 保持 12、latent [4,16,16,8]、input_img_size_numel=2048）。"""

import json
from pathlib import Path

import pytest
import torch

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler


class TestFixtureConfig:
    def test_fixture_config_passes_schema(self, tmp_path: Path) -> None:
        config = Fixture().config(tmp_path)
        assert isinstance(config, CynosureConfig)

    def test_spec_fixture_values(self, tmp_path: Path) -> None:
        """spec「Fixture 策略」的数值：缩小 latent、3 步 ODE、M={1}、G 保持 12。"""
        config = Fixture().config(tmp_path)
        assert config.fixture_mode is True  # 缩小日程的显式声明通道
        assert config.latent_shape == (4, 16, 16, 8)
        assert config.policy.num_inference_steps == 3
        assert config.policy.train_step_indices_m == {1}  # 避开 s≈1 奇异端
        assert config.policy.group_size_g == 12  # G 是方向数，与网络尺寸无关
        assert config.policy.input_img_size_numel == 2048  # = prod((16,16,8)) 数值锚
        assert config.reward.disc_num_layers_d == 1  # fixture 第三维 8 撑不住 2

    def test_config_artifacts_point_into_fixture_dir(self, tmp_path: Path) -> None:
        config = Fixture().config(tmp_path)
        assert config.artifacts.unet_ckpt == tmp_path / "unet.pt"
        assert config.artifacts.net_config_json == tmp_path / "unet_config.json"


class TestFixtureArtifacts:
    def test_writes_network_artifacts(self, tmp_path: Path) -> None:
        artifacts = Fixture().write_artifacts(tmp_path)
        assert artifacts.unet_ckpt.is_file()
        assert artifacts.unet_config_json.is_file()
        assert artifacts.discriminator_ckpt.is_file()
        assert artifacts.discriminator_config_json.is_file()

    def test_same_seed_reproduces_weights(self, tmp_path: Path) -> None:
        torch.manual_seed(7)
        first = Fixture().unet().state_dict()
        torch.manual_seed(7)
        second = Fixture().unet().state_dict()
        for key in first:
            assert torch.equal(first[key], second[key]), key


class TestNetbuildForward:
    """netbuild 按 artifact 构建并返回可前向网络（CPU 可跑）。"""

    @pytest.fixture
    def fixture_artifacts(self, tmp_path: Path) -> object:
        return Fixture().write_artifacts(tmp_path)

    def test_unet_forward_shape(self, fixture_artifacts: object) -> None:
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(fixture_artifacts.unet_config_json),
            checkpoint=fixture_artifacts.unet_ckpt,
        )
        unet = NetworkAssembler.unet(artifact)
        x = torch.randn(1, 4, 16, 16, 8)
        out = unet(
            x,
            timesteps=torch.tensor([500.0]),
            class_labels=torch.randint(0, 128, (1,)),
            spacing_tensor=torch.ones(1, 3),
        )
        assert out.shape == x.shape

    def test_discriminator_forward_patch_logits(self, fixture_artifacts: object) -> None:
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(
                fixture_artifacts.discriminator_config_json,
            ),
            checkpoint=fixture_artifacts.discriminator_ckpt,
        )
        disc = NetworkAssembler.discriminator(artifact)
        out = disc(torch.randn(2, 4, 16, 16, 8))
        patch_logits = out[-1]  # list 末元素 = patch logit 图
        assert patch_logits.shape[0] == 2
        assert patch_logits.shape[1] == 1
        assert patch_logits.dim() == 5  # patch logit 图 [B,1,D',H',W']，非单一标量

    def test_scheduler_timesteps_with_transform(self) -> None:
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=3, input_img_size_numel=2048,
        )
        assert scheduler.timesteps.tolist() == [1000, 442, 165]

    def test_unet_config_json_roundtrip_unknown_keys_ignored(
        self, fixture_artifacts: object,
    ) -> None:
        """网络配置 JSON 里的非构造参数（如基座死参数 scale）被静默过滤。"""
        net_config = NetworkAssembler.load_json(fixture_artifacts.unet_config_json)
        net_config["scale"] = 1.4  # config 字面死参数陷阱（ADR-0002）
        unet = NetworkAssembler.unet(NetworkArtifact(config=net_config))
        assert unet is not None
