"""fixture 生成器 + netbuild 测试：MONAI 微型网络的可前向配置（CPU 可跑）、
fixture 数值锚（3 步 ODE、G 保持 12、latent [4,16,16,8]、input_img_size_numel=2048）。"""

import json
from pathlib import Path

import pytest
import torch
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.netbuild import NetworkArtifact, NetworkAssembler


class TestFixtureConfig:
    def test_fixture_config_passes_schema(self, tmp_path: Path) -> None:
        config = Fixture().config(tmp_path)
        assert isinstance(config, CynosureConfig)

    def test_fixture_config_passes_schema_for_all_three_groups(
        self, tmp_path: Path,
    ) -> None:
        """三组配置矩阵走同一 config schema（issue #23 验收「测试面 #4 绿」
        的 fixture 面）：组1/组2/组3 的 fixture config 全字段校验通过。"""
        for group in ("modal-label", "cross-modal", "sequential"):
            config = Fixture().config(tmp_path, group=group)  # type: ignore[arg-type]
            assert config.experiment.group == group

    def test_cross_modal_fixture_config_carries_controlnet_artifacts(
        self, tmp_path: Path,
    ) -> None:
        """组2/组3 config 携带 ControlNet 工件对（ckpt + 网络配置 JSON，
        schema 强制）。"""
        config = Fixture().config(tmp_path, group="cross-modal")
        assert config.artifacts.controlnet_ckpt == tmp_path / "controlnet.pt"
        assert (
            config.artifacts.controlnet_config_json
            == tmp_path / "controlnet_config.json"
        )

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
        assert artifacts.modality_mapping_json.is_file()
        assert artifacts.controlnet_ckpt.is_file()
        assert artifacts.controlnet_config_json.is_file()

    def test_controlnet_artifact_loads_and_forward_injects_residuals(
        self, tmp_path: Path,
    ) -> None:
        """netbuild 扩展 ControlNet（issue #23）：fixture ControlNet 按
        artifact 装载、前向产出残差并注入同构 fixture UNet（残差注入
        管线 CPU 可跑）；controlnet 残差块零初始化被 fixture 重初始化，
        源影像条件真实参与前向（防组合场退化式静默复现）。"""
        torch.manual_seed(7)
        artifacts = Fixture().write_artifacts(tmp_path)
        controlnet = NetworkAssembler.controlnet(NetworkArtifact(
            config=NetworkAssembler.load_json(artifacts.controlnet_config_json),
            checkpoint=artifacts.controlnet_ckpt,
        ))
        x = torch.randn(2, 4, 16, 16, 8)
        timesteps = torch.tensor([442, 442])
        labels = torch.tensor([29, 34])
        down_residuals, mid_residual = controlnet(
            x=x, timesteps=timesteps, controlnet_cond=x, class_labels=labels,
        )
        assert mid_residual.shape[0] == 2
        assert sum(r.abs().sum().item() for r in down_residuals) > 0.0
        assert mid_residual.abs().sum().item() > 0.0  # 非零初始化：条件真实参与
        unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(artifacts.unet_config_json),
            checkpoint=artifacts.unet_ckpt,
        ))
        out = unet(
            x=x, timesteps=timesteps, class_labels=labels,
            spacing_tensor=torch.full((2, 3), 100.0),
            down_block_additional_residuals=tuple(down_residuals),
            mid_block_additional_residual=mid_residual,
        )
        assert out.shape == x.shape

    def test_modality_mapping_artifact_covers_four_modalities(
        self, tmp_path: Path,
    ) -> None:
        """spec 输入物 modality_mapping 以工件形式落盘（诊断经它装载标签，
        不设代码内常量副本）。"""
        artifacts = Fixture().write_artifacts(tmp_path)
        mapping = json.loads(artifacts.modality_mapping_json.read_text(encoding="utf-8"))
        assert mapping == {"t1n": 29, "t1c": 34, "t2w": 30, "t2f": 31}

    def test_same_seed_reproduces_weights(self, tmp_path: Path) -> None:
        torch.manual_seed(7)
        first = Fixture().unet().state_dict()
        torch.manual_seed(7)
        second = Fixture().unet().state_dict()
        for key in first:
            assert torch.equal(first[key], second[key]), key

    def test_fixture_unet_is_condition_sensitive(self) -> None:
        """条件敏感性守卫（AC「组合场逐字对齐」的前提）：fixture UNet 的
        velocity 须真实响应 label / timestep / spacing——MONAI 的 zero-init
        末层卷积会把未训练网络的 resnet 条件通道数值归零，fixture 对全零
        卷积重初始化，此测试防该修复回退（组合场退化静默复现）。"""
        torch.manual_seed(7)
        unet = Fixture().unet()
        x = torch.randn(1, 4, 16, 16, 8)
        with torch.no_grad():
            kwargs = dict(x=x, spacing_tensor=torch.full((1, 3), 100.0))
            v29 = unet(timesteps=torch.tensor([442]), class_labels=torch.tensor([29]), **kwargs)
            v0 = unet(timesteps=torch.tensor([442]), class_labels=torch.tensor([0]), **kwargs)
            v165 = unet(timesteps=torch.tensor([165]), class_labels=torch.tensor([29]), **kwargs)
            vsp = unet(
                timesteps=torch.tensor([442]), class_labels=torch.tensor([29]),
                x=x, spacing_tensor=torch.full((1, 3), 200.0),
            )
        assert not torch.equal(v29, v0)  # 全零 label ≠ 条件 label（无条件分支语义的前提）
        assert not torch.equal(v29, v165)  # timestep 影响 velocity（日程真实参与）
        assert not torch.equal(v29, vsp)  # spacing 影响 velocity（恒传条件真实参与）


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

    def test_production_schedule_anchors_monai_transform_output(self) -> None:
        """生产数值锚 131072（= 64·64·32）：日程 = MONAI set_timesteps 实际
        输出，timestep transform 生效（无 transform 为 [1000,967,933,…]），
        实际 scale=1.0——config 字面 scale=1.4 是死参数，禁止照抄（ADR-0002）。"""
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=30, input_img_size_numel=131072,
        )
        assert scheduler.timesteps.tolist() == [
            1000, 979, 956, 934, 912, 888, 864, 839, 813, 787,
            760, 732, 704, 675, 644, 613, 581, 548, 514, 479,
            442, 404, 366, 325, 284, 241, 195, 149, 102, 51,
        ]

    def test_unet_config_json_roundtrip_unknown_keys_ignored(
        self, fixture_artifacts: object,
    ) -> None:
        """网络配置 JSON 里的非构造参数（如基座死参数 scale）被静默过滤。"""
        net_config = NetworkAssembler.load_json(fixture_artifacts.unet_config_json)
        net_config["scale"] = 1.4  # config 字面死参数陷阱（ADR-0002）
        unet = NetworkAssembler.unet(NetworkArtifact(config=net_config))
        assert unet is not None


class TestNetbuildVae:
    """netbuild VAE 装配（AutoencoderKlMaisi）：网络配置 JSON + checkpoint
    构建、装载、前向与直接构造一致（生产预编码侧的装载契约）。"""

    VAE_CONFIG: dict = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "num_res_blocks": [1, 1],
        "num_channels": [4, 8],
        "attention_levels": [False, False],
        "latent_channels": 4,
        "norm_num_groups": 4,
        "include_fc": False,
        "use_combined_linear": False,
        "num_splits": 1,  # 微型体积小于默认 16 切分的 save_mem 下限
    }

    def test_vae_assembly_loads_checkpoint_and_matches_direct_construction(
        self, tmp_path: Path,
    ) -> None:
        torch.manual_seed(3)
        direct = AutoencoderKlMaisi(**self.VAE_CONFIG).eval()
        ckpt = tmp_path / "vae.pt"
        torch.save(direct.state_dict(), ckpt)
        (tmp_path / "vae_config.json").write_text(
            json.dumps(self.VAE_CONFIG), encoding="utf-8",
        )
        artifact = NetworkArtifact(
            config=NetworkAssembler.load_json(tmp_path / "vae_config.json"),
            checkpoint=ckpt,
        )
        assembled = NetworkAssembler.vae(artifact).eval()
        image = torch.randn(1, 1, 8, 8, 4)
        with torch.no_grad():
            mu_direct, _ = direct.encode(image)
            mu_assembled, _ = assembled.encode(image)
        assert torch.equal(mu_direct, mu_assembled)
        assert mu_assembled.shape[1] == self.VAE_CONFIG["latent_channels"]
