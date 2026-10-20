"""梯度检查点解耦的测试档（#233）：装配探针 fail-fast + 端到端透明性锚。

- **单元档（无标记，纯 CPU）**：fixture 装配走 ``GroupPolicy.build``
  ——默认开（wrapper 在场、探针通过即装配成功）、fixture 关档（无
  wrapper）、探针检测力（注入前向变异 → bitwise 失配 fail-fast；
  注入 buffer → 零状态守卫 fail-fast）；
  生产 config 关闭拒绝走 schema 守卫测试档
  （tests/test_config_schema.py 生产守卫族）。
- **端到端锚（slow+gpu）**：on/off 两 run 同 seed——训练结果逐位一致
  （检查点对更新相数值零影响的全程形态）。
"""

from pathlib import Path

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)

from cynosure.config import CynosureConfig
from cynosure.fixtures import Fixture
from cynosure.train.policy import GroupPolicy
from cynosure.train import checkpointing as checkpointing_module
from tests.conftest import CliSession, FixtureArtifactLibrary, RunTrajectory
from tests.test_async_executor import ExecutorScenario


def _build_policy(config: CynosureConfig) -> GroupPolicy:
    return GroupPolicy.build(
        config, torch.Generator().manual_seed(0), torch.device("cpu"),
    )


def _has_checkpoint_wrappers(network: torch.nn.Module) -> bool:
    return any(
        isinstance(module, CheckpointWrapper) for module in network.modules()
    )


class TestAssembly:
    def test_default_enables_checkpointing(self, cli: CliSession) -> None:
        """默认 config（fixture_mode=true）装配即开：resnet 块被包装、
        bitwise 探针通过（装配成功本身即探针绿的形态）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        assert config.policy.gradient_checkpointing is True
        policy = _build_policy(config)
        assert _has_checkpoint_wrappers(policy.network)

    def test_fixture_mode_allows_disabled(self, cli: CliSession) -> None:
        """fixture 关档：装配通过且无包装（no-op 缝的真实形态）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        config.policy.gradient_checkpointing = False
        policy = _build_policy(config)
        assert not _has_checkpoint_wrappers(policy.network)

    def test_probe_fail_fast_on_forward_mutation(
        self,
        cli: CliSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """探针检测力（fail-fast 门不是装饰）：包装时注入权重变异 →
        wrapped 腿与 plain 腿梯度分叉 → 装配期 ValueError。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        real_wrapper = checkpointing_module.checkpoint_wrapper

        def tampering(module, **kwargs):
            wrapped = real_wrapper(module, **kwargs)
            with torch.no_grad():
                next(wrapped.parameters()).add_(1e-2)
            return wrapped

        monkeypatch.setattr(checkpointing_module, "checkpoint_wrapper", tampering)
        with pytest.raises(ValueError, match="bitwise"):
            _build_policy(config)

    def test_probe_fail_fast_on_buffer_mutation(
        self,
        cli: CliSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """零状态扰动守卫的判别力（#233 评审补强）：包装层注入新 buffer
        → 探针前后键集漂移 → 装配期 ValueError（「探针不改网络状态」
        是受守卫不变式的失败形态）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        real_wrapper = checkpointing_module.checkpoint_wrapper

        def tampering(module, **kwargs):
            wrapped = real_wrapper(module, **kwargs)
            wrapped.register_buffer("injected", torch.zeros(1))
            return wrapped

        monkeypatch.setattr(checkpointing_module, "checkpoint_wrapper", tampering)
        with pytest.raises(ValueError, match="状态守卫"):
            _build_policy(config)


class TestToleranceTier:
    """DTK 容差档的分支逻辑（#233 归因决议）：CPU 上以 monkeypatch 模拟
    hip 后端，只守分支选择与容差界——生产网络 × 真实 DTK 的实测证明
    在集群装配面（探针审计打印落 pretrain/train 日志）。"""

    @staticmethod
    def _probe() -> checkpointing_module.GradientCheckpointing:
        # 断言面单测不担 config/网络构建重负，跳过 __init__ 直取比较逻辑
        return checkpointing_module.GradientCheckpointing.__new__(
            checkpointing_module.GradientCheckpointing,
        )

    def test_within_tolerance_passes_with_audit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(torch.version, "hip", "6.3.0")
        probe = self._probe()
        plain = torch.tensor([1.0, -2.0])
        probe._assert_recompute_parity(
            [plain], [plain + 5e-4], plain, plain + 5e-4,
        )
        assert "容差档通过" in capsys.readouterr().out

    def test_beyond_tolerance_fails_fast(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.version, "hip", "6.3.0")
        probe = self._probe()
        plain = torch.tensor([1.0, -2.0])
        with pytest.raises(ValueError, match="重算一致性"):
            probe._assert_within_tolerance(plain, plain + 0.5, "参数梯度")

    def test_presence_mismatch_rejected_in_tolerance_tier(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(torch.version, "hip", "6.3.0")
        probe = self._probe()
        grad = torch.ones(1)
        with pytest.raises(ValueError, match="在场性失配"):
            probe._assert_recompute_parity(
                [grad, None], [grad, grad], grad, grad,
            )

    def test_bitwise_tier_untouched_on_cpu(self) -> None:
        """hip 缺省缺席（CPU/CUDA）：超界失配走 bitwise 文案——守卫
        强度在确定性内核层不因容差档放松的形态。"""
        probe = self._probe()
        plain = torch.tensor([1.0])
        with pytest.raises(ValueError, match="bitwise"):
            probe._assert_recompute_parity(
                [plain], [plain + 0.5], plain, plain,
            )


@pytest.mark.gpu
@pytest.mark.slow
class TestCheckpointingEndToEnd:
    def test_on_off_bitwise_identical_training(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """双档全程锚：同 seed 两 run（一开一关）——权重逐位一致
        （NO_REENTRANT 重算对更新相数值零影响的端到端形态）。"""
        scenario = ExecutorScenario(cli, tmp_path)
        on_config = scenario.prepare(seed=8)
        off_config = scenario.prepare(seed=8)
        off_config.policy.gradient_checkpointing = False
        enabled = scenario.build(on_config, run_name="on")
        enabled.run()
        disabled = scenario.build(off_config, run_name="off")
        disabled.run()
        scenario.assert_state_dicts_bitwise(
            enabled.cards[0].replica.policy.full_state(),
            disabled.cards[0].replica.policy.full_state(),
            "policy(on vs off)",
        )
        assert RunTrajectory(scenario.events("on")) == RunTrajectory(
            scenario.events("off"),
        )
