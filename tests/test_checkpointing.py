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
