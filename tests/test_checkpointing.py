"""梯度检查点解耦的测试档（#233）：装配探针 fail-fast + config 双档 +
端到端透明性锚。

- **单元档（无标记，纯 CPU）**：fixture 装配走 ``GroupPolicy.build``
  ——默认开（wrapper 在场、探针通过即装配成功）、fixture 关档（无
  wrapper）、生产 config 关闭拒绝（schema 守卫）、探针检测力（注入
  前向变异 → bitwise 失配 fail-fast）。
- **端到端锚（slow+gpu）**：on/off 两 run 同 seed——训练结果逐位一致
  （检查点对更新相数值零影响的全程形态）。
"""

import json
from pathlib import Path

import pytest
import pydantic
import torch

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
        "checkpointwrapper" in type(module).__name__.lower()
        for module in network.modules()
    )


class TestAssembly:
    def test_default_enables_checkpointing(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """默认 config（fixture_mode=true）装配即开：resnet 块被包装、
        bitwise 探针通过（装配成功本身即探针绿的形态）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        assert config.policy.gradient_checkpointing is True
        policy = _build_policy(config)
        assert _has_checkpoint_wrappers(policy.network)

    def test_fixture_mode_allows_disabled(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """fixture 关档：装配通过且无包装（no-op 缝的真实形态）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        config = Fixture().config(fixture_dir)
        config.policy.gradient_checkpointing = False
        policy = _build_policy(config)
        assert not _has_checkpoint_wrappers(policy.network)

    def test_production_rejects_disabled(
        self, cli: CliSession, tmp_path: Path,
    ) -> None:
        """生产 config（fixture_mode=false）关闭检查点 = schema 拒绝
        （#217 §4：训练相激活全量驻留的 OOM 风险，静默关闭不合规）。"""
        fixture_dir = FixtureArtifactLibrary.artifacts_dir(cli, "modal-label")
        payload = json.loads(
            Fixture().config(fixture_dir).model_dump_json(),
        )
        payload["fixture_mode"] = False
        # 其余 fixture 守卫（日程/样本量/resize 基数）先满足——本测试只
        # 让 gradient_checkpointing 的生产拒绝成为唯一违例
        payload["policy"]["num_inference_steps"] = 30
        payload["schedule"]["baseline_samples"] = 200
        payload["preprocessing"]["resize_base"] = 128
        payload["policy"]["gradient_checkpointing"] = False
        with pytest.raises(pydantic.ValidationError, match="gradient_checkpointing"):
            CynosureConfig.model_validate(payload)

    def test_probe_fail_fast_on_forward_mutation(
        self,
        cli: CliSession,
        tmp_path: Path,
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
