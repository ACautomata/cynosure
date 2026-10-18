"""metrics.jsonl → plots/ 逐指标 SVG 绘图模块的行为验证（issue #256）。

全部经由公共 seam（合成 metrics.jsonl → 断言输出目录产物）独立验证，
不接 CLI（issue #256 明文本票不接线）。"""

import json
from pathlib import Path

from cynosure.plots import MetricsPlotter


def write_metrics(path: Path, events: list[dict]) -> None:
    lines = [json.dumps(event, ensure_ascii=False) for event in events]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_event(iteration: int, **overrides: object) -> dict:
    event = {
        "event": "iter",
        "iteration": iteration,
        "modality": "t1n",
        "anchor_eval_reward": 0.5 + iteration * 0.1,
        "intra_group_reward_std": 0.1,
        "lr": 1e-4,
        "elapsed_s": 30.0 + iteration,
        "loss": {"policy": 0.4, "discriminator": 0.2},
    }
    event.update(overrides)
    return event


def milestone_event(iteration: int, **overrides: object) -> dict:
    event = {
        "event": "milestone",
        "iteration": iteration,
        "fid": 120.0 - iteration,
        "kid": 0.02,
        "ssim": 0.7,
        "mae": 0.25,
        "psnr": 21.0,
        "elapsed_s": 15.0 + iteration,
    }
    event.update(overrides)
    return event


class TestSvgOutput:
    def test_flat_numeric_fields_one_svg_each(self, tmp_path: Path) -> None:
        """数值平铺字段逐指标一张 SVG：iter 与 milestone 事件的全部数值
        字段各成一张图；同名跨事件类型合并单图（elapsed_s 共一张）。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [iter_event(0), iter_event(1), milestone_event(1)])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        produced = {path.name for path in output.glob("*.svg")}
        assert produced == {
            "anchor_eval_reward.svg",
            "intra_group_reward_std.svg",
            "lr.svg",
            "elapsed_s.svg",
            "loss.svg",
            "fid.svg",
            "kid.svg",
            "ssim.svg",
            "mae.svg",
            "psnr.svg",
        }
        for path in output.glob("*.svg"):
            content = path.read_text(encoding="utf-8")
            assert "<svg" in content, path.name
            assert "<image" not in content, path.name

    def test_dict_field_subkeys_stack_in_one_figure(self, tmp_path: Path) -> None:
        """dict 数值字段（loss）一张图内多序列叠加：子键进图例；产物为
        矢量 SVG（<svg 头、无内嵌位图）。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [
            iter_event(0, loss={"policy": 0.4, "discriminator": 0.2}),
            iter_event(1, loss={"policy": 0.3, "discriminator": 0.25}),
        ])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        content = (output / "loss.svg").read_text(encoding="utf-8")
        assert "<svg" in content
        assert "<image" not in content
        assert "policy" in content
        assert "discriminator" in content

    def test_empty_jsonl_creates_output_dir_without_svgs(self, tmp_path: Path) -> None:
        """0 事件（空 jsonl）边界：输出目录存在、无 .svg、不抛异常。"""
        metrics = tmp_path / "metrics.jsonl"
        metrics.write_text("", encoding="utf-8")
        output = tmp_path / "plots"

        produced = MetricsPlotter(metrics, output).write()

        assert output.is_dir()
        assert list(output.glob("*.svg")) == []
        assert produced == []


class TestFieldDiscovery:
    def test_new_numeric_field_plots_without_code_change(self, tmp_path: Path) -> None:
        """字段自动发现：合成事件新增数值字段无需改绘图代码即可成图。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [iter_event(0, **{"new_metric_v2": 1.5})])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        assert (output / "new_metric_v2.svg").exists()

    def test_none_str_bool_fields_skipped(self, tmp_path: Path) -> None:
        """None 与非数值（str/bool）不成图。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [iter_event(
            0,
            heldout_auc=None,
            modality="t1n",
            early_stop=False,
        )])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        produced = {path.name for path in output.glob("*.svg")}
        assert "heldout_auc.svg" not in produced
        assert "modality.svg" not in produced
        assert "early_stop.svg" not in produced

    def test_non_numeric_dict_subkeys_skipped(self, tmp_path: Path) -> None:
        """dict 字段的非数值子键（str/None/嵌套 dict）跳过，数值子键照常。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [iter_event(0, breakdown={
            "score": 0.9,
            "weight": 0.1,
            "label": "good",
            "missing": None,
            "nested": {"deep": 1.0},
        })])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        content = (output / "breakdown.svg").read_text(encoding="utf-8")
        assert "score" in content
        assert "weight" in content
        assert "label" not in content
        assert "missing" not in content
        assert "nested" not in content

    def test_same_field_across_event_types_merges(self, tmp_path: Path) -> None:
        """同名数值字段跨事件类型合并到同一张图：train_pairwise_acc 同时
        出现在 iter 与 overfit_alert 事件，共一张图。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [
            iter_event(0, train_pairwise_acc=0.6),
            {
                "event": "overfit_alert",
                "iteration": 1,
                "stage": 1,
                "modality": "t1n",
                "divergence_ema": 0.2,
                "train_pairwise_acc": 0.65,
                "heldout_auc": 0.45,
            },
        ])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        produced = {path.name for path in output.glob("*.svg")}
        assert "train_pairwise_acc.svg" in produced
        assert "divergence_ema.svg" in produced
        assert "heldout_auc.svg" in produced


class TestXAxis:
    def test_pretrain_events_use_step_axis(self, tmp_path: Path) -> None:
        """x 轴由数据决定：pretrain 事件（无 iteration）取 step，轴名落
        xlabel。"""
        metrics = tmp_path / "metrics.jsonl"
        write_metrics(metrics, [
            {
                "event": "pretrain",
                "step": step,
                "modality": "t1n",
                "loss_discriminator": 0.7 - step * 0.01,
                "heldout_auc": 0.6,
                "lr": 1e-4,
                "elapsed_s": 5.0 + step,
            }
            for step in range(3)
        ])
        output = tmp_path / "plots"

        MetricsPlotter(metrics, output).write()

        content = (output / "loss_discriminator.svg").read_text(encoding="utf-8")
        assert "step" in content
        assert "iteration" not in content
