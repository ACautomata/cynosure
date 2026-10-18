"""metrics.jsonl → plots/ 逐指标 SVG 的共享绘图模块（issue #256）。

train run 与 pretrain run 共用：输入一份 run 的 metrics.jsonl 与输出目录，
自动发现全部事件（iter / milestone / pretrain / 各类 alert）中的数值字段
并成图——数值平铺字段逐指标一张 SVG；dict 数值字段（loss、phase_seconds）
的子键展开为同图内多序列叠加（图例区分子键）；同名数值字段跨事件类型合并
到同一张图。x 轴取 iteration（train 侧事件）或 step（pretrain 事件），由
数据决定。matplotlib Agg 后端直出矢量 SVG；None 与非数值（str/bool/非
数值子键）跳过。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# 图例/轴文字以 <text> 落 SVG（默认 fonttype=path 会把文字转成 glyph
# 路径——不可检索、不可访问）
matplotlib.rcParams["svg.fonttype"] = "none"

from matplotlib import pyplot as plt

_X_AXIS_FIELDS = ("iteration", "step")
"""x 轴候选字段（按序取用）：train 侧事件带 iteration、pretrain 事件带
step——由数据决定，二者互斥。轴字段本身不作为 y 指标成图。"""


class MetricsPlotter:
    """指标流的逐指标绘图器：metrics.jsonl → 输出目录每数值指标一张 SVG。"""

    def __init__(self, metrics_path: Path, output_dir: Path) -> None:
        self.metrics_path = Path(metrics_path)
        self.output_dir = Path(output_dir)

    def write(self) -> list[Path]:
        """发现指标序列并逐指标渲染 SVG，返回产出文件路径。"""
        events = self._read_events()
        series = self._collect_series(events)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        produced = []
        for field, sequences in series.items():
            path = self.output_dir / f"{field}.svg"
            self._render(path, field, sequences)
            produced.append(path)
        return produced

    def _read_events(self) -> list[dict]:
        text = self.metrics_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def _collect_series(
        self, events: list[dict],
    ) -> dict[str, dict[tuple[str, str], list[tuple[float, float]]]]:
        """数值字段自动发现：字段名 → (x 轴名, 序列标签) → 点列。

        平铺字段的序列标签为空串（单序列不成图例，混 x 轴时图例以轴名
        区分）；dict 字段按数值子键展开（子键即标签）；None 与非数值
        （str/bool/非数值子键）跳过。
        """
        series: dict[str, dict[tuple[str, str], list[tuple[float, float]]]] = {}
        for event in events:
            axis = self._axis_of(event)
            if axis is None:
                continue
            axis_name, x = axis
            for field, value in event.items():
                if field in _X_AXIS_FIELDS:
                    continue
                if self._is_numeric(value):
                    points = series.setdefault(field, {}).setdefault(
                        (axis_name, ""), [],
                    )
                    points.append((x, float(value)))
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if self._is_numeric(subvalue):
                            points = series.setdefault(field, {}).setdefault(
                                (axis_name, str(subkey)), [],
                            )
                            points.append((x, float(subvalue)))
        return series

    def _axis_of(self, event: dict) -> tuple[str, float] | None:
        for name in _X_AXIS_FIELDS:
            value = event.get(name)
            if self._is_numeric(value):
                return name, float(value)
        return None

    @staticmethod
    def _is_numeric(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _render(
        self,
        path: Path,
        field: str,
        sequences: dict[tuple[str, str], list[tuple[float, float]]],
    ) -> None:
        fig, ax = plt.subplots()
        axis_names: list[str] = []
        for (axis_name, label), points in sorted(
            sequences.items(),
            key=lambda item: (_X_AXIS_FIELDS.index(item[0][0]), item[0][1]),
        ):
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            ax.plot(xs, ys, marker="o", markersize=3, label=label or axis_name)
            if axis_name not in axis_names:
                axis_names.append(axis_name)
        ax.set_xlabel(" / ".join(axis_names))
        if len(sequences) > 1:
            ax.legend()
        ax.set_ylabel(field)
        fig.savefig(path, format="svg")
        plt.close(fig)
