"""cynosure 命令行：train / eval / prepare 三子命令与 config schema 校验
——全库唯一测试 seam（spec「Testing Decisions」）。

三子命令共享同一 config schema；本 ticket（#16，prefactor）交付 seam 与
run 目录契约最小版，训练 / 评测的实际循环由后续 ticket 填充（prepare
数据工件管线已由 #18 交付，fixture 合成数据下端到端可跑）。

单进程 seam：torchrun 入口（FSDP 初始化等）属 orchestration ticket；
分布式启动须经显式 --run-dir（跨 rank 的 run 目录 barrier 见 train.RunArtifacts）。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.reward import PreparePipeline, SyntheticLatentEncoder
from cynosure.train import RunArtifacts

_EXIT_USAGE_ERROR = 2


class CynosureCli:
    """命令分发（Command Pattern）：参数解析、config 校验、子命令 handler。"""

    def __init__(self, argv: list[str], stdout: TextIO, stderr: TextIO) -> None:
        self._argv = argv
        self._stdout = stdout
        self._stderr = stderr

    def run(self) -> int:
        parser = self._build_parser()
        args = parser.parse_args(self._argv)
        # 三子命令共享同一 config schema：dispatch 前统一校验
        config = self._load_config(args.config)
        if config is None:
            return _EXIT_USAGE_ERROR
        handlers = {
            "train": self._train,
            "eval": self._eval,
            "prepare": self._prepare,
        }
        return handlers[args.command](args, config)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="cynosure",
            description="MAISI 3D latent rectified-flow checkpoint 的 Granular-GRPO RL 后训练",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)
        for name, help_text in (
            ("train", "启动 RL 训练（run 目录 + 指标流 + checkpoint）"),
            ("eval", "从 checkpoint + Real sample pool 产出评测指标"),
            ("prepare", "构建 Real sample pool / Held-out real / per-channel 统计量"),
        ):
            sub = subparsers.add_parser(name, help=help_text)
            sub.add_argument("--config", required=True, help="config JSON 路径")
            if name in ("train", "eval"):
                sub.add_argument(
                    "--run-dir", default=None,
                    help="run 目录（默认 $HOME/.cynosure/runs/<时间戳>-<组>）",
                )
        return parser

    def _load_config(self, config_arg: str) -> CynosureConfig | None:
        path = Path(config_arg)
        if not path.is_file():
            print(f"config 文件不存在: {path}", file=self._stderr)
            return None
        try:
            return ConfigLoader.load(path)
        except ValidationError as exc:
            print("config 校验失败：", file=self._stderr)
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                print(f"  - {location}: {error['msg']}", file=self._stderr)
            return None
        except json.JSONDecodeError as exc:
            print(f"config 不是合法 JSON: {exc}", file=self._stderr)
            return None

    def _train(self, args: argparse.Namespace, config: CynosureConfig) -> int:
        if args.run_dir is None and "RANK" in os.environ:
            # 默认 run 目录按进程时间戳生成：多 rank 下无法对齐、会静默分裂 run
            print(
                "检测到 torchrun 环境（RANK="
                f"{os.environ['RANK']}）：默认 run 目录按进程时间戳生成、"
                "无法跨 rank 对齐，分布式启动必须显式指定 --run-dir",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        run_root = (
            Path(args.run_dir) if args.run_dir
            else RunArtifacts.default_root(config)
        )
        try:
            artifacts = RunArtifacts.init(config, run_root)
        except FileExistsError:
            print(
                f"run 目录已存在（不静默覆盖；续训请走续训入口）: {run_root}",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        except TimeoutError:
            print(f"等待 rank 0 创建 run 目录超时: {run_root}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(f"run 目录已就绪: {artifacts.paths.root}", file=self._stdout)
        if os.environ.get("RANK") in (None, "0"):  # 非 0 rank 只采用、不落盘
            print(
                "工件契约最小版已落盘：config.json / metrics.jsonl / manifest.json /"
                " checkpoints/（训练循环由后续 ticket 填充）",
                file=self._stdout,
            )
        return 0

    def _eval(
        self, args: argparse.Namespace, config: CynosureConfig,
    ) -> int:
        if args.run_dir is not None:
            run_root = Path(args.run_dir)
            if not run_root.is_dir():
                print(f"run 目录不存在: {run_root}", file=self._stderr)
                return _EXIT_USAGE_ERROR
            # milestone 事件写入既有 run 目录的 metrics 流（eval ticket 交付）
            print(f"评测目标 run 目录: {run_root}", file=self._stdout)
        print(
            f"评测计划：group={config.experiment.group}"
            f" N_baseline={config.schedule.baseline_samples}"
            f" 里程碑间隔={config.schedule.milestone_interval} iteration"
            "（像素域 2.5D FID 基础设施由 eval ticket 交付）",
            file=self._stdout,
        )
        return 0

    def _prepare(
        self, args: argparse.Namespace, config: CynosureConfig,
    ) -> int:
        if not config.fixture_mode:
            # 生产预编码（MONAI AutoencoderKlMaisi 装载 vae_ckpt）待基座
            # checkpoint 落地后的 ticket 校准交付，当前显式拒绝、不静默产出
            print(
                "prepare 当前仅支持 fixture 合成数据端到端：生产 config 须"
                "经 fixture_mode=true 显式声明（MONAI VAE 预编码由后续 "
                "ticket 交付）",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        try:
            report = PreparePipeline(config, SyntheticLatentEncoder()).run()
        except (ValueError, FileNotFoundError) as exc:
            print(f"prepare 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(
            f"prepare 完成（病例级 split seed={config.schedule.seed}："
            f"train {report.split_sizes['train']} / val "
            f"{report.split_sizes['val']} / test {report.split_sizes['test']}）:",
            file=self._stdout,
        )
        print(
            f"  - Real sample pool: {report.pool_manifest}"
            f"（{report.pool_entries} 条，按序列分层）",
            file=self._stdout,
        )
        print(
            f"  - Held-out real: {report.heldout_manifest}"
            f"（{report.heldout_entries} 条，与 pool 病例级不相交、"
            "永不参与判别器更新）",
            file=self._stdout,
        )
        print(
            f"  - per-channel 标准化统计量: {report.channel_stats}"
            f"（mean/std × {len(report.mean)} 通道）",
            file=self._stdout,
        )
        return 0


def main() -> None:
    """console-script 入口。"""
    raise SystemExit(CynosureCli(sys.argv[1:], sys.stdout, sys.stderr).run())
