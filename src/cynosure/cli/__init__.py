"""cynosure 命令行：train / eval / prepare / pretrain / fid / fid-floor
子命令与 config schema 校验——全库唯一测试 seam（spec「Testing Decisions」）。

四子命令共享同一 config schema；dispatch 前统一校验。train 执行
Granular-GRPO 训练循环（MGAI → 逐 k 梯度步 → 判别器 Online update →
iter 事件流 + checkpoint），执行序 = async 单进程多卡门面
（``AsyncTrainingExecutor``：每卡完整副本 + 静态分配表轮 + 逐 k barrier
收集-同步，#217/#226 切换期第一步起生产入口单口径）；torchrun 多进程
启动显式拒绝（多进程拓扑属已退役的旧执行序），``--dump-trajectory``
额外产出 fixture 轨迹诊断工件（组1 采样场对照，``trajectory.json``）；
``--resume`` 从既有 run 目录的 v12 续训分片恢复训练（须显式 --run-dir）。
pretrain 执行判别器 warm-start 预训练（ADR-0007）：密集步进至 held-out
AUC 达过线阈值（预训练棘轮）或步数上限，产出判别器 checkpoint + 预训练报告；
单进程多卡 async 执行（ADR-0018），torchrun 启动显式拒绝。
fid/fid-floor/base-smoke 保持单进程（ADR-0018 决策 10：RANK 注入源
死化后无守卫面——torchrun 环境下误启动属操作错误，非代码路径）。

fid 是裁决性 MR FID 读数仪器（#73 双轨之一，wayfinder #79 移植）：
独立 ``MrFidConfig`` schema（9 项冻结变量的载体），单进程执行，
特征缓存带几何口径 fingerprint 防护。fid-floor 是 real-vs-real
地板的病例级半分工具（#73 裁决八）：读 MR-RATE 评估 manifest，
seed 冻结落盘（split_record.json + 逐格双侧清单），产物直接喂 fid。
两子命令与训练族 config 完全分离（fid 不需要训练工件）。base-smoke
是基座 checkpoint 装载与前向自检（wayfinder #120）：独立
``BaseSmokeConfig`` schema，单进程执行，装载上游发布件（UNet 训练
checkpoint 容器 + VAE 裸权重）后出定点前向指纹与 VAE 生产尺寸往返
读数，报告落盘供 #121/#122 消费。
"""

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import TextIO, TypeVar

import torch
from pydantic import BaseModel, ValidationError

from cynosure.config import ConfigLoader, CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.eval.mr_fid import MrFidConfig, MrFidInstrument
from cynosure.eval.real_real_floor import (
    DEFAULT_PATH_TEMPLATE,
    DEFAULT_SEED,
    RealRealFloorSplit,
)
from cynosure.policy import TrajectoryDiagnosticRunner
from cynosure.pretrain import PretrainRun
from cynosure.pretrain.driver import PretrainDriver
from cynosure.reward import PreparePipeline
from cynosure.smoke import BaseSmokeConfig, BaseSmokeRunner
from cynosure.train import RunArtifacts
from cynosure.train.executor import AsyncTrainingExecutor, TrainingAborted

_EXIT_USAGE_ERROR = 2

_SchemaT = TypeVar("_SchemaT", bound=BaseModel)
"""独立 schema 子命令的装载返回类型（``fid`` 的 MrFidConfig / ``base-smoke``
的 BaseSmokeConfig）。"""


class CynosureCli:
    """命令分发（Command Pattern）：参数解析、config 校验、子命令 handler。"""

    def __init__(self, argv: list[str], stdout: TextIO, stderr: TextIO) -> None:
        self._argv = argv
        self._stdout = stdout
        self._stderr = stderr

    def run(self) -> int:
        parser = self._build_parser()
        args = parser.parse_args(self._argv)
        # fid / fid-floor 走独立 schema（裁决性评测仪器，与训练 config 分离）
        if args.command == "fid":
            return self._fid(args)
        if args.command == "fid-floor":
            return self._fid_floor(args)
        # base-smoke 走独立 schema（基座装载自检，与训练 config 分离）
        if args.command == "base-smoke":
            return self._base_smoke(args)
        # 四子命令共享同一 config schema：dispatch 前统一校验
        config = self._load_config(args.config)
        if config is None:
            return _EXIT_USAGE_ERROR
        handlers = {
            "train": self._train,
            "eval": self._eval,
            "prepare": self._prepare,
            "pretrain": self._pretrain,
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
            ("pretrain", "判别器 warm-start 预训练（warm-start 装载守卫的产物源）"),
            ("fid", "裁决性 MR FID 读数（fork 口径 2.5D 仪器，#73 双轨）"),
            ("fid-floor", "real-vs-real 地板半分（病例级 seed 冻结 + 逐格清单）"),
            ("base-smoke", "基座 checkpoint 装载与前向自检（wayfinder #120）"),
        ):
            sub = subparsers.add_parser(name, help=help_text)
            if name == "fid-floor":
                sub.add_argument(
                    "--manifest", required=True, type=Path,
                    help="MR-RATE 评估 manifest（eval_manifest.csv，#78 工件）",
                )
                sub.add_argument(
                    "--seed", type=int, default=DEFAULT_SEED,
                    help=f"半分 seed（默认 {DEFAULT_SEED}；冻结记录随落盘，重算必须同值）",
                )
                sub.add_argument(
                    "--output-dir", required=True, type=Path,
                    help="冻结工件输出目录（split_record.json + 逐格双侧清单）",
                )
                sub.add_argument(
                    "--path-template", default=DEFAULT_PATH_TEMPLATE,
                    help="卷路径模板（manifest 行字段插值；默认 MR-RATE "
                         "gauss 落位相对路径）",
                )
                continue
            sub.add_argument("--config", required=True, help="config JSON 路径")
            if name == "fid":
                sub.add_argument(
                    "--comparison-tag", default="",
                    help="自由标识，写进 FidResult provenance"
                         "（如 floor_half_a_vs_half_b、label9_seed42）",
                )
            if name == "train":
                sub.add_argument(
                    "--run-dir", default=None,
                    help="run 目录（默认 $HOME/.cynosure/runs/<时间戳>-<组>）",
                )
                sub.add_argument(
                    "--resume", action="store_true",
                    help="从 run 目录最新续训状态恢复训练（须显式 --run-dir；"
                         "仅单阶段组，config 除 schedule.max_iterations 外"
                         "须与原 run 一致）",
                )
                sub.add_argument(
                    "--dump-trajectory", action="store_true",
                    help="fixture 诊断开关：落盘 per-step 轨迹统计/哈希、log-prob 对"
                         "与双样本分布统计量（限 fixture_mode=true）",
                )
            elif name == "eval":
                sub.add_argument(
                    "--run-dir", default=None,
                    help="run 目录（评测目标 run）",
                )
            elif name == "pretrain":
                sub.add_argument(
                    "--run-dir", default=None,
                    help="预训练 run 目录（默认 = config 的 "
                         "reward.pretrain_report_json 所在目录）",
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
        # 单进程多卡执行序（#217/#226）：多进程拓扑属已退役的旧执行序，
        # 拒绝在 run 目录预占之前（usage 错误不落任何工件）
        env_rank = DistributedContext.env_rank()
        if env_rank is not None:
            print(
                f"train 以单进程多卡执行（async 执行序），"
                f"拒绝 torchrun 启动（RANK={env_rank}）——进程内多卡由"
                "设备发现承担（CUDA_VISIBLE_DEVICES 裁剪卡集）",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        if config.experiment.group == "sequential":
            # 组3 序贯两阶段编排是旧执行序组件（SequentialTrainer，随
            # 切换期退役）：新执行序为单阶段门面，两阶段由两次独立 run
            # 衔接（stage-1 产物 checkpoint 经 config 工件路径装配）
            print(
                "组3（sequential）的序贯两阶段编排已随旧执行序退役："
                "请分别以组1（modal-label）与组2（cross-modal）run 执行，"
                "stage-1 产物 checkpoint 经 config 工件路径衔接",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        resume = args.resume
        if resume and args.run_dir is None:
            print(
                "续训须显式指定 --run-dir（恢复目标 run 目录；默认目录按"
                "时间戳生成，不可能指向既有 run）",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        if args.dump_trajectory and not config.fixture_mode:
            # 诊断工件属 fixture 诊断模式（spec「产物工件契约」）：生产采样
            # 诊断随训练循环 ticket 交付，当前显式拒绝、不建 run 目录
            print(
                "轨迹诊断当前仅支持 fixture（fixture_mode=true）：生产 config "
                "须先经 fixture_mode=true 显式声明",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        if args.dump_trajectory and config.experiment.group != "modal-label":
            # trajectory.json 的 MONAI 直接步对照是组1（CFG=10 组合场）专属
            # 诊断：组2 跳过该工件而非静默产出组1 对照（训练侧 log-prob 对
            # 已随旧执行序退役，async 门面无诊断路径）
            print(
                "轨迹诊断工件（trajectory.json）当前仅覆盖组1"
                "（modal-label）采样场，本运行跳过",
                file=self._stderr,
            )
        run_root = (
            Path(args.run_dir) if args.run_dir
            else RunArtifacts.default_root(config)
        )
        if resume:
            paths = RunArtifacts.layout(run_root)
            if not paths.config_snapshot.is_file():
                print(
                    f"续训目标不是有效 run 目录（缺 config 快照）: {run_root}",
                    file=self._stderr,
                )
                return _EXIT_USAGE_ERROR
            artifacts = RunArtifacts(paths)
            print(f"续训目标 run 目录: {artifacts.paths.root}", file=self._stdout)
        else:
            artifacts = None  # 新 run 的创建在执行序装配之前
        return self._run_training(
            config, artifacts, args.dump_trajectory,
            resume=resume, run_root=run_root,
        )

    def _rollback_untouched_run(self, artifacts: RunArtifacts) -> None:
        """训练装配/启动失败且回滚 run 目录：目录若除 init 契约最小集外
        未产出任何工件（无 iter 事件、无 checkpoint、无诊断工件），删除
        本次预占的目录——用户修复输入后可用同一 --run-dir 重试（run 目录
        已存在语义拒绝重跑）。已有真实产出（如 η=0 对照的 trajectory.json
        在训练拒绝前产出，spec 诊断契约）则保留目录。"""
        paths = artifacts.paths
        produced = (
            paths.metrics.stat().st_size > 0
            or paths.trajectory_diagnostic.exists()
            or any(paths.checkpoints.iterdir())
        )
        if not produced:
            shutil.rmtree(paths.root)

    def _run_training(
        self, config: CynosureConfig, artifacts: RunArtifacts | None,
        dump_trajectory: bool, resume: bool = False,
        run_root: Path | None = None,
    ) -> int:
        """训练循环（async 执行序单进程门面，#226 切换期第一步起生产
        入口单口径）：``AsyncTrainingExecutor.build`` 装配（设备发现 →
        分配表 + 槽注册表 → 每卡完整副本与绑卡线程）→ ``run()`` 主循环
        （Baseline 采样 → 逐 iteration → 里程碑评测/早停 → RL 后重采 →
        v12 续训分片与产物 checkpoint 落盘）；``--dump-trajectory`` 额外
        产出轨迹诊断工件（trajectory.json，组1 专属独立单进程路径）；
        ``--resume`` 从 run 目录的 v12 续训分片恢复。续训失败不回滚
        run 目录（既有产物非本次预占）。

        装配/启动失败的 run 目录回滚在本进程内直接执行（单进程执行序，
        无旧执行序的 rank 非对称动作与广播裁决面）。"""
        if artifacts is None:
            try:
                artifacts = RunArtifacts.init(config, run_root)
            except FileExistsError:
                print(
                    f"run 目录已存在（不静默覆盖；续训请走 --resume 入口）: "
                    f"{run_root}",
                    file=self._stderr,
                )
                return _EXIT_USAGE_ERROR
            print(f"run 目录已就绪: {artifacts.paths.root}", file=self._stdout)
        if dump_trajectory and config.experiment.group == "modal-label":
            # 轨迹诊断是独立单进程路径（policy 采样场对照，与训练循环无关）
            if self._dump_trajectory(config, artifacts) != 0:
                self._rollback_untouched_run(artifacts)
                return _EXIT_USAGE_ERROR
        # 构造期 = 装配/输入契约（网络与 manifest 工件装载、跨字段守卫）：
        # checkpoint 键/shape 不匹配的严格装载失败（RuntimeError）与
        # 损坏文件的反序列化失败（UnpicklingError）同属输入契约违反，
        # 得到干净消息 + 未产出工件的 run 目录回滚
        try:
            executor = AsyncTrainingExecutor.build(
                config, artifacts, resume=resume,
            )
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"训练输入契约违反: {exc}", file=self._stderr)
            if not resume:
                self._rollback_untouched_run(artifacts)
            return _EXIT_USAGE_ERROR
        try:
            completed = executor.run()
        except TrainingAborted as exc:
            # fail-fast 门面（槽线程异常 / barrier 硬超时 / 跨卡权重分叉）：
            # 进程级退出的入口层承载（executor docstring 记档），干净消息
            # 非 usage 错误（既有产物保留——续训可从中断现场恢复）
            print(f"训练中止（fail-fast）: {exc}", file=self._stderr)
            return 1
        except (ValueError, FileNotFoundError) as exc:
            print(f"训练输入契约违反: {exc}", file=self._stderr)
            if not resume:
                self._rollback_untouched_run(artifacts)
            return _EXIT_USAGE_ERROR
        print(
            f"训练完成（{completed} iteration）：iter 事件流 "
            f"{artifacts.paths.metrics}、checkpoint {artifacts.paths.checkpoints}",
            file=self._stdout,
        )
        return 0

    def _dump_trajectory(
        self, config: CynosureConfig, artifacts: RunArtifacts,
    ) -> int:
        """fixture 轨迹诊断（--dump-trajectory）：policy 采样路径 + MONAI
        直接步对照 + log-prob 对 + 双样本分布统计量落盘为诊断工件。"""
        try:
            report = TrajectoryDiagnosticRunner(config).run()
        except (ValueError, FileNotFoundError) as exc:
            print(f"轨迹诊断输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        artifacts.paths.trajectory_diagnostic.write_text(
            report.model_dump_json(indent=2), encoding="utf-8",
        )
        print(
            f"轨迹诊断工件已落盘: {artifacts.paths.trajectory_diagnostic}"
            f"（η={report.eta}、扰动步 {report.perturbation_steps}、"
            f"log-prob 对 {len(report.logprob_pairs)} 组）",
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
            print(f"评测目标 run 目录: {run_root}", file=self._stdout)
        print(
            f"评测计划：group={config.experiment.group}"
            f" N_baseline={config.schedule.baseline_samples}"
            f" 里程碑间隔={config.schedule.milestone_interval} iteration"
            f" N_plateau={config.schedule.n_plateau}。里程碑解码评测"
            "（2.5D FID/KID，milestone 事件入训练指标流）随 train 循环在"
            "里程碑间隔触发；Baseline 采样与 RL 后重采随 train 落盘"
            " manifest 样本路径。独立 eval 子命令（验收阶梯汇总）由后续 "
            "ticket 交付",
            file=self._stdout,
        )
        return 0

    def _load_detached_config(
        self, config_arg: str, schema: type[_SchemaT],
    ) -> _SchemaT | None:
        """独立 schema 子命令（fid / base-smoke）的 config 装载：与训练
        config 完全分离——裁决性评测仪器的冻结变量与基座装载自检的定点
        输入都不经由训练 schema。错误面与行为同训练族装载（不存在 /
        非法 JSON / 字段级校验失败均落 exit 2 的可读输出）。"""
        path = Path(config_arg)
        if not path.is_file():
            print(f"config 文件不存在: {path}", file=self._stderr)
            return None
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            return schema.model_validate(data)
        except ValidationError as exc:
            print("config 校验失败：", file=self._stderr)
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"])
                print(f"  - {location}: {error['msg']}", file=self._stderr)
            return None
        except json.JSONDecodeError as exc:
            print(f"config 不是合法 JSON: {exc}", file=self._stderr)
            return None

    def _fid(self, args: argparse.Namespace) -> int:
        """裁决性 MR FID 读数（#73 双轨之一）：单一对比的一次执行。
        装配期与执行期失败（权重/清单/缓存口径指纹/损坏工件的
        反序列化）= 输入契约违反，exit 2——fid 是只读评测，执行期的
        RuntimeError 面就是权重与缓存 .pt 的装载失败，与 prepare/
        pretrain 装载期同口径收窄，不裸 traceback；成功则三面 FID +
        均值 stdout 展示，FidResult provenance 落盘。"""
        config = self._load_detached_config(args.config, MrFidConfig)
        if config is None:
            return _EXIT_USAGE_ERROR
        try:
            instrument = MrFidInstrument(config)
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"fid 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        try:
            result = instrument.run(comparison_tag=args.comparison_tag)
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"fid 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(f"FID XY: {result.fid_xy:.4f}", file=self._stdout)
        print(f"FID YZ: {result.fid_yz:.4f}", file=self._stdout)
        print(f"FID ZX: {result.fid_zx:.4f}", file=self._stdout)
        print(f"FID Avg: {result.fid_avg:.4f}", file=self._stdout)
        print(
            f"FID 结果已落盘: {config.result_json}"
            f"（modality={result.modality}、ratio={result.center_slices_ratio}、"
            f"num_images={result.num_images}，provenance 随盘）",
            file=self._stdout,
        )
        return 0

    def _fid_floor(self, args: argparse.Namespace) -> int:
        """real-vs-real 地板半分：manifest → 病例级 seed 半分 → 冻结落盘。
        产物（split_record.json + 逐格双侧清单）直接作为 fid 子命令的
        real/synth 清单输入（每对同格清单 = 一次地板对比）；输出目录
        已有不一致的冻结记录时拒绝（冻结工件不重算）。"""
        try:
            record = RealRealFloorSplit(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                seed=args.seed,
                path_template=args.path_template,
            ).run()
        except (ValueError, FileNotFoundError) as exc:
            print(f"fid-floor 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(
            f"病例 {len(record.half_a) + len(record.half_b)} → "
            f"half_a={len(record.half_a)}、half_b={len(record.half_b)}"
            f"（seed={record.seed}，来源 {record.validation_source}）",
            file=self._stdout,
        )
        print(
            f"冻结记录与逐格双侧清单已落盘: {args.output_dir}"
            "（split_record.json + filelist_half_<a|b>[_<格>].txt）",
            file=self._stdout,
        )
        return 0

    def _base_smoke(self, args: argparse.Namespace) -> int:
        """基座 checkpoint 装载与前向自检（wayfinder #120）：装载上游发布件
        （UNet 训练 checkpoint 容器 + VAE 裸权重）→ 定点 latent/模态 token
        前向指纹 → VAE 生产 latent 尺寸往返 → 报告落盘。

        装载期与执行期的失败（工件缺失 / 转写配置键不完整 / 参数量对账
        不符 / 定点前向逐位不可复现）= 自检未通过，exit 2：退出码即
        结论，不留一份「半绿」的报告。"""
        config = self._load_detached_config(args.config, BaseSmokeConfig)
        if config is None:
            return _EXIT_USAGE_ERROR
        try:
            report = BaseSmokeRunner(config).run()
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"base-smoke 未通过: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(
            f"装载：UNet {report.loading.unet_parameters} 参数（权重文件 "
            f"{report.loading.unet_checkpoint_parameters}）、VAE 权重文件 "
            f"{report.loading.vae_checkpoint_parameters} 参数",
            file=self._stdout,
        )
        print(
            f"定点前向：latent {list(report.latent_shape)} @ t="
            f"{report.timestep}、模态 token={report.modality_token}、"
            f"CFG={report.cfg_weight}、spacing={list(report.spacing)} → "
            f"velocity {list(report.velocity_shape)} sha256="
            f"{report.velocity_sha256}（两次前向逐位一致："
            f"{report.velocity_repeat_identical}）",
            file=self._stdout,
        )
        print(
            f"VAE 往返：{list(report.image_shape)} → "
            f"{list(report.encoded_shape)} → {list(report.decoded_shape)}"
            f"（decode 口径 {report.decode_autocast_dtype} autocast、"
            f"roi={list(report.decode_roi_size)}、"
            f"overlap={report.decode_overlap:.4f}、scale="
            f"{report.loading.latent_scale_factor} 来自 "
            f"{report.loading.latent_scale_factor_source}）",
            file=self._stdout,
        )
        print(
            f"基座装载自检报告已落盘: {config.output_json}",
            file=self._stdout,
        )
        return 0

    @staticmethod
    def _prepare_device() -> torch.device:
        """prepare 编码设备（单进程、不起进程组）：有 DCU/CUDA 用 0 号卡。"""
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _prepare(
        self, args: argparse.Namespace, config: CynosureConfig,
    ) -> int:
        # 装载期（同 train 构造期口径）：网络工件严格装载失败
        # （RuntimeError）与损坏 checkpoint 反序列化失败（UnpicklingError）
        # 同属输入契约违反；执行期窄面在下方（OOM 等运行时故障不混入归因）
        try:
            encoder = PreparePipeline.build_encoder(config, self._prepare_device())
            vocabulary = PreparePipeline.load_vocabulary(config)
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"prepare 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        try:
            report = PreparePipeline(config, encoder, vocabulary).run()
        except (ValueError, FileNotFoundError) as exc:
            print(f"prepare 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        print(
            f"prepare 完成（split seed={config.schedule.seed}："
            f"train {report.split_sizes['train']} / val "
            f"{report.split_sizes['val']} / test {report.split_sizes['test']}）:",
            file=self._stdout,
        )
        print(
            f"  - Real sample pool: {report.pool_manifest}"
            f"（{report.pool_entries} 条，按"
            + (
                "生成条件分层"
                if report.condition_counts is not None
                else "序列分层"
            ) + "）",
            file=self._stdout,
        )
        if report.condition_counts is not None:
            detail = "、".join(
                f"{condition}×{count}"
                for condition, count in sorted(report.condition_counts.items())
            )
            print(f"    - 逐条件: {detail}", file=self._stdout)
        print(
            f"  - Held-out real: {report.heldout_manifest}"
            f"（{report.heldout_entries} 条，与 pool 病例级不相交、"
            "永不参与判别器更新）",
            file=self._stdout,
        )
        if report.sampling_manifest is not None:
            print(
                f"  - 配额抽样留痕: {report.sampling_manifest}"
                "（逐卷归属 + 评估集互斥守卫读数，#131）",
                file=self._stdout,
            )
        print(
            f"  - per-channel 标准化统计量: {report.channel_stats}"
            f"（mean/std × {len(report.mean)} 通道）",
            file=self._stdout,
        )
        return 0

    def _pretrain(
        self, args: argparse.Namespace, config: CynosureConfig,
    ) -> int:
        """判别器 warm-start 预训练（ADR-0007 + ADR-0008-04；ADR-0018
        单进程多卡 async 化）：per-condition 密集步进（主控单点测量
        模板 + 卡轴分片扇出 + gate inline 主控、产物主控唯一写者），
        跑至全部轮转条件过线或步数上限（过线条件为空不拒跑——报告与
        checkpoint 落盘供诊断）。

        run 目录默认 = config 的 ``reward.pretrain_report_json`` 所在
        目录（产物位置在 config 里声明，train 上岗按同一路径装载）；
        ``--run-dir`` 可显式覆盖（产物路径以 config 声明为准——覆盖
        目录与声明路径分叉时拒绝：train 按 config 声明装载，分叉即
        missing-report 或静默装旧报告）。

        torchrun 启动显式拒绝（ADR-0018 的历史回环：ADR-0016 决策 3
        恰好退役的 RANK env 守卫在本票恢复）——执行模型已单进程多卡化，
        进程内多卡由设备发现承担、多进程拓扑退役；拒绝在 run 目录
        预占之前（usage 错误不落任何工件）。"""
        env_rank = DistributedContext.env_rank()
        if env_rank is not None:
            print(
                f"pretrain 以单进程多卡执行（ADR-0018），"
                f"拒绝 torchrun 启动（RANK={env_rank}）——进程内多卡由"
                "设备发现承担（CUDA_VISIBLE_DEVICES 裁剪卡集）",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        run_root = (
            Path(args.run_dir) if args.run_dir
            else Path(config.reward.pretrain_report_json).parent
        )
        # 产物路径一致性不变式（init 之前校验：分叉配置不预占目录）：
        # 报告是 run 目录布局内的契约文件，train 按 config 声明的精确
        # 路径装载——两者不一致时 producer/consumer 断链。比对**归一化
        # 后**的路径而非字面拼写：相对 vs 绝对、``.`` 分量、符号链接
        # 祖先都是同一位置的不同写法，字面比较会把合法调用误判成分叉。
        declared_report = Path(config.reward.pretrain_report_json)
        produced_report = PretrainRun.layout(run_root).report
        if produced_report.resolve() != declared_report.resolve():
            print(
                f"预训练产物路径与 config 声明分叉：本次将产出 "
                f"{produced_report}，train 按声明装载 {declared_report}"
                "（换目录请同步更新 reward.pretrain_report_json）",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        try:
            run = PretrainRun.init(config, run_root)
        except FileExistsError:
            print(
                f"预训练 run 目录已存在（不静默覆盖）: {run_root}",
                file=self._stderr,
            )
            return _EXIT_USAGE_ERROR
        # 装配期 = 输入契约（网络/工件装载、跨字段守卫）：失败回滚本次
        # 预占的 run 目录（尚无任何产出）
        try:
            driver = PretrainDriver.build(config, run)
        except (
            ValueError, FileNotFoundError, RuntimeError,
            pickle.UnpicklingError,
        ) as exc:
            print(f"pretrain 输入契约违反: {exc}", file=self._stderr)
            shutil.rmtree(run_root)
            return _EXIT_USAGE_ERROR
        try:
            report = driver.run()
        except (ValueError, FileNotFoundError) as exc:
            print(f"pretrain 输入契约违反: {exc}", file=self._stderr)
            return _EXIT_USAGE_ERROR
        outcome = (
            "已达标（全部条件过线）" if report.all_conditions_passed
            else "未达标（步数上限耗尽）"
        )
        conditions_passed = (
            "、".join(report.conditions_passed)
            if report.conditions_passed else "（空）"
        )
        print(
            f"预训练完成（group={report.group}，步数 "
            f"{report.steps_completed}/{config.reward.pretrain_max_steps}）：",
            file=self._stdout,
        )
        print(
            f"  - 过线条件: {conditions_passed}（阈值 {report.pass_threshold}，{outcome}）",
            file=self._stdout,
        )
        # 判据口径随报告明示（ADR-0012 决策 5 的审计面）：预训练的
        # recon-AUC 与在线 iter 事件的 rollout-AUC 不同构、不可横向比较
        # ——准入体检 vs 在岗考核，跨阶段比读数会被误导
        print(
            f"  - 判据口径: {report.auc_criterion}（held-out real 原始 vs "
            "冻结基座同源重构体；与在线 rollout-AUC 不可横向比较）",
            file=self._stdout,
        )
        for modality, auc in report.condition_auc.items():
            volumes = report.condition_volumes.get(modality)
            scope = "" if volumes is None else f"，{volumes} 卷"
            print(
                f"  - recon-AUC[{modality}]: {auc:.4f}{scope}",
                file=self._stdout,
            )
        print(
            f"  - 判别器 checkpoint: {run.paths.discriminator_ckpt}",
            file=self._stdout,
        )
        print(
            f"  - 预训练报告: {run.paths.report}",
            file=self._stdout,
        )
        print(
            f"  - 预训练曲线: {run.paths.metrics}"
            f"（{len(run.read_events())} 条 pretrain 事件，离线查看收敛"
            "曲线与校准门槛阈值的数据源）",
            file=self._stdout,
        )
        return 0

def main() -> None:
    """console-script 入口。"""
    raise SystemExit(CynosureCli(sys.argv[1:], sys.stdout, sys.stderr).run())
