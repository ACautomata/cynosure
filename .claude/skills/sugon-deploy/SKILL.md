---
name: sugon-deploy
description: 把 cynosure 训练部署到曙光 SothisAI DCU 实例并启动——同步代码、装 DCU 依赖、数据落位（group_data/private_data 分区纪律）、torchrun 启动 prepare/pretrain/train、metrics.jsonl 监控。触发：用户要在曙光/Sugon/SothisAI/DCU 集群上部署或跑训练、同步代码上集群、启动/恢复 train 或 pretrain 或 prepare、或问训练在集群上怎么跑。集群级打底（SSH sugon 别名、双 source、hy-smi）见用户级 sugon-bootstrap。
---

# 部署训练上曙光 DCU

一次部署 = 代码上集群 → 数据落位 → 起一个 run。集群级打底（SSH `sugon`
别名、**双 source**、hy-smi、持久分区常识、DCU 依赖陷阱）由用户级
sugon-bootstrap 负责，本 skill 不重复；前置不满足时先跑它。

**隐私纪律**：本 skill 与一切部署产物（脚本、文档、release 资产）不写平台
账号、实例端口/网关、代理地址与凭据。连接只经 `ssh sugon` 别名，凭据只活在
集群上的平台代理文件里（双 source 自动消费），临时脚本进 run 目录
`scripts/`（见 experiment-release 对 release 资产的扫查纪律）。

## 落位纪律（部署前记牢）

二分标准：**下载/复制的公共资产进团队盘，本项目产出的进私有盘**。

| 资产 | 落位 |
|---|---|
| 数据集（`dataset_root`）、公开预训练权重（`unet_ckpt`、`vae_ckpt`、`radimagenet_weights` 等下载件） | `/root/group_data/cynosure/`（团队持久盘，跨实例复用，缺了才下载） |
| run 目录、checkpoint、`pretrain_report_json`、判别器权重等训练产物 | `/root/private_data/cynosure/`（`deployment.output_root` 默认值） |
| 代码、pip 包装进系统 site-packages | 系统盘（易失但可重建：rsync 重传、pip 重装） |

任何产物落 `/` 或 `/tmp` = 实例一重置全没，等于丢数据。

## 步骤

### 1. 前置门：双 source 自动生效

```bash
ssh sugon 'echo "proxy=${https_proxy:-UNSET}"; command -v hy-smi >/dev/null && echo hy-smi-OK || echo hy-smi-MISSING'
```

判据：`proxy=http://…` 有值且 `hy-smi-OK`。显示 UNSET/MISSING → 跑
sugon-bootstrap §4 的 bashrc 注入（bashrc 在易失盘，实例重置后需重注入），
不满足不继续。

### 2. 代码上集群（rsync）

仓库根目录下把本地工作树 rsync 直传到集群（走 ssh 通道，不占集群外网；pip 仍走双 source 代理，前置门已保证）。同步的是工作树本身——未提交改动照传，无需先提交，发布可追溯性由 experiment-release 的「先提交」纪律兜底。`--delete` 使集群目录严格镜像本地，`--exclude` 的模式同时保护集群侧同名文件不被清掉：

```bash
rsync -acv --delete \
  --exclude .git --exclude .venv --exclude .claude --exclude .remote --exclude scripts \
  --exclude __pycache__ --exclude .pytest_cache --exclude .mypy_cache --exclude .ruff_cache \
  --exclude '*.egg-info' --exclude dist --exclude build \
  ./ sugon:/root/cynosure/
```

判据：`-c` 按逐文件校验和比对（不靠时间戳），命令正常结束、只有统计行即两侧一致——比对工作由同步本身完成。同步完成后重复执行应零文件行，出现文件行即本地又有新改动未传。

首次切换：集群 `/root/cynosure` 若还是旧 git 检出，先 `ssh sugon 'rm -rf /root/cynosure/.git'`（此后该目录只是 rsync 镜像）；实例重置后目录整体消失，rsync 会自动重建。

依赖装进**系统 python**（3.11，DCU torch 唯一宿主）——本地「先激活
`.venv/`」的惯例到此为止，venv 里没有 DCU torch：

```bash
ssh sugon 'pip install -e /root/cynosure --no-deps'   # --no-deps 防 pip 从 PyPI 拉 CUDA torch 顶掉 DCU torch
ssh sugon 'python -c "import torch, cynosure; import numpy; print(torch.__version__, numpy.__version__)"'
```

判据：torch 2.9.x 且 numpy 1.x（集群 pin `numpy==1.26.4`；import 失败按
sugon-bootstrap 的 pitfalls 排查——装任何 ML 依赖后都要回验这一条）。
`flash_attn`/`triton` 缺失走 sugon-bootstrap 的 `ensure_dcu_ops.sh`。

### 3. 数据落位

按落位纪律核对目标 run 的 config `artifacts` 段每个路径：团队盘上的
下载件缺了才下载（走代理，落 `/root/group_data/cynosure/`）；私有盘上的
产物路径由步骤 5 的 run 目录创建。

判据：`ssh sugon 'ls <每个数据集/权重路径>'` 全部存在。

### 4. run 配置

以集群上最近一个 run 的 `config.json` 为底
（`/root/private_data/cynosure/runs/<最近>/config.json`），只改 artifacts
路径与本次超参——不从零写 schema（887 行，全字段见
`src/cynosure/config.py`）。三个硬点：

- `reward.pretrain_report_json` **必填无默认**，schema 层拦截（判别器
  warm-start 报告，来自 pretrain）；
- `--run-dir` 必须显式给 `/root/private_data/cynosure/runs/<YYYYMMDD>-<名>`
  （RANK 存在时代码硬性要求），命名沿 T11 先例 `20260906-t11-smoke`；
- 路径一律集群绝对路径，项目无环境变量机制。

判据：`cynosure train --config <file>` 能通过 pydantic 校验进入装配
（缺件会被 schema 拒绝并报字段名）。

### 5. 启动

前置链：`prepare`（构建 real sample pool / held-out / channel stats）→
`pretrain`（判别器 warm-start，产出 `pretrain_report_json`）→ `train`。
已有产物的环节跳过。`pretrain` 与 `train` 同走 torchrun（ADR-0016：
pretrain 测量批按卷分片到各 rank + rank0 gate 单点判定 + 判别器 DDP，
单进程 World-1 退化路径仅测试口径）。

实例无作业调度器，长跑进 tmux：

```bash
ssh sugon
tmux new -s <run名>
# pretrain：torchrun 启动，nproc = 实例 DCU 卡数（8 卡实例 = 8）
CYNOSURE_PG_TIMEOUT_MIN=40 torchrun --nproc_per_node=8 -m cynosure.cli pretrain \
  --config /root/private_data/cynosure/runs/<run>/config.json \
  --run-dir /root/private_data/cynosure/runs/<run>/pretrain_run
```

- `CYNOSURE_PG_TIMEOUT_MIN=40`：同实例其他任务会间歇饿死 RCCL 端点，
  watchdog 调到 40 分钟（`src/cynosure/distributed/process.py`）；
- `--run-dir` 必须显式给：分布式启动（检测到 RANK）硬性要求（跨 rank
  目录对齐，train 同款规则），且目录须与 config
  `reward.pretrain_report_json` 声明一致——train 按声明路径装载，
  分叉即 usage error 拒绝；
- train 段同款（`--run-dir` 给 run 目录本身）：

```bash
CYNOSURE_PG_TIMEOUT_MIN=40 torchrun --nproc_per_node=8 -m cynosure.cli train \
  --config /root/private_data/cynosure/runs/<run>/config.json \
  --run-dir /root/private_data/cynosure/runs/<run>
```

判据：run 目录 `metrics.jsonl` 出现首条事件（pretrain 相 = 首条
`pretrain` 事件，train 相 = 首条 `iter` 事件——进程活着 ≠ 在训练）。

### 6. 监控

```bash
ssh sugon 'hy-smi'                                                     # 卡占用
ssh sugon 'tail -f /root/private_data/cynosure/runs/<run>/metrics.jsonl'  # iter/milestone/pretrain 事件流
```

指标契约只有 `metrics.jsonl`（JSONL 事件流，拒 NaN/Inf）；项目不接
wandb/tensorboard，监控与出图都从它出发。

**读数口径（ADR-0016）**：单卡 run 与多卡 run 的预训练读数**不可横向
比较逐位值**——测量批分块边界与判别器有效 batch（K×world_size）随
world_size 变化；多卡重放锚是「同 config + 同 world_size 重跑逐位
一致」，跨卡只保统计等价（卷积算法随 batch shape 的 ~1e-12 量级尾差）。
对比实验的「同机制」判断锚在机制语义（ADR-0012）而非执行面数值。

### 7. 收尾

发布走 experiment-release skill：run 三件套（`config.json`、`metrics.jsonl`、
`manifest.json`）齐才发布，checkpoint 留集群持久分区不上 release。

## 参考

- 启动编排、M0 门槛与 T11 实测结论：`docs/spec/orchestration.md`
- 决策记录：`docs/adr/0003`（torchrun+FSDP）、`0005`（SothisAI 迁移）、`0007`（pretrain warm-start）、`0016`（pretrain torchrun 化：测量批分片 + rank0 gate + 判别器 DDP）
- config 全字段：`src/cynosure/config.py`
- 集群级故障排查（双 source 失效、numpy 顶坏 torch、pip 超时、sourcefind wheel）：
  sugon-bootstrap 的故障排查表与 `references/dcu-pitfalls.md`
