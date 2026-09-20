# MR-RATE real 生产数据链（#132 / spec #125 T6）

prepare 数据链在**生产体量**下的落地产物登记：MR-RATE 全库（train 候选域）的
分层抽样 pool + patient 级 held-out 集，经生产 VAE 编码为 latent 工件 +
per-channel 强度统计量。判别器预训练（#122 延伸）与正式训练票（#134）的数据基座。

本目录是登记的摘要 manifest；latent 本体与影像不落仓库（见「落位」）。

## 来源与口径（与评估集 #78 完全同源）

- 数据集：`Forithmus/MR-RATE`（HF，gated），快照 `f6e39794`（与评估集同一 release）。
- 词汇表条件 = `classified_modality`（小写化）× `acquisition_plane`（小写化）；
  列名映射由数据落位面完成（#132）。oblique 及词汇表外组合不入层
  （词表域外 11,299 卷）。
- split：官方 `splits.csv` patient 级连接，**train 候选域**（638,345 series）
  经互斥守卫后 627,046 卷进入抽样域——评估集（#78，2,514 卷）以
  `study_uid + series_id` 与 patient 两粒度显式排除，**实测零命中**
  （`plan_summary.json` 的 `eval_exclusion_*` 字段）。
- 互斥纪律（#73 决策）由 `EvalSetExclusion` 在装配链内 fail-fast 强制。

## 配额（spec「配额为 config knob，执行期定版」）

| 项 | 值 | 裁决 |
|---|---|---|
| 头部 10 格 `real_pool_quota` | 各 2,000 | 2026-09-17 用户裁决（spec「头部模态各数千条」带内下探；下载 1.6TB→283.9GB 实测、编码体量可控） |
| `mra/all-planes` | 全量 150（pool 140 + held-out 10） | spec「MRA 全量」（#78 普查口径） |
| `heldout_quota_volumes` | 512/格 | ADR-0008 决策 6 支撑度界一个数量级之上 |
| `heldout_fraction` | 0.1（patient 级二分） | config 默认 |
| `schedule.seed` | 20260912 | 与评估集同值（不同机制域，无耦合） |

抽样语义零复制：下载计划导出与 prepare 装配复用同一批 `MrRateAssembly`
方法（同 seed / 同输入 / 同代码），下载全集与抽样 manifest 逐键一致；
fixture 场景对拍冒烟（12 键）验证过该一致性。

## 产出规模

- **25,270 卷 / 21,512 study-zip**：pool 20,140 + held-out 5,130
  （逐条件计数见 `sampled_counts.csv`；计划明细见 `plan_summary.json`）。
- 下载实测 283.9GB（zip 解包前），全部 verdict ok（含 14 study 因 hf-mirror
  代理缺陷改直连补齐，见执行记录）。

## 落位

| 位置 | 路径 | 内容 |
|---|---|---|
| gauss（编码源） | `/data72/junran/mrrate_real/prepare/` | latents（逐条件目录）、per-channel 统计量、抽样 manifest、`audit.json` |
| sugon（训练用副本） | `/root/group_data/mrrate_real/` | prepare 工件 + 元数据（`export_group/` 层：`metadata.csv` / `splits.csv` / `eval_manifest.csv` / 下载面清单）+ 原始卷副本，经 SCNet 快传同步 |
| 原始影像 | gauss `/data72/junran/mrrate_real/volumes/`；副本 sugon `/root/group_data/mrrate_real/volumes/` | 265GB（25,270 卷）平铺 nii.gz，SCNet 快传同步 |

加载契约：`PreparePipeline`（#121/#163 落库）按 config 的 `artifacts.*`
路径装载；`audit.json` 为验收工件（per-condition 形状、分层完备、MRA
全量、held-out patient 级二分、统计量规模）。

## 编码与统计量（生产编码完成，实测补登 2026-09-19）

- 词汇表 per-condition 统一网格（ADR-0008 / #130），latent 形状 = grid/4；
  ≤16.38M voxels 整前向，否则滑窗（roi 320×320×160，overlap 0.4）。
- per-channel 强度统计量（`ChannelRunningStats`，float64 可合并）在 pool
  编码时增量累积；worker 分片可加性见 ADR-0011。
- 统计量实测（`channel_stats.json`，source = real_pool.json）：4 通道
  mean = [-0.0526, -0.1145, -0.0529, 0.0725]，std = [0.9803, 1.0229,
  0.9843, 1.0106]；num_latents = 20,140（pool 全量）；统计量锚形状
  [4, 64, 64, 32]；intensity_clip = False（NVIDIA v1 / #71 裁决）。
- 验收：`audit.json` **all_pass = true**（12 项检查全过）：pool / held-out
  键与下载清单逐键一致；eval 集 series + patient 双粒度零重叠（AC1）；
  held-out patient 级二分（pool 16,280 / held-out 3,400 patients，AC3）；
  逐条件配额齐 + MRA 140/10 全量；全部 latent 形状与词汇表网格一致
  （20,140 抽查 / mismatch 0，AC2）；统计量数量与形状锚定。
- 成本实测：编码墙钟 ~21.4h × 2× RTX A6000（GPU 1,3）≈ **42.8 GPU·h**
  （好于冒烟外推 105 GPU·h）；latent 体量 **131GB**（25,270 卷，~5.2MB/卷，
  与冒烟 5.0MB/卷一致）；audit 21min；下载面 283.9GB / 16.1h（执行记录）。

## 复现与脚本

- 链路执行脚本（chain / download / encode 封装 / watcher）在
  `run/132-mrrate-real-prod/scripts/`（不入仓，随 experiment-release
  归档）；prepare 装配与审计由 `src/cynosure` prepare 管线（#121/#163）
  承担。
- 生产运行代码锚：commit `e5095b9`——prepare 链代码该 commit 已全量
  落库，运行期间无 `src/` 未提交改动；`audit.json` 的
  `provenance.source_commit: null` 为工件原样拷贝，未回填。
- 复现要点：同 snapshot `f6e39794` + 同 seed `20260912` + 同配额 →
  抽样 manifest 逐键一致（对拍 `plan_summary.json`）。

## 文件

| 文件 | 内容 |
|---|---|
| `audit.json` | 生产验收工件原始拷贝（all_pass 与全部检查读数） |
| `plan_summary.json` | 抽样计划原始导出：逐格候选普查、配额实取、held-out 计数、互斥审计读数、seed |
| `sampled_counts.csv` | 逐条件 pool/held-out 卷数摘要 |
| `README.md` | 本文件 |
