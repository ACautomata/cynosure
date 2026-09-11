# MR-RATE 基线评估集（wayfinder #78 / 地图 #67）

纯 Granular-GRPO 后训练基线的**留出评估集**：零下载普查 → 分层抽样 → 互斥登记 → 按需下载。
本目录是登记的 manifest；影像本体不落仓库（见「影像落位」）。

## 互斥纪律（硬约束，来自决策 #73）

**将来 RL real pool / replay 数据链必须显式排除 `eval_manifest.csv` 中的全部卷**
（以 `study_uid + series_id` 为键）。否则评估集被 RL 判别器见过，RL 后 FID 吃到利好偏差。

> 旁注：fork T6 的遗忘参考集（val split 202 卷，replay/forgetting 线）与本评估集
> 重叠 12 卷。两者都是留出侧评估工件、互不在对方训练链中，不影响互斥纪律；
> 若将来要让两类读数完全独立，可从本集剔除这 12 卷（重抽代价为零，seed 在案）。

## 来源与口径

- 数据集：`Forithmus/MR-RATE`（HF），快照 `f6e39794`（2026-09-12 经 hf-mirror 拉取；
  与 2026-08 首次拉取 md5 逐文件一致，同一 release）。
- split：`splits.csv` 经 `patient_uid` 连接（patient 级一致；文件本身 study 级，98,334 行 =
  73,516 患者）。**留出池 = val + test**（train 留给上游语料/RL 数据链）。
- 质量旗标：`is_derived` / `is_localizer` / `is_subtraction` 在本 release 全部为 False
  （剔除为零，已审计）；`StudyFound` 全 True。
- 模态白名单：`classified_modality ∈ {T1w, T2w, FLAIR, SWI, MRA}`（本 release 无其它取值）。
- **oblique 一律不入层**（与官方 FOV 表口径一致：官方表只覆盖正交三平面，oblique 8,806 另计）。

## 分层与抽样（seed = 20260912，全程可复现）

| 层 | 机制 | 产出 |
|---|---|---|
| 10 头部格（T1w/FLAIR/T2w × AXIAL/SAGITTAL/CORONAL + SWI/AXIAL） | plane 级 N=250/格 | 2,500 卷 |
| MRA | modality 级全取（留出池仅 14 卷） | 14 卷 |
| SWI SAGITTAL / CORONAL | 格级全取 | **0 卷**（留出池为零，与官方「极稀格」警告一致） |

**对 #73 裁定的一处偏离（有据）**：#73 将 T2w 定为「稀疏 modality 级全取」，依据是官方
§3.4 的 T2w 669（source-volume 口径）。零下载普查证伪了该前提——本 release 分类口径下
T2w train 达 176,297 series（留出池 18,609），全取将使 T2w 成为评估集最大层（下载量放大
~9 倍）。2026-09-12 用户裁决：**T2w 头部化**，plane 级 N=250×3，机制规模与其余头部格一致。

抽样细节见 `sampling_seed.json`：固定 `random.Random(seed)`，格顺序固定，池内按
`(study_uid, series_id)` 排序后 `sample()`。

## 零下载普查要点（详见 census_* 文件）

- 元数据全量 705,254 series = train 638,345 + val 27,003 + test 39,906
  （T6 replay 报告的「705,254 (Train, quality flags excluded)」实为**全 split** 计数，
  旗标剔除在本 release 是空操作——两处口径勘误）。
- 与官方 `docs/inference.md` v1 表核对：**总量吻合**（train+val 三平面格求和 665,348 vs
  官方 665,371，差 23 ≈ 0.003%），但**逐格组成有系统差**：T2w 为官方 ~124×（见上），
  FLAIR/SWI 的 SAGITTAL/CORONAL 我们低 30–40%，T1w/FLAIR/SWI 的 AXIAL 我们高 3–34%。
  结论：两版分类/计数口径不同（官方 v1 语料的确切构成未公开，见
  `research/mrrate-data-spec.md` §6/§10）。**本集统一使用本 release 的
  `classified_modality` / `acquisition_plane`**——RL 数据链从同一口径取数，评估条件与
  训练条件定义一致，这是比「对齐官方表」更硬的约束。

## 文件

| 文件 | 内容 |
|---|---|
| `eval_manifest.csv` | 评估集主清单（2,514 卷）：stratum、sampling_role、split、batch_id、patient_uid、study_uid、series_id、modality、plane、array_shape、array_spacing_mm、array_fov_mm |
| `download_list.csv` | 下载清单：2,200 个 study-zip（batch_id、study_uid、n_eval_series、zip_repo_path） |
| `availability.csv` | 实际可得 (modality, plane) 卷数表（requested vs obtained，按需下载核验后产出） |
| `sampled_counts.csv` | 抽样产出逐格计数 |
| `sampling_seed.json` | seed、格表、池过滤器、复现命令 |
| `census_by_split_modality_plane.csv` | (split × modality × plane) 全量交叉计数（留出池 66,909） |
| `census_official_reconcile.csv` | train 普查 vs 官方 v1 表逐格核对 |
| `census_flag_excluded.csv` | 旗标剔除审计（本 release 为空） |
| `census_summary.json` | 总量对账（含 39,883 缺口定案：test split = 39,906 行） |
| `census_latent_grid_distribution.csv` | 评估集逐格 latent 网格分布（resize 口径见 `research/mrrate-data-spec.md` §3.1/§4.1；地图 #67「尚未明确-1」的输入） |

## 影像落位（gauss）

- 体数据：`/data72/junran/mrrate_eval_volumes/<batch_id>/<study_uid>/{img,seg}/`
  （img = 原始 NIfTI；seg = HD-BET brain-mask；zip 拉完即删）
- 逐卷判定：`gauss:/home/junran/mrrate_eval/verdicts.csv`（append-only，续跑点）
- 下载日志：`gauss:/home/junran/logs/mrrate_eval_download.log`

## 复现

```bash
# gauss 上（HF_ENDPOINT=https://hf-mirror.com）
python3 scripts/mrrate_eval_census.py census <snap_dir> <out>   # 零下载普查
python3 scripts/mrrate_eval_census.py sample <out>              # 分层抽样
python3 scripts/mrrate_eval_download.py <out> <volumes_root>    # 按需下载+核验
```

脚本按仓库纪律留在 run 目录 `scripts/`（gitignored），随 experiment-release 归档。
将来若评估集扩充（top-up），**必须沿用留出池 + 新 seed 重抽并全量登记**，不得从
train split 补数（互斥纪律优先于层满额）。
