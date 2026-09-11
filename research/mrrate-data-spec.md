# MR-RATE 数据规格与上游预处理链（rflow-mr-brain v1 口径）

- **交付对象**：wayfinder 地图 #67「MR-RATE 上游域 RL 后训练基线」/ 数据规格票 #68
- **来源标注约定**（**本版按真上游重新核定**，凡引用必带标记）：
  - **【上游】** = `https://github.com/NVIDIA-Medtech/NV-Generate-CTMR`，本地快照 `/tmp/nvidia-upstream/NV-Generate-CTMR-main/`（HEAD `da438fe`，2026-08-19）。凡本文标【上游】者，均已用 `diff -rq` 与 fork 逐文件比对确认同内容。
  - **【fork】** = 仅存在于 fork 的文件/fork 改动。fork 旧态 `~/Documents/NV-Generate-CTMR/`（remote `ACautomata/NV-Generate-CTMR`，HEAD `764e4fb`）；fork 新态 `~/Documents/MR-Generate/`（`8a59815`）。
  - **【MR-RATE】** = 数据集官方 guide / HF 卡片（gated 页面）。
- **勘误（相对本票上一版口头汇报）**：上一版把 `~/Documents/NV-Generate-CTMR/scripts/compute_fid_2-5d_ct.py` 的 MR 分臂当作"官方口径旁证"——**不成立**：官方版 FID 仪器是**纯 CT**（只有 `SpatialPadd(value=-1000)` + `ScaleIntensityRanged(-1000,1000)`，`【上游】scripts/compute_fid_2-5d_ct.py:565,570`，无 `--modality` 参数、无 MR 分支），MR 支持是 fork 的扩展。clip 结论不依赖该条（见 §3.2）。
- **不重复的既有情报**：`MG/docs/research/20260908-upstream-rflow-mr-brain-recipe.md`（训练配方/采样/CFG）、`MG/docs/research/distribution-gap-rflow-mr-brain-vs-brats.md`（BraTS 域分布差）。

---

## 1. 结论摘要（硬约束清单，可直接给施工用）

| # | 约束 | 值/规则 | 来源 |
|---|---|---|---|
| 1 | **强度归一化** | MRI 臂 `ScaleIntensityRangePercentilesd(0.0, 99.5 → [0,1], **clip=False**)`；禁 z-score、禁 spacing 重采样 | 【上游】`scripts/transforms.py:64` |
| 2 | **方向** | `Orientationd(axcodes="RAS")`，只置换/翻转；斜采集残余斜度不纠正 | 【上游】`scripts/diff_model_create_training_data.py:58` |
| 3 | **尺寸** | 每轴 `max(round(轴长/128), 1)×128`，轴长取 **RAS 之后**的值；`round` 是 Python **half-to-even**（192→256、320→256，**不是**四舍五入）；latent 每轴 = 32 的倍数且 ≥32，**形状不固定** | 【上游】`:77-90,247-249` |
| 4 | **spacing** | 不做重采样；**conditioning 用 resize 后的后置 spacing**（= 原始 spacing × 原始轴长/取整轴长），侧车 ×1e2 | 【上游】`scripts/diff_model_train.py:120-121,450-451` |
| 5 | **切片下限** | brain v1 **无**（`<48 slices` 只属全身版 `rflow-mr`） | 【上游】`data/README.md:143` |
| 6 | **标签映射** | `t1w/t2w/flair/swi/mra` → `mri_t1/t2/flair/swi/mra` = 9/10/11/20/16；skull-stripped = 29–33。**上游权威**（映射表就在上游仓库里，fork 只加了 BraTS 的 40–43） | 【上游】`configs/modality_mapping.json` |
| 7 | **每序列双条目** | whole-brain + 现场 derive 的 skull-stripped（官方只发 `img/` + `seg/`，无现成 skull-stripped 文件） | 【MR-RATE】+【上游】`docs/inference.md:138` |
| 8 | **条目数/体量** | 665,371 × 2 ≈ **1,330,742 条**；latent 存储 **~1.5–3 TB（fp16）/ 3–6 TB（fp32）**；原始 zip 整库 **8.1 TB** | §5 推算 |
| 9 | **编码网格** | 恒整前向（本仓 T12 裁决），**不得**套上游 `SlidingWindowInferer`；latent 通道置末 `(X,Y,Z,4)` | 【上游】`:174-191` + 本仓 `docs/spec/data-preparation.md:28` |
| 10 | **许可** | 数据 **CC BY-NC-SA**（非商业 + ShareAlike）；模型 NVIDIA Open Model License（两者不同） | 【MR-RATE】卡片 |
| 11 | **划分** | 留出集必须取官方 `splits.csv`（**patient 级**，同患者所有 study 同 split） | 【MR-RATE】guide |

---

## 2. 获取

### 2.1 仓库与容量【MR-RATE】

| 仓库 | 容量 | 是否 v1 用到 |
|---|---|---|
| `Forithmus/MR-RATE` | **8.1 TB** | ✅ 唯一用到（native-space defaced 影像 + HD-BET 脑掩码 + 元数据/report/pathology/splits） |
| `Forithmus/MR-RATE-coreg` | 17.6 TB | ❌ |
| `Forithmus/MR-RATE-atlas` | 12.3 TB | ❌（fork 的 spine filter 标定借用了它的 study 名单） |
| `Forithmus/MR-RATE-nvseg-ctmr` | 415 GB | ❌（多标签分割，与 rflow-mr-brain 无关） |

官方明示："While we currently only use native-space data for our model training"（guide §3）——**拉主仓库即可**。

### 2.2 zip 布局、batch、单批容量【MR-RATE】

- `mri/batch00/<study_uid>.zip` … `mri/batch27/<study_uid>.zip`，**28 批**，每批 ~3,500 studies（guide:149；分批动机是 HF 单目录 ~10,000 文件上限）。
- **zip 形态 = 无法单文件取**：一个 study zip 内是 `<study_uid>/img/<study_uid>_<series_id>.nii.gz` + `<study_uid>/seg/<study_uid>_<series_id>_brain-mask.nii.gz`（guide + 【fork】`scripts/mrrate_series.py:24-28` 同构）。要取单序列也得下整个 zip。
- 容量：单 study zip 平均 **~74 MB**（N=1000 manifest 实测均值）；官方 batch 级模式 **~290 GB/batch**；28 × 290 GB ≈ 8.1 TB = HF 标称 ✅。
- **下载策略**：按 study 粒度 `hf download --include "mri/batchXX/<study_uid>.zip"`，拉完即删（一个 zip 只有 2 个文件有用）。该做法出自【fork】`scripts/download_replay_subset.py:19-24`——【上游】仓库**没有** MR-RATE 下载脚本（`scripts/download_model_data.py` 只下**模型权重**）。
- 元数据 `metadata/batchXX_metadata.csv` 是 series 级，**单独可拉、不需要影像**（§5.4）。

### 2.3 许可与访问【MR-RATE】

- **CC BY-NC-SA**（卡片原文："all elements of the MR-RATE dataset are released under a Creative Commons Attribution–NonCommercial–ShareAlike (CC BY-NC-SA) license"，商用联系 `contact@forithmus.com`）→ 非商业 + 相同方式共享：**用 MR-RATE 产出的 latent/权重/release 都受 NC 约束并须同许可分发**。注意：`rflow-mr-brain` **模型**是 NVIDIA Open Model License（可商用）——"基座可商用"≠"产物可商用"。
- **gated**：需 HF 账号 + 接受条款；未授权时 raw 读取返回 `Access to dataset Forithmus/MR-RATE is restricted`。是否自动批准未验证。团队账号已有访问（T6 replay run 已实拉）。

---

## 3. 预处理链：原始 NIfTI → VAE 输入

### 3.1 六步链【上游】`scripts/diff_model_create_training_data.py:33-90`

| # | 步骤 | 关键参数 | 代码 |
|---|---|---|---|
| 1 | `LoadImaged` | NIfTI 读取 | `:56` |
| 2 | `EnsureChannelFirstd` | → `[1,X,Y,Z]` | `:57` |
| 3 | `Orientationd(axcodes="RAS")` | 只置换/翻转 | `:58` |
| 4 | `EnsureTyped(dtype=torch.float32)` | | `:59` |
| 5 | `define_fixed_intensity_transform(modality)` | mri 臂 percentile 0.0–99.5 → [0,1]，**clip=False** | `:49` → `transforms.py:64` |
| 6 | `Resized(spatial_size=dim, mode="trilinear")` | `dim` = 逐轴 `max(round(轴长/128),1)×128` | `:63,77-90,247-249` |

**不在链中**（同样是结论）：无 spacing 重采样、无 crop/foreground、无 `NormalizeIntensity`/z-score、无模态堆叠（每序列独立处理、独立 label）。

**resize 边界**：`dim` 从 **RAS 之后**的 `meta["dim"]` 读（`:247-248` 先用含 Orientationd 的 `plain_transforms` 跑一遍）；`round()` 是 half-to-even（192/128=1.5→**256**；320/128=2.5→**256**）——写成"四舍五入"会差一整档 128。

### 3.2 clip 证据链（判定：`clip=False`）——**写死**

1. 【上游】**`scripts/transforms.py` 与 fork 版逐字节相同**（`diff -rq` 未列入差异；lead 的 md5 复核一致）。该文件由 NVIDIA 作者 `Can-Zhao` 以 "initial version"（`2cb90dd`, 2025-10-23）引入 → `:64` 的 `clip=False` **是 NVIDIA 官方原文**，不是"MONAI 默认"的转述。
2. 【上游】**`scripts/diff_model_create_training_data.py` 与 fork 版逐字节相同**，其 `:29` `from .transforms import SUPPORT_MODALITIES, define_fixed_intensity_transform`，`:49` 调用 → **v1 的训练 embedding 就是这条 clip=False 链产出的**。
3. 【fork】`clip=True` 在 `MG/src/ctmr/infrastructure/maisi_engine/instance_definition.py:76`，其 docstring `:25-39` **自认**："One recorded deviation from the byte-for-byte rule (issue #251, series-② T4): the mri arm's normalization flag is `clip=True` — **upstream shipped `clip=False`**"，理由是本仓 BraTS-t1c 的 >1.0 外推带落在冻结 `autoencoder_v1` 重建域外（外推带自评 MAE 0.8673 vs 截断 0.0062），并明写后果："**re-encode all training embeddings after this flag: clip=False-encoded embeddings are not reusable in the clip=True world**"。

→ **判定：`rflow-mr-brain v1` 发布权重的训练口径 = `clip=False`。** fork 的 `True` 是本仓 BraTS-P3 线的偏差，不是 v1 配方。

**旁证（同向，均为 fork 侧）**：fork 的 MR-RATE replay latent 走"original, unmodified encoding pipeline"（【fork】`scripts/create_replay_latent_dataset.py` 文档串）即 clip=False；【fork】`scripts/check_scale_factor.py` 用 v1 checkpoint 的 `scale_factor` 做预训练前置闸门（"超阈值 ⇒ 阻断并报告预处理 OOD 排查"）。

### 3.3 编码步骤与 spacing【上游】`:131-191`、`utils.py:787-817`

AMP fp16 → `SlidingWindowInferer(roi_size=[320,320,160], sw_batch_size=1, mode="gaussian", overlap=0.4)` 包 `encode_stage_2_inputs`（`:174-183`）→ `z.squeeze().transpose(1,2,3,0)` **通道置末** `(X,Y,Z,4)`，用 **resize 后的 affine** 存 `*_emb.nii.gz`（`:188-191`）。
`dynamic_infer`（【上游】`utils.py:787-817`）：**总体素数 ≤ 320×320×160 = 16.38M 时走整前向**，否则才滑窗。

> **本仓落点（勿照抄滑窗）**：`docs/spec/data-preparation.md:28` 已裁决**恒整前向 + 超界显式拒绝**（MONAI `SlidingWindowInferer` 的多分辨率拼合对上采样分割网络设计，对下采样 encoder 会把通道维折进空间维、静默产出错误 latent）。#67 沿用；>16.38M 体素的卷另立策略。

**spacing 的准确口径**：训练侧读每个 latent 旁的侧车 `{"spacing":[...], "modality":"..."}`（【上游】`diff_model_train.py:120-121,450-451`，读入 ×1e2），值 = **resize 后的后置 spacing**。fork 的侧车写入器把这条写明了（【fork】`scripts/latent_sidecars.py:29-31`："it is the post-resize physical spacing the condition actually sees"）。→ 我们 rollout 喂 `spacing_tensor` 也必须用后置值；`docs/spec/data-preparation.md:34` 现写的"源 zooms ×1e2"对 MR-RATE 不成立（须改按 latent affine 计算）。

### 3.4 标签映射的权威性（lead 问的那条）

- **映射本身是上游权威**：`【上游】configs/modality_mapping.json` 已含 `mri_t1:9 / mri_t2:10 / mri_flair:11 / mri_mra:16 / mri_swi:20` **以及 `mri_t1_skull_stripped:29 … mri_mra_skull_stripped:33`**。fork 的 diff 只**新增** `mri_t1n:40 / mri_t1ce:41 / mri_t2w:42 / mri_t2f:43`（BraTS 线）。→ 五个脑模态 + 五个 skull-stripped 码**不是 fork 推断**，是随模型一起发布的上游配置。
- "whole-brain 与其 skull-stripped 孪生配对" 也有上游依据：`【上游】docs/inference.md:138` "Whole-brain and skull-stripped share the same FOV since they are **two preprocessings of the same subject**"，且 `data/README.md:172-173` 两行逐模态同数。fork 的 `mrrate_series.py` 只是把这套配对**实现**出来。
- **但 v1 的训练清单 `dataset.json` 未公开**：【上游】`configs/environment_maisi_diff_model_rflow-mr-brain.json:4` 指向 `./dataset.json`，而仓库里**没有该文件**（全仓 grep 只有配置引用与 skills 里的用户数据示例）。→ **「哪些卷进了训练」只能靠计数反推**（§6），标签映射可用、卷清单不可考。

---

## 4. latent 形态与尺寸分布

### 4.1 公式与硬约束

- 源轴长 `a`（RAS 后）→ `r = max(round(a/128),1)×128` → latent 轴 = `r/4`（VAE 空间压缩 4×，`latent_channels=4`；【上游】`configs/config_network_rflow.json:4,18-26`）。**latent 每轴 ∈ {32,64,96,128,…}，形状不固定。**
- 轴序：`Orientationd` 只置换/翻转 → **轴长多重集与朝向无关**，可由元数据 `array_shape` 直接算出（§5.4）。
- **上游推理侧的 dim 硬约束**（运行时校验，`【上游】scripts/sample_mask.py:284-329` `check_input_mr`；skills 文档同）：
  - `output_size[2] ∈ {128, 256}`（**只有这两种**）；
  - `output_size[2]=128` → `output_size[0]=output_size[1] ∈ {128,256,384,512}`；
  - `output_size[2]=256` → 只允许 `[128,256,256] / [256,128,256] / [256,256,256]`；
  - `spacing ∈ [0.4, 5.0] mm/轴`。
  - 另注：`docs/inference.md:188` 写 `dim` "Must be divisible by 16"，skills 写 `dim[0..2]` 至少两个相等——**训练网格与推理 API 允许网格不完全一致**（训练网格可以出现 z=384 这种推理 API 不接受的值）。#67 的 rollout 请求必须落在上面这套允许集内。
- 上游文档记录过的 latent 尺寸（【上游】`docs/performance.md` 两表）：`4x64x64x32 / 4x64x64x64 / 4x128x128x32 / 4x128x128x64 / 4x128x128x128 / 4x128x128x192`（对应 image 256×256×128 … 512×512×768，训练内存表最高 512×512×768）。

### 4.2 典型值（按【上游】`docs/inference.md:142-158` 中位 FOV，~1mm 等效体素示意）

| 模态/平面 | 中位 FOV (mm) | resize 后 | latent |
|---|---|---|---|
| T1 axial | 240×240×174 | 256×256×128 | **4×64×64×32** |
| T1 sagittal | 176×250×250 | 128×256×256 | **4×32×64×64** |
| T1 coronal | 240×200×240 | 256×256×256 | **4×64×64×64** |
| FLAIR axial | 250×250×175 | 256×256×128 | 4×64×64×32 |
| SWI axial | 230×230×145 | 256×256×128 | 4×64×64×32 |
| MRA axial | 220×220×158 | 256×256×128 | 4×64×64×32 |
| T2 axial | 240×240×158 | 256×256×128 | 4×64×64×32 |

长尾：细间距（MRA 0.5 mm → 440 轴 → `round(3.4375)=3` → 384 → latent **96**；480 轴 → 512 → latent **128**）；厚层（5 mm × 32 层 → 128 → latent **32，被上采样 4×**）。单卷 latent 体素数跨度可达 16 倍以上。
→ **不能按固定 `[4,64,64,32]` 做 rollout/判别器/显存预算。**

### 4.3 "higher resolution embeddings" 的准确含义（`data/README.md:168`）

原文：*"Unique scans are counted per source volume; volumes that span a larger field of view produce additional embeddings at higher resolution used during training but are not double-counted."*

- **机制（可证）**：resize 是"取最近的 128 倍数"，**不是**统一压到 128/256——大 FOV / 细体素的卷因此落进更大网格（256/384/512），latent 轴随之变大（64/96/128），这就是 "higher resolution"。
- **"additional embeddings"（复数）**：上游发布的编码脚本**每条目只产 1 个 embedding**（`:239-258` 逐条遍历清单，无内部多分辨率循环），所以"同一源卷多个 embedding"来自 NVIDIA 构造训练清单的方式（同一源卷的不同派生物/平面/网格）。与 §6 的计数差（每源卷 ~4.1 个 embedding）方向一致。
- **"not double-counted"**：指 `data/README.md` §3.4 的模态计数按 source volume 去重。
- **未定案**：源卷 → embedding 的**确切展开规则**上游未公布。

---

## 5. 编码体量定价（lead 点名要的估算）

### 5.1 单卷 latent 体积（C=4、每轴 1/4 下采样）

| 输入 image 尺寸 | latent | fp32 磁盘体积 | **fp16** 磁盘体积 | VAE 时间（A100 80G，解码器，上限参考） |
|---|---|---|---|---|
| 256×256×128 | 4×64×64×32 | 2.10 MB | **1.05 MB** | 1 s |
| 256×256×256 | 4×64×64×64 | 4.19 MB | **2.10 MB** | 2–5 s |
| 512×512×128 | 4×128×128×32 | 8.39 MB | **4.19 MB** | 6–8 s |
| 512×512×512 | 4×128×128×128 | 33.6 MB | **16.8 MB** | 29–32 s |
| 512×512×768 | 4×128×128×192 | 50.3 MB | **25.2 MB** | 57 s |

（落盘实际是 **fp32**：`nib.Nifti1Image(np.float32(out_nda), ...)`，【上游】`:190`。fp16 列供我们自建格式参考。VAE 时间取自【上游】`docs/performance.md` 的 `VAE Time` 列——**该列是 VAE 解码器**、A100 80G，作为编码耗时的**上限参考**；encoder 通常不高于 decoder。）

### 5.2 全量总量级

- 训练条目：**665,371（whole-brain）+ 665,371（skull-stripped）≈ 1,330,742 条**。
- **存储**：
  - 若形状以 256×256×128 为主（T1/FLAIR/SWI/MRA axial 的中位情形）：fp32 **≈ 2.8 TB**，fp16 ≈ 1.4 TB。
  - 若长尾（细间距/大 FOV）占相当比例：fp32 **3–6 TB**，fp16 1.5–3 TB。
  - 对照：原始 zip 整库 **8.1 TB**（下完即可删，不驻留）。
- **算力（上限参考）**：1,330,742 × 1 s ≈ **370 GPU·h**（A100，取 256×256×128 档）；若平均按 2 s 估 → **~740 GPU·h**。DCU 上按实测折减（经验系数 2× 计 → 740–1500 GPU·h）。
- **排期**：8 卡 A100 约 **2 天**（按 370 GPU·h）；DCU 8 卡按 2× 折减约 **4 天**。**单一 512³ 档不可能出现**（全按该档算要 1.1 万 GPU·h，与中位 FOV 表不符）。
- **真瓶颈很可能不是 GPU 而是 I/O**：8.1 TB 下载 + 解压 + 1.33M 个小文件读写（inode/元数据压力）。建议按 shard 打包（每 shard ~1k latents），不要散文件直存。
- **本地承载**：cynosure 侧「manifest + latent 工件对」在 1.33M 量级下**必须分片 manifest**（单文件会到 GB 级），且按模态分层（T2w 669 / MRA 157 与 T1w 54,511 差三个量级，分层不当会让小模态在 shuffle 中消失）。
- 现成闸门可复用：【fork】`scripts/check_scale_factor.py`（latent std ↔ v1 `scale_factor`，阈值 0.2）就是"预处理是否 OOD"的探针，编码完成后先跑它再谈训练。

### 5.3 先用元数据做零下载普查（推荐先做）

`metadata/batchXX_metadata.csv` 逐 series 带 `array_shape` / `array_spacing_mm` / `array_fov_mm` / `acquisition_plane` / `Patient'sAge` / `FieldStrength_T` / `Manufacturer`。因为 resize 逐轴取整、`Orientationd` 只置换，**latent 形状的多重集与体素总量可从 `array_shape` 直接算出**——不下载任何影像就能得到：① latent 形状精确分布与体素总和（→ 精确存储/算力）；② 原生 spacing 分布（校验 §4.2 假设）；③ §6 计数口径的定案。团队已有这份元数据（【fork】`scripts/spine_filter.py:36-42` 的标定就是跑它）。

---

## 6. 划分与用量

- **官方 split**：`splits.csv`，**patient 级**（同一患者所有 study 同 split），study 级存储（guide）。→ 留出集必须用它，自己乱切会因同患者多 study 而**泄漏**。
- **NVIDIA 的用量**（【上游】`docs/inference.md:138`、`docs/data.md:34`、`data/README.md:170-174`）：
  - v1 whole-brain "unique scans"（含 oblique）= **665,371**，unique subjects = **73,516**；
  - 按模态（§3.4，whole-brain 行；skull-stripped 行同数）：T1w 54,511 / T2w 669 / FLAIR 57,641 / SWI 48,205 / MRA 157。
- **我的算术核对（可复核）**：`docs/inference.md:142-158` FOV 表 **v1 列 15 行求和 = 656,565**，+ oblique **8,806** = **665,371** ✅ → v1 列即 665,371 按 (模态, 平面) 的划分，计数单位是 **series/embedding**；而 §3.4 的模态列求和 = **161,183**，两者**不是同一单位**（§3.4 是 source-volume 级，"counted per source volume … not double-counted"）。(64,64,32) 类形状平均每源卷展开 ~4.1 个 embedding。
- **⚠️ 推断（明确标注，未验证）**：705,254（【fork】`scripts/spine_filter.py:36-42` 所述"28 批全量元数据、Train split、已排除质量标记"的 series 数）− 665,371 = **39,883** ≈ 一个 test split 的量级（lead 记的 39,906 与之相差 23）。**这只是一个量级吻合的推断**：35 万→39.9k 的缺口也可能来自额外质检剔除或计数单位差异。**定案前不要把"我们对齐全量"当成"与 v1 语料逐卷相等"。**

---

## 7. 官方文档自相矛盾之处（上游内部，非 fork 引入）

1. **`cfg_guidance_scale`**：`configs/config_maisi_diff_model_rflow-mr-brain.json:34` = **2**，而 `docs/inference.md:195` 写 "MR → **10**（shipped defaults — keep）"、`:214` 写 "is at the shipped default of **10** (not 0)"。**同一上游仓库，配置与文档互相打架**（既有情报 `MG/docs/research/20260908-upstream-rflow-mr-brain-recipe.md` 已记，此处确认纯属上游而非 fork 改动）。
2. **计数口径**：`data/README.md:172` 的 T1w **54,511** vs `docs/inference.md:144` 的 T1 axial v1 **98,769**——不是矛盾，是**计数单位不同**（source-volume 级 vs series/embedding 级，§6），但上游从未说明，极易误读为"上游数字打架"。
3. **v0/v1 两列的语义**：`docs/inference.md:142-158` 的 "Unique scans (v0)/(v1)" 两列比值逐行 1.91–2.08。结合 `README.md` News（"**[August 2026]** — Updated NV-Generate-MR-Brain model weights from **v0 to v1**. Code usage is unchanged." 与 "**[March 2026]** Released ... v0"）→ **v0/v1 是模型版本**；v1 语料约为 v0 的两倍（是"v1 起加入 skull-stripped 双产"还是"v1 重拉了数据"，上游未说明，**未定案**）。
4. **训练网格 vs 推理 API 允许网格**：§4.1（训练网格可以是任意 32 倍数 latent；推理 `check_input_mr` 只接受 z ∈ {128,256} 的少数组合）。

---

## 8. 对 #67 的移植清单

### 直接可用

| 项 | 说明 |
|---|---|
| 六步链语义（§3.1） | 与 BraTS 线同构，仅 clip 参数与 resize 基数需注入 |
| `round_number` 规则 | 逐轴、floor 128、half-to-even（【上游】`:77-90`） |
| 标签映射与双产配对 | 【上游】`configs/modality_mapping.json`（9/10/11/20/16 + 29–33） |
| latent 通道置末 4D 契约 | 与 fork 的 `EmbeddingShapeContract`（`MG/src/ctmr/domain/generation/embedding_shape.py`）同构 |
| 侧车格式/路径约定 | 【上游】`diff_model_train.py:120-121,450-451` 读 `*_emb.nii.gz.json` 的 `spacing` |
| 预训练 OOD 闸门 | 【fork】`check_scale_factor.py`（latent std ↔ v1 `scale_factor`，阈值 0.2） |
| 官方 patient 级 splits | 留出集唯一合法来源 |

### 需要改造

| 项 | 改造点 |
|---|---|
| **强度变换 clip** | 本仓写死 `clip=True`（`src/cynosure/reward/preprocessing.py:65`，ADR-0006）。MR-RATE 线必须参数化为 **False**（§3.2）；并**修正 ADR-0006 的归因错误**——"clip=False 是 MONAI 上游 MAISI 的默认"应为"clip=False 是 NVIDIA NV-Generate-CTMR 的官方代码"（逐字节证据） |
| **latent 域混用** | fork 原话"两边 embedding 不可互用"。若 #67 的 real 池要混 BraTS real（clip=True）与 MR-RATE real（clip=False），判别器会把"归一化制度"当可分性来源（RL 伪信号）。混合前必须统一口径或分层隔离 |
| **可变形状** | rollout/判别器/评测须接受 per-axis 32 倍数的 4D latent，并满足 §4.1 的推理 dim 允许集 |
| **spacing 侧车** | 改用 latent affine 的后置 spacing（§3.3） |
| **存储与 manifest** | 分片 manifest + shard 化 latent；按模态分层 |
| **编码器** | 恒整前向；>16.38M 体素的卷另立分块策略 |
| **FID 仪器** | 【上游】`compute_fid_2-5d_ct.py` **只有 CT 分支**；MR 支持是 fork 扩展。若 #67 要"上游自己的仪器"，须先确定用哪一版并把差异写清 |

### 不适用

| 项 | 原因 |
|---|---|
| `<48 slices` 规则 | 属全身版 `rflow-mr`（`data/README.md:143`），brain v1 无此约束 |
| coreg / atlas / nvseg-ctmr 三仓库 | v1 只用 native-space（【MR-RATE】guide） |
| 上游 `SlidingWindowInferer` 编码 | 对下采样 encoder 静默产出错误 latent（本仓 T12 裁决） |
| 按 batch 整批下载 | 290 GB/batch × 28；须 study-zip 粒度按需拉 |
| z-score / spacing 重采样 / crop | 不在上游链中（照抄即对齐） |

---

## 9. 权威信源索引

| 来源 | 路径 / URL | 用于本文的什么 |
|---|---|---|
| **上游（真）** | `https://github.com/NVIDIA-Medtech/NV-Generate-CTMR`；本地快照 `/tmp/nvidia-upstream/NV-Generate-CTMR-main/`（`da438fe`, 2026-08-19） | 预处理链、config、modality_mapping、data/README、docs、skills |
| fork（旧态） | `~/Documents/NV-Generate-CTMR/`（remote `ACautomata/NV-Generate-CTMR`，HEAD `764e4fb`） | replay 管线、spine filter、sidecar 写入器、`compute_fid_2-5d_ct.py` 的 MR 扩展 |
| fork（新态） | `~/Documents/MR-Generate/`（`8a59815`） | clip=True 偏差记录（`instance_definition.py:25-39,76`）、`embedding_shape.py` |
| MR-RATE 数据集 | `https://huggingface.co/datasets/Forithmus/MR-RATE`（+`-coreg` / `-atlas` / `-nvseg-ctmr`）；guide（Meta 节、§2 Repository Overview、batches 节） | 布局、容量、许可、splits、元数据列 |
| 模型卡 | `https://huggingface.co/nvidia/NV-Generate-MR-Brain`（v1 权重） | 版本时间线 |
| 本仓既有情报 | `docs/research/20260908-upstream-rflow-mr-brain-recipe.md`、`docs/research/distribution-gap-rflow-mr-brain-vs-brats.md`（在 `~/Documents/MR-Generate/`） | 训练配方、分布差（本文不重复） |

---

## 10. 未决问题（交给后续票）

1. **705,254 ↔ 665,371 的 39,883 缺口**（§6）：是 test split 剔除还是额外质检？只能用 §5.3 的元数据普查定案。**这是"全量对齐"能否定义为"逐卷相等"的前提。**
2. **v0/v1 语料翻倍的成因**（§7-3）：加了 skull-stripped 双产，还是 v1 重拉了 MR-RATE？
3. **源卷 → embedding 的展开规则**（§4.3）：代码不可判，需普查 + 计数反推。
4. **clip 的实证裁决**：本文按证据链判定 `clip=False`；若要更硬，可用 fork 的两臂强度域仪器（`MG/src/ctmr/application/acceptance/distribution/intensity_domain.py`）在 MR-RATE 样本上量两臂 latent 统计与重建域，再用 `check_scale_factor` 对照 v1 `scale_factor` 出读数。
5. **T2w 669 / MRA 157 的计数语义**（source-volume 级还是其它），直接决定稀疏模态配平是否有意义（与既有情报未决 #5 同源）。
