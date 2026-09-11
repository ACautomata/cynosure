# 上游评估协议调研：MAISI / NV-Generate-CTMR 的 2.5D FID 仪器

> 日期：2026-09-11
> 服务对象：cynosure MR-RATE 换域实验（wayfinder 地图 #67 → 研究票 #69）
> 信源：
> - **官方上游** `https://github.com/NVIDIA-Medtech/NV-Generate-CTMR`，本地快照 `/tmp/nvidia-upstream/NV-Generate-CTMR-main/`（HEAD `da438fe`，2026-08-19）
> - **团队 fork** `ACautomata/NV-Generate-CTMR`。本机有该 fork 的**两次 checkout**，引用时必须带目录名：
>   `~/Documents/NV-Generate-CTMR`（**重构前**，保留 `scripts/` 层）与 `~/Documents/MR-Generate`（**重构后**，`src/ctmr/` 布局）。详见 §4.4。
> - MAISI v1（arXiv:2409.11169，WACV 2025）与 MAISI-v2（arXiv:2508.05772，AAAI 2026）论文全文
> - `nvidia/NV-Generate-MR-Brain` 模型卡与 commit 历史
>
> 相关背景：`CONTEXT.md`、`docs/spec/experiment-design.md`、`src/cynosure/eval/`

---

## 0. 三层口径必须先分清（全文最重要前提）

调研中反复出现的混淆源是「上游」在本项目里指两个不同的东西，且代码已有第三份实现：

| 层 | 位置 | FID 工具状态 |
|---|---|---|
| **官方上游** | 快照 `/tmp/nvidia-upstream/NV-Generate-CTMR-main/scripts/compute_fid_2-5d_ct.py` | `main()` 在 **L368**，**无 `modality` 参数——纯 CT 仪器** |
| **团队 fork** | `~/Documents/NV-Generate-CTMR/scripts/compute_fid_2-5d_ct.py`（904 行） | `main()` 在 **L518**，**有 `--modality ct\|mr`**；由 commit `ab6d3f0`（2026-09-10，T4 / #33 / closes #20）加入 |
| **cynosure 本仓** | `src/cynosure/eval/` | 零依赖重实现（`features.py` / `frechet.py` / `milestone.py`），不 import 上游 |

**两条已核实的事实：**

1. **官方上游是纯 CT 仪器，没有 MR 路径可用。** 对官方快照 grep：`MODALITY_PREPROCESSING` 出现 **0** 次；`main()` 形参表（L368-386）无 `modality`；强度预处理硬编码在 L570：`ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=-1000, b_max=1000, clip=True)`，padding 硬编码在 L565：`value=-1000`。官方 `docs/evaluation.md` 全文只讲 CT / autoPET 2023。

2. **MR 支持是 fork 的推断性扩展，不是上游行为。** fork 新增了 `ModalityPreprocessing` 数据类与 `modality` 参数（"ct"/"mr"），MR 分支为「动态 percentile 映射（padding 0，与训练管线相同的 (0, 99.5) 百分位窗口，映射到生成器侧 (0, 1000) 输出域）」。**这是团队自研的，官方没有。** 但它也是我们唯一现成的 MR FID 实现，对本项目的价值反而更高。

**信源可用性说明**：`~/Documents/NV-Generate-CTMR/docs/` 与 `data/README.md` 已核实与官方一致（fork 只多 `docs/agents/` 三个文件）；`scripts/transforms.py` 两版 **md5 相同**（`ac1f89b56370c703defccdf11019c2af`），是权威文件。引用这些文档安全。

---

## 1. 仪器规格：2.5D 三正交面 FID

来源：官方与 fork 的 `scripts/compute_fid_2-5d_ct.py` + `docs/evaluation.md`（87 行）。除 §1.6 标注处外，下列规格两版一致。

### 1.1 入口与分布式

- **CLI 是 Python Fire 不是 argparse**：`fire.Fire(main)`，参数即 `main()` 形参，`--flag value` 由 Fire 映射。官方 `setup.md:33` 列 `fire` 的用途为 "CLI for FID evaluation script"。
- `dist.init_process_group(backend="nccl", ...)`，`device = torch.device("cuda", local_rank)`。
- 数据分片：`monai.data.partition_dataset(shuffle=False, num_partitions=world_size, even_divisible=False)[local_rank]`——**按 filelist 排序后轮流分片**。
- 特征跨 rank all-gather 后**只在 rank 0 算 FID**，`pad_to_max_size` 补零再按真实长度裁剪。

### 1.2 样本量口径

- `num_images`（默认 **100**）是**每个数据集各自的上限**：`real_lines.sort(); real_lines[:num_images]` 与 `synth_lines.sort(); synth_lines[:num_images]` 分别截断。**不是两侧合计。**
- filelist 是纯文本、每行一个相对路径（相对 `*_dataset_root`），先 `sort()` 再截断——**顺序无关、集合决定**。
- 送进 FID 的特征行数 = Σ (每卷保留切片数) × 3 平面，**可远大于 `num_images`**。
- **小样本警告**：fork T4 commit 的端到端验证是 3 real + 2 synth，得 FID avg 8.65，commit message 自注 "sample size statistically meaningless by design"。正式对比须回到 `experiment-design.md` 的 N=200–500。

### 1.3 三平面切片（逐字核对）

输入 `(B, C, H, W, D)`，先单通道 `repeat(1,3,1,1,1)` 复制成 3 通道，再 `image = image[:, [2, 1, 0], ...]` 翻转通道序。

| 平面 | 切片轴（`torch.unbind` 的 dim） | 代码 |
|---|---|---|
| XY | **D 轴（dim=-1）** | `torch.unbind(image[:, :, :, :, start_d:end_d:k], dim=-1)` |
| YZ | **H 轴（dim=2）** | `torch.unbind(image[:, :, start_h:end_h:k, :, :], dim=2)` |
| ZX | **W 轴（dim=3）** | `torch.unbind(image[:, :, :, start_w:end_w:k, :], dim=3)` |

命名陷阱：脚本里的 "XY 平面" 是**沿 D 轴切出的 (H, W) 面**，与直觉的「轴标签 = 法轴」不同，照抄时不要按名字推断轴。

- 中心窗口：`start = int((1.0 - ratio) / 2.0 * N)`，`end = int((1.0 + ratio) / 2.0 * N)`，`N` 为该平面法轴长。
- `sample_every_k` 默认 1。
- **`drop_empty` 默认 False**，主流程调用**不传该参数 → 空切片过滤实际未启用**。`empty_threshold` 默认 `-700` 也是 HU 语义。

### 1.4 强度归一化（易踩）

`radimagenet_intensity_normalisation(images_2d)` 在 `torch.cat(slices, dim=0)` **之后**调用，且 `norm2d` 默认 False：

- 走 4D 分支：`max3d = torch.max(volume)` / `min3d = torch.min(volume)` 是**整卷该平面全部切片的全局 min-max**，不是逐切片。因为 DataLoader 是 `batch_size=1`，等价于「每卷 × 每平面」一个 min/max。
- 再 `subtract_mean`，逐字常量 `mean = [0.406, 0.456, 0.485]`——注意这是**通道序翻转后的** ImageNet 值（标准 ImageNet mean 为 [0.485, 0.456, 0.406]，配合 `[2,1,0]` 翻转自洽）。
- **特征网络前不缩放切片到 224×224**——官方脚本直接送原尺寸切片进 ResNet50；224 是 cynosure 侧 `features.py` 的口径（见 §3.2）。

### 1.5 特征网络与距离核

- `torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True, trust_repo=True)`；否则 fallback `torchvision.models.squeezenet1_1(pretrained=True)`。
- 特征出口：`spatial_average(..., keepdim=False)`——4D 输入取 `mean([2,3])`，得 **2048 维**池化特征。
- 距离核直调 MONAI：`from monai.metrics.fid import FIDMetric`，逐面 `fid(synth_xy, real_xy)`。
- 聚合：`fid_avg = (fid_res_xy + fid_res_yz + fid_res_zx) / 3.0`——**算术平均，非加权**。

### 1.6 预处理参数语义

| 参数 | 语义 | 默认 |
|---|---|---|
| `enable_center_slices_ratio` | float 或 None；None = 该轴取满。窗口见 §1.3 | `None` |
| `enable_padding` | `SpatialPadd(spatial_size=target_shape, mode="constant", value=padding_value)` | `True` |
| `enable_center_cropping` | `CenterSpatialCropd(roi_size=target_shape)` | `True` |
| `enable_resampling_spacing` | str `"1.0x1.0x1.0"` 或 None；None 时 `rs_spacing_tuple=(1.0,1.0,1.0)` 但**不建 Spacingd** | `None` |
| `target_shape` | `"512x512x512"` | `"512x512x512"` |
| `num_images` | 每侧体数据上限，见 §1.2 | `100` |
| `model_name` | `radimagenet_resnet50` / `squeezenet1_1` | `radimagenet_resnet50` |
| `ignore_existing` | False 时命中 `.pt` 特征缓存直接 load | `False` |
| `modality` | `ct` / `mr`（**fork 独有**） | `ct`（仅 fork 有） |
| `comparison_tag` / `result_json` | 落盘 `FidResult` provenance（**fork 独有**） | `""` / `None` |

**官方变换链顺序**（L556-570，硬编码）：

```
LoadImaged → EnsureChannelFirstd → Orientationd(RAS)
  → [Spacingd] → [SpatialPadd(value=-1000)] → [CenterSpatialCropd]
  → ScaleIntensityRanged(-1000, 1000, b_min=-1000, b_max=1000, clip=True)
```

**fork 变换链顺序**（`ModalityPreprocessing.compose`）：

```
LoadImaged → EnsureChannelFirstd → Orientationd(RAS)
  → [Spacingd] → [CenterSpatialCropd] → 强度映射（按 modality 分支）
  → [SpatialPadd(value=padding_value)]
```

即 fork 把 **crop 提前、padding 推后、强度夹在中间**。为何必须如此见 §4.3。

> 注：官方 FID 脚本**不 import** `scripts/transforms.py`，它把自己的变换链完整内联。`transforms.py` 是**训练管线**的强度 recipe，仅供参照。

### 1.7 特征缓存陷阱（最隐蔽的坑）

缓存路径 = `output_root[/<modality>]/<features_dir>/<镜像相对路径>.pt`（`<modality>` 段是 fork 加的）。`ignore_existing=False` 时命中即复用。

**`enable_center_slices_ratio` / `target_shape` / `enable_resampling_spacing` 等几何参数的变化不会使缓存失效**——改了 ratio 却复用旧 `.pt` 会静默产出错误数字。fork 的 `FidResult` 落盘 provenance 供事后识别，但运行时无防护。实务纪律：每次换几何口径必须 `--ignore_existing True` 或用不同 `features_dir`。

`FidResult` 字段：`comparison_tag, fid_xy, fid_yz, fid_zx, fid_avg, modality, model_name, num_images, real_filelist, synth_filelist, target_shape, center_slices_ratio, enable_padding, enable_center_cropping, enable_resampling_spacing`。

---

## 2. 可引用数字（逐表标注模态）

### 2.1 全部生成质量数字都是 CT —— MR/brain 数字为「无」

| 表 | 出处 | 模态 / 参考集 | 可引用 |
|---|---|---|---|
| 官方/fork `docs/evaluation.md` Table 1 | FID 逐面 + avg | **CT** / autoPET 2023 | ✅ |
| 官方/fork `docs/evaluation.md` Table 2 | VAE LPIPS/SSIM/PSNR | CT ×2 + **BraTS18 MRI ×1** | ⚠️ 唯一 MRI 数字，是**重建**不是生成 |
| `README.md` §4 benchmark | FID | **CT** / autoPET 2023 | ✅ |
| MAISI v1 论文 Table 2 | FID（含 real-vs-real 地板） | **CT** / MSD Task06・LIDC-IDRI・TCIA COVID-19 | ✅ |
| MAISI v1 论文 Table 3 | FID 逐面 + avg | **CT** / autoPET 2023 | ✅ |
| MAISI v2 论文 Table 1 | FID 逐面 + avg | **CT** / AutoPET2023 | ✅ |
| MAISI v2 论文 Table 3 | FID 步数消融 | **CT** / AutoPET23 | ✅ |
| MAISI v2 论文 Table 4 | 分割 Dice | CT 肿瘤任务 | ✅ |
| `nvidia/NV-Generate-MR-Brain` 模型卡 | — | — | ❌ **无任何数字** |
| MR-RATE 数据集卡 | — | — | ❌ 论文 "Coming Soon" |

### 2.2 官方上游的 benchmark（`README.md` §4，autoPET 2023）

| Model | FID Score | Inference Steps | Speed vs ddpm-ct |
|---|---|---|---|
| `rflow-ct` | **5.124** | 30 | **33x faster** |
| `ddpm-ct` | 6.083 | 1000 | baseline |

`docs/evaluation.md` Table 1 给出逐面拆分：

| Method | FID(XY) | FID(YZ) | FID(ZX) | FID(Avg) |
|---|---|---|---|---|
| DDPM | 18.524 | 23.696 | 25.604 | 22.608 |
| LDM | 16.853 | 10.191 | 10.093 | 12.379 |
| HA-GAN | 17.432 | 10.266 | 13.572 | 13.757 |
| MAISI (`ddpm-ct`) | 3.301 | 5.838 | 9.109 | 6.083 |
| MAISI (`rflow-ct`) | 2.685 | 4.723 | 7.963 | **5.124** |

Table 2（VAE 重建）：

| Dataset | Model | LPIPS | SSIM | PSNR | GPU |
|---|---|---|---|---|---|
| MSD Task07 | MAISI-v1 VAE | 0.038 | 0.978 | 37.266 | 0h |
| MSD Task08 | MAISI-v1 VAE | 0.046 | 0.970 | 36.559 | 0h |
| **Brats18** | MAISI-v1 VAE | **0.026** | **0.977** | **39.003** | **0h** |
| Brats18 | Dedicated VAE | 0.030 | 0.975 | 38.971 | 672h |

Brats18 = post-contrast T1-weighted MRI，是**上游全部公开材料里唯一带 MRI 的量化数字**，且是 VAE 重建指标、单一序列、无 FID。

### 2.3 MAISI v1 论文（arXiv:2409.11169）

Table 2 —— **real-vs-real FID 地板**，方法学上最值得抄的一项：

| FID(Avg) | MSD Task06 | LIDC-IDRI | TCIA COVID-19 |
|---|---|---|---|
| Real MSD Task06 | – | 3.987 | 1.858 |
| Real LIDC-IDRI | 3.987 | – | 4.744 |
| Real TCIA COVID-19 | 1.858 | 4.744 | – |
| HA-GAN | 98.208 | 116.260 | 98.064 |
| MAISI DM | 4.349 | 6.200 | 8.346 |

Table 3（unseen autoPET 2023 为参考）：

| Method | Axial | Sagittal | Coronal | Avg |
|---|---|---|---|---|
| DDPM | 18.524 | 23.696 | 25.604 | 22.608 |
| LDM | 16.853 | 10.191 | 10.093 | 12.379 |
| HA-GAN | 17.432 | 10.266 | 13.572 | 13.757 |
| MAISI DM | 3.301 | 5.838 | 9.109 | **6.083** |

论文的 FID 协议描述仅一句，**未点名特征骨干**：

> "We use the Fréchet Inception Distance (FID) [27] as the metric for evaluating the similarity between the distributions of generated images and real counterparts from varied sources."

v1 全文无 "central"、无 "2.5D"——切片口径是 v2 才写清楚的。

### 2.4 MAISI v2 论文（arXiv:2508.05772）

协议原文（逐字）：

> "The metric we use is **2D Fréchet Inception Distance (FID)** between synthesized images and the unseen AutoPET2023 dataset for xy, yz, and xz-planes."
> "To avoid empty slices, we compute FID on the **central 50%** of the slices."
> "The reason we use 2D FID in three planes instead of 3D FID is to better reflect human perception."

Table 1（512³ @ 1 mm³，H100）：

| Model | Steps | FID xy | FID yz | FID xz | FID avg |
|---|---|---|---|---|---|
| HA-GAN | 1 | 13.813 | 12.567 | 14.405 | 13.595 |
| MedSyn (2-stage DDIM) | 50+20 | 18.662 | 22.171 | 33.293 | 24.709 |
| GenerateCT (2D EDM) | 25×201 | 7.909 | 9.256 | 15.106 | 10.757 |
| MAISI (DDPM) | 1000 | 2.199 | 2.480 | 2.642 | **2.441** |
| MAISI (DDIM) | 30 | 4.855 | 4.703 | 4.770 | 4.776 |
| MAISI-v2 (Rectified Flow) | 30 | 2.217 | 2.211 | 2.538 | **2.322** |

Table 3 —— 步数消融，**每格都是 autoPET23（CT）**：

> "Table 3: FID scores across inference steps for MAISI-v2 LDM. Each value represents the score on the OOD **AutoPET23** dataset (lower is better)."

| Metric | 5 | 10 | 20 | 30 | 40 | 50 | 70 | 100 |
|---|---|---|---|---|---|---|---|---|
| FID xy | 22.774 | 4.232 | 2.538 | 2.217 | 2.088 | 1.966 | 1.818 | 1.783 |
| FID yz | 13.907 | 3.765 | 2.446 | 2.211 | 2.119 | 2.001 | 1.970 | 1.960 |
| FID xz | 24.321 | 5.266 | 2.952 | 2.538 | 2.405 | 2.225 | 2.156 | 2.159 |
| **FID avg** | **20.334** | **4.421** | 2.645 | **2.322** | 2.204 | 2.064 | **1.981** | **1.967** |

### 2.5 MR / brain 数字：明确为「无」——查过的范围

**（c）答案：官方与 fork 都不含任何 MR/brain 的 FID 数字。**

**MAISI v2 全文词频**（纯文本 grep）：`MRI` = **1**、`MR` = **0**、`MR-RATE`/`RATE` = **0**、`brain` = 4。

- 唯一的 `MRI` 在 Limitations：

  > "**Limitations and Future Work:** This work has three main limitations. **(1) We only trained on CT scans. In the future, we plan to include other types like MRI and PET.** (2) We only tested the model on segmentation tasks. ..."

- 4 次 `brain` 无一是脑部评估：1 次是 quality-check 器官 HU 中位数列表，3 次是参考文献标题。

**MAISI v1 全文**：`MRI` 11 次，全部是训练数据组成（17,887 MRI volumes 训练 / 940 验证，措辞为 "to potentially support MRI modality in future work"）、动机论述或参考文献；`BraTS` 1 次（§2.2 的 VAE 表）。v1 Limitations 只谈人口统计学偏差与算力，**未提 MRI**。

**官方快照全仓扫描**：`docs/` 内 FID 引用只出现在 `evaluation.md`（纯 CT）与 `setup.md:33`（`fire` 依赖说明）；`README.md` 的 FID 表只有 `rflow-ct` / `ddpm-ct`。MR 相关的 config（`environment_rflow-mr.json`、`config_maisi_diff_model_rflow-mr-brain.json`、`label_dict_ctmr.json` 等）全部是**推理/训练**配置，**无一是评估**。**官方没有 MR FID 脚本。**

**`nvidia/NV-Generate-MR-Brain` 模型卡全文**：无 Evaluation 章节、无任何 FID 数字。唯一质量陈述是定性的：

> "...designed to generate high-quality synthetic brain magnetic resonance imaging (MRI) images, **achieving the highest resolution and best FID scores among comparable models**."

卡片其余量化信息：240M 参数、分辨率 0.45×0.45×0.7mm、最大 512×512×256、数据划分约 28,000 / 8,000 / 4,000 扫查（train/test/eval）、四序列 T1/FLAIR/T2/SWI 整数标签选择、MONAI Core 1.5、A100/H100。

两处细节：frontmatter 写的是 `arxiv: 2409.11169`（**MAISI v1**）而 References 同时列 v1/v2——**该模型卡没有绑定到任何给出它自己数字的论文**；卡片 "Model Version(s): 0.1"，与权重实际版本脱节。

**commit 历史**（`huggingface.co/api/models/nvidia/NV-Generate-MR-Brain/commits/main`）共 19 条，权重相关两条：

```
2026-08-12  Add diff_unet_3d_rflow-mr-brain_v1.pt model weights   ef9759bf
2026-03-15  add model v0                                          67ba36a4
```

**2026-08-12 是纯权重 commit，未附带任何评测说明，README 在该 commit 中未改动。**

**MR-RATE 数据集卡**：引用区块为占位符 —— Paper badge "Paper (Coming Soon)"、"Citing Us: Coming soon"。无 arXiv ID / DOI / 期刊。规格：705,254 volumes、98,334 studies、83,425 患者、序列 T1w/T2w/FLAIR/SWI/MRA。**无任何性能指标。**

**NVIDIA 技术博客**：只有推理步数（MR 模型 30 步）与数据集规模，无质量指标。

### 2.6 三处口径不一致（引用时必须声明）

1. **同一模型两篇论文数字差 2.5 倍**：MAISI DDPM 1000 步，v1 Table 3 报 **6.083**，v2 Table 1 报 **2.441**。baseline 集合也整体更换（v1 的 LDM 在 v2 中消失，换成 MedSyn/GenerateCT）。
2. **fork 公开数字与论文数字对不上**：`rflow-ct` 在 `docs/evaluation.md` 是 **5.124**，MAISI-v2 论文的 Rectified Flow 是 **2.322**。
3. **中心切片比例不一致**：论文写 central **50%**，`docs/evaluation.md` 的示例命令用 `--enable_center_slices_ratio 0.4`。

三处共同指向同一结论：**协议细节（切片比例、样本量、参考集版本）主导数值量级，跨来源的绝对 FID 不可直接比较。**

---

## 3. 复现所需组件清单

### 3.1 权重

- **RadImageNet-ResNet50**：官方路径是 `torch.hub.load("Warvito/radimagenet-models", ...)`——**需要外网**，离线集群不可用。
- cynosure 侧改为本地文件注入：`config.artifacts.radimagenet_weights`（`src/cynosure/eval/features.py`），下载脚本属施工。
- 装载坑（已记录在 `features.py`）：RadImageNet 官方发布（Keras 转换，conv 全带 bias）在 MONAI resnet50 拓扑外**多 49 个 conv bias 键**，装载前按拓扑键过滤；拓扑键缺失则显式失败、不静默随机初始化。丢弃 bias 后前向残留 γb/σ 的 per-channel 偏移，**实测池化特征相对差异 1.2%（8 样本 max 1.3%）**——对跨里程碑相对比较无实质影响，但阻断与 RadImageNet 原版前向的绝对对齐。
- 基座 checkpoints：`diff_unet_3d_rflow-mr-brain_v1.pt`（nvidia/NV-Generate-MR-Brain）+ `autoencoder_v1.pt`（nvidia/NV-Generate-CT）。

### 3.2 依赖与 cynosure 的偏离

| 上游依赖 | cynosure 对应 | 偏离理由 |
|---|---|---|
| `monai.metrics.fid.FIDMetric` | `src/cynosure/eval/frechet.py::FrechetDistance` | MONAI 的 `FIDMetric` **依赖 scipy** 做矩阵平方根；scipy 不在 import 白名单、且集群侧有 numpy 钉版约束。cynosure 改用 `torch.linalg.eigh` 谱分解 |
| 数值细节 | `compute_frechet_distance(..., epsilon=1e-6)`，对角加 `eye*epsilon` 后 `_sqrtm((Σa+εI)·(Σb+εI))` | cynosure 走对称化 `√Σa·Σb·√Σa` + `clamp_min(0)` 钳负特征值，**无 epsilon 对角加载** |
| `torchvision.models.squeezenet1_1` fallback | 无 | cynosure 只有 RadImageNet / fixture stub 两条路 |
| 切片前不 resize | `features.py` 先 `interpolate` 到 **224×224** bilinear 再复制 3 通道 | **口径差异，须核对**：官方脚本直接把原尺寸切片送进 ResNet50 |
| `fire` CLI | config 驱动 | 项目风格 |
| NCCL | 集群为 RCCL 兼容层 | `backend="nccl"` 在 torch-dcu 上是别名，通常可用；需实测 |

cynosure 还额外实现了 MONAI 没有的 **KID**：`KernelMmd`（无偏 MMD²，多项式核 `(x·y/d + 1)³`）+ `BootstrapKernelMmd`（**无放回 m-out-of-n 半样本子抽样** CI，非 Efron 有放回 bootstrap）。

### 3.3 DCU（曙光 SothisAI）可行性

- **纯 PyTorch/MONAI 变换**（`Spacingd` / `CenterSpatialCropd` / `ScaleIntensityRangePercentilesd` / `SpatialPadd`）——CPU/PyTorch 实现，DCU 无特殊障碍。
- `torch.distributed` + `monai.data.partition_dataset`——无特殊依赖，走 RCCL。
- `device = torch.device("cuda", local_rank)`——torch-dcu 上 `cuda` 是别名，可用。
- **阻断项**：`torch.hub.load(...)` 需要外网。集群上必须离线注入权重（cynosure 的本地文件注入设计正是为此）。
- 按 `sugon-bootstrap` 的纪律：装任何依赖后回验 `import torch`，必要时钉 `numpy==1.26.4`。

---

## 4. fork 的 MR 分支：自洽性裁定与自建改动清单

### 4.1 官方 CT 版 → MR 的完整改动清单（回答 b）

要把官方 CT 仪器改成 MR 仪器，**必须**改以下 6 处（fork 已全部实施，编号对应 fork diff）：

| # | 位置 | 官方 CT 版 | 改动 |
|---|---|---|---|
| 1 | `main()` 形参表 | 无 `modality` | 加 `modality: str = "ct"` + `MODALITY_PREPROCESSING` 查表校验（拒绝未知值） |
| 2 | 变换链顺序 | **pad → crop → 强度** | **crop → 强度 → pad**（MR 下必须，见 §4.3） |
| 3 | 强度变换 | 硬编码 `ScaleIntensityRanged(-1000,1000, b_min=-1000, b_max=1000, clip=True)` | 按 modality 分支：MR 走 `ScaleIntensityRangePercentilesd(0, 99.5, b_min=0, b_max=1000, clip=False)` |
| 4 | padding 值 | 硬编码 `value=-1000` | 参数化为 `padding_value`（MR=0，CT=-1000） |
| 5 | 特征缓存路径 | `output_root/<features_dir>` | `output_root/<modality>/<features_dir>`（防跨模态复用 `.pt`） |
| 6 | 结果落盘 | 仅 log | （fork 的 T8 增强）`FidResult` dataclass + `--result_json` + `comparison_tag` |

**结构性要点**：官方 FID 脚本**刻意不复用** `transforms.py` 的强度 recipe。对比同一份 `transforms.py`（md5 两版一致）的 CT 分支 `ScaleIntensityRanged(-1000,1000, b_min=0.0, b_max=1.0, clip=True)` 与 FID 脚本的 CT 分支 `b_min=-1000, b_max=1000`——**两者输出域不同，因为服务对象不同**：

- `transforms.py` → **VAE 输入域**（[0,1]）
- FID 脚本 → **生成器输出域**（CT `(-1000,1000)`，MR `(0,1000)`，与 `utils_infer.py` 一致）

fork 的 MR 分支沿用了同一设计原则（瞄准生成器侧输出域），它的 docstring 引用 `transforms.py` 只是为了说明**百分位窗口的来源**，不是输出域。这个区分是理解整套设计的钥匙。

**注意区分两个正交的轴**：「输出**域**」（由消费侧决定，见上）与「**clip 取值**」（同一域内的 recipe 选择，见 §4.4）。前者是设计原则、必须按消费域分；后者是 fork 的一处主动偏离、cynosure 已裁定对齐真上游。不要把两者混为一谈。

### 4.2 MR 分支的完整口径引用（标注为 fork 推断性扩展）

`ModalityPreprocessing` docstring 逐字：

> "CT has an absolute physical scale (HU), so both datasets share a fixed [-1000, 1000] clip window. **MR intensity is scanner/sequence dependent — real volumes can far exceed 1000 — so both datasets are instead mapped with the dynamic percentile window the training pipeline uses (scripts/transforms.py: ScaleIntensityRangePercentilesd(0, 99.5), clip off) into the generator-side (0, 1000) output domain (scripts/utils_infer.py).** The same single Compose instance built by `compose` must serve both the real and the synth loader so FID stays comparable."

常量表逐字：

```python
MODALITY_PREPROCESSING = {
    "ct": ModalityPreprocessing(name="ct", padding_value=-1000, output_range=(-1000, 1000)),
    "mr": ModalityPreprocessing(name="mr", padding_value=0, output_range=(0, 1000), percentile_range=(0.0, 99.5)),
}
```

commit message 逐字（`ab6d3f0`）：

> "mr: pad value 0 and dynamic intensity mapping — real MR volumes far exceed 1000 (measured BraTS t1n range 0-3290), so a fixed clip would truncate the bright tail. Both real and synth are mapped with the training pipeline's `ScaleIntensityRangePercentilesd(0, 99.5)` window into the generator-side (0, 1000) output domain, clip off (matches `utils_infer.py` which only clips the lower bound)."

**生成侧 (0, 1000) 域的来源已核实**（`utils_infer.py`）：

```python
if modality_tensor is not None and modality_tensor <= 7:
    a_min, a_max = -1000, 1000     # CT
else:
    a_min, a_max = 0, 1000         # MRI
b_min, b_max = 0.0, 1.0            # autoencoder output range

if modality <= 7:  synthetic_images = torch.clip(synthetic_images, b_min, b_max)
else:              synthetic_images = torch.clip(synthetic_images, b_min, None)   # MR 只 clip 下界
synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
synthetic_images = synthetic_images * (a_max - a_min) + a_min
```

MR 落盘域 = `x * 1000 + 0` = **(0, 1000)，仅下界 clip**——与 FID 工具 MR 路径的目标域一致，**自洽**。

### 4.3 自洽性裁定（回答 a）：**一致，已数值验证**

fork 声称其 MR 分支用的是「训练管线的 (0, 99.5) 窗口」。核对结果：

| 维度 | **路径 A** `scripts/transforms.py:64`（md5 `ac1f89b5` 官方与 fork 一致） | fork FID MR 分支 | 判定 |
|---|---|---|---|
| 百分位窗口 | `lower=0.0, upper=99.5` | `lower=0.0, upper=99.5` | ✅ 一致 |
| `clip` | `False` | `False` | ✅ 一致 |
| 输出域 | `b_min=0.0, b_max=1` → **[0, 1]** | `b_min=0, b_max=1000` → **(0, 1000)** | ⚠️ 不同，但**不可见**（见下） |

**输出域差异被下游归一化完全抵消。** 理由：`ScaleIntensityRangePercentiles` 在 `b_min=0` 时输出 = `(x − p0) / (p99.5 − p0) · b_max`，是输入的正线性缩放；紧随其后的 `radimagenet_intensity_normalisation`（`norm2d=False` 的 4D 分支）做**逐卷逐平面全局 min-max** `(v − min) / (max − min)`，对正缩放因子完全免疫。

**实测验证**（MONAI 1.6.0，BraTS 风格非负亮尾强度 seed=0）：

```
训练域范围 [0.0000, 1.0093]
FID 域范围  [0.0000, 1009.3409]
原始域比例  1000.000000  (应 = 1000)
归一化后最大绝对差 5.960e-08
完全一致: True
```

**裁定：fork 的 MR 分支在数值上自洽。** docstring 唯一的不精确之处是把 `transforms.py` 的「百分位窗口」与「输出域」写成了一个短语；实际上窗口共享、(0,1000) 输出域来自生成侧 `utils_infer.py`，而两者在最终特征上等价。

**顺带确证 fork 的「强度先于 padding」修复是必要且有效的**：

1. MONAI 的 `CenterSpatialCropd` 在 `roi_size` 大于影像尺寸时**原样返回、不补零**（实测：4³ 影像用 `roi_size=(8,8,8)` 裁剪 → 输出仍是 4³）。因此 fork 的 crop → 强度 → pad 顺序下，**百分位统计只看真实内容，没有被任何零填充污染**。修复成立。
2. 若不改顺序（官方 pad → crop → 强度），padding 的 0 会进入百分位统计。实测（16³ 内容 pad 到 64³，内容仅占 1.56%）：内容范围从 `[0.0, 1009.3]` 变为 `[0.0, 2217.8]`，**缩放约 2.2 倍**，且倍数**随 padding 体素数（即体积大小）变化**——正如 fork docstring 所述 "padding voxels can outnumber the content and would dilute the percentiles by a volume-size dependent share, making real/synth normalization inconsistent"。

3. **CT 路径的顺序调整也确证数值等价**（fork 的 claim 成立）：官方 pad(-1000) → crop → clip，fork crop → clip → pad(-1000)。当影像 > target 时两者都是「中心裁剪」；当影像 < target 时官方补 -1000、fork 补 padding_value=-1000；clip 窗口与 -1000 相容，裁剪与填充在轴上独立。故 CT 数字不受此次重排影响——**T4 没有改动任何已冻结的 CT 记录**。

### 4.4 编码端 `clip` 歧义：已裁定（cynosure#71 关闭）

> **本节是对报告初版的更正。** 初版称「fork 内唯一的百分位变换是 `clip=False`，cynosure 的 `clip=True` 与 fork 不一致」——那是我**只 grep 了 `scripts/` 目录**造成的范围错误。fork 实际有**两条**强度路径，且 cynosure 跟随的是另一条。

**fork 有两条强度路径：**

| 路径 | 文件 | clip | 服务对象 |
|---|---|---|---|
| **A** | `scripts/transforms.py:64` | `False` | VAE 训练 / ControlNet 数据 |
| **B** | `src/ctmr/infrastructure/maisi_engine/instance_definition.py:76` `define_fixed_intensity_transform` | **`True`** | **DM 训练数据编码链**（`diff_model_create_training_data.py` → `create_training_data.py`） |

**B 与 A 是同源关系，不是并列实现。** `instance_definition.py` 模块头逐字：

> "- `define_fixed_intensity_transform` from the retired scripts layer (git history; ``transforms``).py"

即 fork 把 `scripts/transforms.py` 的函数**搬迁**进了 `src/ctmr/`，并在搬迁时把 mri 臂改成了 `clip=True`。这是**明文记录的、唯一的一次**对「逐字节照搬」规则的偏离：

> "One recorded deviation from the byte-for-byte rule (issue #251, series-② T4; the one recipe delta the retrain ticket pins): the mri arm's normalization flag is ``clip=True`` -- upstream shipped ``clip=False``. Job C measured the trade ...: the unclipped affine extrapolates the top ~0.5% of training t1c voxels above 1.0, out of the frozen autoencoder_v1's reconstruction domain (extrapolated-band self-eval MAE 0.8673 with intra-tumour negative-value artifacts vs 0.0062 for truncated inputs); truncation aligns the encoding input domain with the frozen VAE."

**裁定**：cynosure 的 `clip=True` **不是跟错，是忠实地跟随了 B**；偏差的源头在 fork（#251），不在 cynosure。

**歧义的适用边界**：B 修的是**编解码往返**（encode→decode 重建质量，MAE 0.8673 → 0.0062，且伴随瘤内负值伪影）。而 RL 的 real pool **只需要编码侧**——不需要解码回来。所以 B 的收益（重建保真）在此路径上拿不到，而它的代价（把最亮的 0.5% 体素截断到 1.0，使编码输入分布偏离真上游）却全额继承。这正是「照抄 B」在此处不划算的原因。

**已裁定**：cynosure#71 关闭，用户决定**编码端改用 `clip=False`（对齐真上游 / 路径 A）**。

**注意这与 FID 仪器的 `clip` 无关，两者按消费域分属不同设定**：FID 脚本的 MR 分支用 `clip=False`，因为它服务的是**生成器输出域**（见 §4.1）；编码链的 `clip` 服务 **VAE 输入域**。两者不必一致，也不应一致。

**顺带澄清本地目录布局**：`~/Documents/NV-Generate-CTMR/` 与 `~/Documents/MR-Generate/` 的 remote **都是** `ACautomata/NV-Generate-CTMR`，只是两次不同时期的 checkout——前者是**重构前**（保留 `scripts/` 层，路径 A），后者是**重构后**（`src/ctmr/` 布局，路径 B，commit `8a59815`）。所以「fork 是哪一份」在本机是**歧义表述**，引用时必须带目录名。（`MR-Generate` 下**没有** FID 脚本，FID 仪器只在 `NV-Generate-CTMR/scripts/` 一份。）

### 4.5 其余 MR 适配注意

- `drop_empty` 在 MR 路径**同样未启用**（主流程不传该参数）。其 `empty_threshold=-700` 是 HU 语义，若将来要在 MR 上启用空切片过滤，阈值须重新标定。
- `modality` 同时驱动特征缓存命名空间，切换时**务必** `--ignore_existing True`。
- `--modality` **未写进 `docs/evaluation.md` 的示例命令**（该文件未随 T4 更新，官方与 fork 均无）——照抄文档会漏掉这个参数。

---

## 5. 自测对比的判据设计

冻结基座 vs RL 后 policy 的配对对比，须控制的变量：

### 5.1 必须冻结的变量（任一变动即破坏可比性）

1. `num_images` —— 双侧各自截断；两侧基数变化会同时改变 FID 的有偏程度。
2. `real_filelist` / `synth_filelist` 的**集合**（脚本会 `sort()`，行序无关）。real 侧参照集必须逐字节固定。
3. 几何口径：`target_shape` + `enable_resampling_spacing` + `enable_padding` + `enable_center_cropping`。
4. `enable_center_slices_ratio` —— **论文 0.5 vs docs 示例 0.4，必须在 config 里钉死一个值**，并写进 `FidResult`。
5. `modality` —— 决定强度域，同时决定特征缓存命名空间。
6. `model_name` —— `radimagenet_resnet50` 与 `squeezenet1_1` 是**完全不同的特征空间**，数字不可比。
7. RadImageNet 权重**文件本体**（同版本同文件；bias 键过滤行为随文件而变）。
8. 特征提取设备与 dtype（`EXTRACT_BATCH` 分块本身不改数值，但设备浮点差异会）。
9. `ignore_existing` / `features_dir` —— §1.7 的缓存陷阱；**换几何口径必须重算特征**。

### 5.2 配对设计（cynosure 已有实现）

- **同 manifest 条目 = 同初始噪声 + 同条件集**，差异唯一归因于 RL（`src/cynosure/eval/__init__.py::ManifestEvaluation`；`docs/spec/experiment-design.md`「对照基线」）。
- 冻结模型**只采一次**；RL 后 policy 用相同 seed 与条件重采（`sample_baseline()` / `resample()`）。
- 参照分布取自**病例级不相交**的 held-out（BraTS 病例级 70/10/20），不越过分割。
- **分层出数**：cynosure `MilestoneEvaluator` 已按**目标序列**分层（`MilestoneMetrics`「按目标序列分层的 KID」）；上游无此口径，属自建。

### 5.3 补 real-vs-real 地板（建议采纳上游做法）

MAISI v1 论文 Table 2 报告了真实数据两两之间的 FID（1.858–4.744）。**没有地板值，绝对 FID 无法解读**——这正是同一模型能在两篇论文里报出 2.441 与 6.083 的原因。

建议：在 MR-RATE / BraTS 上按同样的 2.5D 口径算 real-vs-real 地板，作为解读所有绝对数字的锚点，并作为「仪器是否正确接线」的自检（地板值应与合成值同量级，而非数量级更小）。

### 5.4 统计口径

- **KID + bootstrap CI** 是 FID 之外的必配：FID 是有偏估计，小样本下系统性偏高，**只做相对比较、不跨样本量比绝对值**。
- cynosure 的 CI 用**无放回 m-out-of-n 半样本子抽样**（`BootstrapKernelMmd`），非 Efron 有放回 bootstrap——理由：重复行会抬高组内相似度、使重复分布系统性偏离点估计（小样本下尤甚）。
- 样本量回到 `experiment-design.md` 的 **N = 200–500**；fork T4 的 3+2 验证（FID 8.65）只证明管线跑通，不构成任何质量结论。
- 主判据是**相对各自 no-RL 基线的统计显著提升**（配对检验），不是绝对阈值。

---

## 6. 结论与移交

1. **官方上游没有 MR FID 仪器**（无 `--modality`、强度与 padding 硬编码为 CT），**也没有任何 MR/brain 的公开 FID 数字**（模型卡无数字、`README.md` benchmark 表只有 CT、MR-RATE 论文未发布、MAISI v1/v2 均为 CT-only）。**cynosure 无法与上游公开数字做对照**；能做的是在 CT 侧复现仪器以证明实现正确，再在 MR 上自建基线。

2. **fork 的 MR 分支在数值上自洽，可以作为自建基准。** 已实测确证：百分位窗口 (0, 99.5) 与 `clip=False` 与**路径 A**（`~/Documents/NV-Generate-CTMR/scripts/transforms.py:64`，与 FID 脚本同一 checkout）完全一致；输出域 (0,1000) vs [0,1] 的差异被下游 `radimagenet_intensity_normalisation` 的全局 min-max 完全抵消（归一化后最大差 5.96e-08）。「强度先于 padding」的修复必要且有效（不改顺序会因零填充使内容缩放约 2.2 倍且随体积大小漂移）。

3. **自建 MR FID 的改动清单是 6 处**（§4.1），其中 3 处是关键：变换链顺序（crop → 强度 → pad）、强度变换换成 `ScaleIntensityRangePercentilesd(0, 99.5, 0, 1000, clip=False)`、`padding_value` 参数化为 0。

4. **编码端 `clip` 歧义已裁定（cynosure#71 关闭）**：fork 有两条强度路径（A `scripts/transforms.py:64` = `clip=False`；B `src/ctmr/.../instance_definition.py:76` = `clip=True`，为 fork 明文记录的 #251 偏离），cynosure 的 `clip=True` 是**忠实跟随 B**、不是跟错。B 修的是编解码往返，而 RL 的 real pool 只需编码侧，收益拿不到、代价全继承。**用户已决定编码端改用 `clip=False`（对齐真上游）。** 详见 §4.4。

5. **引用纪律**：任何 MAISI 的 FID 数字（含步数消融 20.33 → 1.97）都**必须标注为 CT / autoPET23**；Brats18 的 0.026/0.977/39.003 是 VAE 重建、单一序列、非 FID。

6. **落盘纪律**：每次 FID 运行写 `FidResult` JSON，并在报告数字时附带 `center_slices_ratio` / `num_images` / `modality` / 权重文件标识。

7. **本报告新增证据的可复现性**：§4.3 的两项数值验证均在 MONAI 1.6.0 + CPU 上以约 20 行脚本完成（方差来源：`ScaleIntensityRangePercentiles` 的 `b_min=0` 缩放不变性、`CenterSpatialCropd` 在 roi > size 时不补零），建议固化为一条 fixture 测试。
