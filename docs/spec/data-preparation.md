# 数据准备方案：BraTS 加载对齐上游训练 recipe

> 本章由 grilling 会话决议产出（2026-09-04）。上游（NV-Generate-CTMR fork）预处理链事实经只读代码核查；决策记录见 `docs/adr/0006-brats-preprocessing-upstream-recipe.md`。MR-RATE 换域段（文末）由地图 #67 / #121+#131 产出（2026-09-16）。

## 范围与原则

- **零依赖原则**：上游 fork 代码只读参照、永不 import。对齐的是**数据处理语义**，不是代码。
- **对齐目标 = 上游训练数据创建链**（fork `src/ctmr/infrastructure/maisi_engine/create_training_data.py`），非其推理/评测链（`fid_2d5.py` 的 Spacingd、instrument 链 GridResampler、mask 增强等均不适用）。
- **权威性**：recipe 偏差以 fork 实际代码为准，不以 MONAI 上游 MAISI 默认为准（见 ADR 的 clip 讨论）。
- **改造点唯一**：全仓库读原始 NIfTI 的位置只有 `PreparePipeline._encode_one`（`src/cynosure/reward/pipeline.py`）；训练/评测侧只消费预编码 latent，本次零改动。

## 上游 recipe（权威事实，逐项）

fork `create_training_data.py:55-96` `create_transforms` 的六步链：

1. `LoadImaged`（NIfTI 读取）
2. `EnsureChannelFirstd`
3. `Orientationd(axcodes="RAS")`
4. `EnsureTyped(dtype=torch.float32)`
5. `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=True)` —— mri 臂；**clip=True 是本 fork 对 MONAI 上游 MAISI（clip=False）的唯一有意 recipe 偏差**（fork issue #251），两边 embedding 不可互用
6. `Resized(spatial_size=dim, mode="trilinear")`，目标 dim 由 `round_number` 计算：**每轴 `max(round(size/128), 1) * 128`，size 从 RAS 重定向后的 spatial shape 读取**（fork issue #312：从 NIfTI header 存储轴读对轴置换方向会错；BraTS flip-only 幸免）。BraTS 240→256、155→128

明确不存在于链中的步骤（同样是对齐结论）：

- **无 Spacing 重采样 / 无 CropForeground / 无 NormalizeIntensity / 禁 z-score**（fork census 明令：z-score 与 rflow-mr-brain v1 训练分布不一致）。物理 spacing 不保持，体数据靠 resize 改网格。
- **无模态堆叠**：四序列（t1n/t1c/t2w/t2f）逐条独立处理、独立编码、独立 modality token（29/34/30/31，skull-stripped 码；cynosure `fixtures.py` 映射已一致）。

编码侧事实（引用备查，**不在本次 scope**）：`SlidingWindowInferer(roi_size=[320,320,160], mode="gaussian", overlap=0.4)` 包 `encode_stage_2_inputs`，AMP autocast；latent 全局标量 scale_factor（=1/std(z)），复用 checkpoint 值、不重算，无 per-channel scale/shift。**cynosure 落点（T12 裁决，#140 复核改判 + #143 交付）**：原裁决「编码改**恒整前向** + 超界显式拒绝」的依据是 MONAI `SlidingWindowInferer` 多分辨率拼合为上采样分割网络设计、对下采样 encoder 把通道维折进空间维（实测 [4,16,16,8] 产出 [1,16,16,8]）；T12 复核探针（#140，MONAI 1.6 z_scale 路径）证伪该依据——z_scale 路径对下采样网络首窗输出自动探测缩放比、逐格产出期望 latent 形状。超界随改判交付**滑窗分支（b 语义，#143）**：`SlidingWindowInferer` 包 encoder 确定性前向，roi [320,320,160] 影像空间逐轴 clamp、overlap 0.4、sw_batch_size 1、gaussian（逐项锚 NVIDIA `create_training_data`），逐窗 (z_mu, z_sigma) 在 latent 网格高斯加权拼合后**单次** seeded 采样——对上游逐窗采样拼接（a 语义，接缝方差收缩）为记录在案偏离（保重跑零漂移幂等 + 接缝带方差均匀，CONTEXT.md「b 语义」词条；改判裁决与探针证据链见 ADR-0010）。豁免判定与上游 `dynamic_infer` 逐字同构：单样本空间体素数 ≤ prod(roi) 恒整前向、行为不变，BraTS [1,1,256,256,128]=8.39M ≤ 16.38M 生产全语料不触发滑窗。

## cynosure 落点

- **transform 链**：`_encode_one` 的裸 `LoadImage` 换成 MONAI `Compose` 六步链（单影像场景可用非 dict 版 transform）。末端保持 `[1,D,H,W]` float32（`LatentEncoder.encode` 契约）；latent 契约 `[4,64,64,32]` 由 resize 步达成。
- **dim 公式参数化**：resize 基数（上游 128）做成可注入参数。fixture config 注入小基数——fixture 影像尺寸不变，但 orientation/强度/dtype 步在 fixture 下全走；链逻辑单份、测试覆盖全链。**fixture 不是对齐对象**，其参数独立于上游。
- **spacing 侧车**：prepare 逐 case 读 NIfTI header `get_zooms()[:3]` × 1e2 存入 manifest（per-case）；rollout 侧源条件的 `spacing_tensor` 接线 manifest（替换现恒定 fixture 值）。BraTS 1mm iso → `[100.0, 100.0, 100.0]`，与现 fixture 值巧合相同，但语义从「写死」变「来自数据」。
- **train/val 划分**：保持 cynosure 现状（排序 + seed 洗牌病例级 70/10/20，`reward/dataset.py`）。上游 fold 字段机制的生成脚本已退休不可考，不复刻伪对齐；两边「病例级 70/10/20」划分原则一致（`experiment-design.md:63` 本取自 fork 事实）。
- **latent 存储域**：manifest 存 **seeded 后验采样 z**（`z_mu + eps(noise_seed)·z_sigma`，上游 `encode_stage_2_inputs` 的确定性重写；种子按（schedule seed, 病例, 序列）内容寻址，重跑零漂移——**逐位**归测试口径，生产 pipeline 的 VAE 前向有浮点噪声级漂移，ADR-0011），**未乘 scale_factor**；checkpoint scale_factor 的域缩放语义归 policy 采样 ticket，判别器侧 fake 由 rollout 相在消费点除回归位。域裁决依据（T12 集群探针）：raw z_mu 全局 std≈0.48 与 policy rollout 终点域 std≈0.94 分布级错配，采样 z std≈1.00 同域——判别器 real/fake 比较要求两侧同为后验采样分布。组2 源条件 latent 同取此域（组2 专用裁决随其 ticket）。

## 边界（不在本方案内）

- **生产 VAE encoder**：**已交付（初版 T12/#27；滑窗分支 #143，ADR-0010 改判）**——`MaisiLatentEncoder`（`reward/encoder.py`，豁免/滑窗两分支 + fp16 autocast，见上「cynosure 落点」）经 `PreparePipeline.build_encoder` 分派装载（`vae_config_json` + `vae_ckpt` 工件对 strict-load）；CLI 非 fixture 拒绝已移除。
- **decode 后逆变换**（合成影像 [256,256,128] → 回原生形状供 nnUNet/FID）→ eval ticket。
- 上游 P1 的 modality label 扰动增强、MR-RATE replay 1:1 混合：属上游训练编排，非数据加载，不适用。

## 验收

1. **方向断言**：任一 BraTS case 加载后 affine 轴码 = RAS（BraTS 原生 ~89% LPS，flip-only 无轴置换，翻转后达成）。
2. **形状契约**：真实 BraTS 影像经链后为 [1,256,256,128]，latent [4,64,64,32] 通过既有契约检查。
3. **fixture 端到端**：`tests/test_prepare.py` 全链（含新 transform 步）通过，工件契约（序列分层、病例级不相交、幂等）不回归。

## MR-RATE 换域：real sample pool + per-channel 统计量（#121/#131，spec #125 实现决策 3）

地图 #67 的 prepare 数据链换域。装配语义按 `experiment.dataset` 分派到策略（`reward.mrrate` 的 `BratsAssembly` / `MrRateAssembly`）——编排骨架（计划 → 失效 → 编码 → 统计量 → 落盘）两域单份，BraTS 线语义零改动。

### 预编码 resize/网格口径（#111 网格裁决的落地）

- **裁决 = 多网格案（逐条件统一网格）**：#111 点名裁决点「单一网格案 vs 多网格案」由 spec #125 实现决策 1/3 定案——11 生成条件各有统一网格（条件词汇表工件 `data/conditions/mrrate_conditions.json` 携带，= #78 普查逐条件众数 latent 网格 ×4），rollout 条件解析形状（#129）、判别器输入按条件分层、里程碑评测与 prepare 编码同一网格口径。同条件内任意原生形状的 real 卷影像域 trilinear resample 到该条件统一网格后编码——**同条件出链 latent 形状唯一**，real/fake 同网格不喂「网格差异」判别捷径；条件间异形状（#78 普查实测逐格 5–11 种原生 latent 网格）在一份工件内按条件登记（`LatentManifest.condition_latent_shapes`）。
- **强度臂 = 官方 clip=False**（#71 裁决，NVIDIA v1 训练口径；两臂 embedding 不可互用）：MR-RATE 线 `preprocessing.intensity_clip=false`、BraTS 线恒 `true`（ADR-0006 fork 锚）——两域取值由 config schema 锁死，显式携带错误值即拒绝。
- **spacing = 条件属性**：等效 spacing = 推荐 FOV / 统一网格（spec #125 决策 6），条目值 = 等效 spacing ×1e2，同条件严格同值——BraTS 线的 per-case zooms 侧车消费（组2 语义）在 MR-RATE 线由条件属性取代，堵死「spacing 差异」判别捷径；BraTS 线侧车机制原样保留。

### 装配流程（`MrRateAssembly.plan()`，四步全确定性）

1. **候选域**：series 级元数据 CSV × 官方 patient 级 splits CSV join（悬挂判定按 splits **全集**——元数据患者不在任何 split = join 完整性破坏、可读拒绝；val/test split 卷是评估留出池、不进 real 数据链候选、计数留痕；生产元数据覆盖全 split 是常态）× 条件词汇表归属解析（`MrConditionVocabulary.resolve_condition`；MRA 全平面单格、SWI 仅 axial 在词汇，域外卷计数留痕不进工件）。
2. **评估集互斥硬守卫**（#131 AC2）：候选域对 #78 评估清单（`artifacts.eval_manifest_csv`）做 series 键（study_uid + series_id）与 patient 集合双粒度零交集校验——任一命中即 fail-fast（官方 split 下 train 与 val/test 天然不相交，守卫防口径漂移静默吃掉互斥性）。守卫读数（键基数、命中数恒 0）随抽样 manifest 落档。
3. **held-out 二分**（train split 内 patient 级，#73 原则/#121 AC3）：候选 patients 排序 + seed 洗牌 + 按 `reward.heldout_fraction` 切出 held-out 侧（同 patient 全部卷同侧 = 病例级不相交）；与评估留出池（官方 val+test）的不相交由 train split 边界 + 互斥守卫共同保证。held-out 池为空显式拒绝（失去 out-of-sample 信号语义）。
4. **逐条件配额抽样**（pool 侧，#78 抽样机制同款）：条件内排序 + seed 洗牌 + 截取 `reward.real_pool_quota` 上限（头部模态各数千条、MRA 全量 ≈ 110 的登记形态；候选不足取全量，配额是上限非硬指标）。同 seed 重跑抽样 manifest 逐字节零漂移。

### 工件契约（分层键泛化）

- **Real sample pool / Held-out real manifest**（`LatentManifest` 泛化，可扩不改名）：条件键两域同名（`PoolEntry.modality`，#129 统一面——BraTS = 序列名、MR-RATE = 生成条件名，判别器条件匹配采样与分层计数的同一归因轴）、卷键 = `<study_uid>/<series_id>`、`modalities` 逐条件计数（由条目派生）+ `condition_latent_shapes` 逐条件 latent 形状契约（多条件域必带、与词汇表逐条件同形，装载期逐条目对账）、spacing = 条件属性值。可被 reward 数据管线既有契约装载（`LatentManifest.load` + `RealPoolSampler.sample(modality=...)` 条件匹配采样——同条件同形，批 stack 前提）。
- **per-channel 统计量**（#121 AC2）：MR-RATE pool 重算（异形状不影响 per-channel 归约），随工件落 `provenance`（数据域、release 快照 `artifacts.mrrate_data_snapshot`、来源 commit `artifacts.source_commit`（运行环境显式声明注入）、强度臂 clip、resize 口径 = uniform-grid、上游锚 = NVIDIA v1 clip=False）。
- **配额抽样留痕**（`SamplingManifest`，`reward.sampling_manifest_json`）：seed / 快照 / 配额 / 逐条件候选与实抽计数 / pool 与 held-out 逐卷归属（patient/study/series/modality/plane/condition/role）/ 互斥守卫读数——prepare 幂等与 held-out 互斥的「落档可查」登记面。
- **容量装配守卫**（ADR-0008-03 口径，#121 AC5）：逐（条件, 全量）容量 ≥ `disc_batch_size_k × world_size`（条件全集 = 词汇表 11 格，稀疏模态小池触发口径——任一条件不足即装配期可读拒绝，开工前失败而非训练中途）。守卫落在 manifest **落盘之前**：失败时盘上 manifest 明确缺失（latents 已写但无索引指向），维持「要么全量一致、要么明确缺失」的工件契约。train 装配期的同款守卫（rank 切片口径）语义不变。

输入工件四件套（MR 线 schema 必填、BraTS 携带即拒）：`mrrate_metadata_csv` / `mrrate_splits_csv` / `eval_manifest_csv` / `mrrate_data_snapshot`；影像落位 = `<dataset_root>/<study_uid>_<series_id>.nii.gz`（官方 zip 内文件名平铺，#132 生产落位同构）。fixture 通道 = `SyntheticMrRateDataset` 夹具（同布局缩小版，`tests/test_mr_prepare.py` 端到端）。
