# 实验设计与验收：三组实验矩阵、基线与评估指标

> 本章是整体实施 spec（地图 #2「CTMR Granular-GRPO RL 后训练方案」）的实验设计与验收章节，由 ticket #8 决议产出。终稿由 ticket #9 汇总。策略侧衔接 `docs/spec/policy-modeling.md`（ticket #5）、reward 侧衔接 `docs/spec/reward-model.md`（ticket #6）、编排侧衔接 `docs/spec/orchestration.md`（ticket #7）。

## 实验基座

三组实验统一落在 **rflow-mr-brain_v1** 基座 + **BraTS2023** 数据上——下游 nnUNet 仪器与跨模态 ControlNet 都在 BraTS 脑肿瘤域，real 库必须与评估目标域一致。

- base UNet = `diff_unet_3d_rflow-mr-brain_v1.pt`（nvidia/NV-Generate-MR-Brain）+ 图像 VAE `autoencoder_v1.pt`（nvidia/NV-Generate-CT）。
- 组2 ControlNet = fork 在 BraTS 上自训的 P3 跨序列 ControlNet（本地工件；rflow-mr-brain 的公开 ControlNet「Coming soon」暂无）。
- **排除 rflow-ct / rflow-mr**：其 ControlNet 是 mask 向或未发布，无跨模态能力。

## 三组实验配置矩阵

| 组 | 训练对象 | 条件 c | 采样场 | ckpt 输入 | 规模 |
|---|---|---|---|---|---|
| 组1 模态标签 | base UNet 全参数 | BraTS 序列 modality token（t1n/t1c/t2w/t2f → 29/34/30/31） | CFG=10 组合场 | rflow-mr-brain_v1 UNet + autoencoder_v1.pt | ~200–500 iter |
| 组2 跨模态 | ControlNet（base 冻结） | 源序列 latent × scale_factor + 源序列 token（ControlNet）/ 目标序列 token（UNet） | 裸条件单前向（CFG=0） | 冻结 base UNet + fork P3 跨序列 ControlNet | ~200–500 iter |
| 组3 序贯 | 先组1 后组2 | 继承各阶段 | 随所训阶段 | 组1 产出 base + 预训练 ControlNet | 组1 + 组2 步数 |

- **组2 跨模态方向** = 脑 MRI 四序列 `t1n/t1c/t2w/t2f` 的 **12 个有序 src→tgt 对**（每序列作 anchor、其余三序列为目标），**非 CT↔MR**（真正的 CT↔MR 合成只在外部仓 `brudfors/maisi-mr-to-ct`，不在本仓库）。组2 条件分布 = 12 对均匀采样。
- **组2 条件 = 双 label（#115）**：同一 (源, 目标) 有序对同时产出两个序列 token——**源序列 token 随 ControlNet 前向**（解读源影像）、**目标序列 token 随 UNet 前向**（生成目标模态）；源序列 latent 与源序列 token 同源于同一源序列条目。分路的权威记述见 `docs/spec/policy-modeling.md`「MDP 形式化」，组别与 CFG 语义见 `CONTEXT.md`「CFG 组合场」。
- **组3 序贯衔接**：第二阶段 base（组1 RL 产出）**冻结**、ControlNet **训练**——从**预训练 ControlNet checkpoint** 复用作初始化，非冻结照搬、非挪用组2 独立产出。若 ControlNet 对组1 新 base 不匹配（残差是对旧 base 的加性修正），「对组1 产出 base 再微调 ControlNet」作消融、非默认步骤。
- **组3 预训练产物衔接（操作面，#116）**：跑序贯需**两份预训练产物**，config 分别指定——stage-1（组1 配置）消费 modal-label 预训练报告（`reward.pretrain_report_json`），stage-2（组2 配置）消费 cross-modal 预训练报告（`experiment.stage2_pretrain_report_json`，序贯必填、非序贯组携带即被 schema 拒绝）；**cross-modal 预训练 run 需真实执行一次**（`pretrain` 子命令按 config 组别走同一条 driver 路径，代码现成）。stage-2 指向异组报告被装载守卫拒绝（#113 组别等值守卫，报错指引绑定配置面）——序贯不静默继承 stage-1 报告。
- **训练规模**：每组默认 **~200–500 iteration**（先跑 50 iter sanity 再扩），rollout = 条件组 × G=12 方向；三组共用同一量级保证横向可比。实际吞吐以 `orchestration.md` 的 rollout profile 为准。
- **判别器**：每组 RL 各自独立的在线判别器与回放缓冲（判别器在线跟踪当前 policy 的 fake 分布，跨组/跨阶段不复用）。

## 条件词表口径（MR-RATE 换域线，#119 → #127 工件化）

地图 #67 的 MR-RATE 上游域 RL 后训练基线与本页 BraTS 三组矩阵**共用同一基座**（rflow-mr-brain_v1），条件口径两套并存、经 `experiment.dataset` **互斥激活**（schema 层二选一：词汇工件绑定与登记域校验在装载期字段级拒绝）：

| 口径 | 数据集字段 | 模态词表 | token 映射 | 条件单位 |
|---|---|---|---|---|
| BraTS（本页矩阵，行为不变） | `BraTS2023`（默认） | 四序列 t1n/t1c/t2w/t2f | 29/34/30/31（工件装载，`artifacts.modality_mapping_json`） | 组1 = 序列；组2 = 12 有序对 |
| MR-RATE（地图 #67） | `MR-RATE` | 五模态 t1w/t2w/flair/swi/mra | whole-brain 9/10/11/20/16（工件装载，`artifacts.condition_vocabulary_json`；skull-stripped 29–33 不进本轮生成词汇） | 11 生成条件（#81 白名单，五元组工件） |

MR-RATE 词表四要素（#119 定过口径）自 **#127 起整体工件化**——唯一来源是仓库登记工件 `data/conditions/mrrate_conditions.json`，config 不内嵌词表、代码内无常量副本：

- **token 映射**：whole-brain `t1w/t2w/flair/swi/mra → 9/10/11/20/16`，**上游权威**（NV-Generate-CTMR `configs/modality_mapping.json`，`research/mrrate-data-spec.md` §3.4）；条件 token 由映射按模态派生（文件内不重复登记，单一来源）；skull-stripped 29–33 不进本轮生成词汇（prepare 数据链双产条目的消费口径由数据装配票承接）；
- **生成条件五元组**：11 个 (modality token, plane, 推荐 FOV, 统一网格, 等效 spacing) 格 = #81 终审白名单全量（T1w/T2w/FLAIR 各三平面 + SWI/AXIAL + MRA/ALL-PLANES）。统一网格 = #78 普查工件逐条件众数 latent 网格 ×4（RAS 轴序，薄轴按 FOV 归位）；等效 spacing = 推荐 FOV / 统一网格（条件属性，spec #125 决策 6）。FOV 数值：10 格取官方 docs/inference.md 推荐 FOV 表，MRA（无 all-planes 官方行）取 #78 评估 manifest 中位（#80 实测口径）；工件逐条件带 `fov_source` 标注；
- **装载面**：`cynosure.conditions.MrConditionVocabulary.load(path)`（`artifacts.condition_vocabulary_json` 提供路径）。装载期校验：字段缺失/多余即 pydantic 字段级拒绝；token 映射须恰好覆盖五模态（缺模态可读拒绝）；统一网格逐轴 32 倍数（UNet 跳连约束）；FOV 薄轴与网格薄轴同位；等效 spacing ∈ [0.4, 5.0] mm（上游 check_input_mr 域）；生产模式（`fixture_mode=False`）恒要求 11 条件全量且**逐条件网格与 #78 普查期望网格（众数）一致**——缺格、多格、网格不符即拒绝，「条件 → latent 形状」以普查工件为权威对照；
- **fixture 通道**：非 11 条件全量的小词汇表只能经 `fixture_mode=True` 显式装载（同 `resize_base` 显式声明纪律）；fixture 工件由 `Fixture.write_artifacts` 落盘（2 条件**异形状**小网格——t1w/axial [64,64,32]、flair/axial [32,32,64]，#129 全链验收的输入面）。

schema 语义（与 BraTS 线的隔离保证）：

- **绑定互斥**：`dataset=MR-RATE` 时 `artifacts.condition_vocabulary_json` 必填（缺词表绑定 = 生成条件取数域无从装配）；BraTS config 携带即拒绝（拼错 dataset 时两套口径静默共存比显式拒绝危险）；
- **dataset 登记域**：`experiment.dataset` 从 Literal 定死放宽为 str + 登记域 validator（`REGISTERED_DATASETS`，#127 泛化）——未登记域字段级拒绝；新增数据域扩登记处并接线其词汇来源，不再动类型层；BraTS 取值与全部既有语义零变化（回归测试守住「泛化不改语义」）；
- **无共享可变状态**：词表装载产物不可变（frozen 值对象 + 只读映射视图），同进程先后装载互不污染；
- **MR-RATE 线只定义组1**：上游无 MR ControlNet，非 modal-label 组别即拒绝（跨模态/序贯是 BraTS 语义）。

**运行时消费边界（#127 装载层 → #129 贯通层）**：#127 交付到工件装载与 schema 绑定；**#129 接线 rollout 条件组装与 latent 形状按条件贯通**——`cynosure.conditions.ConditionVocabulary` 协议是两域统一解析面（`names` / `latent_shape` / `latent_numel` / `token` / `spacing_condition`——末者与 #130 的 `MrConditionVocabulary.spacing_condition` 同名同语义：合并期统一命名，换算因子取 config 单一来源 `SPACING_CONDITION_SCALE`），MR 侧 = 词汇表工件装载产物、BraTS 侧 = 单域常量策略（四序列、任意条件恒 config `latent_shape`——单域 = 单条件词汇特例）。落地口径：

- **rollout 条件组装**：MR 组1 条件分布 = 词汇表生成条件均匀轮转（`MrConditionSampler`，spec #125 决策 5 默认口径），token 与等效 spacing ×1e2 都是条件五元组属性、经协议取数（单一来源）；BraTS 组1 = 四序列采样（`ModalLabelConditionSampler`，行为不变）。条件名随 `RolloutCondition.name` 贯通采样场。
- **latent 形状按条件贯通**：rollout 初始噪声/扰动噪声、base 分区量产、eval/baseline 逐条目采样的形状一律从批次条件经协议解析（批内同条件即同形状；GRPO 组内天然同条件同形状，advantage 无跨形状问题）。base 分区量产与 buffer `fill_base` 收逐条目张量清单（跨条件异形状无从 cat）；回放采样与判别器输入的同条件过滤使批内同形成为结构事实。
- **sigma 日程逐条件锚（ADR-0002 语义逐条件化）**：锚 = 该条件空间 numel（`latent_numel(name)`），与形状同源派生（词汇表单一入口，结构性防日程静默错位）；`ConditionSchedules` 按条件名选日程（MR = 逐条件惰性 `TrajectoryCursor`，numel 碰撞的条件日程天然等价；BraTS = 单条件日程表，数值零漂移）。MR 线 config **显式携带单域锚字段即字段级拒绝**（`latent_shape` / `policy.input_img_size_numel`——默认值在 MR 线无消费，显式声明即「以为全局锚仍生效」的意图表达）；MR 序列化产物自动排除两字段（JSON 往返不被误拒）。
- **评测批组织**：里程碑评测按目标条件分组（组内同形 → 分组解码/分组参照 stack），读数逐条件产出（`fid_target_*` 键 = 条件名）；Baseline/重采落盘逐条目（分组解码、条目体 clone 独立存储）。条件词汇表工件与 `stage_condition_vocabulary` 的 MR 语义在 schema 层显式拒绝（BraTS 口径方法不动），MR 的条件集与里程碑样本面守卫在装配期消费词表。
- **续训持久化（分片 v7）**：replay buffer 两区 latents 从单一堆叠张量改为逐条目张量清单（异形状条目可持久化），恢复对账逐条目经词汇表按条件校验形状；旧 v6 分片被版本对账显式拒绝。real pool / held-out manifest 的 MR 多条件形态携带逐条件形状契约（`condition_latent_shapes`，装载期逐条目对账）；BraTS 单域工件不带（全局形状对账照旧）。
- **real pool 切片轴与预训练报告条件域**：rank 条带切片的分层轴 = 活动条件集（`RankSlicedPool` 经装配注入条件名，不设四序列代码内副本——换域线多 rank 运行的结构前提）；real pool / held-out manifest 的逐条件形状契约在装配期与词汇表逐条件对照（同名异形 = 词表工件改动而 manifest 未重建，装配期显式拒绝而非等到判别器拼接 real 与 fake 时才炸）。预训练报告的条件域随之泛化：`condition_auc` / `gate_whitelist` 键 = 本域条件名（BraTS 线仍是四序列、行为不变；MR-RATE 条件名此前被四序列字面量域挡在 schema 外，MR 预训练在收尾处 ValidationError、报告落不了盘）；多条件线报告不记单域全局 `latent_shape`（形状逐条件派生自词表工件，口径由 provenance 的 `condition_vocabulary_sha256` 承载），装载守卫另加条件集对照与词表内容指纹（工件漂移而 real 侧工件未变时，报告的白名单与实测值对另一份 fake 分布负责）。**域边界同批收口**：多条件域 real 工件必须携带逐条件形状契约（缺表即装配期拒绝——缺表会让 `load_latent` 静默回退全局 `latent_shape` 对账，异形条件在 real 采样/gate 重算期才炸、同形条件带着错误的全局口径入训；判据 = `ConditionVocabulary.single_condition`）；尚未交付的 MR 面在构造/装配期显式拒绝，不把能力边界藏成 BraTS 布局扫描或单域默认锚的错误——prepare 管线（`PreparePipeline`，BraTS 病例布局 + 四序列预编码）、里程碑参照影像库（`ManifestEvaluation.build`，`RealVolumeStore` 是 BraTS 布局参照库）、轨迹诊断（`TrajectoryDiagnosticRunner`，单条件日程 + 全局形状噪声 + 四序列条件）三处。
- **仍由后续票接线**：预处理统一网格与 spacing 条件属性的消费面（#130）、prepare 数据链配额装配（#131）。BraTS 线的全部既有消费点（`MODALITIES`、`stage_condition_vocabulary` 等）不接管、不修改。

**消费接线进度（#121/#131 交付，2026-09-16）**：预处理统一网格与强度臂参数化、prepare 数据链装配（官方 split join / 评估集互斥守卫 / patient 级 held-out 二分 / 逐条件配额抽样 / 条件分层工件 / 容量守卫）已落地——口径见 `data-preparation.md`「MR-RATE 换域」节。`latent_shape(name)` 的 rollout/评测侧消费（#129）仍待接线。

## 对照基线（no-RL）

- **组1 基线** = 冻结 base UNet @ CFG=10；**组2 基线** = 冻结 base UNet + 冻结预训练 ControlNet。
- 基线样本量 **N = 200–500**（与评估集同规模），用**同一组初始噪声 + 同一条件集**采样；冻结模型**只采一次**。
- RL 后 policy 用**相同 seed 与条件**重采，使差异唯一归因于 RL 训练。

## 评估指标组合

**主判据 = 复用基座的验收阶梯**（相对各自 no-RL 基线统计显著更优），三块：

1. **定量评测（quantitative evaluation）**：像素域 **2.5D FID**（XY/YZ/ZX 三正交面、RadImageNet-ResNet50）+ KID/bootstrap CI；跨模态组另加 3D SSIM + MAE。**直接复用基座 `fid_2d5` 解码 FID 基础设施，不做 latent FID**——latent 域无现成 3D 特征提取器（FID 需特征网），硬造一个引入新训练开销与不确定性。
2. **下游指标分布对齐（downstream distribution alignment）**：nnUNet 仪器（nnunetv2 2.8.1，BraTS2023SSA fold_0，250 epoch）产出 WT/TC/ET 体积、质心、ET/WT，对合成影像与真实分布做 **TOST/KS/EMD 对齐**。**不比 Dice**——合成影像无 GT，Dice 只能用于 P2 回切一致性，不构成生成验收门槛。
3. **专家目检（expert review）**：复用基座盲审流水线（**5×5=200 条**，视觉图灵 balanced accuracy CI⊂[0.40,0.60] + 4 维 5 分 Likert≥4.0 + Fleiss' κ）；双盲、影像背景同事、非作者。

**在线健康监控**（不进验收门槛）= reward 侧信号：held-out 判别器 AUC 近 chance / 组内 reward std / anchor eval reward 背离（`reward-model.md` 防 reward hacking 信号集）。各项指标按目标序列分层出。

## real 样本库

- real = **BraTS train split（病例级 70%）全量 VAE 预编码 latent**，按序列 token 分层（GLI 全量约 1251 例 → train 约 800+，确切体量以基座 data/README 为准）；**不混入 MR-RATE 全库**（避免域漂移 + 库过大）。
- 组1 按序列 token 分层；组2 按 4 序列分层。
- **MR-RATE 换域线（地图 #67）的 real 样本库 = 独立口径**：官方 train split 逐条件配额抽样 + held-out train 内 patient 级二分 + 评估集互斥硬守卫，分层键 = 生成条件（11 格）——口径见 `data-preparation.md`「MR-RATE 换域」节（#121/#131）。

## 成功判据与早停准则

spec **只钉判据形态 + 触发规则，不钉绝对数值**：

- **成功** = 主判据（定量评测 + 下游分布对齐 + 专家目检）相对各自 no-RL 基线**统计显著提升**（配对检验）。
- **早停** = 主判据连续 **N 个里程碑** plateau + 判据不再升 + 出现 hacking 签名（held-out AUC 近 chance 且 eval reward 仍升）即停。N 为运行时参数（默认 3）。
- 绝对阈值（如「FID 必须 < X」）是训练期经验数据才能定的，写进 spec 是假的——标为**运行时 knobs + 默认初值**，属执行，超出本地图 spec 范围。

## 待定 / 移交

- 绝对阈值数值、N（plateau 里程碑数）等 → 训练期经验数据（执行，超出本地图 spec 范围）。
- 消融矩阵执行（`reward-model.md` 判别器尺度/聚合、`policy-modeling.md` Λ/M 扫描）→ 施工。
- 组3 ControlNet 不匹配时的「再微调 ControlNet」消融 → 仅在触发时做。

## 依据

- 基座 fork NV-Generate-CTMR 评测与数据事实：`fid_2d5.py`（RadImageNet-ResNet50 2.5D FID）、`nnunet_runner.py`（nnunetv2 2.8.1 BraTS2023SSA fold_0）、`cross_modal/plan.py`（四序列 12 有序对）、BraTS2023 病例级 70/10/20。
- 衔接 `docs/spec/policy-modeling.md`（组1 CFG=10 / 组2 CFG=0、G=12、单步 SDE）、`docs/spec/reward-model.md`（hacking 信号集、online update）、`docs/spec/orchestration.md`（rollout 吞吐绑定、单节点优先）。
- 地图 #2 Destination / Notes。
