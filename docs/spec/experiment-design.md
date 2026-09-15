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

## 条件词表口径（MR-RATE 换域线，#119）

地图 #67 的 MR-RATE 上游域 RL 后训练基线与本页 BraTS 三组矩阵**共用同一基座**（rflow-mr-brain_v1），条件口径两套并存、经 `experiment.dataset` **互斥激活**（schema 层二选一，改动词表值即字段级拒绝）：

| 口径 | 数据集字段 | 模态词表 | token 映射 | 条件单位 |
|---|---|---|---|---|
| BraTS（本页矩阵，行为不变） | `BraTS2023`（默认） | 四序列 t1n/t1c/t2w/t2f | 29/34/30/31（工件装载，`artifacts.modality_mapping_json`） | 组1 = 序列；组2 = 12 有序对 |
| MR-RATE（地图 #67） | `MR-RATE` | 五模态 t1w/t2w/flair/swi/mra | whole-brain 9/10/11/20/16 + skull-stripped 29–33 | 11 生成条件（#81 白名单） |

MR-RATE 词表四要素全部入 config schema（`experiment.conditioning` 段，`src/cynosure/config.py` 的 `MrRateConditioning`）：

- **模态集**：t1w/t2w/flair/swi/mra 五模态（#81 swap 探针五 token 全 responsive 的证明面）；
- **token 映射**：whole-brain `t1w/t2w/flair/swi/mra → 9/10/11/20/16`，**上游权威**（NV-Generate-CTMR `configs/modality_mapping.json`，`research/mrrate-data-spec.md` §3.4）；
- **序列词表**：每序列**双条目**——whole-brain（官方直发）+ skull-stripped（29–33，现场 derive：官方只发 `img/` + `seg/`）；skull-stripped 条目属 prepare 数据链的双产形态；
- **生成条件分组**：11 个 (模态, 平面) 格 = #81 终审白名单全量（9 读数格欠训、0 饱和、0 标签存疑）：T1w/T2w/FLAIR 各三平面 + SWI/AXIAL（仅轴位可得）+ MRA/ALL-PLANES（全平面一格；T2w 读数三格并池但条件独立成格）。分组 token 恒为 whole-brain 条目（#81 swap 生成口径，skull-stripped 码不进生成分组）。

schema 语义（与 BraTS 线的隔离保证）：

- **缺省自动填充**：MR config 不必抄录词表，`conditioning` 缺席时 schema 以定死默认值填充（单一来源）；
- **互斥携带即拒**：BraTS config 携带 `conditioning` 段即拒绝（拼错 dataset 时两套口径静默共存比显式拒绝危险）；MR-RATE 词表字段改值（含分组集合偏离白名单）即字段级拒绝；
- **无共享可变状态**：词表容器经 `default_factory` 每实例独立构造，同进程先后加载互不污染；
- **MR-RATE 线只定义组1**：上游无 MR ControlNet，非 modal-label 组别即拒绝（跨模态/序贯是 BraTS 语义）。

**装载层交付边界（#119）**：本段口径只到 schema 校验。词表的运行时消费——RolloutCondition 组装（组条件 label 取数）、prepare 数据链（双条目序列的装载与编码）、预训练 per-condition 分组（ADR-0008 的按条件统计键）——由地图 #67 的后续施工票接线；BraTS 线的全部既有消费点（`MODALITIES`、`stage_condition_vocabulary` 等）不接管、不修改。

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
