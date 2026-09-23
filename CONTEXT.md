# cynosure

cynosure 为 MAISI 3D latent rectified-flow 医学影像 checkpoint 设计并实施基于 Granular-GRPO 的 RL 后训练。**零依赖原则**：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；网络类来自 MONAI 库本身。

## Language

### 生成与表征

**Latent（潜变量）**:
VAE（AutoencoderKlMaisi）把影像体压缩成的 4 通道低维表征。一切 RL 采样与打分都在 latent 域进行，不回像素域。
_Avoid_: 特征图、embedding、编码

**Latent 域载体（DomainLatent）**:
latent 张量 + 域标签的薄包装（ADR-0015，实施票未启动）：域枚举两值——存储域 / policy 工作域；「干净域 vs 带噪域」不在本枚举（第二票，见 干净域语义 = ADR-0012 判别器输入恒干净域）。域间换算只存在于载体语义级方法（`to_working` / `to_storage` 命名方向），载体方法是唯一被允许触碰 `latent_scale_factor` 的位置；**无裸函数约束**（维护者强约束）：latent 域操作一律挂载体类方法下，模块级裸函数不得触碰域换算（与域无关的统计裸函数不受约束）。错域 = 构造期拒绝，取代散布 6 处 / 13 文件的手写乘除纪律；一次性切换、不留双口径，数值零变化由既有 roundtrip / 逐位一致测试把守。
_Avoid_: 裸张量手写 ×÷ scale factor（本词条落地后即违例）、把干净域混入本枚举、Tensor 子类（autocast 输出丢失子类类型，ADR-0015 已排除）

**Policy（策略）**:
被 RL 训练的扩散模型——模态标签阶段是 base UNet，跨模态影像阶段是 ControlNet。
_Avoid_: 模型、网络、生成器

**Base model（基座）**:
RL 起始的冻结预训练 checkpoint；RL 不改动它本身，只从它出发微调出 policy。
_Avoid_: 参考模型（reference model，是另一个概念，本项目起步不使用）

**Rollout（滚动采样）**:
当前 policy 从纯噪声到 latent 的一条完整去噪轨迹。
_Avoid_: 采样、生成

**Capability stage（能力阶段）**:
NV-Generate-CTMR 的分阶段能力。本轮实验只用「模态标签条件生成」与「跨模态影像条件生成」两阶段；「掩码条件生成」不在本轮范围。
_Avoid_: 任务、phase、step

### 数据对齐

**上游（Upstream）**:
NV-Generate-CTMR fork——基座 checkpoint 的训练方项目。cynosure 对齐的是它的训练数据 recipe 与推理实际行为，不是代码：上游代码只读参照、永不 import（零依赖原则）。
_Avoid_: 基座（base model 指 checkpoint 本体）、参考实现

**RAS 方向**:
全链影像方向约定：进入 VAE 的体数据一律先重定向到 RAS 轴码；BraTS 原生 LPS 靠轴翻转达成，无轴置换。
_Avoid_: canonical 方向、ToCanonical（那是实现名）

**上游 `dynamic_infer`（参照函数）**:
NV-Generate-CTMR `utils.py` 的体积分派函数：单样本体素数 ≤ roi 元素数走整前向，否则 roi 逐轴 clamp 到图像尺寸后走滑窗。只读参照、永不 import（零依赖原则）；本仓同语义实现 = decode 侧 `LatentDecoder`（含滑窗分支）、encode 侧 `MaisiLatentEncoder`（豁免/滑窗两分支，滑窗走「b 语义」偏离）。
_Avoid_: Dynamic_Inferer（全库无此类，上游是函数不是 Inferer 类）

**Decode（解码）**:
latent → 像素域的 VAE 解码，只发生在评测路径（Baseline 采样、里程碑评测、RL 后重采），不进逐 iteration 训练循环。滑窗口径与上游 `dynamic_infer` 同语义（单通道空间体素数 ≤ prod(roi) 走整前向豁免，否则 `SlidingWindowInferer` 高斯滑窗，latent 空间 roi=[48,48,48]、overlap 2/3，锚 NVIDIA `config_infer.json`；豁免口径修正见 ADR-0010），前向冻结 fp16 autocast 口径。
_Avoid_: Dynamic_Inferer、逐 iteration 解码

**预编码（Encode）**:
影像体 → latent 的 VAE 编码，发生在 prepare 阶段（`PreparePipeline._encode_one` 是全仓唯一读原始 NIfTI 的位置）；产物 = seeded 后验采样 z（上游 `encode_stage_2_inputs` 的确定性重写；幂等重跑 = 语义层，生产 pipeline 的 VAE 前向有浮点噪声级漂移，逐位归测试口径——ADR-0011），存储域不乘 scale_factor。豁免判定 = 单样本体素数 ≤ prod(roi)（影像单通道，与上游逐字同构，BraTS 全语料恒整前向）；超界走滑窗分支（b 语义，NVIDIA 语义锚 roi=[320,320,160]、overlap 0.4；ADR-0010 改判、#143 交付）。
_Avoid_: 编码器推理、「超界显式拒绝」作现状引用

**b 语义（blend-then-sample）**:
encode 滑窗的采样编排：MONAI `SlidingWindowInferer` 包 encoder 确定性前向，逐窗 (z_mu, z_sigma) 在 latent 网格高斯加权拼合，拼合**后**以单一内容寻址种子采样一次 eps——重跑零漂移幂等保持、接缝带方差与体心均匀。对 NVIDIA 的逐窗采样拼接（a 语义，接缝方差收缩）为记录在案偏离（ADR-0010；#139/#143）。
_Avoid_: 逐窗采样拼接当本仓语义、a/b 语义混称

**上游锚（Upstream anchor）**:
对齐的双重锚。recipe 级分线：BraTS 线锚 fork（ADR-0006，`clip=True` 为记录在案故意偏差），MR-RATE 线锚 NVIDIA 真上游 v1（`clip=False`）。机制级（滑窗、采样、scale factor）fork 与 NVIDIA 逐字节相同（`da438fe` 对拍 `utils.py`/`create_training_data.py`/`utils_infer.py` 零差异），两锚无分歧。说「与上游对齐」必须指明哪一锚。
_Avoid_: 上游（不指明锚的泛指）

### 配置与域

**域资格（Domain qualification）**:
config schema、装配分派与测试构造对「数据域」的单一事实源（ADR-0013，实施票未启动）：dataset → 域内字段集 / 工件集 / 错误文案 / 按域取行为的查询入口，登记在 `cynosure.domains`（待建）。schema validator 退化为查表接线（错误文案跟字段清单走同一知识单点），运行时分派（词汇表装配、schedule 装配、policy 构建、prepare 装配、eval 取数、baseline manifest 条件解析）经查询入口取本域行为——调用方不再写 `experiment.dataset == "MR-RATE"` 字面分支；新域 = 加条目不改消费面。schema 结构不动（注册表收拢、非 discriminated union）；MR-only 字段（`reward.real_pool_quota` / `heldout_fraction` / `heldout_quota_volumes` / `reward.sampling_manifest_json`）的「域内才合法」语义由查表守卫承载，取代散布 14 处 / 8 文件的字面分支与三份同步（validator / 分派 / fixtures）。
_Avoid_: `experiment.dataset == "MR-RATE"` 字面判断（本词条落地后即违例）、域知识多处各写各的、为第三域预先设计（新域只是加条目的顺带能力）

### 策略建模

**Velocity（速度场）**:
UNet 的输出语义：`v = x0 − noise`（v-prediction），决定去噪方向。
_Avoid_: 噪声预测、epsilon

**CFG 组合场**:
条件与无条件 velocity 的线性组合 `v_uncond + w·(v_cond − v_uncond)`，w 为引导强度。policy 的有效采样场按组对齐基座推理的**实际行为**：组1 = CFG=10 组合场（无条件分支 = 全零 label）；组2 = **裸条件单前向**（基座代码强制 `cfg==0`）——「CFG=0」标记的是**组合机制缺席**，不是把 w 代入 0（代入 0 仍要算无条件分支，组2 根本不组织它）。**无条件分支缺席 ≠ 条件注入缺席**：组2 的模态标签注入通道全程在场，且两路模态标签**各收其职**（#115）——ControlNet 收**源**模态标签（解读源影像的模态先验）、UNet 收**目标**模态标签（生成目标模态的模态先验）。
_Avoid_: guidance（单独使用时）、CFG=0 当成 w=0（组2 是组合机制缺席，不是把 w 代入 0）、label 同源（ControlNet 与 UNet 收同一标签的旧语义，已被各收其职取代）

**Anchor 轨迹**:
从同一初始噪声确定性 ODE 采出、逐步存下 latent 的参考轨迹；组内全部方向共享它，使 reward 差异唯一归因于被优化那一步的扰动。
_Avoid_: 参考轨迹、主轨迹

**单步 SDE 扰动（Singular Stochastic Sampling）**:
把随机性限制在单个被优化步——仅在该步把确定性 ODE 步替换为带噪高斯核，其余步保持确定性。
_Avoid_: 全程加噪、SDE 采样

**MGAI（Multi-Granularity Advantage Integration）**:
多粒度 advantage 集成：每个粒度 λ 的 advantage 各自组内标准化后直接求和的融合方式。

**前向激活预算（Forward Activation Budget）**:
rollout 续跑里单次 policy 前向允许吃掉的激活显存上限（config `policy.forward_activation_budget_gib`）：超预算的 G 方向整批前向按 latent 体素切子批**顺序**积分，峰值以子批为界（#123 首跑 OOM 修复——探针实测激活 ≈3.2 KiB / latent 体素 / 前向样本，而 UNet 权重驻留仅 0.687 GiB）。缺省按设备**总**显存的比例自动探测（同设备可复现；共享实例不反映他进程占用，须显式钉值），显式值在设备总显存可探测时（CUDA）超过即装配期拒绝。**G 不是显存旋钮**：分块改的是「同时算几个样本」，不改任何样本的数值路径（调度等价，实测 rel ≈4e-7）。**分布式下分块上限必须 rank 一致**：FSDP 逐前向参数 all-gather 的调用序列与「前向调用次数」绑定，而条件逐 rank 独立采样（seed 的 rank 派生是设计意图）使各 rank 的体素数/本地上限不同——本地各自取值即前向次数分叉、集合序列错配、全体互等挂死（#165 review P1，sugon train2 首跑实录：iter 0 rollout 后全 rank 阻塞直至 watchdog abort）。故本地预算先经 `all_reduce_min` 取全 rank 最小再截断（子批只小不大，任何一侧的显存上界语义都不破）。
_Avoid_: 用降 G 换显存（改变算法口径的最后手段）、把预算当空闲显存配额、按空闲显存探测（同设备不可复现）、分布式下各 rank 本地各自取分块上限（前向次数分叉 = 集合互等挂死）

### Reward model

**Reward model（奖励模型）**:
给 rollout 的 latent 打标量分的在线判别器，该分即 RL 的 reward。

**PatchDiscriminator（PatchGAN 判别器）**:
MONAI 的 PatchGAN 判别器（Pix2PixHD 式），输出 patch logit 图而非单一标量；本项目用它当 reward model。

**判别器深度（Discriminator depth, num_layers_d）**:
单臂 PatchDiscriminator 内部 stride-2 下采样卷积的层数——决定单尺度感受野与 patch 粒度（2 层 → latent 每维 /4，输出 16×16×8 patch logit 图）。消融轴 {1, 2}（reward-model 章臂 A）。
_Avoid_: 判别器尺度（另一条轴）、网络总层数（另含 initial/final conv，非同一口径）

**多尺度判别器（Multi-scale discriminator, num_d）**:
并联的 PatchDiscriminator 臂数（num_d=1 即单尺度）：各臂吃同一 latent 输入、靠逐臂加深的下采样链取得由细到粗的感受野阶梯（MONAI 乘法阶梯——第 i 臂层数 = num_layers_d×(i+1)），各臂 patch logit 图先各自 mean 成标量、再跨臂相加成 reward（reward-model 章裁决）。消融轴 {1, 2, 3}（臂 B）。
_Avoid_: 粒度（λ 时间步间隔，另一概念）、image pyramid（各臂吃原始分辨率输入，非预先下采样）、判别器深度（臂内层数，另一条轴）

**Real sample（真实样本）**:
训练集影像经 VAE 预编码的 latent，作为判别器的「真」，固定不更新。

**Real sample pool（真实样本库）**:
训练集影像经 VAE 预编码的 latent 集合，判别器的「真」与评测参照都取自它——同源重构（ADR-0012）后也是判别器「假」的源材料。分层键按域：BraTS 线 = train split（病例级 70%）全量、按序列 token 分层；MR-RATE 线 = 官方 train split 逐条件配额抽样（配额为上限、固定 seed、排序后抽样、幂等留痕）、按生成条件分层（11 格，每条件统一网格——同条件 latent 同形，异条件异形状随 `condition_latent_shapes` 契约登记）。
_Avoid_: 真样本集

**Held-out real（留出真样本）**:
held-out 监控侧的预编码 latent 工件，与 Real sample pool 病例级不相交、永久不参与判别器更新——保证 held-out AUC 是 out-of-sample 的 hacking 监控信号（reward-model 章，`prepare` 产出）。来源按域：BraTS 线 = val split（病例级 70/10/20 之 10%）；MR-RATE 线 = train split 内 patient 级二分（`heldout_fraction` 配比）再按条件封顶（`heldout_quota_volumes`，监控集不是越大越好——预训练按条件读全量卷级聚类，无上限即按 10% patient 的体量整条件上卡）——与官方 val/test 评估留出池的不相交由官方 split 边界 + 装配期评估集互斥守卫共同保证（#73 原则）。
_Avoid_: 验证集（val split 是划分段，held-out real 是其预编码工件）

**配额抽样 manifest（Sampling manifest）**:
MR-RATE prepare 的抽样留痕工件（`reward.sampling_manifest_json`，#131/#78 机制同款）：seed、数据 release 快照、逐条件配额与实抽计数、pool 与 held-out 逐卷归属（patient/study/series/modality/plane/condition/role）、评估集互斥守卫读数（series 键 + patient 级双守卫，合法装配恒 0）——prepare 幂等（同 seed 重跑零漂移）与 held-out 互斥「落档可查」的可审计登记面。
_Avoid_: 抽样清单（泛指）、下载清单（#78 的另一工件）

**Fake sample（伪样本）**:
Real sample pool 真实 latent 的同源重构体（构造见 同源重构），作为判别器的「假」——与 real 同内容、仅生成伪影不同。
_Avoid_: 生成样本、负样本、rollout 产物当判别器 fake（旧语义，ADR-0012 取代）

**同源重构（Reconstruction fake）**:
判别器 fake 的构造原语（ADR-0012）：真实 latent 乘回 policy 工作域后从日程步点 s 加噪（rectified flow 插值 x_s = (1−s)·x + s·noise），再用**当时的 policy**（预训练 = base 冻结权重、在线 = 当前 policy）以 η=0 确定性 ODE 去噪回 σ=0、归位 pool 存储域；s 逐样本均匀抽自该条件被优化步（`policy.train_step_indices_m`）的日程点，静态逐条件从各自 sigma 日程导出（与 rollout 同一 `ConditionSchedules`）。判别器输入恒干净域——加噪只发生在 fake 构造的输入端，不进判别器前向；real/fake 同内容配对使「记 Real sample pool 病例共性」捷径结构性失效。构造走专属命名随机流（先抽水平 s、后抽 ε，随续训分片落盘）。
_Avoid_: 重加噪/加噪判别（判别器输入带噪的旧注入语义，已废）、partial diffusion、把重构体当数据增强（它是 fake 的定义本身）

**Warm-start pre-training（判别器预训练）**:
RL 启动前对 reward model 的离线密集训练：real 取 Real sample pool，fake 取同批 real 的 base policy 同源重构；训练至通过 RM readiness gate，产物作为在线更新的初始权重（ADR-0007；fake 构造 ADR-0012）。
_Avoid_: 一次性预训练、离线 reward model（RLHF 语境指冻结，本项目预训练后仍在线更新）、量产 rollout 当 fake 源（旧语义，ADR-0012）

**RM readiness gate（RM 上岗门槛）**:
RL 启动的硬前置：判别器预训练后按条件报告 recon-AUC（held-out real 原始 vs 同源重构体——测判别器在训练任务上的 out-of-sample 泛化力，ADR-0012），过线条件构成条件白名单；白名单为空拒绝开跑，非空即放行。口径交接：预训练判据（recon-AUC）与在线运行口径（fake 侧 = rollout 终点 latent）不同构、不可跨阶段比较绝对值——预训练放行、在线实测重新洗牌名单是预期行为（准入体检 vs 在岗考核，ADR-0012）。条件闸总开关（`reward.condition_gate_enabled`，默认开）关闭时本门槛整体不适用（白名单既不作上岗判据也不作更新开关，见 条件闸）；**warm-start 权重装载不属门控链、不受开关影响**。上岗口径含组别绑定（#113）：预训练报告的 group 与消费 config 的组别严格等值——warm-start 装载守卫（assert_data_provenance）对照，跨组消费显式拒绝、无逃生门（组间 fake 分布不同，per-condition 判定只在本组分布上测量；组3 序贯 stage-2 的合法消费路径 = `experiment.stage2_pretrain_report_json` 绑定的 cross-modal 报告，stage 级绑定、不继承 stage-1 报告——#116）。
_Avoid_: 软警告、早停（早停是训练期机制，门槛是启动期机制）、池化达标（全池单一标量口径，已被按条件取代）、跨组上岗（报告组别 ≠ 消费组别的装载——守卫期即拒绝，非 gate 白名单判定对象）

**条件闸（Condition gate）**:
held-out AUC 驱动更新决定的整条链的总开关（ADR-0008 决策 5/7/8 的统一关闭形态，维护者裁决 2026-09-17）：`reward.condition_gate_enabled=false` 时上岗判定不拒绝开跑、运行时白名单退化为全条件放行、逐 iteration 门控与 EMA 动态恢复停步——policy 每 iteration 对目标条件全量更新。**关的是「AUC 驱动决定」，不是「AUC 被测量」**：AUC 照常逐 iteration 测量并落 `iter` 事件、分叉监控（ADR-0009）照常——观测面正是本开关的裁决输入。诊断/执行期口径，不改变条件匹配采样、容量守卫、支撑度规则任何语义。
_Avoid_: 关监控（观测面不退化）、关判别器训练（判别器照常受训与更新）、白名单降级（`gating_dynamic_recovery=false` 是静态白名单降级，本开关幅度更大）

**支撑度规则（Support rule）**:
门槛判定的统计形态（ADR-0008 决策 6）：条件 held-out 卷数 < 支撑度界（暂定 20，config `reward.gate_support_min_volumes`）时，该条件过线判据从池化点估计改为 bootstrap CI 下界 ≥ 门槛——重采样单元是卷级聚类（每卷一组 patch 分数整卷进出；patch 级打散把同卷强相关 patch 当独立观测、低估 CI 宽度），重复数与分位固化在 `cynosure.reward.support`。MRA ≈ 16 卷命中（bootstrap 判定）、T2w ≈ 67 卷不命中（点估计判定）。
_Avoid_: 小支撑的点估计直接过线、patch 级 bootstrap、换被估计量（两口径同为池化 AUC，只差判据形态）

**条件白名单（Condition whitelist）**:
RM readiness gate 的产物：预训练后逐条件判定的「判别器在该条件上有分辨率」清单。RL 期间它是 policy 更新的按条件开关（消费见 梯度门控）——名单内正常更新，名单外只跑 rollout 与判别器更新。名单不是静态产物：动态恢复（EMA 滞回）驱动名单进出——gated 条件的在线 per-condition held-out AUC 经 EMA 平滑越过 enter 阈值即恢复更新，名单内条件跌破 exit 阈值即重新门控（enter/exit/EMA 跨度三 knob 进 config、暂定值待校准；可配置关闭，静态白名单为降级路径）。门控决定全 rank 集体口径（rank 0 判定 + broadcast），门控状态随续训分片落盘逐位复原。条件闸关闭时名单不参与任何决定（见 条件闸）。
_Avoid_: 条件调度（rollout 条件分布的配平，另一概念）

**生成条件（Generation condition）**:
RL 条件的按 (模态, 平面) 分组单位（MR-RATE 换域线口径，BraTS 线条件单位仍是序列/有序对）：#81 终审白名单全量 11 个——T1w/T2w/FLAIR 各三平面 + SWI/AXIAL（仅轴位可得）+ MRA/ALL-PLANES（全平面一格；T2w 读数三格并池但条件独立成格）。每条件是**五元组**（modality token / plane / 推荐 FOV / 统一网格 / 等效 spacing），唯一来源是仓库工件 `data/conditions/mrrate_conditions.json`（#127 工件化，取代 #119 的 config 内嵌词表——config 不内嵌词表、代码内无常量副本）：token 映射 9/10/11/20/16 为上游 `configs/modality_mapping.json` 权威、条件 token 由映射派生（skull-stripped 29–33 不进本轮生成词汇）；统一网格 = #78 普查工件逐条件众数 latent 网格 ×4、等效 spacing = FOV / 网格（条件属性）。装载面 = `cynosure.conditions.MrConditionVocabulary`（config 经 `artifacts.condition_vocabulary_json` 携带路径）：装载期字段级拒绝（缺格 / 网格不符 / 字段缺失），生产模式逐条件对账普查期望网格；小词汇表只能经 `fixture_mode=True` 显式装载。两套口径经 `experiment.dataset` 互斥激活（`BraTS2023` 默认、既有 BraTS 线行为不变；`MR-RATE` 必带词汇工件绑定、BraTS 携带即拒），MR-RATE 线只定义组1（上游无 MR ControlNet）。词表的运行时消费（#129 + #131 均已接线）：`cynosure.conditions.ConditionVocabulary` 协议是两域统一解析面（rollout 条件组装、latent 形状按条件贯通、逐条件 sigma 锚、token/spacing 取数）——MR 侧 = 词汇表工件装载产物、BraTS 侧 = 单域常量策略（四序列、任意条件恒 config `latent_shape`，单域 = 单条件词汇特例）；rollout 初始噪声与 eval/baseline 采样的形状逐条件解析（批内同条件即同形状）、sigma 日程经 `ConditionSchedules` 按条件名选择（锚 = 该条件空间 numel，与形状同源派生、结构性防错位；MR 线 config 显式携带单域锚字段 `latent_shape`/`policy.input_img_size_numel` 即字段级拒绝）；real pool 条带切片的分层轴、manifest 逐条件形状契约的装配期对照与预训练报告的条件域/词表指纹同属词表消费面（#129：切片轴经注入、同名异形在装配期拒绝、报告键 = 本域条件名 + 词表工件内容 sha256）；域边界守卫同属 #129 消费侧收口——多条件域 real 工件缺逐条件形状契约即装配期拒绝（判据 = `ConditionVocabulary.single_condition`；缺表会静默回退全局 `latent_shape` 对账），未交付的 MR 面在构造/装配期显式拒绝而非深炸 BraTS 布局错误或单域默认锚（轨迹诊断一处）。里程碑参照影像库 #124 已交付（`MrReferenceVolumeStore`：参照卷集与卷→条件映射唯一来源 = real pool manifest 条目——病例级 train split 泄漏守卫与 BraTS 侧同源；组1 参照轮转**按目标条件过滤**（一卷一条件，BraTS 式全病例池轮转在条件维度上不成立）；装载 = dataset_root 平铺 NIfTI 经逐条件统一网格预处理链，与 prepare 预编码同口径；两域共同满足 `ReferenceVolumes` 协议、装配期 `_build_reals` 单点分派）——监控相装配守卫绑定「本 run 有无里程碑触发点」不变：`schedule.max_iterations < milestone_interval` 的 run 不装配监控相、不消费参照库。预处理半边 #130 已落地（`UpstreamPreprocessChain` 两臂旋钮——强度臂 BraTS `clip=True` / MR-RATE `clip=False` #71 裁决、resize 目标 BraTS dim 公式 / MR-RATE 词汇表统一网格绝对目标；`MrConditionVocabulary.spacing_condition` 条件属性解析面与 `latent_shape` 对偶，换算因子 `SPACING_CONDITION_SCALE` 提升至 config 单一来源）；prepare 数据链 #131 已落地（#121 票内：官方 split join + 评估集互斥硬守卫 + train split 内 patient 级 held-out 二分 + pool 与 held-out 两侧逐条件配额抽样 + 抽样留痕 + 装配期容量守卫，条件分层经 `LatentManifest.modalities`/`condition_latent_shapes`——口径见 `docs/spec/data-preparation.md`「MR-RATE 换域」节）。完整口径见 `docs/spec/experiment-design.md`「条件词表口径」节。
_Avoid_: 把 skull-stripped 码当生成 token（生成分组恒用 whole-brain 条目）、在 BraTS config 里携带 MR 词汇工件（互斥携带即拒）、在 MR config 里声明单域锚字段（形状/锚逐条件派生，#129 互斥携带即拒）、在代码里写 token/网格常量副本（唯一来源是工件）

**梯度门控（Gradient gating）**:
白名单的按 iteration 消费（ADR-0008 决策 7）：目标条件不在名单 → 该 iteration 跳过 policy 更新——rollout、判别器更新（同源重构 fake 现做现用）、iter 事件照常。语义 = 拒绝在 RM 无分辨率的样本上做策略梯度（GRPO 无效样本不参与 advantage 的既有实践），不引入第二重 reward、KL 或参考模型。被门控条件的判别器持续受训——其建立判别力是白名单动态恢复的前提。门控决定是全 rank 集体口径：任一 rank 的条件被门控即全体跳过（policy 更新的 FSDP 梯度 allreduce 是全 rank 集合操作，部分 rank 跳过会互等死锁）；iter 事件以 policy_gated 标记区分门控 iteration（loss 缺 policy 项）。名单恢复由 条件白名单 词条的动态恢复机制驱动。条件闸关闭时本词条整体停用（无 iteration 被跳过，policy_gated 恒 false；见 条件闸）。
_Avoid_: 条件过滤（判别器侧条件匹配采样，另一概念）

**Online update（在线更新）**:
reward model 在 RL 期间每个 iteration 用新 fake 样本继续重训（紧随预训练 warm-start）：fake = 当前 policy 的同源重构、现做现用（ADR-0012），追踪 policy 演化、抗 Reward hacking——对抗博弈里判别器一侧的必要运动（见 对抗博弈）。
_Avoid_: 在线从零（冷启动形态，已被预训练取代）

**条件匹配采样（Condition-matched sampling）**:
判别器一步更新的 real 侧按本 iteration 的目标模态过滤——判别器永远在「同一目标模态的 real vs fake」内比较，杜绝跨模态混采喂出的模态分类捷径；fake 侧经同源重构自动匹配（fake 由同批 real 重构而来，ADR-0012）。real 池逐（目标模态, 每 rank 切片）容量 ≥ K 是装配期硬守卫（不足 fail-fast 可读报错；无放回采样语义不动，不引入有放回采样补洞）。
_Avoid_: 混采（real 全池混采的旧口径，已被本词条取代）

**过拟合分叉监控（Overfit divergence monitoring）**:
判别器内收敛健康度观测面（ADR-0009-β 在线侧 / γ 预训练侧）：分叉 = EMA(train pairwise acc − held-out AUC)，两侧统一干净域、同一 Mann-Whitney pairwise 占比估计量（不同采样平面）——train 侧每判别器步用干净域输入 no_grad 复算一次准确率（更新前快照、随单步更新报告上行；不复用 loss 伴生量——复算保证两侧同一估计量口径），held-out 侧消费现成 per-condition AUC 流（更新前快照）。健康判别器两侧近似相等、分叉贴 0；判别器记住训练批共性而非真假分界时 train 侧被 in-sample 拟合抬高、分叉上行——hacking 后果出现前的病因信号。分叉按条件、按 rank 独立记账（rank 间离散 = 数据切片异质性的诊断信号，不跨 rank 平均），EMA 跨度与报警阈值进 config（`reward.overfit_ema_span` / `reward.overfit_alert_divergence`，暂定 8 / 0.2，MR-RATE 预训练曲线校准后定版）。分叉 EMA 自下而上越线 → `overfit_alert` 事件进指标流（modality、分叉值、train acc、held-out AUC、rank + γ 的相判别字段 `phase`；事件契约「可扩不可改名」、非有限浮点构造期拒绝）——只报警、人工裁决：不自动移出白名单（升级项留曲线校准后另议）。预训练与在线两阶段同一套组件、同一 knobs（共享装配缝挂进 RewardCoordinator 的 OverfitMonitor）：warm-start 预训练每个更新步喂入两侧干净域读数，per-condition 分叉监控在 RM readiness gate 之前即暴露稀疏模态（MRA）记忆化；预训练相告警随 pretrain 事件之后写出，`phase="pretrain"` 登记 EXEMPT 记账（预训练执行史全量保留——不参与续训回退重写），RL 相告警按 iteration 轴参与回退记账（随所属 iteration 删除、回退重执行重发）；分布式预训练（ADR-0016）下越线告警经 gather 归并到 rank0 事件流、仅 rank0 写出（train 侧 EventMerger 同构，rank 归因观测 rank，写出序 = 步序 + 步内 pretrain 先于告警）；per-condition EMA 状态随续训分片落盘（v6），恢复逐位复原。
_Avoid_: 训练/验证损失分叉（机器学习泛指——本项目分叉轴是 in-sample 训练批 vs held-out 池）、自动动作（只报警、人工裁决；旧「自动降 σ」升级项随噪声注入取消作废，ADR-0012）、跨 rank 平均的分叉读数（rank 离散本身是诊断信号）、预训练/在线口径断层（两阶段同一套组件与 knobs，γ 已收口）

**Reward hacking（奖励攻击）**:
policy 学会骗过判别器拿高分，而非真正提升样本质量——静止判别器的必然结局（见 对抗博弈）。

**Environment（环境）**:
RL 意义上 policy 之外的固定世界：真实数据分布（Real sample pool / Held-out real，全程冻结）与去噪动力学。Reward model 不属于环境——它是对该分布的判别器与估计器，属博弈一方（见 对抗博弈）。
_Avoid_: 环境漂移、非平稳环境（把判别器划进环境的口径）

**Adversarial game（对抗博弈）**:
policy 与 reward model 的关系定性：判别器与被其分数经 GRPO 驱动的 policy 构成双人博弈。判别器持续移动（Online update）是博弈成立的必要条件——静止判别器必然被 exploit；收敛性依据 two-timescale 随机逼近（Borkar 1997；TTUR, Heusel et al. 2017），不依赖单智能体 MDP 的平稳性假设。
_Avoid_: 时变 reward、非平稳 MDP（同上）

**Two-timescale separation（时间尺度分离）**:
对抗博弈收敛的工程条件：判别器的有效时间尺度（步数 × 学习率 × 收缩率）应快于 policy，使打分所见的判别器近似「当前 policy 下的收敛判别器」。分离坍塌的签名 = policy 追打过期判别器（Reward hacking）或双侧震荡。
_Avoid_: 学习率比例（有效时间尺度不止 lr）

**Piecewise-stationary scoring（分段平稳打分）**:
iteration 内全部 rollout 由同一判别器快照打分（打分先于该 iteration 的判别器更新），组内 advantage 同尺可比；非平稳性只以 iteration 间阶跃形态存在，每个 policy 更新步面对平稳子问题。
_Avoid_: 边训边打、中途换尺

**Group（组）/ Advantage（优势）**:
GRPO 中共享同一初始噪声的 G 条 rollout 为一组；advantage 是该组内标准化后的 reward。

**Granularity（粒度）**:
Granular-GRPO 里续跑采样所用的时间步间隔 λ；多粒度（multi-granularity）指多个 λ 的 reward 融合。
_Avoid_: 分辨率、尺度（尺度另有所指，见 多尺度判别器）

### 分布式执行

**Rank（进程秩）**:
torchrun 进程组内进程的全局编号（0 起）；rank 0 独占产物写盘（run 目录创建、指标归并、checkpoint），其余 rank 只参与集合通信。
_Avoid_: 进程号、节点（node 是机器，rank 是进程）

**World size（进程组规模）**:
参与训练的 rank 总数（torchrun ``--nproc_per_node`` 语义）；续训状态的 world_size 契约对账拒绝跨拓扑恢复。

**FSDP full-shard（全分片）**:
可训练网络的参数/梯度/优化器状态按 rank 切分、前向按需重组的数据并行方式；梯度 allreduce 保证各 rank 权重同步（ADR-0003：同构 rank、无角色划分）。

**DDP replica（判别器完整副本）**:
判别器不参与分片的分布式口径：每 rank 完整副本 + 标准 DDP 梯度 allreduce；各 rank 用本 rank fake + Real sample pool 切片更新。
_Avoid_: 判别器分片

**Rank-sliced pool（真实样本库切片）**:
Real sample pool 按**活动条件集**分层的条带切片（每条件 entries[rank::world]），各 rank 判别器 real 侧只见本切片；条件集经装配注入（#129：BraTS = 四序列、MR-RATE = 词表条件集，代码内不设四序列副本）；held-out real 不切（out-of-sample 监控保持全量）。

**Resume generation marker（续训代际标记）**:
全部 rank 分片均已持久化到同一 iteration 的提交记录（resume_generation.json，save 的 barrier 之后由 rank 0 写出）；恢复对账标记代际、混代际分片（保存中途崩溃现场）显式拒绝——各 rank 必须从同一 iteration 继续。

**Metric merge（指标归并）**:
iter 事件由各 rank gather 到 rank 0、按 (iteration, rank) 稳定序写出的合并写出——无重复、无丢失。

**World-1 degeneration（world-1 恒等退化）**:
单进程 = world size 1 的退化实现：不初始化进程组、集合通信原语恒等（barrier/gather 直接返回），训练循环对单进程/分布式走同一条执行序。
_Avoid_: 单机模式（单机也可多进程）

**分片自持（Component-owned resume state）**:
续训分片的读写知识归各协作者自身（ADR-0014，实施票未启动）：协作者实现 `state()` / `adopt()` 小接口（gating / overfit 既有雏形命名），分片键由组件自持声明，resume 只跨 trainer 一道 seam、不再穿透组件树（旧形态：`trainer.rewards.update.optimizer` 三跳 + 8 个转发 property）；`adopt` 的 dict 形态校验为共享 helper 单点。分片格式变更循升版拒旧先例（v10 清单退役、legacy 拒载），不写迁移读取。
_Avoid_: resume 穿透属性链（本词条落地后即违例）、转发 property（interface 由消费者需求长出）、迁移读取（先例是升版拒旧）

### 实验设计与验收

**Baseline（无 RL 基线）**:
冻结基座 checkpoint（不做 RL）的采样结果，作 RL 收益的对照基准；每组各一条（组1 = base UNet @ CFG=10；组2 = base UNet + 冻结预训练 ControlNet）。
_Avoid_: 对照组、基准线

**Quantitative evaluation（定量评测）**:
像素域 2.5D FID（XY/YZ/ZX 三正交面、RadImageNet-ResNet50）+ KID/bootstrap CI 的自动化质量评测；跨模态组另加 3D SSIM/MAE/PSNR（合成 target 与同一病例 ground-truth target 配对比较）。
_Avoid_: L1

**Milestone evaluation（里程碑评测）**:
按里程碑间隔（默认每 50 iteration）在 train 循环内触发的解码评测——VAE 解码**监控子样本**（Baseline manifest 条目前缀 K，`milestone_eval_samples`，同 seed 同条件 → 跨里程碑可比）到像素域算 FID/KID（按目标条件分层宏平均），结果以 `milestone` 事件写入训练指标流，事件携带监控成本读数（总卡时 + decode/fid 相位分解，#111 监控账的成本行来源）。解码只发生在 Baseline / 里程碑 / 重采三条评测路径，不进逐 iteration 训练循环（iter 事件账无 decode 行项）。参照影像库按域分派：BraTS = 病例目录布局、病例×序列矩阵取体；MR-RATE = 平铺影像树、参照卷集锁 real pool train split 且轮转**按目标条件过滤**（一卷一条件，跨条件取卷即跨域比较，#124）。解码前向冻结为 fp16 autocast 口径（官方 NV-Generate-CTMR utils_infer 同款；不得换 bf16——与官方口径输出差 ~6.8e-02；生产 config `norm_float16=true` 下纯 fp32 前向对 fp32 conv 抛 dtype 错，无条件包 autocast、不分设备分支），出口统一上浮 fp32 供指标/落盘消费。
_Avoid_: 定期评测、周期评测

**Monitoring subsample（监控子样本）**:
监控/评测路径专属的 decode 样本面：Baseline/重采 = N_baseline 全量 manifest 条目；里程碑 = 条目前缀 K（`milestone_eval_samples`，小样本相对信号，只服务于跨里程碑 plateau 比较）。decode 从不为训练主循环发生——主循环全程 latent 域（维护者裁决，#111 两本账口径），监控成本（decode+FID 卡时）只在里程碑事件上读数。
_Avoid_: 评测集（评估集是数据侧互斥分区的概念，样本面是 run 内 manifest 条目）

**Adjudication FID（裁决性 FID 读数）**:
fork 口径 2.5D FID 仪器（切片原尺寸直进 ResNet50、MR 百分位强度臂 crop→强度→pad、padding=0）产出的基线与 RL 终评读数。与里程碑评测**双轨**：里程碑维持 224×224 设施口径只看趋势，两侧数字不可互比；裁决性读数走冻结变量 config 校验（`mr_fid` schema）+ `FidResult` provenance 落盘，特征缓存带口径 fingerprint 防护。解读锚点 = real-vs-real 地板（病例级 seed 半分互比）。
_Avoid_: 验收 FID、最终 FID（验收是阶梯概念，本仪器只产读数）

**Real-vs-real floor（real-vs-real 地板）**:
同仪器下 real 集合的病例级 seed 半分（half_a vs half_b）互比 FID——「该仪器在该数据上该样本量的噪声下限」。没有地板，绝对 FID 无法解读；地板同时是仪器接线自检（应与合成读数同量级）。半分记录（seed + 两半 + 来源）冻结落盘一次、永不重算。
_Avoid_: 基线 FID（基线是 policy 概念，地板是数据切分概念）

**Downstream distribution alignment（下游指标分布对齐）**:
用 nnUNet 仪器产出肿瘤体积/质心/ET-WT，对合成影像与真实分布做 TOST/KS/EMD 对齐检验；合成影像无 GT，不比 Dice。
_Avoid_: L2、Dice 主判据

**Expert review（专家目检）**:
盲审流水线（视觉图灵 balanced accuracy + 4 维 5 分 Likert + Fleiss' κ）的人工终审。
_Avoid_: L3、专家打分
