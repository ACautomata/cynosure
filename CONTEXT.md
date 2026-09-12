# cynosure

cynosure 为 MAISI 3D latent rectified-flow 医学影像 checkpoint 设计并实施基于 Granular-GRPO 的 RL 后训练。**零依赖原则**：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；网络类来自 MONAI 库本身。

## Language

### 生成与表征

**Latent（潜变量）**:
VAE（AutoencoderKlMaisi）把影像体压缩成的 4 通道低维表征。一切 RL 采样与打分都在 latent 域进行，不回像素域。
_Avoid_: 特征图、embedding、编码

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

### 策略建模

**Velocity（速度场）**:
UNet 的输出语义：`v = x0 − noise`（v-prediction），决定去噪方向。
_Avoid_: 噪声预测、epsilon

**CFG 组合场**:
条件与无条件 velocity 的线性组合 `v_uncond + w·(v_cond − v_uncond)`，w 为引导强度。policy 的有效采样场按组对齐基座推理的 w（组1=10，组2=0）。
_Avoid_: guidance（单独使用时）

**Anchor 轨迹**:
从同一初始噪声确定性 ODE 采出、逐步存下 latent 的参考轨迹；组内全部方向共享它，使 reward 差异唯一归因于被优化那一步的扰动。
_Avoid_: 参考轨迹、主轨迹

**单步 SDE 扰动（Singular Stochastic Sampling）**:
把随机性限制在单个被优化步——仅在该步把确定性 ODE 步替换为带噪高斯核，其余步保持确定性。
_Avoid_: 全程加噪、SDE 采样

**MGAI（Multi-Granularity Advantage Integration）**:
多粒度 advantage 集成：每个粒度 λ 的 advantage 各自组内标准化后直接求和的融合方式。

### Reward model

**Reward model（奖励模型）**:
给 rollout 的 latent 打标量分的在线判别器，该分即 RL 的 reward。

**PatchDiscriminator（PatchGAN 判别器）**:
MONAI 的 PatchGAN 判别器（Pix2PixHD 式），输出 patch logit 图而非单一标量；本项目用它当 reward model。

**Real sample（真实样本）**:
训练集影像经 VAE 预编码的 latent，作为判别器的「真」，固定不更新。

**Real sample pool（真实样本库）**:
训练集（本轮 = BraTS train split）全量影像经 VAE 预编码的 latent 集合，按序列 token 分层；判别器的「真」与评测参照都取自它。
_Avoid_: 真样本集

**Held-out real（留出真样本）**:
基座 val split（BraTS 病例级 70/10/20 之 10%）影像经 VAE 预编码的 latent 工件，按序列分层；与 Real sample pool 病例级不相交、永久不参与判别器更新——保证 held-out AUC 是 out-of-sample 的 hacking 监控信号（reward-model 章，`prepare` 产出）。
_Avoid_: 验证集（val split 是划分段，held-out real 是其预编码工件）

**Fake sample（伪样本）**:
当前 policy rollout 的去噪输出 latent，作为判别器的「假」。
_Avoid_: 生成样本、负样本

**Warm-start pre-training（判别器预训练）**:
RL 启动前对 reward model 的离线密集训练：real 取 Real sample pool，fake 取 base policy 量产 rollout；训练至通过 RM readiness gate，产物作为在线更新的初始权重（ADR-0007）。
_Avoid_: 一次性预训练、离线 reward model（RLHF 语境指冻结，本项目预训练后仍在线更新）

**RM readiness gate（RM 上岗门槛）**:
RL 启动的硬前置：判别器预训练后按条件报告 held-out AUC，过线条件构成条件白名单；白名单为空拒绝开跑，非空即放行。
_Avoid_: 软警告、早停（早停是训练期机制，门槛是启动期机制）、池化达标（全池单一标量口径，已被按条件取代）

**支撑度规则（Support rule）**:
门槛判定的统计形态（ADR-0008 决策 6）：条件 held-out 卷数 < 支撑度界（暂定 20，config `reward.gate_support_min_volumes`）时，该条件过线判据从池化点估计改为 bootstrap CI 下界 ≥ 门槛——重采样单元是卷级聚类（每卷一组 patch 分数整卷进出；patch 级打散把同卷强相关 patch 当独立观测、低估 CI 宽度），重复数与分位固化在 `cynosure.reward.support`。MRA ≈ 16 卷命中（bootstrap 判定）、T2w ≈ 67 卷不命中（点估计判定）。
_Avoid_: 小支撑的点估计直接过线、patch 级 bootstrap、换被估计量（两口径同为池化 AUC，只差判据形态）

**条件白名单（Condition whitelist）**:
RM readiness gate 的产物：预训练后逐条件判定的「判别器在该条件上有分辨率」清单。RL 期间它是 policy 更新的按条件开关——名单内正常更新，名单外只跑 rollout 与判别器更新，待其在线判别力出带自动恢复。
_Avoid_: 条件调度（rollout 条件分布的配平，另一概念）

**Online update（在线更新）**:
reward model 在 RL 期间每个 iteration 用新 fake 样本继续重训（紧随预训练 warm-start），追踪 policy 演化、抗 Reward hacking。
_Avoid_: 在线从零（冷启动形态，已被预训练取代）

**条件匹配采样（Condition-matched sampling）**:
判别器一步更新的 real 侧与回放半区都按本 iteration 的目标模态过滤——判别器永远在「同一目标模态的 real vs fake」内比较，杜绝跨模态混采喂出的模态分类捷径。
_Avoid_: 混采（real 全池混采的旧口径，已被本词条取代）

**Replay buffer（回放缓冲）**:
封顶 FIFO 的 fake latent 存库（base 时期 + 近期），条目带条件标记（目标模态标签——组2 跨模态条目按目标端归因）；更新判别器时按比例混入、回放抽取与本 iteration 条件匹配，防漂移、防灾难性遗忘。
条件标记的落地面（ADR-0008-01）：base 分区种子按每条件配额量产（配额 ≥ 判别器更新回放半区需求是装配期硬守卫，预训练 driver 与 train 装配同口径）；回放采样按条件过滤、该条件候选不足时显式拒绝（可区分「条件不足」与「总数不足」，绝不静默回退全池混采）；续训状态分片 v3 随条目标记升级，旧格式分片被版本对账显式拒绝。
_Avoid_: 无条件回放（旧裸 latent FIFO 口径，已被带标签 FIFO 取代）

**Reward hacking（奖励攻击）**:
policy 学会骗过判别器拿高分，而非真正提升样本质量。

**Group（组）/ Advantage（优势）**:
GRPO 中共享同一初始噪声的 G 条 rollout 为一组；advantage 是该组内标准化后的 reward。

**Granularity（粒度）**:
Granular-GRPO 里续跑采样所用的时间步间隔 λ；多粒度（multi-granularity）指多个 λ 的 reward 融合。
_Avoid_: 分辨率、尺度（尺度另有所指，见单/多尺度判别器）

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
Real sample pool 按序列分层的条带切片（每序列 entries[rank::world]），各 rank 判别器 real 侧只见本切片；held-out real 不切（out-of-sample 监控保持全量）。

**Per-rank replay buffer（每 rank 回放缓冲）**:
每 rank 独立的 fake 回放缓冲（rollout 数据各 rank 独立演化）；续训状态同样按 rank 分片落盘（resume_state_rank{R}.pt），恢复时对号取回。

**Resume generation marker（续训代际标记）**:
全部 rank 分片均已持久化到同一 iteration 的提交记录（resume_generation.json，save 的 barrier 之后由 rank 0 写出）；恢复对账标记代际、混代际分片（保存中途崩溃现场）显式拒绝——各 rank 必须从同一 iteration 继续。

**Metric merge（指标归并）**:
iter 事件由各 rank gather 到 rank 0、按 (iteration, rank) 稳定序写出的合并写出——无重复、无丢失。

**World-1 degeneration（world-1 恒等退化）**:
单进程 = world size 1 的退化实现：不初始化进程组、集合通信原语恒等（barrier/gather 直接返回），训练循环对单进程/分布式走同一条执行序。
_Avoid_: 单机模式（单机也可多进程）

### 实验设计与验收

**Baseline（无 RL 基线）**:
冻结基座 checkpoint（不做 RL）的采样结果，作 RL 收益的对照基准；每组各一条（组1 = base UNet @ CFG=10；组2 = base UNet + 冻结预训练 ControlNet）。
_Avoid_: 对照组、基准线

**Quantitative evaluation（定量评测）**:
像素域 2.5D FID（XY/YZ/ZX 三正交面、RadImageNet-ResNet50）+ KID/bootstrap CI 的自动化质量评测；跨模态组另加 3D SSIM/MAE/PSNR（合成 target 与同一病例 ground-truth target 配对比较）。
_Avoid_: L1

**Milestone evaluation（里程碑评测）**:
按里程碑间隔（默认每 50 iteration）在 train 循环内触发的解码评测——VAE 解码当前 policy 采样到像素域算 FID/KID，结果以 `milestone` 事件写入训练指标流。解码只发生在里程碑路径，不进逐 iteration 训练循环。
_Avoid_: 定期评测、周期评测

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
