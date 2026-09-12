# Reward Model 方案：在线 latent PatchDiscriminator

> 本章是整体实施 spec（地图 #2「CTMR Granular-GRPO RL 后训练方案」）的 reward-model 章节，由 ticket #6 决议产出。终稿由 ticket #9 汇总。

## 范围与输入

- **零依赖原则**：不 import NV-Generate-CTMR 任何代码。reward model 只依赖 MONAI 库与 VAE 预编码 latent。输入物 = {UNet ckpt, VAE ckpt, 网络配置 JSON, modality_mapping}（跨模态实验加 ControlNet ckpt）。
- **latent 形状 = `[4, 64, 64, 32]`**：[256,256,128] 影像体经 `AutoencoderKlMaisi`（`num_channels=[64,128,256]`，**4× 空间压缩**，`latent_channels=4`）得到。各向异性（层间 32），但物理体素 ~6.8×6.8×8 mm，**近似各向同性**。
- **real** = 训练集影像的 VAE 预编码 latent（固定，不更新）。
- **fake** = 当前 policy rollout 的去噪输出 latent。
- **latent 域打分**，不经 decoder——相比 Granular-GRPO 原版的像素域解码打分，**省去全部 VAE 解码开销**。

## 判别器架构

基类用 MONAI `PatchDiscriminator` / `MultiScalePatchDiscriminator`（Pix2PixHD 式）。

- **in_channels = 4**；输入 latent 先做 per-channel 标准化（用训练集统计量）。
- **归一化 = GroupNorm**，**不用默认 `BATCH`**：判别器在线小 batch 更新、fake 分布每个 RL iteration 都在漂移，BatchNorm 的 running stats 不稳定且泄漏 batch 统计，直接污染 reward。可选在 conv 上叠 **SpectralNorm**——Lipschitz 约束既稳住在线训练，也是一道轻量防 hack（防止判别器过锐被 policy 钻空）。
- **patch 粒度（关键）**：MONAI 默认 `num_layers_d=3` 时总 stride=8、感受野 ~70³，在 `[4,64,64,32]` 上**覆盖整个 latent 体、退化为 image-level 判别器**。故往下调：
  - `num_layers_d=1` → 感受野 ~16³，stride 2，输出 32×32×16；
  - `num_layers_d=2` → 感受野 ~34³，stride 4，输出 16×16×8（真·局部 patch，推荐起步）。
- **尺度单/多不定死**，作为消融轴（见「消融矩阵」）。
- `kernel=4` 固定；各向异性不做特殊处理（物理体素近似各向同性）。

## 损失与 reward 标量

- **损失 = LSGAN（least squares）**；**reward = raw real-logit（不过 sigmoid）**。
- **理由**：GRPO 的 advantage 是组内 `(r − mean)/std`，**scale-invariant**，要的是 reward 的**组内分辨率**而非绝对有界性。raw logit 不饱和、对「接近真实」的样本持续有区分度；sigmoid 概率一旦饱和（判别器有把握 → 全组趋近 1.0）会杀掉组内方差，触发 1e-8 保护 / advantage 爆炸。LSGAN 同时给判别器非饱和的训练梯度。
- **聚合**：patch logit 图 `[B,1,D′,H′,W′]` → 标量。**mean 为主**；**min**（对局部伪影更敏感）作为与尺度正交的第三维消融。多尺度臂：各尺度先各自 mean 成标量，再**跨尺度相加**（对齐 Granular-GRPO 的 advantage 求和哲学，免调跨尺度权重）。
- **有界化**：起步**不额外有界**，靠组内标准化 + adv clamp ±5 吸收量纲；**tanh 压 (-1,1) 作为一行代码的廉价保险**，监控发现 logit 幅度持续膨胀时再开。

## 在线更新机制

> RL 启动前先经 **Warm-start 预训练**（ADR-0007，见下节）——本节的在线节奏适用于 RL 期间：预训练后的判别器继续随每个 iteration 在线更新（ADR-0001 核心保留，仅「在线从零」被修订）。

- **节奏**：`N=1`（每个 RL iteration 都更新判别器），每批 `K` 小，**D:G 更新比 ≈ 1:1**；优化器 AdamW，LR 1e-5~1e-4（与 policy 同量级）。判别器是几层 3D conv，相对 UNet rollout（30 步 ODE × G 方向）算力可忽略，故「每 iter 更新」几乎免费；真正的约束是 rollout 吞吐（见 ticket #7 编排）。`N/K/LR` 标为 **tunable**，待 profile 后定。`weight_decay` 显式落位（`disc_weight_decay`，与 policy 侧同值口径 1e-4）——此前隐式取 PyTorch 默认 0.01 的不对称已消除（ADR-0007 卫生项）。
- **条件匹配监督（ADR-0008）**：判别器一步更新内 real 侧与回放半区都按本 iteration 的目标模态过滤——判别器永远在「同一目标模态的 real vs fake」内比较，杜绝跨模态混采在稀疏模态上喂出的模态分类捷径（real 全池混采的旧口径废止）。Replay buffer 条目带模态标签，回放不足时该步退化为纯 current 半区并落盘；real / 回放候选容量在装配期逐（切片, 模态）校验、不足 fail-fast。
- **fake 缓冲**：封顶 **FIFO 回放缓冲** = base 时期样本（初始冻结 policy 产出）+ 近期 policy 样本，按 **50% 当前 / 50% 回放** 混合采样。real 侧固定训练集 latent，不漂。理由：防止判别器随 policy 变好而**灾难性遗忘**「明显假」长什么样，稳定在线训练、抗漂移（GAN-RL 标准做法；代价仅是显存里存数百~数千个小 latent）。
- **梯度门控（ADR-0008）**：条件不在 RM readiness 白名单（见下节）→ 该 iteration 跳过 policy 更新，rollout、fake 入 buffer、判别器更新照常——拒绝在 RM 无分辨率的样本上做策略梯度（不引入第二重 reward、KL 或参考模型）。白名单动态恢复：该条件在线 per-condition AUC 的 EMA 越 enter 阈值（暂定 0.55）恢复更新、跌破 exit 阈值（暂定 0.52）重新门控。盲条件的组内标准化噪声梯度（std+1e-8 尺度不变 + clamp ±5）由此道防线拦截。

## Warm-start 预训练与 RM readiness gate（ADR-0007 / ADR-0008）

T12/T13 取证（#56）：判别器在线 1 step/iter 的训练量结构性不足——100 iter 全程徘徊 chance 带，advantage 信号近噪声。ADR-0007 修订 ADR-0001 的「在线从零」：RL 启动前新增判别器**密集预训练**，产物作为在线更新的初始权重。

- **入口**：`pretrain` 子命令（与 train / eval / prepare 共享同一 config schema 与 dispatch 前校验）。单进程执行（World-1 退化路径），产物全局唯一——判别器是 DDP 完整副本口径，多 rank 各自预训练会分叉（torchrun 启动显式拒绝）。
- **数据**：real = Real sample pool manifest（kind 守卫装载）；fake = base policy 冻结 rollout 量产，复用回放缓冲的 base 分区采样入口（批量分块、独立随机流、输出归一到 pool 存储域）。组1 / 组2 各自预训练 run（fake 分布不同）：采样场与条件分布经 `GroupPolicy` 按 config 分派——**同一条代码路径，仅 config 不同**。
- **训练循环（ADR-0008 per-condition 步进的**最小诚实形态已落**：每步单条件量产 + real 同条件匹配；均匀轮转调度与 per-condition AUC 归因随 ADR-0008-04 落码）**：复用在线期同款判别器单步更新原语（`OnlineUpdate.step` + LSGAN + AdamW）密集步进，**无第二套判别器训练逻辑**。每步先抽一个条件（当前为条件分布的随机采样，均匀轮转归 ADR-0008-04），再量产该条件 fake 批——`update_step` 的回放按该条件过滤、real 同条件匹配（混采量产批没有诚实标签可穿，ADR-0008-03）；「预训练期混采退化为 base fake 库内采样」的旧口径废止，AUC 归因暂为全池混采（per-condition 归因随 ADR-0008-04）。终止 = 全部条件最近一次 per-condition AUC 过线即停（白名单空 → 拒跑）；当前实现的终止 = held-out AUC ≥ 门槛（`pretrain_gate_auc`，暂定 **0.65**、chance 带外，用预训练曲线校准后定版）**且换批复测仍 ≥ 门槛**（两次独立测量都达标才终止，报告值取两次较小者——train 侧 gate 按独立采样对同一阈值重算，单批贴线越过的 checkpoint 会被非确定性拒绝，producer 侧成功判据须对单批测量噪声鲁棒）或步数上限（`pretrain_max_steps`），两者皆配置化。最终 per-condition AUC（当前实现：单一 `final_heldout_auc`）与落盘 checkpoint 同快照。
- **产物契约**：判别器 checkpoint（可装载 state_dict，与训练期产物 checkpoint 同构）+ 预训练报告（`kind="pretrain_report"`：组别、**per-condition 最终 AUC 与条件白名单**（ADR-0008，取代单一 `final_heldout_auc` 标量）、门槛与达标与否、数据口径指纹——ChannelStats / Real sample pool manifest / held-out manifest / 判别器网络配置的内容 sha256）。报告落盘路径与 `pretrain_report_json` 声明的一致性是 pretrain 入口的不变式（分叉即 usage error 拒绝——train 按声明精确路径装载，分叉即 missing-report 或静默装旧报告）。装载走守卫入口（`PretrainReport.load` → `assert_data_provenance` → `load_discriminator`）：**缺报告 / kind 不符 / latent 形状不符 / 工件指纹不符 / 形态指纹不符即拒绝**。
- **指标事件**：`pretrain` 事件类型（步号 + loss + held-out AUC + buffer 占用）写入预训练 run 目录的 metrics.jsonl（event 判别字段与 iter / milestone 混存同一流）；事件类型清单与各型的回退记账口径见下节。
- **RM readiness gate（train 入口硬前置；ADR-0007 口径已交付，ADR-0008 按条件化为定版设计）**：判别器装配从预训练报告守卫重载（latent 形状对照 → 数据口径三工件指纹对照 → 形态指纹对照 → checkpoint 严格装载——缺报告 / kind 不符 / 工件损坏 / 口径指纹不匹配即拒绝，冷启动训练路径在 train 侧废弃，``discriminator_ckpt`` 不再是 train 装配消费点——resume 占位装配同样不消费任何 checkpoint 工件，预训练产物清理不阻断续训）；启动期按**当前 run 的数据口径**（held-out real + 本 rank base fake 批）**重算** held-out AUC——不信任预训练报告旧值，重算与预训练 gate 测量同一份 ``HeldOutAuc.compute`` 口径（全池混采），同 scorer 快照 + 同 fake 批下重算值与预训练任一次测量逐位可比；报告 ``final_heldout_auc``（= 跨界测量与换批复测的较小者）是 producer 侧留痕，重算不复现它也不必复现——门槛判定只依赖本次重算与阈值；重算值低于 ``pretrain_gate_auc`` 即给出含实测值与阈值的可读报错并回滚 run 目录（沿用 preflight 失败语义），达标放行。检查发生在 Baseline 采样等昂贵启动动作之前；分布式下各 rank 以本 rank base fake 独立重算、经集合裁决全体一致拒绝（任一 rank 眼里判别器失明都不得开跑；本地重算的任何失败——含 held-out 工件读盘异常——都收敛为该 rank 的裁决输入，不越过集合点）。**resume 跳过门槛与 warm-start 装载**——续训状态已含判别器全量状态（恢复点判别器已在岗），预训练产物被清理的 run 仍可续训；``pretrain_report_json`` 字段引入前的旧 run 快照在续训对账装载期被 schema 必填校验拒绝，旧口径 run 不再支持续训。fixture 不设豁免：fixture config 以低阈值（0.51，chance 带上沿之上）+ 自产小产物经同一条代码路径通过门槛。``pretrain_report_json`` schema 字段（生产配置**必填无默认**）已落位——RL 不带 warm-start 工件在 schema 层就无法启动。ADR-0008 按条件化落地形态：判定对象改为 per-condition AUC + **条件白名单**（过线条件清单，在线期梯度门控按此名单执行，见「在线更新机制」），0.65 为候选阈值（用 MR-RATE 预训练曲线校准后定版），条件 held-out 卷数 < 20 时用 bootstrap CI 下界判定（小样本支撑度规则）；**白名单空 = 拒绝开跑**（硬前置保留），非空即开跑、未过线条件由在线期梯度门控兜底；train 侧重算的白名单化改造随 per-condition 步进代码后续 ticket 交付。门槛是启动期机制，与「防 reward hacking」节的训练期监控（AUC 掉回 chance 带）互不替代。


## 指标事件流的事件类型清单（metrics.jsonl）

指标流是 run 目录的契约工件：一行一事件的 JSONL，`event` 为类型判别字段。契约口径**可扩不可改名**——新增事件类型 = 新判别值 + 新字段，既有类型的判别值、字段名与语义不动；三处同批同步：事件模型（`train/artifacts`）、回退记账登记表（`REWIND_ACCOUNTING`）、本节清单。消费方按判别字段分派，未知类型跳过而非报错。

| 事件类型 | 产出方 | 字段 | 回退（rewind）记账口径 |
|---|---|---|---|
| `iter` | RL 逐 iteration（train 循环） | `iteration` / `stage` / `rank` / `modality` / `anchor_eval_reward` / `intra_group_reward_std` / `heldout_auc` / `loss` / `buffer_current_fraction` / `buffer_replay_fraction` / `buffer_replay_degraded` / `buffer_base_occupied` / `buffer_recent_occupied` / `lr` / `elapsed_s` | 以 0-based iteration 号记账：保留号 < 恢复点 |
| `milestone` | 里程碑解码评测（train 循环） | `iteration` / `stage` / `fid` / `kid` / `ssim` / `mae` / `psnr` / `criteria_summary` / `early_stop` / `early_stop_reason` | 以完成数记账：保留完成数 ≤ 恢复点（评测与恢复点 checkpoint 同批产出） |
| `pretrain` | 判别器 warm-start（pretrain 子命令） | `step` / `loss_discriminator` / `modality`（本步条件，ADR-0008 per-condition 步进）/ `heldout_auc`（本步条件的 AUC）/ `buffer_base_occupied` / `buffer_recent_occupied` / `lr` / `elapsed_s` | **不参与回退**：全量保留 |

- **预训练事件排除在回退口径外的理由**：warm-start 执行史没有对应的 checkpoint 可重放，按任何边界删都是永久丢失——预训练收敛曲线断点、RM readiness gate 的阈值校准（`pretrain_gate_auc` 定版）失去数据基础。
- **曲线的读法**：`heldout_auc` 一律是「本步更新**前**」的快照（与在线期 iter 事件同口径：更新后测同一 fake 批会把 in-sample 拟合计入 AUC）。终止那两次测量（达标跨界测量 + 换批复测 / 步数耗尽后的补测）不进事件流——报告的 `final_heldout_auc` = 跨界测量与复测中的较小者（与落盘 checkpoint 同快照），离线画预训练收敛曲线时两端拼读。
- **登记表 = 删除的准入名单**：回退只对表内口径为删除的轴做判定，表外（未登记 / 新增未声明）的事件类型一律保留——宁可留痕不可误删。
- **同流混存的口径**：流的类型契约不假设一份流里有哪几型事件——当前 CLI 布局下 warm-start 与 RL 各在自己 run 目录（pretrain 写 `pretrain_report_json` 所在目录，train 另建 run 目录），两者分居两流；同流时（warm-start 历史并入 RL run 流）续训回退只重写恢复点之后的 RL 半截执行史，预训练事件**逐字**保留（真 `--resume` 回退路径的专属用例锁死）。


## KL / 稳定性锚定

- **起步不加 KL、不放参考模型**——忠于 Granular-GRPO（原文 β=0、无参考模型），**省一整份 UNet 显存**（对 3D 大 latent 意义重大）。稳定靠 `clip_range=1e-4` 的极窄 PPO trust region + 在线判别器 + 监控。
- **升级项 = 参数 EMA 锚**：不存完整参考模型，用滑动平均权重做软约束，出现漂移/hacking 时启用。优于「对 base ckpt 的 KL」（后者需常驻一份参考模型，吃显存）。

## 防 reward hacking（design 级）

监控最小信号集：

1. **held-out real vs 当前 fake 的判别器准确率 / AUC**——掉到近 chance（~50%）而 eval-reward 仍在升 = **典型 hacking 签名**；
2. **组内 reward std**——→0 则分辨率丢失；
3. **anchor 终点 eval reward 曲线**（Granular-GRPO 本就落盘），与 (1) 背离即报警。

- **EMA 锚触发点** = (1) 近 chance + eval reward 仍上升。
- **按条件归因（ADR-0008）**：AUC 信号一律按 `record.modality` 归因（iter 事件已如此）；盲条件的「满幅噪声梯度」破坏由梯度门控防线治理。hacking 签名的判型（判别器冷 / 盲 / 被错误引导三分）依赖按条件口径，另行开票。
- **阈值具体数值与完整早停准则不在本章锁死** → 移交 ticket #8（实验设计与验收）与地图 fog「Reward hacking 的监控指标与早停准则」（依赖训练期经验数据，属执行，超出本地图 spec 范围）。

## 消融矩阵（实验计划）

固定 GroupNorm + LSGAN + raw logit，扫以下三维，**小步扫、不做全组合爆破**：

| 维度 | 取值 |
|---|---|
| 臂 A：单尺度深度 | `num_layers_d ∈ {1, 2}` |
| 臂 B：多尺度 | `num_d ∈ {2, 3}` |
| 维 C：聚合 | `{mean, min}` |

每组用 **held-out 判别器 AUC** + **组内 reward std** 两指标定胜负。实际执行属 ticket #8 / 后续施工，超出本地图（spec）范围。

## 待定 / 移交

- 精确 `N/K`、判别器 LR、replay buffer 容量 → rollout 吞吐 profile 后定（ticket #7 编排、ticket #9 终稿）。
- **门槛阈值 0.65 定版与动态门控 enter/exit/EMA 跨度定版** → MR-RATE 预训练曲线校准（ADR-0008；chance 带 ≈ 0.5±0.02 来自 T13 实测）。
- hacking 监控阈值、早停准则 → ticket #8 + 对应 fog（依赖经验数据）。
- **预案 A（冻结判别器 + EMA 锚）**：hacking 签名（`anchor_eval_reward` 升 + milestone FID 同步恶化）触发时切换——实现另开 ticket（ADR-0007 Considered Options A）。
- 若将来加第二 reward（重建 loss / 分割指标）→ 各自组内标准化后 advantage 相加（Granular-GRPO 双 reward 做法）。

## 依据

- MONAI `PatchDiscriminator` / `MultiScalePatchDiscriminator` 源码（Pix2PixHD 式，输出 patch logit 图 + 中间特征）。
- NV-Generate-CTMR `configs/config_network_p3.json`（`AutoencoderKlMaisi` `num_channels=[64,128,256]`、`latent_channels=4`、`RFlowScheduler`）与 `config_maisi_diff_model_rflow-{ct,mr-brain}.json`（inference dim `[256,256,128]`）。
- Granular-GRPO 调研 `research/granular-grpo.md`（无 KL/参考模型、`clip_range=1e-4`、advantage 组内标准化 + clamp±5、依赖组内 reward 方差非零）。
