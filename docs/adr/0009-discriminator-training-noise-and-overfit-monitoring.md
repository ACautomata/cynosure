# 判别器训练期对称噪声注入与过拟合分叉监控（承接 ADR-0008 遗留半边）

ADR-0008 的条件匹配采样把判别器 real 侧骤缩到单模态池：MRA train split ≈ 110 条（全池），经 `RankSlicedPool` 按序列条带切片后每 rank 仅 ≈ 110/world 条（world=8 → ≈ 14 条），装配期容量守卫要求 K ≤ 池大小。叠加 ADR-0008 裁决 3 的预训练 per-condition 均匀轮转（默认 `pretrain_max_steps=2000`，MRA 占 1/5 ≈ 400 步）与 K=8 无放回采样：**每 rank 判别器把同一条 MRA latent 反复看到 ~229 遍**。现状判别器零增强（无 augmentation / 噪声注入 / dropout / R1；仅 AdamW weight_decay=1e-4 与默认关闭的 spectral_norm），注定记住「这 14 例的共性」而非「真假 MRA 的分界」——held-out AUC（病例级不相交，MRA 仅 ≈ 16 卷）因分布共性虚高，ADR-0008 的条件白名单 gate 被假阳性骗过、无信号条件上岗。这是 MRA 上最可能的假阳性来源（`research/discriminator-saturation-risk.md` §5.6），比小样本统计问题更危险；ADA（Karras et al., arXiv:2006.06676）给出直接文献支撑：「小数据集上判别器过拟合训练样本，对生成器的反馈退化为只关注少数特征」，增强是标准对策。ADR-0008 裁决 4 已拒有放回采样（小池 bagging 是过拟合加速器），其容量守卫只防「抽不出」、不防「反复抽同一样」——本 ADR 补齐后半，即 ADR-0008 Consequences 末条所立之票的裁决。**决定：主防线 = 判别器训练期对称噪声注入（training-only augmentation），配套 per-condition 过拟合分叉监控（只报警不自动动作），预训练与在线两阶段同一套；不构成对上游 recipe 的偏离。**

**Status**: accepted

## Decision

1. **主防线 = 训练期对称噪声注入**：判别器参数更新的前向中，real 与 fake 两侧 latent 同分布加噪。注入点在 `ChannelNormalizer` 之后（归一化域），σ 以「相对通道 std 的比例」参数化——免依赖 latent 存储域量级，不依赖 K。机制 = 冲掉单样本精确值指纹，逼判别器学平滑特征（ADA 核心洞见）；latent 域噪声注入是 Diffusion-GAN（arXiv:2206.02262）/DMD2（arXiv:2405.14867）一族的简化形态。
2. **对称性语义 = training-only augmentation**：噪声只进判别器参数更新的前向；reward 打分、held-out AUC、train-side 监控复算全部留在干净域。否决双侧同噪——打分与训练共用 `patch_logits` 路径（含 normalize），注入点后移即打分自动带噪，reward 每步 i.i.d. 抖动直接进 GRPO advantage，组内标准化把噪声方差放大到与信号同尺度（盲条件 reward std 0.003 的教训）；否决采样平均消抖——rollout 打分成本 ×N。σ 小时分界面对扰动稳健，AUC 作为排序统计可用。
3. **强度形态 = 每样本随机 σ ~ U[0, σ_max]，σ_max = 0.2 暂定**：随机下限 0 为判别器保留干净样本的梯度锚；随机强度提供多尺度指纹冲刷（ADA / Diffusion-GAN 共同形态）。σ_max = 0.2 标注「MR-RATE 预训练曲线校准后定版」（与 ADR-0008 对 0.65 / enter / exit 同一口径），校准路径 = 观察分叉监控与 held-out AUC 曲线对 σ_max 的响应。
4. **在线检测 = per-condition 分叉监控**：分叉 = EMA(train pairwise acc − held-out AUC)，两侧统一干净域。train 侧每判别器步用干净域输入 no_grad 复算一次准确率（`PatchDiscriminator` 2 层、K ≤ 池大小的小批，成本可忽略）——不直接用 loss 伴生量，因其带噪输入使训练批任务天然更难、系统性低估分叉；held-out 侧用现成 per-condition AUC 流（本就在干净域）。两阶段（预训练 driver 与在线 `OnlineUpdate.step`）同一注入点、同一监控、一份 config knobs——不制造预训练/在线口径断层。
5. **报警不动作**：新事件 `overfit_alert` 进事件流（带 modality、分叉值、两侧原始量；事件契约「可扩不可改名」兼容），只报警、人工裁决——不自动移出白名单、不自动调 σ。预训练曲线校准前，任何自动动作的阈值都是盲定；升级项（分叉联动白名单 exit、σ 自适应 = ADA 启发式 p 提升）留校准后另议。
6. **升级路径（不实现，记录触发条件）**：σ_max 校准不足或 MRA 分叉持续越线 → Diffusion-GAN 式 timestep 条件增强（real/fake 同过前向扩散 q(x_t|x₀)，D(x_t, t)）——需判别器输入契约变更（timestep 通道 + 噪声 schedule 重校），改动面大一个量级，本 ADR 不做。
7. **上游一致性口径**：训练期噪声注入**不构成对上游 recipe 的偏离**。上游 recipe 约束的是基座训练与数据管线（强度变换、FOV、resize、VAE 编码）；判别器是 cynosure 自有 RL 组件，上游无此物。增强只作用于判别器的输入 latent，不触碰 VAE 编码前的影像、不动基座权重、不改评测仪器（2.5D FID）——与 ADR-0008「判别器口径按条件化」同类，判别器口径从不在「与上游一致」约束范围内。以本 ADR（0009）承载，非 0008 修订；0008 Consequences 末条「另开 ticket」由本 ADR 闭环。

## Considered Options

- **每卷重复采样硬上限**（滑动窗口内同一 real 条目不得复现）：per-rank MRA 池 ≈ 14 条，硬上限即步数锁——直接卡死 MRA 预训练步数或迫使放弃该条件；且重复本身不可避免，真正的问题是判别器从重复中学到什么，硬上限不回答此问，否决。
- **real:fake 配比按条件调整**：过拟合源于 real 侧多样性不足，不是 fake 太多——配比调整治不了病根，否决。
- **有放回 bagging**：ADR-0008 裁决 4 已否决（过拟合加速器），不因本票重启。
- **训练与打分双侧同噪 / 打分采样平均消抖**：见 Decision 2，分别以 reward 噪声面与 rollout 成本否决。
- **跨病例 latent mixup**：制造非物理的解剖学混合体，判别器可能学到「混合伪影」分界而非真假分界，且 patch 域混合语义不明，否决。
- **R1 / WGAN-GP / 谱归一化当信号源、单向条件扰动**：`research/discriminator-saturation-risk.md` §5.7 已否决（防幻觉 ≠ 造信号；单向扰动制造与质量无关的捷径信号），本票不重启。

## Consequences

- 新 config knob：`disc_noise_sigma_max`（暂定 0.2，注释标「MR-RATE 预训练曲线校准后定版」）。
- `reward/update.py` 判别器输入装配处新增噪声注入（仅参数更新前向）；打分路径 `patch_logits` 不动，`RewardScorer` 契约不变。
- train-side 分叉量：每判别器步一次干净域 no_grad 复算前向（新增，成本可忽略）；分叉 EMA 跨度进 config（与 ADR-0008 的 EMA(AUC) 跨度 8 iter 口径对齐，暂定同值）。
- 事件契约：新增 `overfit_alert` 事件类型（modality、分叉值、train acc、held-out AUC）；现有事件「可扩不可改名」兼容。
- **预训练侧落地顺序依赖**：分叉监控的预训练侧挂点在 ADR-0008 per-condition 步进落地之后（`PretrainEvent` 增 modality 字段为其先决，ADR-0008 Consequences 已列）；在线侧无此前置。
- **BraTS 线同样生效**：训练期增强改变判别器训练动态，BraTS 线旧 run 数字不与新 run 横向比较——ADR-0008 已声明的校准断层之上再叠一层来源，不另设开关回退旧口径。
- CONTEXT.md：新增「判别器训练期噪声注入」「过拟合分叉监控」词条，`overfit_alert` 入事件流词条。
- 地图雾区收窄：「稀疏模态的采样配平策略」中 real 侧已决，仅剩 rollout 条件分布加权。
