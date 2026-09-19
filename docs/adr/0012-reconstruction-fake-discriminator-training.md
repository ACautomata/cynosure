# 同源重构判别器训练：fake 构造换域，取代训练期噪声注入

ADR-0008/0009 链路下判别器以 rollout 终点 latent 为 fake、以训练期对称噪声注入（σ ~ U[0, σ_max]）为过拟合防线；MR-RATE 线判别器预训练的实验结果不佳，维护者裁决旧机制整体不满意。根因：fake（rollout 产物）与 real 的差异混杂「内容分布差」与「生成伪影差」两层，判别器可走内容捷径（记 Real sample pool 病例共性），注入只是输入增强补丁、不改变任务结构。**决定：判别器 fake 构造换域为同源重构体（reconstruction fake）——真实 latent 加噪到被优化步的日程噪声水平、用当时的 policy 以确定性 ODE 去噪回 σ=0，real/fake 同内容配对；判别器输入恒干净域，ADR-0009 的注入决策（1/2/3）随之取消，其监控决策（4/5）保留。**同内容配对使「记病例共性」捷径结构性失效——任务结构上根治而非增强；s 取自被优化步使重构伪影谱贴近 rollout fake，判别力向打分对象（rollout 终点）迁移；机制面净简化：replay buffer 判别器链路、预训练量产 rollout、σ_max 强度校准问题一并退役。

**Status**: accepted

## Decision

1. **fake = 同源重构体**：取 Real sample pool 真实 latent（同批既供 real 侧也供 fake 构造），乘回 policy 工作域后从日程步点 s 加噪（rectified flow 插值 x_s = (1−s)·x + s·noise），用当时的 policy（预训练 = base 冻结权重；在线 = 当前 policy）以 η=0 确定性 ODE（与 anchor 续跑同一 kernel/`continue_to_terminal` 语义）去噪回 σ=0，除回 pool 存储域。real/fake 标准二分类各自独立前向，PatchDiscriminator 单输入契约零改动。
2. **s 的分布**：逐样本均匀抽自该条件 `policy.train_step_indices_m`（{2..15}）的 sigma 日程步点——G2RPO 单步扰动的注入点，「当前采样噪声水平」的语义落点；静态、逐条件从各自 `ConditionSchedules` 日程导出（与 rollout 同锚同源）。无自由强度参数——ADR-0009 的 σ_max 曲线校准问题消解。s=1（纯噪声端）天然排除于 M 之外。
3. **判别器输入恒干净域，注入取消**：`disc_noise_sigma_max` knob、`disc_noise` 命名随机流、`training_patch_logits` 带噪入口退役；ADR-0009 决策 1（主防线）、2（training-only 对称语义）、3（σ_max 形态）superseded。重构构造自身的加噪走新命名随机流（先抽 s 后抽 ε，随续训分片落盘），与训练/评测/AUC 流不交叉。ADR-0009 决策 4/5（per-condition 分叉监控、报警不动作）原样保留——监控与增强正交，分叉监控的 train 侧复算继续干净域同估计量口径。
4. **打分与 advantage 不变**：reward 仍打 rollout (k,λ) 终点与 anchor 终点（pool 存储域）；不做打分侧重构（成本 ×N + reward 随机性，与 ADR-0009 否决采样平均消抖同一逻辑）。在线 per-condition held-out AUC 的 fake 侧维持 rollout 终点 latent——gate 的运行语义 = 判别器对打分对象的分辨力，不变。
5. **gate 判据口径 = recon-AUC**：RM readiness gate 预训练报告以 held-out real 原始 vs 同源重构体为判据（判别器训练任务的 out-of-sample 泛化力），不新增专职 rollout 采样。口径交接：预训练判据（recon-AUC）与在线运行口径（rollout-AUC）不同构、不可跨阶段比较绝对值——准入体检 vs 在岗考核；预训练放行、在线实测重新洗牌白名单是预期行为，动态恢复（EMA 滞回）语义不变。
6. **替换而非并存**：判别器 fake 侧只有重构体（现做现用）；replay buffer 判别器链路整体退役（base 分区 seed_base 量产、recent 分区回放、`replay_current_fraction`、`pretrain_fake_batch`）。rollout latent 只承担打分、advantage、在线 AUC。
7. **两阶段同一原语保持**：预训练 driver 与在线 trainer 经同一 `OnlineUpdate.step` 消费同一组 knobs（ADR-0009 决策 4 的装配缝不动）；重构的 UNet 调用按组采样场语义（组1 CFG 组合场 / 组2 裸条件单前向，#115 各收其职）。
8. **条件匹配采样语义收窄**：real 侧按目标条件过滤 + 容量硬守卫照旧；fake 侧由同源自动匹配（fake 由同批 real 重构而来），回放半区的条件过滤随 buffer 退役消失。

## Considered Options

- **保留注入（σ_max 不变或对齐日程）**：输入增强治标不治本，任务结构不变、内容捷径仍在；维护者明示「Reward Model 不应该输入噪声」，否决。
- **注入 σ 对齐采样日程**（原提案的字面解读之一）：σ_start = 1.0 恒为纯噪声，带状对齐的高噪声端两侧信号全灭；且日程 σ（插值系数）与注入 σ（相对通道 std）两域换算无自然映射，否决。
- **重构体与 rollout fake 并存**：replay 回放的时间结构套不上现做重构；两分布混合稀释同源配对的捷径屏蔽，否决。
- **只预训练换、在线不变**：两阶段构造不同构，warm-start 权重面临分布跳变，与「warmup 和正式训练一致」的诉求相反，否决。
- **打分前把 rollout 终点重构一遍再打分**（彻底统一训练/打分分布）：成本 ×N + reward 注入随机性，与 ADR-0009 否决「采样平均消抖」同一逻辑，否决。
- **预训练 gate 配专职 rollout 采样保口径连续**（每条件 32~64 条纯 anchor ODE 专供 AUC）：口径断裂真实但代价是一次性机制新增；精简裁决取 recon-AUC 判据 + 在线实测重新洗牌，否决。
- **判去噪轨迹中间 latent（timestep 条件化判别器）**：需 PatchDiscriminator 输入契约变更（ADR-0009 决策 6 已列为未实现升级路径），改动面大一个量级，否决。

## Consequences

- **退役面**：`reward.disc_noise_sigma_max`、`disc_noise` 随机流、`training_patch_logits` 带噪入口、ReplayBuffer（base/recent 分区与条件过滤回放）、`reward.replay_current_fraction`、`disc_batch_size_k` 的 current/replay 语义、`reward.pretrain_fake_batch` 量产。CONTEXT.md 删「判别器训练期噪声注入」「Replay buffer」词条。
- **新增面**：同源重构构造原语（real 工作域换算、日程步点加噪、确定性 ODE 续跑、pool 域归位）、recon 构造命名随机流、预训练报告 recon-AUC 口径。
- **核心风险（记录在案）**：判别器训练分布（重构体）≠ 打分分布（rollout 终点），判别力迁移是赌注，靠 s 取自被优化步压差；观测仪器 = 在线 rollout-AUC（失辨触发白名单 exit，保守方向）+ overfit 分叉 + OOD reward 抖动（GRPO 组内标准化放大，盯 reward 分布）。
- **成本曲线**：预训练 fake 由 30 步全 ODE → 平均 ~21 步重构（更便宜）；在线判别器更新新增 ~K×21 次 UNet 前向/更新步（no_grad、bf16，按 `update_interval` 节奏摊薄）。
- **范围边界（stage-2 非目标）**：跨模态阶段（ControlNet）的同源重构需要（源影像, 目标标签）配对条件化，real pool 现无源影像配对——组1/stage-1（模态标签）不受影响；stage-2 到来时 fake 构造另行设计。
- **BraTS 线旧 run 数字与新 run 不横向可比**：fake 构造换域即判别器任务变更，在 ADR-0008/0009 已声明的校准断层之上再叠一层来源。
- 条件闸（`condition_gate_enabled`）语义不受影响：新机制首跑可选择关闭条件闸全条件更新、纯观测攒曲线（run 级 config 决定，非本 ADR 锁定）。
