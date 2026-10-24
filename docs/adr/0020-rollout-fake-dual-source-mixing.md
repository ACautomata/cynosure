# 判别器 fake 双源并存：on-policy rollout 终点按 ρ 混入（修订 ADR-0012 决策 6/7）

**Status**: accepted（#274 方案 A 决策档，#275；实现拆票 #277 任务双型指派 / #278 rollout 供给接线 / #279 事件契约 / #280 pretrain 侧混入 / #281 gauss 三臂验收；本票为 expand 阶段只加不改——ADR 落盘 + ρ 配置缝，ρ=0 默认下全仓行为逐位不变）

## Agreements

- 用户同意：判别器训练的改进目标 = ①判别器训练持续改进或至少保持稳定（held-out AUC 不崩入 chance 带下方）；②FID 持续下降（#274）。
- 用户同意：fake 供给口径从 ADR-0012 决策 6「同源重构替换 replay、替换而非并存」修订为「同源重构与 on-policy rollout 终点按混合比 ρ 并存」；决策 7「两阶段同一原语」的字面随之修订、精神保留。
- 用户同意：pretrain 侧采纳子选项 A-i——预训练 fake 侧按 ρ 混入冻结基座的 rollout 终点（保住两阶段同分布精神）；A-ii（pretrain 保持纯重构、接受在线跳变）记录为被否选项。
- 用户同意：reward 有界变换（log D / tanh / sigmoid）不做——文献裁决单调变换保持排序不变、治幅度不治排序（AUC 0.24 是排序反转），且 PAR 明载 sigmoid 类 shaping 在 GRPO 下无明确增益。
- 用户同意：ρ 的实验校准包络以 0.5 为顶（理由：内容捷径回归风险，见 Consequences）；config schema 只守物理域 [0,1)，>0.5 不 schema 拒绝、由臂设计裁决。
- 用户同意：expand 阶段只加不改——本票仅落 ADR 与 `disc_rollout_fake_ratio` 配置缝，ρ 默认 0.0 时全仓行为与现状逐位一致，行为面实现拆票执行。

## Context

run `20261020-t134-gate-on-r3` 实测（#274 Problem Statement）：train 阶段判别器 held-out AUC 从 iter0 的 0.858 崩至 iter8/9 的 0.23~0.25（**低于随机 = 排序反转**），iter9 触发 `overfit_alert`（divergence_ema 0.21，train_pairwise_acc 0.496）；FID 4.83（iter5）→ 5.63（iter10）同步恶化。关键细节：判别器 train loss 不高但 in-sample pairwise acc 也跌至 ≈0.5——**坍缩先于过拟合**。

ADR-0012 把判别器 fake 从 rollout 终点换域为同源重构体时，其 Consequences 已把核心风险记录在案：「判别器训练分布（重构体）≠ 打分分布（rollout 终点），判别力迁移是赌注」。r3 是该赌注的实锤兑现。文献调研（`research/reward-signal-paradigm-survey.md`，一手来源逐条标注）裁决：本设计是文献离群者——GAIL/AIRL（Ho & Ermon arXiv:1606.03476 §4；Fu et al. arXiv:1710.11248 §3）、DRAIL（arXiv:2405.16194，与本项目结构最接近的扩散+在线判别器+RL 先例）、AdvDMD（arXiv:2604.28126）、Adv-GRPO（arXiv:2511.20256）**无一例外让判别器在 policy 当前输出上训练并打分**；训练/打分分布错配是有名字、有对策的失败模式（OCRM arXiv:2507.15507 形式化「静态 RM 在 policy 漂移下失效」，对策族 = 用 policy 当前样本在线刷新 RM）。迭代 RLHF 的教训（arXiv:2505.18126：早期过优化几乎不可恢复）与「开局数个 iteration 内崩盘」的时标相容——修复窗口在开局。

**决定：判别器 fake 供给从「替换」修订为「按 ρ 并存」——同源重构体保留为 fake 主体（同内容配对的内容捷径防线按 (1−ρ) 比例保留），rollout 终点按混合比 ρ 进入 fake 侧，让判别器在「real vs 当前 policy 输出」这个真实打分任务上受训，训练分布向打分分布对齐。ρ = `disc_rollout_fake_ratio` ∈ [0,1)，默认 0.0 = 现状逐位不变；消融臂由配置显式开启。**

## Decision

1. **fake 双源并存，修订 ADR-0012 决策 6**：判别器窗口任务分 recon / rollout 两型。recon 型保持现状（同源重构体，ADR-0012 决策 1/2 构造原语不动）；rollout 型的 fake 直接取自本 (iteration, 槽) rollout 相已产出的终点 latent（现成数据、零额外 policy 前向）。混合比 ρ = `reward.disc_rollout_fake_ratio`，schema 守物理域 [0,1)（负数无语义；ρ=1 时 fake 侧全是 rollout 终点——内容捷径全面回归且预训练 recon-AUC 判据失真，故为开区间），默认 0.0。ADR-0012 决策 1~5、8（构造原语、s 分布、干净域、打分与 advantage 不变、recon-AUC gate 判据、条件匹配语义）全部不动。
2. **指派确定性**：任务型指派 = (seed, 窗口号, 任务创建序) 的确定性纯函数——每 ⌈1/ρ⌉ 个任务指派一个 rollout 型；零新增 RNG 流，续训重导出语义不变（任务账仍是 (seed, 步号) 的确定性函数）。ρ=0 时任务集与现状逐位一致。指派纯函数与容器不感知 fake 来源（配对记录/桶构造期不变式不破）是预登记测试缝（#274 Testing Decisions 缝 2/3）。
3. **rollout 型供给规则**：fake 选取 = 任务序号的纯函数（如序号 mod 终点数），不消耗任何随机流；条件自动匹配（分配表同源），不引入跨条件污染新通道。时序前提已满足：SlotExample 构造序为 rollout → 打分 → 窗口对装配，终点集在对装配前已落。显存路径沿用 rollout 产物 CPU 暂存、步前 join 后回迁（#220 决议 16 同一路径）。损失、跨卡归约、步末 broadcast 全部不变——改动面收敛在任务层。
4. **pretrain 侧采纳 A-i，修订 ADR-0012 决策 7 字面**：预训练 fake 侧同样按 ρ 混入冻结基座 rollout 终点（rollout 机器现成、权重冻结），保住「预训练与在线同分布」的精神；两阶段仍消费同一 `OnlineUpdate.step` 同一组 knobs，装配缝不动。预训练报告判据的口径细分（recon/rollout 分侧留痕或混合口径记档）随 #280 落地时裁决，本 ADR 不预设。
5. **事件契约可扩**：`disc_update.conditions[*]` 增 `recon_pairs` / `rollout_pairs` 计数（可扩不可改名），REWIND_ACCOUNTING 登记表同步（#279 票）。
6. **观测语义升级**：方案 A 落地后，held-out AUC 再崩的含义从「分布错配伪影」变为「真失效信号」——两种病因首次可分（分布错配通道被 ρ 比例关闭）。
7. **观测仪器延续**：在线 rollout-AUC（失辨触发白名单 exit）、overfit 分叉监控、OOD reward 抖动监控全部延续（ADR-0012 Consequences 观测面不变）；gauss 三臂 ρ ∈ {0, 0.25, 0.5} 的验收判据（AUC 稳于 chance 带上、FID 单调降或平台、分叉 EMA 不越线）见 #281。

## Considered Options

- **A-ii：pretrain 保持纯重构、在线按 ρ 混入（接受两阶段跳变）**：两阶段 fake 构造不同构，warm-start 权重面临分布跳变，与「warmup 和正式训练一致」的诉求相反——ADR-0012 决策 7 当初否决「只预训练换、在线不变」的同一逻辑。否。
- **reward 有界变换（log D / tanh / sigmoid）**：单调变换保持排序不变，AUC 0.24 是判别器学错特征方向的排序反转，有界化只压幅度漂移、不能恢复排序（调研 §0.3、§2.2）；PAR（arXiv:2502.18770）明载 sigmoid 类 shaping 在 GRPO（critic-free 组内标准化）下无明确增益。作为独立消融轴留档（`reward_tanh_bounding` knob 已有），本票不开启。否。
- **全量替换（ρ=1，fake 侧全是 rollout 终点）**：内容捷径全面回归——rollout 终点与 real 不同内容，「记 Real sample pool 病例共性」捷径（ADR-0012 根治对象）重新可用；且预训练 recon-AUC 判据口径失真。否。此为 schema 取开区间 [0,1) 的理由。
- **回到 ADR-0008/0009 的 replay/注入老路**：任务结构上根治路线已被 ADR-0012 裁决，r3 病因是分布错配而非防线不足；注入只改输入增强、不改任务结构。否（沿用 ADR-0012 Considered Options 结论）。

## Consequences

- **新增配置面**：`reward.disc_rollout_fake_ratio`，[0,1)、默认 0.0。expand 阶段只加不改：ρ=0 时任务集、损失、容器、事件流与现状逐位一致（既有测试全套不动是本票验收）。实现拆票：#277（任务双型与确定性指派）/ #278（rollout 型供给接线）/ #280（pretrain 侧混入）/ #279（事件契约扩展）/ #281（gauss 三臂验收）。
- **ρ 实验包络上限 0.5（校准而非 schema 约束）**：rollout 终点与 real 非同内容配对——判别器可重新学「real 病例共性 vs policy 输出」的内容分界，ADR-0012 结构性失效的内容捷径按 ρ 比例回归。ρ ≤ 0.5 保证窗口内多数 fake 仍受同内容配对约束（反捷径压力 (1−ρ) ≥ 0.5 过半）；>0.5 的臂不 schema 拒绝（物理域内），由实验设计裁决，须以在线 AUC 与分叉监控背书。验收三臂取 {0, 0.25, 0.5}（#281）。
- **反内容捷径压力按 (1−ρ) 保留**：real 侧抽取语义不变（窗口起点全局无放回，#274 User Story 6），recon 型任务的同内容配对继续屏蔽病例共性捷径。
- **准备期算力下降**：每个 rollout 型任务省一次 ODE 续跑前向（判别器准备期更便宜），为后续 `disc_batch_size_k` 上调腾出吞吐空间。
- **预训练口径交接再叠一层**：fake 构造随 ρ 变更即判别器任务变更，在 ADR-0012 已声明的「BraTS 线旧 run 数字与新 run 不横向可比」之上再叠来源层；ρ>0 的 run 与 ρ=0 的 run 比较须先对 ρ。
- **gate 判据口径**：ADR-0012 决策 5 的 recon-AUC 判据在 A-i 落地后 fake 侧含 rollout 终点，预训练报告口径细分随 #280 裁决（本 ADR 不预设）；在线运行口径（fake 侧 = 按 ρ 混合）与预训练口径仍守「不同构、不跨阶段比较绝对值」原则。
- **风险记录在案**：ρ>0 引入的第二阶风险——判别器在 rollout 终点上直接过拟合 policy 当前缺陷（on-policy 训练的对抗面），由 overfit 分叉监控 + 白名单 exit 观测；若三臂验收显示 ρ>0 臂 AUC 稳但 FID 不改善，回落方向是 ρ=0 + 预案 B（reward 中性化，#274 Out of Scope 另开）。
- **GLOSSARY.md 词条更新**：本票先行登记「rollout 型任务」术语并修订「Fake sample」「判别器窗口」两词条（均标注实施票未启动 / ρ=0 默认下现状语义逐位成立）；「Online update」「条件匹配采样」词条的 fake 双源口径改写随实现票（#277–#280）落地后修订。
