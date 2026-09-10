# 判别器冷启动修复：预训练 warm-start + RM 上岗门槛（修订 ADR-0001）

T12/T13（#56）取证：判别器 held-out AUC 100 iter 全程徘徊 chance 带，RL 信号近噪声。根因判定为**判别器训练量结构性不足**——在线 D:G≈1:1（N_d=1）下每个 RL iteration 判别器只攒 1 个 AdamW step，100 iter ≈ 100 步梯度，从零初始化的 3D PatchGAN 出不了冷启动。决定：**RL 启动前新增判别器离线密集预训练（warm-start），预训练通过 RM 上岗门槛（held-out AUC ≥ 0.65 暂定，chance 带外）才允许进入 RL；RL 期间保留 ADR-0001 的逐 iter 在线更新**。本 ADR 修订 ADR-0001 的「在线从零」部分，其余决策（LSGAN、raw logit、GroupNorm、replay buffer、无 KL/参考模型）不变。

**Status**: accepted

## Considered Options

- **A. 纯冻结（RLHF 式离线 RM）**：预训练后 RL 期间不再更新。否决为默认——本项目 `kl_beta=0`、无参考模型，唯一信任域 `clip_range=1e-4` 是步长约束而非分布约束；无 KL 锚的裸冻结 RM 提供一套固定盲点集合，确定性 policy gradient 必然攻克（Goodhart）。且冻结使 held-out AUC 语义含混：回落既可能是 policy 真进步也可能是 hacking，#55 早停通道的时序前提失效。**保留为武装预案**：hacking 签名（`anchor_eval_reward` 升 + milestone FID 同步恶化）触发时切换「冻结判别器 + EMA 锚」——即「冻结 + 分布锚」的 RLHF 正确移植姿势（EMA 锚为 ADR-0001 预留升级项）。
- **B. 预训练 warm-start + 继续在线更新**：采纳。治根因（步数不足），保住 ADR-0001 的全部 hacking 防线；组内标准化 advantage（单 iteration 单 D 快照打分、scale-invariant）结构性抵消移动靶的大部分危害。预训练无新增数据管线：real = Real sample pool，fake = base policy 量产 rollout（即 replay buffer「base 时期样本」管线的放大）。
- **C. 仅加密在线更新（调 N_d / LR / 容量，#56 候选表）**：不单独采纳——在线节奏下判别器梯度步数被 RL iteration 数钳死，冷启动可能结构性无解；调参可叠加于 B 之上，移交实现 ticket。
- **D. 「RM 不该动」的非平稳性顾虑**：理论上成立（policy gradient 收敛性假设固定 reward），但被 advantage 设计结构性缓解；实测的病是 D 不会打分而非 D 在动。顾虑以预案 A 的形式保留在决策树中。

## Consequences

- ADR-0001 的「在线从零」部分由本 ADR 修订（其状态行已加指针）；CONTEXT.md「Online update」词条同步修订，新增「Warm-start pre-training」「RM readiness gate」词条。
- 编排新增预训练阶段与上岗门槛硬检查（T14 前置 ticket）。
- 门槛阈值 0.65 为暂定值，用预训练曲线校准后定版。
- T12/T13 与后续 run 口径断层：标注「冷启动失败组」，T14 前定版判别器形态。
- 卫生项：判别器 AdamW `weight_decay` 当前隐式取 PyTorch 默认 0.01（policy 侧显式 1e-4），实现时显式对齐。
- 预案 A 的触发判定依赖 milestone FID 语义完好（与 #55 早停通道的交互见原 ticket）。
