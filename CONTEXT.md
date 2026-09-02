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

**Fake sample（伪样本）**:
当前 policy rollout 的去噪输出 latent，作为判别器的「假」。
_Avoid_: 生成样本、负样本

**Online update（在线更新）**:
reward model 随 RL 训练每个 iteration 用新 fake 样本重训，而非一次性预训练。

**Replay buffer（回放缓冲）**:
封顶 FIFO 的 fake latent 存库（base 时期 + 近期），更新判别器时按比例混入，防漂移、防灾难性遗忘。

**Reward hacking（奖励攻击）**:
policy 学会骗过判别器拿高分，而非真正提升样本质量。

**Group（组）/ Advantage（优势）**:
GRPO 中共享同一初始噪声的 G 条 rollout 为一组；advantage 是该组内标准化后的 reward。

**Granularity（粒度）**:
Granular-GRPO 里续跑采样所用的时间步间隔 λ；多粒度（multi-granularity）指多个 λ 的 reward 融合。
_Avoid_: 分辨率、尺度（尺度另有所指，见单/多尺度判别器）
