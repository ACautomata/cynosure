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

- **节奏**：`N=1`（每个 RL iteration 都更新判别器），每批 `K` 小，**D:G 更新比 ≈ 1:1**；优化器 AdamW，LR 1e-5~1e-4（与 policy 同量级）。判别器是几层 3D conv，相对 UNet rollout（30 步 ODE × G 方向）算力可忽略，故「每 iter 更新」几乎免费；真正的约束是 rollout 吞吐（见 ticket #7 编排）。`N/K/LR` 标为 **tunable**，待 profile 后定。
- **fake 缓冲**：封顶 **FIFO 回放缓冲** = base 时期样本（初始冻结 policy 产出）+ 近期 policy 样本，按 **50% 当前 / 50% 回放** 混合采样。real 侧固定训练集 latent，不漂。理由：防止判别器随 policy 变好而**灾难性遗忘**「明显假」长什么样，稳定在线训练、抗漂移（GAN-RL 标准做法；代价仅是显存里存数百~数千个小 latent）。

## KL / 稳定性锚定

- **起步不加 KL、不放参考模型**——忠于 Granular-GRPO（原文 β=0、无参考模型），**省一整份 UNet 显存**（对 3D 大 latent 意义重大）。稳定靠 `clip_range=1e-4` 的极窄 PPO trust region + 在线判别器 + 监控。
- **升级项 = 参数 EMA 锚**：不存完整参考模型，用滑动平均权重做软约束，出现漂移/hacking 时启用。优于「对 base ckpt 的 KL」（后者需常驻一份参考模型，吃显存）。

## 防 reward hacking（design 级）

监控最小信号集：

1. **held-out real vs 当前 fake 的判别器准确率 / AUC**——掉到近 chance（~50%）而 eval-reward 仍在升 = **典型 hacking 签名**；
2. **组内 reward std**——→0 则分辨率丢失；
3. **anchor 终点 eval reward 曲线**（Granular-GRPO 本就落盘），与 (1) 背离即报警。

- **EMA 锚触发点** = (1) 近 chance + eval reward 仍上升。
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
- hacking 监控阈值、早停准则 → ticket #8 + 对应 fog（依赖经验数据）。
- 若将来加第二 reward（重建 loss / 分割指标）→ 各自组内标准化后 advantage 相加（Granular-GRPO 双 reward 做法）。

## 依据

- MONAI `PatchDiscriminator` / `MultiScalePatchDiscriminator` 源码（Pix2PixHD 式，输出 patch logit 图 + 中间特征）。
- NV-Generate-CTMR `configs/config_network_p3.json`（`AutoencoderKlMaisi` `num_channels=[64,128,256]`、`latent_channels=4`、`RFlowScheduler`）与 `config_maisi_diff_model_rflow-{ct,mr-brain}.json`（inference dim `[256,256,128]`）。
- Granular-GRPO 调研 `research/granular-grpo.md`（无 KL/参考模型、`clip_range=1e-4`、advantage 组内标准化 + clamp±5、依赖组内 reward 方差非零）。
