# Reward model：在线 latent PatchDiscriminator（LSGAN + raw logit，起步不加 KL）

RL 后训练需要一个 reward model 给 policy rollout 的 latent 打分。在大方向已定（MONAI 3D PatchDiscriminator、latent 域打分、real=训练集 VAE 预编码 latent、fake=当前 rollout 去噪输出、在线更新）之上，本章钉死：**损失用 LSGAN、reward 取 raw real-logit（不过 sigmoid）、归一化用 GroupNorm（弃默认 BatchNorm）、在线更新配封顶 FIFO 回放、起步不加 KL/参考模型而以 EMA 锚为升级项**。完整设计见 `docs/spec/reward-model.md`。

**Status**: accepted

## Considered Options

- **损失**：选 LSGAN。弃经典 BCE——sigmoid 概率饱和会杀掉 GRPO 组内 reward 方差（触发 1e-8 保护 / advantage 爆炸）；弃 WGAN-GP——critic 分虽与质量更线性，但要多算 gradient penalty，对在线小判别器不值。
- **reward 取值**：选 raw logit。advantage 是组内 `(r−mean)/std`、scale-invariant，要的是组内分辨率；raw logit 不饱和、持续有区分度。sigmoid 概率有界但饱和，分组内「接近真实」的样本。
- **归一化**：选 GroupNorm。默认 BatchNorm 在「在线小 batch + fake 分布逐 iter 漂移」下 running stats 不稳且泄漏 batch 统计，污染 reward。可选叠 SpectralNorm 稳住在线训练并轻量防 hack。
- **KL/锚定**：起步不加（省一整份 UNet 显存，靠 `clip_range=1e-4` 窄 trust region）。备选「对 base ckpt 的 KL」需常驻参考模型、吃显存；选 **EMA 锚** 作升级项（滑动平均权重、不存完整参考模型）。
- **判别器尺度**：单尺度 vs 多尺度**不定死**，连同深度（`num_layers_d`）、聚合（mean/min）一并作消融轴，用 held-out AUC + 组内 reward std 定胜负。

## Consequences

- 判别器永远在线更新 → 必须配 replay buffer 防灾难性遗忘「明显假」。
- reward 无界（raw logit）→ 依赖组内标准化 + adv clamp ±5；tanh 压 (-1,1) 留作一行保险，logit 幅度持续膨胀时启用。
- 出现 hacking 签名（held-out 判别器 AUC 掉到近 chance 而 eval reward 仍升）→ 触发 EMA 锚。
- 精确更新节奏 N/K、LR、buffer 容量、监控阈值与早停准则依赖 rollout 吞吐 profile 与训练期经验数据，移交 ticket #7 / #8 与对应 fog；实际消融实验执行属施工，超出本地图（spec）范围。
