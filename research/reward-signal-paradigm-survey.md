# 判别器即 reward 的范式调研：在线判别器 RL 后训练扩散模型

> 日期：2026-10-23
> 关联：`research/discriminator-saturation-risk.md`（#70 取证）、`docs/spec/reward-model.md`、`docs/adr/0012-reconstruction-fake-discriminator-training.md`
> 触发故障：train 开始数个 iteration 内判别器 held-out AUC 从 0.86 崩到 0.24（低于随机 = 排序反转）、reward 转负、policy 反向优化、FID 4.83 → 5.63。工作假设：训练分布（同源重构体）≠ 打分分布（rollout 终点）的错配 + raw logit 无界。
> 调研方法：一手来源优先（arXiv 原文、官方代码库），每个论断标注 URL 与章节位置；二手转述不作为依据。

---

## 0. 判定摘要

**五条核心结论，逐条给证据：**

1. **同范式先例存在，且它们的做法与我们 ADR-0012 恰好相反。** 「在线训练的判别器当 reward」最成熟的先例是 GAIL/AIRL 模仿学习一族（判别器在 policy 当前 rollout 上训练、同一样本上算 reward，Ho & Ermon, arXiv:1606.03476 §4、Fu et al., arXiv:1710.11248 §3）；扩散侧有 DRAIL（扩散策略 + 扩散判别器联合在线训，reward = log D − log(1−D)，arXiv:2405.16194）、AdvDMD（在线 DMD2 判别器直接作 GRPO reward，arXiv:2604.28126）、Adv-GRPO（判别头在线对抗共训，arXiv:2511.20256）。**这些工作无一例外让判别器在「policy 当前输出」上训练并打分**；而扩散 RLHF 主线（DPOK/DDPO/D3PO/ReFL/AlignProp/Diffusion-DPO）用的是**冻结的外部 reward model**。我们「训练=同源重构体、打分=rollout 终点」的构造在文献里是离群者——文献里找不到「故意把训练分布与打分分布分开」的先例。

2. **判别器输出 → reward 的变换有成熟族谱，raw logit 无先例。** 文献中出现过的变换：−log(1−D)（GAIL）、log D − log(1−D)（AIRL 的 f−log π、DRAIL）、VDB 瓶颈平滑后的 −log(1−D)（VAIL，arXiv:1810.00821）、Wasserstein critic 值（WAIL，arXiv:1906.08113——raw 线性输出的唯一先例，且**前提是显式 Lipschitz 约束**）。raw logit 直接作 reward 未找到先例。注意一个常被忽略的事实：**LSGAN 的均衡判别器本身有界**（D\* = p_data/(p_data+p_g) ∈ (0,1)，Mao et al., arXiv:1611.04076），raw logit 无界的是「未收敛 D 的瞬态输出」——我们的故障正发生在瞬态区。

3. **单调变换救不了排序反转，这是方向 (e) 裁决的关键。** log D、tanh、sigmoid 都是单调变换，**保持排序不变**；AUC 0.24（反转）是判别器学错了特征方向的问题，有界化只能压幅度漂移与组间噪声放大，不能恢复排序。有界 reward 的文献价值在别处：PAR（sigmoid 中心化 reward，arXiv:2502.18770 Thm 3.1/3.2 证有界 reward 上界化 return 方差）、DreamerV3 symlog（arXiv:2301.04104，重尾 reward 鲁棒）。且 PAR 明载 **sigmoid 类 shaping 在 GRPO 下无明确增益**（critic-free 组内标准化已吸收线性 shaping）——我们的组内标准化设定下，(e) 的收益进一步收窄。

4. **训练/打分分布错配是有名字、有对策的失败模式。** 静态 RM 在 policy 漂移下失效（RM 打分持续升、真实偏好恶化）被 OCRM 形式化（arXiv:2507.15507：分布漂移导致 RM 参数与 policy gradient 估计同时不一致），对策族 = **用 policy 当前样本在线刷新 RM**（OCRM 重要性加权、DIAL arXiv:2501.00911 域不变自适应、迭代 RLHF arXiv:2312.11456 / arXiv:2405.07863 的 on-policy 偏好刷新）。这正是方向 (a) 的文献身份。另：迭代 RLHF 的教训（arXiv:2505.18126）——**早期过优化几乎不可恢复**——与我们「几个 iteration 内崩盘」的时标相容，提示修复窗口在开局。

5. **五个改进方向的裁决一览**（详 §5）：(a) rollout 终点混入 fake 侧 = **强支持**（这就是 GAIL/DRAIL/AdvDMD/Adv-GRPO 的标准做法）；(b) AUC 落 chance 带 → reward 中性化 = **弱-中支持**（机制先例在 GRPO 社区的 zero-variance group masking，无判别器/AUC 门控先例）；(c) spectral norm + label smoothing = **中支持**（SN 是稳定项不造信号；label smoothing 有 GAN 侧证据但只 smoothing real 侧、且会移动最优判别面，对 reward 质量的影响无先例）；(d) 提高更新间隔与批量 = **拆开裁决：批量↑强、间隔↑与文献方向相反**（TTUR/Adv-GRPO 10:1/AdvDMD 5:1 全部是判别器更快）；(e) 有界变换 = **中支持但对当前故障不对症**（治幅度不治排序）。

---

## 1. 同范式的先例（调研问题 1）

### 1.1 判别器即 reward：GAIL/AIRL 一族是教科书先例

「判别器在线更新 + 判别器输出即 reward + RL 更新 policy」这个三角结构不是我们的发明，是**模仿学习的主流传输带**：

- **GAIL**（Ho & Ermon, arXiv:1606.03476，§3 算法）：判别器 D(s,a) 在「专家样本 vs policy 当前 rollout」上交替训练（每轮 policy 更新后重新采样 rollout 训 D），policy 用 cost `log(1−D)`（即 reward = −log(1−D)）做 TRPO。**判别器的训练分布 = 打分分布 = policy 当前输出，三者恒同**——这正是我们 ADR-0012 刻意拆开的东西。官方实现默认提供 log D − log(1−D)（logit）变体（HumanCompatibleAI/imitation 的 gail.py 文档，https://imitation.readthedocs.io/en/latest/_modules/imitation/algorithms/adversarial/gail.html）。
- **AIRL**（Fu et al., arXiv:1710.11248，§3）：判别器重参数化 `D = exp(f)/(exp(f) + π)`，reward = `log D − log(1−D) = f − log π`。即「判别器 logit 减 policy log-prob」的 reward 形态出处。实现侧有已知分歧：imitation 库训练用 f 单独、推理用 f（GitHub issue #451，https://github.com/HumanCompatibleAI/imitation/issues/451）——说明该族对「reward 到底取判别器的哪一段」敏感且长期有争议。
- **VAIL**（Peng et al., arXiv:1810.00821，ICLR 2019）：给判别器加变分信息瓶颈（VDB），reward = −log(1−D(E[z|s,a]))。动机直接相关：**无约束判别器会过拟合/给出过于自信的信号，瓶颈把 reward 磨平**；VAIRL 变体明言学出比 AIRL 更平滑的 reward，在 maze 迁移上 AIRL reward 过拟合训练布局（VAIRL 平均 return ~156 vs AIRL ~0）。
- **DAC**（Kostrikov et al., arXiv:1809.02925，ICLR 2019）：指出 GAIL/AIRL reward 的**系统性偏差**（−log(1−D) 永不为负 → 偏向长 episode；absorbing state 处理引入额外偏差），给出无偏 reward 修正。说明「判别器输出直接当 reward」从第一天起就被发现带结构性 bias，变换选择不是无害细节。
- **DRAIL**（Lai et al., arXiv:2405.16194，NeurIPS 2024；官方代码 github.com/NVlabs/DRAIL）：**与本项目结构最接近的控制领域先例**——policy 是扩散模型，判别器是扩散式判别分类器（conditional diffusion U-Net），二者**联合在线交替训练**，reward = log D − log(1−D)，PPO 更新。即「扩散模型 + 在线判别器 reward + RL」三要素齐备的工作。

**判定**：判别器作在线 reward 的先例充分存在；且先例的共性是训练/打分同分布、判别器跟随 policy 当前样本。

### 1.2 扩散生成侧的判别器 reward：AdvDMD 与 Adv-GRPO

- **AdvDMD**（arXiv:2604.28126）：DMD2 的对抗判别器直接 repurposed 为 GRPO 的 reward model，**随蒸馏模型在线更新**（"updated online with the distilled model"）；reward 打在**中间与最终去噪态**上（per-timestep，组内标准化 advantage）；**解耦更新频率**——训练初期优先 DMD loss 与判别器建立基线能力、推迟 GRPO（"we employ a decoupled update frequency for DMD and GRPO training"），实现上 fake score 与判别器每 generator step 联合更新 5 次；靠移动对抗信号防 reward hacking。判别器训练样本 = 当前 student 的生成输出（on-policy）。
- **Adv-GRPO**（arXiv:2511.20256，标注 CVPR 2026）：reference 图作正样本、当前 policy 生成作负样本，判别头在线对抗共训（"iteratively updates both the reward model and the generator"），**D:G 更新比 10:1**；当生成样本平均 reward 超过 reference 时**触发** reward model 的对抗微调。报告 KL 约束反而降低 reward 与视觉质量（与我们的 kl_beta=0 一致）。
- **底座 DMD2**（arXiv:2405.14867）：判别器挂 fake denoiser bottleneck，区分生成样本与真图；正文未报告 spectral norm 或 label smoothing（检索其 HTML 全文未见 R1/SN 字样）。

**判定**：扩散侧先例同样遵循「判别器在 policy 当前输出上训练并打分」。AdvDMD 的两个可借手段（per-timestep reward、解耦/前置的判别器建立期）在 `discriminator-saturation-risk.md` §6.3 已有记载，本报告确认其第一手表述。

### 1.3 对照组：扩散 RLHF 主线的 reward model 全部冻结

DPOK（arXiv:2305.16381，Eq.8 带 KL）、DDPO（arXiv:2305.13301，无 KL，"no general-purpose method for preventing overoptimization"）、ImageReward/ReFL（arXiv:2304.05977）、AlignProp（直接 reward 反传）、Diffusion-DPO（arXiv:2311.12908）——**全部使用训练期冻结的外部 reward**（美学分类器、人类偏好 RM），饱和问题交给 early stopping 或 KL。**「reward model 在线更新」的扩散 RL 在文献里是少数派**，除 §1.2 外检索未见第四家（GAN-RM 一类判别式 reward modeling 工作存在但面向 LLM 偏好标注替代，如 OpenReview 2024 的 GAN-RM，未成扩散 RL 先例）。

### 1.4 LLM 侧：迭代/在线 RM 的成熟实践（协同演化治理的出处）

- **Iterative on-policy RLHF**：Xiong et al., arXiv:2312.11456（ICML 2024，KL 约束下迭代偏好学习 + 理论）；Dong/Xiong et al., arXiv:2405.07863（TMLR 2024，「From Reward Modeling to Online RLHF」实用配方 + RLHFlow 流水线）——RM 随 policy 迭代**用当前 policy 样本重新标注刷新**，是该范式的公认做法。
- **Self-Rewarding LMs**（arXiv:2401.10020）与 **OAIF**（Guo et al., arXiv:2402.04792）：judge 在线给当前 policy 输出打分构造偏好对，迭代 DPO——reward 信号逐轮刷新，专治离线偏好数据的分布漂移（OAIF 明言动机是 on-policy 持续刷新的 AI feedback）。
- **协同演化的风险记录**：arXiv:2505.18126（*Reward Model Overoptimisation in Iterated RLHF*）——**早期迭代就过优化的 policy 后续几乎不可恢复**，重置到 SFT 初始化 vs 沿用末代 policy 的初始化策略决定后续迭代成败。对我们「开局即崩」的故障时标是直接警示。

**判定**：policy/RM 协同演化在 LLM 侧有完整正（§1.4 前段）反（2505.18126）两面记录；扩散+医学影像侧的先例就是 §1.1/1.2 这几家。

---

## 2. 判别器输出 → reward 的变换（调研问题 2）

### 2.1 变换族谱与各自的理论依据

| 变换 | 出处 | 理论依据 / 已知问题 |
|---|---|---|
| reward = −log(1−D)（cost = log(1−D)） | GAIL, arXiv:1606.03476 §3；imitation 库默认 | JS 散度族；**饱和区梯度消失**（Goodfellow et al., arXiv:1406.2661 §3 的 saturating loss 讨论）；**永不为负的 reward bias**（DAC, arXiv:1809.02925：偏向长 episode、absorbing state 偏差） |
| reward = log D − log(1−D)（logit） | AIRL, arXiv:1710.11248 §3；DRAIL, arXiv:2405.16194 | = f − log π，可恢复最优 soft advantage；**log π 项对低概率动作给巨大负 reward**（高方差），Orsini et al.（arXiv:2106.00672）实测 "the logit shift significantly hurts the performance"、与 off-policy SAC 不兼容 |
| reward = −log(1−D(E[z]))，z 过信息瓶颈 | VAIL, arXiv:1810.00821 | 瓶颈约束 I(z;x) ≤ I_c → reward 平滑、防 D 过自信、防梯度消失；VAIRL 学出比 AIRL 更平滑的 reward |
| reward = Wasserstein critic 值 f | WAIL, arXiv:1906.08113；WDAIL, arXiv:2006.03503 | critic/Kantorovich potential 作 reward；**前提是 WGAN 式 Lipschitz 约束**（权重裁剪/梯度惩罚）——raw 输出的唯一先例，且从不裸用 |
| reward = sigmoid(中心化 raw reward) | PAR, arXiv:2502.18770 | 有界（Thm 3.1：上界 return 方差）、logistic 噪声下最小方差无偏 shaping（Thm 3.2）、中心点处斜率最大；**需 reference reward 才有效，零 reference 裸 sigmoid 显著变差**；明载 GRPO 下无明确 hacking 增益 |
| reward 幅度变换 symlog / two-hot | DreamerV3, arXiv:2301.04104 | 重尾/跨域 reward 幅度的鲁棒化；PPO 侧消融（arXiv:2310.17805）：去掉 reward clip 后 symlog 预测显著改善 |

**大规模消融的经验裁决**（Orsini et al., arXiv:2106.00672, NeurIPS 2021，"What Matters for Adversarial Imitation Learning?"）：FAIRL 最差（"much worse than all others"，对超参最敏感，reward clipping 显著救它）；裸 log D 略差（"performs a bit worse than the other two"）；AIRL reward 配 RL 生成的 demos 最好、GAIL 的 −log(1−D) 配人类 demos 更好。**没有一个变换在所有设定下最优——变换选择是与数据分布耦合的经验题。**

### 2.2 raw logit 直接作 reward：无先例，且有两个结构性问题

1. **无界性的真实来源**：LSGAN 均衡判别器有界——`D* = p_data/(p_data+p_g) ∈ (0,1)`（Mao et al., arXiv:1611.04076，Proposition/§3 推导，最优解与 vanilla GAN 相同）。但均衡是极限态；**训练中途的 D 输出不受约束**，raw logit 直接暴露瞬态幅度漂移给 advantage。我们观测到的 reward 转负、幅度异常正是瞬态区行为。
2. **排序才是命门**：任何单调变换（log、tanh、sigmoid、symlog）都不改变排序。我们的故障签名是 AUC < 0.5（反转），变换族谱里**没有任何一个变换以「修复反转排序」为功能**——防反转的手段在别处（在线刷新训练分布，§4；平滑/瓶颈，VAIL；有界+中心化，PAR）。

**raw logit 作 reward 未检索到直接先例**；最接近的是 WGAN critic 裸输出，但那伴随 Lipschitz 硬约束。我们 spec 当初选 raw logit 的理由（组内标准化 scale-invariant、sigmoid 饱和杀组内方差，reward-model.md:25-30）在文献里无对偶证据——文献中的 reward 变换全部把「有界/平滑」当作优点，无人用「无界不饱和」换「组内分辨率」。

---

## 3. 在线 reward model 的防崩手段（调研问题 3）

### 3.1 RM 侧已证实的手段

| 手段 | 出处与结论 | 对我们 |
|---|---|---|
| **RM ensemble** | Costa et al., arXiv:2310.02787（缓解 overoptimization）；Eisenstein et al., arXiv:2312.09244（COLM 2024，标题即结论："Mitigate but Do Not Eliminate"；**pretrain-seed 多样性 > fine-tune-seed 多样性**；失败根因 = RM 欠定，分布漂移下成员分歧大） | 证据强但成本高（N 份判别器）；可作白名单判定的投票面，不消除 hacking |
| **有界/中心化 reward（PAR）** | arXiv:2502.18770：PPO 下 ≥5pt 胜率提升、两 epoch 后仍稳健；**GRPO 下无明确增益** | 部分适用（见 §5(e)） |
| **奖励解耦（ODIN）** | arXiv:2402.07319：双头分解相关/不相关成分，丢弃 length 头 | 思路可迁移（把「重构伪影」当不相关头丢弃），但属新机制 |
| **在线校正（OCRM）** | arXiv:2507.15507：重要性加权迭代校正 RM，无需新标注 | 直接支持方向 (a)，见 §4 |
| **KL / clipped KL** | DPOK arXiv:2305.16381；arXiv:2602.21765（泛化误差分解为采样误差 + reward shift 误差 + KL clip 误差） | 定死约束 kl_beta=0 不动；2602.21765 从反面量化「无 KL 时 reward shift 误差无抵消项」 |
| **冻结期 / early stopping** | DDPO（arXiv:2305.13301）：early stopping 与 KL 惩罚经验等价；Gao et al.（arXiv:2210.10760）：proxy-gold 曲线 | 我们已有 milestone FID + AUC 监控面；「开局即崩」说明冻结期要前置到开局 |
| **迭代节奏教训** | arXiv:2505.18126：早期过优化难恢复、初始化策略决定后续成败 | 开局数 iteration 是黄金修复窗口 |
| **RM 的 EMA** | **未检索到直接先例**。最近邻 = Salimans et al.（arXiv:1606.03498）的 historical averaging（参数偏离历史均值的惩罚，属 G 侧技巧）与 GAN 生成器 EMA（StyleGAN）。我们 spec 的 EMA 锚是自己开的设计，文献无对偶 | 诚实标注：无先例 |

### 3.2 GAN 侧防崩手段在「判别器作 reward」场景的适用性裁决

| 手段 | GAN 侧证据 | 判别器作 reward 场景的证据 |
|---|---|---|
| **TTUR / 双时间尺度** | Heusel et al., arXiv:1706.08500（NeurIPS 2017，局部 Nash 收敛 + FID 出处） | 已被我们 spec 采纳为设计原则（reward-model.md「理论定性」节）；**判别器应更快** |
| **Spectral norm** | Miyato et al., arXiv:1802.05957（稳定训练） | **AdvDMD/DMD2/Adv-GRPO/DRAIL 均未报告使用**。只约束 Lipschitz 上界，不防退化（repo §6.2 已有同判） |
| **R1 / R2** | Meschede et al., arXiv:1801.04406 | 朝常数解收缩，与「制造判别力」方向相反（repo §6.2 裁决不变） |
| **Label smoothing** | Salimans et al., arXiv:1606.03498：**one-sided**（只平滑 real 侧，0→0.9）；明言双边平滑会让 D 行为错误（最优 D 变为 α·p_data/(p_data+p_model)） | **零先例**。对 LSGAN 尤其微妙：LSGAN 均衡 D 本已在 (0,1) 有界，平滑 real 目标到 0.9 会把均衡点系统性拉偏、改变 reward 校准；对「判别器输出作 reward」的影响无人测过 |
| **ADA / 增强** | Karras et al., arXiv:2006.06676（判别器过拟合 → 反馈失真 → 发散；train/held-out 分叉是诊断信号） | 已转化为我们的 overfit 分叉监控（ADR-0009）；**增强对 RM 场景的副作用未验证**（repo §5.7 扰动类方案已有否决项） |

**判定**：GAN 防崩技巧在「判别器作 reward」场景**几乎没有被验证过**——§1.2 的扩散侧先例一个都不报告 SN/smoothing/R1。可照搬的只有 TTUR（方向：D 更快）与 ADA 的过拟合诊断思想。把 SN + label smoothing 当「修复」卖没有文献撑腰，最多当「稳定项」。

---

## 4. 训练/打分分布错配（调研问题 4）

### 4.1 这个失败模式在文献里有名字

- **OCRM**（arXiv:2507.15507）：PPO-RLHF 中 policy 漂移使 "responses generated by the LM no longer resemble the responses seen by the RM during training"，RM 估计不一致导致 policy gradient 不一致——**静态 RM + 移动 policy = 收敛失败（even with unlimited data）**；对策 = 重要性加权迭代校正，无需新标注。
- **DIAL**（arXiv:2501.00911）：off-policy 偏好上训练的 RM 在 policy 输出分布下失准（引 Ziegler et al. 2019）；对策 = 训练期让 RM 适配当前 policy 输出。
- **MetaRM**（AAAI, https://ojs.aaai.org/index.php/AAAI/article/view/34552/36707）：meta-learning 适配漂移分布，免连续新标注。
- **理论面**：arXiv:2602.21765 把泛化误差分解为采样误差 + **reward shift 误差** + KL clip 误差——在无 KL 约束（kl_beta=0）时，reward shift 误差项没有抵消机制，只能靠在 RM 侧消化。
- **欠定性**：Eisenstein et al.（arXiv:2312.09244）：in-distribution 表现相近的 RM 在分布漂移下给出**截然不同**的 reward——RM 的 off-distribution 行为本质不可从训练分布预测。

### 4.2 对策的共识与我们的位置

文献对策收敛到一条：**让 RM 持续见到 policy 当前输出**（on-policy 刷新：迭代偏好学习 2312.11456/2405.07863；OAIF 2402.04792 的在线 judge；OCRM 的加权校正；DRAIL/AdvDMD/Adv-GRPO 的判别器直接吃 policy 当前样本）。

我们的 ADR-0012 当时换域到同源重构体，是为了结构性掐死「判别器记 real 病例共性」的内容捷径（ADR-0012 正文，该捷径在 rollout-fake 时代是实测病灶）。**这是一个真实的两难**：rollout-fake 侧有内容捷径、重构-fake 侧有分布错配；文献的 GAIL 系没有这层矛盾，因为 GAIL 的 real/fake 不同源、判别器本就该比较分布。我们的同内容配对是「防记忆化」的特化设计，代价就是打开了错配之门——s 取自被优化步只是压缩差、不消除差（ADR-0012 Consequences「核心风险」节自认「判别力迁移是赌注」）。

**与观测的对接**：AUC 0.86 → 0.24 的反转与文献机制相容——判别器在重构体上学到的 fake 签名（ODE 重构残差伪影）不等于 rollout 终点的缺陷签名；policy 一开训，rollout 分布离开重构体流形，D 在 rollout 上的排序可以系统性反转（低于随机的 AUC 即「rollout 比 real 更像重构 fake 的反义侧」）。这属于 §4.1 的 reward shift，而非 D「崩溃」。

---

## 5. 对我们五个改进方向的文献裁决（调研问题 5）

### (a) rollout 终点混入判别器 fake 侧 —— **支持（强）**

- 文献身份：这就是 GAIL（1606.03476）、AIRL（1710.11248）、DRAIL（2405.16194）、AdvDMD（2604.28126）、Adv-GRPO（2511.20256）的标准做法——**判别器训练分布 = 打分分布 = policy 当前输出**；OCRM/DIAL/OAIF 从 RM 侧给出同一对策的独立证据。
- 具体风险可控：ADR-0012 防的内容捷径来自「fake 与 real 共享内容」——混入 rollout 终点时保留一部分同源重构 fake（配对监督）即可维持捷径屏蔽，rollout 终点只作为分布对齐的补充负样本。文献无「配对 fake + on-policy fake 混合」的精确先例（无先例点只在这个配方细节上），但两个成分各有强先例。
- 注意：GAIL 系交替更新时 D 与 policy 同步小步移动；我们 N_d=1/iter 的更新节奏下混入 rollout fake 的收益依赖 D 真的吸收新样本——配合 (d) 的批量放大。

### (b) AUC 落 chance 带时 per-condition reward 中性化 —— **支持（弱-中）**

- 机制先例在 GRPO 社区：zero/low-variance 组的 ε-floor 爆炸是已知病（MAD GRPO 诊断、NousResearch/atropos #457 hardening）；**DAPO 的做法 = 直接丢弃/重采零方差组**（GRPO/Dr.GRPO/DAPO 统一视角：alphaxiv 2607.00152「three operations on one number: σ」）。「无信号组不贡献梯度」有社区级共识。
- 无先例的部分：用**外部仪表（per-condition AUC）**而非组内 std 来做门控。按 std 门控会把「弱信号」误判为「无信号」（repo §5.4 已裁决）；按 AUC 门控语义更准但文献中无人这样做过。且我们现行架构下「policy 每 iteration 无条件更新、无 AUC 消费路径」（ADR-0017 门控链退役）——(b) 等于部分恢复门控链，是机制回摆，需要明确记为新决策。
- 中性化的形态（reward 置常数 vs skip policy 更新）文献无偏好证据；skip 更干净（与 DAPO 弃组同构）。

### (c) spectral norm + label smoothing —— **部分支持（中），定位为稳定项而非修复**

- SN：GAN 侧证据强（1802.05957）；判别器作 reward 场景零先例（§1.2 四家都不用）；repo §6.2 裁决不变——只约束 Lipschitz 上界，D 退化时常数解不会被拉回来。**可开，但不指望它修 AUC 反转**。
- Label smoothing：**只支持 one-sided**（1606.03498 明言双边平滑有害，最优 D 变 α·p_data/(p_data+p_model)）；对 LSGAN 的特殊性（均衡 D 本已有界，arXiv:1611.04076）意味着平滑 real 目标会系统性偏移 reward 校准，对组内分辨率的影响无人测过。**若试：只平滑 real 目标、小量（0.9）、并把「reward 分布漂移」加进监控面**。判为中偏弱。

### (d) 提高判别器:policy 更新间隔与批量 —— **拆开裁决：批量↑（强）、间隔↑（弱/反向）**

- 批量↑：ADA（2006.06676）与 StyleGAN 系的大 batch 实践、repo §5.2 的 K=8→32-64 建议方向一致（real 侧独立体数据太少，样本多样性不足）；扩散侧判别器先例的 batch 均远大于我们。**强支持**。
- 间隔↑（D 更新更稀）：**与所有双时间尺度文献相反**——TTUR（1706.08500）要求 D 更快；Adv-GRPO 10:1、AdvDMD 5:1、DMD2 5:1 全是 D 更密。判为弱/不建议按字面执行。
- 文献支持的等价重述：**保持 D 快（或更快），但把「每 iter 一次小更新」改为「每 iter 一次大批量更新 / 每 N iter 一次多步大批量更新」**——总量与方差账不变的前提下提高每次更新的信噪比，并让 D 在每组 rollout 打分前更接近「当前 policy 下的收敛判别器」（Borkar 两时间尺度的设计原则，repo reward-model.md 理论出处节已引）。AdvDMD 的「先建立判别器基线再开 GRPO」就是该原则的工程形态。**中-强支持（按重述版）**。

### (e) reward 改用有界变换（log D 或 tanh）替代 raw logit —— **支持（中），但对本次故障不对症**

- 支持面：PAR（2502.18770，有界 reward 的方差上界 + GRPO 下线性 shaping 无效的定理化观察）、DreamerV3 symlog（2301.04104）、VAIL 平滑（1810.00821）——「有界/平滑 reward 利于稳定」有正面证据链。
- 反对/限制面：(i) **单调变换不改变排序，治不了 AUC 反转**（§2.2）；(ii) PAR 实测 sigmoid 类 shaping 在 GRPO 下无明确 hacking 增益（critic-free 组内标准化已吸收），我们的设定恰是 GRPO；(iii) PAR 的 sigmoid 需要 reference reward 中心点才有效，裸 sigmoid 显著变差——照抄必须带中心化，而中心化的自然取值（anchor/参考 logit）在我们无参考模型约束下需要自定。
- 若做：**tanh(logit·τ) 或 sigmoid((logit−ref)/T)** 都行，定位是「压瞬态幅度漂移 + 防组间噪声经 std 标准化放大」，须在监控上验证 logit 幅度确实在漂——否则是先找药后看病。判为中（作为稳定项）/无先例（作为反转修复）。

---

## 6. 对我们架构的具体建议清单

按「先修错配、再稳信号、后做保险」排序；证据强度标注每条。

| # | 建议 | 对应方向 | 证据强度 |
|---|---|---|---|
| 1 | **fake 侧改为「同源重构体 + rollout 终点」混合负样本**（重构体维持配对监督屏蔽内容捷径，rollout 终点对齐打分分布；比例可从 100%:0 起扫），让判别器训练分布覆盖打分分布 | (a) | **强**（GAIL/DRAIL/AdvDMD/Adv-GRPO 标准做法 + OCRM/DIAL 的 RM 侧独立证据；混合配方的细节无精确先例——中） |
| 2 | **修复窗口前置**：开局若干 iteration 设判别器建立期（per-condition AUC 出带前 policy 不更新或仅微更新）——AdvDMD「先建立判别能力再开 GRPO」+ 迭代 RLHF「早期过优化不可恢复」的双重依据 | (a)/(d) | **中-强** |
| 3 | **判别器单步更新批量放大**（K 32–64 起，real 侧独立样本数优先），可用「每 N iter 一次多步大批量更新」吸收同样的 D 总预算 | (d) | **强**（批量）；**中**（多步化的具体节奏，机制先例 AdvDMD 5:1/Adv-GRPO 10:1 支持 D 更密） |
| 4 | **不要提高 D 更新间隔**（与 TTUR 及全部扩散侧先例反向）；若担心 D 噪声，用 #3 的大批量/多步化而非拉稀 | (d) | **强**（反对） |
| 5 | **per-condition AUC 落 chance 带 → 该条件本 iteration 跳过 policy 更新**（reward 中性化的 skip 形态，与 DAPO 弃组同构）；按条件 AUC 而非组内 std 做门控语义（避免弱信号误判，repo §5.4 裁决）。机制上是 ADR-0017 门控链的部分恢复，须作为新决策记录 | (b) | **弱-中**（GRPO masking 机制先例强；AUC 门控本体无先例） |
| 6 | **spectral norm 可开**（一行成本、方向无害），但定位为稳定项；监控预期：不改善 AUC 反转 | (c) | **中**（GAN 侧强、判别器作 reward 场景零先例） |
| 7 | **label smoothing 缓行**：仅 one-sided（real 目标 0.9）、小步试；对 LSGAN 均衡点的偏移先在小规模预训练曲线上验证 reward 分辨率（组内 std）不退化再上 RL | (c) | **弱**（双边有害有证据；LSGAN+RM 组合零先例） |
| 8 | **有界变换（tanh/sigmoid 中心化）作为幅度保险而非修复**：先核实 logit 幅度漂移的实测（metrics 加 logit 范数统计），确认漂移存在再开；开则带中心化参考（如 anchor logit 滑动均值），不带中心化的裸 sigmoid 有反例 | (e) | **中**（有界化一般性收益）/**无先例**（作为反转修复） |
| 9 | **root cause 优先级**：本次故障的修复组合是 #1 + #2 + #5——错配是病因（§4），反转是症状；(c)/(e) 只动症状层 | 综合 | **强**（§4.1 错配失败模式的文献命名与对策共识） |
| 10 | **记录迭代 RLHF 教训的监控含义**：开局即崩的模式与 2505.18126 相容——把「前 N iteration 的 per-condition AUC/reward/FID」设为最高优先级的监控窗，修复后若再崩，优先查初始化策略而非继续堆稳定项 | 综合 | **中** |

---

## 7. 来源索引与诚实标注

### 7.1 一手来源（arXiv 原文 / 官方库）

- GAIL — Ho & Ermon, arXiv:1606.03476, §3（算法与 cost 定义）
- AIRL — Fu et al., arXiv:1710.11248, §3（D = exp(f)/(exp(f)+π)、reward = f−log π）
- LSGAN — Mao et al., arXiv:1611.04076（最优 D\* = p_data/(p_data+p_g) ∈ (0,1)）
- Improved GAN techniques — Salimans et al., arXiv:1606.03498（one-sided label smoothing；双边平滑的最优 D 偏移）
- VAIL/VDB — Peng et al., arXiv:1810.00821（瓶颈平滑 reward；VAIRL vs AIRL 迁移实验）
- DAC — Kostrikov et al., arXiv:1809.02925（GAIL/AIRL reward bias）
- What Matters for AIL — Orsini et al., arXiv:2106.00672（reward 选择大规模消融；logit shift 与 SAC 不兼容）
- WAIL — Xiao et al., arXiv:1906.08113（critic 值作 reward）；WDAIL — arXiv:2006.03503
- GAN 稳定化：TTUR arXiv:1706.08500；SN arXiv:1802.05957；R1 arXiv:1801.04406；ADA arXiv:2006.06676（均见 repo `discriminator-saturation-risk.md` §6.2 的逐篇适用性裁决）
- DRAIL — arXiv:2405.16194 + github.com/NVlabs/DRAIL（扩散判别器在线联合训练，reward = log D − log(1−D)）
- DMD2 — arXiv:2405.14867（判别头挂 fake denoiser bottleneck；正文未见 R1/SN 报告）
- AdvDMD — arXiv:2604.28126（在线判别器作 GRPO reward、per-timestep、解耦频率、5:1）
- Adv-GRPO — arXiv:2511.20256（10:1 更新比、触发式对抗微调、KL 反效果）
- 扩散 RLHF 冻结 RM 主线：DPOK arXiv:2305.16381、DDPO arXiv:2305.13301、ImageReward/ReFL arXiv:2304.05977、Diffusion-DPO arXiv:2311.12908、AlignProp（ICLR 2024）
- 迭代/在线 RLHF：Xiong et al. arXiv:2312.11456；Dong/Xiong et al. arXiv:2405.07863（RLHFlow）；Self-Rewarding arXiv:2401.10020；OAIF arXiv:2402.04792；SPIN-Diffusion arXiv:2402.10210
- 错配与校正：OCRM arXiv:2507.15507；DIAL arXiv:2501.00911；MetaRM AAAI（ojs.aaai.org/index.php/AAAI/article/view/34552）；Reward shift + clipped KL 理论 arXiv:2602.21765
- Reward hacking 缓解：PAR arXiv:2502.18770（Thm 3.1/3.2、GRPO caveat）；ODIN arXiv:2402.07319；Ensembles arXiv:2310.02787 与 arXiv:2312.09244；Iterated RLHF overoptimisation arXiv:2505.18126；Scaling laws arXiv:2210.10760
- 有界/鲁棒 reward 变换：DreamerV3 arXiv:2301.04104（symlog/two-hot）；PPO 侧消融 arXiv:2310.17805
- GRPO 零方差组：DAPO/Dr.GRPO/GRPO 统一视角 arXiv:2607.00152（alphaxiv）；NousResearch/atropos issue #457
- 实现层分歧记录：imitation 库 GAIL reward 形态（imitation.readthedocs.io，gail.py/airl.html）；AIRL f vs f−logπ 分歧（github.com/HumanCompatibleAI/imitation/issues/451）

### 7.2 未验证 / 存疑（诚实标注）

1. **AdvDMD（2604.28126）与 Adv-GRPO（2511.20256）均为 2026 年预印本**，未经独立复现检验（与 repo §7.3 同一保留）。本报告引用其**机制设计**（在线更新、更新比、触发条件），不引用其绝对性能结论。
2. **DMD2/AdvDMD 的判别器正则化**：DMD2 HTML 全文检索未见 R1/SN 字样；若官方代码（bespokesynth? 实为官方 dmd2 仓库）实际使用 R1，则 §3.2「扩散侧先例不用 R1」的表述需修正——本报告只负搜索到论文正文层面。
3. **「判别器 AUC 低于随机 = 排序反转」的直接文献**：未检索到把「判别器作 reward 时排序反转」作为具名现象的论文。§4.2 的机制叙述（重构伪影签名 ≠ rollout 缺陷签名）是**由一手数据 + §4.1 错配文献外推**，与 repo §7.3-4 的诚实标注一致。
4. **label smoothing 对 LSGAN-均衡点偏移的影响**：理论推导明确（最优 D 变为 α·p/(p+q)），但对「组内 reward 分辨率 → GRPO advantage 质量」的实证影响无人测过，§5(c)/建议 7 的缓行立场是保守推断。
5. **PAR 的 GRPO caveat** 出自其正文实验节（LLM 设定，sigmoid shaping + GRPO）；我们的是图像扩散 + 在线判别器，旁证强度降一档，标注为中。
6. **GAN-RM**（OpenReview 2024，GAN 式判别 reward modeling 替代人工标注）检索到但未纳入：面向 LLM Best-of-N/SFT/DPO 偏好构造，与在线判别器 RL 只有方法学远亲关系。
