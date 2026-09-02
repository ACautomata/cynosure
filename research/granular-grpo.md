# Granular-GRPO (G²RPO) 深度调研

- **论文**：*Fine-Grained GRPO for Precise Preference Alignment in Flow Models*（arXiv:2510.01982v3，CVPR 2026）
- **仓库**：https://github.com/bcmi/Granular-GRPO （本地克隆 `/tmp/granular-grpo`，commit `753b2fc`，2026-06-01）
- **一手来源**：论文 HTML 全文（arxiv.org/html/2510.01982v3）、仓库 README、训练脚本 `fastvideo/train_g2rpo_hps.py` / `train_g2rpo_hps_clip.py`、启动脚本 `scripts/finetune/finetune_g2rpo_hps*.sh`

---

## 1. 方法核心："Granular"指什么

**"Granular" 是双重含义：(a) 步级（step-wise）的细密 credit assignment；(b) 多种去噪粒度（multi-granularity）的 reward 评估组合。**

- **Motivation**（论文 §1、§3.2、§3.3）：标准 flow-based GRPO（Flow-GRPO / DanceGRPO）有两个问题：
  1. **稀疏 reward（sparse reward）**：SDE 采样在**每一步**都注入噪声，但 terminal reward 被**均匀广播到所有步**，无法与每步的采样方向精确对齐（"the final reward signal is uniformly assigned to each SDE sampling step"，§1）。
  2. **固定粒度评估不完整（incomplete evaluation）**：每条 denoising 方向只按固定步数续跑到终点，只产生单一粒度的输出图像；不同步数续跑出的图像在细节上存在差异，导致 reward model 打分不一致，单一粒度无法稳健评估方向质量（§3.3）。
- **对策一：Singular Stochastic Sampling（奇异随机采样）**：把随机性**限制在单个优化步**——先确定性 ODE 采到某个中间步 x_k，只在这一步做 SDE 扰动产生 G 个方向，再各自确定性 ODE 续跑到 x_0。于是组内 reward 方差**完全由这一步注入的噪声决定**，terminal reward 与噪声强相关，实现精确的步级归因（§3.2，Eq. 8-9；README Method Overview）。
- **对策二：Multi-Granularity Advantage Integration (MGAI)**：对 SDE 步 k 的每个方向，用多种**间隔采样粒度** Λ={λ}（每 λ 步取一个 sigma，续跑步数不同）分别续跑、解码、打分，各粒度的 advantage 直接**求和融合**（§3.3，Eq. 10-12）。
- 结论：G²RPO 的 "fine-grained" = 步级 reward 归因 + 多粒度 reward 组合，**不是**简单地"每步都打分"（中间步不打分，见 §3 下文）。

## 2. MDP 建模

- **状态**：s_t = (c, t, x_t)，c 为 prompt 条件，x_T ~ N(0,I)，x_0 为最终图像（§3.1）。
- **动作**：a_t 即单步去噪结果 x_{t-1} ~ π_θ(x_{t-1} | x_t, c)（§3.1）。
- **策略与 log-prob**：单步策略是 SDE 离散化产生的高斯核。ODE（Eq. 1, dx_t = v_θ dt）被转为**保边缘分布的 SDE**（Eq. 2，源自 Flow-GRPO/DanceGRPO 的构造）：

  d**x**_t = [v_θ + σ_t²/(2t) · (x_t + (1−t)v_θ)] dt + σ_t d**w**_t，  σ_t = η·√(t/(1−t))

  Euler–Maruyama 离散化（Eq. 3）后，一步动作 x_{t-1} 服从均值为确定性漂移、标准差 σ_t√Δt 的各向同性高斯，log-prob 即该高斯的 log 密度（§3.1 Eq. 2-3）。
- **代码实现**：核心在 `flow_grpo_step()`（`fastvideo/train_g2rpo_hps.py:63-103`）：
  - sigma 日程 `torch.linspace(1,0,T+1)` 经 `sd3_time_shift(shift=3.0)` 移位（`train_g2rpo_hps.py:331-332`，shift 默认 3.0，`:1117-1121`）；
  - `std_dev_t = sqrt(σ/(1−σ))·η`，σ=1 奇异点用 σ_max 钳制（`:76,81`）；
  - 均值 `prev_sample_mean = x·(1+σ²ᵗ/(2σ)·Δt) + v·(1+σ²ᵗ(1−σ)/(2σ))·Δt`（`:83`，即 Eq. 3 漂移项）；
  - `log_prob` 为完整高斯 log 密度再对非 batch 维取均值（`:94-101`）。
- **关键点**：G²RPO **没有**像 DDPO 那样把 DDIM 全局改写成等效 SDE 再逐步加噪；它是"ODE 主干 + 单步 SDE 扰动"。确定性 ODE 步就是 η=0 的特例（`run_anchor_sample_step` / `run_ode_sample_step` 以 `eta=0` 调用 `flow_grpo_step`，`:185,279`），只有被优化的那一步用 η=0.7 注入噪声（`run_sde_sample_step`，`:231`）。Flow-GRPO/DanceGRPO 的等效 SDE 构造只在该单步上被引用。

## 3. Reward 的步级归因

- **机制**：不是给中间步图像打分，也**没有任何 credit-assignment 公式/学习 value 函数**；而是**反事实重采样**——对训练步集合 M（论文取前半程 8 步）中的每个 k：
  1. 从同一初始噪声 ODE 采出 anchor 轨迹，存下每一步的 x_k（`run_anchor_sample_step`，`train_g2rpo_hps.py:146-190`，`:188` 把全程 latent stack 下来）；
  2. 在 x_k 处做单步 SDE 扰动得 x_{k-1}^i（i=1..G，`run_sde_sample_step:192-239`，`prev_sample=None` → 新采噪声）；
  3. 从 x_{k-1}^i 用 ODE 续跑到 x_0（`run_ode_sample_step:241-281`），**VAE 解码成图后由 reward model 打 terminal 分**（`:463-486`）。
- 由于除第 k 步外全组共享同一确定性轨迹，组内 reward 差异唯一来源于 x_k 处注入的噪声，因此 R(x_{0←k}^i) 可以无损归因到第 k 步的方向——这是"步级 dense reward"的真正含义（§3.2）。
- **中间"图像"的来源**：就是真实续跑到 x_0 再解码，不是把带噪中间 latent 直接喂给 reward model。
- **多粒度**：对每个 (i, k)，按 stride λ∈Λ 从 sigma 日程中抽稀后续点续跑（`train_g2rpo_hps.py:440-459`，`suffix = sigma_schedule[eta_step+2::g]`），每个粒度各解码、各打分，得到 per-λ 的 reward。
- 附加：每次迭代还用 anchor 全 ODE 终点解码打分记录训练曲线（eval reward，写入 `hps_reward.txt`，`:607-613`），不参与 loss。

## 4. GRPO 细节（全部经代码核实）

- **组大小**：G = 12（`--num_generations 12`，`finetune_g2rpo_hps.sh`；论文 §4.1）。同组共享初始噪声（`--init_same_noise`，`train_g2rpo_hps.py:356-361,1110-1115`）。
- **Advantage**：对每 (prompt 组, 步 k, 粒度 λ) 各自做组内标准化 (r − mean)/std（std 加 1e-8），再**对 λ 求和**（`train_g2rpo_hps.py:615-630`，对应 Eq. 11）；双 reward 版再把 HPS 与 CLIP score 的 advantage **相加**（`train_g2rpo_hps_clip.py` diff：`samples["advantages"] = hps_advantages + clip_advantages`）。
- **Advantage 截断**：clamp 到 ±5（`--adv_clip_max 5.0`，`:659-661,1128-1133`；论文 "advantage clip ε=5"）。
- **Ratio clip**：PPO 式 clip，但范围极窄——`--clip_range 1e-4`（`:663-671,1122-1127`），即 ratio 被夹在 [0.9999, 1.0001]。论文 Eq. 5-7 写的是标准 1±ε 形式。
- **KL / 参考模型**：**无 KL 正则、无参考模型**（论文：β=0，follow DanceGRPO/MixGRPO；代码中 `sample_reference_model` 名字有误导性——采样用的就是当前 policy 本身，全仓无 second model 拷贝）。
- **每个 k 一次独立梯度步**：更新循环对每个训练步 k 单独 forward→loss.backward→optimizer.step（`:638-687`），即一个数据批次产生 |M|=8 次优化器更新。组内共享 anchor 使第 k 步的 forward 只需 batch=1 再 repeat 成 G（`pred.repeat(num_generations)`，`:647-648`），这是论文 §3.2 宣称的效率技巧。
- **学习率**：2e-6，AdamW（betas 0.9/0.999，weight decay 1e-4），constant_with_warmup，bf16 autocast + fp32 master weights（`finetune_g2rpo_hps.sh`；`train_g2rpo_hps.py:798-817`；论文 §4.1）。
- **batch**：每 rank 每步 1 个 prompt × 12 generations；16 GPU。训练 301 steps（`finetune_g2rpo_hps.sh --max_train_steps 301`）。
- **超参总表**（论文 §4.1 + 启动脚本）：T=16 步采样，η=0.7，Λ={1,2}（脚本；论文 v3 写 {1,2,3}），M=前半程 8 步（脚本 `--eta_step_list 0..7`；论文写 {16,…,9}），shift=3.0。

## 5. 训练框架与编排

- **框架**：FastVideo 仓库的 fork（含 ByteDance 修改头注，`train_g2rpo_hps.py:1-10`），纯 PyTorch + **torch.distributed FSDP**（full sharding，`train_g2rpo_hps.py:763-771`）+ 梯度检查点（`:773-776`）。无 Ray、无 TRL、无 vLLM 式分离推理。启动用 `torchrun` 2 节点 × 8 卡（`finetune_g2rpo_hps.sh`）。
- **流水线组织**：**同卡交替**——同一进程内先 `transformer.eval()` + no_grad 做 anchor/SDE/ODE 采样与 reward 打分，再 `transformer.train()` 做 8 次参数更新；无独立 rollout worker，无参数服务器。每 k 更新后有 `dist.barrier()`（`:683-687`）。采样轨迹和 log-prob 以 bf16/fp32 tensor 常驻显存（`all_input_latents`/`all_output_latents`/`all_log_probs`，`:503-506`）。
- **显存管理**：policy（FLUX 12B，fp32 master + FSDP 分片）+ VAE（bf16，`vae.enable_tiling()`）+ reward model（HPSv2 的 CLIP ViT-H-14）**全部常驻同一 GPU**，无 offload（reward model 加载见 `:723-752`；双 reward 版再加一个 DFN5B-CLIP，`:815-822`）。无参考模型副本（省一份大模型显存）。
- **数据**：prompt 的文本嵌入预先离线计算（`scripts/preprocess/preprocess_flux_rl_embeddings.sh`，数据集 `LatentDataset` 直接读 embedding 文件，`dataset/latent_flux_rl_datasets.py`）；prompts.txt 为 HPSv2 的 103,765 条文本 prompt。`scripts/evaluation/` 提供 HPS/CLIP/PickScore/ImageReward/UnifiedReward 评测。

## 6. 基座模型与采样

- **仅 FLUX.1-dev**（rectified-flow / flow-matching 类 MMDiT，12B，720×720，guidance 固定 3.5，`train_g2rpo_hps.py:173-177`）。README 的模型准备一节只有 FLUX；论文 §4.1 基座也只有 Flux.1-dev。仓库内虽残留 SD1.x 的 `pipeline_with_logprob*`/`ddim_with_logprob*` 文件（DDPO 遗留，含 DDIM-with-logprob），但**训练脚本不 import 它们**，实际训练路径只在 FLUX 上。
- **采样步数**：训练时 T=16；推理时用 10/20/38 步评测均不掉点（论文 Table 3）。
- **Reward model**：训练用 HPS-v2.1 或 HPS+CLIP-Score(DFN5B) 组合；评测另加 PickScore、ImageReward、UnifiedReward（域外评测）。

## 7. 其他值得注意的事实

- 论文 Table 1：HPS 训练相对 DanceGRPO 提升 6.52%（HPS 0.353→0.385）；HPS+CLIP 联合训练同时提升 HPS(0.376) 与 CLIP(0.406)。
- MGAI 消融（Table 2）：Λ 从 {1}→{1,2}→{1,2,3} 单调提升，验证多粒度融合有效。
- 开销（§5 Limitation）：MGAI 使训练采样步数从 ~128 增至 184（+45.7%），仅训练期开销，推理延迟不变；且对齐速度显著快于 baseline。
- 代码与论文的小出入：脚本 `granular_list 1 2`（论文 {1,2,3}）；argparse 里 `clip_range` 默认 1e-4（极窄，实际生效值）；脚本存采样图用于调试（hps 版 `:470-473`）。

---

## 对 CTMR 场景的移植清单

目标场景：3D latent rectified-flow 模型（MONAI `RFlowScheduler`，30 步 ODE）；reward 为 MONAI 3D PatchDiscriminator 在 **latent 域**打分；医学影像，无文本 prompt，条件 = 模态标签/掩码/影像。

### 直接可用

| 设计 | 说明与出处 |
|---|---|
| MDP 形式化 s_t=(c,t,x_t), a_t=x_{t-1} | 条件 c 换成 (模态标签, 掩码, 条件影像) 即可，结构不变（§3.1） |
| 单步 SDE 扰动 + 高斯 log-prob（`flow_grpo_step`） | 核心 ~40 行，与具体模型无关；σ 日程换成 MONAI RFlow 的 sigma（注意 MONAI 用 alpha=1−σ 且带 shift，方向要对应：其 `sigmas` 即 t=1−alpha），奇异点钳制写法照搬（`train_g2rpo_hps.py:76-83`） |
| Singular Stochastic Sampling 整体流水线 | anchor ODE 轨迹 → 单步 SDE → ODE 续跑 → terminal 打分 → 归因于该步。条件组内共享 anchor 的一次 forward 技巧（batch=1 再 repeat G，`:647-648`）在"同一体数据、G 个噪声方向"下完全成立 |
| 组内 advantage 标准化 + 多粒度求和 | `train_g2rpo_hps.py:615-630` 可直接复用；30 步 ODE 下 M 可取前 15 步，Λ 取 {1,2,3} 甚至更大（步数多，粒度空间更宽裕） |
| GRPO loss（ratio clip + adv clamp，无 KL 无参考模型） | `:659-671`。省一份参考模型显存，对 3D 大 latent 模型意义重大 |
| 双 reward 加权方式（advantage 相加） | `train_g2rpo_hps_clip.py`：若以后加第二个 reward（如重建 loss/分割指标），同样各自标准化后相加 |
| 训练曲线 eval（anchor 终点打分落盘） | `:607-613`，便于监控对齐进度 |

### 需要改造

| 设计 | 改造点 |
|---|---|
| Reward 打分路径 | G²RPO 在像素域解码后打分（VAE decode per (i,k,λ)，192 次/步）；CTMR 的 discriminator 直接在 latent 域打分——**省去全部 VAE 解码**， but 需确认 PatchDiscriminator 对"早停续跑"latent 的分数校准（不同粒度 λ 的 latent 统计分布不同，必要时 per-λ 归一化 reward，仿照 MGAI 原始做法是对各 λ 的 advantage 而非原始 reward 求和，天然缓解该问题） |
| 条件注入 | 代码里 c 是预计算的 T5/pooled 文本嵌入（`LatentDataset` 读离线 embedding）；CTMR 需改为在线组 batch 的 (模态标签, 掩码, 条件影像)，并处理"掩码/条件影像在组内固定、噪声方向变化"的重复利用 |
| 3D 显存与并行 | FLUX 用 FSDP full-sharding + sp_size=1；3D latent 模型需评估 FSDP 分片 + 序列/切片并行（FastVideo 有 sp 基础设施可借），gradient checkpointing 必须开；G=12 × 3D latent 的采样轨迹显存需实测，可能需降低 G 或逐 k 释放 |
| sigma 日程索引 | G²RPO 用 linspace(1,0)+shift；MONAI RFlowScheduler 有自己的 timesteps/sigmas 与 shift 逻辑，需统一"σ=1 起点"约定并验证 SDE 噪声尺度 σ_t=η√(t/(1−t)) 在 MONAI 参数化下仍保边缘分布（建议先做小样本噪声注入 sanity check：扰动后续跑 ODE 的样本分布应与原模型一致） |
| 训练步 k 的选择 | 论文取前半程（高噪区）；3D 医学 latent 的语义形成步区间可能不同，需扫描（例如 k∈{2..15}） |
| 判别器 reward 的组内方差 | PatchDiscriminator 输出 logits，需确认其分数对微小噪声扰动有足够分辨率（G²RPO 依赖组内 reward std 非零；std 过小会使 advantage 爆炸，注意 1e-8 保护和 adv clamp=5） |

### 不适用

| 设计 | 原因 |
|---|---|
| 文本 prompt 相关全部基础设施 | 离线 T5 嵌入预处理（`preprocess_flux_rl_embeddings.sh`）、HPS/CLIP 文本-图像 reward、tokenizer——CTMR 无文本条件，整体替换 |
| `scripts/inference/infer.py`、VAE 解码-后处理-安全检测链路 | 像素域产物；CTMR 输出即 latent/重建体，评测链路需自建 |
| 残留 SD1.x 的 `pipeline_with_logprob*`/`ddim_with_logprob*` | 训练脚本本就不 import（DDPO 遗留），对 FLUX 与 CTMR 皆无用 |
| 45.7% 量级的多粒度开销预算 | 数值本身仅适用于 16 步 FLUX；30 步 3D 模型需按实际算（大致 ∝ Σ_k⌈k/λ⌉），且单卡上 policy+discriminator 共存时 rollout 吞吐可能是新瓶颈，同卡交替编排是否够快需 profile |
