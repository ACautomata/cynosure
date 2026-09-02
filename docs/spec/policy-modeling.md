# 策略建模方案：单步 SDE 噪声注入、log-prob 与 granularity 适配

> 本章是整体实施 spec（地图 #2「CTMR Granular-GRPO RL 后训练方案」）的策略建模章节，由 ticket #5 决议产出。终稿由 ticket #9 汇总。reward 侧衔接 `docs/spec/reward-model.md`（ticket #6）。

## 总原则：与基座实际推理行为对齐

policy 的采样分布必须**逐组复刻基座（NV-Generate-CTMR 各阶段推理）的实际生效行为**——CFG 取值、batch 组织、sigma 日程、timestep transform，全部照搬。对齐的是**实际行为，不是 config 字面值**（见「sigma 日程」的 scale 陷阱）。任何偏离都会让 RL 优化一个与推理不一致的分布。

## 范围与输入

- **零依赖原则**：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；网络类（`DiffusionModelUNetMaisi` / `AutoencoderKlMaisi` / `ControlNetMaisi` / `RFlowScheduler`）来自 MONAI 库本身。
- 输入物 = {UNet ckpt, VAE ckpt, 网络配置 JSON, modality_mapping}；跨模态实验加 ControlNet ckpt。
- **latent = `[4, 64, 64, 32]`**（256×256×128 影像体 ÷4）；网络输出 = **velocity**（v-prediction，`v = x0 − noise`）。
- class 条件 = `nn.Embedding(128, 256)` 查表后加到 time embedding；**无条件分支 = 全零 label**（训练期 `ModalityLabelPerturber` 以 prob=0.1 清零 label，使 label 0 实际承担「无条件」语义）。
- `spacing_tensor`（体素间距 ×1e2）恒传（基座 `include_spacing_input=true`）。

## MDP 形式化

- **state** `s_t = (c, t, x_t)`；`t` 取自 RFlowScheduler 变换后的 timesteps（0..1000，1000=纯噪声）。
- **action** `a_t = x_{t+1} ~ π_θ(x_{t+1} | s_t)`——单步 SDE 高斯核的采样输出（velocity 只决定该高斯的均值，不是 action 本身）。
- **条件 c 按组**：
  - 组1（模态标签条件）：`c = (modality label, spacing)`；
  - 组2（跨模态影像条件）：`c = (源影像 latent × scale_factor [ControlNet 条件], modality label, spacing)`——**两个条件都带**；
  - 组3（序贯）：先组1 后组2，各自沿用。

## Policy = 按组对齐基座 CFG 的采样场

| 实验组 | 基座推理 CFG | RL policy 采样场 | log-prob 场 |
|---|---|---|---|
| 组1 模态标签 | **10.0** | **CFG=10 组合场**：batch=2 单次前向 `chunk(2)`，序 [cond, uncond]，uncond=全零 label | 组合场 `v_cfg` |
| 组2 跨模态 | **0.0**（基座代码强制 `cfg==0`） | **裸条件单前向**：frozen base UNet + trainable ControlNet，ControlNet 残差每次前向都参与 | 单前向 velocity |
| 组3 序贯 | 继承各阶段 | 随所训阶段套用上面两行 | — |

- 组1 组合公式与 batch 组织逐字复刻基座：`v_cfg = v_uncond + 10·(v_cond − v_uncond)`。
- **无 KL、无参考模型**（与 reward 章 ADR-0001 一致；参数 EMA 锚为升级项）。

## 单步 SDE 注入 → MONAI RFlowScheduler 映射

### sigma 日程（照搬基座）

- MONAI `RFlowScheduler` **无 sigma 数组**，全程 timestep 域 0..1000（1000=纯噪声）；噪声水平 **`s = t/1000`**（前向加噪 `x_t = (1−s)·x0 + s·noise`，速度目标 `v = x0 − noise`）。
- `set_timesteps(num_inference_steps=30, input_img_size_numel=131072)`，`use_timestep_transform=true`——SD3 式 timestep transform 在 MONAI 内部触发，ratio 由 latent 尺寸决定（`(131072/32³)^(1/3) ≈ 1.587`）。
- ⚠️ **scale 陷阱**：config 里 `"scale": 1.4` 是**死参数**——MONAI 1.5.0 / 1.6.0 / dev 的 `set_timesteps`/`sample_timesteps` 调 `timestep_transform` 时均不传 `scale=`，实际生效 **1.0**。复刻日程必须按实际行为（1.0），照抄 config 字面会让整个日程错位。
- 实操：直接调 MONAI `set_timesteps(30, input_img_size_numel=prod(latent.shape[2:]))` 读 `scheduler.timesteps`；轨迹游标自持（快照 timesteps 防共享调度器被复写、`next_timesteps` 按位取、末位补 0——参照 NV-Generate-CTMR domain 层薄封装）。

### 确定性步（η=0，即 MONAI `step()`）

```
x_{k+1} = x_k + v·Δs，    Δs = s_k − s_{k+1} = (t_k − t_{k+1})/1000 > 0
```

### 单步 SDE（Singular Stochastic Sampling）

**唯一被优化的训练步 k** 用高斯核替换确定步（步序号沿 timesteps 数组，0=最噪端）：

```
g(s_k) = η·√( s_k / (1−s_k) )                       # s_k ≤ s_max 钳制 σ→1 奇异点
μ = x_k·(1 + g²/(2s_k)·Δs) + v_θ·(1 + g²(1−s_k)/(2s_k))·Δs
x_{k+1} = μ + g·√Δs·ε,   ε ~ N(0,I)                 # Δs 同上
```

`v_θ` 按组取 `v_cfg`（组1）或单前向 `v`（组2）。**η=0 时上式精确退化为 MONAI `step()`**——确定性锚点，封装正确性的自检依据。

## log-prob 口径

```
log π_θ(x_{k+1} | s_k) = log N(x_{k+1}; μ, g²·Δs·I)     # 对非 batch 维取均值（沿用 G²RPO）
```

GRPO ratio 的 π_new/π_old 均在**各自组的采样场**上重算：组1 每次 log-prob 评估 = 1 次 batch=2 前向；组2 单前向。组1 的无条件分支与 G 个方向共享同一 `x_k`，`v_uncond` 每次** batch=1 评估一次即可全组复用**，不随条件分支复制。

## 实现接缝（policy 薄封装）

零依赖自实现；参照 NV-Generate-CTMR domain 层的薄封装形态（`DiffusionScheduler` 把全部 step 算术委托 MONAI；`DiffusionModel._unet_output` 是 CFG 组合唯一发生地）：

1. **anchor 轨迹**：同一初始噪声全 ODE（η=0）采出，逐步存下 `x_k`；
2. **第 k 步 SDE 核**替换 `step()`，产生 G 个方向（组内共享 anchor：条件分支一次 forward batch=1 再 repeat G，`v_uncond` 全组一次评估——G²RPO 效率技巧）；
3. 各方向 **ODE 续跑到 x_0**（确定性）；
4. 终点 latent 交 reward model 打分（latent 域、不经 decoder，见 `reward-model.md`）；
5. 组2：**仅训 ControlNet**，base UNet 冻结；`ControlNetMaisi` 每步产出 `(down_block_res_samples, mid_block_res_sample)` 注入 UNet（条件 = 源影像 latent × scale_factor）。

## MGAI 30 步适配

- **Λ（间隔集合）可配置**：`{1,2}` 或 `{1,2,3}`，消融对比，不定死；
- **融合照搬**：各 λ 的 advantage 各自组内标准化后**求和**（对 λ 的 advantage 而非原始 reward 求和，天然缓解不同 λ 的 latent 分数校准差异）；
- **M（被优化的训练步集合）默认 {2..15}**（高噪前半程，跳过 s≈1 奇异端）+ 扫描接口；
- Λ/M 消融的**评估指标**依赖 ticket #8；实际消融执行属施工，超出本地图（spec）范围。

## 超参初始值

| 项 | 值 | 备注 |
|---|---|---|
| 组大小 G | 12 | 实测显存，不够降 6~8 或逐 k 释放轨迹 |
| advantage clamp | ±5 | |
| ratio clip | 1e-4 | 极窄 trust region——无 KL 时的主要稳定器 |
| SDE 噪声强度 η | 0.7 | |
| KL / 参考模型 | 无 | EMA 锚为升级项（ADR-0001） |
| 优化器 | AdamW，lr=2e-6 | bf16 autocast + fp32 master weights |

## 开工前门槛（sanity checks）

1. **噪声注入 sanity**：单步 SDE 扰动后续跑 ODE，样本分布应与原模型一致（验证边缘分布保持；以 η=0 为对照）。
2. **组内 reward 方差检查**：组1（CFG=10 组合场可能压缩 G 个方向的差异）与组2（CFG=0）各自确认 G 个方向的 reward std 足够非零——raw-logit reward（`reward-model.md`）依赖组内方差，std→0 则 advantage 失效（1e-8 保护 + clamp 兜底）。判别器分辨率侧的对应监控见 `reward-model.md`「防 reward hacking」。

## 待定 / 移交

- G 的显存实测、逐 k 释放与 rollout 吞吐 → ticket #7（编排 / profile）。
- η / G / Λ / M 的最终取值与扫描、组3 序贯衔接（RL 后新 base 与已训 ControlNet 的匹配）→ ticket #8（实验设计与验收）。
- 本章公式落码后的数值对照验证（η=0 锚 + 与基座采样的 parity 测试）→ 施工阶段。

## 依据

- MONAI 1.6.0 `monai/networks/schedulers/rectified_flow.py`（RFlowScheduler：无 sigma、timestep transform、纯 Euler `step()`、`V_PREDICTION`、scale 死参数）与 `monai/apps/generation/maisi/networks/diffusion_model_unet_maisi.py`（forward 签名、`nn.Embedding(128,256)`、ControlNet residual 注入口）。
- NV-Generate-CTMR（只读参照）：`src/ctmr/infrastructure/maisi_engine/diff_model_infer.py:172-219`（30 步主循环、CFG `chunk(2)`）、`src/ctmr/domain/generation/scheduler.py:48-116`（轨迹游标薄封装）、`src/ctmr/domain/generation/model.py:191-217`（`_unet_output` CFG、sample 主循环）、`src/ctmr/domain/generation/bypass.py:46-62`（ControlNet 残差）、`configs/config_network_rflow.json:134-141`（scheduler 配置、`scale:1.4` 死参数）、`configs/config_p3_controlnet_infer.json` 与 `cross_modal/candidate.py:133-136`（组2 CFG=0 强制）、`modality_label/token_swap_sampling.py:69-70`（组1 CFG=10）。
- Granular-GRPO 调研 `research/granular-grpo.md`（Singular Stochastic Sampling、单步高斯核 Eq.2-3、MGAI、G=12、clip 1e-4、η=0.7、无 KL/参考模型）。
- reward 侧衔接 `docs/spec/reward-model.md`（raw logit 组内分辨率、组内 std 监控、无 KL 一致性）。
