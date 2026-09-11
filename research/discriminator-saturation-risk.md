# 判别器饱和风险：在预训练域上做 RL 的梯度可行性

> 日期：2026-09-11
> 关联：issue #70（本报告）、#67（MR-RATE 上游域 RL 地图）、#56 / #55 / #57 / #60
> 一手数据：GitHub Release `exp/20260908-t12-g1` 与 `exp/20260909-t13-g2` 的 `metrics.jsonl` / `verdict.json` / `rollout_domain_report.json` / `config.json`（集群 run 目录已随 experiment-release 归档，本机可复现下载）

---

## 0. 判定摘要

**核心结论：判别器能提供 RL 梯度的域，恰好等于基座的残差域。在基座已经拟合的条件上，判别器不是「学得慢」，而是任务不可分。**

三条判定，逐条给证据：

1. **BraTS 的失败不是「判别器冷启动」。** 这条叙事（#56、ADR-0007）建立在一个未经检验的前提上。重分析 T12 一手 `metrics.jsonl`：判别器 held-out AUC 按条件分层后，**t1c 的均值是 0.668、峰值 0.773，99% 的事件落在 chance 带外**；T12 的 `verdict.json` 里 `ever_above_gate: true`——**判别器早已越过 ADR-0007 暂定的 0.65 上岗门槛**。「100 iter 全程徘徊 chance 带」是全体条件混采后的聚合假象（全体 62.5% 在带内），不是判别器的性质。

2. **在同一 run、同一判别器、同一训练预算（100 step × lr 5e-5）下，t1c 到 0.77、t1n/t2w/t2f 停在 0.51——瓶颈在任务，不在优化。** 且 t1c 正是基座生成质量最差的条件（T12 milestone：t1c 目标 FID 63.8，其余三条 1.25–2.98；T13 独立复现同一模式：t1c 5.36 最差、其余 0.52–2.14）。**两个 run 独立复现「唯一有判别力信号的条件 = 唯一基座欠佳的条件」。**

3. **盲条件下 policy 收到的不是「零梯度」，而是被组内标准化放大到满幅的噪声梯度。** `MgaiAdvantage._normalize` 除以 `std + 1e-8`（`src/cynosure/grpo/advantage.py:46`），reward 是纯噪声时 normalized advantage 照样是 O(1)，再被 clamp 到 ±5。T12 的后果直接可见：**判别器有信号的 t1c，FID 63.8 → 51.4（改善 −19%）；三条盲条件全部恶化**（t1n 2.18 → 4.62、t2f 2.98 → 3.87、t2w 1.25 → 1.65）。这推翻「无信号 = 无操作」的隐含假设——**盲条件是主动破坏源，不是惰性区。**

**对 MR-RATE 的判定：判别器可以在 MR-RATE 上工作，但只在「基座欠训的条件」上工作；「全模态 RL 后训练整体提升」这个前提本身不成立。** 且 MR-RATE 的稀疏模态（T2w 0.1% / MRA 0.02%）会让现有代码的 real 侧混采口径**必然**退化成模态分类器捷径——这是阻断级缺陷，必须先修（§5.1）。

---

## 1. 一手取证：T12/T13 重分析

### 1.1 数据来源

集群 run 目录本机不可达（SSH 不通），但两跑均已按 experiment-release 归档为 GitHub Release，含完整 `metrics.jsonl`：

```
gh release download exp/20260908-t12-g1 -D /tmp/cyn-t12
gh release download exp/20260909-t13-g2 -D /tmp/cyn-t13
```

两跑各 400 条 `iter` 事件 = 100 iteration × 4 rank，加 2 条 `milestone`（iter 50 / 100）。运行配置（`config.json`）：

| 项 | T12 | T13 | 出处 |
|---|---|---|---|
| 组 | modal-label（CFG=10） | cross-modal（CFG=0） | — |
| `disc_lr` | 5e-5 | 5e-5 | `config.py:432`（区间下沿） |
| `disc_batch_size_k` | 8 | 8 | `config.py:428`（K=4 当前 + 4 回放） |
| `disc_update_interval_n_d` | 1 | 1 | `config.py:423` |
| `replay_buffer_capacity` | 64 | 64 | `config.py:444` |
| `disc_num_layers_d` / 谱归一化 | 2 / 关 | 2 / 关 | `config.py:383,403` |
| `kl_beta` | 0.0 | 0.0 | `config.py:352` |
| **判别器累计梯度步** | **100** | **100** | 100 iter × N_d=1 |

### 1.2 held-out AUC 按条件分层（决定性证据）

| 条件 | n | min | max | mean | sd | 落 ±0.02 带内比例 |
|---|---|---|---|---|---|---|
| **T12 t1c** | 99 | 0.5135 | **0.7728** | **0.6677** | 0.0715 | **1.0%** |
| T12 t1n | 86 | 0.4922 | 0.5430 | 0.5176 | 0.0117 | 58.1% |
| T12 t2w | 93 | 0.4911 | 0.5293 | 0.5123 | 0.0066 | 91.4% |
| T12 t2f | 122 | 0.4936 | 0.5256 | 0.5085 | 0.0071 | 93.4% |
| T13 t1c | 90 | 0.4971 | 0.5829 | 0.5323 | 0.0195 | 26.7% |
| T13 t1n | 116 | 0.4802 | 0.5230 | 0.5020 | 0.0078 | 99.1% |
| T13 t2w | 100 | 0.4960 | 0.5330 | 0.5121 | 0.0079 | 83.0% |
| T13 t2f | 94 | 0.4878 | 0.5268 | 0.5077 | 0.0080 | 93.6% |

T12 t1c 的逐 iteration 轨迹（4 rank 均值）：首段 0.533–0.547 → 中段 0.673–0.707 → 末段 0.745–0.773，**单调爬升**。T13 t1c 同向但幅度小得多（0.50 → 0.55–0.57）。

`verdict.json` 的机器判定与 issue 叙事直接冲突：

```json
// /tmp/cyn-t12/verdict.json
"cold_start": { "auc_mean": 0.5507, "in_band_fraction": 0.625,
                "ever_left_band": true, "ever_above_gate": true }
// /tmp/cyn-t13/verdict.json
"cold_start": { "auc_mean": 0.5127, "in_band_fraction": 0.775,
                "ever_left_band": true, "ever_above_gate": false }
```

**T12 的 `ever_above_gate: true` 是本次取证最重要的单条事实**：取证脚本自己记录并已越过门槛，但该结论没有被消费——#56 与 ADR-0007 的「冷启动未解决」判定沿用了不带条件分层的聚合口径。

### 1.3 判别器 loss 曲线：健康的收敛，不是震荡或发散

T12 判别器 LSGAN 总损失（`src/cynosure/reward/scorer.py:188-190`，`mean((D(real)−1)²) + mean(D(fake)²)`）：

| iter 段 | [0,10) | [20,30) | [40,50) | [60,70) | [80,90) | [90,100) |
|---|---|---|---|---|---|---|
| mean | 5.663 | 2.336 | 1.236 | 0.863 | 0.704 | **0.694** |
| sd | 1.841 | 0.335 | 0.101 | 0.081 | 0.067 | **0.065** |

T13 同形（5.648 → 0.742，末段 sd 极小）。**单调下降、末段标准差 0.065——这是收敛得很干净的判别器，不是「训不动」。**

末段 0.69 的含义：LSGAN 在 real/fake 不可分时的最优常数解 loss = 0.5（D≡0.5）。0.69 说明判别器在 3/4 条件上**已接近「放弃分辨」的常数解**，只在 t1c 上保留残差信号。这与 AUC 分层的结果完全自洽：`loss ≈ 0.69`（几乎盲）与 `t1c AUC = 0.77`（该条件不盲）可以同时为真——loss 被三条盲条件主导。

### 1.4 milestone FID 按条件：与 AUC 一一对应

| run | iter | t1c | t1n | t2f | t2w | 聚合 FID |
|---|---|---|---|---|---|---|
| T12 | 50 | **63.83** | 2.18 | 2.98 | 1.25 | 17.56 |
| T12 | 100 | **51.42** | 4.62 | 3.87 | 1.65 | 15.39 |
| T13 | 50 | **5.36** | 1.47 | 2.14 | 0.52 | 2.37 |
| T13 | 100 | **7.84** | 3.57 | 1.57 | 0.78 | 3.44 |

**两个 run、两个完全不同的组（组1 CFG=10 无条件 / 组2 CFG=0 跨模态），「最差 FID 条件 = 唯一有 AUC 信号条件」的模式独立复现。** 这不是巧合，机制是显然的：判别器测的是「与 real 池可分的差异」；基座已经生成得好的条件，可分差异 ≈ 0。

T12 的 FID 走向还给出因果方向：**唯一有 reward 梯度的 t1c 改善（63.8 → 51.4，−19%），三条盲条件全部恶化（+112% / +30% / +32%）。** 见 §3。

**两跑的 milestone 都被判为 reward hacking，且两跑都未触发早停生效**：

```json
// /tmp/cyn-t12/verdict.json 与 /tmp/cyn-t13/verdict.json，iter 50 与 iter 100 各一条，四条的字段值完全同形
{ "iteration": 50, "hacking_signature": 1.0, "early_stop": true, "early_stop_reason": "reward_hacking" }
```

这条证据的两个含义：

1. `hacking_signature: 1.0` 说明现有监控**确实在工作**——它按 ADR-0001 的信号集（held-out AUC 掉回 chance 带 + intra-group std）判定两跑都在 hacking。**即：现有监控把「判别器盲」这件事正确识别出来了，但没有能力把它与「判别器被错误引导」区分开**，两者都输出同一个 1.0。
2. `early_stop: true` 却跑满了 100 iteration——早停标记被写入 verdict 但没有中止 run。这不影响本次结论，但说明 `early_stop` 字段当前的语义是「事后标注」而非「在线制动」。

### 1.5 域错配（data-preparation.md:36 的旧账）：已修复，不是本次失败的原因

`rollout_domain_report.json`（T12 探针）实测：

```json
{ "mean_global_std": 0.9437,
  "references": { "raw_sampling": 1.003, "scaled": 0.973, "z_mu": 0.48 },
  "verdict": "to_scaled",
  "distances": { "to_scaled": 0.029, "to_raw_sampling": 0.059, "to_z_mu": 0.464 } }
```

- rollout 终点在 **scaled 域**（std 0.9437），与 `scaled` 参照（0.973）距离 0.029；
- Real sample pool 按 `data-preparation.md:36` 契约存 **raw 后验采样 z**（std 1.003，未乘 scale_factor）；
- `RolloutPhase._to_pool_domain`（`src/cynosure/train/rollout.py:335-339`）在打分/入 buffer 前除 `latent_scale_factor`（config 值 0.9697）归位：0.9437 / 0.9697 = **0.973 vs pool 的 1.003，残差 3.0%**。

**判定：域错配已被 `_to_pool_domain` 修复，两侧同域比较成立。** 剩余 3.0% 的全局 std 亏空不是 bug，而是**基座生成样本的真实质量差**（policy 采出的 latent 就是比真实后验采样「平」3%）。

但它引出一个设计层面的副作用，值得单列：**判别器输入前的 per-channel 标准化 + 判别器内部的 GroupNorm，联合抹掉了全局尺度/对比度这条最可靠的 real/fake 统计量。** `ChannelNormalizer.normalize`（`src/cynosure/reward/scorer.py:87-108`）按 pool 统计量归一，real → std 1.0、fake → std 0.973；随后 GroupNorm 又把中间激活的逐样本统计量归一。**结果：一个 3% 的全局尺度差——理论上一步就能学到的信号——被网络结构主动丢弃了。** GroupNorm 是为「在线小 batch 稳定」选的（ADR-0001），但这个选择与「检测全局分布差」是冲突的。

### 1.6 并发缺陷：AUC 条件匹配、判别器训练条件不匹配

| 环节 | 代码 | real 侧口径 |
|---|---|---|
| held-out AUC 测量 | `trainer.py:351-353` → `rewards.py:61-68` → `auc.py:49-74` | **按本 iteration 目标序列过滤** |
| 判别器 online update | `trainer.py:357` → `rewards.py:47` → `update.py:82` | **全池混采（`modality=None`）** |

即：判别器被训练去分辨「**单条件 fake** vs **四序列混合 real**」，却被评测成「单条件 fake vs **同条件 real**」。在 BraTS 四序列大致均衡时，这个不一致尚可容忍；**在 MR-RATE 的 0.1% / 0.02% 稀疏模态下会直接致命**（§4.2）。

`OnlineUpdate.step(current_fakes)` 的签名里**根本没有条件**，`UpdateReport` 也没有——所以修这个不是配置改动，要穿参。

---

## 2. 归因判定

对 #70 提出的四个候选，逐条裁决：

### (a) 判别器容量 / LR / 更新节奏不足 —— **不是主因**

- **反证一（同预算内分化）**：同一判别器、同一 100 步预算、同一 lr 5e-5，t1c 到 0.77。若瓶颈是「步数/容量/LR 不足」，四条条件的曲线应当同向慢爬；实际是**一条起飞、三条贴地**。同一网络在同一 run 内分化 → 瓶颈不在优化侧。
- **反证二（loss 已收敛）**：末段 loss 0.694、段内 sd 0.065，是收敛到近常数解，不是欠拟合的信号（欠拟合会表现为 loss 高位 + 高方差）。
- **量级校核**：AdamW 的累计参数位移上界 ≈ `n_step × lr`（`|m̂/(√v̂+ε)| ≲ 1`）= 100 × 5e-5 = **5×10⁻³**。这确实很小——但 t1c 证明了这点预算**足以**学出 0.77。所以预算小是「天花板不高」，不是「学不到」。

结论：**(a) 是真实的收紧项（100 步 ≪ `pretrain_max_steps` 默认 2000，见 `config.py:474`），但不能解释三条盲条件。**

### (b) warm-start 未真正接入 —— **成立，但只是解释「为何没更早发现」，不解释失败**

- T12 于 2026-09-08 跑，T13 于 2026-09-09 跑；`pretrain` 子命令落地于 commit `ac03308`（2026-09-10，#58/#63），T12/T13 的 `config.json` 里 `pretrain_report_json` 根本不存在（schema 字段当时未引入）。
- **即：T12/T13 是冷启动，warm-start 从未在任何真实数据上跑过。** ADR-0007 的整个修复方案（预训练 + 0.65 门槛）目前仍是**纸面设计，零实测**。
- 但这不是失败的原因——t1c 在冷启动条件下已经过线。真正的意义是：**warm-start（20× 步数）是一个未被执行的判决性实验，而不是一个已知有效的修复。**

### (c) 任务本身不可分 —— **部分成立，且是主导因素；但要精确表述**

不是「全部不可分」，而是：

> **判别器可分辨的部分 = 基座的残差。基座已拟合的条件上，残差 ≈ 0，任务不可分。**

证据是 §1.2 与 §1.4 的双 run 独立复现。这个结论比「不可分」更强也更有用——它给出了**预测规则**：判别器能否工作，可以在跑 RL 之前用「基座在该条件下的 FID / 生成质量」预测。

**必须诚实保留的不确定性**：t1n 的 AUC 在 100 步里从 0.4926 缓慢升到 ~0.53（T12）、t2w/t2f 几乎不动。缓慢上升不能排除「只是学得慢」。**判决性实验是 warm-start：2000 步下若 t1n/t2w/t2f 仍停在 0.51，则 (c) 确认；若升过门槛，则退回 (a)。** 这是 §5.2 把 warm-start 列为第一步的原因——它既是最小改动，也是判决实验。

### (d) 其它 —— **两个真实缺陷，但都不是「AUC 不出带」的原因**

1. **域错配**：已修复（§1.5）。剩余 3% 是真实质量差，不是 bug。
2. **测量口径**：`HeldOutAuc` 的实现本身无可指摘——#56 的取证回放已确认三重可信性（时序无伪影、样本无泄漏、RNG 无交叉），本次重分析确认代码与取证描述一致（`auc.py:49-74` 用 midrank 秩统计，`no_grad` 前向，held-out manifest 有 kind 守卫）。
   **真正的口径问题是聚合层级**：`in_band_fraction`、`auc_mean` 这类全体混采统计量**掩盖了按条件的结构性分化**。这一条是本次「误判为冷启动」的直接原因。
3. **`intra_group_reward_std` 未塌缩但对结论无意义**：`verdict.json` 报 `collapsed: false`（std_min 0.0031 ≫ 1e-8），因此「std 塌缩」被排除。**但这个判定用错了尺子**——std 是否塌缩看的是「相对于 1e-8 保护项」，而真正的问题是「std 里有没有信号」。std = 0.003 在 AUC = 0.51 下就是纯噪声，而 `_normalize` 照样把它放大到 O(1)。见 §3。

### 裁决表

| 候选 | 判定 | 关键证据 |
|---|---|---|
| (a) 容量/LR/节奏 | 真实收紧项，非主因 | 同预算内 t1c=0.77 vs 三条 0.51；loss 已收敛 |
| (b) warm-start 未接入 | 成立（从未跑过），非失败原因 | 时间线 `ac03308`(09-10) 晚于 T12(09-08)/T13(09-09) |
| **(c) 任务不可分** | **主导，但需按条件表述** | 最差 FID 条件 = 唯一有 AUC 信号条件，双 run 复现 |
| (d) 域错配 / 测量口径 | 域已修复；聚合口径是误判来源 | `rollout_domain_report.json`；§1.6 |

---

## 3. 最致命的一条：盲条件提供的是满幅噪声梯度

这不是 #56 的候选方向之一，但它是本次取证里**对实验方向威胁最大**的发现。

**机制**（三行代码）：

```
src/cynosure/grpo/advantage.py:46   (rewards − mean) / (std + 1e-8)
src/cynosure/grpo/advantage.py:40   total = total + normalize(...)     # 跨 λ 求和
src/cynosure/grpo/advantage.py:41   return total.clamp(−5, +5)
```

组内标准化是 **scale-invariant** 的——它把「std 是信号」和「std 是噪声」同等对待。AUC ≈ 0.5 时，组内 reward 的 0.003 标准差没有任何与质量相关的成分，但除以 std 后 advantage 仍是 O(1)，再被 clamp 到 ±5。**于是盲条件给 policy 的不是零梯度，而是与有效条件同量级的随机梯度。**

**T12 的直接观测后果**：

| 条件 | 判别器 AUC | FID@50 → FID@100 | 方向 |
|---|---|---|---|
| t1c | **0.668**（有效） | 63.83 → 51.42 | **改善 −19%** |
| t1n | 0.518（盲） | 2.18 → 4.62 | 恶化 +112% |
| t2f | 0.509（盲） | 2.98 → 3.87 | 恶化 +30% |
| t2w | 0.512（盲） | 1.25 → 1.65 | 恶化 +32% |

（T13 四条件全恶化、t1c 幅度也小，与其 t1c AUC 只到 0.53 一致。）

### 这条发现推翻的两个既有假设

1. **「reward 平坦 ≠ 失败」**（#55 triage 附注、#56 影响节）。在盲条件下 reward 不是「平坦」而是「纯噪声」；而噪声经组内标准化后**有全幅效果**。平坦的 anchor reward 曲线（T12 首→末 0.44 → 0.45，见 `curves_table.md`）恰恰是噪声的签名——一个有效 reward 不会在 100 iter 内毫无趋势。
2. **「AUC 掉回 chance 带 = hacking 签名」**（ADR-0001 监控信号 #1）。该签名假设「AUC 高 = 判别器在工作 = reward 可信」。但 §1.4 显示 AUC 高只说明「有可分差异」，不说明「差异与质量同向」。**当可分差异是「基座特有的伪影/尺度/条件痕迹」时，policy 会沿该方向放大它——AUC 反而上升，FID 恶化。** 这是现有 hacking 监控的结构性盲区：`#55` 修的是「冷判别器误触发」，没覆盖「热判别器错误引导」。§1.4 的四条 `hacking_signature: 1.0` 是这一盲区的实证——监控报出了 hacking，但报不出**是哪一种** hacking，因此也无法给出正确的应对。EMA 锚预案 A 的触发条件（`anchor_eval_reward` 升 + milestone FID 恶化）恰好能覆盖这一条，但目前只是预案、未实现。

---

## 4. MR-RATE 外推

### 4.1 结论

**判别器能在 MR-RATE 上工作，但只在基座欠训的条件上。**「换到 MR-RATE 让整套 RL 流程跑通并整体提升」这个前提**不成立**——它与「基座 `rflow-mr-brain_v1` 就是在 MR-RATE 上训的」直接冲突。

按 §2(c) 的预测规则推演：

| 条件类型 | 基座状态 | 判别器 | 预期 RL 效果 |
|---|---|---|---|
| 基座擅长的常见条件（665k 扫描的主力模态） | 已拟合 | AUC → 0.5，噪声 | **负**（§3 的破坏机制） |
| 稀疏模态（T2w 669 / MRA 157） | 必然欠训 | 有信号 | 正，但受 §4.2 与 §4.3 约束 |
| 跨模态组2（若源-目标对罕见） | 欠训 | 有信号 | 正（t1c 在 T13 已现弱信号） |

**这与 BraTS 的观察完全同构**：t1c 在 BraTS 上「坏」有两种可能来源，MR-RATE 上都会重演——
- **(i) 真实欠训**：条件在训练语料中占比低 → 基座学得差 → 判别器可分 → RL 真能提升。这是**真的增益**。
- **(ii) 标签映射/条件失效**：基座对该 label token 近乎没训过（embedding 停在初始化附近）→ 生成的是「无条件垃圾」→ FID 极端差 → 判别器**极其容易**可分 → 门槛轻松过 → RL 大幅「改善」的其实是「从垃圾恢复到可用」。

**本次取证无法区分 t1c 属于 (i) 还是 (ii)**（本地无 `modality_mapping.json` 与基座的 label 使用统计）。这个区分对 MR-RATE 是**成败攸关**的：若 MRA 是 (ii)，那「MRA 上 RL 巨大提升」是一个标签 bug 的自我修复，不是方法有效性的证据，且论文式的结论会被审稿人一击击穿。

**行动要求**：在 MR-RATE 上跑 warm-start 之前，先做一次廉价的**基座能力基线**——对每个模态条件，用冻结基座采样 N 条、与同条件 real 算 FID（现有 milestone 仪器即可）。**判别器 AUC 应当与该 FID 强相关**（BraTS 上已验证这个关联）。若某条件 FID 极端差（比同批其他条件差一个数量级），先排查标签映射再归因于欠训。

### 4.2 阻断级缺陷：稀疏模态下 real 混采 = 必然的模态分类器捷径

这是 §1.6 的口径不一致在 MR-RATE 上的后果，**它是必然发生的，不是概率事件**：

- fake：每个 iteration 单条件（组1 四序列均匀采样，组2 十二有序对均匀采样）；
- real：全池均匀采样，`modality=None`（`update.py:82`）；
- MR-RATE 池中 MRA 占 **0.02%** → 一个 K=8 的 real 批里期望出现 MRA 的条数是 **0.0016 条**，实际永远是 0。

于是判别器在 MRA iteration 上收到的是「**MRA-fake** vs **非 MRA-real**」。**任务的最优解不是「分辨真假」而是「分辨模态」**——而模态在 latent 域是一个极强的全局信号。判别器一步就能学到它。后果：

1. held-out AUC 会**很高**（因为 MRA-real 与 MRA-fake 同模态、而训练时 real 从未含 MRA…… 实际符号取决于边界落在哪侧）——**门槛会被轻松骗过**；
2. reward 变成「离 MRA 有多远」，policy 被推向**摧毁模态特征**；
3. 这个失败**不会触发任何现有监控**：AUC 高、hacking 签名要求 AUC 掉回带内（#55 的时序前提），两个通道都哑。

**修法（最小）**：`OnlineUpdate.step` 必须接收本批 fake 的条件，real 侧按同一条件过滤采样——即把 `HeldOutAuc` 已有的条件匹配口径（`rewards.py:61-68`）复制到训练侧，保持两侧一致。

### 4.3 稀疏模态上 AUC 的统计支撑

T2w 669 卷、MRA 157 卷。按病例级 70/10/20 划分，held-out 侧 MRA 约 **16 卷**。`auc.py:70-74` 在 patch logit 上做 Mann-Whitney（每卷 16×16×8 = 2048 个 patch），统计量本身不缺样本，**但有效样本量是「卷」而非「patch」**——16 卷上的 AUC 由个案特异性主导，置信区间会非常宽，`pretrain_gate_auc` 的判定在小样本条件上不可靠。

**要求**：按条件报 AUC 时同时报**该条件的 held-out 卷数**，并对小样本条件用 bootstrap CI 代替点估计。这一点在现有 `HeldOutAuc` 接口里没有暴露。

---

## 5. 出路：最小改动方案

约束不变：**在线判别器 reward + 无 KL + 无参考模型**（`policy-modeling.md:35`、`config.py:352-375` 的 `kl_beta` 定死守卫）。以下方案均在此前提下。

### 5.0 优先级

| 级别 | 动作 | 成本 | 理由 |
|---|---|---|---|
| **P0 阻断** | real 侧条件匹配采样 | 小（穿参） | §4.2，MR-RATE 上必然失败 |
| **P0 阻断** | gate 与梯度按条件门控 | 小 | §2(c) + §3，盲条件主动破坏 |
| **P1 判决** | 跑 warm-start 预训练（含按条件 AUC 埋点） | 中（已有实现） | §2 的判决实验；同时是 (a)/(c) 的分离器 |
| **P2 增益** | 稀疏模态配平 + 条件匹配的 batch 组成 | 中 | §4.1 唯一有真实增益的域 |
| **P3 可选** | 噪声级条件判别器 | 大 | §5.5 |

### 5.1 P0：real 侧条件匹配采样（阻断级）

**问题**：`update.py:82` `reals = self._real_sampler.sample(self._batch_size_k)`，无条件。

**改法**：`OnlineUpdate.step(current_fakes, modality)` → `real_sampler.sample(K, modality=modality)`；`RewardCoordinator.update_step` 与 `trainer.py:357` 穿参。`RealPoolSampler.sample` 已支持 `modality=` 过滤（`sampler.py:60-81`），无需新逻辑。

**注意**：预训练路径同样需要——`PretrainDriver.run` 的 fake 批由 `base_partition_samples(batch)` 内部跨条件混合产生并丢弃条件（`rollout.py:309-333` 返回的只有 terminals）。预训练要么改成每步单条件采样，要么把条件一起返回。**这是 warm-start 在 MR-RATE 上跑之前必须先修的东西**，否则预训练出的判别器从第一天起就是一个模态分类器。

**为什么不选 projection discriminator（Miyato & Koyama, arXiv:1802.05637）**：条件判别器确实是文献正解，公式 `f(x,y) = yᵀVφ(x) + ψ(φ(x))` 正是「在条件内比较」的结构化实现。但它比条件匹配 real 采样**改动大得多**（新增 embedding 头、改网络配置契约、改消融锚），而后者已经消掉了 §4.2 的捷径。**建议先做条件匹配；只有当「条件匹配后判别器仍无法在稀疏模态上建立判别力」时，再上 projection 头。** 这符合 #67 已定的「先试现有流程」路线。

### 5.2 P1：warm-start 配方

已有实现（`src/cynosure/pretrain/driver.py`，commit `ac03308`），**但从未跑过**。参数建议：

| 项 | 现默认 | MR-RATE 建议 | 依据 |
|---|---|---|---|
| `pretrain_max_steps` | 2000 | **2000 起步，不要降** | 判决实验需要 20× 于冷启动的步数预算 |
| `pretrain_fake_batch` | 16 | 保持 ≥ `ceil(K × current_fraction)` | 装配期守卫（`driver.py:60-68`）已强制 |
| `disc_lr` | 5e-5 | **1e-4**（区间上沿，`config.py:434` spec 区间 1e-5~1e-4） | 冷启动证据显示 5e-5 在 100 步内位移仅 5e-3；预训练要的是「尽快建立判别力」，取上沿 |
| `disc_batch_size_k` | 8 | **32–64** | K=8（4 real + 4 fake）对 3D PatchGAN 是极小批；patch 级统计量虽多，但 **real 侧只有 4 条独立体数据/步**，样本多样性不足 |
| `disc_num_layers_d` | 2 | 保持 2 | 消融轴，先不动 |
| `spectral_norm_enabled` | off | 保持 off | 见 §5.6 |
| fake 构成 | base fake 库内混采 | **按条件分层，每条件最低配额** | §5.4 |
| real:fake 配比 | 50% 当前 / 50% 回放 | 保持 | ADR-0001 定死（`config.py:493-498`） |

**终止口径必须改**（见 §5.3）：达标判定按条件，而非池化。

### 5.3 P0：RM readiness gate 门槛——从池化改按条件

**现状缺陷（两向都错）**：`driver.py:144` `auc = self._rewards.auc.compute(fakes)` 不带 `modality`，退化为全池混采（`auc.py:56-58` 的 docstring 明确说明预训练走全池口径）。后果：

- **假阳性**：t1c 一条条件到 0.77、三条 0.51，池化 AUC 可能被拉到 0.6+ 而过线——**判别器对 75% 的条件是盲的，却拿到了上岗证**；
- **假阴性**：反过来，若只有少数条件有信号，池化可能压不过 0.65，**把「部分可用」误判为「不可用」**（T12 的 `ever_above_gate: true` 建立在 t1c 峰值上，池化均值 0.5507 是低于门槛的）。

**改法**：

1. `pretrain` 事件的 `heldout_auc` 拆成按条件的字段（或在 `PretrainReport` 里加 `per_condition_auc: dict[Modality, float]`）；
2. gate 判定改为**按条件出报告 + 明确的最低可用条件清单**，而不是单个标量；
3. **0.65 这个数值不要在 MR-RATE 上直接沿用**——它是 BraTS 域上（且事实上是 t1c 单条件上）定的。建议：
   - 保留 0.65 作为「单条件可用」的候选阈值，但用 MR-RATE 自己的预训练曲线校准后定版（`config.py:467-473` 的字段注释本就写明「用预训练曲线校准后定版」）；
   - **新增一个支撑度门槛**：held-out 卷数 < 20 的条件，AUC 判定必须用 bootstrap CI 下界而非点估计（§4.3）。
4. **gate 语义扩展**：门槛的产物不应只是「开跑/拒跑」，而应是**「哪些条件允许进 RL」的白名单**。这正是 §5.4 的输入。

**为什么不直接把门槛降到 0.55 让它过**：那会把 §3 的噪声梯度机制原样放进来，重演 T12 的 FID 恶化。门槛的作用不是「让实验能开跑」，而是「保证开跑的每个条件上 reward 有分辨率」。

### 5.4 P0：盲条件的梯度门控（治 §3）

**改法**：在训练循环里，对**本 iteration 的条件**查 RM readiness 白名单（§5.3 的产物）：

- 条件在白名单（判别器有判别力）→ 正常 `updater.step`；
- 条件不在白名单 → **跳过 policy 更新**（`trainer.py:125` 那条路径），只保留 rollout 与指标落盘。

**为什么这不算违背「纯用我们的方法」**：它没有引入第二重 reward、没有 KL、没有参考模型——它只是**拒绝在奖励模型没有分辨率的样本上做策略梯度**。这在语义上等价于「数据的有效性过滤」，是 GRPO 的既有实践（无效样本不参与 advantage）。而且它是**可逆的**：判别器后续在在线更新中若在该条件上建立判别力（AUC 出带），该条件自动恢复更新。

**替代方案（不推荐作为主路）**：把 advantage 的 `1e-8` 保护换成「std 低于阈值时 advantage 置零」。这更简单，但把「std 小」与「无信号」混为一谈——std 小也可能是有信号但扰动弱（`policy-modeling.md:101-103` 的 sanity check 提到的正是这种情形）。**按条件 AUC 门控比按 std 门控语义更准。**

### 5.5 P3：可分辨性来源清单（按性价比）

按「是否需要改判别器」排序：

| 来源 | 做法 | 是否需要新信号 | 评价 |
|---|---|---|---|
| **1. 条件匹配的 real 采样** | §5.1 | 否（口径修正） | **必做**。这是唯一「零新增、纯修 bug」的来源 |
| **2. 基座残差本身** | 什么也不做，只选对条件 | 否 | 唯一已被实证有效的（BraTS t1c） |
| **3. 稀疏模态的低覆盖率** | 按条件分层 batch（§5.6） | 否 | MR-RATE 上唯一有真实增益的域；但需先排除标签映射问题（§4.1） |
| **4. 噪声级条件判别器** | 对 real/fake **同时**做前向扩散到随机 t，判别器变 `D(y_t, t)` | **是**（改网络输入） | 文献最有力的一条：Diffusion-GAN（arXiv:2206.02262）证明显式地让判别器在噪声水平上工作，能在「原始 JSD 不连续」处恢复有用梯度；DDGAN（arXiv:2112.07804）在 latent 上按 `(x_{t−1}, x_t, t)` 判别。**但它是大改**（改判别器输入契约、改 `OnlineUpdate`、改 `HeldOutAuc`），且现有证据不支持「必做」——先做 1–3 |
| **5. per-timestep reward** | 对中间去噪态打分（AdvDMD, arXiv:2604.28126 的做法） | 是 | 与你现有的 rollout 轨迹天然契合（`StepRollout` 已有 `anchor_latent` 与 `directions`），但会改变 reward 语义与 spec 的「终点打分」口径，属方向性改动 |
| **6. R1 / R2 / 谱归一化** | — | 否 | **明确不用于「制造信号」**，见 §5.7 |
| **7. 条件扰动（ModalityLabelPerturber）** | 对 fake 的 modality token 以 prob 置 unknown | 否 | **不推荐作为主路**，见 §5.7 |

### 5.6 稀疏模态配平方案

**目标**：让判别器在 T2w / MRA 这类稀有条件上也能建立判别力，同时不让常见条件主导 real 侧。

**batch 组成**（配合 §5.1 的条件匹配）：

1. **条件调度**：rollout 的条件分布不应是均匀的。当前 `ModalLabelConditionSampler`（`policy/condition.py:113-130`）对四序列**均匀**采样，`CrossModalConditionSampler`（`:198-223`）对 12 有序对均匀。在 MR-RATE 上这会让 MRA 以 25% 的 iteration 占比出现，而它在 real 池里只有 0.02%——**条件占比与数据占比相差 1250 倍**。建议改为 **sqrt-inverse-frequency** 或直接按「基座在该条件下的欠训程度」加权，并且**让 rollout 的条件分布与 real 侧的采样分布对齐到同一口径**。
2. **real 侧最低配额**：条件匹配采样下，MRA 的 real 条目只有 157 × 70% ≈ 110 条（train split）。K=32 的 real 批若按条件匹配，MRA iteration 会反复抽同样这 110 条 → **判别器会在 MRA 上过拟合到具体病例**。必须配 **real 侧的样本级增强或跨病例混合上限**，否则 held-out AUC 会因「记住了训练病例」而虚高——**这是 MRA 上最可能的假阳性来源，比 §4.3 的统计问题更危险**（held-out 与 train 病例级不相交，但只有 110 条 train 时，判别器学过拟合到「这 110 例的共性」）。
3. **预训练 fake 配额**：`pretrain_fake_batch` 内部按条件分层，保证每条件每步至少有 `max(1, batch // 条件数)` 条，避免均匀采样下 MRA 步数不足。

### 5.7 明确不建议的做法（附理由）

| 做法 | 否决理由 |
|---|---|
| **R1 / R2 正则制造信号** | R1 的目标是**把判别器朝常数解收缩**（目标梯度 0，arXiv:1801.04406 式 (9)）。在 real≈fake、判别器已收敛到近常数（loss 0.69）的区间里，R1 只会**进一步压低** AUC。它是「防幻觉」而非「造信号」（见 §6.2） |
| **WGAN-GP 式梯度惩罚** | 惩罚中心在 1，采样点是 real↔fake 连线上的插值（arXiv:1704.00028）。real≈fake 时这条路径退化成一个点，惩罚会把判别器推向**人为陡峭**的解——正是「用噪声造伪信号」 |
| **谱归一化** | 只约束 Lipschitz **上界**，判别器退化为常数时不会把它拉回来（arXiv:1802.05957）。作为稳定性项无害，但不能当信号源 |
| **用 CFG scale / 采样步数造正负对** | 检索未找到先例，且有反向证据：偏好模型对高 CFG 有系统性偏置（arXiv:2602.22570）。**Diffusion-DPO（arXiv:2311.12908）并未使用 CFG/步数造对**——它用的是 Pick-a-Pic 现成人类偏好对。不要把它当依据 |
| **单向条件扰动（只扰 fake 的 modality label）** | 会制造「unknown label 样本长什么样」的捷径信号，且该信号与质量无关。上游 `ModalityLabelPerturber` 的语义是**训练期条件 dropout**（`policy-modeling.md:14`，prob=0.1，使 label 0 承担无条件语义），是**为了让模型对条件鲁棒**，不是为了让判别器可分。若要借用，必须**对 real 侧施加同分布扰动**，把它变成对称的数据增强而非单向标记。另注：该类在 cynosure 与两个本地 fork clone 中**均无实现**（§7.3），采用等于新写，不是「现成复用」 |
| **纯冻结判别器（ADR-0007 选项 A）作为默认** | 理由未变（无 KL 无参考模型，固定盲点必被确定性 policy gradient 攻克）。但注意 §3 的发现让「冻结 + EMA 锚」预案的**触发条件**（`anchor_eval_reward` 升 + FID 恶化）成为**必备监控**，而非可选 |

---

## 6. 文献依据

### 6.1 判别器在 real/fake 同分布时的行为

- **Goodfellow et al., *Generative Adversarial Nets*, arXiv:1406.2661** — https://ar5iv.labs.arxiv.org/html/1406.2661
  > "For p_g = p_data, D*_G(x) = 1/2"；"**Theorem 1.** The global minimum of the virtual training criterion C(G) is achieved if and only if p_g = p_data. At that point, C(G) achieves the value −log 4."
  适用性：这是 §2(c) 的理论上界情形——p_g = p_data 时最优判别器的决策面是平的，raw logit 的结构只由采样噪声支撑，正是观测到的 AUC 0.5±0.02。**注意**：原文的「梯度消失」讨论针对的是相反方向（G 差时 `log(1−D)` 饱和），并未把 p_g=p_data 与梯度消失写成同一句。

- **Arjovsky & Bottou, arXiv:1701.04862** — https://ar5iv.labs.arxiv.org/html/1701.04862
  > "the optimal discriminator will be perfect and its gradient will be zero almost everywhere"；"**Corollary 2.1**: lim_{‖D−D∗‖→0} ∇θ𝔼[...] = 0."；"This shows that as our discriminator gets better, the gradient of the generator vanishes."
  适用性：方向相反（判别器太强 → 梯度消失），但共同点是判别器层可提供的信息量趋零。**重要**：该文的补救方向是「加噪让分布重叠」（§3 "Towards softer metrics"），与我们要的「制造可分性」正好相反——引用时不要混。

- **Karras et al., *ADA*, arXiv:2006.06676** — https://ar5iv.labs.arxiv.org/html/2006.06676
  > "The key problem with small datasets is that the discriminator overfits to the training examples; its feedback to the generator becomes meaningless and training starts to diverge."；"Without augmentations, the gradients the generator receives from the discriminator become very simplistic over time — the discriminator starts to pay attention to only a handful of features."
  适用性：**这是 §5.6 第 2 条（MRA 只有 ~110 条 train real → 判别器过拟合到病例）的直接文献支撑**，也是「判别器没在学有用东西」的可操作诊断（D 的训练/验证准确率分叉）。

### 6.2 判别器正则化

- **Meschede et al., arXiv:1801.04406** — https://ar5iv.labs.arxiv.org/html/1801.04406
  > Eq.(9) `R₁(ψ) := γ/2 E_{p_D(x)}[‖∇D_ψ(x)‖²]`（只在真数据上）；"the discriminator cannot create a non-zero gradient orthogonal to the data manifold without suffering a loss in the GAN game."
  适用性：R1 的作用是**把判别器朝常数解收缩**，在判别器已收敛到近常数（T12 loss 0.69）的区间会进一步压低 AUC。**不是造信号的手段。** 该文正文未给推荐 γ 数值（只说试了 3 个，数值在补充材料）；可引用的实践值来自 StyleGAN2（arXiv:1912.04958）"γ=10"，但那是**不同论文**，勿混引。

- **Miyato et al., *Spectral Normalization*, arXiv:1802.05957** — https://ar5iv.labs.arxiv.org/html/1802.05957
  > Eq.(8) `W̄_SN(W) := W/σ(W)`；"a novel weight normalization method called spectral normalization that can stabilize the training of discriminator networks"；"Lipschitz constant is the only hyper-parameter to be tuned"
  适用性：约束 Lipschitz **上界**，判别器退化为常数时不会把它拉回来。作为零成本稳定项可以开，但不能期待它造信号。

- **Gulrajani et al., *WGAN-GP*, arXiv:1704.00028** — https://arxiv.org/pdf/1704.00028
  > "We implicitly define P_x̂ sampling uniformly along straight lines between pairs of points sampled from the data distribution P_r and the generator distribution P_g."；"We encourage the norm of the gradient to go towards 1 (two-sided penalty) instead of just staying below 1 (one-sided penalty)."；"All experiments in this paper use λ=10"
  适用性：惩罚中心在 1 且采样于 real↔fake 插值路径。real≈fake 时路径退化成一个点 → 推向人为陡峭的解。**不采用**（与 R1 的中心在 0 形成对照）。

### 6.3 扩散 RL 后训练中的 reward 饱和

- **Black et al., *DDPO*, arXiv:2305.13301** — https://arxiv.org/html/2305.13301
  > "Unlike DPOK, we do not employ KL regularization."；"There is currently no general-purpose method for preventing overoptimization."；"existing solutions, including KL-regularization, may be empirically equivalent to early stopping"；reward 用现成的 `LAION aesthetics predictor` / JPEG 可压缩性 / BERTScore。
  适用性：DDPO 是「固定现成 reward」路线，把饱和问题留给 early stopping。**你的设定（在线判别器 + 无 KL）在 DDPO 里没有先例可抄。**

- **Fan et al., *DPOK*, arXiv:2305.16381** — https://arxiv.org/html/2305.16381
  > Eq.(8) 含 `β Σ_t KL(p_θ(x_{t−1}|x_t,z) ‖ p_pre(x_{t−1}|x_t,z))`；"the model may overfit to the reward and discount the 'skill' of the initial diffusion model"；无 KL 时 "can generate lower-quality images (e.g., over-saturated colors and unnatural shapes)"
  适用性：**注意——KL 不能救盲判别器**。DPOK 的 KL 是防止 policy 跑离基座，不提供判别力。你的 `kl_beta=0` 是 ADR-0001 的定死决策，本报告不主张改它；§5.4 的条件门控是「不加 KL 也能防止噪声驱动」的替代。

- **Xu et al., *ImageReward / ReFL*, arXiv:2304.05977** — https://ar5iv.labs.arxiv.org/html/2304.05977
  > "we re-weight ReFL loss and regularize with pre-training loss."；"if only the gradient of the last denoising step is retained, the training is proved very unstable"
  适用性：**注意更正**：任务书给的 ReFL = arXiv:2304.12824 有误，2304.12824 是 CEP/QGPO。ReFL 出自 ImageReward 论文 §3。ReFL 的 re-weighting（ReLU 截断 reward 分）+ 混入预训练 loss，是「无 KL」路线的替代稳定手段；在你的场景等价于「用基座的自监督去噪 loss 做 anchor」。

- **Wallace et al., *Diffusion-DPO*, arXiv:2311.12908** — https://arxiv.org/html/2311.12908
  > "the objective is to maximize reward while regularizing the KL-divergence from a reference distribution"；数据 "a prompt c and a pairs of images generated from a reference model"（Pick-a-Pic，851,293 对）
  适用性：**明确更正**——Diffusion-DPO **没有**用 CFG/步数构造偏好对，它用的是现成人类偏好对 + 参考模型 KL。不要拿它当「CFG 造对」的依据。

- **Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*, arXiv:2210.10760** — https://arxiv.org/html/2210.10760
  > "optimizing its value too much can hinder ground truth performance, in accordance with Goodhart's law."；RL 拟合 `R_{RL}(d)=d(α_{RL}−β_{RL}log d)`，`d:=√(D_KL(π‖π_init))`；"the effect of the penalty on the gold score is akin to early stopping"
  适用性：你的判别器 = proxy reward，训练集 latent = gold。**该曲线刻画的是「proxy 上升、gold 先升后降」**；T12 的失效模式比这更早——proxy（判别器打分）本身就没有分辨率。这是「reward model 与 policy 同分布」超出该文献覆盖范围的地方。

- **Wang et al., *AdvDMD*, arXiv:2604.28126** — https://arxiv.org/html/2604.28126
  > "AdvDMD employs the adversarially trained discriminator from DMD2 as the reward model"；"It is trained on both intermediate and final states of the denoising process and updated online with the distilled model, thus mitigating reward hacking during training."；"we employ a decoupled update frequency for DMD and GRPO training"；"we prioritize more frequent optimization of the DMD loss at the outset to rapidly establish baseline generation and discrimination capabilities, deferring full GRPO updates until these components are sufficiently stabilized"
  适用性：**与你架构最接近的工程先例**（在线判别器 + GRPO + 无固定 RM），两个手段可借：per-timestep reward、解耦更新频率 + 「先让判别器建立能力再开 GRPO」。**但它没有讨论「同分布导致判别器无信号」**，且它的判别器在有真实 gap（少步蒸馏模型 vs 真图）的场景下训练，成功前提与你不一致。**引用时不要当作「同分布下也能 work」的证据。** 预印本，未经充分复现检验。

- **Mao et al., *Adv-GRPO*, arXiv:2511.20256** — https://arxiv.org/html/2511.20256
  > "10:1 update ratio, meaning that the discriminator is updated for 10 steps for every 1 generator step."；"When the average reward of generated images surpasses that of reference images ... we trigger adversarial fine-tuning of the reward model."
  适用性：它的前提是正样本 = 参考真图分布、负样本 = policy，即**假设了真实 gap**；你那套「触发式对抗微调」在当前场景会一直不触发。**但它的 10:1 更新比对 §5.2 的批量/节奏建议有参考价值。**

- **arXiv:2406.07971, *It Takes Two*（RLHF 饱和现象）** — https://ar5iv.labs.arxiv.org/html/2406.07971
  > "beyond a certain threshold, improvements in the quality of the RM and PM do not translate into increased RLHF performance"
  适用性：**旁证，非同一命题**——它说的是 RM 与 PM 的**不匹配**，不是「RM 与 policy 同分布」。诚实标注。

### 6.4 用扰动制造可分性

- **Wang et al., *Diffusion-GAN*, arXiv:2206.02262** — https://ar5iv.labs.arxiv.org/html/2206.02262
  > "We use the same diffusion process and mixture distribution for both the real samples 𝒙∼p(𝒙) and the generated samples 𝒙g∼pg(𝒙)."；"a smaller t makes it more confident and a larger t makes it more cautious. Thus the diffusion acts like a scale to balance the power of the discriminator."；"the optimal discriminator under the original JS divergence is discontinuous and unattainable. With diffusion-based noise, the optimal discriminator changes with t"；"the black line with t=0 shows the original JSD, which is not even continuous, while as the diffusion level t increments, the lines become smoother and flatter."；"the vanilla GAN exhibits severe mode collapsing ... However, Diffusion-GAN successfully captures all the 25 Gaussian modes and the discriminator is under control to continuously provide useful learning signals."
  适用性：**§5.5 第 4 条（噪声级条件判别器）的直接文献依据**——把「real latent / fake latent」同时过前向扩散到随机 t，判别器变 `D(y_t, t)`；当 t=0 那一个切片不可分时，中间 t 带通常仍存在可分性。**这是「制造可分性」这一族里最成熟的现成方案。**

- **Xiao, Kreis, Vahdat, *DDGAN*, arXiv:2112.07804** — https://ar5iv.labs.arxiv.org/html/2112.07804
  > `D_ϕ(x_{t−1}, x_t, t)`；"takes the N-dimensional x_{t−1} and x_t as inputs, and decides whether x_{t−1} is a plausible denoised version of x_t."；"the true and generator distributions are closer to each other, making the discrimination harder and hence resulting in higher discriminator loss."
  适用性：latent 域 + timestep 条件的判别器是标准工程实践。注意**该文没有「干净样本上判别器会失效」的显式论断**，不要过度引用。

- **Yin et al., *DMD2*, arXiv:2405.14867** — https://ar5iv.labs.arxiv.org/html/2405.14867
  > "where D is the discriminator, and F is the forward diffusion process (i.e., noise injection)"；"we add a classification branch on top of the bottleneck of the fake diffusion denoiser."；"The weight for the GAN loss is set to 3×10^{-3}"
  适用性：判别头挂在去噪网络 bottleneck + 前向扩散注入噪声，是 AdvDMD 的底座。

- **Chen et al., *SimCLR*, arXiv:2002.05709** — https://ar5iv.labs.arxiv.org/html/2002.05709
  > "two correlated views of the same example, denoted x̃_i and x̃_j, which we consider as a positive pair."；"Composition of multiple data augmentation operations is crucial in defining the contrastive prediction tasks that yield effective representations."；"When composing augmentations, the contrastive prediction task becomes harder, but the quality of representation improves dramatically."
  适用性：**最强的类比支撑**——把「不可分的全局分布」重构成「可分的关系型任务」。但方向相反：SimCLR 用增强制造**不变性**（把增强当噪声丢掉），你要用扰动制造**差异性**（把扰动当信号保留）。**类比成立但需要改造，不能直接照搬。**

- **Xie et al., *Guidance Matters*, arXiv:2602.22570** — https://arxiv.org/abs/2602.22570
  > 人类偏好模型 "biased towards large guidance scales"；"simply raising CFG scales can match most of them"（即便图像质量下降）
  适用性：**「CFG 造对」的反向证据**——偏好模型对高 CFG 有系统性偏置，用 CFG 差异造正负对会把偏置当信号学进去。不建议作为主路线。

- **arXiv:2402.10210, *SPIN-Diffusion*** — https://arxiv.org/html/2402.10210
  > "Generate real diffusion trajectories x₁:T ∼ q(x₁:T|x₀)" / "Generate synthetic diffusion trajectories x'₀:T ∼ p_θk(·|c)."；opponent "designed to be previous copies of the main player."；"eliminat[es] the necessity for human preference data"
  适用性：**比「CFG 造对」更值得抄的替代路线**——「旧 checkpoint = loser / 真实数据 = winner」，天然存在可分性（旧 policy 与当前 policy 的差异）且无需 reward model。但注意它与「在线判别器 reward」是不同的方法学路线，**采纳它等于换赛道**，本报告不主张。

- **arXiv:2311.13231, *D3PO*** — https://ar5iv.labs.arxiv.org/html/2311.13231
  > "we also assume that if the segment is preferred, then any state-action pair of the segment is better than the other segment."
  适用性：把终局偏好摊派到每个去噪步，可与 §5.5 第 4 条的 timestep 条件判别器叠加。

---

## 7. 附录

### 7.1 一手证据文件（本机已可复现）

| 文件 | 内容 | 关键发现 |
|---|---|---|
| `/tmp/cyn-t12/metrics.jsonl`（源 release `exp/20260908-t12-g1`） | 400 iter + 2 milestone | t1c AUC 0.6677/峰值 0.7728；其余三条件 0.508–0.518 |
| `/tmp/cyn-t12/verdict.json` | 机器判定 | **`ever_above_gate: true`**；两条 milestone 均 `hacking_signature: 1.0` + `early_stop_reason: "reward_hacking"` |
| `/tmp/cyn-t12/curves_table.md` | 分段统计 | disc loss 5.663 → 0.694（末段 sd 0.065） |
| `/tmp/cyn-t12/rollout_domain_report.json` | 域探针 | rollout std 0.9437 vs pool 1.003 → 残差 3.0% |
| `/tmp/cyn-t12/config.json` | 运行配置 | disc_lr 5e-5、K=8、capacity 64、N_d=1、kl_beta 0 |
| `/tmp/cyn-t13/*`（源 release `exp/20260909-t13-g2`） | 同上 | t1c AUC 0.5323、`ever_above_gate: false`；同一「最差 FID 条件 = 唯一 AUC 信号条件」模式 |

复现命令：

```bash
gh release download exp/20260908-t12-g1 -D /tmp/cyn-t12
gh release download exp/20260909-t13-g2 -D /tmp/cyn-t13
```

> 注：任务书指引的 `runs/20260908-t12-g1/` 与 `runs/20260909-t13-g2/` 在 worktree 与主 checkout 中均**不存在**（`runs/` 被 `.gitignore` 排除），集群 run 目录 SSH 不可达（`sugon` 连接被关闭）。本次取证改走 experiment-release 归档，与集群 run 目录同源。

### 7.2 本地代码引用

| 位置 | 事实 |
|---|---|
| `src/cynosure/grpo/advantage.py:40-46` | 组内标准化除以 `std + 1e-8` 后 clamp ±5——噪声被放大到满幅 |
| `src/cynosure/reward/update.py:82` | `reals = self._real_sampler.sample(self._batch_size_k)`——real 侧无条件，全池混采 |
| `src/cynosure/train/rewards.py:47` vs `:61-68` | `update_step` 无条件；`heldout_auc` 有条件——口径不一致 |
| `src/cynosure/train/trainer.py:351-357` | AUC 传 `record.modality`；`update_step` 不传 |
| `src/cynosure/pretrain/driver.py:144` | `auc = self._rewards.auc.compute(fakes)`——预训练走全池口径 |
| `src/cynosure/train/rollout.py:309-333` | `base_partition_samples` 内部跨条件混合且丢弃条件 |
| `src/cynosure/reward/scorer.py:87-108, 188-190` | ChannelNormalizer + GroupNorm 抹掉全局尺度；LSGAN 两项 |
| `src/cynosure/reward/auc.py:49-74` | patch 级 Mann-Whitney midrank，real 侧按 modality 过滤 |
| `src/cynosure/fixtures.py:97` | 模态映射 t1n/t1c/t2w/t2f → 29/34/30/31 |
| `docs/spec/policy-modeling.md:14` | 上游 `ModalityLabelPerturber` prob=0.1（cynosure 无实现） |
| `docs/spec/policy-modeling.md:35` | 无 KL、无参考模型（本方案的硬约束） |
| `docs/spec/reward-model.md:34-48` | 在线更新机制与 warm-start / gate 设计 |
| `docs/spec/data-preparation.md:36` | 域错配的原始记录与修复裁决 |
| `docs/adr/0001-...md` / `docs/adr/0007-...md` | 损失/取值/归一化决策；warm-start 与 0.65 门槛 |

### 7.3 未验证 / 存疑（诚实标注）

1. **`ModalityLabelPerturber` 的「~19% 步」**：任务书给出该数字，但 cynosure 仓库、`~/Documents/NV-Generate-CTMR`、`~/Documents/MR-Generate` 三处均**未搜到该类**。cynosure spec 只记载 `prob=0.1`（`policy-modeling.md:14`）。**该机制在本地无实现，采用等于新写。**
2. **t1c 的「坏」是欠训还是标签映射失效**：本地无 `modality_mapping.json` 与基座 label 使用统计，**无法区分**。MR-RATE 上这是成败攸关的前置问题（§4.1）。
3. **稀疏模态计数**（T2w 669 / MRA 157，0.1% / 0.02%）来自任务书转述，未在本地工件中核对；报告按 0.02% 推算的 MRA held-out ≈ 16 卷、train ≈ 110 条，**若原始计数不同，§4.3 与 §5.6 的量化结论需重算**。
4. **「reward model 与 policy 同分布 → 无信号」这一命题**：检索未找到直接文献。最接近的是 Gao et al.（over-optimization）与 arXiv:2406.07971（RM/PM saturation），但两者都不是该命题。**本报告的该判定由 §1.2/§1.4 的一手数据 + §6.1 的 GAN 理论外推得出，不依赖文献。**
5. **AdvDMD（arXiv:2604.28126）会议状态**：核对 abs 页（2026-04-29 提交），未见会议接收信息；Adv-GRPO 标注 CVPR 2026。两者均为很新的预印本，未经充分复现检验。

### 7.4 给 #67 地图的四条待升级迷雾

1. **基座能力基线必须先于 warm-start**：按条件测冻结基座 FID（现有 milestone 仪器），作为「判别器 AUC 预测器」与「标签映射排查」的双重输入（§4.1）。
2. **条件白名单机制**：gate 的产物从「开跑/拒跑」扩展为「哪些条件允许进 RL」，训练循环按白名单门控（§5.3 + §5.4）——这需要新票。
3. **稀疏模态的 real 侧过拟合防线**：MRA train real 仅 ~110 条，判别器过拟合到病例的风险 + 对应的增强/配额设计（§5.6）——这需要新票。
4. **hacking 签名分型**：当前 `hacking_signature` 是布尔量，把「判别器冷」、「判别器盲」、「判别器被错误引导」三种病因压成同一个 1.0（§1.4）。至少要拆出「按条件 AUC 带内占比」与「按条件 FID 走向」两个正交轴，才能自动给出正确的应对——这需要新票。
