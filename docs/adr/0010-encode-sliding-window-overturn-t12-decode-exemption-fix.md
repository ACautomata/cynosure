# encode 超界滑窗改判（b 语义）与 decode 豁免口径修正（推翻 T12 编码裁决）

T12 编码裁决（记录于 `docs/spec/data-preparation.md`「上游 recipe」一节，非 ADR 形态）决定 prepare 编码**恒整前向**、超界（单卷 > 16,384,000 体素）**显式拒绝**，依据是一条集群实测：MONAI `SlidingWindowInferer` 的多分辨率拼合为上采样分割网络设计，对下采样 encoder 把通道维折进空间维（实测 [4,16,16,8] 产出 [1,16,16,8]，静默产出语义错误 latent）。该裁决与 MONAI 1.6 实际行为冲突：1.6 起的 `z_scale` 路径原生支持下采样网络的输出网格拼合，且上游生产链 665k 训练 embeddings 的长尾正是此路径编码；#141 零下载普查判定全语料 705,254 卷中 61.43%（433,255 卷）resize 后超界——拒绝路径对 MR-RATE 线是结构性丢料（细间距/大 FOV 卷系统性缺失），不是随机减采样。另有独立缺口：decode 整前向豁免判定把 4 通道因子算进 numel，与上游 `dynamic_infer` 单通道口径差 4 倍，中等体 latent 输出不可比。#139 决议「先探针后改判」：#140 集群数值探针（P1 encode 确定性对拍 → P2 seeded 采样 + scale factor 全链 → P3 无 Diffusion VAE 重建闭环）在真实 BraTS 体三目标网格 + sanity 网格上对拍整前向 vs NVIDIA 语义滑窗，fp16 重跑噪声地板先行标定、验收刻度相对地板。**决定：推翻 T12 编码裁决**——超界卷改走机制参数逐项锚 NVIDIA 的滑窗分支，采样编排取「b 语义」（对上游 a 语义为记录在案偏离）；探针越阈值读数经归因裁决为「上游滑窗语义的组成部分」记录在案；decode 豁免判定同步修正为与上游逐字同构的单通道口径。探针报告（证据链）随 release `exp/20260914-t12-probe` 归档。

**Status**: accepted

## Decision

1. **T12 改判：裁决理由证伪，超界改滑窗。** 探针（2026-09-14，DCU 单卡，MONAI 1.6.0 / torch 2.9.0，checkpoint `autoencoder_v1.pt` sha256 前 12 位 `1f8a7a056d0e`，体源 `BraTS20_Training_001_t1ce.nii.gz`）在全部目标网格上验证 MONAI 1.6 `z_scale` 路径滑窗输出形状逐格等于期望 latent 形状（含 512×512×256 的 27 窗拼合）——「通道维折进空间维」未复现，原裁决描述的机制在 MONAI 1.6 上不存在。超界分派改为：豁免判定与上游 `dynamic_infer` 逐字同构（单样本单通道空间体素数 ≤ prod(roi) 恒整前向，BraTS [1,1,256,256,128]=8.39M 全语料不触发、行为不变）；超界走 `SlidingWindowInferer`（1.6 `z_scale` 路径）包 encoder 确定性前向，机制参数逐项锚 NVIDIA `create_training_data`：roi [320,320,160]（影像空间）逐轴 clamp、overlap 0.4、sw_batch_size 1、gaussian。
2. **探针越阈值读数的裁决：接受为上游滑窗语义的组成部分。** 探针在确定性运行环境下跨进程逐位复现，噪声地板实测为零，阈值回落绝对下限（z_rms ≤ 1e-2、PSNR ≥ 40 dB）；实测超界网格 μ rms(Δμ)=2.5–2.9e-2（相对信号 5–6%）、重建 PSNR 39.04–39.91 dB（SSIM ≥ 0.9967）——判定式字面 FAIL 的事实记录在案。归因三条支持接受：dtype 无关（roi 缩小强制多窗对拍，fp32 4.017e-02 vs fp16 4.012e-02，比值 1.00）= 滑窗窗口分解对卷积感受野的截断效应，机制固有；非接缝集中（μ 差异带/心 0.50–0.61，全域分布）；重建域近似等效（MAE ≈ 4.0e-3）。裁决依据：对齐对象是上游实际行为（机制级锚 NVIDIA）——其 665k 训练 embeddings 本身即此路径编码，把上游自身语义的机制差异当缺陷否决与对齐目标矛盾。
3. **b 语义（blend-then-sample）= 对 NVIDIA 的记录在案偏离。** 上游 a 语义逐窗采样后拼 z，接缝带方差收缩（带/心 std 比 0.85–0.89）；本仓逐窗 (z_mu, z_sigma) 在 latent 网格高斯加权拼合、拼合**后**以单一内容寻址种子采样一次 eps（带/心 0.93–0.97，与体心均匀）。偏离理由：(a) 重跑零漂移幂等契约——单次采样消除窗口枚举序 RNG 依赖；(b) real pool 分布卫生——接缝降方差 artifact 不得喂进判别器的「真」侧。参数通道走 config `preprocessing.encode_roi_size` / `encode_overlap`（NVIDIA 锚缺省 + schema 范围校验，fixture 与生产同链）。
4. **decode 豁免口径修正：单通道空间体素数。** 修正前 `scaled[0].numel()` 把 4 通道因子算进豁免判定，与上游单通道口径差 4 倍——空间体素数 ∈ (27,648, 110,592] 的中等体 latent，上游整前向、本仓滑窗，输出不可比。修正为 `torch.numel(scaled[0:1, 0:1, ...])`，与上游 `dynamic_infer` 判定式逐字同构；滑窗参数不动（latent roi [48,48,48]、overlap 2/3，锚 NVIDIA `config_infer.json`）。修正属现网行为变化，里程碑基准复测已显式记录读数平移并归因（BraTS 现网尺度零平移、Δ=0 逐位确认；翻转区间 FID +0.517 / KID +0.373，#142）。
5. **512×512×256 量级：现网只有滑窗可行。** 整前向在探针实例现网余量下 OOM（HIP 8 GiB 分配失败），滑窗臂 27 窗正常完成且形状正确——该量级编码路径以滑窗为准，空卡整前向可行性不作为分派条件。裁剪 fallback 维持文档语义、不实现（#139 Out of Scope 维持）。

## Considered Options

- **维持 T12 显式拒绝**：探针已证伪裁决依据；61.4% 结构性丢料使 MR-RATE 线可编码语料只剩 38.6%，real pool 网格分布与上游 v1 语料错位 61 个点。维持拒绝等于放弃对齐目标，否决。
- **照抄上游 a 语义（逐窗采样拼接）**：接缝方差收缩 artifact 进判别器「真」侧 + 窗口枚举序 RNG 依赖破坏幂等契约。否决，取 b 语义记录在案偏离（Decision 3）。
- **以判定式字面 FAIL 否决滑窗**：验收刻度设计为「相对地板」，零地板使绝对下限过严；差异归因为与上游共享的机制固有效应（Decision 2）。把上游语义当缺陷否决与对齐目标矛盾，否决（FAIL 数字照录）。
- **自研「输出网格直接加权拼合」**：MONAI 1.6 `z_scale` 已覆盖，探针未证伪，重启条件不成立，否决（仅当探针证伪时重启，#139 Out of Scope）。
- **decode 维持含通道因子旧口径**：豁免边界与上游差 4 倍，中等体 latent 输出持续不可比，否决。

## Consequences

- prepare 对超界卷不再 raise：MR-RATE 长尾 61.4% 恢复可编码；BraTS 生产全语料恒整前向不变，历史工件与读数可比性不受影响。
- encode 滑窗 latent 与上游存在两层记录在案差异：滑窗分解机制固有截断差异（与上游共享，Decision 2）与 b 语义采样偏离（本仓独有，Decision 3）——对齐讨论引用本 ADR，T12 旧裁决不再作现状引用。
- decode 修正后的翻转区间量级评测读数与修正前历史读数不可直接横向比较（可归因平移 FID +0.517 / KID +0.373，#142 复测记录为准）。
- 文档同步改判随本 ADR 落地：`docs/spec/data-preparation.md` 编码侧段落、MR-RATE 调研笔记编码网格条目（`research/mrrate-data-spec.md`）、CONTEXT.md 词条（预编码 / Decode / b 语义）锚定本 ADR。
- 证据链：探针报告 + 双跑 JSON + 脚本（钉 `3cf1050`）随 release `exp/20260914-t12-probe` 归档；provenance 见 Decision 1。
- 探针失败分支未触发：证伪时本应维持显式拒绝、改判转向裁剪/自研拼合方向（#139 行为契约）——本次探针通过，该分支关闭。
