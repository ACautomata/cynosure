# 评估与验收：复用基座验收阶梯（像素域定量评测 + 下游分布对齐 + 专家目检，不做 latent FID、不比 Dice）

RL 后训练的验收定为**复用基座的验收阶梯**——定量评测（像素域 2.5D FID + KID 重采样 CI，跨模态加 3D SSIM/MAE）+ 下游指标分布对齐（nnUNet 仪器产出肿瘤体积/质心/ET-WT，做 TOST/KS/EMD）+ 专家目检（盲审流水线视觉图灵 + Likert + Fleiss' κ）——并**不做 latent FID、不直接比 Dice**。完整设计见 `docs/spec/experiment-design.md`。

**Status**: accepted

## Considered Options

- **latent FID**：弃。latent 域无现成 3D 医学特征提取器（FID 需特征网，像素域有 RadImageNet-ResNet50，latent 域没有等价物）；硬造一个引入新训练开销与不确定性。且基座已有解码 FID 基础设施（`fid_2d5`），像素域 FID 复用即可。
- **nnUNet Dice 作唯一主判据**：弃。合成影像**无 GT**，Dice 无法对合成样本计算；基座终验也不以 Dice 作硬门槛（Dice/HD95 只用于校准与 P2 回切一致性），而是用肿瘤指标的**分布对齐**（TOST）——无逐例 GT 时，正确验收是「合成分布与真实分布对齐」，而非「逐例 Dice」。
- **复用基座验收阶梯（选定）**：定量评测（FID）看生成质量、下游分布对齐（nnUNet 仪器 + TOST）看下游效用、专家目检兜住「指标都好但仍有肉眼伪影」的盲区；三块相对 no-RL 基线统计显著提升即成功。

## Consequences

- 评测必须解码到像素域（FID、专家目检都要 pixel/NIfTI），RL 打分仍在 latent 域——解码只发生在**评测路径**（Baseline 采样、里程碑评测、RL 后重采；#25 的 manifest 契约要求 Baseline 与 RL 后同条目解码样本对），不进逐 iteration 训练循环。
- **FID/KID 按目标序列分层宏平均**（#25 里程碑路径）：fid/kid = 各 target 逐面距离的算术均值，分层值随 `criteria_summary` 落盘（`fid_target_*` / `kid_target_*`）。条件模型忽略/交换模态标签而保持总体混合时全池聚合距离不可见（合成池 = 参照池的混合分布、距离 0），宏平均使坍缩直接进主判据；单一 target 时退化为原全池口径。
- nnUNet 是**测量仪器**（产出分布，不比 Dice），与训练图解耦，只评估时调用。
- 成功判据只钉「相对基线显著提升 + 触发规则」，绝对阈值（FID < X 等）为运行时 knobs，依赖训练期经验数据，超出 spec 范围。
- 三块主判据 + 在线 reward 侧信号（held-out AUC / 组内 std / anchor eval reward）共同构成完整验收与早停。
