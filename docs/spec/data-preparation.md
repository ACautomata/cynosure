# 数据准备方案：BraTS 加载对齐上游训练 recipe

> 本章由 grilling 会话决议产出（2026-09-04）。上游（NV-Generate-CTMR fork）预处理链事实经只读代码核查；决策记录见 `docs/adr/0006-brats-preprocessing-upstream-recipe.md`。

## 范围与原则

- **零依赖原则**：上游 fork 代码只读参照、永不 import。对齐的是**数据处理语义**，不是代码。
- **对齐目标 = 上游训练数据创建链**（fork `src/ctmr/infrastructure/maisi_engine/create_training_data.py`），非其推理/评测链（`fid_2d5.py` 的 Spacingd、instrument 链 GridResampler、mask 增强等均不适用）。
- **权威性**：recipe 偏差以 fork 实际代码为准，不以 MONAI 上游 MAISI 默认为准（见 ADR 的 clip 讨论）。
- **改造点唯一**：全仓库读原始 NIfTI 的位置只有 `PreparePipeline._encode_one`（`src/cynosure/reward/pipeline.py`）；训练/评测侧只消费预编码 latent，本次零改动。

## 上游 recipe（权威事实，逐项）

fork `create_training_data.py:55-96` `create_transforms` 的六步链：

1. `LoadImaged`（NIfTI 读取）
2. `EnsureChannelFirstd`
3. `Orientationd(axcodes="RAS")`
4. `EnsureTyped(dtype=torch.float32)`
5. `ScaleIntensityRangePercentilesd(lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=True)` —— mri 臂；**clip=True 是本 fork 对 MONAI 上游 MAISI（clip=False）的唯一有意 recipe 偏差**（fork issue #251），两边 embedding 不可互用
6. `Resized(spatial_size=dim, mode="trilinear")`，目标 dim 由 `round_number` 计算：**每轴 `max(round(size/128), 1) * 128`，size 从 RAS 重定向后的 spatial shape 读取**（fork issue #312：从 NIfTI header 存储轴读对轴置换方向会错；BraTS flip-only 幸免）。BraTS 240→256、155→128

明确不存在于链中的步骤（同样是对齐结论）：

- **无 Spacing 重采样 / 无 CropForeground / 无 NormalizeIntensity / 禁 z-score**（fork census 明令：z-score 与 rflow-mr-brain v1 训练分布不一致）。物理 spacing 不保持，体数据靠 resize 改网格。
- **无模态堆叠**：四序列（t1n/t1c/t2w/t2f）逐条独立处理、独立编码、独立 modality token（29/34/30/31，skull-stripped 码；cynosure `fixtures.py` 映射已一致）。

编码侧事实（引用备查，**不在本次 scope**）：`SlidingWindowInferer(roi_size=[320,320,160], mode="gaussian", overlap=0.4)` 包 `encode_stage_2_inputs`，AMP autocast；latent 全局标量 scale_factor（=1/std(z)），复用 checkpoint 值、不重算，无 per-channel scale/shift。

## cynosure 落点

- **transform 链**：`_encode_one` 的裸 `LoadImage` 换成 MONAI `Compose` 六步链（单影像场景可用非 dict 版 transform）。末端保持 `[1,D,H,W]` float32（`LatentEncoder.encode` 契约）；latent 契约 `[4,64,64,32]` 由 resize 步达成。
- **dim 公式参数化**：resize 基数（上游 128）做成可注入参数。fixture config 注入小基数——fixture 影像尺寸不变，但 orientation/强度/dtype 步在 fixture 下全走；链逻辑单份、测试覆盖全链。**fixture 不是对齐对象**，其参数独立于上游。
- **spacing 侧车**：prepare 逐 case 读 NIfTI header `get_zooms()[:3]` × 1e2 存入 manifest（per-case）；rollout 侧源条件的 `spacing_tensor` 接线 manifest（替换现恒定 fixture 值）。BraTS 1mm iso → `[100.0, 100.0, 100.0]`，与现 fixture 值巧合相同，但语义从「写死」变「来自数据」。
- **train/val 划分**：保持 cynosure 现状（排序 + seed 洗牌病例级 70/10/20，`reward/dataset.py`）。上游 fold 字段机制的生成脚本已退休不可考，不复刻伪对齐；两边「病例级 70/10/20」划分原则一致（`experiment-design.md:63` 本取自 fork 事实）。
- **latent 存储域**：manifest 存 encode 原始输出（未乘 scale_factor）；checkpoint scale_factor 的域缩放语义归 policy 采样 ticket。

## 边界（不在本方案内）

- **生产 VAE encoder**（`AutoencoderKlMaisi` + 滑窗 + AMP 装载）→ 基座 checkpoint ticket（现 CLI 显式拒绝非 fixture 模式的状态维持）。
- **decode 后逆变换**（合成影像 [256,256,128] → 回原生形状供 nnUNet/FID）→ eval ticket。
- 上游 P1 的 modality label 扰动增强、MR-RATE replay 1:1 混合：属上游训练编排，非数据加载，不适用。

## 验收

1. **方向断言**：任一 BraTS case 加载后 affine 轴码 = RAS（BraTS 原生 ~89% LPS，flip-only 无轴置换，翻转后达成）。
2. **形状契约**：真实 BraTS 影像经链后为 [1,256,256,128]，latent [4,64,64,32] 通过既有契约检查。
3. **fixture 端到端**：`tests/test_prepare.py` 全链（含新 transform 步）通过，工件契约（序列分层、病例级不相交、幂等）不回归。
