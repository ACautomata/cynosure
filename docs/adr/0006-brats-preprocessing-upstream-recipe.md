# BraTS 预处理对齐上游训练 recipe：resize 拉伸、0–99.5 百分位 clip=True、RAS，不做 spacing 重采样

cynosure 的 prepare 管线原本裸读 NIfTI 直接 VAE 编码：真实 BraTS（240×240×155）与 latent 契约 `[4,64,64,32]` 不兼容，且与基座 checkpoint 的训练分布脱节——reward 判别器的「真」与 policy 的 rollout 会落在错误的影像域。我们决定逐项复刻 NV-Generate-CTMR fork 的训练数据创建链（只读参照，零依赖原则不动），对齐的是数据处理语义而非代码。

**Status**: accepted

## Considered Options

- **形状转换**：选 `Resized(trilinear)` 直接拉伸到每轴 128 的倍数（240→256、155→128）。弃 pad 到 256×256×160——那是 fork census 的首选建议但**不是 fork 实际代码**；弃 crop+pad 保物理 spacing——同理，RL 优化的分布必须等于基座实际训练分布，不是理想分布。
- **强度归一化**：选 `ScaleIntensityRangePercentilesd(0.0, 99.5 → [0,1], clip=True)`。弃 clip=False——那才是 MONAI 上游 MAISI 的默认，本 fork 有意偏差（issue #251），两版 embedding 不可互用；**照抄 fork，不照抄 MONAI**。弃 z-score——fork census 明令禁止（与 rflow-mr-brain v1 训练分布不一致）。
- **spacing 重采样**：不做。fork 全链无 `Spacingd`，物理 spacing 不保持；spacing 仅作为条件张量（per-case 原生 header zooms ×1e2）进入网络。

## Consequences

- 换 recipe = 全部预编码 latent 工件重编。当前无任何真实数据工件（tickets 未开始），处于零成本窗口期；此后再改 recipe 需全量重编。
- resize 拉伸引入网格各向异性失真，spacing 条件值与 resize 后网格的物理间距**不对应**——这是上游自身的选择，照抄，不自作主张修正。
- 未来任何「改进预处理」的提议（pad、spacing 对齐、z-score）都等于换基座训练分布，须以偏离上游为代价重新论证。
- 逐项落点见 `docs/spec/data-preparation.md`。
