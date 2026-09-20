# 预训练编排：torchrun 测量批分片 + rank0 gate，判别器 DDP 同在线

ADR-0003 把 RL 主循环定在 torchrun + FSDP 的编排上；判别器预训练（ADR-0007/0008）因「多 rank 各自预训练会分叉判别器」保持单进程 World-1 退化路径，CLI 层以 RANK env 显式拒绝 torchrun 启动。ADR-0012 落地后的生产数据重跑（#174）给出成本结构实测：预训练每步 = **测量批**（该条件全量 held-out 512 卷的冻结基座同源重构，逐卷平均 ~21.5 次 UNet 前向、事件流 `reconstruction_forwards=11032`）+ 判别器更新（K=8 配对批，毫秒级）——99% 的墙钟在测量批重构；MR-RATE 各平面尺寸极差（t1w/coronal ≈ 8× t1w/axial）下 coronal 单步 71 分钟，`pretrain_max_steps=100` 的最坏走满是 2 天以上。判别器 warm-start 是一次性成本、后续全部 RL iter 复用，墙钟压缩 = 把唯一的大头并行掉。**决定：预训练执行面 torchrun 化——测量批按卷分片到 N rank 独立重构与打分，分数 gather 到 rank0 做全局 recon-AUC 与 gate 判定（`broadcast_flag` 分发步进决策）；判别器走与在线期同款的 DDP 数据并行；冻结基座不 FSDP（每 rank 完整副本）；World-1 退化路径保持，单进程行为与现行逐位一致。**判别器机制面（fake 构造、gate 判据语义）零改动——本 ADR 是执行面变更，不是 ADR-0012 的修订。

**Status**: accepted

## Decision

1. **并行边界：只分片测量批**。并行收益全部来自测量批的重构前向与打分（11032 前向/步 vs 更新步毫秒级的 99/1 成本比）；判别器更新的装配、AUC 的秩统计计算、gate 判定与白名单棘轮不追求并行。测量批按**卷**切分（每 rank 512/N 卷各自跑完整 ODE 链）——ODE 逐样本独立、`continue_to_terminal` 的 batch 维并行天然适配；不按 σ 组跨 rank 切（会把单卷的积分链拆到多 rank、每积分步一次通信）。

2. **判别器更新 = 在线期同款 DDP 数据并行**。`TrainingRuntime.assemble_rewards` 在分布式下自动完成 `ReplicatedDiscriminator.replicate`（梯度 allreduce 平均、更新后各 rank 权重逐位一致）与 `RankSlicedPool` 条带切片（real 池按 rank 切、容量守卫按 `K×world_size` 把门）——预训练 driver 已经由同一装配缝组装，分布式化的装配面改动落在 runtime/driver 的编排层而非网络层。有效 batch = K×N：这是 ADR-0012 决策 7「两阶段构造同构」在执行面的延伸——预训练锚**在线口径**（在线 train 就是 DDP 每 rank K=8）而非单卡口径；config 字面不变（`disc_batch_size_k` 是 per-rank 值），`world_size` 是唯一变量。

3. **CLI 守卫退役，单进程与 torchrun 同一代码路径**。`_pretrain` 的 RANK env 拒绝（exit 2）移除，与 train 同款「单进程与分布式同路径」；分布式启动必须显式 `--run-dir`（跨 rank 目录对齐，CLI 既有规则）。fid/fid-floor/base-smoke 的 `_reject_torchrun` 泛化守卫不受影响。

4. **测量批 real 侧：同排列、切加载**。held-out 全量索引排列经 `heldout_auc` 命名流（rank0 的派生 seed = seed 恒等，跨 rank 同序）一次抽出，每 rank 只 load 本地切片（512/N 卷）上卡——测量批 = 单卡 rank0 全量排列的**精确切片**，且把「测量批不整批驻留」的原则（#174 OOM 修复）延伸到数据加载侧（coronal 全量 512 卷 ≈ 8.6 GB，切加载后每 rank ≈ 1.1 GB）。ε 侧批次复位测量流是 shared seed 派生的 CPU generator（`measure_condition` 既有设计，跨 rank 一致），切片 randn 与整批 randn 同位逐位一致。复测换批 = 流同步推进抽新排列（单卡版「流推进 → 新排列」语义不变）。实现缝：`HeldOutAuc` 需暴露「索引抽取 / 按索引加载」的分解面。

5. **gate 控制流：rank0 集中决策**。每 rank 对本地切片打分 → 逐卷分数 object-gather 到 rank0 → rank0 `torch.cat` 重算全局 `auc_from_scores`（Mann-Whitney estimand 不变；卷级归属保留、条件池同形保证 reshape 形态）→ `SupportRule` 判定、复测确认、白名单棘轮、`steps_completed` 计数全部 rank0 单点 → `broadcast_flag` 分发步进四态（更新 / 复测 / 确认 / 终止）。复测确认步全 rank 一致跳过更新——集合序列因此逐 rank 对齐（gather 与 broadcast 的调用次数由控制流决定，控制流单点即序列单点）。

6. **冻结基座不 FSDP**。重构前向是 no_grad 推理负载，180M 参数 bf16 ≈ 0.4 GB/rank 完整副本可忽略——分片省 0.35 GB/rank 的代价是每前向参数 all-gather 与 #165 式集合序列绑定。pretrain 保持无 FSDP、不注入 `chunk_sync`（各 rank 同条件同尺寸，分块 cap 天然一致，无集合可绑定）。

7. **数值口径：8 卡重放锚**。重放承诺改为「同 config + 同 world_size 重跑逐位一致」（前向分块边界由 world_size 决定、随之固定）；与单卡重放只保**统计等价**——卷积算法随 batch shape 选择引入 ~1e-12 量级尾差（#174 报告 7.1 的判读口径）。World-1 退化路径数值与现行逐位一致（同一代码路径、同一 RNG 消耗序）。

8. **产物写者 rank 门**。事件流（`metrics.jsonl`）、判别器 checkpoint、预训练报告全部 rank0 落盘——`PretrainRun`「单进程唯一写者」契约由调用方 rank 门满足（train 侧 `EventMerger` 先例：gather 到 rank0、仅 rank0 append）。`OverfitMonitor` 保持 rank 本地状态（ADR-0009 决策 4/5 语义：per-rank 离散本身是诊断信号，不跨 rank 平均），预训练相告警的归并按 train 侧先例落 rank0 事件流。

## Considered Options

- **判别器单点更新 + broadcast 参数**（rank0 每 rank-step K=8 更新后广播权重）：per-step 语义与单卡完全相同，但需绕开 `assemble_rewards` 自动发生的 DDP replicate、给 rank0 的更新批开全 pool 视图（`RankSlicedPool` 只见 1/8 条带）、每步新增一次权重 broadcast——三处反向改动换一个与在线不同构的特例语义，否决。
- **每 rank 独立 gate 判定**（gather 后 all-gather AUC、各 rank 同函数同输入）：省一次 broadcast，但决策逻辑 ×8 份，任何分支分叉即集合错位挂死；控制流单点才有单点语义，否决。
- **冻结基座 FSDP 分片**：省 0.4 GB/rank，换来每前向 all-gather + 分块 cap 的跨 rank 集合绑定（`chunk_sync` 死锁面，#165 先例），推理负载下纯负债，否决。
- **测量批 per-rank 独立子集**（heldout 流 per-rank 派生 seed）：子集互斥性无保证、重放锚从 1 份变 8 份，测量可复算性（ADR-0012 决策 5 的测量面契约）受损，否决。
- **rank0 抽全量测量批再 scatter**：coronal 全量 8.6 GB 张量的 scatter，发送端显存峰值与带宽双输，否决。
- **测量批按 σ 组跨 rank 切分**：单卷 ODE 链拆到多 rank，每积分步一次通信（通信量 ×平均 21.5 步），且把「逐样本独立」的天然并行换成强耦合，否决。
- **维持单进程不动、以降采样测量批换墙钟**（512 → 128 卷）：gate 判据的统计支撑度直接受损（#85 支撑度规则退到 bootstrap 下界、门变严），测量口径随规模变化而与既有读数不可比——用判据质量换墙钟，与本 ADR 的方向（保判据、并行化）相反，否决。

## Consequences

- **退役面**：`_pretrain` 的 RANK env 拒绝守卫；「预训练以单进程执行」的 CLI 提示与 driver docstring 口径更新（World-1 退化路径本身保留）。
- **新增面**：`HeldOutAuc` 的索引抽取/按索引加载分解缝；测量批分数的 object-gather 与 rank0 全局 AUC 重算；`broadcast_flag` 步进四态协议（更新/复测/确认/终止）；事件/checkpoint/report 的 rank0 写者门。
- **成本曲线**：测量批墙钟 ≈ ÷N——8 卡下 coronal 单步 71 → ~9 分钟，`pretrain_max_steps=100` 最坏走满从 2 天+ 压到 ~7 小时；判别器更新与 AUC 计算的并行开销（gather/broadcast）为毫秒级，可忽略。
- **核心风险（记录在案）**：集合序列错位面——gather/broadcast 的调用次数由 rank0 控制流决定，复测/确认/终止路径上任一 rank 的控制流分叉即挂死；测试档的 torchrun 多 rank slow 档专门覆盖四态路径。有效 batch = K×N 的收敛节奏变化（gate 过线可能更快或更慢）记录而不锚定，由首个多卡 run 的 recon-AUC 曲线实测。
- **测试档**：CPU 单进程现有测试全保（World-1 退化路径数值逐位不变是回归锚）；新增 torchrun 2 卡 gloo 多 rank 测试进 slow 档（切片语义、集合序列、四态控制流）；gauss 2×A6000 冒烟；sugon 8 卡真机验收。
- **与既有 ADR 的关系**：ADR-0003（train 编排）不修订——本 ADR 把 pretrain 接入同一 torchrun 家族但显式排除 FSDP（负载结构不同：推理为主）；ADR-0012（判别器机制）零改动，本 ADR 全部裁决在其机制语义之上；#85 支撑度规则、#171 成本读数面（`reconstruction_forwards`/`measurement_volumes` 事件字段）语义不变。
- **口径交接**：单卡 run（如 #174 t174-rerun）与多卡 run 的预训练读数**不可横向比较逐位值**（分块边界与有效 batch 不同）——对比实验的「同机制」判断锚在机制语义（ADR-0012）而非执行面；#174 对比报告的 pretrain 段以单卡口径收尾，多卡口径自本 ADR 后的首个多卡 run 起算。
