# 预训练执行模型：单进程多卡 async 化（测量模板直锚 + 卡轴分片 + gate inline 主控）

**Status**: accepted（取代 [ADR-0016](0016-pretrain-measurement-sharding.md)；0017 由 #219 先落、本票后继——注记防占号颠倒）

## Context

ADR-0016 把判别器预训练执行面 torchrun 化（测量批按卷分片到 N rank、rank0 gate 单点、判别器 DDP）。加厚期路线图（#226）随后裁决了整个 RL 执行模型的单进程多卡化：GIL 对照 spike **go**（#229）、每卡完整副本 + 主控单点集合通信的门面形态（#217）。#221 结票决议冻结了预训练 driver 的 async 化全口径（22 条）：torchrun 多进程拓扑（进程组、rank0 广播/gather、DDP、EventMerger 归并）在单进程多卡下没有存在语义——集合序列对齐的死锁面（ADR-0016 记录在案的核心风险）随进程组消失而结构性消失。

**决定：预训练执行面迁移到单进程多卡 async 执行序——步进循环 resident 主控（编排层），每卡一条静态绑卡执行线程，任务粒度 = 每卡每相一个单元任务直交绑卡线程；算法语义零变化（per-step 单条件轮转 + per-condition 棘轮终止 + 复测确认 + 补测循环全保留）。**

## Decision

1. **执行形态**。主控（编排层）持步进循环，#217 门面形态（绑卡线程直交提交；预训练无分配表、无 k 循环、无协程语义面，任务载体不承载任何决策）。单步序：排列抽取（主控单点）→ 测量扇出（每卡一片）→ join → 卡序 plain 拼接卷级分数聚类 → pooled AUC（CPU）→ SupportRule / 复测确认 / 棘轮 / steps_completed 判定 **inline 主控** → 更新扇出（每卡 K 对装配 + 本地前向反向）→ join → 判别器步（逐对等权 loss 的跨卡梯度 SUM allreduce（M1 形态：主控单线程驱动多卡 NCCL group，与在线判别器步同构）→ 各卡 AdamW → 步末 u/v broadcast（卡 0 权威））→ 事件主控直写 + 告警卡序追加。

2. **测量批 = 主控单点复位测量模板**（#221 决议 5/9）。模板 = `schedule.seed + 19` **显式直锚**（数值与现行 `shared_seed+19` 恒等；卡轴化 recon 流下 `initial_seed()+10` 的实现缝会静默随卡漂移，故直锚不派生）。每次测量：复位 → 条件构造一次 → 按卡序逐卡抽本地 ε 行直接交任务；`volume_offset` 参数与前缀消耗**整体退役**。与 ADR-0016 否决项「rank0 抽全量再 scatter」的距离：否决的是全量驻留 scatter 张量，本机制逐段瞬态生成（主控瞬态 = 单卡段 ⌈V/D⌉ 行）、无全量搬运。

3. **顺序流等价性 = 受锚守护的不变式**（对抗审核实证降格）。float32 `randn` 走 ATen normal_fill 的 16 元素块 Box-Muller，分段抽 ≡ 全量对应行**当且仅当行宽 ≡ 0 (mod 16)**（同时保证 fallback 路径不可达）；现行安全靠数据巧合（测试行宽 8192、生产 4 通道×偶³ latent）。新增**装配期不变式断言**：每条件测量批行宽（latent numel）≡ 0 mod 16，违者 fail-fast，报错文案引导「词表/manifest 工件异常」而非机制缺陷（`LatentManifest.assert_measurement_row_width`）。#198 三组锚**平移接棒、不退役**：断言形态 = 同复位状态主控逐段抽 vs 全量对应行逐位（宿主 = `MeasurementTemplate`，tests/test_pretrain_measurement）。

4. **分片纯函数化**。`ShardPlan.split(V, D)`：卡轴连续段 + 前余均分（现行 `_shard_bounds` 提为纯函数，测量批与 real 侧共用）；σ 轮转全量位次 = 全量 σ 列表切片（零偏移算术）；装配期守卫每条件 held-out ≥ 卡数（现行 world-1/distributed 两态守卫统一换名）；real 侧容量守卫 ≥ K×卡数零改动。

5. **gate 四态 inline 主控**。SupportRule bootstrap = `seed+7` 直派单点；四态不设代码实体——update/remeasure/confirm/halt 即主控 `_step_loop` 的 if/continue/break 控制流；`broadcast_object`/`broadcast_flag`/object-gather/EventMerger 归并/rank0 写者门退役——单进程无集合可对齐（#220 决议 19 的主控单点分发在单进程的落点 = inline）。

6. **更新批 σ/ε 卡轴化 + real 侧主控全局抽取**（#221 决议 11/12）。recon 流按槽公式以卡号派生（`seed + 卡×10⁶ + 9`，卡 0 恒等 = 现行 rank0 recon 数值）；每卡 σ 位与 ε 独立、「先 s 后 ε」次序契约保持。real 侧 = 主控对全池全局无放回抽 K×卡数 → ShardPlan 切片到卡（跨卡无重复的条带互斥语义保持）；`real_pool` 流 seed+1 单点直派；`RankSlicedPool` 退出 pretrain 路径（本体随旧执行序衰减窗口保留）。**accepted drift**：real 抽取空间条带→全池、抽取者各卡→主控、卡 ≥1 的 σ/ε。

7. **随机面恒等/drift 对照表**（#221 决议 20，迁移验收归因基底）：

   | 面 | 流 | 口径 |
   |---|---|---|
   | 恒等 | held-out 排列 | `seed+3` 主控单点（与现行 rank0 `condition_order` 逐位恒等——randperm 一处、构造零消耗） |
   | 恒等 | 测量模板 | `seed+19` 主控显式直锚（数值与现行恒等） |
   | 恒等 | SupportRule bootstrap | `seed+7` 单点（现行 rank0 数值不变） |
   | 恒等 | 冷启动判别器 | `seed+6` fork（`assemble_scorer` 既有口径） |
   | 恒等 | 卡 0 recon | `seed+9`（= 现行 rank0 recon 数值） |
   | drift | real 侧 | 抽取空间条带→全池、抽取者各卡→主控 |
   | drift | 卡 ≥1 σ/ε | recon 流卡轴派生（每卡独立） |

8. **产物契约零改动**（#221 决议 17/18）。checkpoint = 判别器步末**卡 0 副本直写**（确定性 allreduce + u/v broadcast 保证全卡逐位一致，含 `_u`/`_v` buffer；`loadable_state_dict` 键集/格式不变；不设 checkpoint 周期 bitwise 守卫 = 显式裁决——run 短风险低）；PretrainReport 字段集与 provenance 指纹面不变；PretrainEvent 字段不动（reconstruction_forwards / measurement_volumes = 全量 σ 列表推算与全量排列长度，分段求和的加法结合恒等）；train 侧消费面（load_discriminator + 组别绑定守卫）零改动。告警轴 (步, rank) → (步, 卡) 的迁移对象**仅 OverfitAlertEvent**（rank 字段归因观测卡；OverfitMonitor 每卡实例——本地 train acc + 全局 AUC 的卡轴诊断保持）；PretrainEvent 本无 rank 字段。

9. **复现承诺口径**（取代 ADR-0016 决策 7）：#218 决议 0 双口径（生产统计等价 / 测试进程逐位）+ **同 config 同卡数重放逐位**；跨卡数重跑 = 新 run（预训练无 payload、天然无拒绝面——与 RL 跨拓扑显式拒绝的差异记一句）。复现锚 = #218 三层锚形态套预训练（派生公式单元锚 + CPU fixture 重放逐位 + gauss `--run-slow` 多卡档）。

10. **torchrun 守卫历史回环**：CLI pretrain 恢复 ADR-0016 决策 3 恰好退役的 RANK env 拒绝守卫（执行模型已单进程多卡化，进程内多卡由设备发现承担）；`fid`/`fid-floor`/`base-smoke` 的 `_reject_torchrun` 泛化守卫随 RANK 注入源死化一并退役。

11. **第三批删除面收口**（#238）：`ReplicatedDiscriminator` 本体（含 `distributed` 导出与 `RewardScorer.adopt_distributed`/DDP 解包）、`EventMerger` 本体（旧 trainer 归并点降级为单事件直写——衰减窗口内单进程语义与 world-1 恒等逐位同形）、`_reject_torchrun` 死守卫、torchrun slow 档测试退役（CPU fixture 档与 gauss 多卡档接替）。`DistributedContext`/`PolicySharding`/`RankSlicedPool` 为旧执行序 trainer 的衰减窗口保留面（切换期第二步整删）。

## #220 决议修订记档（#221 决议 5 移交）

- **决议 17 前半句退役**：「各卡模板状态逐位相等断言」失去对象（模板单份，主控单点持有）。
- **决议 18 前半句改判退役**：「ε 前缀消耗保留」→ 按卡序逐段顺序抽取（本 ADR 决策 2/3）。
- **决议 18 后半句落地**：σ 定序轮转纯函数化（全量位次 = 全量 σ 列表切片）。
- **决议 17 后半句强化落地**：测量模板 seed+19 显式直锚（决策 2）。

## Consequences

- **退役面**：torchrun 多进程预训练拓扑整体（进程组装配、broadcast/gather、DDP、EventMerger、rank0 写者门）；`volume_offset`/`measure_condition`/`measurement_forward_count` 从装配原语退役（测量面迁 `MeasurementTemplate`）。
- **新增面**：`pretrain/sharding.ShardPlan`（分片纯函数）、`pretrain/measurement.MeasurementTemplate`（主控测量面原语）、`PretrainCardWorker`（绑卡直交执行线程）、`MeasurementSources`（排列/real 抽取主控单点）、行宽 mod 16 装配期断言、`HeldOutAuc.volume_clusters` 打分段静态面。
- **成本曲线**：测量批墙钟 ≈ ÷卡数（同 ADR-0016 的收益结构——并行边界不变，载体从多进程换绑卡线程）；主控 join 与 CPU 侧 AUC 为毫秒级。性能验收读数 = driver 每步 `elapsed_s`（PretrainEvent 既有字段，不新增埋点）。
- **集合序列错位面消失**：单进程无互等挂死（ADR-0016 记录在案的核心风险结构性消除）；fail-fast = 卡任务异常经 Future 传播中止（daemon 线程口径与 async 门面一致）。
- **测试档**：CPU 单进程全保 + 新增测量面锚（test_pretrain_measurement）；torchrun spawn 档退役；gauss 多卡 e2e + 同 seed 重放锚挂 slow+gpu。
- **与既有 ADR 的关系**：ADR-0003（train 编排）不修订（旧执行序衰减窗口内照旧）；ADR-0012（判别器机制）零改动；#220 决议修订见上节；ADR-0016 superseded（其 World-1 退化路径语义由「同 config 同卡数重放逐位」取代——单卡即卡数 1 的特例）。
