# RL 训练执行模型切换（第一步）：生产入口单口径 async 门面 + nominal-v12 收口

**Status**: accepted（#226 切换期两步之第一步，#240；第二步 = #242 全删除面，须待 #241 性能正式验收通过后执行）

## Context

加厚六期（rollout / policy 更新 / 判别器链 / 续训与事件契约 / 评测顺迁 / pretrain driver，#231–#238）已把 async 执行模型的全设计面落地：新执行序（`AsyncTrainingExecutor`，#217 门面形态）仅被 fixture 测试驱动，生产入口仍指旧执行序（`GranularGrpoTrainer` 单进程 world-1 / torchrun 多进程 + FSDP/DDP）。#226 决策 1 的「加厚期窗口」随两扇轨迹对照门（RL 门 #235 / pretrain 门 #239）与双门全绿关闭。

**决定：生产入口改指新执行序，单口径合入（无运行时开关、无新旧选择）；续训分片版本口径收口为 v12 单常量；旧执行序本体转为零入口死代码，待第二步整删。**

## Decision

1. **CLI train 单口径**。`train` 子命令装配 `AsyncTrainingExecutor`（设备发现 → 分配表 + 槽注册表 + 判别器窗口计划 → 每卡完整副本与绑卡线程 → 评测三路径 + v12 续训）。torchrun 启动（RANK env）显式拒绝——多进程拓扑属旧执行序，拒绝在 run 目录预占之前（与 pretrain 的 ADR-0018 决策 10 同款历史回环）。卡集裁剪 = `CUDA_VISIBLE_DEVICES`；调度槽数 = `execution.coroutines`（缺省 = 卡数）。

2. **组3 序贯编排退役于入口**。`SequentialTrainer`（两阶段单次运行的编排器）是旧执行序组件，随其唯一依赖（旧 trainer）一并删除；CLI 对 `group="sequential"` 显式拒绝。两阶段语义由两次独立 run 衔接（stage-1 产物 checkpoint 经 config 工件路径装配——`policy_iter*.pt` 契约名不变）。schema 的 sequential 分支（`stage1_run_dir` / `stage2_pretrain_report_json` 绑定守卫）暂保留为死面，随第二步与配置面残余一并裁决。

3. **nominal-v12 单常量收口**。旧执行序 `ResumeStore`（v11、per-rank 分片 + 代际标记 + 集合化拒绝）随旧 trainer 删除；`ASYNC_RESUME_FORMAT_VERSION`（v12）成为续训分片契约的唯一版本单点。行为面：v12 读写闭环、非 v12 代际分片（历史 v11 / 更早）拒载——「跨执行器拒绝由版本号承载」（#222）的加厚期双常量互拒形态终结为单边拒载。**续训跨代际不迁移**：切换前旧 run 的 v11 分片不可恢复（从产物 checkpoint 重启新 run），升版拒旧先例（v9→v10→v11）的自然延伸。

4. **训练侧 log-prob 诊断工件退役**。`training.json`（训练侧 π_old 录取/重算对）的唯一生产者是旧 trainer 循环，随删；π_old 一致性的组件级锚（`StepRollout` 记录 vs `evaluate_log_prob` 重算，test_policy）存续。`--dump-trajectory` 保留组1 采样场对照（`trajectory.json`，`TrajectoryDiagnosticRunner` 独立单进程路径，与训练循环无关）；async 门面的诊断路径（trajectory 相）留待诊断加厚票（#237 docstring 既有记档）。

5. **iter 事件相位面收窄为三相**（#237 口径的入口侧显式化）：`rollout` / `policy_update` / `discriminator`——held-out AUC 的池化原料打分计入 rollout 相（判别器链期 #234 的池化语义），旧四相的 `heldout_auc` 独立相位不再存在；判别器相仅在判别器步 iteration 出现（窗口节奏 N_d）。

6. **分配表 tile 边际勘误提请**（#226 接缝备忘的记档义务）：`AllocationTable` 对 C < D 的条件 tile 采用均匀 tile（每条件 ⌊D/C⌋ 或 ⌈D/C⌉ 槽）；#217 决议 1 字面「⌈D/C⌉ 或 ⌈D/C⌉+1」与均匀 tile 的正确边际差一，按实现口径勘误记档（allocation.py docstring 在案）——提请维护者在 #226 结票时对决议字面做同款勘误注记。

7. **测试面处置**（#226 决策 10 三桶的切换期落点，#242 收口其余）：
   - **退役**：`test_sequential.py` 整文件（编排器删除面）；训练侧 log-prob 诊断族（training.json 消费）；旧判别器侧编排族（`update_step` 注入形态的时序/条件锁——新链由 `test_discriminator_chain` 窗口/桶/池化族与 `test_async_executor` 重构族接替）；旧 resume 拒载族（marker 兼容、CUDA availability、v2/v10 代际、旧 store 拒 v12）。
   - **改写为锚**：CLI e2e 全量改指新执行序生产入口（断言槽自适应：每 iteration 每调度槽恰一条 iter 事件，`execution_slot_count` 基数——本机 CPU 单槽 / gauss 双卡双槽同断言成立）；直构 trainer 的装配面测试改走 `AsyncTrainingExecutor` API（optimizer 超参 / RNG 注册表 / 设备放置 / spectral norm checkpoint 逐位还原 / 仅 UNet / 仅 ControlNet）；续训 roundtrip 族改经 v12 分片（MidRunCrash 注入点 `IterationLoop.update_policy` → `AsyncTrainingExecutor._run_iteration`）；新增 v11 拒载 / 扁平 generators 双保险拒载锚。
   - **roundtrip 跨实例比对容差化**：截断/崩溃 resume 的恢复段与独立 baseline 的对比是**两次独立 `AsyncTrainingExecutor.run` 的世界对**——GPU 库层 1-2 ulp 重算噪声（确定性模式不可根除）经 AdamW 归一化步长链式放大（gauss 双卡 4-iteration 实测：事件指标最大 1.6e-3、收官 checkpoint 最大 1.5e-5；共享集群负载窗口另有不可控波动增量，带 5e-3/1e-4 容差仍观测到低频越线）。恢复段比对取 atol=1e-2（事件面）/ 1e-3（checkpoint 面）：恢复语义的结构面（iteration 序、事件数、恢复点对账、rewind 边界、milestone 记账）全部严格相等是硬锚，数值面只承载大方向恢复缺陷（RNG 流错位 / EMA 错位 → O(0.1) 级，检测余量 10 倍+）；同进程重放（流回填、分片读写闭环、自 run 内 checkpoint 一致性）仍逐位。依据 ADR-0018 决策 9 accepted drift 与 `RunTrajectory` docstring 既有的两级语义。
   - **EMA 拒绝面上移**：`ema_anchor_enabled=true` 的装配期拒绝改为 schema 级（config 装载即拒）——旧 trainer 的构造期守卫删除后拒绝单点不得悬空。

8. **跨卡权重一致性的物理现实对齐**（gauss 共享负载压测实证，切换期验收暴露）：「单进程多卡 + 每卡完整副本 + 梯度 allreduce」架构下，GPU 库层归约噪声（共享集群负载窗口放大）使 1-2 ulp 级跨卡权重漂移成为概率事件——压测实测 policy 与判别器皆可随机出现 2.4e-7 ~ 2.6e-6 失配，逐位校验以 ~1/10 概率在 checkpoint 点误触发 fail-fast。三处落点：(a) `synchronize_spectral` 的步末同步面从 spectral u/v 扩展为**判别器全参数 + u/v**（broadcast 卡 0 权威，#220 决议 2 口径的自然延伸——参数同样不可平均；幅度即噪声级、不改训练数值语义）；(b) `detect_card_divergence` 校验从逐位改为浮点 allclose(rtol=0, atol=1e-5)、非浮点仍逐位——噪声上界 2.6e-6（4 倍余量）之下不触发，结构性分叉（单步 AdamW ~lr=5e-5 量级，5 倍检测余量）仍 fail-fast；(c) 测试面的跨实例/跨卡数值比对同步容差化——续训 roundtrip 跨实例世界对（事件 atol=1e-2 / checkpoint atol=1e-3，结构面严格）、多卡同 seed 重放（iter ≥1 事件 rel 1e-1→3e-1、权重 atol 5e-5→2e-4，#234 包络 2.2% 与共享负载实测 12.7% 在案）、跨卡一致性比对（atol=1e-5，spectral buffer 因 broadcast 直接复制保留逐位）。#217「分叉是吸收态」的语义保留给真实信号；逐位数值重放的判别力由 CPU 单卡多协程档独占（该几何实测稳定）。

9. **衰减窗口保留面清单**（#242 删除对象，本票不动）：`TrainingRuntime.build`（旧分布式装配，零消费者）、`DistributedContext` 控制面（pretrain 的 RANK 守卫与超时解析仍消费，随 #242 与 pretrain 侧一起去留裁决）、`PolicySharding` / `RankSlicedPool`、`ShardingConfig` 与 sequential schema 死分支、`rollout.IterationRollout` 的旧消费残余。本票合入后上述面全部零入口（生产入口不再触达）。

## Consequences

- 生产 run 的执行形态从「torchrun 多进程 + FSDP」变为「单进程多卡 + 每卡完整副本」：多卡机器的显存占用上界从分片共享变为逐卡副本 × 卡数（fixture/生产网络的 #123 前向激活预算口径不变；权重驻留 0.687 GiB/卡的量级下 4 卡可承载）。
- 性能正式验收（#241，≤10% 底线）在本票后、#242 前执行；基线采自 #219 后旧执行序、sugon 同机同卡新跑（#226 决策 9）。
- 旧 run 目录（v11 分片时代）在本票后只可读产物 checkpoint，不可 `--resume`。
