# 条件闸门控链退役：从「可关闭」到「无此链」（取代 ADR-0008 决策 5/7/8）

ADR-0008 决策 5/7/8 构成的 held-out AUC 门控链（RM readiness gate 上岗判定 + 逐 iteration 梯度门控 + EMA 动态恢复）在 2026-09-17 被维护者裁决以总开关 `reward.condition_gate_enabled` 统一关闭（ADR-0008 尾部修订段）。#219 grilling 对「关着的链是否退役」专项裁决：**整条链删除、不留死开关**，监控面（held-out AUC 测量、ADR-0009 分叉监控）与预训练棘轮保留。本 ADR 记录取代决定与删除面口径。理由：**有效性有限 + 架构复杂度**——#122 实测 11 条件 held-out AUC 全落 chance 带（0.4934–0.5203，门槛 0.65 不可达），空名单说明该信号在本任务上不具区分力；而门控链的全 rank 集体口径（观测 all_gather + rank0 判定 + 快照 broadcast + 集体跳过）、EMA 滞回判定、续训落盘面与 5 个 config knob 的架构复杂度，与「关闸后零消费」的现状不成比例。

**Status**: accepted（2026-09-26 #219 结票裁决，#228 实施；取代 ADR-0008 决策 5/7/8，修订 ADR-0007 的上岗门槛语义）

## Decision

**删除面**（一次性删除，不留运行时开关）：

- 文件整删：`train/gate.py`（ReadinessGate）、`train/gating.py`（DynamicWhitelist + ConditionAucEma）、`train/whitelist.py`（ConditionWhitelist）；trainer 的 readiness 构造与启动期检查、门控观测相（`phases.mark("gating")`）、条件更新分支（policy 恢复无条件更新）。
- config：`condition_gate_enabled` / `gating_dynamic_recovery` / `gating_enter_auc` / `gating_exit_auc` / `gating_ema_span` 五 knob 与 `_gating_hysteresis_band` validator 删除。
- 事件契约收缩特例：`IterEvent.policy_gated` 字段删除、`phase_seconds` 键集去 `"gating"`。「可扩不可改名」是事件契约的常规纪律；本条是其显式记录的**收缩特例**——消费方（离线分析脚本）按缺失即「正常更新步」解读，旧流新读、新流旧读均不破坏解析。
- resume v11 bump：payload 删除 `gating` 键，`RESUME_STATE_FORMAT_VERSION` v10→v11，旧分片被版本对账显式拒绝（先例：v9→v10 的退役删除同形态）。config `extra="forbid"` 下旧 run config 快照含已删字段——分片版本对账先行拒绝，语义自洽。
- gate 术语一次清干净（**范围 = 字段/knob/打印面**，见 #219 改名表）：PretrainReport 字段族改名 `gate_whitelist`→`conditions_passed`、`gate_auc`→`pass_threshold`、`gate_passed`→`all_conditions_passed`、`gate_criterion`→`auc_criterion`（缺省值随之对齐当前口径 `"recon_auc"`）；config knob `pretrain_gate_auc`→`pretrain_pass_threshold`；报告改名同样走 schema 拒绝承载断代（旧键名报告在 `extra="forbid"` 下整体拒绝装载，无静默误装窗口）。
  **残余「gate」措辞显式保留**（不在清理面）：预训练测量路径的行文（`reward/assembly.py` 的「gate 测量批」、`reward/auc.py` 的「预训练 gate」等——指预训练棘轮的过线测量，属保留面）、`reward/support.py` 与 `gate_support_min_volumes`（Q1 保留：其唯一消费方是棘轮）、`pretrain/driver.py` 的「rank0 gate 单点/四态广播」（ADR-0016 分布式执行机制，同字不同义，#219 显式不触碰——退役归 async 执行模型票）。「清干净」的判据是**字段/knob/打印面零残留**，不是删除代码库中所有 gate 字样。

**修订 ADR-0007 的「RM 上岗门槛才允许进入 RL」语义**：预训练本体（判别器密集预训练 warm-start）、warm-start 装载守卫全链（`PretrainReport.load` → `assert_data_provenance` → `load_discriminator`：组别绑定、latent 形状/条件集对照、数据工件与词表指纹、形态指纹、checkpoint 严格装载）与「缺报告拒绝启动」的 schema 必填**全部保留**——退役的只是「白名单空拒绝开跑」的启动期门槛判定与启动期重算语义（后者在 ADR-0008 决策 5 落地时已废止）。预训练棘轮接棒：per-condition recon-AUC 过线判定（含支撑度规则）是预训练自身的终止判据，报告照常落盘过线条件清单与达标与否，供人工判读。

**显式不在取代面**：ADR-0008 决策 6（支撑度规则）——`reward/support.py` 与 `gate_support_min_volumes` 的唯一消费方是预训练棘轮（`SupportRule` 仅在 pretrain driver 构造），属棘轮判据统计形态，不在门控链。决策 1–4（条件匹配采样、容量守卫等）同此。

**保留面（监控面升格为唯一安全网）**：held-out AUC 逐 iteration 测量照旧落 iter 事件 `heldout_auc`（失去 EMA 消费方后测量与事件本身不变）；ADR-0009 分叉监控全链（OverfitMonitor、`overfit_alert` 事件、两 knob、resume `overfit` 键）只报警不动作；预训练 recon-AUC 测量 + 棘轮终止 + 换批复测全保留。

## Consequences

- **系统侧无任何自动动作的结构性保证**：门控退役后，代码中不存在任何「AUC 读数驱动训练决定」的机制——报警信号面（`overfit_alert` 事件、`heldout_auc` 序列跌回 chance 带）的裁决主体是维护者人工读 per-condition 曲线；人工裁决职责边界落档于 `docs/spec/reward-model.md`「防 reward hacking」节。
- RL 主循环少一个相位（`gating`）与一个 FSDP 集体跳过分支——每 iteration 的 policy 更新无条件执行；`phase_seconds` 相位集合随之收缩（字典键集收缩，解析无破坏）。
- 性能基线口径：async 执行模型迁移（#226）的性能基线采自本 ADR 落地后的旧执行序，「扣 gating 相」的历史防御性注记随之失效。
- 报告 schema 与事件 schema 的双收缩由既有断代机制承载：resume 走版本 bump 拒旧、报告走 `extra="forbid"` 拒旧键名——跨口径不可静默续用。
- 删除面不触碰 RNG 流与任何数值路径：`observe` 的 all_gather 集合调用消失，无数值/RNG 效应；同 seed 重放的确定性锚以删除后的执行序重新定锚。
