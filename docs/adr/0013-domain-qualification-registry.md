# 域资格登记表：dataset 域知识单点化，取代字面分支散布

`experiment.dataset == "MR-RATE"` 字面分支全仓 14 处 / 8 文件——schema 层 7 个 validator（互斥绑定：MR-RATE 缺工件即拒、BraTS 携带即拒，#127/#121/#131/#130 的延续形态）、运行时 6 个分派点（词汇表装配 `ConditionVocabulary.assemble`、schedule 装配 `TrainingRuntime.assemble_schedules`、policy 构建 `GroupPolicy.build`、prepare 装配 `PreparePipeline.build_assembly`、eval 取数 `ManifestEvaluation._build_reals`、baseline manifest 条件解析 `BaselineManifest._stage_conditions`）、fixtures 测试构造 payload 分支。schema 无域分离：MR-only 字段（`reward.real_pool_quota`、`heldout_fraction`、`heldout_quota_volumes`、`reward.sampling_manifest_json`）BraTS config 也必须携带，只能靠 validator 事后互斥拒绝；「字段→域」知识三份表达（validator 群 / 分派点 / fixtures）永远同步。每张换域票（#127→#129→#130→#131）都在 validator 群与分派点各加分支，`test_config_schema.py`（1239 行）逐 validator 镜像同步膨胀。**决定：新建 `src/cynosure/domains.py` 域资格登记表——dataset → 域内字段集 / 工件集 / 错误文案 / 按域取行为查询入口的单一事实源；validator 退化为薄查表循环，运行时分派改走查询入口，调用方不再写字面分支。** schema 结构不动（注册表收拢，非 discriminated union）；下层 seam（`ConditionVocabulary` / `ReferenceVolumes` / `AssemblyPlan` / `ConditionSchedules` 协议）已 Protocol 化，分派点散布是写法 locality 问题、不是接口形状问题。

**Status**: accepted（实施票未启动）

## Decision

1. **注册表条目 = 域标识 + MR-only 字段集 + 工件集 + 错误信息文案**：文案与字段清单同一知识单点（#127 互斥哲学的表述跟随清单走），validator 变薄查表循环、错误信息仍保持现有详尽中文口径。
2. **validator 测试镜像同步收敛**：`test_config_schema.py` 逐 validator 镜像瘦身为「注册表单点测试 + validator 接线抽样」——validator 只需证明查表接线正确，域语义测试集中在登记表。
3. **运行时 6 分派点消费「按域取行为」查询入口**：各分派点语义独立（装配词汇表 ≠ 装配 schedule ≠ eval 取数），收的是查询入口、不是把六个行为捏成一个函数；调用方不再出现 `experiment.dataset ==` 字面串。
4. **不为第三域设计**：注册表按「域标识 → 条目」形态组织，新域 = 加条目不改消费面——顺带能力，非设计目标（YAGNI）。
5. **schema 分域（discriminated union）为后续演进选项，本票不做**：结构硬保证（MR-only 字段在 BraTS 类型上不存在）但 `test_config_schema.py` + fixtures + `_serialize_without_mr_excluded_fields` 等序列化特殊逻辑全部重写，爆炸半径与当前摩擦不成比例。

## Considered Options

- **schema 真分域（discriminated union on dataset）**：结构上不可能携带错域，但测试面 1239 行镜像、fixtures、序列化特殊逻辑全重写；当前摩擦是知识散布而非 schema 表达力不足，否决（留作决策 5 的演进选项）。
- **注册表只收 schema 层**：运行时 6 处字面分支仍在，第三域仍要 6 处加分支，locality 收益减半，否决。
- **注册表只返回域枚举、调用方保留 switch**：14 处字面串换成 14 处枚举 switch，靶心（消灭散布分支）未中，否决。
- **注册表内嵌 config.py**：1513 行文件继续膨胀；域资格横跨 schema/运行时/测试三面，是独立概念，否决。

## Consequences

- **新增面**：`src/cynosure/domains.py` 登记表 + 查询入口 + 注册表单点测试；CONTEXT.md「域资格」词条。
- **收敛面**：7 个 validator 的重复互斥模式、6 处运行时字面分支、fixtures payload 域分支、config schema 测试镜像。
- **风险（记录在案）**：查询入口若设计成转发层（只查字段不承载「按域取行为」语义）会通不过 deletion test——入口必须收行为分派而非转发 config 字段；实施票评审以此为验收轴。
- **BraTS/MR-RATE 两域行为零变化**：本票动分派写法，不动任何域行为；既有全部域互斥测试应原样通过（错误信息文案经决策 1 保持等值）。
