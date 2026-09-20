# 续训分片自持：协作者 state/adopt Protocol，取代 trainer 穿透

`resume.py` 对 `GranularGrpoTrainer` 组件树 15+ 处穿透——最深三跳（`trainer.rewards.update.optimizer`）、两跳（`trainer.policy.optimizer.param_groups[0]["lr"]`、`trainer.rewards.gating.adopt`、`trainer.rng.named()`）；trainer 以 8 个转发 property（unet/policy/rewards/updater/rollout/rng/device/amp，注释自认「既有公开访问面（tests 与 resume 模块消费）」）供养这条穿透路径。「什么进分片」的知识与 trainer 组件树逐层对应——组件每挪一层，分片键跟着变，与「per-rank 续训分片逐位复原」的核心不变式正面相抵（resume 格式 v4→v10 的每次演进都是这个耦合的代价）；`adopt` 的手写 dict 形态校验（逐字段类型检查 + 拒绝）三处同款（gating / overfit / DynamicWhitelist）。既有雏形：gating 与 overfit 已自持 `state()` / `adopt()`——自持模式存在且被验证，缺的是推广。**决定：分片读写知识自持到各协作者——Protocol `state()` / `adopt()`，trainer 聚合注册，resume 只跨 trainer 一道 seam；转发 property 随穿透路径一并退役，分片格式升 v11。**

**Status**: accepted（实施票未启动）

## Decision

1. **Protocol 命名沿用 `state()` / `adopt()`**：跟随 gating/overfit 两处既有先例，不引入 torch `state_dict()` / `load_state_dict()` 第二套命名。
2. **分片键由各组件自持声明**；adopt 的 dict 形态校验收成共享 helper 单点（三处手写副本退役）。
3. **8 个转发 property 全部删除**：deletion test 不过——property 的存在理由就是 resume 穿透，resume 走 seam 后是纯 pass-through；tests 直接消费的改为测协作者自身 interface（fixtures 已可独立构造，测试面错位由测试改、不由接口将就）。
4. **分片格式升 v11、拒绝旧版，不写迁移读取**：循 v10 清单退役先例与 `test_resume_rejects_legacy_v2_shard` 先例——格式口径变更以升版 + 显式拒旧表达。
5. **落地前置检查**：确认 sugon/gauss 上无待续训存量 run；若有，v11 落地时机排在该 run 结束之后。

## Considered Options

- **trainer 单点聚合（操作协作者提供的片段视图）**：resume 不再穿透属性链，但组件重排仍要动聚合处的键组织——locality 弱于自持，否决。
- **torch `state_dict()` / `load_state_dict()` 命名**：与仓库 state/adopt 先例并存两套语义相近的命名，增加而非减少认知负担，否决。
- **保留 tests 在用的转发 property、只删 resume 独占的**：pass-through 无存在理由；隔着 trainer 测协作者是测试面错位，否决。
- **写 v10→v11 迁移读取**：与仓库「升版拒旧」既定策略相反，为一次性兼容引入永久迁移面，否决。

## Consequences

- **新增面**：state/adopt Protocol + 共享 adopt 校验 helper + v11 分片格式；CONTEXT.md「分片自持」词条。
- **退役面**：8 个转发 property、resume.py 全部穿透路径、三处手写 adopt 校验、旧版分片载入。
- **核心收益**：组件重排不再动 resume（locality 落到组件自身）；trainer interface 由设计产生、不由消费者长出；v11 键组织获得组件自持的稳定锚。
- **风险（记录在案）**：v11 落地与存量 run 的时点互斥（决策 5 检查项）；分片键自持后「全分片清单」的完整性从 resume 单点转移到「 trainer 聚合注册是否漏组件」——以注册时逐组件登记 + resume generation marker 对账把守。
