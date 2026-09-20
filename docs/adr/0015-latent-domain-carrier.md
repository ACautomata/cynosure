# Latent 域载体 DomainLatent：域标签构造期核对，取代手写乘除纪律

`latent_scale_factor` 出现 13 个文件；乘除动作 6 处——`train/rollout.py` `_to_pool_domain`（除回存储域）、`reward/assembly.py` 重构核乘入 + 除回（ADR-0012 同源重构）、`eval/decode.py`（除回）、`smoke.py`（乘回）、`policy/field.py` BareConditionField 源 latent（乘入）；来源裁决 3 处（`netbuild.checkpoint_scale_factor`、`smoke._resolve_scale_factor`、`config.policy.latent_scale_factor`）。「这个 latent 此刻在存储域还是 policy 工作域」没有类型表达，全靠调用点纪律 + 注释维持（CONTEXT.md「存储域」词条锚）；ADR-0012 同源重构使乘除点翻倍，stage-2（跨模态 ControlNet）还将增长。这是全仓唯一「出错不抛错、静默错域」的知识——其余架构摩擦 fail-fast，唯域漂移静默。**决定：新建薄包装类 `DomainLatent`（`policy/latent.py`）——latent 张量 + 域标签，域间换算只存在于载体语义级方法；无裸函数约束（维护者强约束，落档防未来 review 重提）；一次性切换 13 文件，不留双口径。**

**Status**: accepted（实施票未启动）

## Decision

1. **域枚举两值：存储域 / policy 工作域**。「干净域 vs 带噪域」（ADR-0012：判别器输入恒干净域）不进本枚举——该维度当前由同源重构构造原语的 interface 形状保证（加噪只发生在 fake 构造输入端），留第二票。
2. **换算为语义级载体方法（`to_working` / `to_storage` 命名方向）**，载体方法是唯一被允许触碰 `latent_scale_factor` 的位置；消费点退化为声明域，不再手写乘除。
3. **无裸函数约束（维护者强约束）**：latent 域操作一律挂载体类方法下，模块级裸函数不得触碰域换算；与域无关的统计裸函数（`auc_from_scores`、`bootstrap_ci_lower_bound`——口径单点健康面）不受约束。
4. **一次性切换、不留双口径**：同一票内先立载体、随即把 6 处裸乘除全切完——并存期「新旧混用」恰是静默错域的温床。验收 = 裸乘除归零 + 既有 roundtrip/逐位一致测试全绿（数值零变化）。
5. **Tensor 子类方案排除**：autocast 遍布 15 文件，autocast 产出的张量不保证保留子类类型——自动传播的前提不成立。

## Considered Options

- **函数式域函数单点（`to_working_domain` 等裸函数收单点，张量裸传）**：把纪律从 6 处散布收成单点已消除主要摩擦、半径小得多，但不解决张量身份——调错方向函数仍静默；维护者裁决取硬保证（构造期核对），否决。
- **每域一个子类（StorageLatent / WorkingLatent）**：域判定从枚举比较升级为类型分派，更硬；但 pydantic/批处理装箱、序列化各多一层，YAGNI，否决。
- **Tensor 子类（域随张量自动传播）**：见决策 5，autocast 兼容风险有实据，否决。
- **新旧并存渐进迁移**：双口径并存期正是静默错域的高发窗口，与本票的动机自相矛盾，否决。
- **干净域纳入同一枚举**：现有保证是构造原语 interface 形状而非类型系统，纳入是真实改进但当前守卫已足；扩域留第二票，否决（本票范围）。

## Consequences

- **新增面**：`policy/latent.py` 的 `DomainLatent` 载体 + 域枚举 + 语义级换算方法；CONTEXT.md「Latent 域载体」词条。
- **收敛面**：6 处手写乘除、13 文件的 scale factor 认知负担、来源裁决 3 处的重复；stage-2 新消费点不再重学整条域链。
- **风险（记录在案）**：包装类穿过 rollout / 同源重构 / 判别器测量 / decode 全链，类型改动半径 13 文件——由一次性切换（决策 4）+ 既有逐位一致测试把守；批处理路径（批内 latent 装箱）需要载体感知的容器约定，实施票内定。
- **stage-2 到来时干净域维度再议（第二票）**：跨模态阶段新增（源影像, 目标标签）条件化前向，域链将再一次扩张——那是重估干净域入枚举的自然时点。
