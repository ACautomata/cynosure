# 判别器监督与门控按条件化：条件匹配采样 + 条件白名单（修订 ADR-0007）

MR-RATE 换域调研（#70 → `research/discriminator-saturation-risk.md`）暴露两类阻断级缺陷。其一，判别器更新的 real 侧全池混采（`update.py`，`modality=None`），而 fake 侧单条件：MRA 占 real 池 0.02%，K=8 的 real 批永远抽不到 MRA，判别器任务退化为「MRA-fake vs 非 MRA-real」——最优解是分辨模态而非分辨真假，held-out AUC 门槛被轻松骗过，policy 被推向摧毁模态特征，且 AUC 高、reward 升两个监控通道同时失明。其二，调研以 BraTS T12/T13 一手数据证实「盲条件给 policy 的是满幅噪声梯度」：组内标准化 scale-invariant，AUC≈0.5 条件的 reward std 0.003 照样被放大到 O(1)，T12 唯一有信号的条件改善 −19%、三条盲条件全部恶化 +30~112%。同时池化 gate 口径两向皆错——多数盲条件可被单条件拉动假阳性过线，也可把「部分可用」压成假阴性拒跑。决定：**判别器的监督（采样）、上岗（gate）、消费（policy 梯度）三个口径全部按条件化**。本 ADR 修订 ADR-0007 的 gate 语义与数值部分（warm-start 预训练本体保留），其余 ADR-0001 决策不变。

**Status**: accepted

## Decision

1. **匹配键 = 目标模态**（`record.modality`，与 held-out AUC 现行归因口径同源）。组2 跨模态亦按目标模态匹配——source 影像不在 Real sample pool，配对语义不存在。
2. **条件匹配贯彻 fake 全侧**：Replay buffer 条目带模态标签，回放抽取按本 iteration 条件过滤；该条件回放不足时该步退化为纯 current 半区（回放 0 条）并落盘事件。base 分区 seeding 同步带标签。
3. **预训练改 per-condition 步进**：每步先抽一个条件（按目标模态均匀轮转），再量产该条件 fake 批、real 同条件匹配、AUC 归因该条件。终止 = 全部条件最近一次 per-condition AUC 过线即停；`pretrain_max_steps` 耗尽 → 白名单 = 已过线者；白名单空 → 拒跑。
4. **无放回采样语义不变 + 装配期容量守卫**：`RealPoolSampler` 采样语义不动（候选不足显式拒绝）；新增装配期逐 (rank 切片或全池, 模态) 校验 real 容量 ≥ K、base 分区该条件 ≥ 回放半区需求，不足 fail-fast（预训练 driver 已有同款先例守卫）。不引入有放回采样——小池 bagging 是过拟合加速器（ADA, arXiv:2006.06676）。
5. **gate 产物 = per-condition AUC 报告 + 条件白名单**：白名单空拒绝开跑（保留硬前置语义），非空即开跑；未过线条件不阻塞 run，由 Decision 7 的逐 iteration 门控兜底。
6. **门槛数值与统计形态**：0.65 保留为候选阈值（config 注释标明「MR-RATE 预训练曲线校准后定版」）；新增支撑度规则——条件 held-out 卷数 < 20 时，判定用 bootstrap CI 下界 ≥ 门槛而非点估计（MRA ≈ 16 卷命中，T2w ≈ 67 卷不命中）。
7. **梯度门控 = 白名单跳过 policy 更新**：条件不在白名单 → 该 iteration 跳过 policy 更新，rollout、fake 入 buffer、判别器更新照常。语义 = 「拒绝在 RM 无分辨率的样本上做策略梯度」（GRPO 无效样本不参与 advantage 的既有实践），不引入第二重 reward、KL 或参考模型。
8. **白名单动态恢复**：在线 per-condition AUC 流（iter 事件已按 `record.modality` 归因，信号现成）驱动——EMA(AUC) 越过 enter 阈值恢复更新、跌破 exit 阈值重新门控（暂定 enter 0.55 / exit 0.52 / EMA 跨度 8 iter，进 config 待校准）。gated 条件的判别器持续受训，是其建立判别力、白名单得以恢复的前提。

## Considered Options

- **维持 real 全池混采**：即现状。MR-RATE 上必然踩模态捷径（非概率事件），且监控双通道失明，否决。
- **Projection discriminator（条件判别器头，Miyato & Koyama, arXiv:1802.05637）**：文献正解，但新增 embedding 头 + 网络配置契约 + 消融锚的改动远大于条件匹配采样；后者已消掉捷径。留作「条件匹配后判别器仍无法在稀疏模态建立判别力」时的升级项。
- **std 阈值置零 advantage**：把「std 小」混同「无信号」——有信号但扰动弱的条件会被误杀（`policy-modeling.md` 的 sanity check 正是此情形），否决。
- **按 AUC 自适应缩放 advantage**：引入「AUC→缩放系数」映射的新超参面，且 AUC 是带噪估计、噪声乘噪声；P0 修复不引入新超参面，否决。
- **静态白名单（pretrain 定死、训练期不自动恢复）**：零新超参的最小面，但条件恢复要人肉改配置重启——判别器在 gated 条件上持续受训、判别力出带后却不能自动上岗，与 Decision 8 的目标矛盾。留作动态版的降级实现路径（配置开关）。
- **gate 池化口径维持 + 降门槛**：把 0.65 降到 0.55 会把噪声梯度机制原样放进来重演 T12；池化口径本身两向皆错（假阳/假阴），否决。

## Consequences

- `OnlineUpdate.step` 签名穿参（breaking change）：`step(current_fakes, modality)`，`UpdateReport` 随之带条件；`RewardCoordinator.update_step`、train 循环、预训练循环同批更新。
- `PretrainReport` schema 按条件化：单一 `final_heldout_auc` 标量 → per-condition dict + 条件白名单字段；`PretrainEvent` 增 `modality` 字段（事件契约「可扩不可改名」兼容；`iter` 事件已带 `modality`，无需改动）。
- Replay buffer 从裸 latent FIFO 变带标签 FIFO：续训状态分片（`resume_state_rank{R}.pt`）格式随条目标记变化——恢复对账按版本拒绝跨口径续训。
- **BraTS 线校准断层**：条件匹配对 BraTS 线同样生效（real 侧从混采变按条件），T12/T13 及更早 run 的数字不与新 run 横向比较；ADR-0007 的「冷启动失败组」断层标注之上再叠本断层。
- 新 config knobs：支撑度界（held-out 卷数 < 20）、动态门控 enter/exit/EMA 跨度——全部标注「暂定，MR-RATE 预训练曲线校准后定版」。
- 监控语义修复：盲条件不再产生假阳性 AUC 与噪声 reward；hacking 签名的「AUC 掉回带内」通道在按条件归因后才具备判型能力（冷/盲/被错误引导的签名分型另开 ticket）。
- CONTEXT.md：RM readiness gate 词条修订、新增「条件白名单」「条件匹配采样」词条、Replay buffer 词条补条件标记。
- 稀疏模态 real 侧小池（MRA ≈ 110 条）的过拟合防线（样本级增强 / 重复采样上限 / 配额）不在本 ADR 范围——容量守卫只防「抽不出」，不防「反复抽同一样」；另开 ticket。
