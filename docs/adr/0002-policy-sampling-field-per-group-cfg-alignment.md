# Policy 采样场按实验组对齐基座实际推理行为（per-group CFG）

RL 的 policy 定义为「与基座各阶段推理**实际生效行为**逐字对齐的采样场」：组1（模态标签）基座推理用 CFG=10 → policy 是 CFG=10 组合场（batch=2 `chunk(2)`，无条件分支=全零 label）；组2（跨模态 ControlNet）基座推理强制 CFG=0 → policy 是裸条件单前向。对齐的是实际行为而非 config 字面——例如 scheduler config 写 `"scale": 1.4`，但 MONAI 各版本从不把该参数传入 timestep transform（死参数），实际生效 1.0，复刻日程必须按 1.0。

**Status**: accepted

## Considered Options

- 两组统一 CFG=10 组合场（算力形状一致）：组2 与基座推理分布不一致，RL 会优化一个推理时不会出现的分布，且放弃组2 单前向的算力节省。
- 两组统一 CFG=0 单前向（最省算力）：组1 与基座 CFG=10 分布有 gap，reward 在错误分布上评估。
- **per-group 对齐（选定）**：分布正确性优先；组2 恰好单前向，算力反而更省。

## Consequences

- log-prob 必须在各自采样场上计算：组1 ratio 的每次评估是 batch=2 前向，组2 单前向——算力与显存预算按组不同，编排（ticket #7）需感知。
- 组1 的无条件分支与 G 个方向共享同一 `x_k`：`v_uncond` 每次 batch=1 评估一次全组复用，不随条件分支复制。
- 复刻 sigma 日程禁止照抄 config 字面值（`scale=1.4` 死参数陷阱）；以 MONAI `set_timesteps` 的实际输出为准。
