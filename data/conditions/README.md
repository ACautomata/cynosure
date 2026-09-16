# MR-RATE 条件词汇表工件（cynosure #127）

生成条件的**五元组登记表**（modality token / plane / 推荐 FOV / 统一网格 /
等效 spacing）——「条件 → latent 形状」的单一权威来源。装载面
`cynosure.conditions.MrConditionVocabulary`；config 经
`artifacts.condition_vocabulary_json` 携带路径（#125 spec 实现决策 1）。

| 文件 | 内容 |
|---|---|
| `mrrate_conditions.json` | 11 生成条件五元组（#81 白名单全量）+ whole-brain token 映射（上游权威 9/10/11/20/16）+ 普查引用 |

- **token 映射**：NV-Generate-CTMR `configs/modality_mapping.json` whole-brain
  条目；skull-stripped 29–33 不进本轮生成词汇。
- **统一网格**：#78 普查工件
  （`../eval/mrrate-baseline/census_latent_grid_distribution.csv`）逐条件
  众数 latent 网格 ×4（影像域，RAS 轴序，薄轴按 FOV 归位）。
- **推荐 FOV**：10 格取官方 `docs/inference.md` 推荐 FOV 表（RAS 轴序）；
  MRA（无 all-planes 官方行）取 #78 评估 manifest 中位（#80 基线同口径实测）。
- **等效 spacing**：推荐 FOV / 统一网格（条件属性，spec #125 决策 6）；
  合法域 [0.4, 5.0] mm（上游 `check_input_mr` 约束）。
- **装载期守卫**：缺格 / 多格 / 网格不符 / 字段缺失即字段级拒绝；生产模式
  恒 11 条件全量且逐条件网格对账普查众数；小词汇表只能经
  `fixture_mode=True` 显式装载（fixture 工件由 `Fixture.write_artifacts`
  落盘，不登仓）。

修改本工件须同步跑 `pytest tests/test_condition_vocabulary.py`——
测试内独立登记上游权威数值与普查导出值，与工件交叉复核。
