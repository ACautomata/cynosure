# MR-RATE 全语料 >16.38M 长尾普查产物（cynosure #141）

零下载元数据普查的机器可读产物；口径、公式出处与结论见
`research/mrrate-tail-census.md`（阅读顺序：先报告后本目录）。

| 文件 | 内容 |
|---|---|
| `tail_summary.json` | 主数字：全语料条目数、>16.38M 卷数与占比（总体 + 分 split）、体素分位数、top 超界网格、审计计数 |
| `grid_buckets.csv` | 按 resize 后 canonical 网格（三轴降序）分桶主表：image/latent 体素数、超界标记、计数（total/train/val/test）、占比 |
| `over_by_modality_plane.csv` | 超界卷 (modality × plane) 分布与逐格超界率 |

- 阈值：`320×320×160 = 16,384,000` 影像体素（【上游】`utils.py:787-817` `dynamic_infer` 豁免边界，≤ 走整前向）。
- 公式：逐轴 `max(round(a/128),1)×128`，round 为 Python half-to-even（【上游】`diff_model_create_training_data.py:77-89,247-249`）。
- 数据源：`gauss:/data72/junran/mrrate_replay/metadata/batch[00-27]_metadata.csv` + `splits.csv`
  （HF `Forithmus/MR-RATE` 快照 `f6e39794`，与评估集普查 #78/#91 同源）。
- 脚本：`gauss:~/mrrate_tail_census.py`（临时脚本不入库；复现命令见报告 §7）。
