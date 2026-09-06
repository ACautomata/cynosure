# cynosure

cynosure 为 MAISI 3D latent rectified-flow 医学影像 checkpoint 设计并实施基于 Granular-GRPO 的 RL 后训练。**零依赖原则**：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；网络类来自 MONAI 库本身。

实施 spec 见 GitHub issue #15（配合 `docs/spec/` 四章节与 `docs/adr/`）；领域术语表见 `CONTEXT.md`。

## 开发

```bash
# MONAI/torch 对 Python 3.14 无兼容 wheel，本地 venv 用 3.12–3.13
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest                     # 本地测试入口（CLI seam + fixture + 静态零依赖检查）
```

CLI：

```bash
cynosure train   --config config.json   # 训练（run 目录 + 指标流 + checkpoint）
cynosure eval    --config config.json   # 里程碑评测
cynosure prepare --config config.json   # Real sample pool / held-out / 统计量构建
```

三子命令共享同一 config schema（全量配置项 + 定死/tunable 状态标注，见
`src/cynosure/config.py`）；config 校验失败时输出字段级错误并以退出码 2 拒绝。

分布式训练（torchrun + FSDP full-shard，判别器 DDP 不分片）：

```bash
torchrun --nproc_per_node=4 -m cynosure.cli train \
    --config config.json --run-dir /root/private_data/cynosure/runs/<run>
```

- 分布式启动必须显式 `--run-dir`（默认目录按进程时间戳生成，多 rank
  无法对齐）；产物 checkpoint 由 rank 0 独写、指标流由 rank 0 归并
  （事件按 (iteration, rank) 顺序稳定）、续训状态每 rank 一个分片文件
  （`checkpoints/resume_state_rank{R}.pt`）；
- 单进程（无 torchrun 环境）与分布式走同一条训练循环；续训须以同一
  world size 恢复（跨拓扑续训被拒绝）；
- 本地 CPU fixture 多进程验证用 gloo 后端；DCU/RCCL 集群侧门槛见
  实施 spec（issue #15）M0 清单。
