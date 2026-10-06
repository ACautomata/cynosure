# cynosure

cynosure 为 MAISI 3D latent rectified-flow 医学影像 checkpoint 设计并实施基于 Granular-GRPO 的 RL 后训练。**零依赖原则**：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；网络类来自 MONAI 库本身。

实施 spec 见 GitHub issue #15（配合 `docs/spec/` 四章节与 `docs/adr/`）；领域术语表见 `CONTEXT.md`。

## 开发

```bash
# MONAI/torch 对 Python 3.14 无兼容 wheel，本地 venv 用 3.12–3.13
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest                     # 本地测试入口（CLI seam + fixture + 静态零依赖检查；默认跳过 slow 大轮次）
pytest --run-slow          # 全量（含 slow 大轮次：完整训练 / 多卡 e2e / 像素域评测）
```

CLI：

```bash
cynosure train   --config config.json   # 训练（run 目录 + 指标流 + checkpoint）
cynosure eval    --config config.json   # 里程碑评测
cynosure prepare --config config.json   # Real sample pool / held-out / 统计量构建
```

三子命令共享同一 config schema（全量配置项 + 定死/tunable 状态标注，见
`src/cynosure/config.py`）；config 校验失败时输出字段级错误并以退出码 2 拒绝。

train / pretrain 以**单进程多卡**执行（async 执行序，#217/#226：每卡完整
副本 + 静态分配表 + 逐 k barrier 收集-同步；进程内多卡由设备发现承担）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 cynosure train \
    --config config.json --run-dir /root/private_data/cynosure/runs/<run>
```

- torchrun 多进程启动显式拒绝（多进程拓扑属已退役的旧执行序）；
  卡集裁剪经 `CUDA_VISIBLE_DEVICES`；
- 调度槽数 = `execution.coroutines`（缺省 = 卡数）；
- 续训 `--resume --run-dir <run>`：v12 单文件分片
  （`checkpoints/resume_state.pt`），恢复须同协程数拓扑（卡数不进对账）；
- 多卡语义的机器面锚（跨卡 allreduce / 重放逐位）由集群 `--run-slow`
  全量覆盖，见 `docs/adr/0018`（pretrain driver 同款执行形态）。
