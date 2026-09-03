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
