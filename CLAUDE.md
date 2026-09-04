# cynosure

## 项目结构

- Python 项目，采用 **source layout**：包代码位于 `src/cynosure/`，不在仓库根目录。
- 构建后端为 **hatchling**，项目元数据与依赖定义在 `pyproject.toml`。
- 实验期间编写的临时脚本（启动、结果计算）一律写进 run 目录的 `scripts/` 子目录，不提交进仓库；发布时随 experiment-release skill 归档进实验 release。

## 环境与命令

- 虚拟环境使用 **venv**，位于 `.venv/`：

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -e .            # 以可编辑模式安装本项目（hatchling 后端）
```

- 运行 Python、pip 或任何项目工具前，先激活 `.venv/`；不要把依赖安装到全局解释器。
- 新增依赖写入 `pyproject.toml` 的 `dependencies`（开发依赖写 `optional-dependencies`），不要引入游离的 requirements 文件。

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label triage vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
