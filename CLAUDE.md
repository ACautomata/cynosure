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

## 测试（一律上集群）

- 所有测试一律上集群执行：sugon 实例的真实 DCU/DTK 加速卡栈（`ssh sugon`）。集群上用系统 python（DCU torch 唯一宿主），`.venv/` 惯例到本机为止。
- 同步走 **git**：本地提交后 `ssh sugon 'cd /root/cynosure && git pull'`，两侧 `git rev-parse HEAD` 一致即同步完成。
- 验证以集群上仓库根目录的 `pytest` 全绿为准；本机执行不计数。缺 pytest 时装进系统 python。
- 集群级打底（SSH 别名、双 source、DCU 依赖陷阱）见用户级 **sugon-bootstrap** skill；部署与训练全流程见项目 **sugon-deploy** skill。

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label triage vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
