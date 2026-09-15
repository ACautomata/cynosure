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

## 集群环境分派（测试走 gauss，训练部署走 sugon）

- **测试一律上 gauss**（`ssh gauss`，4× RTX A6000，驱动 575 / CUDA 12.9）：NVIDIA 栈独占性可预期，不受共享实例抢占；sugon 共享 DCU 实例只用于训练与部署。
- 代码同步走 **rsync**：本地工作树直传 `gauss:~/cynosure/`（未提交改动照传，无需先提交）；排除清单与完成判据沿用 sugon-deploy 的「代码上集群」口径，主机换 gauss。
- gauss 上用专用 venv 跑测试：`/tmp/cyno-gauss/venv/bin/python -m pytest`（仓库 `pythonpath=["src"]`，无需安装本包）。venv 在 /tmp 易失，重建分两步（/home 配额紧张，缓存与临时目录一律指 /tmp）：

  ```bash
  python3 -m venv /tmp/cyno-gauss/venv
  TMPDIR=/tmp/cyno-gauss /tmp/cyno-gauss/venv/bin/pip install --cache-dir /tmp/cyno-gauss/pipcache \
      torch "monai>=1.6,<2" "pydantic>=2.7" "einops>=0.8" "numpy>=1.26" "nibabel>=5" "pytest>=8" "pytest-xdist>=3"
  TMPDIR=/tmp/cyno-gauss /tmp/cyno-gauss/venv/bin/pip install --cache-dir /tmp/cyno-gauss/pipcache \
      --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1"
  ```

  第二步是驱动守卫：默认 wheel 为 cu130 构建，超出驱动 575 承载（CUDA 直接不可用），必须锁 cu128。
- 测试分两档：日常开发 `pytest` 默认跳过 `slow` 标记的特别耗时测试（多 iteration 完整训练 / torchrun 多进程 / 像素域解码评测 / 满步数真实 rollout 的状态机轮次）；全量验证以 gauss 仓库根目录的 `pytest --run-slow` 全绿为准，本机执行不计数。
- **训练与部署走 sugon**：真实 DCU/DTK 加速卡栈（`ssh sugon`），系统 python 为 DCU torch 唯一宿主，`.venv/` 惯例到本机为止。集群级打底（SSH 别名、双 source、DCU 依赖陷阱）见用户级 **sugon-bootstrap** skill；部署与训练全流程见项目 **sugon-deploy** skill。

## Agent skills

### Issue tracker

Issues are tracked in this repo's GitHub Issues (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label triage vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
