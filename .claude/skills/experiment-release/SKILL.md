---
name: experiment-release
description: 实验完成后把 run 目录结果、实验临时脚本与运行源码打包发布为 GitHub Release（tag exp/<run-dir名>，源码 tarball 按运行 commit 用 git archive 固定）。触发：一次 train/eval run 跑完需要发布结果；用户要求发布/上传实验结果或建实验 release。
---

# 实验发布：run 目录 → GitHub Release

一次发布 = 一个 run 目录 + 产生它的源码 commit，绑成一个 GitHub Release。两条纪律：

- **溯源**：结果必须与实际跑出它的源码树绑定，绑错 commit 整个 release 作废；
- **临时脚本只活在两处**：run 目录的 `scripts/` 与 release 资产，唯独不进仓库。
  启动脚本、结果计算脚本（指标汇总、图表、评测）都按此办理。

## 步骤

### 1. 定位并校验 run 目录

run 目录须在本地（还在 SothisAI 实例上的先按 sugon 惯例拉回），且在仓库树外。
三件套齐才算可发布：

- `config.json`、`manifest.json` 存在且非空；
- `metrics.jsonl` 至少一行事件。

缺件或空流 → 停下向用户报告，不发布残缺 run。

顺手盘点 `scripts/`：空或缺失不算残缺，但启动与结果计算必然跑过——脚本散落
在实例他处或 repo 工作区的，先收进 `scripts/`；确实没有（如纯 CLI 启动）向
用户确认一句。

### 2. 溯源：确定运行 commit

找到实际运行这次实验的源码 commit，可靠来源只有两处：

- 本会话刚启动的 run：用启动训练时的 HEAD commit；
- 其余一律问用户——本地 HEAD 往前走了，从 HEAD archive 出来的就不是
  跑出这份结果的代码。

运行时工作树带未提交改动 → `git archive` 拿不到它们：请用户先把改动提交
（或指认一个等价 commit）再继续。未提交改动指**被跟踪源码的改动**；散在
工作区的临时脚本不算，也不许借这一步提交进仓库。

### 3. 打包资产

- 源码：`git archive --format=tar.gz --prefix=cynosure-<short-sha>/ -o
  <run-dir>/source-<short-sha>.tar.gz <commit>`；
- 脚本：从 `<run-dir>/scripts/` 按一条标准挑——release 里没有它，换个人能否
  复现启动与结果计算？能收的打成 `scripts-<short-sha>.tar.gz`（`tar -czf …
  -C <run-dir>/scripts <挑出的文件>`）；一次性探针、临时 sanity 这类过不了
  标准的留在目录里，不上；
- 结果：run 目录下的 `config.json`、`metrics.jsonl`、`manifest.json`，以及
  除 `checkpoints/`、`scripts/` 外的其他结果文件（图表、评测报告等）；
- checkpoint 默认不上 release：权重留集群持久分区，用户点名要的才打。

每个资产过一遍 `du -h`：单资产上限 2 GiB，超限挡下（GitHub 拒收，release
会建到一半失败）。上传前扫一遍资产内容——release 是对外可见面，别让密钥
或内部信息混进去（临时脚本最爱夹带密钥、代理账密与内部路径，重点扫）。

### 4. 起草 release notes

从 `metrics.jsonl` 与 `config.json` 汇总，中文、短：group、seed、末次
iteration 与 `anchor_eval_reward`、里程碑 FID 轨迹、`heldout_auc` 末值、
是否 early-stop；标出源码 commit short sha；末尾一行列出脚本 tarball 里
收了哪些脚本。写入临时文件供 `--notes-file`。

### 5. 发布并验证

```bash
gh release create "exp/<run-dir 目录名>" <资产...> \
  --target <commit> --title "实验 <group> @ <时间戳>" --notes-file <notes>
```

`--target` 把 tag 钉在运行 commit 上——tag 即溯源锚点。目录名不适合做 tag
（空格等）时先问用户。tag 已存在即失败，报告用户，不覆盖。

完成后 `gh release view "exp/<run-dir 目录名>"` 逐项核对：资产清单与第 3 步
一致（含 `scripts-<short-sha>.tar.gz`）、notes 要素齐全。核对通过才算发布
完成。
