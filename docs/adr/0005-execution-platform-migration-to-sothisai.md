# 执行平台迁移：zzeshell SLURM 集群 → 曙光云 SothisAI DCU 实例

RL 后训练的执行平台从 zzeshell SLURM 集群迁移到曙光云 SothisAI DCU 实例（实测 2026-09-03：BW/gfx936，4×64GiB/实例，DTK 26.04，torch 2.9.0 DCU 构建，持久分区 `/root/private_data` 1000G、系统盘 `/` 易失）。背景：zzeshell SLURM 集群访问权永久失去（无恢复预期）；SothisAI 与 zzeshell 同属 scnet 超算互联网体系、同为 BW/gfx936 卡型、同 DTK 26.04 软件栈——是迁移成本最小的新平台。编排裁决不变：torchrun + FSDP 同卡交替（ADR-0003 原地不动）；跨实例多节点降格为未验证远期选项；config 的 `SlurmConfig` 由 `DeploymentConfig` 取代。部署细节见 `docs/spec/orchestration.md`「实例部署结构」。

**Status**: accepted

## Considered Options

- **留守 zzeshell**：不可行。SLURM 集群访问权永久失去（2026-09-03 确认），无恢复路径。
- **迁移到 SothisAI DCU 实例（选定）**：同属 scnet 超算互联网体系、同卡型 BW、同栈 DTK 26.04——改动收敛在平台假设层（编排章部署与门槛节、config 部署字段、实例冒烟脚本），算法层（policy / reward / GRPO / 实验设计）与唯一代码 seam（CLI + config schema）零改动。
- **等待其他 SLURM 集群配额**：弃。无到货时间承诺，阻塞开工；且目标同为 DCU 栈的话，后续迁移成本与本次相同，先迁无损失。

## Consequences

- 「Slurm 作业结构」被「实例部署结构」取代：无作业调度器，生产与调试统一 SSH + tmux/nohup + torchrun；单实例 4 卡 `torchrun --nproc_per_node=4`（原 zzeshell 单节点 8 卡）。
- 实例系统盘 `/` 易失（实例重置即丢环境与产物）：双 source（DTK `source /opt/dtk/env.sh` + 平台代理）可幂等注入 bashrc，实例冒烟脚本（M0 ⑥）一键重建与验证；checkpoint 与产物只落持久分区 `/root/private_data`，绝不落系统盘。
- 断点续训语义从「跨 sbatch 恢复」改为「跨实例重启/重置恢复」：checkpoint 周期（每 10 iteration + 每里程碑强制）不变。
- M0 开工前门槛七条 → 六条：删多节点 RCCL 条（跨实例多节点 = 未验证远期选项，不作路径承诺、不进验证清单），其余在新平台重验。
- 4 rank（原 8 rank）下 rollout 吞吐约为原来一半、每 iter wall-clock 约 ×2——已定决策：iteration 数 200–500 不动，接受 wall-clock ×2，必要时运行期再议缩 iteration。
- config 部署字段替换发生在同一 seam 面内：`DeploymentConfig`（`nproc_per_node=4`、产物根目录 `/root/private_data/cynosure`）取代 `SlurmConfig`（partition/gres/ntasks/num_nodes/walltime），旧 slurm 段被 schema 显式拒绝。
