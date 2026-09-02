# 编排架构方案：torchrun + FSDP 同卡交替（Ray 为升级项）

> 本章是整体实施 spec（地图 #2「CTMR Granular-GRPO RL 后训练方案」）的编排章节，由 ticket #7 决议产出。终稿由 ticket #9 汇总。策略侧衔接 `docs/spec/policy-modeling.md`（ticket #5）、reward 侧衔接 `docs/spec/reward-model.md`（ticket #6）。

## 总原则：单一角色，最小编排

本 workload 的角色图坍缩为**单一同构角色**——没有 vLLM 等效快推理引擎（rollout 就是 policy UNet 的前向）、没有参考模型（KL=0）、没有文本/tokenizer、reward 是几层 3D conv。因此不存在「rollout engine / trainer / reward / reference」的多角色切分需求；编排只做一件事：**起 N 个同构 rank，各自同卡交替采样 + 训练**。编排层越薄越好。

## 裁决：两层分开选（先纠类别错误）

「Ray vs FSDP」是类别错误：**Ray = 编排层**（调度/角色切分/异步/容错），**FSDP / DeepSpeed ZeRO = 分片层**（参数怎么切、梯度怎么 allreduce）。专业 RL 框架是两层都用（verl = Ray + FSDP/Megatron；OpenRLHF = Ray + ZeRO-3）。本 workload 两层分别裁决：

| 层 | 选定 | 弃 |
|---|---|---|
| 编排 | **torchrun**（多节点 rank launch + `torch.distributed` init） | Ray Core / Ray Train |
| 分片 | **FSDP full-shard + 梯度检查点** | DeepSpeed ZeRO-3（仅显存 fallback） |

Ray 不进入主路径：它的三项独有能力（actor 级容错、placement groups、异步角色池）在本 workload 全部用不上——训练是同步的 FSDP 集体通信 lockstep，中途节点故障必须整作业 checkpoint 重启（Ray 的 actor 重启无法救回 in-flight allreduce）；且 Ray-on-DCU 需付全额的设备可见性地雷 + 实验性代价（`research/ray-on-dcu-slurm.md` §1.3）。**Ray 记为升级项**：若未来加独立 rollout 引擎或 KL/参考模型，再启用（且须先过 DCU 实测）。

## 进程拓扑（ticket 问题①）

- **N 个同构 FSDP rank，无角色划分**。每 rank 在同一进程内交替两相（照搬参考实现 `train_g2rpo_hps.py` 的 eval/train 交替）：
  1. `eval()` + `no_grad`：anchor ODE 轨迹 → 单步 SDE → ODE 续跑 → latent 域打分（见 `policy-modeling.md` / `reward-model.md`）；
  2. `train()`：每个训练步 k 一次独立 forward → backward → optimizer.step（|M| 次），再判别器在线更新。
- **无独立 rollout worker、无 reward actor、无参数服务器**；rollout → train 是同进程内的 mode toggle，不是跨 actor 权重同步。
- **显存规划**（目标 zzeshell BW1000，gfx936，8×64GB/节点）：policy UNet 约 1e8 级参数量（用户口径，待 checkpoint 实测校正）→ fp32 master + AdamW 优化器经 FSDP full-shard 后每卡仅数百 MB；**内存不是瓶颈**。真正的绑定是 **rollout 吞吐**（每 iter 数百至上千次 CFG-batched UNet 前向，train:rollout ≈ 5%）。G=12 轨迹 latent 每张 `[4,64,64,32]` ~1–2MB，全量常驻也仅数百 MB，但**仍以 profile 实测为准**（见「开工前门槛」）。

## Slurm 作业结构（ticket 问题②）

- **单节点优先（默认）**：`sbatch --partition hx1hdnormal --gres=dcu:8 --ntasks-per-node=1`（一个节点 8 卡），`torchrun --nproc_per_node=8`。8×64GB=512GB 对 ~1e8 参数量富余，且彻底回避跨节点 RCCL。
- **多节点 = rollout 吞吐升级杠杆**：本 workload 瓶颈在 rollout，更多 rank = 每 iter 采样更多组。单节点 wall-clock 不够时 `torchrun --nnodes=2+`（`--gres=dcu:8` × N 节点），走跨节点 RCCL。
- 作业内：`module load compiler/dtk/26.04` + `source /opt/dtk/env.sh`（DTK 环境）；计算节点离线，checkpoint/中间产物落 `$HOME`（/public 持久，**不落 /tmp**，TMP_SHARED=no）。
- 生产走 `sbatch`；调试走 `salloc` / `srun --pty bash`（终端断开即失效）。
- MAX_WALLTIME=3 天；长训需**断点续训**（policy + 判别器 + 回放缓冲 + optimizer state 定期落盘，跨 sbatch 恢复）。

## 权重同步（ticket 问题③）

**蒸发**。无独立 rollout worker——rollout 读的就是当前进程内同一份 FSDP 分片权重，参数更新经 **allreduce + `dist.barrier()`** 生效。不存在「policy 更新后分发到 rollout worker」的 NCCL broadcast / 共享存储 checkpoint 路径；跨 rank 唯一通信是 FSDP 梯度 allreduce（RCCL）。

## 判别器在线更新：数据并行

PatchDiscriminator 几层 3D conv、算力可忽略，**不参与 FSDP 分片**，每 rank 持一份完整副本：

- 用**本 rank 的 fake latent** + **共享 real 池的切片**更新（real 池 = 训练集 VAE 预编码 latent，固定不更新）；
- 梯度 **allreduce**（标准 DDP），回放缓冲 **per-rank FIFO**（`reward-model.md` 的封顶 FIFO）。

## 降级预案（ticket 问题④）

主路径已是 torchrun，降级方向**反转**为「多节点 → 单节点」：

1. 多节点 RCCL 不稳 → 退回单节点 `--gres=dcu:8`；
2. FSDP full-shard 出问题 → 退 **DDP**（~1e8 参数量单卡可整存）；
3. 显存仍紧 → ZeRO-3 / offload（fallback，非默认）。

最高风险项与编排无关、两路都要验：**DTK pytorch-dcu wheel + `numpy==1.26.4` pin（装任何依赖后回验 `import torch`）+ RCCL 调优**。**Ray 是升级项，不是降级项**。

## 开工前门槛（sanity / 验证清单）

1. DTK 环境 `import torch` 通过（source DTK 后）；钉 `numpy==1.26.4`，装 ML 依赖后回验。
2. gfx936（BW1000）上 `bf16 autocast + fp32 master` 生效（profile 验证 dtype）。
3. 单节点 8 卡 RCCL allreduce 正常（FSDP 分片 + 梯度同步）。
4. G=12 × 30 步轨迹 + MGAI 终点的实际显存（64GB/卡）实测；不够则降 G 或逐 k 释放。
5. **rollout 吞吐 profile**（真正绑定项）：单节点 8 rank 的每 iter wall-clock，决定是否需多节点。
6. Slurm 模板 `--gres=dcu:8 --ntasks-per-node=1` + DTK module/source 在 DCU 队列跑通。
7. （如需多节点）跨节点 RCCL allreduce 与权重同步实测。

## 待定 / 移交

- UNet 精确参数量与激活显存 → checkpoint 实测校正（本章按 ~1e8 口径）。
- G 显存实测、逐 k 释放、rollout 吞吐、单/多节点最终定 → profile（施工，超出本地图 spec 范围）+ ticket #8 实验设计。
- 判别器更新节奏 N/K、LR、buffer 容量 → 依 rollout 吞吐 profile 后定（`reward-model.md` 已标 tunable，移交 ticket #9 终稿）。

## 依据

- `research/ray-on-dcu-slurm.md`（Ray-on-DCU 可行性、§1.3 设备地雷、§2 Slurm 部署、§3.2 diffusion-RL 单节点常态、§5.3 验证清单）。
- `research/granular-grpo.md`（参考实现 §5：torchrun 2×8 + `torch.distributed.fsdp` full-shard + 梯度检查点、同卡交替、无 Ray/无独立 rollout worker）。
- `docs/spec/policy-modeling.md`、`docs/spec/reward-model.md`（rollout 是 UNet 前向、无 vLLM、无参考模型、latent 域打分、real=VAE 预编码）。
- `scnet-hpc/clusters/zzeshell.conf`（BW1000 gfx936、8×64GB/节点、131 节点、DTK 26.04、bf16 支持、`--gres=dcu:N`）。
- 对抗验证（ticket #7 内）：21 条 claim 各经 3 反方驳斥，7 条存活、14 条被驳，合成 verdict=torchrun+FSDP。反方确证 verl 有 diffusion-RL/FlowGRPO 模式——「Ray 不适合 diffusion-RL」为假；正确边界是「本 workload 无 vLLM/无参考模型/无文本，单一角色让 Ray 角色分离无利可图」。
