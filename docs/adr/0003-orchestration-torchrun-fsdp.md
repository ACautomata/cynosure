# 编排架构：torchrun + FSDP 同卡交替，Ray 为升级项（非基线）

RL 后训练的编排定为 **torchrun + PyTorch FSDP（full-shard + 梯度检查点）** 的**同卡交替**形态——N 个同构 rank，各自在同一进程内先 `eval()`/`no_grad` 采样打分、再 `train()` 更新，无独立 rollout worker、无 reward actor、无参数服务器。Ray 记为升级项（若未来加独立 rollout 引擎或 KL/参考模型再启用，且须先过 DCU 实测），DeepSpeed ZeRO-3 记显存 fallback。完整设计见 `docs/spec/orchestration.md`。

**Status**: accepted

## Considered Options

- **Ray 角色分离（verl/OpenRLHF 范式）**：弃。该范式的价值（vLLM 独立 decode 引擎 + sleep-mode 权重共享 + reference model）在本 workload 全部缺失——无 vLLM（rollout 就是 UNet 前向）、无参考模型（KL=0）、无文本/tokenizer；Ray 的独有能力（actor 级容错、placement groups、异步角色池）用不上，而 Ray-on-DCU 需付设备可见性地雷（ROCR/HIP/CUDA_VISIBLE_DEVICES）+ 实验性状态；且 verl 的 AMD 支持只验 MI300/MI250、未验 DCU。
- **torchrun + FSDP 同卡交替（选定）**：参考实现 Granular-GRPO 同款（torchrun 2×8 + `torch.distributed.fsdp` full-shard + 梯度检查点），单角色天然拟合；唯一需要的编排（多节点 rank launch + distributed init）torchrun 全给。
- **DeepSpeed ZeRO-3**：不作默认。~1e8 参数量用不上分区；且 OpenRLHF 的 ZeRO-3 只验 MI300/MI250、未验 DCU gfx936。仅作显存 fallback。
- **accelerate + FSDP**：薄封装，无额外收益，多一个 DCU 上要钉的依赖。

## Consequences

- 无独立 rollout worker，权重同步 = FSDP allreduce + `dist.barrier()`（ticket #7 问题③蒸发）。
- 判别器数据并行（每 rank 一份，本地 fake + 共享 real 池切片，per-rank FIFO 回放）。
- 单节点优先（8×64GB=512GB 对 ~1e8 参数量富余）；多节点 = rollout 吞吐升级杠杆。
- 风险集中在 DCU 配置（DTK pytorch-dcu wheel + `numpy==1.26.4` + RCCL 调优），与 torchrun/Ray 选择无关，两路都要验。
- Ray 留档为升级项：加独立 rollout 引擎 / KL / 参考模型时再启用（须先过 DCU 实测）。
