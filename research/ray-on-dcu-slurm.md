# Ray on Hygon DCU + Slurm 编排 RL 后训练（GRPO）基础设施调研

> 日期：2026-09-02
> 目标环境：中科曙光 / SothisAI Hygon DCU 集群（Slurm 调度，DTK 软件栈，ROCm 兼容，PyTorch ROCm 版）
> 目标 workload：GRPO 后训练（多 rollout 采样 + policy 更新 + 在线 reward model 更新）

---

## 1. Ray 对加速器的要求与 ROCm/DCU 支持现状

### 1.1 Ray 的 GPU 资源模型（与厂商无关的部分）

- Ray 本身是 CPU 编排框架。task/actor 使用 GPU 靠 `@ray.remote(num_gpus=N)` 资源声明：Ray 调度到有剩余 GPU 资源的节点、在执行前通过**环境变量注入**的方式把指定设备"划给"该 worker，并预留资源。
- NVIDIA 下注入的是 `CUDA_VISIBLE_DEVICES`；Ray 启动时自动探测节点物理 GPU 数作为资源量（可用 `ray start --num-gpus=N` 覆盖）；actor 内可用 `ray.get_runtime_context().get_accelerator_ids()["GPU"]` 查询分配到的设备（一般不需要，因为环境变量已自动设置）。
- **重要限制**：Ray 只做资源记账，不强制隔离——代码若忽略分配到的设备、直接用全部可见设备，Ray 不会阻止。
- 来源：https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html

### 1.2 Ray 官方对 AMD ROCm 的支持状态

- Ray 官方文档将 AMD GPU 支持标为 **experimental / community-supported**，资源类型名仍是 `GPU`（与 NVIDIA 共用），设备隔离**不用 `CUDA_VISIBLE_DEVICES` 而用 `ROCR_VISIBLE_DEVICES`**（文档示例：`ROCR_VISIBLE_DEVICES=1,3 ray start --head --num-gpus=2`）。官方文档没有提到 `HIP_VISIBLE_DEVICES`。
- 来源：https://docs.ray.io/en/latest/ray-core/scheduling/accelerators.html

### 1.3 版本地雷：ROCR_VISIBLE_DEVICES vs HIP_VISIBLE_DEVICES vs CUDA_VISIBLE_DEVICES

这是 Ray-on-ROCm（含 DCU）最集中的坑，2024–2026 年有大量 issue，核心脉络：

1. **Ray 2.45 起弃用 `ROCR_VISIBLE_DEVICES`**，改用 `HIP_VISIBLE_DEVICES` 管理 AMD 设备可见性。下游 verl 的 worker 代码曾依赖旧变量，已跟进修复（verl issue #1399，修复 PR #1369）。来源：https://github.com/volcengine/verl/issues/1399
2. **只设 `CUDA_VISIBLE_DEVICES` 会触发断言**：`AssertionError: Inconsistent values found. Please use either HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES`（ray#52701）。因为大量"CUDA 迁移"软件在 import 时调用 `torch.cuda.*`，只认 `CUDA_VISIBLE_DEVICES`。
3. **修复 PR #52794（被 #53531 取代并合并）**：允许 `CUDA_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` 同时存在（值相同时不报错），改善 PyTorch-on-ROCm 兼容性。来源：https://github.com/ray-project/ray/pull/52794 、https://github.com/ray-project/ray/issues/52701
4. **`ROCR_VISIBLE_DEVICES` 与 `HIP_VISIBLE_DEVICES` 同时设置时 `import ray` 直接 RuntimeError**：`RuntimeError: Please use HIP_VISIBLE_DEVICES instead of ROCR_VISIBLE_DEVICES`（ray#53737，Ray 2.46.0，Ubuntu 24.04）。而"双变量同时存在"恰是 ROCm/DTK 环境的常见默认。**规避方法：import ray 前 `unset ROCR_VISIBLE_DEVICES`**。来源：https://github.com/ray-project/ray/issues/53737
5. **Ray Train / RLlib / vLLM 组合下的 invalid device ordinal**：`ray.get_gpu_ids()` 与实际可见设备序号不一致（ray#49260，Ray 2.40 + torch 2.5+rocm6.1；ROCm#5780 在 MI300X 上复现）。相关 workaround 环境变量：`RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1`、`RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`（在 PR #52794 讨论中提及）。来源：https://github.com/ray-project/ray/issues/49260 、https://github.com/ROCm/ROCm/issues/5780
6. **import 时初始化 CUDA/HIP 会"毒化"进程**：vLLM 在模块 import 阶段调用 `torch.cuda.get_device_properties()`，早于 Ray worker 设置可见性环境变量，导致所有 worker 都落到 GPU 0（vllm#33938）。规避：在 Ray actor 内部、设置好环境之后再 import vLLM/torch 相关模块。来源：https://github.com/vllm-project/vllm/issues/33938
7. PyTorch 侧配套修复：PyTorch 2.6.0 起尊重 `ROCR_VISIBLE_DEVICES` 做设备发现（pytorch#140318 / PR #144026）。来源：https://github.com/pytorch/pytorch/issues/140318 、https://github.com/pytorch/pytorch/pull/144026
8. 社区实测：裸 actor + `num_gpus=2` + `ROCR_VISIBLE_DEVICES` 在 ROCm 多卡上可用（Ray Discuss，2024-12）；但 `TorchTrainer` 在当时版本有 ordinal 问题。来源：https://discuss.ray.io/t/torchtrainer-fails-rocm-multi-gpu-invalid-device-ordinal/21041

### 1.4 DCU/DTK 特有的坑（搜到的公开资料）

- **没有搜到"Ray + 曙光 DCU"的直接 issue**——DCU 的 HIP 兼容层对 Ray 是透明的，Ray 看到的仍是"ROCm 风格的 AMD GPU"。DCU 层面的坑主要来自 DTK 与上游 ROCm 的衍生差异：
  - **不能用 PyTorch 官方 ROCm wheel，也不能用 pip 官方 torch**：必须用 DTK 定制编译的 pytorch-dcu wheel（曙光云本地目录或光合开发者社区 sourcefind 下载），否则 ABI 不互通、算子库无法加载。来源：https://www.scnet.cn/help/docs/mainsite/ai/practice/application/ultralytics/index.html 、https://github.com/FlyAIBox/dcu-in-action/blob/main/docs/01-dcu-installation.md
  - **DTK 26.04 改了库路径**（`ROCM_PATH=/opt/dtk/.hyhal`），第三方 GPU 探测工具因此失效（gpustack#5334）。Ray 若用 ROCm 路径探测需实测。来源：https://github.com/gpustack/gpustack/issues/5334
  - DTK 环境激活：`module load compiler/dtk/xx.xx` 或 `source /opt/dtk/env.sh`；设备可见性用 `HIP_VISIBLE_DEVICES`；DCU 监控命令是 `hy-smi`（ROCm `rocm-smi` 的对应物）。来源：https://blog.csdn.net/2501_94013662/article/details/162236949 、https://github.com/FlyAIBox/dcu-in-action/blob/main/docs/01-dcu-installation.md
  - 海光 DCU 架构为 gfx 系列（如 gfx906/gfx926/gfx928 等，依卡型），DTK 文档与 wheel 需按 DTK 版本匹配（如 `dtk2604` ↔ torch 2.9）。来源：https://github.com/kvcache-ai/ktransformers/issues/2065
- **本仓库 skill 已记录的集群事实**（`sugon-bootstrap` / `scnet-hpc`）：sourcefind 是文件浏览器不是 pip 索引（须找 wheel 直链 + `--no-deps --no-index` 安装）；DCU torch 2.9 按 numpy 1.x 编译，装任何 ML 依赖后必须回验 `import torch`，必要时钉 `numpy==1.26.4`。

**小结**：Ray-on-DCU 的设备识别链是 `Ray → ROCR/HIP 环境变量 → DTK HIP 运行时 → DCU`，链上每一环都有已知 issue，但都有明确规避手段；风险主要不在"能不能跑"，而在**版本组合与 import 顺序的纪律性**。

---

## 2. Ray on Slurm 的标准部署模式

### 2.1 官方推荐模式（docs.ray.io）

- 官方指南指出 Slurm 的"同程序多副本"模型与 Ray 的 head-worker 单入口模型不匹配。两种模式：
  1. **`ray symmetric-run`（Ray 2.49+，推荐）**：一条 `srun` 在所有节点起 Ray runtime，入口脚本只在 head 节点执行一次。不支持多租户环境。
  2. **手动 head/worker**：head 上 `ray start --head --node-ip-address=... --port=6379 ... --block &`，各 worker 上 `ray start --address=$ip_head ... --block &`，再从 head 用 `ray job submit` 或直接 `srun` 跑入口脚本。
- 官方 sbatch 模板要点：`#SBATCH --nodes=N --ntasks-per-node=1 --cpus-per-task=... --gpus-per-task=...`；head IP 取 `scontrol show hostnames "$SLURM_JOB_NODELIST"` 的第一个；symmetric-run 示例：
  ```bash
  srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
      ray symmetric-run --address "$ip_head" --min-nodes "$SLURM_JOB_NUM_NODES" \
      --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus="${SLURM_GPUS_PER_TASK}" \
      -- python -u simple-trainer.py
  ```
- 注意事项：`python -u` 防止输出缓冲异常；多用户共用集群时必须显式指定互不冲突的端口段（`--port`、`--min-worker-port` 等）；某些集群只允许内网网卡（需 `RAY_NETWORK_INTERFACE`）；IPv6 支持需 Ray 2.49+；Slurm 里跑 Docker 版 Ray 要用 `--init` 防僵尸进程。
- 来源：https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html 、https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-basic.html
- 注意：该指南在官方文档的 **community** 路径下——`ray up` cluster launcher **没有 Slurm provider**（只有云厂商/K8s），Slurm 上不存在 `ray up` 一键方案。来源：https://docs.ray.io/en/latest/cluster/vms/references/ray-cluster-configuration.html

### 2.2 机构级实操参考（LBL Lawrencium）

- 模块加载后用 `srun -n 1 --nodes=1 -w $head_node ray start --head ... --block &` 起 head，再 `srun --exclude=$head_node ray start --address=... --block &` 起 workers；批处理作业内完成，也有 OOD 交互式 App。
- 关键警告："Ray is designed to manage all resources on a node, **use an exclusive partition when possible**"——Ray 按整节点管理资源，与共享分区冲突。后挂 worker 的 walltime 不得超过已有 head session 剩余时间。
- 来源：https://scienceit-docs.lbl.gov/hpc/software/ml/ray/

### 2.3 多节点 RL 训练如何用 Ray 编排（rollout 并行 + 训练）

- 主流 LLM RLHF 框架（verl、OpenRLHF）都把 Ray 用作**集群内编排层**：Ray 负责把 GPU 分配给不同角色（rollout engine / trainer / reward / reference），多节点扩展由 Ray 集群承担，Slurm 只负责拿到 N 个独占节点。
- verl 官方多节点文档列了四种方式：手动 Ray cluster、SkyPilot、Slurm（改 `examples/tutorial/slurm/ray_on_slurm.slurm` 后 `sbatch`）、dstack；多网卡集群需设 `RAY_NETWORK_INTERFACE`。来源：https://verl.readthedocs.io/en/latest/start/multinode.html
- 角色分配模式（OpenRLHF）：`PPORayActorGroup` 管理 Actor/Critic/Reward/Reference 的 DP 分片，`LLMRayActor` 承载 vLLM engine，支持 colocate（同卡混部）+ sleep mode + 异步 rollout。来源：https://github.com/OpenRLHF/OpenRLHF 、https://arxiv.org/html/2405.11143v5

---

## 3. 替代 / 补充方案

### 3.1 现成 RLHF 框架（都基于 Ray，风险与 Ray-on-DCU 同源）

| 框架 | 编排 | 训练后端 | rollout | 备注 |
|---|---|---|---|---|
| verl | Ray single-controller | FSDP/Megatron | vLLM HybridEngine（进程内共享权重） | 官方有 AMD/ROCm 教程（MI300/MI250 一等支持） |
| OpenRLHF | Ray actor pool | DeepSpeed ZeRO-3 | vLLM 独立 Ray actor pool | 支持 colocate、异步、GRPO |

- verl 的 AMD 支持是**针对 MI300/MI250（gfx942/gfx90a）验证的**，要求的组合：ROCm ≥7.0（0.5.x）、vLLM >0.11.0（sleep mode）、vLLM 从源码 `VLLM_TARGET_DEVICE=rocm` 构建。已知坑：多节点 CUDA graph 捕获崩溃（需设 `cudagraph_capture_sizes=[1,2,4,8,16,32,64]` 并 `enforce_eager=False` 谨慎开启）、`HIPBLAS_STATUS_NOT_SUPPORTED`（FP8 GEMM，设 `TORCH_BLAS_PREFER_HIPBLASLT=0`）、NCCL 环境变量组。**这些经验对 DCU 有直接参考价值，但 gfx 架构与 DTK 衍生差异意味着不能假设"MI300 能跑 DCU 就能跑"。**
- 来源：https://verl.readthedocs.io/en/latest/amd_tutorial/amd_vllm_page.html 、https://github.com/volcengine/verl 、https://github.com/OpenRLHF/OpenRLHF

### 3.2 Diffusion-RL / Granular-GRPO 类工作流的常见编排

- **DDPO 原版**（Black et al. 2023, arXiv:2305.13301）：作者 PyTorch 复刻版 `kvablack/ddpo-pytorch` 就是 **accelerate 单机**（`accelerate launch scripts/train.py`，LoRA 后 <10GB 显存）。TRL 的 `DDPOTrainer`、CarperAI 的 DRLX 同为 accelerate 系。来源：https://github.com/kvablack/ddpo-pytorch 、https://huggingface.co/blog/trl-ddpo
- **GRaN-GRPO / visual-gen RL**（如 DGPO，ICLR 2026）：以**单节点脚本**为主（`scripts/single_node/*.sh`），多节点则 slurm + torchrun 自行扩展；NVIDIA cosmos-rl 支持 FlowGRPO/DDRL 等 diffusion RL 算法（torchrun/FSDP 体系，非 Ray）。来源：https://github.com/Luo-Yihong/DGPO 、https://arxiv.org/html/2508.10316v1 、https://nvidia-cosmos.github.io/cosmos-rl/
- **TRL GRPOTrainer + vLLM**：社区在多节点 Slurm 上的用法仍在摸索（open-r1#180 只有问题没有成熟答案），实际多为单节点 DeepSpeed/FSDP。来源：https://github.com/huggingface/open-r1/issues/180
- 结论：diffusion-RL 工作负载（采样是几十步去噪、吞吐远低于 LLM decode）**单机/单节点编排是常态**；多节点需求不强时，accelerate/torchrun 方案的工程风险显著低于 Ray。

### 3.3 降级方案（Ray 在 DCU 上不可行时的 Plan B）

1. **纯 torchrun + 自研队列**（Slurm 标准多卡模式）：sbatch 申请 `--gres=dcu:N` 独占节点，`torchrun --nproc_per_node=N`（或 `srun --mpi=pmix`）起训练；rollout 采样用同一批卡的训练进程内同步完成（diffusion-RL 的 DDPO/DGPO 模式），或通过 `--ntasks` 切分节点角色、以文件/共享存储交换权重与样本。这是曙光 DCU 文档与社区教程的默认做法。来源：https://mcresearch.github.io/abacus-user-guide/abacus-dcu.html 、https://doku.lrz.de/5-3-slurm-batch-jobs-multi-gpu-1898974517.html
2. **训练/推理进程分离 + 权重文件同步**：trainer（torchrun）与 rollout 服务（DCU 版 vLLM/diffusers 推理）各占节点，每 step 通过共享文件系统换权重与 rollout 缓存——不依赖 Ray，但吞吐与显存利用率低。
3. **单节点优先**：若 DCU 单节点卡数（通常 4–8 卡）能满足 GRPO 组采样 + 训练的显存需求，优先单节点方案，彻底回避多节点编排与跨节点 RCCL 问题。

---

## 4. Slurm 侧约束

- **批处理 vs 交互式**：生产作业走 `sbatch`；调试用 `salloc` / `srun --pty bash`（终端断开即失效）；部分集群提供 OOD 交互式 Ray Cluster App。长时间 RL 训练必须 sbatch。来源：https://scienceit-docs.lbl.gov/hpc/software/ml/ray/ 、https://blog.csdn.net/2501_94013662/article/details/162236949
- **srun 与 ray 进程共处**：标准模式是**每个节点恰好一个 Ray runtime**（`--ntasks-per-node=1` / `--tasks-per-node=1`），由 `srun` 以 job step 方式拉起 `ray start ... --block`；用户入口脚本在 head 节点执行（symmetric-run 自动保证）。不要用 srun 的 task 模型把 Ray worker 当 MPI rank 用。
- **多节点 job 内启动 ray cluster 的标准模式**：head 节点取 `SLURM_JOB_NODELIST` 第一个 → `scontrol show hostnames` 解析 IP → head `ray start --head --node-ip-address=$ip --port=6379` → workers `ray start --address=$ip:6379` → `ray job submit --address http://$ip:8265` 或直接 head 上 `python -u`。端口段要按用户/作业隔离。来源：https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html 、https://scienceit-docs.lbl.gov/hpc/software/ml/ray/
- **DCU 资源申请**：曙光/SothisAI 集群用 `--gres=dcu:N`（不是 `gpu:N`），队列名按集群而定；`hy-smi` 查看卡状态。**Ray 侧 `--num-gpus` 的数字要与 srun/sbatch 申请到的卡数对齐**（官方模板用 `${SLURM_GPUS_PER_TASK}` 透传）。来源：https://mcresearch.github.io/abacus-user-guide/abacus-dcu.html 、https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-basic.html
- **独占分配**：Ray 按整节点管理 CPU/GPU 资源，官方与 LBL 文档都建议独占节点（exclusive partition / 整节点申请），否则 Ray 资源视图与 Slurm cgroup 冲突。来源：https://scienceit-docs.lbl.gov/hpc/software/ml/ray/
- **walltime 一致性**：后启动的 worker 作业时长 ≤ 已有 head session 剩余时长；RL 长作业建议单 sbatch 内完成整个 cluster 生命周期。

---

## 5. 结论与风险清单

### 5.1 可行性判断：**可行但有坑**

Ray 在 ROCm 上官方标记为实验性支持但已被 verl/OpenRLHF 等大厂框架生产使用；DCU 的 DTK 是 ROCm 兼容栈，Ray 看到的设备模型与 AMD GPU 一致，没有发现"Ray 在 DCU 上完全不工作"的公开证据。真正的风险集中在三处：(1) **Ray 版本 × ROCm/DTK 环境变量组合的地雷**（§1.3，都有规避手段但必须在目标集群实测）；(2) **vLLM-on-DCU**（rollout 引擎，比 Ray 本身风险更高：DTK 定制 wheel、CUDA graph、sleep mode、NCCL）；(3) **多节点跨机通信**（RCCL/网络）。按"Ray 可用、vLLM 需专项验证"的预期推进是合理的。

### 5.2 推荐编排架构

**单 sbatch 独占多节点 → Ray cluster → 角色化 actor**。作业内：申请 `--gres=dcu:N` 独占节点，source DTK 环境 + unset `ROCR_VISIBLE_DEVICES`；Ray 2.49+ 用官方 `ray symmetric-run` 模板（多租户/旧版本用手动 head/worker + 显式端口段）；集群内按 OpenRLHF/verl 模式分配角色——vLLM rollout engine（colocate + sleep mode）、FSDP policy trainer、reward model actor、reference actor，权重经 NCCL/RCCL 同步。若 Ray 或 vLLM 在 DCU 上验证失败，降级为 **torchrun + accelerate 自研循环**（§3.3）：单节点内完成"多 rollout 采样 → group norm 优势 → policy 更新 → reward 更新"，跨阶段用共享文件系统交换权重；diffusion-RL 类 workload 因单样本采样成本高、节点内即可并行，天然适合这一降级形态。

### 5.3 必须实测验证的事项清单（按优先级）

1. **import 链测试**：DTK 环境下（`source /opt/dtk/env.sh`）`import ray` 是否触发 #53737 的 RuntimeError（检查环境里是否同时存在 `ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES`）；确定 Ray 版本与 #53531 修复的对应关系，选定版本并钉死。来源：https://github.com/ray-project/ray/issues/53737 、https://github.com/ray-project/ray/pull/52794
2. **设备探测**：`ray start --head` 后 `ray status` / 节点资源是否正确显示 DCU 数量（DTK 26.04 的 `ROCM_PATH` 变更是否影响探测）。来源：https://github.com/gpustack/gpustack/issues/5334
3. **单 actor 单卡**：`@ray.remote(num_gpus=1)` 的 actor 内 `torch.cuda.device_count()==1` 且 ordinal 正确（验证 #49260 类 ordinal 错位；必要时试 `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1`）。来源：https://github.com/ray-project/ray/issues/49260
4. **import 顺序**：vLLM/diffusers 必须在 Ray worker 内、设备环境变量设置之后才 import（#33938）；写探针验证。来源：https://github.com/vllm-project/vllm/issues/33938
5. **pytorch-dcu wheel × Ray × numpy 组合**：安装任何依赖后回验 `import torch`；钉 `numpy==1.26.4`；wheel 全部走 sourcefind 直链 `--no-deps --no-index`。（本仓库 sugon-bootstrap skill 经验）
6. **vLLM on DCU**：DTK 定制 vLLM wheel 或源码编译（`VLLM_TARGET_DEVICE=rocm`，路径指向 DTK）；`enforce_eager` 基线、CUDA graph 谨慎开启；sleep mode 可用性。来源：https://verl.readthedocs.io/en/latest/amd_tutorial/amd_vllm_page.html
7. **多节点 RCCL/NCCL**：跨节点 allreduce 与权重同步（设 `NCCL_IB_HCA` 等）；torchrun 与 Ray 两条路径都要测。来源：https://verl.readthedocs.io/en/latest/amd_tutorial/amd_vllm_page.html
8. **Slurm 模板**：`--gres=dcu:N` 与 `--num-gpus` 对齐；`--tasks-per-node=1`；独占队列；端口段隔离；`python -u`；`RAY_NETWORK_INTERFACE`。来源：https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
9. **对称模式边界**：`ray symmetric-run` 不适用多租户——确认目标队列是否独占，否则退回手动模式。
10. **降级路径演练**：同一 workload 用 torchrun + accelerate 跑通单节点版，作为随时可切换的 Plan B。
