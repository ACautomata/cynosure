# research：DTK/CUDA 栈单进程多卡的并发与梯度同步机制

- **票**：[ACautomata/cynosure#215](https://github.com/ACautomata/cynosure/issues/215)（地图 #214）
- **日期**：2026-09-26
- **方法与证据等级**：只读 SSH 实证探测（sugon 8×DCU / gauss 4×A6000，全部加 `-o BatchMode=yes -o ConnectTimeout=10`，无任何写操作/安装/部署/训练）+ 一手文档与 PyTorch 源码核对（WebFetch）。置信度标签：**[事实·实证]**（本环境跑出来的）、**[事实·文档]**（官方文档原文）、**[事实·源码]**（源码核对）、**[推断]**（基于以上分析，未独立验证）、**[待验证]**（需后续实验）。
- **检索局限声明**：WebSearch 本周配额已耗尽（重置于 2026-09-27），DuckDuckGo/Bing 网页版被反爬拦截，故 DTK vs CUDA 的**生态级差异文献**（如 kernel 覆盖矩阵、triton/flash-attn DCU 构建现状）未能以外部文献补强；本报告该项以「本环境实证 + sugon-bootstrap skill 记载」为准，并明确标注。

## 探测环境基线 [事实·实证]

| | sugon（DTK / 海光 DCU） | gauss（CUDA / NVIDIA） |
|---|---|---|
| torch | 2.9.0 | 2.9.1+cu128 |
| HIP / CUDA | HIP 6.3.26093（`torch.version.hip`） | CUDA 12.9（驱动 575） |
| 加速卡 | 8× 「BW」cc 9.3，64 GB（海光 DCU） | 4× RTX A6000 cc 8.6，47.4 GB |
| 集合通信 | RCCL 2.22.3（`torch.cuda.nccl.version()`），gloo/MPI 可用 | NCCL 2.27.5，gloo 可用 |
| P2P 能力位 | 8×8 **全互联**（`can_device_access_peer` 全 Y） | 4×4 **全互联**（全 Y） |
| P2P 运行时 | 跨卡 `copy_`（含 0↔7 跨端）实测通过 | 跨卡 `copy_`（含 0↔3 跨端）实测通过 |

另有环境噪音 [事实·实证]：sugon 每次触发 `import torch` 打印 `WARNING: /opt/hyhal/lib/cmake/rocm_smi doesn't exist ... The rocm_smi_lib will be used in default`（driver/DTK 版本错位的告警，本次探测未见实际影响，但属 DTK 环境常态噪音）。

---

## 1. 机制排序（推荐 + 回落）

### 排序总表

| 排序 | 机制 | 结论 | 置信度 |
|---|---|---|---|
| **M1 推荐** | `torch.cuda.nccl` 单进程原生绑定（RCCL 同接口） | 双栈一次跑通：单次 CPU 调用驱动全卡 all_reduce，逐位可复现，comm 缓存后 CPU 侧 0.1–0.3 ms | 事实（实证+源码+文档） |
| M2 回落 | 手工 P2P 归约（跨卡 `copy_` 到根卡 `add_` + 广播） | 双栈实测正确，构造性确定；星型拓扑，大张量慢于 ring | 事实（实证）+ 推断（性能） |
| M3 兜底 | host 内存中转（`.cpu()` 累加后回传） | 双栈实测正确；host 带宽瓶颈；零集合通信依赖，最鲁棒 | 事实（实证）+ 推断（性能） |
| M4 不推荐 | c10d（`init_process_group`/ProcessGroupNCCL）单进程多 rank | **无公开支持路径**：默认进程组进程内单例（二次 init 实录报错），PG 构造绑定单 rank 语义 | 事实（实证+源码分析） |
| 附注 | gloo | 单进程下无必要（同地址空间，不存在跨进程传输问题）；保留给 CPU fixture（仓库现状） | 推断 |

### M1（推荐）：`torch.cuda.nccl` 原生绑定

**核心实证**（两端同日探测，脚本见附录 A.1）[事实·实证]：

- gauss：`nccl.all_reduce([t0..t3])` → 每卡 10.0 = 1+2+3+4，正确。
- sugon：8 卡 → 每卡 36.0 = 1+…+8，正确。
- **无需** `init_process_group`、无需 MASTER_ADDR/PORT、无需 rank 文件：传入「每卡一个张量」的 list，一次调用完成全卡归约。
- 128 MB fp32 ×8 卡，两次独立 run 结果**逐位一致**（gauss True / sugon True）[事实·实证]。
- CPU 侧开销：首次调用含 communicator 初始化（gauss 648 ms / **sugon 4307 ms**），之后每次 0.12 ms / 0.25 ms [事实·实证]。

**机制解剖** [事实·源码]（torch v2.9.0 `torch/csrc/cuda/nccl.cpp`）：

- 静态缓存：`static std::unordered_map<device_list, NcclCommList, ...> _communicators;`——按「设备列表」键缓存，`ncclCommInitAll` 只在首次该组合时调用；`comms=None` 走缓存，用户显式传 comms 绕过缓存。
- 每次调用的多卡发起被 `AutoNcclGroup` 包裹（构造 `ncclGroupStart()`、析构 `ncclGroupEnd()`）——**一次 Python 调用 = 一个原子 NCCL group**，这是「单线程驱动多 rank 不死锁」的结构性保证。
- `comm_destroy` 为 no-op（源码注释：segfault workaround），comm 与进程同寿命——对本设计（固定全卡组、进程常驻）无碍。
- 缓存访问由 mutex 保护（源码注释 "guarded by THC's CudaFreeMutex"）。

**为什么它正中本设计** [推断，基于上述事实]：

1. 裁决的「逐 k 收集 → allreduce → 一次 step」在所有卡上是**同一个世界、同一种归约、固定卡序**——恰好是 M1 缓存键恒定、ring 顺序恒定的理想形态。
2. 没有任何「多 rank 时序对齐」问题：所有 rank 的发起由**同一次 CPU 调用**完成（group 内原子发射），NCCL 文档最担心的「跨 rank 发起顺序不一致 → 死锁」在本拓扑下**结构性不存在**。
3. coroutine 单线程调度天然满足 NCCL 线程安全规则（见 §5-1）。

**API 形态** [事实·源码]（`torch/cuda/nccl.py`，v2.9.0）：

```python
uid = torch.cuda.nccl.unique_id()
comms = torch.cuda.nccl.init_rank(num_ranks, uid, rank)   # 可选手动建 comm
torch.cuda.nccl.all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None)
torch.cuda.nccl.broadcast(inputs, root=0, streams=None, comms=None)
torch.cuda.nccl.all_gather(inputs, outputs, streams=None, comms=None)
torch.cuda.nccl.reduce_scatter(inputs, outputs, op=SUM, streams=None, comms=None)
```

输入校验 [事实·源码/实证]：张量须 CUDA（HIP 同接口）、连续、各卡唯一（重复设备直接拒绝）；`init_rank` 第三参须为 int（传 list 报 `'list' object cannot be interpreted as an integer`，且**sugon 的 torch 构建上调 `init_rank` 报 `PY_SSIZE_T_CLEAN macro must be defined`——DTK torch 2.9.0 包的 C 扩展打包缺陷**）。但 `init_rank` 并非必需：`comms=None` 缓存路径完全覆盖本设计需求（实证即走此路径）。**前提**：设备列表一旦变化就触发新的 comm 初始化（sugon 上一次 ~4.3 s）——归约组必须固定全卡、全程不变 [事实·实证 + 推断]。

**启动预热前提** [推断]：首个 k 同步点会吃掉 sugon ~4.3 s 的 comm 初始化；应在 rollout 正式计时前（进程启动时）对全卡 dummy 张量跑一次 `all_reduce` 预热，把这笔开销移出性能底线（≤10% 墙钟）的测量窗口。

### M2（回落）：手工 P2P 归约

**核心实证**（附录 A.2）[事实·实证]：`grads[i].to(root).add_()` 累加 + 广播回各卡——gauss 4 卡、sugon 8 卡均正确；跨卡 `copy_` 双向通过（gauss 含 0↔3，sugon 含 0↔7）。

- 归约顺序由代码写死 → **构造性确定**（不依赖 NCCL 算法选择）。
- 能力位与运行时双重确认：`can_device_access_peer` 全 Y，且跨卡 copy 运行时成功 [事实·实证]。注意：`copy_` 成功不区分底层走 P2P 直传还是驱动回退 host 中转（torch 语义：不可 P2P 时自动回退），**底层路径未直接观测** [待验证]。XGMI/PCIe 拓扑差异会显著影响此路径速度。
- 性能 [推断]：单根星型 = O(W) 串行段拷贝，根卡带宽是瓶颈；对 GB 级梯度比 NCCL ring 慢（经验上 2–4×），k 间隔小、卡多时会吃墙钟。**定位：M1 出现 RCCL 层问题时的诊断性回落**，不是默认路径。

### M3（兜底）：host 内存中转

**核心实证**（附录 A.2）[事实·实证]：`sum(g.cpu() for g in grads)` 再 `.to(cuda:i)` 回传，双栈正确。零加速卡集合通信依赖（连 P2P 都不需要）。host 带宽与同步拷贝延迟决定其只适合兜底/调试与 CPU fixture。仓库现行 `distributed/process.py` 的 gloo/CPU fixture 模式与之兼容（status quo 保留）[事实·源码]。

### M4（不推荐）：c10d 单进程多 rank

**为什么此路不通**：

- [事实·实证] 同进程第二次 `dist.init_process_group` → `ValueError: trying to initialize the default process group twice!`（gauss 实录）。默认进程组是进程级单例。
- [事实·源码分析] `ProcessGroupNCCL(store, rank, size)` 构造器绑定**单一 rank** 语义；要单进程扮演 W 个 rank 需要 W 个不同前缀的 PG 对象、且每个 PG 的全 rank 实例必须齐备才能完成 uid 交换——c10d 没有公开的「一进程多 rank」配方。NCCL 本体层面单进程多 communicator 是官方支持的一等模式（`ncclCommInitAll` 文档原文："You can also call the ncclCommInitAll operation to create n communicator objects at once within a single process." [事实·文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)；torch 也正是为此保留 M1 绑定），**缺的只是 c10d 的封装**——所以正确答案是绕过 c10d 直接用 M1，而不是在 c10d 上硬凿。

### gloo 的角色重估

gloo 是**跨进程**传输层（TCP/共享内存）。单进程模型下所有张量同处一个地址空间，「CPU 中转」退化为普通的 `.cpu()`/`.to()` 拷贝（M3），不需要 gloo 参与归约。gloo 仅保留现有用途：CPU fixture 单测 [推断，基于 c10d/gloo API 形态与单进程语义；仓库现状为事实·源码]。

---

## 2. torch.Generator 线程安全与确定性

**实证设计**（附录 A.3，gauss 与 sugon **结果完全一致**）[事实·实证]：

1. **per-task generator 并发逐位可重放**：W 线程并发，每任务独立 `torch.Generator().manual_seed(s)`（CPU 侧）+ `torch.Generator(device).manual_seed(s)`（设备侧），与顺序执行的参考序列 `torch.equal` **全部 True**（双栈）。
2. **CPU generator 不能喂设备算子**（双栈同报错）：`torch.randn(device='cuda:0', generator=cpu_gen)` → `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`。per-task 模式必须**设备匹配**地建两个 generator（或任务全部 draw 在 CPU 再搬运）。
3. **共享 generator 并发 draw = 调度依赖、不可重放**（双栈一致）：两线程并发从同一 CPU generator 各取 1000 个 draw，结果 ≠ 顺序 A→B 的期望序列（`False`）。不崩溃、不报错——**静默地按线程调度顺序消耗状态**，这正是重放破坏的来源。

**规则提炼**（供「确定性分配表 + RNG 流重设计」直接引用）[推断，由上述事实支撑]：

- 每卡每任务独占 generator 实例，seed = `base_seed` 经**确定性分配表**派生（如 `base_seed + f(卡号, 任务序)`，派生式本身进 spec）。
- **同一 generator 只允许一个执行流 consume**——coroutine 单线程模型下天然成立（同一卡的协程串行）；若走线程宿主实现，须保证 generator 归属卡上的任务不跨线程共享。
- 逐位重放的条件：同 seed + 同 generator 实例 + 同 draw 调用序列 + 同设备类型。draw 本身（MT19937 / Philox）与调度并发无关，前提是上面两条归属规则成立。
- **全管线逐位**（不止 RNG）：还需确定性算子开关（`torch.use_deterministic_algorithms(True)`，CUDA 侧另需 `CUBLAS_WORKSPACE_CONFIG`）消除 atomics 类算子的非确定性——属 RNG 之外的补充前提 [推断/待验证：该开关与 CUBLAS 环境变量在 DTK/hipBLAS 上的行为未在本轮验证]。

---

## 3. DTK vs CUDA 兼容度核实

维护者判断「DTK ≈ CUDA 完全兼容」——结论：**API/源码级兼容（对本项目用到的一切已实证），二进制级不兼容（打包生态隔离）**。逐项：

| 维度 | 结论 | 等级 |
|---|---|---|
| 二进制分发 | **不兼容**：torch/monai/flash_attn/triton 必须 DCU 构建自 sourcefind 镜像，PyPI CUDA 版「装上去会崩或静默错算」（sugon-bootstrap skill 原文口径） | [事实·skill 记载] |
| `torch.cuda.*` API 面 | **兼容**：`device_count / get_device_properties / can_device_access_peer / set_device / synchronize / Generator(device='cuda') / randn(device) / copy_ / add_ / nccl 模块` 全部在 DTK 上行为与 CUDA 一致（`device.type` 均为 `"cuda"`） | [事实·实证] |
| dist backend | **同接口**：`backend="nccl"` 在 DTK 上自动映射 RCCL（仓库 `distributed/process.py` 现行口径即如此，torchrun+DDP 已在 sugon 生产跑通） | [事实·源码 + 实证] |
| 单进程多卡绑定 | `torch.cuda.nccl` 在 DTK 上可用且 all_reduce 正确；唯 `init_rank` 的 C 扩展在 sugon 构建上有 `PY_SSIZE_T_CLEAN` 打包缺陷（`comms=None` 缓存路径不受影响） | [事实·实证] |
| 集合通信版本 | RCCL 2.22.3 vs NCCL 2.27.5：M1 用到的能力（group 发射、comm 缓存、all_reduce/broadcast）在两者都属基础能力，无版本风险；c10d 侧新特性（如 device_id eager init）的版本差未验证 | [推断] |
| 性能 | 8×128MB all_reduce sugon wall 2.5 ms / gauss 4 卡 21.7 ms——均无病理性征兆（拓扑不同，不直接比大小） | [事实·实证，样本有限] |
| stream 语义 | 本轮未触及自定义 stream 用法（M1 支持 `streams=` 参数）；当前裁决「rollout 全并发 no_grad + 逐 k barrier」不依赖跨 stream 细粒度同步 | [待验证，风险低] |
| 生态缺口 | flash-attn/triton 需 DCU 构建（skill 记载）；rocm_smi 告警噪音（实证）；kernel 覆盖矩阵未获文献佐证（检索受限，见局限声明） | [事实·skill] / [待验证] |

**对「DTK ≈ CUDA 完全兼容」的修正表述** [推断]：在「torch 官方发行路径 + 标准 ATen/NCCL 接口」的边界内兼容度极高（本轮全部探测零分歧）；出了这个边界（二进制包、`init_rank` 这类冷门 C 绑定、生态库构建）就是两个世界。执行模型设计应显式约束在边界内。

---

## 4. 风险 / 前提清单（逐条）

1. **M1 并发纪律**：NCCL 官方规则——「it is not allowed to issue NCCL operations to a single communicator in parallel with multiple threads」「It is safe to operate a communicator from multiple threads as long as users can guarantee only one thread operates the communicator at a time」[事实·文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html)。⇒ **逐 k 同步相必须由单一驱动者串行触发**：coroutine 单线程调度天然满足；若实现走线程宿主，须在同步相汇聚到单点（barrier 后单线程调 `all_reduce`，或每卡线程各自 await 一个统一的 asyncio 任务）。这是本设计的硬约束，写进 spec。
2. **固定归约组**：comm 缓存按设备列表键控，sugon 冷启动 ~4.3 s/次 [事实·实证]。⇒ 全程固定全卡组 + 启动预热一次。任何动态子组（如条件数 < 卡数时分组归约）都会引入新的 comm 初始化与新的确定性顺序问题——设计上应避免 [推断]。
3. **多 communicator 全局序**：NCCL 文档（2.26 前）要求多 communicator 并发使用时保持一致全序，否则可能死锁 [事实·文档，communicators 页，链接见附录]。⇒ 本设计一次 group 发射 + 单点驱动，结构性满足 [推断]。
4. **rollout 相并发模型**：coroutine 全并发 = 单线程 Python 逐卡 enqueue，靠 CUDA/HIP 异步执行达成 GPU 并行 [事实·语义，实证于 kernel 异步性]。风险：若 per-step kernel 太碎，Python enqueue 速率成瓶颈（DataParallel 时代同款问题）[推断]；本模型 3D diffusion 前向 kernel 大，预计不敏感 [推断，待 sugon 基准 run 验证——即地图性能底线票的测量对象]。
5. **RNG 三律**（§2）：per-task 独占 + 设备匹配 + 单流 consume。违者静默错（不崩溃）[事实·实证]。回归测试应包含「并发跑 N 步 vs 顺序重放逐位对比」。
6. **全管线确定性开关**：`use_deterministic_algorithms` + `CUBLAS_WORKSPACE_CONFIG` 在 DTK 上的等价行为未验证 [待验证]；且判别器/隐式 atomics 算子可能被该开关直接 raise——需在 spec 的「新锚重建」票里逐算子过闸 [推断]。
7. **`torch.cuda.nccl` API 定位**：文档定位偏内部（DataParallel 遗产），长期 API 稳定性承诺弱于 c10d [推断，源码无 docstring 声明佐证]。缓解：本项目只用 `all_reduce/broadcast/all_gather` 四个稳定函数，且双栈已实证；若未来移除，M2/M3 是纯张量运算兜底 [推断]。
8. **P2P 实际路径未观测**：能力位 Y + `copy_` 成功，但未区分直传/回退 [待验证]；只影响 M2 的性能评估，不影响 M1。
9. **DTK torch 构建质量**：`init_rank` 的 `PY_SSIZE_T_CLEAN` 缺陷说明 DTK 构建在冷门路径上有打包疏漏 [事实·实证]；`comms=None` 主路径健康，但 spec 应写明「不碰 `init_rank`」。
10. **gauss 先行验证有效性**：双栈在 M1/M2/M3/RNG 三律上**行为完全一致**（本轮全部探测）[事实·实证] ⇒ 「gauss 先行、sugon 验收」的两段式验证策略成立 [推断]。

---

## 5. 对执行模型 spec 的落地建议 [推断，基于以上全部事实]

1. 同步相原语选 `torch.cuda.nccl.all_reduce(grads_per_device)`（M1），归约组 = 全卡固定序，进程启动时预热一次。
2. 同步相由调度器单点触发（coroutine 模型天然满足）；rollout 相零集合通信。
3. RNG 按三律改造：`base_seed` → 确定性分配表 → 每卡每任务 `(cpu_gen, device_gen)` 对。
4. 落地顺序：gauss tracer-bullet（M1 + RNG 三律 + 逐位重放回归）→ sugon 全量验收（RCCL 行为 + 4.3s 预热 + 墙钟底线）。
5. spec 明文禁项：`init_process_group` 二次初始化、动态设备子集归约、跨线程共享 generator、`init_rank` 绑定。

---

## 附录 A：探测实录

### A.1 M1 双栈 all_reduce（2026-09-26）

```
gauss: nccl.all_reduce(4×1024 tensors on cuda:0..3) → [10.0, 10.0, 10.0, 10.0]（init_rank 传 list 报 'list' object cannot be interpreted as an integer；不传 comms 直接成功）
sugon: nccl.all_reduce(8×1024 tensors on cuda:0..7) → [36.0 ×8]（init_rank 报 PY_SSIZE_T_CLEAN macro must be defined for '#' formats；不传 comms 直接成功）
```

### A.2 M2/M3 + 跨卡 copy（摘录）

```
gauss: p2p manual reduce ok: [10.0×4] / cpu staging ok: [10.0×4] / copy 0->1,1->2,2->3,0->3 ok
sugon: p2p manual reduce ok: [36.0×8] / cpu staging ok: [36.0×8] / copy 0->1,3->4,0->7 ok
```

### A.3 RNG 三律（双栈输出逐字一致）

```
per-task gen bitwise under concurrency: cpu True device True
cpu-gen-on-cuda raises: Expected a 'cuda' device type for generator but found 'cpu'
shared-gen concurrent == (A,B) sequential expectation: False
```

### A.4 开销与确定性

```
gauss(4卡, 128MB fp32): run1 cpu_enqueue=648.54ms wall=669.89ms; run2 cpu_enqueue=0.12ms wall=21.74ms; bitwise identical: True
sugon(8卡, 128MB fp32): run1 cpu_enqueue=4306.78ms wall=4309.39ms; run2 cpu_enqueue=0.25ms wall=2.51ms; bitwise identical: True
```

### A.5 c10d 双重初始化（gauss 实录）

```
dist.init_process_group(backend="nccl", rank=0, world_size=1) ×2
→ ValueError: trying to initialize the default process group twice!
```

### 证据链接

- NCCL Communicators（ncclCommInitAll 单进程多 communicator、全序要求、同设备多 rank 禁止）: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
- NCCL Thread Safety（单 communicator 多线程并行禁止、单线程独占安全）: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
- torch/cuda/nccl.py（v2.9.0 绑定与签名）: https://github.com/pytorch/pytorch/blob/v2.9.0/torch/cuda/nccl.py
- torch/csrc/cuda/nccl.cpp（communicator 缓存、AutoNcclGroup、destroy no-op）: https://github.com/pytorch/pytorch/blob/v2.9.0/torch/csrc/cuda/nccl.cpp
- 仓库分布式现状: `src/cynosure/distributed/process.py`（`backend="nccl"` 双栈同接口、gloo/CPU fixture）
- 集群环境口径: 用户级 sugon-bootstrap skill（DTK 二进制不兼容、双 source、hy-smi）
