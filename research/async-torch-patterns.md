# research：asyncio + PyTorch 异步前向/反向的工程模式与陷阱

- **票**：[ACautomata/cynosure#216](https://github.com/ACautomata/cynosure/issues/216)（地图 #214）
- **日期**：2026-09-26
- **目的**：为「rollout 相全并发 + policy 更新相逐 k barrier」的 async 执行模型（单进程多卡、每卡完整副本、静态分卡）收集工程模式与陷阱。
- **置信度标注**：〔事实〕= 有官方文档/源码直接支撑（附链接）；〔推断〕= 由事实应用到本项目结构的推理，未直接验证。

---

## TL;DR

1. CUDA 调用本身是异步入队，真正的宿主阻塞点只有 D2H 同步拷贝（`.item()`/`.cpu()`）、`synchronize()` 与阻塞式 collective——这些绝不能跑在事件循环线程上。**协程留在主线程做调度，每卡一个专用 worker 线程承载全部 torch 调用**是最稳的映射；current device / current stream 是线程局部状态，线程↔卡的静态绑定天然成立。
2. 「k 内并发收集」**可以包含跨卡并发 backward**：autograd 官方支持多线程并发 backward（各得额外并发），梯度写竞争只发生在**多线程向同一批 `.grad` 累积**时（官方原话 "technically not safe... race condition"）。本设计的「每卡完整副本 + 每卡串行消费」恰好使并发 backward 触碰的参数集互斥，竞争面在结构上不存在——**backward 无需跨卡互斥，只需卡内串行（静态分配已保证）**。
3. 业界（verl/OpenRLHF/vLLM/accelerate/DeepSpeed）清一色选择**多进程 per-GPU**，单进程协程直驱多卡 GPU 属少见形态；torch 自己的 nn.DataParallel 是唯一「单进程多线程多卡」的官方先例，且官方文档明确以 GIL 争用为由判其慢于多进程 DDP。可借鉴的是 verl single-controller 的 dispatch/collect 抽象与 vLLM 的「asyncio 只做门面、GPU 在专职执行体」分层，而非其进程拓扑。
4. RNG 确定性在并发下的实质是**消耗序确定性**：generator 单次调用不会因 GIL 损坏，但交错消耗会重排抽取序。做法 = 按（卡 × 流名）的 per-generator 注册表 + generator 只被所属卡线程消费 + 全局流在 barrier 相单点消费。现行 `TrainingRngStreams` 的命名流 + seed 偏移派生结构可直接平移为「卡派生」。

---

## 一、RQ1：协程 ↔ 线程 ↔ CUDA stream 的编排

### 1.1 事实基础

- **CUDA 异步执行**：GPU 操作默认异步，「the operations are *enqueued* to the particular device, but not necessarily executed until later」；宿主侧通常无感，因为「PyTorch automatically performs necessary synchronization when copying data between CPU and GPU」。即 D2H/H2D 拷贝点是隐式同步点（`non_blocking=True` 可绕开）。〔事实，[CUDA note: Asynchronous execution](https://docs.pytorch.org/docs/2.14/notes/cuda.html)〕
- **stream 语义**：stream 是「a linear sequence of execution that belongs to a specific device」；同流操作按序，跨流并发无序，「unless explicit synchronization functions」（`synchronize()`/`wait_stream()`）。默认流下 PyTorch 自动补同步；**非默认流下同步是用户责任**（`record_stream` 防缓存分配器复用导致的 use-after-free）。backward 的每个 op 跑在其 forward 所在的流上；PyTorch ≤1.9 的「backward 后默认流自动安全消费梯度」模式已不安全，消费方须 `wait_stream`。〔事实，[CUDA note: CUDA Streams](https://docs.pytorch.org/docs/2.14/notes/cuda.html)、[streams.py docstrings](https://github.com/pytorch/pytorch/blob/main/torch/cuda/streams.py)〕
- **thread ↔ device ↔ stream 是线程局部**：`torch.cuda.device` / `torch.cuda.stream` 上下文管理器改变的是当前线程的 selected device / current stream（CUDA runtime 语义本就 per-thread）；`StreamContext.__enter__` 显式处理跨设备换流并恢复。〔事实，[torch/cuda/__init__.py docstrings](https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py)；"Streams are per-device" 见 `stream` 上下文管理器 docstring〕
- **GIL 与 torch**：
  - backward 明确释放 GIL：`THPEngine_run_backward` 用 `pybind11::gil_scoped_release no_gil;` 包住 `engine.execute(...)`，引擎 worker 线程（`pt_autograd_<device>`）以 `gil_scoped_acquire` 按需取 GIL 跑 Python hook。〔事实，[python_engine.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_engine.cpp)〕
  - 部分 ATen 手写绑定同样 `gil_scoped_release` 后才执行 ATen 调用。〔事实，[python_torch_functions_manual.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_torch_functions_manual.cpp)——但该文件无解释性注释，泛化为「大多数前向 op 释放 GIL」属高置信推断〕
  - 官方承认单进程多线程多卡受 GIL 制约：DDP 教程对比 nn.DataParallel——「Due to GIL contention across threads, per-iteration replicated model, and additional overhead introduced by scattering inputs and gathering outputs, `DataParallel` is usually slower than `DistributedDataParallel` even on a single machine.」〔事实，[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)〕
- **torch.distributed 线程约束**：「Initialization is not thread-safe. Process group creation should be performed from a single thread」；「If multiple threads within a process issue collectives, explicit synchronization is necessary to ensure consistent ordering」；NCCL async collective「enqueued on a separate CUDA stream」，消费前须 `work.wait()`。〔事实，[torch.distributed docs](https://docs.pytorch.org/docs/2.14/distributed.html)〕

### 1.2 模式清单（适配「每 k 收集-同步」）

| # | 模式 | 说明 | 依据/置信度 |
|---|------|------|------------|
| P1 | **事件循环线程只做调度** | 主线程跑 asyncio loop，协程持有任务 DAG；一切 torch 调用（含「看似无害」的 `.item()`、reward 的 D2H、`dist.barrier()` 阻塞式）经 `asyncio.to_thread` / per-device executor 下放 | 〔事实×〔CUDA note〕+〔推断〕编排〕 |
| P2 | **每卡一个专用 worker 线程，静态绑定** | 线程启动时 `torch.cuda.set_device(i)` 一次，之后所有任务闭包在该线程执行 → current device/stream 天然线程局部，无锁 | 〔事实（线程局部语义）+〔推断〕映射〕 |
| P3 | **per-device 串行任务队列** | 静态分卡已保证「每卡同时只有一个 (组, k) 任务」；队列深度 1 即天然背压，卡内序列性免锁成立 | 〔推断（由裁决结构推出）〕 |
| P4 | **stream 按卡分配（每卡一条，或直接用默认流）** | 卡内任务串行 → 无跨流数据竞争面；只有当同卡内要重叠 H2D/评测解码与计算时才引入 side stream + `wait_stream`/`record_stream` 纪律。「per task 一条 stream」在本设计下是过度供给 | 〔事实（stream 语义）+〔推断〕取舍〕 |
| P5 | **CUDA 上下文预热在主线程单点完成** | 首次 CUDA 调用初始化 context 开销大；`torch.cuda` 模块用线程局部 `_tls` 管 `_lazy_init` 重入。并行相启动前逐卡 `synchronize()` 一次预热 | 〔事实（`_lazy_init` TLS，[__init__.py](https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py)）+〔推断〕时点〕 |
| P6 | **asyncio.to_thread 用 per-device single-thread executor** | `loop.run_in_executor(per_card_executor[i], fn)` 比 `asyncio.to_thread`（默认共享 ThreadPoolExecutor）更能钉死 线程↔卡 绑定，避免任务漂移到异卡线程后 device 状态错位 | 〔推断（标准库语义 + 本设计要求）〕 |

---

## 二、RQ2：共享参数上的并发前向与反向

### 2.1 事实基础

- **no_grad 是线程局部的**：docstring 原文「This context manager is thread local; it will not affect computation in other threads.」（`enable_grad` 同）。→ 协程里进的 `no_grad` 不会带到 worker 线程，**每个执行线程自管 grad mode**。〔事实，[grad_mode.py](https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py)〕
- **多线程并发 backward 本身受支持**：autograd note 明确多 CPU 线程各自调 `backward()`/`grad()` 时「you are expecting to see extra concurrency instead of serializing all the backward calls」；每个 backward 是独立 GraphTask。〔事实，[Autograd mechanics: Multithreaded Autograd](https://docs.pytorch.org/docs/2.14/notes/autograd.html)〕
- **`.grad` 累积竞争是真的**：同一批共享叶子上多线程 backward →「multiple threads may access and try to accumulate the same `.grad` attribute during gradient accumulation … This is technically not safe, and it might result in race condition and the result might be invalid to use.」官方缓解 = 用函数式 `torch.autograd.grad()` 绕开 `.grad`。另注意共享中间节点的图销毁竞争（等价于漏 `retain_graph` 的报错）。〔事实，同上〕
- **autograd 引擎的多线程语义**：`set_multithreading_enabled` 控制**单次 backward** 是否在引擎的 device worker 线程上展开（默认 True；False 则「the backward pass runs on the calling thread instead」；线程局部 flag，不影响其他线程）。引擎对内置 C++ 节点（AccumulateGrad 等）自带互斥锁；自定义 Python `autograd.Function` 因 GIL 天然安全；C++ hook 无引擎保证。〔事实，[autograd note](https://docs.pytorch.org/docs/2.14/notes/autograd.html) + [grad_mode.py](https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py)〕
- **backward 的流语义**：backward 各 op 跑在其 forward op 的流上；跨流消费梯度须显式 `wait_stream`。〔事实，[CUDA note](https://docs.pytorch.org/docs/2.14/notes/cuda.html)〕

### 2.2 对「每 k 收集-同步」的判定（barrier 协议形态）

〔以下为推断，但每步挂在上述事实上〕

1. **k 内并发能并发到什么程度**：forward 与 backward 都可以跨卡并发——因为每卡完整副本使各卡 backward 触碰的 `.grad` 是**互斥的内存集**，不落入官方警告的「同一 `.grad` 多线程累积」场景；卡内并发则被静态分卡（每卡同时至多一个任务）排除。**结论：barrier 协议的「k 内并发收集」可以包含各卡并发 forward+backward，backward 不需要跨卡互斥，也不需要分卡副本以外的任何保护；唯一的硬约束是卡内串行（静态分配已保证）**。
2. **更保守的替代**（如果未来打破「每卡一任务」的静态性）：同一卡上多任务并发 backward 时，用 `torch.autograd.grad()` 取函数式梯度、各自累积到任务局部 buffer，再在收集相求和——绕开 `.grad` 竞争面。当前裁决下不需要。
3. **allreduce 相**：每 k 一次跨卡梯度 allreduce（sum）→ 各卡副本持有相同总梯度 → 各卡 `optimizer.step()`，即手工 DDP。collective 从**单一线程按固定顺序发起**（或在每卡线程上各卡发自己的 allreduce——同一 PG 上的顺序一致性由「所有线程都在同一个 k 点进入同一个 collective」保证，但更脆弱）。NCCL async 路径记得 `work.wait()` 后再 step。
4. **rollout 相零梯度耦合**：全并发 no_grad 推理 + 判别器重构任务，无 `.grad` 写面，唯一共享状态是参数只读——多线程并发读安全。注意每线程自管 `no_grad`（P1/P2 下 worker 线程内显式进入）。

---

## 三、RQ3：业界先例

### 3.1 verl（HybridFlow）——single-controller 抽象〔事实，[single_controller](https://verl.readthedocs.io/en/latest/single_controller.html)、[hybrid_flow](https://verl.readthedocs.io/en/latest/hybrid_flow.html)、[agent_loop](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)〕

- 控制流单进程（controller），计算在多进程 worker；`@register(dispatch_mode=...)` 装饰器把「split input → dispatch → collect」封装成一次逻辑调用，PPO 主循环读起来像单进程代码。
- **agent loop 就是 asyncio 协程模型**：「AgentLoopWorker schedules multiple coroutines concurrently」，每 prompt 一个 `async def run` 协程，跑完后 manager gather 回 controller。
- 可借鉴：**dispatch/collect 的声明式抽象**（本设计对应「按卡切分的 DataProto→组任务」与「per-k 收集」）；**协程只承载 rollout 生成、重计算在专职执行体**的分层。不可照搬：其多进程 worker 拓扑（我们是单进程多线程 + 每卡副本，无跨进程权重传输问题）。
- v1 async trainer 的 off-policy 治理（model-version 计数、drop/wait 策略）与本图「逐 k 顺序更新、无 off-policy」正交，仅作背景。〔事实，[v1_async_trainer](https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html)〕

### 3.2 OpenRLHF〔事实，[README](https://github.com/OpenRLHF/OpenRLHF/blob/main/README.md)〕

- Ray + vLLM，actor/reward/critic 按角色分 GPU；async（`--train.async_enable`）= 队列缓冲 + partial rollout（vLLM pause/resume 换权重），「Maximum overlap; Most aggressive off-policy」。可借鉴：其「串行 generate→train cycle」的 colocate 模式正是本图每 k barrier 的粗粒度版；其结论「async 会引入 off-policy，需要 importance-sampling 校正」反证了本图「逐 k 全同步」在算法语义上的保守收益。

### 3.3 vLLM V1〔事实，[arch_overview](https://docs.vllm.ai/en/latest/design/arch_overview.html)〕

- API server 进程（asyncio + 媒体线程池）↔ engine core 进程（调度 busy loop）↔ 每 GPU 一个 worker 进程，ZMQ 互联。**要点：asyncio 只做门面，GPU 执行在专职执行体**——即使 vLLM 也没让 asyncio loop 直接驱动 CUDA；区别只在它用进程、我们用线程（单进程多卡是本图裁决）。

### 3.4 accelerate / torchrun / DeepSpeed〔事实，[accelerate launch](https://huggingface.co/docs/accelerate/basic_tutorials/launch)、[torchrun](https://docs.pytorch.org/docs/2.14/elastic/run.html)〕

- 清一色 **process-per-GPU**：`accelerate launch` 包装 torchrun（「torchrun --nproc_per_node=2」），torchrun「spawns up multiple distributed training processes on each of the training nodes」、「each distributed process will be operating on a single GPU」。对「单进程多卡」无直接可借鉴的编排原语——反衬本图裁决是在业界主流形态之外走线，需要自己把 DDP 语义手工化（收集→allreduce→step）。

### 3.5 nn.DataParallel——torch 自带的唯一单进程多线程多卡先例〔事实，[DataParallel docs](https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html)、[DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)〕

- 单进程、多线程、batch scatter 到多卡、每 forward 复制模块、梯度求和回原模块。官方列举的坑（每迭代复制成本、副本状态更新丢失、hook 触发 ×卡数次、GIL 争用 → 「usually slower than DDP even on a single machine」）。
- 可借鉴的教训：**不要每任务复制模块**（本图每卡常驻完整副本，已规避）；**不要在 forward 内改可变状态**（判别器 warm-up 计数、runcate 之类的状态必须放任务闭包外的主循环）；GIL 争用规模 ≈ 卡数（4 卡级别可控，任务内纯 Python 段要短）。

---

## 四、RQ4：与「同 seed 同结果」契约的交互

### 4.1 事实基础

- `torch.manual_seed()` 一并 seed 所有设备；恒定 seed 下「the same series of random numbers will be generated each time the application is run in the same environment」。〔事实，[randomness note](https://docs.pytorch.org/docs/2.14/notes/randomness.html)〕
- 显式 `torch.Generator` 可按设备创建（`torch.Generator(device='cuda')`），state 是 ByteTensor 可存取；`set_rng_state` 仅 CPU（CUDA 用 `manual_seed`）。〔事实，[Generator docs](https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html)、[random.py](https://github.com/pytorch/pytorch/blob/main/torch/random.py)〕
- `fork_rng`：作用域内克隆 RNG、退出时还原（「Forks the RNG, so that when you return, the RNG is reset to the state that it was previously in」），设备多时慢。DataLoader 的标准配方 = per-worker generator + `worker_init_fn` 以保复现；另有 `thread_safe_generator` 提供 DataLoader 线程局部 generator。〔事实，[random.py](https://github.com/pytorch/pytorch/blob/main/torch/random.py)、[randomness note](https://docs.pytorch.org/docs/2.14/notes/randomness.html)〕
- 确定性算法面：`use_deterministic_algorithms`、`cudnn.benchmark=False`、SDPA backend 钉死（FLASH/EFFICIENT 的 backward 非确定）。〔事实，[randomness note](https://docs.pytorch.org/docs/2.14/notes/randomness.html)〕

### 4.2 应用于本设计〔推断，锚定现行代码〕

- **并发下的 RNG 问题不是原子性而是消耗序**：GIL 使单次 generator 调用不会损坏，但两个执行体交错消费同一条流 = 抽取序重排 = 同 seed 不同结果。解法是把「谁消费哪条流、按什么顺序」变成静态事实：
  1. **per-(卡 × 流名) generator 注册表**：现行 `TrainingRngStreams`（[src/cynosure/train/rng.py](../../src/cynosure/train/rng.py)）的命名流 + seed 偏移派生结构直接平移，派生轴从「rank 派生」改为「卡派生」（单进程内卡索引代替 rank）。
  2. **generator 只被所属卡线程消费**：rollout 主流的消耗序 = 该卡静态任务序列的消耗序，确定性由分配表确定性传递。
  3. **全局流（如现行 `recon`，跨 rank 一致语义）在 barrier 相由单点消费**（事件循环线程或 rank0 线程在 k 收集点单线程抽取后再分发），或按 k 静态枚举预抽取——现行注释「s 抽样的调用结构是分布式集合序列的一部分，必须跨 rank 一致」在单进程下变为「必须在 barrier 相单点/单序消费」。
  4. 任务枚举（哪卡拿哪个条件/噪声）在**派发前**由事件循环确定性完成（分配表即 RNG 消耗计划），协程内不再隐式碰全局 RNG。
  5. `fork_rng` 适合包住评测/里程碑解码这类「不能扰动主流」的旁路消耗（注意多设备时慢的警告）。

---

## 五、陷阱清单

| # | 陷阱 | 定性 |
|---|------|------|
| T1 | **事件循环被 torch 阻塞调用卡死**：`.item()`/`.cpu()`/D2H 拷贝、`torch.cuda.synchronize()`、CPU 后端阻塞 collective、checkpoint 写盘——任一发生在 loop 线程即冻结全部协程并发 | 阻塞点清单〔事实〕；本项目哪些调用踩点〔推断〕 |
| T2 | **`no_grad` 不跨线程**：协程内 `with torch.no_grad()` 后把张量工作下放线程，线程内不在 no_grad 中 → 建图 + 显存暴涨。每个 worker 线程入口自设 grad mode | 〔事实（docstring）〕 |
| T3 | **多线程向同一批 `.grad` 累积 = 静默坏梯度**：官方「technically not safe... invalid to use」；不报错、只产错。本设计靠每卡副本 + 卡内串行规避；一旦打破静态分卡立即暴露 | 〔事实〕 |
| T4 | **side stream 的 use-after-free**：非默认流使用张量而其在别的流上释放 → 缓存分配器复用导致的静默脏数据；须 `record_stream`。只在引入 side stream 后才存在 | 〔事实（CUDA note）〕 |
| T5 | **backward 梯度跨流消费无同步**：backward op 在 forward 的流上执行，另一流消费梯度前必须 `wait_stream`；旧版「默认流自动同步」心智已失效 | 〔事实（CUDA note）〕 |
| T6 | **collective 顺序/线程纪律**：PG 初始化只能单线程；多线程发 collective 须保证一致顺序；NCCL async op 入队于独立流，消费前 `work.wait()` | 〔事实（distributed docs）〕 |
| T7 | **GIL 争用随并发线程数上升**：torch 张量操作释放 GIL（backward 确证、前向部分确证），但任务闭包内的纯 Python 段（reward 计算、事件组装、json）全程持 GIL；卡数少时可控，纯 Python 段要短、或移到 loop 线程/独立池 | 〔事实（源码+DDP 教程）+〔推断〕量级〕 |
| T8 | **共享 generator 交错消耗**：同 seed 不同结果，且不报错。任何「在协程里顺手 randint 一下」都是契约违约 | 〔事实（manual_seed 语义）+〔推断〕并发归因〕 |
| T9 | **取消协程 ≠ 停止 GPU 工作**：asyncio cancel 只打断 Python 侧，已入队 kernel 照跑；需要 generation/epoch 标记丢弃陈年结果，防旧噪声组写回新状态 | 〔推断〕 |
| T10 | **worker 线程异常通道**：专用线程 + 队列模式下，任务异常不会自动传回 await 点，需显式 error channel（`to_thread`/`run_in_executor` 则自动传播） | 〔推断（标准库语义）〕 |
| T11 | **CUDA 上下文懒初始化竞争**：多线程同时首触不同设备触发并行 lazy init；启动时主线程逐卡预热一次 | 〔事实（`_lazy_init` TLS）+〔推断〕时点〕 |
| T12 | **每任务复制模块/每 forward 改状态**（DataParallel 教训）：本图每卡常驻副本已规避复制；判别器 warm-up 计数、棘轮等可变状态必须留在主循环，绝不进 forward | 〔事实（DataParallel 官方坑表）〕 |

---

## 六、推荐编排

### 首选：「asyncio 门面 + 每卡专用线程 + 每卡一条流」

```
主线程    asyncio 事件循环
          ├─ 协程只做: 任务枚举(确定性分配表) / await 下放结果 / 每 k 的 barrier 状态机 / RNG 全局流单点消耗
          └─ 绝不直接调用任何阻塞 torch API (T1)

每卡 i    专用 worker 线程 (single-thread executor / Thread + queue)
          ├─ 启动时: set_device(i); 预热 context; 自设 grad mode (rollout 相 no_grad)
          ├─ 串行消费静态分配给本卡的 (组, k) 任务闭包 —— 卡内序列性免锁
          ├─ 全部张量操作走本卡默认流 (或一条常驻 side stream, 无跨流共享则无 T4/T5)
          └─ 本卡专属 generator 组 (per-(卡,流名) 注册表), 只被本线程消费

每 k barrier
          k 内: 各卡并发 forward(+重构) / 打分 D2H 在卡线程内完成
          收集: loop 等齐各卡 Future → 单线程按固定顺序发起 NCCL allreduce(grads, async_op=True)
          同步: work.wait() → 各卡 optimizer.step() (副本同参数化 → 等价 DDP)
          → k+1
```

- **为什么是它**：协程层给了 rollout 相「每卡一协程」的自然并发表达与 barrier 状态机的可读性；线程层把 GIL/线程局部性/T1 全部结构性规避；流层因静态分卡的卡内串行而可以退化到「每卡一条」，无需 record_stream 纪律。每一层都没有引入新的锁面——唯一的同步点是 per-k barrier 本身。
- **backward 并发判定**：k 内各卡并发 backward 安全（`.grad` 内存集互斥，§2.2）；卡内天然串行；不需要 `set_multithreading_enabled(False)`（引擎 device 线程与我们的线程正交）。

### 备选 A：「无 asyncio，纯每卡线程 + 队列 + 主线程编排」

放弃协程层，主线程直接当 scheduler：往每卡队列塞任务、收 Future、跑 barrier。verl/vLLM 的事实（§3.1/3.3）说明工业界普遍把 GPU 放专职执行体、asyncio 只做门面——本图任务 DAG 是静态的（分配表编译期已知），asyncio 的动态调度优势吃不满。**若 barrier 状态机不需要协程组合，这层可以砍**，换来更少的 await 遗漏类 bug 面（漏 await 即假并发）。〔推断〕

### 备选 B（微优化，缓行）：「per-k 流重叠 allreduce 与 k+1 forward」

NCCL allreduce 入队独立流后，k+1 的 no_grad forward 理论上可与上一步梯度通信重叠。但 backward op 跟随 forward 流（§2.1），梯度消费须 wait_stream，流纪律与正确性风险（T4/T5）同步上升；先按首选落地、基准不足再开。〔推断〕

---

## 参考来源

1. PyTorch CUDA note（异步执行/流/record_stream/backward 流语义）: https://docs.pytorch.org/docs/2.14/notes/cuda.html
2. PyTorch Autograd note（Multithreaded Autograd/`.grad` 竞争/functional 缓解）: https://docs.pytorch.org/docs/2.14/notes/autograd.html
3. `set_multithreading_enabled` / `no_grad` 线程局部 docstring: https://github.com/pytorch/pytorch/blob/main/torch/autograd/grad_mode.py
4. autograd 引擎 GIL 释放（python_engine.cpp）: https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_engine.cpp
5. ATen 手写绑定 GIL 释放: https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/python_torch_functions_manual.cpp
6. DDP tutorial（DataParallel vs DDP，GIL 争用）: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
7. nn.DataParallel 文档（单进程多线程坑表）: https://docs.pytorch.org/docs/2.14/generated/torch.nn.DataParallel.html
8. torch.distributed（线程/异步 collective/流）: https://docs.pytorch.org/docs/2.14/distributed.html
9. torchrun（process-per-GPU 启动模型）: https://docs.pytorch.org/docs/2.14/elastic/run.html
10. Accelerate launch（包装 torchrun）: https://huggingface.co/docs/accelerate/basic_tutorials/launch
11. torch/cuda/__init__.py（current_stream/set_stream/`_lazy_init` TLS）: https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py
12. torch/cuda/streams.py docstrings: https://github.com/pytorch/pytorch/blob/main/torch/cuda/streams.py
13. torch/random.py（fork_rng/thread_safe_generator）: https://github.com/pytorch/pytorch/blob/main/torch/random.py
14. torch.Generator 文档: https://docs.pytorch.org/docs/2.14/generated/torch.Generator.html
15. Reproducibility note（manual_seed 全设备/确定性算法/SDPA）: https://docs.pytorch.org/docs/2.14/notes/randomness.html
16. verl single_controller 设计: https://verl.readthedocs.io/en/latest/single_controller.html
17. verl HybridFlow guide: https://verl.readthedocs.io/en/latest/hybrid_flow.html
18. verl agent loop（asyncio 协程先例）: https://verl.readthedocs.io/en/latest/advance/agent_loop.html
19. verl v1 async trainer: https://verl.readthedocs.io/en/latest/advance/v1_async_trainer.html
20. OpenRLHF README（Ray+vLLM/async 模式）: https://github.com/OpenRLHF/OpenRLHF/blob/main/README.md
21. vLLM V1 架构（asyncio 门面/进程分层）: https://docs.vllm.ai/en/latest/design/arch_overview.html
