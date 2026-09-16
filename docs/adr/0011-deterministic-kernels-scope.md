# 确定性 kernel 口径的适用范围：pipeline 非确定、测试进程确定

CLI 分发入口原本全局强制确定性 kernel 执行（`use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + `cudnn.benchmark=False`），理由是逐位类断言（同 seed 里程碑 FID、跨 rank 权重对账、续训 roundtrip、轨迹诊断 sha256 数值锚）在 GPU 上依赖 kernel 算法选择确定——缺省的 autotune / split-K 原子归约随负载漂移（同 seed 两 run 的 policy 权重实测 **4e-6 级分叉**、里程碑 FID **逐次漂移 0.04–0.09**）。该口径的代价在 #120 基座自检中实测暴露：DCU（sugon，torch-dcu 2.9.0）上生产尺寸 VAE 解码（latent `[4,64,64,32]` → 像素 `[1,256,256,128]`，滑窗 roi 48³）峰值 **45.6 GiB**，同代码非确定口径 **8.2 GiB**——5.5× 放大，来源是确定性 trilinear 上采样走 torch 分解实现（`torch/_decomp/decompositions.py::_upsample_linear` 的 fp32 中间体），网络结构、权重与输入都不变。64 GB 卡上要整卡独占，而同卡共享场景（`hy-smi` 实测 GPU0 被他人任务占 78%）直接 OOM——解码是评测/rollout 侧的常规动作（#80：decode 占单样本耗时 70–90%），让整条 pipeline 为逐位复现付这份资源账不成比例。**决定：确定性 kernel 口径从运行时收窄为测试进程的属性——生产 pipeline 不开，逐位契约由测试进程（含 spawn 子进程）显式承担；base-smoke 的定点前向按 AC 局部开启。**

**Status**: accepted

## Decision

1. **生产 pipeline 不开确定性模式**：CLI 分发入口不再有任何 kernel 确定性收口（`CynosureCli._enforce_deterministic_kernels` 删除）。train / prepare / eval / pretrain / fid / base-smoke 的实跑路径走 torch 缺省（autotune 允许、原子归约允许、不设 cuBLAS workspace 约束）。资源收益落在全部解码消费点（里程碑评测、Baseline 采样、prepare 生产预编码、#121/#122 的 MR-RATE 数据链）：DCU 解码峰值由 ~46 GiB 回到 ~8 GiB。
2. **逐位复现契约下沉到测试进程**：`tests/conftest.py` 在**导入期**调用 `enforce_deterministic_kernels()`（`CUBLAS_WORKSPACE_CONFIG` 须先于首个 cuBLAS handle，导入期是进程内最早收口），进程内全部 CLI 驱动测试（`CliSession` 与 pytest 同进程）自动获得确定性口径；`tests/test_distributed.py` 的 `TrainWorldWorker`（spawn rank 子进程，独立进程、不继承父进程运行时开关）在 `__call__` 里显式再调用一次——跨 rank 权重逐位对账（RankResumeShards）依赖它。**新增子进程类测试若载荷逐位断言，同样须显式收口。**
3. **base-smoke 按 AC 局部开启**：AC「固定 seed 与确定性 kernels」只覆盖**定点前向**，故确定性作用域只圈前向（`BaseSmokeRunner._deterministic_kernels()` 上下文管理器），退出还原外部状态（pytest 进程已全局开启时还原为开启）；VAE 编解码往返留在 pipeline 口径——解码侧 AC 只要求「跑通 + fp16 autocast 口径」，不要求逐位。
4. **可复现性的承重轴重新声明**：跨 run 可复现由 **RNG 层**承载（内容寻址种子、seeded 后验采样、轨迹种子、续训 RNG 分片——全部不变），kernel 层的逐位浮点一致由测试进程承载。两者不再混为一谈：「同 seed 重跑零漂移」在 pipeline 口径下指语义/统计层面，逐位在测试口径下成立。

## Considered Options

- **维持全局确定性**（现状）：解码峰值 46 GiB、47 GB 显存实例（gauss A6000 类）直接不可用、共享卡必 OOM；为调试便利让常规路径背资源账，否决。
- **仅 decode 局部豁免确定性**（其余路径保持全局开启）：解码峰值同样是 46 GiB 的那一格，真正要解的就是它——豁免 decode 等于本决定的所有收益都拿到，却继续背着「训练/评测全路径确定性」的执行代价（确定性 kernel 普遍更慢、部分算子无确定性实现时直接报错），且 46 GiB 只在确定性模式下存在、与「豁免」自相矛盾，否决。
- **`use_deterministic_algorithms(True, warn_only=True)`**：warn_only 只影响「无确定性实现的算子报错 vs 告警」，不改变算法选择——upsample 仍走分解实现，显存放大照旧，且静默降级会腐蚀逐位断言的判别力，否决。
- **全面取消确定性（测试也不开）**：逐位契约族（跨 rank 权重对账、同进程续训 roundtrip、轨迹 sha256 数值锚）全部退化为容差断言，测试将失去「改动是否引入位级偏移」的判别力，否决。

## Consequences

- 生产 run 之间不再逐位一致：续训恢复后的轨迹、里程碑 FID、prepare 重跑在 kernel 层有浮点噪声（量级 = 上述实测 4e-6 / 0.04–0.09）；**RNG 层契约不变**（seeded 后验采样、轨迹种子、续训 RNG 分片仍逐位可复现）。`reward/pipeline.py` 的「重跑工件零漂移」按此收窄为语义层（统计量归约与 seeded 采样仍确定）。
- 资源：DCU 解码 46 → ~8 GiB（pipeline 口径），同卡共享/小显存实例重新可用；确定性口径下才出现的 `_upsample_linear` 分解实现不再进入生产路径。
- 测试进程：conftest 导入期 + `TrainWorldWorker.__call__` 两处收口；两处之外的子进程（未来新增）需自行收口，已在两处注释与本节声明。
- base-smoke：`BaseSmokeRunner._deterministic_kernels()` 作用域，报告语义不变（`velocity_repeat_identical` 仍是作用域内两次前向的逐位判定）。
- 文档面：CLI 模块 docstring 不再承载该收口的说明；本 ADR 是该口径的唯一权威（原实测证据从 CLI docstring 迁移至此与 conftest 注释）。
- 关联：#120（发现来源与首个消费点）、#121/#122（decode 消费点直接受益）、#111（编排口径中解码成本读数受此影响）。
