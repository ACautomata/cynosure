"""训练循环的命名随机流注册表（续训状态机按名保存/恢复的消费面）。

四条 ``torch.Generator`` 流、两条 seeding 轴：三条**数据侧**流由
rank 派生后的主 seed 各自偏移派生（rollout 相与条件分布共享主流；
real 采样 / held-out AUC 各用独立流——一条流的抽取数变化不扰动其余
流的序列，容量实验等不漂移样本流）；``recon``（同源重构加噪，
ADR-0012 的新流）由 rank 无关的 shared seed 派生——其 s 抽样的调用
结构是分布式集合序列的一部分，必须跨 rank 一致。

续训状态按**流名**保存/恢复（resume 模块经 ``named()`` 枚举），注册
表结构一变即续训状态清单失配、显式拒绝。本类是 RNG 侧**唯一注册
对象**，归 TrainingRuntime 聚合层持有（#218：禁 helper/dict 裸容器；
#230 聚合先行已并入原 generators 裸 dict 载体）——装配缝与消费面
一律经本对象取流。

偏移布局权威（#218 §5 单点化）：本模块注释是流 seed 偏移的唯一权威
登记处——数据侧 +0（rollout）/ +1（real_pool）/ +3（heldout_auc）/
+9（recon，shared 轴或槽轴见下）；+2/+4/+5/+8 为退役流空置偏移、
**不回收再用**（含 v10 退役流的「历史占位不回收」口径）；注册表之外：
+6 = 冷启动判别器初始化的全局 fork seed（跨卡一致，不经派生）、+7 =
预训练 SupportRule 的 bootstrap 流、+19 = 预训练测量模板（ADR-0018 起
宿主 = ``MeasurementTemplate`` 主控显式直锚）。async 执行序（#231
骨架期）的槽轴派生见 ``SlotRngRegistry``（线性步长 = 旧 rank 步长的
沿用，槽 0 恒等）；预训练卡轴 recon 流同公式以卡号派生（#221 决议
11——卡 0 恒等 = 现行 rank0 recon 数值）。

历史（ADR-0012 退役，#173）：``disc_noise``（训练期噪声注入）、
``disc_update``（回放抽样）、``fake_shuffle``（fake 全批置换）、
``base_partition``（base 分区量产）四条流随旧判别器供给机制整体退役
——流名已从注册表移除（续训 payload 契约 v10 起），保留流的 seed 偏移
不变（同 seed 下序列与退役前逐位一致）。
"""

import threading

import torch

SLOT_SEED_STRIDE = 1_000_000
"""async 执行序槽间 seed 派生的间隔步长（#218：线性步长沿用——
``DistributedContext`` rank 步长的同一取值，派生公式本体复用、调用方
dist.rank → 槽号）。槽 0 恒等（数值面与现行单进程派生值逐位相同）；
流轴最大偏移差 19 ≪ 槽间距 10⁶，任何槽 × 流组合不撞位。"""


class TrainingRngStreams:
    """四条命名 ``torch.Generator`` 流的注册表（训练循环的随机性注入口）。"""

    ROLLOUT = "rollout"
    REAL_POOL = "real_pool"
    HELDOUT_AUC = "heldout_auc"
    RECON = "recon"

    def __init__(self, seed: int, shared_seed: int | None = None) -> None:
        """``seed`` = 本 rank 派生后的主 seed（数据侧流的多样性来源）；
        ``shared_seed`` = rank 无关的原 seed，``recon`` 流的派生来源
        （缺省用 ``seed``——单进程两参同值，行为不变）。"""
        self.rollout = torch.Generator().manual_seed(seed)
        self.real_pool = torch.Generator().manual_seed(seed + 1)
        self.heldout_auc = torch.Generator().manual_seed(seed + 3)
        # seed+2（原 disc_update）、+4（fake_shuffle）、+5（base_partition）、
        # +8（disc_noise）随 ADR-0012 退役空出，偏移不回收再用；
        # +6 = 冷启动判别器初始化的全局 fork seed、+7 = 预训练 SupportRule
        # 的 bootstrap 流（均注册表之外）。
        # recon 用 rank 无关的 shared seed：s 抽样决定重构续跑
        # ``continue_to_terminal`` 的调用次数与逐次批量——分布式下它是
        # FSDP 集合序列的一部分（#165 review P1 的同一不变式），逐 rank
        # 相异会让首个判别器更新步集合错位死锁。ε 与 s 同流（先 s 后 ε
        # 的抽取次序契约）；rank 本地的只有真实样本库切片，重构加噪不是。
        self.recon = torch.Generator().manual_seed(
            (shared_seed if shared_seed is not None else seed) + 9,
        )

    def named(self) -> dict[str, torch.Generator]:
        """流名 → Generator 的映射视图（续训状态的保存/恢复枚举面；
        装配期消费不经此视图——直接经属性取流）。"""
        return {
            self.ROLLOUT: self.rollout,
            self.REAL_POOL: self.real_pool,
            self.HELDOUT_AUC: self.heldout_auc,
            self.RECON: self.recon,
        }


class SlotRngRegistry:
    """async 执行序的 per-槽流注册表（#218 全口径的载体）：N = 协程数
    个 ``TrainingRngStreams`` 实例、seed 经槽派生（线性步长，槽 0 恒等）。

    - **RL 侧不传 shared_seed**：recon 流走槽轴（``seed_slot + 9``），
      shared 特例消灭（#217 drift #2 的机制落地；pretrain torchrun
      过渡期继续传 shared_seed，同一类两语境分叉）。
    - **四流全顺序推进 / 在线零复位**：流实例注册期一次性实例化，
      ``get_stream`` 重复取同一实例（无复位入口）；在线主循环按任务
      枚举连续消耗、跨 iteration 不复位（按任务复位 = 同槽跨 iteration
      首任务吃同段初始噪声的 GRPO 组间结构相关，#218 不变式）。
    - **消耗序机器断言 = 出口 owner-thread 断言**（#218 评审修订形态；
      generator 为 pybind C++ 类不可子类化，断言只能挂访问面）：
      ``owner_check=True`` 时 ``get_stream`` 出口记录首次取流线程，
      异线程再取即 fail-fast——共享 generator 并发 draw 是静默不可重放
      失败（#215 三律 3），断言是唯一主动暴露手段。默认关闭（热路径
      零开销），测试档开启。

    本注册对象归 TrainingRuntime 聚合层口径持有（#218 禁裸容器）；
    消费面（RolloutPhase / HeldOutAuc 等）一律经 ``get_stream`` 取流，
    无第二取数通道。
    """

    def __init__(
        self, seed: int, slots: int, *, owner_check: bool = False,
    ) -> None:
        if slots < 1:
            raise ValueError(f"槽数须 ≥ 1，得到 {slots}")
        self._slots = tuple(
            TrainingRngStreams(self.slot_seed(seed, slot))
            for slot in range(slots)
        )
        self._owner_check = owner_check
        self._owners: dict[tuple[int, str], int] | None = (
            {} if owner_check else None
        )

    @property
    def slot_count(self) -> int:
        """注册的槽数（= 协程数；分配表槽轴与静态绑卡的取数面）。"""
        return len(self._slots)

    @staticmethod
    def slot_seed(base: int, slot: int) -> int:
        """逐槽 seed 派生的单点公式：注册表构造面与 v12 分片记录面
        共用（seeds 记录值须与实际派生同源——公式两处手写漂移即
        记录静默失真，#236 review 收拢）。"""
        return base + slot * SLOT_SEED_STRIDE

    def get_stream(self, slot: int, stream: str) -> torch.Generator:
        """槽 × 流名的 generator 取数出口（注册表唯一访问面）。

        ``owner_check`` 开启时本出口记录首次取流的线程 id，异线程再取
        即拒绝（fail-fast）——静态绑卡下每槽的流只应被其绑卡线程触碰
        （主线程装配期取流 + 绑卡线程消费的混用在此显式暴露）。"""
        generator = self._stream(slot, stream)
        if self._owners is not None:
            self._assert_owner(slot, stream)
        return generator

    def stream_state(self, slot: int, stream: str) -> torch.Tensor:
        """槽 × 流的 generator 状态**只读**观测面（锚测试断言流推进/零
        复位的取数口）：读状态不消费流、不触发 owner-thread 断言——
        观测不是消耗，注册表消费面（``get_stream``）与本观测面分离。"""
        return self._stream(slot, stream).get_state()

    def restore_stream(
        self, slot: int, stream: str, state: torch.Tensor,
    ) -> None:
        """槽 × 流的 generator 状态回填（续训恢复入口，v12 分片
        ``generators`` 键的应用面）：恢复是状态写入、不是流消费——
        不触发 owner-thread 断言（与 ``stream_state`` 观测面同口径的
        写对应；#218 恢复序「generators 状态回填、不重派生对账」）。"""
        self._stream(slot, stream).set_state(state)

    def _stream(self, slot: int, stream: str) -> torch.Generator:
        """槽 × 流的 generator 解析（槽界与流名在册校验单点；``get_stream``
        与 ``stream_state`` 的共享前段）。"""
        streams = self._slots[self._resolve_slot(slot)]
        if stream not in streams.named():
            raise ValueError(
                f"流名 {stream!r} 不在注册表（在册：{sorted(streams.named())}）"
            )
        return getattr(streams, stream)

    def _resolve_slot(self, slot: int) -> int:
        if not 0 <= slot < len(self._slots):
            raise ValueError(
                f"槽号 {slot} 越界（注册槽数 {len(self._slots)}）"
            )
        return slot

    def _assert_owner(self, slot: int, stream: str) -> None:
        """出口 owner-thread 断言（#218：结构保证外的观测面）。"""
        assert self._owners is not None
        thread = threading.get_ident()
        key = (slot, stream)
        recorded = self._owners.get(key)
        if recorded is None:
            self._owners[key] = thread
        elif recorded != thread:
            raise RuntimeError(
                f"流 {stream!r}（槽 {slot}）被异线程触碰：注册 owner "
                f"thread {recorded:#x}，当前 thread {thread:#x}——同槽流"
                "只允许绑卡线程消费（#215 三律：跨线程共享 generator "
                "并发 draw 静默不可重放）"
            )


class DropoutGuard:
    """装配期 dropout 守卫（#218 §3）：policy 与判别器构建后断言模块树
    内全部 dropout 概率为 0——dropout 的随机消耗绕开 per-槽流注册表
    （模块内部直抽全局流），是「流派生可重放」锚的结构破坏者；违者
    装配期 fail-fast，机器锚长期成立的前提。"""

    @staticmethod
    def assert_clean(module: torch.nn.Module, origin: str) -> None:
        """模块树内凡持概率属性 ``p`` 的子模块断言其为 0（torch 与
        MONAI 的 dropout 层同约定）；``origin`` = 违例文案的装配来源。"""
        for name, child in module.named_modules():
            probability = getattr(child, "p", None)
            if isinstance(probability, float) and probability > 0:
                raise ValueError(
                    f"{origin} 的子模块 {name!r} dropout={probability}："
                    "dropout 绕开注册表直接消耗随机流（#218），装配期"
                    "拒绝——把网络配置的 dropout 置 0"
                )
