"""torchrun 进程组装配（spec #15 模块划分「环境装配」；ADR-0003 拓扑）。

DistributedContext 是分布式能力的单点装配面：torchrun 注入的 RANK/
WORLD_SIZE/LOCAL_RANK 环境变量 → ``init_process_group``（DCU/RCCL 与
CUDA 同走 nccl 接口；本机 CPU fixture 走 gloo）；无环境变量的单进程
是 world=1 的退化实现（不初始化进程组、集合通信原语恒等退化）——
训练循环由此对单进程/分布式走同一条执行序（spec 执行序不变，
barrier/gather 在 world-1 下恒等）。

seed 派生含 rank 偏移：各 rank 的六条命名 RNG 流独立演化（rollout 的
条件/噪声各 rank 不同 = 分布式 rollout 的数据多样性来源）；rank 0 恒等
偏移（derive_seed(seed) == seed）是「world=1 与单进程逐位一致」的
等价性前提。判别器冷启动初始化不经本派生（跨 rank 同 seed 构建一致
初始权重，见 trainer 装配）。
"""

import os

import torch
import torch.distributed as dist

_RANK_SEED_STRIDE = 1_000_000
"""rank 间 seed 派生的间隔步长（远大于流内偏移 +0..+6，防流间碰撞）。"""


class DistributedContext:
    """torchrun 进程组句柄（rank/world/barrier/gather/seed 派生/销毁）。"""

    def __init__(self, rank: int, world_size: int, distributed: bool) -> None:
        self._rank = rank
        self._world_size = world_size
        self._distributed = distributed

    @staticmethod
    def env_rank() -> int | None:
        """torchrun 注入的 ``RANK`` 环境变量解析（rendezvous 之前的 rank
        判定单点：cli 守卫与产物装配的 rank 0 独写语义消费它；无环境变量
        = None = 单进程）。"""
        raw = os.environ.get("RANK")
        return None if raw is None else int(raw)

    @classmethod
    def bootstrap(cls) -> "DistributedContext":
        """从 torchrun 环境变量装配进程组；无环境变量 = 单进程退化。

        已初始化的进程组（重复 bootstrap）复用现有语义——进程组是进程级
        单例，CLI 层装配一次后注入 trainer（组3 两阶段共享）。
        """
        if cls.env_rank() is None or os.environ.get("WORLD_SIZE") is None:
            return cls(0, 1, False)
        if dist.is_initialized():
            return cls(dist.get_rank(), dist.get_world_size(), True)
        dist.init_process_group(backend=cls._select_backend())
        context = cls(dist.get_rank(), dist.get_world_size(), True)
        # torchrun 只注入 LOCAL_RANK、不替进程选择 CUDA 设备：显式绑定
        # 本 rank 卡，使「未索引 cuda / current device」语义与 object
        # collectives 的暂存设备都落在 LOCAL_RANK 卡上——否则各 rank 默认
        # 都在 GPU 0 上构建网络，与 FSDP/DDP 的 device_id（cuda:LOCAL_RANK）
        # 错位（DDP 拒绝或 NCCL 对象集合走错卡）。
        if torch.cuda.is_available():
            torch.cuda.set_device(context.local_device())
        return context

    @staticmethod
    def _select_backend() -> str:
        """加速器可集合通信的 backend：DCU/CUDA = nccl（RCCL 同接口）、CPU = gloo。"""
        return "nccl" if torch.cuda.is_available() else "gloo"

    @property
    def rank(self) -> int:
        """本进程全局 rank（0 起）。"""
        return self._rank

    @property
    def world_size(self) -> int:
        """参与训练的 rank 总数（torchrun --nproc_per_node 语义）。"""
        return self._world_size

    @property
    def distributed(self) -> bool:
        """是否初始化了真实进程组（False = 单进程退化，集合通信恒等）。"""
        return self._distributed

    def local_device(self) -> torch.device:
        """本 rank 的计算设备（LOCAL_RANK 对应卡；CPU fixture 恒 cpu）。"""
        if torch.cuda.is_available():
            return torch.device("cuda", int(os.environ.get("LOCAL_RANK", self._rank)))
        return torch.device("cpu")

    def derive_seed(self, seed: int) -> int:
        """RNG 主 seed 的 rank 派生（各 rank 独立数据流；rank 0 恒等）。"""
        return seed + self._rank * _RANK_SEED_STRIDE

    def barrier(self) -> None:
        """执行序节奏点（spec 执行序第 3 步）；单进程恒等。"""
        if self._distributed:
            dist.barrier()

    def gather(self, items: list) -> list[list]:
        """逐 rank 收集对象列表到 rank 0（指标归并的通信原语）。

        返回按源 rank 排列的列表（received[src] = rank src 提交的整段
        提交物）——顺序稳定的归并序由此天然成立；单进程返回自身一份。
        非 0 rank 的返回值是占位（归并消费只在 rank 0）。
        """
        if not self._distributed:
            return [items]
        received: list = [None] * self._world_size
        dist.gather_object(
            items, received if self._rank == 0 else None, dst=0,
        )
        return received

    def all_gather(self, items: list) -> list[list]:
        """逐 rank 收集对象列表到**所有** rank（恢复代际对账的「全体同
        见」原语：一致性结论必须各 rank 独立可判定——只有 rank 0 见全貌
        的 gather 会让通过方单方面继续、拒绝方退出，集合操作互等挂死）。

        返回按源 rank 排列的列表（received[src] = rank src 提交的整段
        提交物）；单进程返回自身一份。
        """
        if not self._distributed:
            return [items]
        received: list = [None] * self._world_size
        dist.all_gather_object(received, items)
        return received

    def broadcast_flag(self, value: bool) -> bool:
        """rank 0 的布尔决定广播到所有 rank（早停 verdict 的全局一致性
        消费：训练循环的 break 必须各 rank 一致，分歧会让 barrier 互等
        死锁）。所有 rank 都须调用本方法（集合操作）；rank 0 的 ``value``
        生效，其余 rank 的传入值被覆盖。单进程恒等返回传入值。flag 落
        本 rank 计算设备（NCCL 只支持 CUDA 张量——CPU 张量的广播在
        torchrun 多卡路径上直接失败；gloo/CPU fixture 下即 cpu）。"""
        if not self._distributed:
            return value
        flag = torch.tensor(
            1.0 if value else 0.0, device=self.local_device(),
        )
        dist.broadcast(flag, src=0)
        return bool(flag.item())

    def destroy(self) -> None:
        """进程组销毁（CLI 层 finally 调用；单进程恒等）。"""
        if self._distributed and dist.is_initialized():
            dist.destroy_process_group()
