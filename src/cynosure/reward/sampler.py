"""Real sample 批采样（判别器「真」侧的数据访问）。

Real sample pool manifest 的消费端：均匀无放回采 K 条 latent 成批，
供判别器 online update 的 real 侧。确定性经 torch.Generator 注入
（fixture 与生产共用同一采样语义；固定 seed + 同一 generator 状态
序列 → 采样序列可复现）。pool / held-out 两侧共用本采样器，
kind 语义由 LatentManifest 装载层守卫。

采样面两步分解（ADR-0016 决策 4 的实现缝）：``sample`` = ``permutation``
（索引抽取——heldout 流一次 randperm 定全量条目排列，不加载 latent
本体）+ ``load``（按条目序列上卡）。两步可独立调用——分布式预训练
测量批「排列全量同序、每 rank 只 load 1/N」即落在这道缝上；单进程
全量路径（``sample`` / ``HeldOutAuc.condition_latents``）行为不变。

ADR-0008-03：online update 的 real 侧按本 iteration 目标模态过滤采样
（``sample`` 的 ``modality`` 缺省 None 保留为诊断/预训练 gate 的全池
口径）；按条件匹配后每条件须独立供满无放回 real 批——装配期逐
(rank 切片或全池, 模态) 容量守卫在候选不足处 fail-fast（
``LatentManifest.assert_condition_capacity``），绝不引入有放回采样
补洞（小池 bagging 是过拟合加速器，ADA, arXiv:2006.06676）。
"""

from typing import Protocol, Sequence

import torch

from cynosure.reward.artifacts import LatentManifest, PoolEntry


class RealSampling(Protocol):
    """「真」侧批采样的策略接口（glossary「Real sample」的采样语义）。

    Online update（train real）与 HeldOutAuc（held-out real）依赖本接口；
    pool / held-out 两侧共用 RealPoolSampler 实现，kind 语义由
    LatentManifest 装载层守卫。
    """

    @property
    def size(self) -> int:
        """条目总数（held-out AUC 对称采样的参照）。"""
        ...

    def sample(
        self, count: int, *, modality: str | None = None,
    ) -> torch.Tensor:
        """无放回均匀采 count 条 latent（``modality`` 给定时候选收窄为
        该条件键的条目——条件键两域同名（#129）：BraTS = 序列名、MR-RATE
        = 生成条件名。MR-RATE 线按条件采样是批内同形的前提：同条件同
        统一网格，异条件 latent 异形状）。"""
        ...


class RealPoolSampler:
    """Real sample pool 的批采样器（无放回均匀采样）。

    manifest latents 持久化在 CPU，采样结果经 ``device`` 迁移到消费方
    （判别器/打分）所在设备——CPU generator 的采样决策与设备解耦。"""

    def __init__(
        self,
        manifest: LatentManifest,
        generator: torch.Generator,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._manifest = manifest
        self._generator = generator
        self._device = device

    @property
    def size(self) -> int:
        """pool 条目总数（held-out AUC 对称采样的参照）。"""
        return len(self._manifest.entries)

    def sample(
        self, count: int, *, modality: str | None = None,
    ) -> torch.Tensor:
        """无放回均匀采 count 条 latent；超出候选条目数显式拒绝。

        分层过滤：``modality`` = 条件键（#129 统一面——BraTS 序列名 /
        MR-RATE 生成条件名；online update 的 real 侧按本 iteration 目标
        条件归因；MR-RATE 线批内同形的前提是异条件 latent 异形状，条件
        匹配键 = 生成条件名）。缺省 None 为全池（诊断与预训练 gate
        口径——预训练 fake 批跨条件混合，无单一目标可归因；MR-RATE
        异形状下仅对同形子集可用）。"""
        candidates = self._candidates(modality)
        if count < 1 or count > len(candidates):
            scope = modality or "全池"
            raise ValueError(
                f"采样数 {count} 超出 pool 条目 {len(candidates)}"
                f"（无放回采样；范围 = {scope}）"
            )
        return self.load(self.permutation(modality=modality)[:count])

    def permutation(
        self, *, modality: str | None = None,
    ) -> tuple[PoolEntry, ...]:
        """候选域的**全量条目排列**（索引抽取面，ADR-0016 决策 4）：一次
        randperm 定序、不加载 latent 本体——只定排列、不做 I/O。排列以
        条目序列具象化（``PoolEntry`` 即 manifest 的持久标识——索引的
        域语义载体）。排列可复算：同 seed 同调用序 → 同排列（消耗本采
        样器的持有流，与 ``sample`` 的 randperm 同源同消耗）；加载交给
        ``load``（排列或其切片上卡）。"""
        candidates = self._candidates(modality)
        indices = torch.randperm(len(candidates), generator=self._generator)
        return tuple(candidates[index] for index in indices.tolist())

    def load(self, entries: Sequence[PoolEntry]) -> torch.Tensor:
        """按条目序列加载 latent 批上卡（``permutation`` 排列切片的
        消费面）：逐条目懒加载 + stack + 设备迁移，零随机性（不消耗
        任何流）——切片加载与整批加载的对应切片逐位一致（顺序不重排），
        分布式下每 rank 加载自己的切片不搅动流位置。"""
        return torch.stack([
            self._manifest.load_latent(entry) for entry in entries
        ]).to(self._device)

    def _candidates(self, modality: str | None) -> list[PoolEntry]:
        """候选域（分层过滤）：``modality`` 给定时收窄为该条件键的条目，
        缺省 None 为全池。"""
        candidates = self._manifest.entries
        if modality is not None:
            candidates = [
                entry for entry in candidates if entry.modality == modality
            ]
        return candidates
