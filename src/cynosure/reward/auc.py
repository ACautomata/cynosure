"""held-out AUC 信号（reward-model 章「防 reward hacking」监控信号集第 1 条）。

held-out real vs 当前 fake 的判别器 AUC：掉到近 chance（~50%）而
eval-reward 仍在升 = 典型 hacking 签名。held-out real 与训练 real
病例级不相交、永不参与判别器更新（prepare 工件 + kind 守卫保证），
因此 AUC 是 out-of-sample 的监控信号。

口径：判别器在 patch logit 图的原生判定单位上打分——held-out real 与
当前 fake 的全部 patch logit 展平后做 Mann-Whitney U 检验：real 高于
fake 的配对占比（并列计 0.5，秩统计的标准 tie 口径）。实现为排序
midrank 秩统计（O((n+m)·log(n+m))）：生产 patch 规模（数百 latent ×
2048 patch）的配对枚举达 ~5e11 次比较/iter，监控不得支配训练。
torch 原生算子，不引入白名单外的统计库。``compute()``（在线期 iter 事件
的单标量口径）：fake 侧全量参与，real 侧采样数取 min(fake 批量, held-out
条目数)（两侧数量不必相等，Mann-Whitney 对非对称 n×m 有效），并按本
iteration 采样的目标序列过滤（iter 事件的归因轴）——其他序列的判别器
分数偏移不得伪装成本序列 realism 变化（per-target-sequence 健康监控；
modality 缺省的全池混采仅供诊断）。``compute_volume_clusters()``（预训练
gate 口径，ADR-0008-04）：real 侧由**调用方提供**（全量卷），与 fake 侧
逐样本配对（ADR-0012 同源重构的测量面）。打分前向在 no_grad 下进行
——AUC 非可微、永不 backward，判别器参数 requires_grad 时带图前向白
保留全部激活图。
"""

from dataclasses import dataclass

import torch

from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import LatentScorer


@dataclass(frozen=True)
class VolumeScoreClusters:
    """卷级分数聚类观测面（ADR-0008-02）：每卷一组 patch 分数 + fake 侧
    全量分数。

    支撑度规则（``cynosure.reward.support``）的 bootstrap 原料：重采样
    单元是整卷（``real_volume_scores`` 的一整组进出），patch 级打散把
    同卷内强相关的 patch 当独立观测、低估 CI 宽度——卷才是 i.i.d. 抽
    样单位。构造即校验非空/一维/有限，与 ``HeldOutAuc.auc_from_scores``
    的有限性闸口同口径同语义前置：bootstrap 在重采样簇上重算池化 AUC，
    分数若中途才抛 NaN/Inf 将难以定位。
    """

    real_volume_scores: tuple[torch.Tensor, ...]
    """每卷一组展平 patch 分数（一维、非空、有限；卷间 patch 数可不同）。"""

    fake_scores: torch.Tensor
    """fake 侧全量展平 patch 分数（一维、非空、有限；bootstrap 中固定的对照面）。"""

    def __post_init__(self) -> None:
        if not self.real_volume_scores:
            raise ValueError("卷级聚类需要至少一卷 real 分数")
        for volume in self.real_volume_scores:
            if volume.dim() != 1:
                raise ValueError(
                    f"每卷 patch 分数须为一维展平张量，得到 {tuple(volume.shape)}"
                )
            if volume.numel() < 1:
                raise ValueError("每卷 patch 分数组非空")
        if self.fake_scores.dim() != 1 or self.fake_scores.numel() < 1:
            raise ValueError("fake 侧分数须为一维非空张量")
        for label, scores in (
            ("real", self.real_volume_scores), ("fake", (self.fake_scores,)),
        ):
            bad = [
                index for index, tensor in enumerate(scores)
                if not torch.isfinite(tensor).all()
            ]
            if bad:
                raise ValueError(
                    f"卷级聚类的 {label} 侧分数须为有限值（判别器数值发散或"
                    f"工件损坏）：{label} 侧第 {bad} 组含 NaN/Inf——非有限"
                    "分数在排序口径下会伪装成分数，AUC 因此失去意义"
                )

    @property
    def volume_count(self) -> int:
        """该条件的 held-out 卷数（支撑度判定的卷数输入）。"""
        return len(self.real_volume_scores)

    def pooled_auc(self) -> float:
        """聚类上的池化点估计：每卷自然重数一次，全部 real patch 对
        fake 侧的 Mann-Whitney 口径——与 ``HeldOutAuc.compute`` 的
        estimand 相同（分差只在 real 侧采样面），判据换形态不换被估计量。"""
        return HeldOutAuc.auc_from_scores(
            torch.cat(self.real_volume_scores), self.fake_scores,
        )


class HeldOutAuc:
    """held-out 判别力监控信号（hacking 签名判定的输入）。"""

    SCORE_CHUNK = 8
    """打分前向的定块上界（3D 体数/块）：评估批量 = base 分区
    （capacity//2 体，量产侧限 ``_BASE_BATCH`` 分块生成）或 fake 批，
    一次性全量前向让激活显存随批量无界增长——训练开始前就可能耗尽
    加速器。判别器归一化定死 GroupNorm（前向对 batch 维逐样本独立），
    分块前向与全批逐位等价；AUC 是分数上的 rank 统计，分数级拼接
    不改变口径。"""

    def __init__(
        self,
        heldout_manifest: LatentManifest,
        scorer: LatentScorer,
        generator: torch.Generator,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if heldout_manifest.kind != "heldout_real":
            raise ValueError(
                f"held-out AUC 需 heldout_real 工件，得到 {heldout_manifest.kind}"
                "（train real 冒充 held-out 会失去 out-of-sample 语义）"
            )
        self._manifest = heldout_manifest
        self._scorer = scorer
        self._real_sampler = RealPoolSampler(heldout_manifest, generator, device)

    def _chunked_logits(self, latents: torch.Tensor) -> torch.Tensor:
        """分块打分前向（``SCORE_CHUNK`` 定块，分数级拼接）。"""
        return torch.cat([
            self._scorer.patch_logits(
                latents[start:start + self.SCORE_CHUNK],
            )
            for start in range(0, latents.shape[0], self.SCORE_CHUNK)
        ]).flatten()

    def _pool_size(self, modality: str | None) -> int:
        """该条件的 held-out 条目数（按条件过滤；None = 全池）。"""
        return (
            self._manifest.modalities.get(modality, 0)
            if modality is not None else self._real_sampler.size
        )

    def condition_volume_count(self, modality: str) -> int:
        """该条件的 held-out 卷数（条目数；装配守卫的查询面——预训练
        per-condition 轮转要求条件集每条件 held-out 非空，缺条目的
        条件在装配期显式拒绝而非首步测量时才炸，ADR-0008-04）。"""
        return self._pool_size(modality)

    def condition_latents(self, modality: str) -> torch.Tensor:
        """该条件 held-out **全量卷**的 latent 批（预训练 gate 的 recon
        构造原料，ADR-0012 决策 5）：无放回抽满该条件池——卷数是支撑度
        规则的判定输入，采样子集会让「卷数」一栏与池不一致，故不设下
        采样。同一调用的返回值既是判别器 real 侧、也是重构 fake 侧的
        源（``ReconstructionAssembler.measure_condition`` 与
        ``compute_volume_clusters`` 逐样本配对，AUC 的判别目标只剩重构
        伪影）。

        抽取消耗本对象的持有流（``heldout_auc`` 命名流）——预训练每步
        逐条件调用，抽取次序随轮转序确定；**卷内顺序**不影响读数
        （``auc_from_scores`` 是集合级秩统计）。"""
        pool_size = self._pool_size(modality)
        if pool_size < 1:
            raise ValueError(
                "held-out 全量卷构造需要该条件的条目"
                f"（{modality!r}: {pool_size} 条）"
            )
        return self._real_sampler.sample(pool_size, modality=modality)

    def compute(
        self, fake_latents: torch.Tensor, modality: str | None = None,
    ) -> float:
        """当前 fake 批 vs held-out real 的 patch 级 AUC。

        real 侧按 ``modality`` 过滤（本 iteration 采样的目标序列）后无放
        回采样 min(fake 批量, 该序列 held-out 条目数) 条；缺省 None 为
        全池混采（在线期 iter 事件按序列归因；全池口径仅供诊断——train
        启动期的池化重算已废止，上岗判定读报告条件白名单，ADR-0008
        决策 5；预训练测量走 per-condition 全量卷口径，见
        ``compute_volume_clusters``，ADR-0008-04）。fake 侧全量参与。
        """
        pool_size = self._pool_size(modality)
        count = min(fake_latents.shape[0], pool_size)
        if count < 1:
            raise ValueError(
                "held-out AUC 计算需要非空 fake 批与该序列的 held-out "
                f"real 条目（{modality!r}: {pool_size} 条）"
            )
        with torch.no_grad():
            reals = self._real_sampler.sample(count, modality=modality)
            real_scores = self._chunked_logits(reals)
            fake_scores = self._chunked_logits(fake_latents)
        return self.auc_from_scores(real_scores, fake_scores)

    def compute_volume_clusters(
        self,
        latents: torch.Tensor,
        fake_latents: torch.Tensor,
        modality: str | None = None,
    ) -> VolumeScoreClusters:
        """卷级分数聚类观测面（ADR-0008-02）：每卷一组 patch 分数 +
        fake 侧全量分数。

        ``latents`` = real 侧（**调用方提供**，通常是
        ``condition_latents`` 的返回值）——real 侧不再是本方法内部自抽：
        ADR-0012 的同源重构要求 fake 逐样本由 real 重构而来，两侧必须
        是同一批张量，内部各自抽签会让「同源配对」在测量层悄悄失效
        （AUC 退化为混了内容差的两个分布之间的判别力，恰是本票要根治
        的捷径口径）。逐样本配对由构造保证（``PairBatch``）。

        real 侧**全量卷**整卷暴露（不做 min(fake 批量, 池) 的对称下
        采样）——卷数是支撑度规则（``cynosure.reward.support``）的判定
        输入，下采样会篡改它。卷级分组经「同形 latent → 每卷等 patch
        数」在展平分上 reshape 还原（load_latent 的 latent_shape 守卫
        保证同形）。打分前向同样在 no_grad 下进行、同样 SCORE_CHUNK
        定块；iter 事件的单标量消费路径不经本方法。
        """
        if latents.shape[0] < 1:
            raise ValueError("held-out AUC 卷级聚类需要非空 real 批")
        if fake_latents.shape[0] < 1:
            raise ValueError("held-out AUC 卷级聚类需要非空 fake 批")
        if latents.shape != fake_latents.shape:
            raise ValueError(
                f"real 批 {tuple(latents.shape)} 与 fake 批 "
                f"{tuple(fake_latents.shape)} 不符（同源配对的逐样本对齐"
                "前提——两侧须同量同形）"
            )
        with torch.no_grad():
            real_scores = self._chunked_logits(latents)
            fake_scores = self._chunked_logits(fake_latents)
        patches = real_scores.numel() // latents.shape[0]
        per_volume = real_scores.reshape(latents.shape[0], patches)
        return VolumeScoreClusters(
            real_volume_scores=tuple(per_volume),
            fake_scores=fake_scores,
        )

    @staticmethod
    def auc_from_scores(
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
    ) -> float:
        """Mann-Whitney 口径 AUC：real 高于 fake 的配对占比（并列计 0.5）。

        排序 midrank 秩统计实现（与配对枚举口径严格等价）：U = R_real −
        n(n+1)/2，AUC = U/(n·m)；并列块取平均秩（midrank）恰好等价于
        「并列各计 0.5」。秩平方和 ~1e12 在 float64（2^53）内精确。

        非有限分数显式拒绝（本方法是所有消费点的单一闸口）：排序把
        NaN/Inf 当普通值排（NaN 排尾、+Inf 排头），数值发散或半损坏的
        判别器因此能伪装出高分（实测全 NaN → 1.5、real 单侧 NaN →
        0.875），「失明的判别器」反被认证为高判别力——宁可在测量层
        失败，也不让 gate 拿一个无意义的数做上岗判定。
        """
        if real_scores.numel() == 0 or fake_scores.numel() == 0:
            raise ValueError("AUC 配对统计需要非空 real/fake 分数")
        if not (
            torch.isfinite(real_scores).all() and torch.isfinite(fake_scores).all()
        ):
            nan_real = int(torch.isnan(real_scores).sum())
            nan_fake = int(torch.isnan(fake_scores).sum())
            inf_real = int(torch.isinf(real_scores).sum())
            inf_fake = int(torch.isinf(fake_scores).sum())
            raise ValueError(
                "AUC 配对统计的分数须为有限值（判别器数值发散或工件"
                f"损坏）：real 侧 NaN×{nan_real} Inf×{inf_real}、"
                f"fake 侧 NaN×{nan_fake} Inf×{inf_fake}——非有限分数在"
                "排序口径下会伪装成分数（NaN 排尾、Inf 排头），AUC 因此"
                "失去意义"
            )
        real_count = real_scores.numel()
        combined = torch.cat([real_scores, fake_scores]).double()
        order = combined.argsort()
        sorted_scores = combined[order]
        # 每个排序位置的值在其相等块内的 [first, last] 索引（二分定位，
        # O(n·log n)）：midrank = (first + last)/2 + 1（平均秩，1-based）
        block_start = torch.searchsorted(sorted_scores, sorted_scores, side="left")
        block_end = torch.searchsorted(sorted_scores, sorted_scores, side="right") - 1
        midranks = (block_start + block_end).double() / 2.0 + 1.0
        real_rank_sum = midranks[order < real_count].sum().item()
        u_real = real_rank_sum - real_count * (real_count + 1) / 2.0
        return u_real / (real_count * (combined.numel() - real_count))
