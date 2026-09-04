"""held-out AUC 信号（reward-model 章「防 reward hacking」监控信号集第 1 条）。

held-out real vs 当前 fake 的判别器 AUC：掉到近 chance（~50%）而
eval-reward 仍在升 = 典型 hacking 签名。held-out real 与训练 real
病例级不相交、永不参与判别器更新（prepare 工件 + kind 守卫保证），
因此 AUC 是 out-of-sample 的监控信号。

口径：判别器在 patch logit 图的原生判定单位上打分——held-out real 与
当前 fake 的全部 patch logit 展平后做 Mann-Whitney U 检验：real 高于
fake 的配对占比（并列计 0.5，秩统计的标准 tie 口径）。torch 原生配对
矩阵实现，不引入白名单外的统计库。fake 侧全量参与；real 侧采样数取
min(fake 批量, held-out 条目数)（两侧数量不必相等，Mann-Whitney 对
非对称 n×m 有效）。real 侧按本 iteration 采样的目标序列过滤（iter
事件的归因轴）：其他序列的判别器分数偏移不得伪装成本序列 realism
变化（per-target-sequence 健康监控；modality 缺省的全池混采仅供
诊断）。打分前向在 no_grad 下进行——AUC 非可微、永不 backward，
判别器参数 requires_grad 时带图前向白保留全部激活图。
"""

import torch

from cynosure.config import Modality
from cynosure.reward.artifacts import LatentManifest
from cynosure.reward.sampler import RealPoolSampler
from cynosure.reward.scorer import LatentScorer

_CHUNK_ELEMENTS: int = 2**24
"""单块配对差异矩阵的元素上限（fp32 ≈ 64 MiB）：生产规模（数百 fake ×
2048 patch/latent）的 n×m 配对若整块分配达 TB 级——分块精确累加，内存
有界、tie 口径不变。"""


class HeldOutAuc:
    """held-out 判别力监控信号（hacking 签名判定的输入）。"""

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

    def compute(
        self, fake_latents: torch.Tensor, modality: Modality | None = None,
    ) -> float:
        """当前 fake 批 vs held-out real 的 patch 级 AUC。

        real 侧按 ``modality`` 过滤（本 iteration 采样的目标序列）后无放
        回采样 min(fake 批量, 该序列 held-out 条目数) 条；缺省 None 为
        全池混采，仅供诊断（生产路径按序列归因）；fake 侧全量参与。
        """
        pool_size = (
            self._manifest.modalities.get(modality, 0)
            if modality is not None else self._real_sampler.size
        )
        count = min(fake_latents.shape[0], pool_size)
        if count < 1:
            raise ValueError(
                "held-out AUC 计算需要非空 fake 批与该序列的 held-out "
                f"real 条目（{modality!r}: {pool_size} 条）"
            )
        with torch.no_grad():
            reals = self._real_sampler.sample(count, modality=modality)
            real_scores = self._scorer.patch_logits(reals).flatten()
            fake_scores = self._scorer.patch_logits(fake_latents).flatten()
        return self.auc_from_scores(real_scores, fake_scores)

    @staticmethod
    def auc_from_scores(
        real_scores: torch.Tensor,
        fake_scores: torch.Tensor,
        chunk_elements: int = _CHUNK_ELEMENTS,
    ) -> float:
        """Mann-Whitney 口径 AUC：real 高于 fake 的配对占比（并列计 0.5）。

        配对统计按 ``chunk_elements`` 上限分块精确累加（胜/平计数为整数，
        分块不引入近似；内存有界——整块分配在生产 patch 规模下达 TB 级）。
        """
        if real_scores.numel() == 0 or fake_scores.numel() == 0:
            raise ValueError("AUC 配对统计需要非空 real/fake 分数")
        if chunk_elements < 1:
            raise ValueError(f"chunk_elements 须 ≥1，得到 {chunk_elements}")
        rows_per_chunk = max(1, chunk_elements // fake_scores.numel())
        wins = 0.0
        pairs = 0
        for start in range(0, real_scores.numel(), rows_per_chunk):
            block = (
                real_scores[start:start + rows_per_chunk, None]
                - fake_scores[None, :]
            )
            wins += float((block > 0).sum()) + 0.5 * float((block == 0).sum())
            pairs += block.numel()
        return wins / pairs
