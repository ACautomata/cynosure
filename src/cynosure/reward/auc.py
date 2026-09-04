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
非对称 n×m 有效）。
"""

import torch

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
        self._scorer = scorer
        self._real_sampler = RealPoolSampler(heldout_manifest, generator, device)

    def compute(self, fake_latents: torch.Tensor) -> float:
        """当前 fake 批 vs held-out real 的 patch 级 AUC。

        real 侧无放回采样 min(fake 批量, held-out 条目数) 条；
        fake 侧全量参与。
        """
        count = min(fake_latents.shape[0], self._real_sampler.size)
        if count < 1:
            raise ValueError(
                "held-out AUC 计算需要非空 fake 批与 held-out real 工件"
            )
        reals = self._real_sampler.sample(count)
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
