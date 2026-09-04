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
torch 原生算子，不引入白名单外的统计库。fake 侧全量参与；real 侧采样
数取 min(fake 批量, held-out 条目数)（两侧数量不必相等，Mann-Whitney
对非对称 n×m 有效）。real 侧按本 iteration 采样的目标序列过滤（iter
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
    ) -> float:
        """Mann-Whitney 口径 AUC：real 高于 fake 的配对占比（并列计 0.5）。

        排序 midrank 秩统计实现（与配对枚举口径严格等价）：U = R_real −
        n(n+1)/2，AUC = U/(n·m)；并列块取平均秩（midrank）恰好等价于
        「并列各计 0.5」。秩平方和 ~1e12 在 float64（2^53）内精确。
        """
        if real_scores.numel() == 0 or fake_scores.numel() == 0:
            raise ValueError("AUC 配对统计需要非空 real/fake 分数")
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
