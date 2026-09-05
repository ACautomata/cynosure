"""像素域定量评测的距离度量（ADR-0004：2.5D FID + KID 重采样 CI）。

全部以 torch 实现（float64 数值口径）：MONAI 自带的 ``FIDMetric`` 依赖
scipy 做矩阵平方根——scipy 在 import 白名单之外、且集群侧有 numpy 钉版
回验约束，故 Frechet 距离的矩阵平方根走 ``torch.linalg.eigh``（对称 PSD
谱分解），KID（无偏 MMD²，多项式核）MONAI 无对应实现，同为 torch 原生。

特征空间由 ``SliceFeatureExtractor`` 策略注入（RadImageNet-ResNet50 /
fixture stub），本模块只消费特征矩阵、不感知特征来源。
"""

import torch


class FrechetDistance:
    """两组特征高斯拟合间的 Frechet 距离（FID 的距离核）。

    ``FID = ‖μa − μb‖² + Tr(Σa) + Tr(Σb) − 2·Tr(√(Σa·Σb))``；
    交叉项经对称化 ``√(Σa)·Σb·√(Σa)`` 的谱分解取特征值平方根和——
    Σa·Σb 非对称，其平方根迹与对称化形一致（相似矩阵特征值不变）。
    """

    def score(self, features_a: torch.Tensor, features_b: torch.Tensor) -> float:
        """两组特征 [N, F] / [M, F] 之间的 Frechet 距离（双侧各需 ≥2 样本
        才有非退化协方差）。"""
        if features_a.shape[1] != features_b.shape[1]:
            raise ValueError(
                f"特征维不一致：{features_a.shape[1]} vs {features_b.shape[1]}"
                "（FID 两侧必须在同一特征空间）"
            )
        for name, features in (("a", features_a), ("b", features_b)):
            if features.shape[0] < 2:
                raise ValueError(
                    f"Frechet 距离需要每侧 ≥2 个样本（得到 {name} 侧 "
                    f"{features.shape[0]}），协方差退化"
                )
        a = features_a.double()
        b = features_b.double()
        mean_a, cov_a = self._mean_covariance(a)
        mean_b, cov_b = self._mean_covariance(b)
        diff = mean_a - mean_b
        cross_trace = self._trace_of_sqrt_product(cov_a, cov_b)
        value = (
            diff.dot(diff)
            + torch.trace(cov_a)
            + torch.trace(cov_b)
            - 2.0 * cross_trace
        )
        return float(value)

    @staticmethod
    def _mean_covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = features - features.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / (features.shape[0] - 1)
        covariance = 0.5 * (covariance + covariance.T)  # 对称化消浮点漂移
        return features.mean(dim=0), covariance

    @staticmethod
    def _trace_of_sqrt_product(
        cov_a: torch.Tensor, cov_b: torch.Tensor,
    ) -> torch.Tensor:
        """Tr(√(Σa·Σb))：对称 PSD 谱分解，负特征值（浮点噪声）钳零。"""
        eigenvalues_a, eigenvectors_a = torch.linalg.eigh(cov_a)
        sqrt_a = (eigenvectors_a * eigenvalues_a.clamp_min(0.0).sqrt()) @ eigenvectors_a.T
        symmetric = sqrt_a @ cov_b @ sqrt_a
        symmetric = 0.5 * (symmetric + symmetric.T)
        eigenvalues = torch.linalg.eigvalsh(symmetric)
        return eigenvalues.clamp_min(0.0).sqrt().sum()


class KernelMmd:
    """无偏平方 MMD 估计（多项式核 k(x,y) = (x·y/d + 1)³）——KID 的距离核。

    无偏估计去掉对角项（k(x,x) 的正偏置），小样本下不系统性偏正；
    核的尺度不变项 (x·y/d) 使特征维数不改变核量级。
    """

    DEGREE = 3

    def score(self, features_a: torch.Tensor, features_b: torch.Tensor) -> float:
        if features_a.shape[1] != features_b.shape[1]:
            raise ValueError(
                f"特征维不一致：{features_a.shape[1]} vs {features_b.shape[1]}"
                "（MMD 两侧必须在同一特征空间）"
            )
        for name, features in (("a", features_a), ("b", features_b)):
            if features.shape[0] < 2:
                raise ValueError(
                    f"无偏 MMD² 需要每侧 ≥2 个样本（得到 {name} 侧 "
                    f"{features.shape[0]}），对角去除无意义"
                )
        a = features_a.double()
        b = features_b.double()
        within_a = self._within_term(a)
        within_b = self._within_term(b)
        cross = self._kernel(a, b).mean()
        return float(within_a + within_b - 2.0 * cross)

    def _within_term(self, features: torch.Tensor) -> torch.Tensor:
        n = features.shape[0]
        gram = self._kernel(features, features)
        return (gram.sum() - gram.diagonal().sum()) / (n * (n - 1))

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        similarity = (x @ y.T) / x.shape[1]
        return (similarity + 1.0) ** self.DEGREE


class BootstrapKernelMmd:
    """KID 的重采样置信区间（bootstrap 族的无放回 m-out-of-n 口径）。

    每次重复从两侧各**无放回**抽取半数样本（m = n/2）算 MMD²，取
    2.5/97.5 分位为 95% CI，点估计用全量特征集。**不是** Efron 有放回
    bootstrap：重复行抬高组内相似度、使重复分布系统性偏离点估计
    （无偏 MMD² 在 K≈里程碑样本量的小样本下尤其明显）；无放回半样本
    子抽样（subsampling CI）估计的是同一统计量的抽样波动、方差行为
    有明确定义（m-out-of-n bootstrap 文献的标准替代）。随机性经注入的
    ``torch.Generator``——置信区间可复现（fixture 确定性契约）。
    """

    LOWER_QUANTILE = 0.025
    UPPER_QUANTILE = 0.975

    def __init__(self, replicates: int, generator: torch.Generator) -> None:
        if replicates < 1:
            raise ValueError(f"重采样重复数须 ≥1，得到 {replicates}")
        self._replicates = replicates
        self._generator = generator
        self._kernel = KernelMmd()

    def score_and_replicates(
        self, features_a: torch.Tensor, features_b: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """(全量点估计, 重复值向量)：点估计与自助分布原料一次产出。
        单面区间可直接对重复值取分位；多面汇总统计量的 CI 由上层对
        重复值聚合后取分位（见 MilestoneEvaluator._plane_metrics）。"""
        return self._kernel.score(features_a, features_b), self.replicate_scores(
            features_a, features_b,
        )

    def replicate_scores(
        self, features_a: torch.Tensor, features_b: torch.Tensor,
    ) -> torch.Tensor:
        """每次重复从两侧各无放回抽取半数样本算 MMD² 的重复值向量
        [replicates]——多面汇总统计量的自助分布原料（上层自行聚合）。"""
        size = max(2, min(features_a.shape[0], features_b.shape[0]) // 2)
        return torch.tensor([
            self._kernel.score(
                self._subsample(features_a, size), self._subsample(features_b, size),
            )
            for _ in range(self._replicates)
        ])

    def _subsample(self, features: torch.Tensor, size: int) -> torch.Tensor:
        indices = torch.randperm(features.shape[0], generator=self._generator)[:size]
        return features[indices]
