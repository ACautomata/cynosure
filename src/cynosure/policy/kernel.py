"""单步 SDE 高斯核（Singular Stochastic Sampling，policy-modeling 章）。

唯一被优化的训练步 k 把确定性 ODE 步替换为带噪高斯核，其余步保持确定性：

    g(s) = η·√(s/(1−s))                              # s 经 s_max 钳制 σ→1 奇异点
    μ    = x·(1 + g²/(2s)·Δs) + v·((1 + g²(1−s)/(2s))·Δs)
    x'   = μ + g·√Δs·ε,   ε ~ N(0,I)                 # Δs = s_k − s_{k+1} > 0

**η=0 时 g=0，上式精确退化为 MONAI ``RFlowScheduler.step()`` 的
``sample + v·Δs``（fp32 逐位一致）——封装正确性的自检锚点**：系数式
``x·(1.0 + 0.0) + v·(1.0·Δs)`` 的每个浮点运算都与 MONAI 同式同序。
log-prob 即该高斯的 log 密度对非 batch 维取均值（沿用 G²RPO 口径；
出自 research/granular-grpo.md §2 Eq.2-3）。
"""

import math
from dataclasses import dataclass

import torch


@dataclass
class SdeTransition:
    """单步高斯核的一次转移：均值 μ、标准差 σ（标量）与采样输出 x'。"""

    mean: torch.Tensor
    std: float
    sample: torch.Tensor


class SdeKernel:
    """单步 SDE 高斯核：η 参数化、s_max 钳制；η=0 精确退化为确定性步。"""

    def __init__(self, eta: float, s_max: float) -> None:
        self._eta = eta
        self._s_max = s_max

    @property
    def eta(self) -> float:
        return self._eta

    @property
    def s_max(self) -> float:
        return self._s_max

    @classmethod
    def deterministic(cls, s_max: float) -> "SdeKernel":
        """η=0 的确定性核：Anchor 轨迹与 ODE 续跑共用（不注入噪声）。"""
        return cls(eta=0.0, s_max=s_max)

    def transition(
        self,
        x: torch.Tensor,
        velocity: torch.Tensor,
        sigma_level: float,
        delta_s: float,
        noise: torch.Tensor | None = None,
    ) -> SdeTransition:
        """一步转移：velocity 只决定该高斯的均值，不是 action 本身
        （policy-modeling 章 MDP：action = x' ~ N(μ, σ²I)）。

        ``noise`` 缺省时输出确定性均值（锚轨迹 / ODE 续跑）；扰动步必须
        显式传入 ε（调用方持 RNG，保证可复现）。
        """
        if sigma_level <= 0.0:
            raise ValueError(
                f"sigma 水平必须为正（s={sigma_level}）；s=0 无去噪步可注入"
            )
        s_eff = min(sigma_level, self._s_max)
        g = self._eta * (s_eff / (1.0 - s_eff)) ** 0.5
        mean = x * (1.0 + g * g / (2.0 * sigma_level) * delta_s) + velocity * (
            (1.0 + g * g * (1.0 - sigma_level) / (2.0 * sigma_level)) * delta_s
        )
        std = g * delta_s ** 0.5
        sample = mean if noise is None else mean + std * noise
        return SdeTransition(mean=mean, std=std, sample=sample)

    def log_prob(
        self, samples: torch.Tensor, transition: SdeTransition,
    ) -> torch.Tensor:
        """log N(x'; μ, σ²I) 对非 batch 维取均值（每 batch 元素一个标量）。"""
        if transition.std <= 0.0:
            raise ValueError(
                "η=0 是确定性步、无高斯密度可求（log-prob 仅在扰动步有意义）",
            )
        per_element = (
            -0.5 / transition.std**2 * (samples - transition.mean) ** 2
            - math.log(transition.std)
            - 0.5 * math.log(2.0 * math.pi)
        )
        return per_element.mean(dim=tuple(range(1, per_element.ndim)))
