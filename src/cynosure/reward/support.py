"""支撑度判定原语 + bootstrap CI 下界（ADR-0008 决策 6，issue #85）。

条件 held-out 卷数 < 支撑度界（暂定 20，config knob
``reward.gate_support_min_volumes``）时，该条件过线判据从池化点估计
改为 **bootstrap CI 下界 ≥ 门槛**；≥ 界维持点估计口径（点估计 ≥
门槛）。两条口径的被估计量相同（该条件的 held-out 池化 AUC，
Mann-Whitney 秩口径）——只差判据形态：低支撑条件的点估计单值波动大，
CI 下界把「运气抽样」拦在门槛外（MRA ≈ 16 卷命中、T2w ≈ 67 卷不命中）。

重采样单元取**卷级聚类**（每卷一组 patch 分数，整卷进出）：patch 级
打散把同卷内强相关的 patch 当独立观测，严重低估 CI 宽度——卷才是
i.i.d. 抽样单位（patch 数是体内容积堆出来的伪重复）。每次重复：有
放回抽 n 卷（n = 该条件卷数）、拼接其 patch 分数、对固定的 fake 侧
全量分数重算池化 AUC（复用 ``HeldOutAuc.auc_from_scores``，单一
MW 实现）；重复数 1000、下界取 2.5% 分位（双侧 95% CI，与 KID 的
``BootstrapKernelMmd`` 分位口径一致），两口径由本模块常量钉死、以
单测固化。随机性经注入的 ``torch.Generator``——bootstrap 可复现
（fixture 确定性契约，同 ``RealPoolSampler`` / ``BootstrapKernelMmd``
的 RNG 注入约定）。

本模块只承载决策 6 的**统计形态**：消费方（预训练 per-condition
gate 的 ADR-0008-04、train 侧 readiness gate 白名单化的 -05）装配
时从 config 读门槛与支撑度界构造 ``SupportRule``；iter 事件的单标量
AUC 消费路径不经本模块（ADR-0008-02：现有观测面不变）。
"""

import torch

from cynosure.reward.auc import HeldOutAuc, VolumeScoreClusters


DEFAULT_REPLICATES: int = 1000
"""bootstrap 重复数：1000 = 分位数估计的标准重复数（再高收益递减）。"""

LOWER_QUANTILE: float = 0.025
"""CI 下界分位：双侧 95% CI 的 2.5% 下尾（与 KID BootstrapKernelMmd 同口径）。"""


def bootstrap_replicates(
    clusters: VolumeScoreClusters,
    *,
    replicates: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """卷级聚类 bootstrap 的重复值向量 [replicates]。

    每次重复：有放回抽 n 卷（n = ``clusters.volume_count``，卷级聚类
    整卷进出——patch 级打散低估 CI 宽度）、拼接其 patch 分数、对固定
    的 fake 侧全量分数重算池化 AUC。重复值向量是自助分布的原料：
    CI 下界取分位（``bootstrap_ci_lower_bound``），分布本身的统计
    形态（支撑、频率）由单测对解析锚固化。
    """
    if replicates < 1:
        raise ValueError(f"bootstrap 重复数须 ≥1，得到 {replicates}")
    volumes = clusters.real_volume_scores
    count = clusters.volume_count
    return torch.tensor([
        HeldOutAuc.auc_from_scores(
            torch.cat([
                volumes[index]
                for index in torch.randint(count, (count,), generator=generator)
                .tolist()
            ]),
            clusters.fake_scores,
        )
        for _ in range(replicates)
    ], dtype=torch.float64)


def bootstrap_ci_lower_bound(
    clusters: VolumeScoreClusters,
    *,
    replicates: int,
    quantile: float,
    generator: torch.Generator,
) -> float:
    """bootstrap 重复分布的 quantile 分位（CI 下界口径）。

    纯函数语义：给定 generator 状态序列可复现。quantile 钉死 2.5%
    （``LOWER_QUANTILE``，双侧 95% CI 下尾）由调用方经 ``SupportRule``
    装配，此处保留参数以锁分位口径的单测可注入。
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"CI 分位须在 (0,1) 内，得到 {quantile}")
    distribution = bootstrap_replicates(
        clusters, replicates=replicates, generator=generator,
    )
    return float(torch.quantile(distribution.double(), quantile))


class SupportRule:
    """支撑度判定原语（ADR-0008 决策 6）：装配期注入门槛/支撑度界/
    随机流，判定本身无共享可变状态。

    - ``decide``：判定原语核心（纯函数）——卷数 < 支撑度界走
      ``ci_lower_bound >= threshold``、≥ 界走 ``point_estimate >=
      threshold``；
    - ``ci_lower_bound``：卷级聚类 bootstrap 的 CI 下界；
    - ``passes``：组合入口——< 界才计算 CI（≥ 界路径省下 1000 次
      重采样），卷数取聚类基数（该条件 held-out 全量卷）。
    """

    def __init__(
        self,
        threshold: float,
        support_bound: int,
        generator: torch.Generator,
        *,
        replicates: int = DEFAULT_REPLICATES,
        quantile: float = LOWER_QUANTILE,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"门槛须在 (0,1) 内（AUC 值域），得到 {threshold}")
        if support_bound < 1:
            raise ValueError(f"支撑度界须 ≥1（卷数下界），得到 {support_bound}")
        if replicates < 1:
            raise ValueError(f"bootstrap 重复数须 ≥1，得到 {replicates}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"CI 分位须在 (0,1) 内，得到 {quantile}")
        self._threshold = threshold
        self._support_bound = support_bound
        self._generator = generator
        self._replicates = replicates
        self._quantile = quantile

    @staticmethod
    def decide(
        point_estimate: float,
        volume_count: int,
        threshold: float,
        support_bound: int,
        ci_lower_bound: float,
    ) -> bool:
        """判定原语（纯函数）：(点估计, 卷数, 门槛, 支撑度界, CI 下界)
        → 过线/不过线。

        卷数 < 支撑度界 → ``ci_lower_bound >= threshold``（bootstrap CI
        下界口径）；≥ 界 → ``point_estimate >= threshold``（点估计口径，
        ``ci_lower_bound`` 不参与）。两分支对另一口径的输入值免疫，
        单测以「本口径不过线的值」钉死分派方向（含卷数恰 = 界的边界）。
        """
        if volume_count < support_bound:
            return ci_lower_bound >= threshold
        return point_estimate >= threshold

    def ci_lower_bound(self, clusters: VolumeScoreClusters) -> float:
        """该条件卷级聚类的 bootstrap CI 下界（重复数/分位 = 构造注入）。"""
        return bootstrap_ci_lower_bound(
            clusters,
            replicates=self._replicates,
            quantile=self._quantile,
            generator=self._generator,
        )

    def passes(
        self, point_estimate: float, clusters: VolumeScoreClusters,
    ) -> bool:
        """组合判定入口：卷数取聚类基数，< 支撑度界先算 CI 下界再委托
        ``decide``（分派规则单点）；≥ 界惰性——CI 不参与判定也不计算
        （省下 1000 次重采样），``decide`` 该分支不消费第 5 参。"""
        volume_count = clusters.volume_count
        ci_lower = (
            self.ci_lower_bound(clusters)
            if volume_count < self._support_bound else 0.0
        )
        return self.decide(
            point_estimate, volume_count,
            self._threshold, self._support_bound, ci_lower,
        )
