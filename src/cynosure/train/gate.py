"""RM readiness gate（ADR-0008 决策 5）：train 入口的白名单上岗检查。

「RL 不得带失明的 reward model 开跑」的结构保证，按条件化口径落实：
train 启动期读预训练报告的条件白名单（gate 产物，ADR-0008-04 产出），
白名单为空（判别器对全部目标条件都无 held-out 分辨率）即拒绝开跑，
报错含各条件 per-condition 实测值；非空即放行——未过线条件不阻塞
run，由逐 iteration 门控（ADR-0008 决策 7/8 的门控消费票）兜底。

上岗判定直接信任报告值：ADR-0007 的「池化单标量逐 rank 重算」启动期
语义随本票废止——重算的全池混采口径两向皆错（ADR-0008 决策 5），且
启动期测量的噪声让门槛判定在噪声带内摇摆；数据口径漂移由 warm-start
装载的指纹对照把守（``PretrainReport.assert_data_provenance``，装配
期），各 rank 读同一报告天然一致——ADR-0007 时代的 all_gather 集体
裁决随之废止。

放行动作同时是运行时白名单的生效点：判定与 train 循环的逐 iteration
查询消费同一 ``ConditionWhitelist`` 实例（RewardCoordinator 持有），
上岗名单与更新开关永不分叉。

门槛是启动期机制，与「防 reward hacking」的训练期监控（AUC 掉回
chance 带）互不替代；续训跳过本检查（续训状态已含判别器全量状态，
恢复点不重查白名单）。拒绝以 ``ValueError`` 抛出——沿 preflight 失败
语义由 CLI 层统一干净报错并回滚未产出工件的 run 目录。
"""

from cynosure.config import CynosureConfig
from cynosure.train.whitelist import ConditionWhitelist


class ReadinessGate:
    """RM readiness gate 的判定对象（装载守卫之后、训练循环之前的
    最后一道启动期检查）。"""

    def __init__(
        self,
        config: CynosureConfig,
        whitelist: ConditionWhitelist,
    ) -> None:
        self._config = config
        self._whitelist = whitelist

    def check(self) -> None:
        """白名单空即拒绝；非空放行。

        运行时白名单已在 RewardCoordinator 接线（与本判定同一实例），
        放行后循环侧 ``modality in whitelist`` 查询即时生效。拒绝报错
        含各条件 per-condition 实测值（报告 ``condition_auc`` 快照）与
        报告路径——诊断入口。"""
        if len(self._whitelist) > 0:
            return
        readings = "、".join(
            f"held-out AUC[{modality}]: {value:.4f}"
            for modality, value in self._whitelist.measured.items()
        )
        raise ValueError(
            "RM readiness gate 未通过：条件白名单为空（无任何条件过线，"
            "判别器对全部目标条件都无 held-out 分辨率）——拒绝带失明的 "
            f"reward model 开跑（ADR-0008 决策 5；预训练报告 "
            f"{self._config.reward.pretrain_report_json}）。"
            f"各条件 per-condition 实测：{readings or '（报告无实测值）'}"
        )
