"""RM readiness gate（ADR-0007）：train 入口的上岗硬检查。

「RL 不得带失明的 reward model 开跑」的结构保证：train 启动期按**当前
run 的数据口径**（held-out real + 本 rank base fake 批）对 warm-start
装载的判别器**重算** held-out AUC——不信任预训练报告的旧值——低于门槛
阈值（``reward.pretrain_gate_auc``）即给出含实测值与阈值的可读报错。
门槛是启动期机制，与「防 reward hacking」的训练期监控（AUC 掉回
chance 带）互不替代；续训跳过本检查（续训状态已含判别器全量状态，
恢复点的判别器已在岗）。

重算口径与预训练 gate 测量同源（同一 ``HeldOutAuc.compute``、全池
混采 ``modality=None``——预训练 fake 批跨条件混合，无单一目标序列可
归因）；同 scorer 快照（warm-start 装载的 checkpoint 权重）+ 同输入
下重算值与预训练任一次测量逐位可比。报告 ``final_heldout_auc`` 是
达标跨界与换批复测的较小者（保守口径），本 rank 重算（对 base 分区
批）不复现它也不必复现——门槛判定只依赖本次重算与阈值。

分布式（ADR-0003）：各 rank 的 base fake 独立演化，重算值本 rank
本地——判定经 all_gather 集合裁决，任一 rank 不达标全体一致拒绝
（分歧退出会让其余 rank 停在集合操作；任一 rank 眼里判别器失明都
不得开跑）。
"""

import torch

from cynosure.config import CynosureConfig
from cynosure.distributed import DistributedContext
from cynosure.reward.auc import HeldOutAuc


class ReadinessGate:
    """RM readiness gate 的判定对象（装载守卫之后、训练循环之前的
    最后一道启动期检查）。"""

    def __init__(
        self,
        config: CynosureConfig,
        auc: HeldOutAuc,
        dist: DistributedContext,
    ) -> None:
        self._config = config
        self._auc = auc
        self._dist = dist

    def check(self, fake_batch: torch.Tensor) -> float:
        """按当前 run 数据口径重算 held-out AUC 并对照门槛阈值。

        ``fake_batch`` = 本 rank 的 base fake 样本集（启动期 buffer
        base 分区产物——冻结初始 policy 的量产，与预训练 fake 同分布
        口径）。达标返回实测值；任一 rank 实测值低于阈值（或重算
        本身失败）时抛含实测值与阈值的 ``ValueError``——沿 preflight
        失败语义由 CLI 层统一报错并回滚 run 目录。
        """
        threshold = self._config.reward.pretrain_gate_auc
        local_error: str | None = None
        measured = 0.0
        try:
            measured = self._auc.compute(fake_batch, modality=None)
            if measured < threshold:
                local_error = (
                    f"RM readiness gate 未通过：重算 held-out AUC "
                    f"{measured:.4f} < 门槛 {threshold:.4f}（判别器对"
                    "当前数据口径的 held-out 判别力未达标，拒绝带失明"
                    "的 reward model 开跑——ADR-0007；预训练产物 "
                    f"{self._config.reward.pretrain_report_json}）"
                )
        except Exception as exc:
            # 捕获面放宽到 Exception（重算的失败面不止 ValueError：
            # held-out manifest 条目缺失 = FileNotFoundError、工件损坏 =
            # 反序列化异常、scorer 前向 shape 错位 = RuntimeError）——
            # 任何本地重算失败都必须成为 all_gather 集体裁决的输入，
            # 捕窄会让失败 rank 先于集合点退出、其余 rank 永等
            #（ResumeStore.restore 的两段集合裁决同款语义）
            local_error = (
                f"RM readiness gate 重算失败（{exc}）——held-out AUC "
                "无法按当前数据口径测得"
            )
        peers = [
            entry[0] for entry in self._dist.all_gather(
                [{"rank": self._dist.rank, "error": local_error}],
            )
        ]
        failed = [peer for peer in peers if peer["error"] is not None]
        if failed:
            peer = failed[0]
            message = peer["error"]
            if len(peers) > 1:
                message += (
                    f"（rank {peer['rank']} 重算未达标；任一 rank 的"
                    " base fake 判别力失明都不得开跑——全体一致拒绝）"
                )
            raise ValueError(message)
        return measured
