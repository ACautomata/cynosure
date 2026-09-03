"""fixture 轨迹诊断回路（``train --dump-trajectory`` 背后，spec「产物工件
契约」的 fixture 诊断模式）。

产出诊断工件（测试面 #1–#3 的信号载体，spec「Testing Decisions」）：

- **per-step 轨迹 latent 统计/哈希**：policy 封装路径（η=0 Anchor 轨迹）与
  MONAI ``RFlowScheduler.step()`` 直接对照路径双列——测试面 #1（η=0 parity）
  的断言面，真值锚 = MONAI 库本身（零依赖，不对照基座代码）；
- **log-prob 对**（扰动时记录值 vs 采样场重算值）——测试面 #3（log-prob
  一致性）的载体；
- **双样本分布统计量**（η=0 对照终点集 vs 扰动+ODE 续跑终点集）——测试面
  #2（噪声注入 sanity）的断言面；η=0 时两列逐位相等，η>0 时分布统计量
  应保持一致（边缘分布保持）。

诊断在 seed 控制下全程确定性：初始噪声、扰动 ε 均出自同一 generator。
"""

import hashlib

import torch
from pydantic import BaseModel, ConfigDict

from cynosure.config import CynosureConfig, MODALITIES
from cynosure.netbuild import NetworkArtifact, NetworkAssembler
from cynosure.policy.condition import ModalityMapping, RolloutCondition
from cynosure.policy.cursor import TrajectoryCursor
from cynosure.policy.field import CfgCombinedField
from cynosure.policy.kernel import SdeKernel
from cynosure.policy.sampler import RolloutSampler

CONDITION_SPACING: float = 100.0
"""诊断条件里的体素间距（1.0 × 1e2，policy-modeling 章：spacing ×1e2 恒传）。"""


class TrajectoryStepStats(BaseModel):
    """per-step 轨迹 latent 的统计/哈希行（诊断工件的双列共享本契约）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    step_index: int
    timestep: float
    mean: float
    std: float
    min: float
    max: float
    sha256: str
    """latent 连续 fp32 字节的 sha256（同内容同指纹，per-step 数值锚）。"""


class LogProbPair(BaseModel):
    """log-prob 对：扰动时记录值 vs 采样场重算值（同权重下应精确相等；
    测试面 #3 的载体——后续训练侧以同结构对断言 π_old 前后一致）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    step_index: int
    noise_index: int
    direction: int
    recorded: float
    recomputed: float


class TerminalSampleStats(BaseModel):
    """单侧终点样本集的分布统计量（双样本分布统计量的单列）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    count: int
    channel_mean: list[float]
    channel_std: list[float]

    @classmethod
    def of(cls, latents: torch.Tensor) -> "TerminalSampleStats":
        """终点 latent 集的 per-channel 统计（count = batch 元素数）。"""
        if latents.ndim < 2:
            raise ValueError(f"终点 latent 须为 [N, C, ...] 张量，得到 {tuple(latents.shape)}")
        return cls(
            count=int(latents.shape[0]),
            channel_mean=latents.mean(dim=(0, 2, 3, 4)).tolist(),
            channel_std=latents.std(dim=(0, 2, 3, 4), correction=0).tolist(),
        )


class TrajectoryDiagnosticReport(BaseModel):
    """轨迹诊断工件（run 目录 ``trajectory.json`` 的契约；字段为最小集，
    施工可扩不可改名）。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    eta: float
    s_max: float
    num_inference_steps: int
    input_img_size_numel: int
    perturbation_steps: list[int]
    schedule_timesteps: list[float]
    """sigma 日程（MONAI ``set_timesteps`` 实际输出，transform 生效、
    实际 scale=1.0）——日程锚定断言面（ADR-0002）。"""
    anchor_trajectory: list[TrajectoryStepStats]
    """policy 封装路径（η=0 Anchor 轨迹）的 per-step 统计/哈希。"""
    monai_reference_trajectory: list[TrajectoryStepStats]
    """MONAI ``step()`` 直接对照路径的 per-step 统计/哈希（真值锚列）。"""
    logprob_pairs: list[LogProbPair]
    control_terminals: TerminalSampleStats
    """η=0 对照终点集（Anchor 全 ODE 终点，原模型分布）。"""
    perturbed_terminals: TerminalSampleStats
    """扰动 + ODE 续跑终点集（被优化步注入噪声后的分布）。"""


class LatentFingerprint:
    """latent 张量的统计/哈希指纹：标量统计 + 连续 fp32 字节 sha256。"""

    def __init__(self, latent: torch.Tensor) -> None:
        values = latent.detach().to(torch.float32)
        self._mean = values.mean().item()
        self._std = values.std(correction=0).item()
        self._min = values.min().item()
        self._max = values.max().item()
        self._digest = hashlib.sha256(
            values.contiguous().numpy().tobytes(),
        ).hexdigest()

    def to_step_stats(self, step_index: int, timestep: float) -> TrajectoryStepStats:
        return TrajectoryStepStats(
            step_index=step_index,
            timestep=timestep,
            mean=self._mean,
            std=self._std,
            min=self._min,
            max=self._max,
            sha256=self._digest,
        )


class TrajectoryDiagnosticRunner:
    """fixture 诊断回路：fixture_mode config + 网络 artifact → 诊断工件。"""

    ANCHOR_NOISES = 4
    """噪声注入 sanity 的初始噪声样本量：四序列各一份（模态条件全覆盖）。"""

    def __init__(self, config: CynosureConfig) -> None:
        unet = NetworkAssembler.unet(NetworkArtifact(
            config=NetworkAssembler.load_json(config.artifacts.net_config_json),
            checkpoint=config.artifacts.unet_ckpt,
        ))
        unet.eval()
        scheduler = NetworkAssembler.rflow_scheduler(
            num_inference_steps=config.policy.num_inference_steps,
            input_img_size_numel=config.policy.input_img_size_numel,
        )
        kernel = SdeKernel(eta=config.policy.sde_eta, s_max=config.policy.sde_s_max)
        self._config = config
        self._scheduler = scheduler
        self._kernel = kernel
        self._cursor = TrajectoryCursor(scheduler)
        self._field = CfgCombinedField(unet)
        self._sampler = RolloutSampler(self._field, kernel, self._cursor)

    def run(self) -> TrajectoryDiagnosticReport:
        with torch.no_grad():
            generator = torch.Generator().manual_seed(self._config.schedule.seed)
            noises = torch.randn(
                (self.ANCHOR_NOISES, *self._config.latent_shape), generator=generator,
            )
            mapping = ModalityMapping.load(self._config.artifacts.modality_mapping_json)
            condition = RolloutCondition(
                label=torch.tensor([mapping.label(name) for name in MODALITIES]),
                spacing=torch.full((self.ANCHOR_NOISES, 3), CONDITION_SPACING),
            )
            anchor = self._sampler.anchor_trajectory(noises, condition)
            reference = self._monai_reference_trajectory(noises, condition)
            pairs, perturbed = self._perturb_all(anchor, generator, condition)
            return TrajectoryDiagnosticReport(
                eta=self._kernel.eta,
                s_max=self._config.policy.sde_s_max,
                num_inference_steps=self._config.policy.num_inference_steps,
                input_img_size_numel=self._config.policy.input_img_size_numel,
                perturbation_steps=sorted(self._config.policy.train_step_indices_m),
                schedule_timesteps=self._cursor.timesteps.tolist(),
                anchor_trajectory=self._fingerprint_trajectory(anchor),
                monai_reference_trajectory=self._fingerprint_trajectory(reference),
                logprob_pairs=pairs,
                control_terminals=TerminalSampleStats.of(anchor[-1]),
                perturbed_terminals=TerminalSampleStats.of(perturbed),
            )

    def _monai_reference_trajectory(
        self,
        noises: torch.Tensor,
        condition: RolloutCondition,
    ) -> list[torch.Tensor]:
        """MONAI ``RFlowScheduler.step()`` 直接驱动的对照轨迹（同一 velocity
        来源——parity 隔离的正是单步算术，velocity 正确性由组合场测试锚定）。"""
        trajectory = [noises]
        x = noises
        for index in range(self._cursor.num_steps):
            velocity = self._field.velocity(x, self._cursor.timestep(index), condition)
            x, _ = self._scheduler.step(
                velocity,
                self._cursor.timestep(index),
                x,
                self._cursor.next_timestep(index),
            )
            trajectory.append(x)
        return trajectory

    def _perturb_all(
        self,
        anchor: list[torch.Tensor],
        generator: torch.Generator,
        condition: RolloutCondition,
    ) -> tuple[list[LogProbPair], torch.Tensor]:
        """每个被优化训练步 k：G 方向扰动 + log-prob 对 + ODE 续跑终点。"""
        group_size = self._config.policy.group_size_g
        pairs: list[LogProbPair] = []
        terminals: list[torch.Tensor] = []
        for step in sorted(self._config.policy.train_step_indices_m):
            for noise_index in range(self.ANCHOR_NOISES):
                x_k = anchor[step][noise_index:noise_index + 1]
                noise_condition = RolloutCondition(
                    label=condition.label[noise_index:noise_index + 1],
                    spacing=condition.spacing[noise_index:noise_index + 1],
                )
                noise = torch.randn(
                    (group_size, *self._config.latent_shape), generator=generator,
                )
                directions, recorded = self._sampler.perturb_group(
                    x_k, step, noise_condition, noise,
                )
                if recorded is not None:  # η=0 确定性步无密度、无 log-prob 对
                    recomputed = self._sampler.evaluate_log_prob(
                        x_k, step, noise_condition, directions,
                    )
                    for direction in range(group_size):
                        pairs.append(LogProbPair(
                            step_index=step,
                            noise_index=noise_index,
                            direction=direction,
                            recorded=recorded[direction].item(),
                            recomputed=recomputed[direction].item(),
                        ))
                terminals.append(self._sampler.continue_to_terminal(
                    directions, step, noise_condition,
                ))
        return pairs, torch.cat(terminals)

    def _fingerprint_trajectory(
        self, trajectory: list[torch.Tensor],
    ) -> list[TrajectoryStepStats]:
        stats = []
        for step_index, latent in enumerate(trajectory):
            timestep = (
                self._cursor.timestep(step_index)
                if step_index < self._cursor.num_steps else 0.0
            )
            stats.append(LatentFingerprint(latent).to_step_stats(step_index, timestep))
        return stats
