"""cynosure：MAISI 3D latent rectified-flow checkpoint 的 Granular-GRPO RL 后训练。

零依赖原则：不 import NV-Generate-CTMR 任何代码，唯一接口是 checkpoint 文件；
网络类（DiffusionModelUNetMaisi / AutoencoderKlMaisi / ControlNetMaisi /
RFlowScheduler / PatchDiscriminator）全部来自 MONAI 库本身。
"""

__version__ = "0.1.0"
