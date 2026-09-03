"""模块骨架：PatchDiscriminator 封装、LSGAN 损失、raw real-logit reward、
Replay buffer（固定 base 分区 + FIFO 近期分区）、held-out 监控信号；
Real sample pool / Held-out real / per-channel 标准化统计量的构建
（``prepare`` 子命令背后）。

关键接口（spec #15 模块划分）：打分（latent → 标量）/ Online update 一步 /
pool 与统计量构建 / 信号导出。实现由后续 ticket 交付；数值语义见
docs/spec/reward-model.md 与 ADR-0001。
"""
