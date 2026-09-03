"""模块骨架：torchrun/FSDP 初始化、判别器 DDP 封装（不分片、每 rank 完整副本 +
梯度 allreduce）、Real sample pool 的 rank 切片、barrier。

关键接口（spec #15 模块划分）：环境装配 / 集合通信辅助。
实现由后续 ticket 交付；拓扑与降级链见 docs/spec/orchestration.md 与 ADR-0003。
"""
