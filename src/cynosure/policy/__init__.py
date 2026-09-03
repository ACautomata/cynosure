"""模块骨架：scheduler 轨迹游标（快照 timesteps 防共享调度器被复写）、Anchor 轨迹、
单步 SDE 核、ODE 续跑、per-group 采样场、log-prob。

关键接口（spec #15 模块划分）：anchor 轨迹采样 / 单步 SDE / ODE 续跑 / log-prob 评估。
实现由后续 ticket 交付；数值语义见 docs/spec/policy-modeling.md（ADR-0002：
对齐基座实际生效行为，CFG 组合场在此模块是唯一发生地）。
"""
