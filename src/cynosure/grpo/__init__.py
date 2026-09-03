"""模块骨架：GRPO 核心——advantage（组内标准化 + clamp）、MGAI（跨 λ 求和）、
ratio clip loss、逐 k 更新语义。

关键接口（spec #15 模块划分）：advantage 计算 / policy loss。
实现由后续 ticket 交付；数值语义见 docs/spec/policy-modeling.md
（MGAI、clip 1e-4、无 KL、无参考模型）。
"""
