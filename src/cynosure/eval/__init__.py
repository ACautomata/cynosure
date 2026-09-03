"""模块骨架：VAE 解码 + 2.5D FID/KID（RadImageNet-ResNet50、三正交面）、
3D SSIM/MAE、nnUNet 对齐接口、盲审导出。

关键接口（spec #15 模块划分）：从 checkpoint + Real sample pool 产出指标与
盲审材料。实现由后续 ticket 交付；验收阶梯见 docs/spec/experiment-design.md
与 ADR-0004（解码只发生在里程碑评测，不进训练循环）。
"""
