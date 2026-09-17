# MR-RATE 基座网络工件（wayfinder #120：零依赖只读转写）

装载 MR-RATE 基座（HF `nvidia/NV-Generate-MR-Brain`）所需的**网络架构
转写工件**。零依赖原则下上游只读参照、永不 import——唯一接口是
checkpoint 文件 + 网络配置 JSON；网络类全部来自 MONAI 库本身
（`netbuild` 装配面）。

## 文件

| 文件 | 内容 | 转写来源 |
|---|---|---|
| `unet_config.json` | `DiffusionModelUNetMaisi` 构造参数（键 = MONAI 参数名） | 上游 `configs/config_network_rflow.json` 的 `diffusion_unet_def` |
| `vae_config.json` | `AutoencoderKlMaisi` 构造参数 | 同上文件的 `autoencoder_def` |

## 转写来源（上游锚）

- **上游仓库**：`NVIDIA-Medtech/NV-Generate-CTMR`（真上游，HEAD
  `da438fe`；本仓 MR-RATE 线一律锚真上游，fork 仅 BraTS 线——见
  CONTEXT.md「上游锚」词条）。
- **来源文件**：`configs/config_network_rflow.json`（网络定义单一来源，
  `docs/inference.md` 指定推理经 `-t` 装载它）。
- **转写规则**：上游 config 是 MONAI ini 化形态——`_target_` 指类、
  `@key` 是节点内插值。转写 = 解开插值、剥掉 `_target_`，落成
  「键 = MONAI 构造参数名」的平面 dict（`netbuild` 装配约定）。逐字段
  对照：

### UNet（`diffusion_unet_def` → `unet_config.json`）

| 上游字段 | 值 | 转写后 | 说明 |
|---|---|---|---|
| `_target_` | `...DiffusionModelUNetMaisi` | （剥除） | 类选择由 `netbuild.NetworkAssembler.unet` 承载 |
| `spatial_dims` | `"@spatial_dims"` | `3` | 插值解包 |
| `in_channels` / `out_channels` | `"@latent_channels"` | `4` | latent 通道（与 latent [4,·,·,·] 契约同构） |
| `num_channels` | `[64,128,256,512]` | 原样 | |
| `attention_levels` | `[false,false,true,true]` | 原样 | |
| `num_head_channels` | `[0,0,32,32]` | 原样 | |
| `num_res_blocks` | `2` | 原样 | |
| `use_flash_attention` | `true` | 原样 | DCU 栈回退 SDPA 实现并告警（Torch 未编 memory-efficient attention）；网络结构不受影响 |
| `include_top_region_index_input` / `include_bottom_region_index_input` | `"@include_body_region"` → `false` | `false` | 插值解包（基座无 body region 条件） |
| `include_spacing_input` | `true` | 原样 | spacing ×1e2 恒传（policy-modeling 章） |
| `num_class_embeds` | `128` | 原样 | 模态 token 查表（`nn.Embedding(128,256)`） |
| `resblock_updown` / `include_fc` | `true` | 原样 | |

### VAE（`autoencoder_def` → `vae_config.json`）

`autoencoder_def` 同样是 CT/MR 共用件：`autoencoder_v1.pt` 是 foundation
VAE（NV-Generate-CT 发布，MR-RATE 基座与之配对——上游
`environment_maisi_diff_model_rflow-mr-brain.json` 的
`trained_autoencoder_path` 即它）。转写映射同上（`@spatial_dims`→3、
`@image_channels`→1、`@latent_channels`→4、`norm_float16=true`、
`num_splits=4`、`dim_split=1`、`use_convtranspose=false`、
三级 `[64,128,256]` 无注意力、无 nonlocal attn、关闭 checkpointing）。

### 未转写的段落（有意缺席）

- `controlnet_def` / `mask_generation_*`：本线无 MR ControlNet（上游未
  发布）、掩码生成不在范围（地图 #67「范围外」）。
- `noise_scheduler`：scheduler 由 `NetworkAssembler.rflow_scheduler`
  按**实际生效行为**装配（config 字面 `scale:1.4` 是死参数、实际生效
  1.0——ADR-0002 + spec「sigma 日程」的 scale 陷阱）。
- `autoencoder_def` 的 `save_mem` / `print_info` 等非架构默认值不在
  转写内（构造器默认即上游生效值）。

## Checkpoint（权重文件事实）

HF `nvidia/NV-Generate-MR-Brain`，两份发布件（集群落位见 sugon-deploy
「数据落位」；当前 sugon 在 `/root/private_data/nv-ctmr/models/`）：

| 文件 | 形态 | 实测 |
|---|---|---|
| `models/diff_unet_3d_rflow-mr-brain_v1.pt` | **上游训练 checkpoint 容器**：`{epoch, epoch_finished, loss, num_train_timesteps, scale_factor, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, unet_state_dict}` | 内层 `unet_state_dict` 435 键、张量总量 **180,500,868**（strict 装载通过 = 与转写架构逐键自洽）；`scale_factor` = **0.9696779847145081**（= 1/std(z)，解码前除回 encoder 域的权威值，MONAI `MetaTensor` 形态落盘） |
| `models/autoencoder_v1.pt` | 裸 state_dict（`AutoencoderKlMaisi` 键形） | 张量总量 **20,944,897** |

装载面的两个承接点（`netbuild`）：训练 checkpoint 容器按包装键
（`unet_state_dict` / `controlnet_state_dict`）解包；容器标量的
MetaTensor 形态经 torch safe globals 白名单登记后仍走
`weights_only=True` 严格反序列化（不开任意反序列化逃生门）。

### 参数量口径与 HF 模型卡差异（登记待解）

- HF 模型卡（`nvidia/NV-Generate-MR-Brain`）：**"Number of model
  parameters: 240M"**。
- 实测：UNet **180.5M**（发布权重逐键自洽），UNet+VAE 合计 201.4M——
  均对不上 240M。卡片数字与发布权重不可复算对上，来源不明（疑沿用
  MAISI CT 线口径）。
- **裁决口径**：以**发布权重工件**为唯一权威——本票 AC「参数量与上游
  口径一致（240M 级）」的执行形态 = `base-smoke` 报告里的
  「装载出的模型参数量 == 权重文件张量总量」对账 + 上表实测锚；
  模型卡 240M 与实测的差异在此登记，不阻塞消费票（#121/#122 不消费
  该数字）。若需向上游求证另开票。

## 自检入口（base-smoke）

```bash
python3 -m cynosure.cli base-smoke --config <smoke config JSON>
# config 字段：unet_ckpt / unet_config_json / vae_ckpt / vae_config_json /
#             output_json（+ 定点输入 knobs：latent_shape / modality_token /
#             spacing / timestep / seed，缺省即生产基准网格 [4,64,64,32]）
```

报告（`output_json`）载：参数量对账、定点 latent + 定点模态 token 前向
的逐位复现判定与 sha256 指纹、VAE 生产尺寸 encode/decode 往返形状与
fp16 autocast 口径读数。sugon DCU（torch-dcu 2.9.0）首跑读数：
velocity sha256 `f44eeb55…`、两次前向与两次独立进程运行逐位一致。

### 边界

- 定点模态 token 缺省 9 = `t1w`（上游 `configs/modality_mapping.json`
  的 MR-RATE 权威映射 t1w/t2w/flair/swi/mra → 9/10/11/20/16）；
  token 词表的 config 化由 #119 承接，本目录不放 mapping 工件。
- 定点 spacing 缺省 [0.94,0.94,1.36]（上游
  `configs/config_maisi_diff_model_rflow-mr-brain.json` 的推理 spacing）。
- 基准网格 [4,64,64,32] smoke 即可；多网格承接由 #111 网格裁决后另行
  表达，不在 #120。

## 许可

`diff_unet_3d_rflow-mr-brain_v1.pt`：NVIDIA Open Model License（HF 模型
卡）；`autoencoder_v1.pt`：同 NV-Generate-CT 发布条款。上游代码只读
参照、永不 import。

## modality_mapping.json（MR-RATE 线装载面工件）

上游 `configs/modality_mapping.json`（真上游 `da438fe`）的转写子集：
whole-brain MR 五模态条目（`mri_t1/t2/flair/swi/mra` → 9/10/11/20/16，
#119 权威口径）+ BraTS 四序列 skull-stripped 条目（29/34/30/31 中的
29/34/31；`t2w` 键在本工件取 MR 值 10——同键两域不同 token，MR-RATE
线的文件以本线口径为准）。

装载语义（`cynosure.conditions.ModalityMapping`）：构造期校验只要求
BraTS 四序列键在册（`t2w` 重复键语义见上）；**MR-RATE 线的条件 token
实际取数走条件词汇表工件的 `modality_tokens`**（单一来源，#127）——
本文件是 `ModalityMapping.load` 的装载面满足 + 跨域映射留档，不是
MR token 的消费来源。
