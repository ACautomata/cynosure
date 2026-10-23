# MedIQA 作为外部 reward model 的可行性评估

> 日期：2026-10-23
> 关联：判别器 held-out AUC 崩坏（AUC 0.24、reward 排序反转、FID 恶化）后的「外部冻结 reward model」路线调研；承接 research/discriminator-saturation-risk.md 的判定
> 信源：
> - 仓库 `https://github.com/siyi-xun/MedIQA`（8 个文件，无子目录，无 LICENSE）
> - 论文 arXiv:2507.19004《MedIQA: A Scalable Foundation Model for Prompt-Driven Medical Image Quality Assessment》（Siyi Xun et al.，MICCAI 2025，pp. 339–349；`https://arxiv.org/abs/2507.19004`）
> - 预训练权重 release V1.0.0（Pre-release）：`https://github.com/siyi-xun/MedIQA/releases/tag/V1.0.0`
> - GitHub REST API（release 资产清单、仓库树）

---

## 0. 判定摘要

**裁决：有条件能——但当前形态不能直接上岗，且存在一个方向性风险（对扩散伪影可能「无感」而非「敏感」）必须先做对照实验排除。**

五条判定：

1. **MedIQA 是一个真正对口领域的 NR-IQA 基础模型**：无参考、跨模态（CT/MRI/眼底）、显式覆盖**脑部 MR（T1/T2/FLAIR）**，3D 体数据走「7 张显著切片 + 学习加权聚合」管线——与我们「多方位 3D 脑部 MR」的目标域是全调研空间里最近的一档。
2. **预训练权重公开可得且已验证可达**：release V1.0.0 挂 7 个 zip（每个约 780 MB，含 MRI_T1 / MRI_T2 / 预训练 state_dict 等），GitHub API 返回完整资产清单，下载 URL 返回 302 签名重定向（资产存在）。**但仓库无 LICENSE——法律上属 all-rights-reserved，研究用途处于灰色地带，商用/修改无授权。**
3. **训练信号里没有任何生成伪影成分**：上游是 DICOM 物理参数（CT mAs、MR 场强 T），下游是放射科专家打分（10 reader 众评）。论文与仓库均**无合成退化（blur/noise 类型标签）、无 GAN/扩散伪影数据、无 artifact-vs-clean 分类实验**。它学的是「真实临床采集成像质量」这个轴，不是「真实 vs 生成」这个轴。
4. **适配是可行的但要过像素域**：打分对象是 2D 224×224 切片（每体 7 张），必须 VAE 解码 + 切片选择后批量推理；ViT-base 量级 backbone，单进程 8×DCU 吞吐完全不是问题，主要新增成本是每 iteration 全量 rollout 的解码。GRPO 只需要组内排序，其标量输出的绝对标度无意义这点不构成本质障碍。**方向性风险：它对扩散模型常见伪影（过度平滑、纹理塑料感、高频虚假细节）可能打出「高质量」分——因为它见过的「低质量」是噪声/剂量不足，不是生成痕迹。这恰好是我们判别器 0.24 AUC 故障的镜像风险。**
5. **文献先例存在且偏正面，但没有 3D 医学影像的先例**：自然图像侧已有 RL+NR-IQA reward 微调扩散恢复模型的直接先例（DeQA-Score 作 reward，arXiv:2511.01645），也有 IQA reward 被 policy hack 后专门做防 hack GRPO 的对策工作（Adv-GRPO，CVPR 2026）。**脑部 MR 扩散生成 + NR-IQA reward 是空白点——意味着没有证据能外推，只能靠我们自己的对照实验。**

**前置验证实验（上岗前必做，按优先级）：**
1. **AUC 探针（一票否决）**：取上一轮 run 的失败 rollout 样本（已知 FID 恶化批）+ 真实 MR-RATE 样本，用 MedIQA 打分，算 AUC / pairwise 排序一致率。**AUC 必须显著 > 0.5 且方向正确（真实 > 生成）**，否则直接出局，连 hack 风险都不用谈。
2. **敏感度曲线**：对同一 latent 终点做 VAE 解码 + 逐步加扩散伪影（ddim step 截断、CFG 漂移），看分数是否单调恶化。若对「生成痕迹加重」毫无反应，则它对 RL 是死信号。
3. **hack 探针**：用当前 policy 的输出生成一批「判别器被骗过但视觉明显异常」的样本，检验 MedIQA 是否给高分。防 hack 是必须持续监测的（GRPO 的 group 内标准化会把任何可挖掘的 reward 方向放大）。
4. **域校准**：prompt 生成依赖一个 ViT 分类器判 modality/region/type（prompt_info_generate.zip），需确认它能正确分类我们的多模态多方位输入，否则 prompt 错位会系统性偏移分数。

---

## 1. MedIQA 是什么

| 维度 | 内容 | 出处 |
|---|---|---|
| 任务 | **无参考 IQA（NR-IQA）**，无 FR 设置 | 论文 §1/§3 |
| 输入 | 2D 图像与 3D 体数据统一处理；3D 走显著切片模块：剔除非 ROI 切片 → Z 轴分 7 区 → 每区取中层切片；全部 resize 到 224×224×3 | 论文 §3；`MedIQA_functions.py`（支持 .dcm/.jpg/.tif，7 图/样本） |
| 模态/部位 | 胸部 CT（肺窗/软组织窗）、**脑部 MRI T1/T2/FLAIR**、乳腺 DCE-MRI、腹部低剂量 CT、眼底彩照 | 论文 §3 |
| 架构 | MANIQA 改造：ViT-base patch8/224 特征 + Transposed Attention Block + Scale Swin Transformer Block，prompt（dimension/modality/region/type 四维，前三者由一个预训练 ViT 分类器自动生成）经 FC 注入 Swin 层；双分支头（patch 质量 × patch 权重加权聚合出标量）+ 7 切片学习加权 | 论文 §3；`backbone_network.py`（`vit_base_patch8_224`）；`prompt_driven_STL.py` |
| 输出 | 单一连续标量质量分；论文未声明标度；下游指标 SRCC/PLCC/RMSE | 论文 §4 |
| 训练数据 | 两阶段共 **15,045 例（8:1:1 切分）**：上游 2,500 例（院内胸 CT + 公开 MR——参考文献指 fastMRI 膝部与乳腺 DCE-MRI，与正文「brain/breast」表述不一致，**成分级规模未报告**）；下游 12,545 例（Chest-CTIQA 3D 胸 CT、LDCTIQAC2023 2D 腹 CT、ADNI 脑 MR 切片、Kaggle 糖网眼底） | 论文 §3 |
| 成绩 | 下游集成平均 SRCC 0.8422 / PLCC 0.8668；脑 T1 0.8681 / T2 0.8861 / FLAIR 0.7654（SRCC） | 论文 §4 |

注：早期检索摘要曾提示论文含 artifact detection 的 Sensitivity/AUC 实验——**复核论文全文后不成立**：唯一分类实验是 prompt 生成器的模态分类（ViT 0.9969），评估指标只有 SRCC/PLCC/RMSE。论文里没有任何 artifact-vs-clean 分类任务。

## 2. 预训练权重可得性与许可证

**Release V1.0.0（Pre-release，2025-06-19，commit f0b03ab）**，经 GitHub API（`api.github.com/repos/siyi-xun/MedIQA/releases/tags/V1.0.0`）取得资产清单，并经 302 签名重定向验证资产存在、可下载：

| 资产 | 大小 | URL |
|---|---:|---|
| MRI_T1_IQA_model.zip | 779,597,830 B | `https://github.com/siyi-xun/MedIQA/releases/download/V1.0.0/MRI_T1_IQA_model.zip` |
| MRI_T2_IQA_model.zip | 779,581,333 B | 同格式 |
| LDCTIQAC_IQA_model.zip | 779,579,601 B | 同格式 |
| Chest-CT_IQA_model.zip | 779,597,830 B | 同格式 |
| Fundus_IQA_model.zip | 779,603,715 B | 同格式 |
| Pretrain_MedIQA_state_dict.zip | 779,551,291 B | 同格式 |
| Prompt_info_generate.zip | 792,125,170 B | 同格式 |

（每个 zip 大小几乎相同，内容构成无法在不下载的情况下确认；未发现 HuggingFace / Google Drive 镜像。）

**许可证：无。** 仓库树（8 个文件：README、3 个主脚本、`backbone_network.py`、`MedIQA_functions.py`、`prompt_driven_STL.py`、`generate_info.py`、`requirements.txt`）中没有 LICENSE 文件，README 与论文均无许可声明。法律默认 all-rights-reserved：下载研究使用实践中常见但无正式授权，**商用与修改无许可支撑**；另外 `requirements.txt` 是把整个 conda 环境 freeze  dump（含 `litemedsam`、`modelscope_studio` 指向清华 OSS 的 wheel 等），直接安装有毒，需人工挑选实际依赖（核心是 torch/timm）。

## 3. 训练信号：对扩散伪影的识别能力是「未验证」而非「已证伪」

- 上游预训练信号 = **DICOM 物理参数**（CT mAs、MR 特斯拉数），作为质量的正相关代理；
- 下游微调信号 = **专家人工打分**（Chest-CTIQA 每体 10 reader；论文未说明聚合方式与评分标度）；
- **没有**：合成退化类型标签（blur/noise/compression 等）、失真参数回归、生成模型伪影样本、artifact 检测任务。

含义：MedIQA 的质量轴是「临床采集成像质量」（噪声、对比度、分辨率、剂量），真实 MR 与扩散输出在它的打分空间里可能**都落在「高质量」区**——扩散输出恰恰以「干净、平滑、无噪声」著称。除非扩散伪影以它学过的形式呈现（如纹理缺失被解读为分辨率低下），否则它对「真实 vs 生成」的区分力是开放问题，这也正是 §0 判定 1（AUC 探针一票否决）存在的理由。反面看，它对「低剂量噪声型退化」极其敏感（LDCTIQAC SRCC 0.9764），与我们判别器故障模式（把噪声当真实信号）不同源，这是它相对在线判别器的一个潜在优点。

## 4. 接入路径与吞吐

管线：latent 终点 [B,4,D,H,W] → VAE 解码回像素 → 按方位/切片策略取 7 张 224² 切片（或按它的 Z 轴 7 区分区取中层的口径对齐）→ prompt 分类器生成四维 prompt → MedIQA 前向 → 7 切片分数学习加权 → 每 rollout 一个标量。

- **解码成本**：每个 RL iteration 全量 rollout 多一次 VAE 解码，相对 rollout 本身的扩散采样是次级开销；口径变化（判分回到像素域）反而与我们 FID 恶化监控同域，可复用同一解码缓存。
- **吞吐**：ViT-base patch8/224 前向，7 slice/volume；批量大时单卡每秒可打分数千切片，8×DCU 单进程下每 iteration 打分远非瓶颈。
- **域差距**：模态匹配好（脑 T1/T2/FLAIR 原生支持；SWI/MRA 需验证 prompt 分类器覆盖——仓库只放了 T1/T2 两个 MR 权重，**FLAIR/SWI/MRA 无对应权重**，只能依赖预训练 state_dict + prompt 迁移，或未验证地挪用 T1/T2 权重）；但「质量轴错位」（采集质量 vs 生成真实性）是核心风险，见 §3。
- **工程债**：代码为研究快照质量（路径硬编码空字符串 `torch.load("")`、数据路径 `../data/`、train/test 脚本残缺），需自行补 loader 与 prompt 生成链路。

## 5. 文献先例：NR-IQA 当 reward 有人做过，医学 3D 是空白

| 先例 | 与我们的关系 | 来源 |
|---|---|---|
| Diffusion 恢复模型 + IQA reward RL（DeQA-Score 作 reward，PPO 式 MDP，难度自适应混合 SFT） | 最直接模板：证明冻结 MLLM-IQA 可作扩散模型 RL reward 且优于重构误差 reward；但任务是 2D 自然图像恢复 | arXiv:2511.01645 |
| PhySe-RPO（手术烟雾去除） | NR-IQA（CEIQ/LIQE）进 RL reward 的医学影像先例，2D | arXiv:2603.22844 |
| PixelPolish | PPO/GRPO + 复合 NR-IQA reward（CLIP-IQA/MUSIQ/NIQE/BRISQUE） | github.com/whycoming/PixelPolish |
| Adv-GRPO /「The Image as Its Own Reward」（CVPR 2026） | **反面必需读**：IQA reward 上 GRPO 的 reward hacking 与对抗对策 | CVPR 2026 |
| VisualQuality-R1 / EvoQuality / Q-Insight | GRPO 训练 IQA 模型本身（reward-model 侧技术，非使用侧） | NeurIPS 2025；ICLR 2026 |

没有人把 NR-IQA reward 用在 3D 医学影像生成上——先例支持「这条路走得通」，但不支持「这个模型在这个域上走得通」，外推无效，只能靠 §0 的四组前置实验。

## 6. 结论

**有条件能。** 它不是判别器的同构替代品（一个是「真实性判别轴」，一个是「采集成像质量轴」），但作为「判别器崩坏后的独立信号源」值得花 1–2 天做 §0 的验证实验：AUC 探针通过（>0.5 且方向正确）→ 敏感度曲线单调 → hack 探针不翻车 → 才谈接入；任一不过即出局。若出局，同族备选（MedCLIP-IQA、QUALIFIER 等）也应按同一套探针流程筛——**探针协议本身比模型选择更保值**。另外权重无 LICENSE 这一点在写入任何实验脚本前需要课题组层面确认可接受。

---

## 附：来源 URL

- 仓库：https://github.com/siyi-xun/MedIQA
- 论文：https://arxiv.org/abs/2507.19004 ；MICCAI 2025 页：https://papers.miccai.org/miccai-2025/0548-Paper2487.html
- 权重 release：https://github.com/siyi-xun/MedIQA/releases/tag/V1.0.0
- 先例：arXiv:2511.01645（RL+IQA reward 扩散恢复）、arXiv:2603.22844（PhySe-RPO）、CVPR 2026 Adv-GRPO、arXiv:2505.14460（VisualQuality-R1）、ByteDance/EvoQuality（HF）
