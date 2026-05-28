# Anomalib 训练脚本 - 模型与参数完整参考

本文档涵盖 `train/train.py` 的所有可用参数、三种预训练权重模式、以及各模型专属配置项。

---

## 目录

1. [快速开始](#1-快速开始)
2. [模型选择](#2-模型选择)
3. [权重模式与 backbone/encoder 对照表](#3-权重模式与-backboneencoder-对照表)
4. [通用参数](#4-通用参数)
5. [数据集参数](#5-数据集参数)
6. [数据加载参数](#6-数据加载参数)
7. [模型特有参数](#7-模型特有参数)
8. [AnomalyDINO 专用参数](#8-anomalydino-专用参数)
9. [阈值与后处理参数](#9-阈值与后处理参数)
10. [训练控制参数](#10-训练控制参数)
11. [可视化与测试参数](#11-可视化与测试参数)
12. [各模型默认值速查表](#12-各模型默认值速查表)

---

## 1. 快速开始

### 最简命令

```bash
# PatchCore (默认模型，resnet18 backbone，timm 权重)
python train/train.py

# AnomalyDINO (DINOv2 ViT-Small)
python train/train.py --model anomaly_dino --dino-weights ./dinov2_vits14_pretrain.pth
```

### 完整命令示例

```bash
# MMDet 模式: PatchCore + ResNet50 检测预训练
python train/train.py \
  --exp-name train_mmdet_R50_patchcore \
  --model patchcore \
  --backbone resnet50 \
  --layers layer2 layer3 \
  --mmdet-weights weights/mmdet/TB_R50_20260520.pth \
  --coreset-sampling-ratio 0.05 \
  --pixel-sensitivity 0.5 \
  --no-vis-test

# Timm 模式: PaDiM + ResNet18 ImageNet 预训练
python train/train.py \
  --model padim \
  --backbone resnet18 \
  --weights-path weights/timm/resnet18.pth \
  --n-features 100

# DINOv2 模式: AnomalyDINO + ViT-Small + 背景掩码抑制
python train/train.py \
  --model anomaly_dino \
  --dino-weights weights/dino/dinov2_vits14_pretrain.pth \
  --encoder-name dinov2_vit_small_14 \
  --masking \
  --image-size 252

# DINOv3 模式: AnomalyDINO + ViT-Base (推荐)
python train/train.py \
  --model anomaly_dino \
  --dino-weights Z:\...\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --encoder-name dinov3_vitb16 \
  --masking \
  --image-size 252
```

---

## 2. 模型选择 (`--model`)

| 值 | 全称 | 训练方式 | 默认 epochs | 精度 | 速度 | 显存需求 |
|---|---|---|---|---|---|---|
| `patchcore` | **PatchCore** | Memory Bank（无梯度） | 1 | ★★★★☆ | ★★★☆☆ | 高（特征库） |
| `simplenet` | **SimpleNet** | 梯度训练 | 100 | ★★★☆☆ | ★★★★★ | 低 |
| `rd` | **Reverse Distillation** | 梯度训练 | 100 | ★★★★☆ | ★★★★☆ | 中 |
| `padim` | **PaDiM** | Memory Bank（统计量） | 1 | ★★★☆☆ | ★★★★★ | 最低 |
| `anomaly_dino` | **AnomalyDINO** | Memory Bank（K-Center-Greedy） | 1 | ★★★★★ | ★★★☆☆ | 高 |

> **Memory Bank 方法** (PatchCore / PaDiM / AnomalyDINO): 1 个 epoch 即可完成"训练"（实际是特征提取+建库），不需要梯度计算。
>
> **梯度训练方法** (SimpleNet / RD): 需要多 epoch 迭代优化网络参数，自动启用 FP16 混合精度节省显存。

---

## 3. 权重模式与 backbone/encoder 对照表

### 三种权重模式概览

| 模式 | 核心参数 | 权重来源 | 适用模型 |
|---|---|---|---|
| **DINO** (`--dino-weights`) | `--encoder-name` | ViT 自监督预训练 `.pth` | **仅** anomaly_dino |
| **MMDet** (`--mmdet-weights`) | `--backbone` | MMDetection 目标检测 `.pt`/`.pth` | patchcore, padim, rd 等 CNN 模型 |
| **Timm** (`--weights-path`) | `--backbone` | timm ImageNet 分类预训练 `.pth` | patchcore, padim, simplenet, rd |

> 三种模式 **互斥**，每次只能指定一种。不指定时使用对应模式的在线下载/内置权重。

---

### 3.1 DINO 模式 — encoder_name 参数值

#### DINOv2 系列（Anomalib 内置支持）

| `--encoder_name` | 架构 | embed_dim | num_heads | 参数量 |
|---|---|---|---|---|
| `dinov2_vit_small_14` | ViT-S/14 | 384 | 6 | ~22M |
| `dinov2_vit_base_14` | ViT-B/14 | 768 | 12 | ~86M |
| `dinov2_vit_large_14` | ViT-L/14 | 1024 | 16 | ~304M |
| `dinov2reg_vit_small_14` | ViT-S/14 + register tokens | 384 | 6 | ~22M |
| `dinov2reg_vit_base_14` | ViT-B/14 + register tokens | 768 | 12 | ~86M |

命名规则：`dinov2[_reg]_vit_{small/base/large}_{patch_size}`

#### DINOv3 系列（需本地 DINOv3 项目）

DINOv3 与 v2 **架构完全不同**：
- 使用 **RoPE 旋转位置编码**（替代 v2 的可学习 pos_embed）
- 使用 **LayerNormBF16**（替代标准 LayerNorm）
- 新增 **storage_tokens** 和 **mask_token**
- Plus 变体使用 **SwiGLU** FFN（`ffn_ratio=6`）

| `--encoder_name` | 实际 hub 函数 | embed_dim | depth | heads | FFN | 特点 |
|---|---|---|---|---|---|---|
| `dinov3_vits16` | `dinov3_vits16()` | 384 | 12 | 6 | MLP | 最轻量 |
| `dinov3_vits16plus` | `dinov3_vits16plus()` | 384 | 12 | 6 | SwiGLU | S+增强版 |
| `dinov3_vitb16` | `dinov3_vitb16()` | 768 | 12 | 12 | MLP | **均衡推荐** |
| `dinov3_vitl16` | `dinov3_vitl16()` | 1024 | 24 | 16 | MLP | 高精度 |
| `dinov3_vitl16plus` | `dinov3_vitl16plus()` | 1024 | 24 | 16 | SwiGLU | L+增强版 |
| `dinov3_vith16plus` | `dinov3_vith16plus()` | 1280 | 32 | 20 | SwiGLU | 超大 |
| `dinov3_vit7b16` | `dinov3_vit7b16()` | 4096 | 40 | 32 | SwiGLU-64 | 巨型 (~7B) |

---

### 3.2 MMDet 模式 — backbone 参数值

用于 MMDetection 目标检测 checkpoint（通常含 `backbone.` 前缀，脚本自动剥离）。

| `--backbone` | 对应 torchvision 类 | 典型场景 |
|---|---|---|
| `resnet18` | `tv_models.resnet18` | 轻量检测 backbone |
| `resnet34` | `tv_models.resnet34` | 中等规模 |
| `resnet50` | `tv_models.resnet50` | **最常用** |
| `resnet101` | `tv_models.resnet101` | 大型检测模型 |
| `resnext50_32x4d` | `tv_models.resnext50_32x4d` | 高吞吐量 |
| `wide_resnet50_2` | `tv_models.wide_resnet50_2` | PatchCore 默认 |
| `wide_resnet101_2` | `tv_models.wide_resnet101_2` | StPM 推荐 |

---

### 3.3 Timm 模式 — backbone 参数值

#### CNN Backbone

| `--backbone` | 架构 | 适用模型 |
|---|---|---|
| `resnet18` | ResNet-18 | PaDiM, RD, CFF |
| `resnet34` | ResNet-34 | 同上 |
| `resnet50` | ResNet-50 | 同上 |
| `wide_resnet50_2` | WRN-50-2 | PatchCore, CFF |
| `wide_resnet101_2` | WRN-101-2 | PatchCore, StPM |
| `tf_efficientnet_b4` | EfficientNet-B4 | 各模型通用 |
| `convnext_tiny` | ConvNeXt-Tiny | 各模型通用 |
| `convnext_small` | ConvNeXt-Small | 各模型通用 |
| `convnext_base` | ConvNeXt-Base | 各模型通用 |

#### ViT Backbone

| `--backbone` | 架构 | 适用模型 |
|---|---|---|
| `vit_base_patch16_224` | ViT-B/16 | PaDiM 等 |
| `vit_large_patch16_224` | ViT-L/16 | PaDiM 等 |

> ⚠️ **SimpleNet 特殊要求**: `--backbone` 必须以 `.tv_in1k` 结尾，如 `resnet18.tv_in1k`

---

## 4. 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--exp-name` | `str` | `train_<时间戳>` | 实验名称，决定输出目录名 (`results/<exp-name>/`) |
| `--model` | `str` | `patchcore` | 模型选择: `patchcore`, `simplenet`, `rd`, `padim`, `anomaly_dino` |
| `--pre-trained` | `bool` | `True` | 是否使用预训练 backbone（`--no-pre-trained` 关闭） |
| `--image-size` | `int` | `256` | 输入图像尺寸（正方形 H=W）。AnomalyDINO 默认 252（能被 14 整除） |
| `--input-size` | `[int,int]` | `[256,256]` | 输入尺寸 (H W)，RD 模型必需，默认跟随 `--image-size` |
| `--seed` | `int` | `42` | 随机种子（数据划分、采样等） |

---

## 5. 数据集参数

数据集采用 **Folder 格式**（anomalib 标准），目录结构：

```
<dataset_root>/
├── <normal_dir>/        # 正常样本（用于训练）
│   ├── img001.bmp
│   └── ...
├── <abnormal_dir>/      # 异常样本（用于测试）
│   ├── img002.bmp
│   └── ...
└── <mask_dir>/          # 异常掩码（可选）
    ├── img002.png
    └── ...
```

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dataset-root` | `str` | `Z:\...\TB` | 数据集根目录 |
| `--normal-dir` | `str` | `OK_V2` | 正常样本子目录名 |
| `--abnormal-dir` | `str` | `defects` | 异常样本子目录名 |
| `--mask-dir` | `str` | `masks` | 掩码标注子目录名 |
| `--extensions` | `[str]` | `.bmp .jpg .png` | 支持的图像扩展名 |
| `--test-split-ratio` | `float` | `0.2` | 从 abnormal 中划出多少比例作为测试集 |
| `--val-split-ratio` | `float` | `0.5` | 测试集中多少比例作为验证集（其余为测试集） |

---

## 6. 数据加载参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--train-batch-size` | `int` | `32` | 训练批次大小 |
| `--eval-batch-size` | `int` | `16` | 评估批次大小（**15GB 显存建议 ≤ 8**，避免阈值计算 OOM） |
| `--num-workers` | `int` | `4` | 数据加载进程数（NAS 盘建议 4~8，SSD 可设更高） |
| `--pin-memory` | `bool` | `True` | 锁页内存加速 CPU→GPU 数据传输 |

---

## 7. 模型特有参数

以下参数仅在特定模型下生效：

### PatchCore

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--coreset-sampling-ratio` | `0.1` | **核心集采样率**。从 Memory Bank 中采样的比例。降低此值可大幅减少内存和推理时间。建议范围 `0.01~0.1`，显存紧张时可设 `0.01` |
| `--backbone` | `resnet18` | CNN 骨干网络名（见第 3 节对照表） |
| `--layers` | `["layer2","layer3"]` | 特征提取层，决定从 backbone 的哪些层提取特征 |

### SimpleNet

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--perlin-threshold` | `0.2` | Perlin 噪声阈值，用于生成伪异常样本 |
| `--backbone` | `resnet18.tv_in1k` | **必须以 `.tv_in1k` 结尾**（torchvision ImageNet 权重格式） |

### PaDiM

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--n-features` | `100` (resnet18) | PCA 降维后的特征维度。`resnet18` 推荐 100，`wide_resnet50_2` 推荐 550。设为 `None` 则不降维 |

### Reverse Distillation (RD)

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--anomaly-map-mode` | `add` | 异常图融合模式：`add`（相加）或 `multiply`（相乘） |
| `--layers` | `["layer1","layer2","layer3"]` | 特征提取层（RD 默认使用 3 层） |

---

## 8. AnomalyDINO 专用参数

这些参数**仅在** `--model anomaly_dino` 时生效：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--encoder-name` | `str` | `dinov2_vit_small_14` | DINO 编码器名称（见 §3.1 完整列表） |
| `--dino-weights` | `str` | `None` | DINO 预训练权重文件路径 (`.pth`)。不指定则在线下载 |
| `--masking` | `bool` | **`False`** | 是否启用 **PCA 掩码背景抑制** |
| `--coreset-subsampling` | `bool` | **`True`** | 是否启用 **Corest 子采样**减少显存占用 |

### 8.1 `--masking` 详细说明

**作用**: 通过 PCA 分析正常图像的特征分布，生成背景掩码来抑制非目标区域（如均匀背景）的特征响应。

- **开启** (`--masking`): 在特征提取阶段计算每个像素位置的主成分方向，只保留方差大的区域特征。适用于**有明显均匀背景**的工业图像（如纹理表面），能有效减少背景噪声对异常检测的干扰。
- **关闭** (`--no-masking`): 保留所有位置的特征。适用于**全图都有信息**的图像或背景复杂的情况。

```bash
# 开启背景掩码抑制
python train/train.py --model anomaly_dino --dino-weights xxx.pth --encoder-name dinov3_vitb16 --masking

# 不使用掩码
python train/train.py --model anomaly_dino --dino-weights xxx.pth --encoder-name dinov3_vitb16 --no-masking
```

### 8.2 `--coreset-subsampling` 详细说明

**作用**: 使用 K-Center-Greedy 算法对完整 Memory Bank 进行子采样，保留最具代表性的特征向量。

- **开启** (`--coreset-subsampling`, 默认): 大幅减少 Memory Bank 大小 → 推理更快、显存更低。对于大尺寸图像或高分辨率输入几乎**必须开启**。
- **关闭** (`--no-coreset-subsampling`): 保留全部特征向量 → 信息无损但 Memory Bank 可能非常大（数 GB），导致推理缓慢甚至 OOM。

```bash
# 开启 coreset 降采样（默认，推荐）
python train/train.py --model anomaly_dino --dino-weights xxx.pth --encoder-name dinov3_vitb16 --coreset-subsampling

# 保留全部特征（仅小数据集/低分辨率时考虑）
python train/train.py --model anomaly_dino --dino-weights xxx.pth --encoder-name dinov3_vitb16 --no-coreset-subsampling
```

### 8.3 AnomalyDINO 算法流程

```
输入图像
    │
    ▼
┌─────────────────────┐
│  DINO Encoder       │  ← 提取 patch-level 特征 (ViT-S/B/L)
│  (DINOv2/DINOv3)   │     输出: (B, N_patches, embed_dim)
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    ▼           ▼
[可选]      继续处理
 PCA Masking
(背景抑制)        │
                 ▼
    ┌─────────────────────────┐
    │  K-Center-Greedy 采样   │  ← 从所有正常特征中选代表性子集
    │  (coreset_subsampling)  │     构建 Memory Bank
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │  Cosine KNN 推理        │  ← 测试时: 每个patch与Memory Bank做余弦相似度
    │  (matmul, 支持 fp16)    │     取 top-k 最近邻的距离
    └───────────┬─────────────┘
                ▼
    ┌─────────────────────────┐
    │  Top-1% Mean 聚合       │  ← 取最高的 1% 距离求均值作为该patch的异常分数
    └───────────┬─────────────┘
                ▼
         异常图 (anomaly_map) + 图像级分数 (image_score)
```

**与 PatchCore 的关键区别**:

| 维度 | AnomalyDINO | PatchCore |
|---|---|---|
| 距离度量 | **余弦距离** (`matmul`) | **欧氏距离** (`cdist`) |
| 精度类型 | 支持 fp16（matmul 天然兼容） | 通常需要 fp32 |
| 聚合策略 | **Top-1% mean** | 取最近邻距离 |
| 默认邻居数 | `num_neighbors=1` | 可配置 |
| 背景处理 | 可选 PCA masking | 无 |
| Backbone | ViT (DINOv2/v3) | CNN (ResNet/WRN) |

---

## 9. 阈值与后处理参数

这两个参数影响**预测阶段的阈值判定**和 **pred_mask 的大小**：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--image-sensitivity` | `float` | `0.5` | **图像级灵敏度** (0~1)。越高 → 更多图像被判为异常（降低漏检，提高误报） |
| `--pixel-sensitivity` | `float` | `0.7` | **像素级灵敏度** (0~1)。越高 → pred_mask 区域越大（更敏感地标记异常像素） |

```bash
# 更保守：减少误报，可能增加漏检
--image-sensitivity 0.3 --pixel-sensitivity 0.5

# 更激进：捕捉更多微小异常，可能增加误报
--image-sensitivity 0.8 --pixel-sensitivity 0.9
```

---

## 10. 训练控制参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--accelerator` | `str` | `gpu` | 计算设备: `gpu` / `cpu` |
| `--devices` | `int` | `1` | GPU 数量 |
| `--max-epochs` | `int` | `1` (MB) / `100` (训练式) | 最大训练轮次。Memory Bank 方法固定为 1；训练式方法按模型默认值 |
| `--save-last` | `bool` | `True` | 保存最后一个 epoch 的 checkpoint (`--no-save-last` 关闭) |
| `--save-top-k` | `int` | `0` | 保存验证指标最好的 k 个 checkpoint（设 0 = 不按指标保存） |
| `--save-weights-only` | `bool` | `True` | 只保存模型权重（不含 optimizer 状态），减小文件体积 |

---

## 11. 可视化与测试参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--skip-test` | `flag` | 关闭 | 跳过测试阶段（显存不足时使用） |
| `--vis-samples` | `int` | `100` | 训练完成后从异常图中随机采样预测可视化的数量。设 `0` 跳过 |
| `--no-vis-test` | `flag` | 关闭 | 测试时不输出图片到 `images/` 目录（只计算评估指标，节省磁盘空间） |

---

## 12. 各模型默认值速查表

切换 `--model` 时，以下参数会**自动适配默认值**（除非你手动指定）：

| 参数 | patchcore | simplenet | rd | padim | anomaly_dino |
|---|---|---|---|---|---|
| **backbone** | `resnet18` | `resnet18.tv_in1k` | `resnet18` | `resnet18` | *(N/A, 用 encoder)* |
| **encoder_name** | — | — | — | — | `dinov2_vit_small_14` |
| **layers** | `layer2, layer3` | `layer2, layer3` | `layer1, layer2, layer3` | `layer2, layer3` | — |
| **max_epochs** | **1** | **100** | **100** | **1** | **1** |
| **n_features** | — | — | — | `100` | — |
| **image_size** | `256` | `256` | `256` | `256` | **252** |
| **masking** | — | — | — | — | `False` |
| **coreset_subsampling** | — | — | — | — | `True` |
| **precision** | FP32 | FP16 | FP32 | FP32 | FP32 |

---

## 快速选择指南

```
你想用什么特征提取器？

┌─ ViT 自监督特征（推荐异常检测，精度最高）
│  └── --model anomaly_dino --dino-weights <path> --encoder-name <name>
│      ├── 轻量快速:  dinov2/dinov3_vits16          (~22M params)
│      ├── 均衡推荐:  dinov2/dinov3_vitb16          (~86M params)  ← 首选
│      └── 高精度:    dinov2/dinov3_vitl16          (~304M params)
│
├─ MMDet 检测预训练（COCO 预训练 CNN，目标检测领域强迁移能力）
│  └── --model patchcore --mmdet-weights <path> --backbone <name>
│      └── 常用: resnet50, wide_resnet50_2
│
└─ ImageNet 分类预训练（通用，最广泛可用）
   └── --model <任意CNN模型> --weights-path <path> --backbone <name>
       ├── 轻量: resnet18, convnext_tiny
       └── 重型: wide_resnet50_2, convnext_base
```

---

## 注意事项

1. **三种权重模式互斥**: `--dino-weights`, `--mmdet-weights`, `--weights-path` 只能指定一个
2. **DINOv3 前提**: 需要本地 `Z:\14-调试数据\lxm\Projects\DINOv3` 项目存在且包含完整源码
3. **MMDet 权重前缀**: 脚本会自动剥离 `backbone.` 前缀；若无此前缀则尝试直接匹配
4. **SimpleNet 特殊**: `--backbone` 必须以 `.tv_in1k` 结尾
5. **FP16 自动切换**: SimpleNet 和 RD 会自动启用 FP16 混合精度；Memory Bank 方法始终使用 FP32
6. **AnomalyDINO image_size**: 建议使用 252（能被 ViT 的 patch_size=14 整除），避免插值伪影
