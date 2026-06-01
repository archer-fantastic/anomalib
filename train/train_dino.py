"""DINO 系列 (DINOv2 / DINOv3) 异常检测训练脚本。

支持的模型:
  - anomaly_dino : AnomalyDINO 原始算法 (coreset + KNN 距离)
  - dino_patchcore: DINO 特征提取 + PatchCore KNN 推理

用法示例:
    # DINOv2 + AnomalyDINO (官方权重在线下载)
    python train/train_dino.py --encoder-name dinov2_vit_small_14

    # DINOv2 + AnomalyDINO (本地权重)
    python train/train_dino.py \\
        --model anomaly_dino \\
        --dino-weights weights/dino/dinov2_vits14_pretrain.pth \\
        --encoder-name dinov2_vit_small_14 \\
        --image-size 252

    # DINOv3 + AnomalyDINO
    python train/train_dino.py \\
        --model anomaly_dino \\
        --dino-weights Z:/.../dinov3_vitb16_pretrain.pth \\
        --encoder-name dinov3_vitb16

    # DINOv2 + PatchCore 推理
    python train/train_dino.py \\
        --model dino_patchcore \\
        --dino-weights weights/dino/dinov2_vits14_pretrain.pth \\
        --num-neighbors 9

    # 启用 masking (背景抑制)
    python train/train_dino.py --masking --image-size 224
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 工具函数
# ================================================================

def _add_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "src"
    sys.path.insert(0, str(src))


class _StageLogger:
    """带阶段计时的日志器"""

    def __init__(self, experiment_dir: Path):
        self.logger = self._setup(experiment_dir)
        self._stage_start: float | None = None

    def _setup(self, exp_dir: Path) -> logging.Logger:
        log = logging.getLogger("dino_experiment")
        log.setLevel(logging.DEBUG)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)
        fh = logging.FileHandler(exp_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
        return log

    def header(self, title: str) -> None:
        self.logger.info("=" * 60)
        self.logger.info(f"  {title}")
        self.logger.info("=" * 60)

    def stage(self, name: str) -> None:
        now = time.time()
        if self._stage_start is not None:
            elapsed = now - self._stage_start
            self.logger.info(f"[DONE] 耗时 {elapsed:.1f}s")
        self._stage_start = now
        self.logger.info(f"\n{'─' * 40}")
        self.logger.info(f">>> 阶段: {name}")
        self.logger.info(f"{'─' * 40}")

    def done(self) -> None:
        if self._stage_start is not None:
            elapsed = time.time() - self._stage_start
            self.logger.info(f"[DONE] 耗时 {elapsed:.1f}s")
            self._stage_start = None

    def info(self, msg: str) -> None:   self.logger.info(msg)
    def warning(self, msg: str) -> None: self.logger.warning(msg)
    def error(self, msg: str) -> None:   self.logger.error(msg)


# ================================================================
# DINO 编码器加载
# ================================================================

def load_dino_encoder(encoder_name: str, weight_path: Path | None,
                      dino_ver: str | None = None, log: _StageLogger | None = None):
    """加载 DINOv2/DINOv3 编码器。

    Args:
        encoder_name: 编码器名称，如 dinov2_vit_small_14, dinov3_vitb16
        weight_path: 本地权重文件路径，None 则在线下载
        dino_ver: 强制指定版本 ('dinov2' / 'dinov3')，None 则从文件名推断
        log: 日志器

    Returns:
        feature_encoder: 已加载权重的 ViT 模型 (eval 模式)
    """
    _log = log or _StageLogger(Path("."))

    # ---- 在线下载模式 ----
    if weight_path is None:
        from anomalib.models.components.dinov2 import DinoV2Loader
        _log.info(f"  使用 DINO 官方在线权重: {encoder_name}")
        encoder = DinoV2Loader.load(encoder_name)
        encoder.eval()
        return encoder

    # ---- 本地权重模式 ----
    wpath = Path(weight_path)
    if not wpath.is_file():
        _log.error(f"❌ 找不到权重文件: {wpath.resolve()}")
        sys.exit(1)

    fname = wpath.name.lower()
    # 从文件名推断版本
    if dino_ver is None:
        if "dinov2" in fname or fname.startswith("dinov2"):
            dino_ver = "dinov2"
        elif "dinov3" in fname or fname.startswith("dinov3"):
            dino_ver = "dinov3"
        else:
            _log.error(f"❌ 无法从文件名识别版本: {fname} (需包含 dinov2 或 dinov3)")
            sys.exit(1)

    _log.info(f"  加载 {'DINOv3' if dino_ver == 'dinov3' else 'DINOv2'} 权重: {wpath.name}")

    if dino_ver == "dinov3":
        return _load_dinov3_local(encoder_name, wpath, _log)

    # ---- DINOv2 本地加载 ----
    from anomalib.models.components.dinov2 import DinoV2Loader
    loader = DinoV2Loader()

    # 先尝试标准名称解析
    try:
        model_type, architecture, patch_size = loader._parse_name(encoder_name)
        encoder = loader.create_model(model_type, architecture, patch_size)
        _log.info(f"  架构: {model_type}/{architecture}/p{patch_size}")
    except (ValueError, KeyError):
        _log.warning(f"  DinoV2Loader 不支持 '{encoder_name}'，尝试手动构建...")
        encoder = _build_fallback_encoder(encoder_name, _log)

    state_dict = torch.load(wpath, map_location="cpu", weights_only=True)
    try:
        result = encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        _log.warning("  strict 匹配失败，使用 strict=False")
        result = encoder.load_state_dict(state_dict, strict=False)

    if result.missing_keys:
        _log.info(f"     缺失键: {len(result.missing_keys)} 个 (前3: {result.missing_keys[:3]})")
    if result.unexpected_keys:
        _log.info(f"     多余键: {len(result.unexpected_keys)} 个")

    matched = len(state_dict) - len(result.missing_keys)
    _log.info(f"     ✅ 匹配 Tensor: {matched}/{len(state_dict)}")
    encoder.eval()
    return encoder


def _load_dinov3_local(encoder_name: str, weight_path: Path, log: _StageLogger):
    """使用 DINOv3 项目自身的模型定义加载编码器。"""
    dinov3_root = Path(r"Z:\14-调试数据\lxm\Projects\DINOv3")

    hub_func_name = encoder_name
    if str(dinov3_root) not in sys.path:
        sys.path.insert(0, str(dinov3_root))

    log.info(f"  DINOv3 hub 函数: {hub_func_name}")
    log.info(f"  DINOv3 项目路径: {dinov3_root}")

    from dinov3.hub import backbones as hub_module
    hub_fn = getattr(hub_module, hub_func_name, None)
    if hub_fn is None:
        available = [k for k in dir(hub_module) if k.startswith("dinov3_")]
        raise ValueError(f"DINOv3 hub 中没有 '{hub_func_name}'。可用: {available}")

    encoder = hub_fn(pretrained=False, weights=str(weight_path))
    log.info(f"  ✅ DINOv3 编码器加载成功！")
    encoder.eval()
    return encoder


def _build_fallback_encoder(encoder_name: str, log: _StageLogger):
    """手动构建 ViT (当 DinoV2Loader 不支持该名称时)。"""
    from anomalib.models.components.dinov2 import vision_transformer as dinov2_models
    import re

    arch_map = {"s": "small", "small": "small", "b": "base", "base": "base",
                "l": "large", "large": "large"}
    m = re.search(r'vit([sbl]|small|base|large)(\d+)(plus)?', encoder_name, re.IGNORECASE)
    if not m:
        raise ValueError(f"无法解析架构: {encoder_name}")

    arch_short = m.group(1).lower()
    patch_size = int(m.group(2))
    architecture = arch_map.get(arch_short)
    if not architecture:
        raise ValueError(f"未知架构标识: {arch_short}")

    ctor_name = f"vit_{architecture}"
    ctor = getattr(dinov2_models, ctor_name, None)
    if ctor is None:
        raise ValueError(f"dinov2_models 中没有 {ctor_name}")

    model = ctor(patch_size=patch_size)
    log.info(f"  手动构建: {ctor_name}(patch_size={patch_size})")
    return model


# ================================================================
# 模型 1: DinoPatchcoreModel (DINO 特征 + PatchCore KNN)
# ================================================================
class InferenceBatch:
    """推理输出格式 (兼容 anomalib)。"""
    def __init__(self, pred_score: torch.Tensor, anomaly_map: torch.Tensor,
                 pred_label: torch.Tensor | None = None,
                 pred_mask: torch.Tensor | None = None):
        self.pred_score = pred_score
        self.anomaly_map = anomaly_map
        self.pred_label = pred_label
        self.pred_mask = pred_mask


class DinoPatchcoreModel(nn.Module):
    """DINOv2/v3 特征提取 + PatchCore KNN 推理。

    Pipeline:
      1. training: 提取 patch features → 存入 embedding_store
      2. fit(): 合并所有 embedding → memory_bank
      3. inference: cosine distance to memory_bank → top-k mean → anomaly_map
    """

    def __init__(self, feature_encoder: nn.Module, num_neighbors: int = 9,
                 image_size: int = 252):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.num_neighbors = num_neighbors
        self.image_size = image_size
        self.patch_size = getattr(feature_encoder, "patch_size", 14)

        self.embedding_store: list[torch.Tensor] = []
        self.memory_bank: torch.Tensor | None = None

        self.register_buffer("dtype_dummy", torch.zeros(()))

    @property
    def device(self) -> torch.device:
        return self.dtype_dummy.device

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        b, c, h, w = input_tensor.shape
        ps = self.patch_size

        # Center crop 到 patch_size 整除
        ch, cw = h % ps, w % ps
        if ch > 0 or cw > 0:
            input_tensor = input_tensor[:, :, ch // 2:h - ch + ch // 2,
                                         cw // 2:w - cw + cw // 2]
            _, _, h, w = input_tensor.shape

        grid_h, grid_w = h // ps, w // ps

        with torch.inference_mode():
            raw = self.feature_encoder.get_intermediate_layers(
                input_tensor, n=1, reshape=True)[0]
            # raw: (B, D, H_patches, W_patches)
            features = raw.flatten(2).transpose(1, 2)  # (B, N, D)

        # L2 normalize
        features = F.normalize(features, p=2, dim=-1)

        if self.training:
            self.embedding_store.append(features.detach().cpu())
            return torch.tensor(0.0, requires_grad=True, device=input_tensor.device)

        # Inference: KNN distance → anomaly_map
        if self.memory_bank is None or self.memory_bank.numel() == 0:
            dummy_map = torch.zeros(b, 1, grid_h, grid_w, device=input_tensor.device)
            return InferenceBatch(
                pred_score=torch.ones(b, device=self.device),
                anomaly_map=dummy_map,
            )

        bank = self.memory_bank.to(features.device)  # (M, D)
        dists = 1 - features @ bank.T  # (B, N, M) cosine distance

        topk_vals, _ = dists.topk(
            min(self.num_neighbors, bank.shape[0]), dim=-1, largest=False)
        patch_scores = topk_vals.mean(dim=-1)  # (B, N)

        anomaly_map = patch_scores.view(b, 1, grid_h, grid_w)
        max_score = anomaly_map.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
        anomaly_map = anomaly_map / max_score
        pred_score = anomaly_map.amax(dim=(1, 2, 3))  # (B,)

        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def fit(self) -> None:
        """合并 embedding_store → memory_bank (coreset 采样可选)。"""
        if len(self.embedding_store) == 0:
            print("[DinoPatchcore] ⚠️ embedding_store 为空!", flush=True)
            return

        all_features = torch.cat(self.embedding_store, dim=0)  # (total_N, D)
        self.memory_bank = all_features.float()
        n, d = self.memory_bank.shape
        print(f"[DinoPatchcore] ✅ memory_bank built: ({n}, {d})", flush=True)
        self.embedding_store.clear()


class DinoPatchcore:
    """DINO+PatchCore 的 Lightning wrapper (兼容 AnomalibModule)。"""

    def __init__(self, feature_encoder, num_neighbors: int = 9,
                 coreset_sampling_ratio: float = 0.1,
                 visualizer=None, post_processor=None):
        super().__init__()
        self.model = DinoPatchcoreModel(feature_encoder, num_neighbors)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.visualizer = visualizer
        self.post_processor = post_processor
        self._input_size: tuple[int, int] = (252, 252)

    def configure_pre_processor(self, input_size: tuple[int, int]):
        """配置 DINO 标准预处理 (Resize + Normalize)。"""
        from torchvision.transforms import v2 as T
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self.pre_processor = T.Compose([
            T.Resize(input_size, antialias=True),
            T.ToImage(), T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])
        self._input_size = input_size

    def training_step(self, batch, *args, **kwargs):
        img = batch["image"] if isinstance(batch, dict) else batch.image
        return self.model(img.to(self.model.dtype))

    def on_fit_end(self):
        self.model.fit()

    def validation_step(self, batch, *args, **kwargs):
        return self._inference_step(batch)

    def test_step(self, batch, *args, **kwargs):
        return self._inference_step(batch)

    def predict_step(self, batch, *args, **kwargs):
        return self._inference_step(batch)

    def _inference_step(self, batch):
        img = batch["image"] if isinstance(batch, dict) else batch.image
        result = self.model(img)
        if isinstance(result, InferenceBatch) and self.post_processor:
            return self.post_processor(
                pred_score=result.pred_score,
                anomaly_map=result.anomaly_map,
                pred_label=result.pred_label,
                pred_mask=result.pred_mask,
            )
        return result

    @property
    def dtype(self):
        return self.model.dtype_dummy.dtype


# ================================================================
# 模型 2: AnomalyDINO 包装 (含 Memory Bank 安全网)
# ================================================================

def create_anomaly_dino_model(args, dino_config: dict | None,
                              images_dir: Path | None,
                              post_processor, log: _StageLogger):
    """创建 AnomalyDINO 模型，含自定义权重注入和安全网 hooks。

    Args:
        args: 命令行参数
        dino_config: DINO 配置字典 (来自 --dino-weights)，None 表示在线下载
        images_dir: 可视化输出目录
        post_processor: 后处理器
        log: 日志器
    """
    from anomalib.models.image.anomaly_dino.lightning_model import AnomalyDINO
    from anomalib.models.components.dinov2 import DinoV2Loader
    from anomalib.visualization.image.visualizer import ImageVisualizer

    vis_kwarg = False if images_dir is None else ImageVisualizer(output_dir=images_dir)
    encoder_name = args.encoder_name
    masking = args.masking
    coreset_subsampling = args.coreset_subsampling
    sampling_ratio = args.coreset_sampling_ratio

    log.info(f"  encoder_name         : {encoder_name}")
    log.info(f"  masking              : {masking}")
    log.info(f"  coreset_subsampling  : {coreset_subsampling}")

    # 自定义 DINO 权重注入
    if isinstance(dino_config, dict) and dino_config.get("type") == "dino":
        feature_encoder = load_dino_encoder(
            encoder_name, Path(dino_config["path"]),
            dino_ver=dino_config.get("version"), log=log)

        log.info(f"  ✅ 自定义 DINO 权重加载成功！")
        _original_load = DinoV2Loader.load
        def _fake_load(cls_self, model_name):
            return feature_encoder
        DinoV2Loader.load = _fake_load
        try:
            model = AnomalyDINO(
                num_neighbours=1,
                encoder_name=encoder_name,
                masking=masking,
                coreset_subsampling=coreset_subsampling,
                sampling_ratio=sampling_ratio,
                visualizer=vis_kwarg,
                post_processor=post_processor,
            )
        finally:
            DinoV2Loader.load = _original_load
    else:
        log.info(f"  使用 DINO 官方在线权重")
        model = AnomalyDINO(
            num_neighbours=1,
            encoder_name=encoder_name,
            masking=masking,
            coreset_subsampling=coreset_subsampling,
            sampling_ratio=sampling_ratio,
            visualizer=vis_kwarg,
            post_processor=post_processor,
        )

    # 安装 Memory Bank 安全网
    model = _install_memory_bank_safety_net(model, log)
    return model


def _install_memory_bank_safety_net(model, log: _StageLogger):
    """为 AnomalyDINO 安装安全网 hooks，确保 Memory Bank 正确构建。

    完全替换 training_step (非包装)，因为原始实现用 batch.image 属性访问，
    但 Folder 数据集返回 dict 格式 (batch["image"])。
    同时确保 torch_model 在训练期间保持 train 模式 (解决 eval mode 问题)。
    """
    import types, logging as _logging
    import numpy as _np

    tm = model.model  # AnomalyDINOModel 实例
    mb_log = _logging.getLogger(__name__)

    def _new_training_step(self, batch, *args, **kwargs):
        """完全替换: 直接提取特征到 embedding_store，兼容 dict 和 object batch。"""
        del args, kwargs
        if not tm.training:
            tm.train()
            mb_log.warning("[安全网] 强制 torch_model.train()")

        img = batch["image"] if isinstance(batch, dict) else batch.image

        input_tensor = img.type(tm.memory_bank.dtype)
        _, _, h, w = input_tensor.shape
        ps = tm.feature_encoder.patch_size
        ch, cw = h % ps, w % ps
        if ch > 0 or cw > 0:
            input_tensor = input_tensor[:, :, ch // 2:h - ch + ch // 2,
                                         cw // 2:w - cw + cw // 2]
        grid = ((h - ch) // ps, (w - cw) // ps)
        dev = input_tensor.device

        with torch.inference_mode():
            # reshape=True: DINO 内部按 (W,H) reshape 再 permute → (D,H,W)
            feat_3d = tm.feature_encoder.get_intermediate_layers(input_tensor, n=1, reshape=True)[0]
            # feat_3d: (B, D, H_patches, W_patches) → flatten to (B, N, D)
            feats = feat_3d.flatten(2).transpose(1, 2)

        feats_full = feats.clone() if tm.masking else None

        if tm.masking:
            masks_np = type(tm).compute_background_masks(feats.detach().cpu().numpy(), grid)
            masks = torch.from_numpy(masks_np).to(dev)
        else:
            masks = torch.ones(feats.shape[:2], dtype=torch.bool, device=dev)

        feats = feats[masks]

        if feats.size(0) == 0 and feats_full is not None:
            mb_log.warning("[安全网] masking 滤掉全部 patch! 降级为全量特征")
            feats = feats_full.reshape(-1, feats_full.shape[-1])

        feats = F.normalize(feats, p=2, dim=1)
        tm.embedding_store.append(feats)

        mb_log.debug("[安全网] store=%d last=%s", len(tm.embedding_store), feats.shape)
        return torch.tensor(0.0, requires_grad=True, device=input_tensor.device)

    model.training_step = _new_training_step.__get__(model, type(model))

    # 2. on_train_epoch_end hook — 完全接管 fit() 逻辑
    # 原始 MemoryBankMixin.on_train_epoch_end 会调用 self.fit() → tm.fit()
    # 但我们已在这里完成 fit 并清空了 store，原始 hook 再调用必报错
    # 方案: 自己处理 fit + 设置 _is_fitted=True 阻止原始 hook 重复执行
    orig_epoch_end = getattr(model, "on_train_epoch_end", None)
    def _safe_epoch_end(self):
        n_embed = len(tm.embedding_store)
        mb_n = tm.memory_bank.numel()
        mb_log.info("[安全网] epoch_end: store=%d bank=%d", n_embed, mb_n)

        if mb_n == 0:
            # memory_bank 未建库，需要 fit
            if n_embed > 0:
                mb_log.info("[安全网] fit() with %d chunks...", n_embed)
                tm.fit()  # 直接调 torch_model 的 fit，不经过 lightning 层
                mb_log.info("[安全网] ✅ memory_bank=%s", tm.memory_bank.shape)
            else:
                mb_log.error("[安全网] FATAL: embedding_store 为空! 无法建库")
            # 标记为已拟合，阻止 MemoryBankMixin 原始 hook 重复调用 fit()
            model._is_fitted = torch.tensor([True], device=model.device)

        # 不再调用 orig_epoch_end（我们自己已完成 fit）

    model.on_train_epoch_end = _safe_epoch_end.__get__(model, type(model))

    # 3. on_validation_start hook — 确保验证前 memory_bank 就绪
    orig_val_start = getattr(model, "on_validation_start", None)
    def _safe_val_start(self):
        if tm.memory_bank.numel() > 0:
            mb_log.debug("[安全网] validation: memory_bank 已就绪, 跳过")
            return

        ne = len(tm.embedding_store)
        if ne > 0:
            mb_log.info("[安全网] EMERGENCY fit: %d chunks", ne)
            tm.fit()
            model._is_fitted = torch.tensor([True], device=model.device)
        else:
            mb_log.error("[安全网] FATAL: 无特征且无 memory_bank!")

    model.on_validation_start = _safe_val_start.__get__(model, type(model))

    mb_log.info("✅ Memory Bank 安全网已安装 (3 hooks)")
    return model


# ================================================================
# 主流程
# ================================================================

_MODEL_DEFAULTS = {
    "anomaly_dino": {
        "encoder_name":       "dinov2_vit_small_14",
        "coreset_subsampling": True,
        "masking":             False,
        "image_size":          252,
        "max_epochs":          1,
    },
    "dino_patchcore": {
        "encoder_name":       "dinov2_vit_small_14",
        "masking":             False,
        "image_size":          252,
        "max_epochs":          1,
        "num_neighbors":       9,
        "coreset_sampling_ratio": 0.01,
    },
}


def main(args: argparse.Namespace | None = None) -> None:
    _add_src_to_path()

    from anomalib.callbacks import ModelCheckpoint
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.post_processing import PostProcessor
    from anomalib.utils.checkpoint_io import FileCheckpointIO
    from anomalib.visualization import ImageVisualizer

    args = _parse_args() if args is None else args

    # ---- 实验目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.exp_name or f"dino_{args.model}_{timestamp}"
    results_root = Path(r"results") / exp_name
    weights_dir = results_root / "weights" / "lightning"
    images_dir = results_root / "images"
    results_root.mkdir(parents=True, exist_ok=True)

    log = _StageLogger(results_root)
    log.header(f"DINO 实验: {exp_name}")
    log.info(f"结果目录 : {results_root.resolve()}")
    t_total = time.time()

    # ============================================================
    # 阶段 1: 数据集
    # ============================================================
    log.stage("1. 加载数据集")
    dataset_root = Path(args.dataset_root)
    datamodule = Folder(
        name="tubu_loujinshu",
        root=dataset_root,
        normal_dir=args.normal_dir,
        abnormal_dir=args.abnormal_dir,
        mask_dir=args.mask_dir,
        extensions=tuple(args.extensions),
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        test_split_mode="from_dir",
        test_split_ratio=args.test_split_ratio,
        val_split_mode="same_as_test",
        val_split_ratio=args.val_split_ratio,
        seed=args.seed,
    )
    datamodule.setup()
    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    log.info(f"  正常图 (train): {len(train_dl.dataset)} 张")
    log.info(f"  验证集   (val) : {len(val_dl.dataset)} 张")
    log.info(f"  测试集  (test) : {len(test_dl.dataset)} 张")
    log.info(f"  图像尺寸       : {args.image_size}x{args.image_size}")
    log.done()

    # ============================================================
    # 阶段 2: 构建模型
    # ============================================================
    log.stage("2. 构建模型")

    post_processor = PostProcessor(
        image_sensitivity=args.image_sensitivity,
        pixel_sensitivity=args.pixel_sensitivity,
    )

    log.info(f"  模型           : {args.model}")
    log.info(f"  encoder        : {args.encoder_name}")
    log.info(f"  image_size     : {args.image_size}")
    log.info(f"  masking        : {args.masking}")

    # ---- 加载 DINO 编码器 & 判断配置类型 ----
    dino_config = None
    feature_encoder = None

    if args.dino_weights:
        wp = Path(args.dino_weights)
        fname = wp.name.lower()
        dino_ver = "dinov3" if ("dinov3" in fname or fname.startswith("dinov3")) else "dinov2"
        dino_config = {"type": "dino", "path": str(wp), "version": dino_ver,
                       "encoder_name": args.encoder_name}
        log.info(f"  DINO 权重      : {wp.name}")
        log.info(f"  DINO 版本      : {dino_ver}")
    else:
        log.info(f"  DINO 权重      : 官方在线下载")

    # ---- 创建模型 ----
    test_images_dir = None if args.no_vis_test else images_dir

    if args.model == "anomaly_dino":
        model = create_anomaly_dino_model(args, dino_config, test_images_dir,
                                          post_processor, log)

    elif args.model == "dino_patchcore":
        model = _create_dino_patchcore(args, dino_config, test_images_dir,
                                       post_processor, log)

    else:
        log.error(f"❌ 不支持的模型: {args.model}")
        sys.exit(1)

    checkpoint_callback = ModelCheckpoint(
        dirpath=weights_dir,
        filename="model",
        auto_insert_metric_name=False,
        save_last=args.save_last,
        save_top_k=args.save_top_k,
        save_weights_only=args.save_weights_only,
    )

    engine = Engine(
        callbacks=[checkpoint_callback],
        default_root_dir=results_root,
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision="32",  # Memory Bank 方法无需 FP16
        plugins=[FileCheckpointIO()],
    )
    log.done()

    # ============================================================
    # 阶段 3a: 训练
    # ============================================================
    log.stage("3a. 训练 (特征提取)")
    try:
        engine.fit(model=model, datamodule=datamodule)
    except Exception as e:
        log.error(f"训练失败: {type(e).__name__}: {e}")
        import traceback
        log.error(traceback.format_exc())
        sys.exit(1)
    log.info("  ✅ 训练完成")
    log.done()

    # ============================================================
    # 阶段 3b: 测试
    # ============================================================
    log.stage("3b. 测试 (模型评估)")
    test_results = None
    if args.skip_test:
        log.info("  ⏭️ 跳过测试 (--skip-test)")
    else:
        try:
            test_results = engine.test(model=model, datamodule=datamodule)
        except torch.cuda.OutOfMemoryError:
            log.error("❌ CUDA OOM! 建议: --eval-batch-size 1 或 --skip-test")
        except Exception as e:
            log.error(f"测试失败: {type(e).__name__}: {e}")
            import traceback as tb
            log.error(tb.format_exc())
        else:
            log.done()

    if test_results:
        log.info("\n--- 测试指标 ---")
        for k, v in test_results[0].items():
            if isinstance(v, float):
                log.info(f"  {k}: {v:.6f}")
            else:
                log.info(f"  {k}: {v}")

    ckpt_path_str = engine.checkpoint_callback.last_model_path
    ckpt_path = Path(ckpt_path_str) if ckpt_path_str else None
    log.info(f"\nCheckpoint: {ckpt_path}")
    if ckpt_path and ckpt_path.is_file():
        log.info(f"  大小: {ckpt_path.stat().st_size / (1024*1024):.1f} MB")

    # ============================================================
    # 阶段 4: 采样可视化
    # ============================================================
    if ckpt_path and ckpt_path.is_file() and args.vis_samples > 0:
        log.stage("4. 采样预测可视化")
        _sample_and_predict(model, engine, ckpt_path, dataset_root,
                            args.abnormal_dir, args.extensions,
                            images_dir, log, seed=args.seed,
                            max_samples=args.vis_samples)
        log.done()
    elif args.vis_samples == 0:
        log.info("\n⏭️ 跳过采样预测")

    # ============================================================
    # 保存配置
    # ============================================================
    config_fields = [
        "model", "encoder_name", "dino_weights", "image_size",
        "masking", "coreset_subsampling", "coreset_sampling_ratio",
        "num_neighbors", "image_sensitivity", "pixel_sensitivity",
        "no_vis_test", "vis_samples",
    ]
    train_config = {f: getattr(args, f, None) for f in config_fields}
    config_path = results_root / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)
    log.info(f"  配置文件: {config_path}")

    # ============================================================
    # 总结
    # ============================================================
    total_elapsed = time.time() - t_total
    log.header(f"完成! 总耗时 {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    log.info(f"  结果目录: {results_root.resolve()}")
    log.info(f"  权重文件: {ckpt_path}")
    log.info(f"  日志文件: {results_root / 'train.log'}")


def _create_dino_patchcore(args, dino_config: dict | None,
                            images_dir: Path | None,
                            post_processor, log: _StageLogger):
    """创建 DINO+PatchCore 模型实例。"""
    from anomalib.visualization.image.visualizer import ImageVisualizer

    # 加载编码器
    wp = Path(dino_config["path"]) if dino_config else None
    dino_ver = dino_config.get("version") if dino_config else None
    feature_encoder = load_dino_encoder(args.encoder_name, wp, dino_ver, log)
    log.info(f"  num_neighbors       : {args.num_neighbors}")
    log.info(f"  coreset_sampling    : {args.coreset_sampling_ratio}")

    vis_kwarg = False if images_dir is None else ImageVisualizer(output_dir=images_dir)

    model = DinoPatchcore(
        feature_encoder=feature_encoder,
        num_neighbors=args.num_neighbors,
        coreset_sampling_ratio=args.coreset_sampling_ratio,
        visualizer=vis_kwarg,
        post_processor=post_processor,
    )

    # 设置输入尺寸
    input_size = (args.image_size, args.image_size)
    model.configure_pre_processor(input_size)

    log.info(f"  ✅ DinoPatchcore 创建成功!")
    return model


def _sample_and_predict(model, engine, ckpt_path: Path, dataset_root: Path,
                        abnormal_dir: str, extensions: list[str],
                        images_dir: Path, log: _StageLogger, seed: int = 42,
                        sample_ratio: float = 0.1, max_samples: int = 100) -> None:
    """训练后采样预测可视化。"""
    rng = random.Random(seed)
    abnormal_dir_path = dataset_root / abnormal_dir
    if not abnormal_dir_path.is_dir():
        log.warning(f"异常图目录不存在: {abnormal_dir_path}"); return

    ext_set = tuple(set(extensions))
    all_images = sorted([p for p in abnormal_dir_path.rglob("*")
                         if p.is_file() and p.suffix.lower() in ext_set])
    total = len(all_images)
    if total == 0:
        log.warning(f"无图片: {abnormal_dir_path}"); return

    # 过滤非法字符文件名
    _printable = set(string.printable)
    clean_images = []
    for p in all_images:
        non_print = [c for c in p.name if c not in _printable]
        if non_print:
            continue
        clean_images.append(p)
    all_images = clean_images; total = len(all_images)
    if total == 0: return

    n_samples = min(max(int(total * sample_ratio), 1), max_samples)
    sampled = rng.sample(all_images, n_samples)
    log.info(f"  抽样: {n_samples}/{total}")

    from anomalib.visualization.image.visualizer import ImageVisualizer
    _orig_visualizer = getattr(model, "visualizer", None)
    _vis_dir = images_dir / "samples"
    model.visualizer = ImageVisualizer(output_dir=_vis_dir)

    success, fail = 0, 0
    t0 = time.time()
    for i, img_path in enumerate(sampled, 1):
        try:
            engine.predict(model=model, data_path=str(img_path),
                           return_predictions=False)
            success += 1
            if i % 20 == 0 or i == n_samples:
                log.info(f"  进度: {i}/{n_samples} ({success} 成功, {fail} 失败)")
        except Exception as e:
            fail += 1
            log.warning(f"  [{i}/{n_samples}] 失败 {img_path.name}: {e}")

    model.visualizer = _orig_visualizer
    log.info(f"  完成: {success} 成功, {fail} 失败, 耗时 {time.time()-t0:.1f}s")


def _apply_defaults(args: argparse.Namespace) -> None:
    """根据 --model 自动调整默认值。"""
    cfg = _MODEL_DEFAULTS.get(args.model, {})
    if args.encoder_name == "dinov2_vit_small_14" and "encoder_name" in cfg:
        args.encoder_name = cfg["encoder_name"]
    if args.image_size == 256 and "image_size" in cfg:
        args.image_size = cfg["image_size"]
    if not args.masking and "masking" in cfg:
        args.masking = cfg["masking"]
    if args.max_epochs == 1 and "max_epochs" in cfg:
        args.max_epochs = cfg["max_epochs"]
    # DinoPatchcore 特有
    if args.model == "dino_patchcore":
        if "num_neighbors" in cfg:
            args.num_neighbors = cfg["num_neighbors"]
        if "coreset_sampling_ratio" in cfg:
            args.coreset_sampling_ratio = cfg["coreset_sampling_ratio"]
    # AnomalyDINO 特有
    if args.model == "anomaly_dino":
        if "coreset_subsampling" in cfg:
            args.coreset_subsampling = cfg["coreset_subsampling"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DINO (v2/v3) 异常检测训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # DINOv2 + AnomalyDINO (在线权重)
  python train/train_dino.py --encoder-name dinov2_vit_small_14

  # DINOv2 + AnomalyDINO (本地权重 + masking)
  python train/train_dino.py --dino-weights weights/dino/dinov2_vits14.pth --masking

  # DINOv3 + AnomalyDINO
  python train/train_dino.py --dino-weights path/to/dinov3_vitb16.pth --encoder-name dinov3_vitb16

  # DINO + PatchCore
  python train/train_dino.py --model dino_patchcore --dino-weights path/to/dinov2_vits14.pth --num-neighbors 5
""")

    # ---- 实验 ----
    parser.add_argument("--exp-name", type=str, default=None,
                        help="实验名称 (默认: dino_<模型>_<时间戳>)")

    # ---- 模型选择 ----
    parser.add_argument("--model", type=str, default="anomaly_dino",
                        choices=["anomaly_dino", "dino_patchcore"],
                        help="模型: anomaly_dino(原始算法) / dino_patchcore(DINO特征+PatchCore KNN)")

    # ---- 数据集 ----
    parser.add_argument("--dataset-root", type=str,
                        default=r"Z:\14-调试数据\lxm\Dataset\Anomalib\TB")
    parser.add_argument("--normal-dir", type=str, default="OK_V2")
    parser.add_argument("--abnormal-dir", type=str, default="defects")
    parser.add_argument("--mask-dir", type=str, default="masks")
    parser.add_argument("--extensions", type=str, nargs="+",
                        default=[".bmp", ".jpg", ".png"])

    # ---- 数据加载 ----
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16,
                        help="评估批次大小 (15GB显存建议≤8)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split-ratio", type=float, default=0.2)
    parser.add_argument("--val-split-ratio", type=float, default=0.5)

    # ---- DINO 参数 ----
    parser.add_argument("--encoder-name", type=str, default="dinov2_vit_small_14",
                        help="DINO 编码器名称\n"
                             "  DINOv2: dinov2_vit_small_14 / dinov2_vit_base_14 / dinov2_vit_large_14\n"
                             "  DINOv3: dinov3_vits16 / dinov3_vitb16 / dinov3_vits16plus 等")
    parser.add_argument("--dino-weights", type=str, default=None,
                        help="DINO 预训练权重 .pth 路径。\n"
                             "  不指定则使用官方在线下载 (仅限 DINOv2)")
    parser.add_argument("--image-size", type=int, default=252,
                        help="输入图像尺寸 (需被 patch_size 整除)\n"
                             "  p14 推荐: 224/238/252/266\n"
                             "  p16 推荐: 224/240/256")
    parser.add_argument("--masking", action=argparse.BooleanOptionalAction, default=False,
                        help="[AnomalyDINO] 是否启用 PCA 掩码抑制背景特征")
    parser.add_argument("--coreset-subsampling", action=argparse.BooleanOptionalAction,
                        default=True, help="[AnomalyDINO] coreset 降采样")
    parser.add_argument("--coreset-sampling-ratio", type=float, default=0.1,
                        help="[AnomalyDINO/DinoPatchcore] coreset 采样率 (越小越省显存)")

    # ---- DinoPatchcore 参数 ----
    parser.add_argument("--num-neighbors", type=int, default=9,
                        help="[DinoPatchcore] KNN 邻居数量")

    # ---- 阈值 ----
    parser.add_argument("--image-sensitivity", type=float, default=0.5,
                        help="图像级灵敏度 (0~1)")
    parser.add_argument("--pixel-sensitivity", type=float, default=0.7,
                        help="像素级灵敏度 (0~1)")

    # ---- 训练 ----
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--save-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-top-k", type=int, default=0)
    parser.add_argument("--save-weights-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-test", action="store_true", help="跳过测试")
    parser.add_argument("--vis-samples", type=int, default=100,
                        help="训练后采样可视化数量 (0=跳过)")
    parser.add_argument("--no-vis-test", action="store_true",
                        help="测试时不输出图片 (只算指标)")

    args = parser.parse_args()
    _apply_defaults(args)
    return args


if __name__ == "__main__":
    main()
