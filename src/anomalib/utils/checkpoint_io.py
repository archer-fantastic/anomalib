from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO


class FileCheckpointIO(CheckpointIO):
    def save_checkpoint(self, checkpoint: dict[str, Any], path: str | Path, storage_options: dict[str, Any] | None = None) -> None:
        del storage_options
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        try:
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def load_checkpoint(self, path: str | Path, map_location: Any = None,
                         weights_only: bool | None = None) -> dict[str, Any]:
        del weights_only  # anomalib checkpoint 包含模型+优化器状态，不能仅加载权重
        path = Path(path)
        return torch.load(path, map_location=map_location, weights_only=False)

    def remove_checkpoint(self, path: str | Path) -> None:
        Path(path).unlink(missing_ok=True)
