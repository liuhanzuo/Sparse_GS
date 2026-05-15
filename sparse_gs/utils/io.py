"""Image and checkpoint IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def save_image(img: torch.Tensor | np.ndarray, path: str | Path) -> None:
    """Save a HxWx3 (or HxWx4) image in [0,1] to ``path`` (PNG/JPG)."""
    import imageio.v2 as imageio  # lazy

    if isinstance(img, torch.Tensor):
        img = img.detach().clamp(0, 1).cpu().numpy()
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = (img.clip(0, 1) * 255.0 + 0.5).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), img)


def load_image_rgba(path: str | Path) -> np.ndarray:
    """Load an image as float32 HxWx{3,4} in [0,1]."""
    import imageio.v2 as imageio

    arr = imageio.imread(str(path))
    arr = np.asarray(arr).astype(np.float32) / 255.0
    return arr


def save_checkpoint(state: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location, weights_only=False)
