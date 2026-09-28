"""Mask morphology and image helpers shared by data prep, providers and evaluation.

Morphology works on bool tensors shaped (H, W) or (B, H, W) with max-pooling, so no scipy.
Erosion pads with zeros: the image border counts as object boundary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def _as_4d(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    ndim = mask.ndim
    x = mask.float()
    x = x[None, None] if ndim == 2 else x[:, None]
    return x, ndim


def _restore(x: torch.Tensor, ndim: int) -> torch.Tensor:
    x = x[0, 0] if ndim == 2 else x[:, 0]
    return x > 0.5


def erode(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    x, ndim = _as_4d(mask)
    k = 2 * radius + 1
    x = F.pad(x, (radius,) * 4, value=0.0)
    return _restore(-F.max_pool2d(-x, k, stride=1), ndim)


def dilate(mask: torch.Tensor, radius: int = 1) -> torch.Tensor:
    x, ndim = _as_4d(mask)
    k = 2 * radius + 1
    return _restore(F.max_pool2d(x, k, stride=1, padding=radius), ndim)


def depth_map(mask: torch.Tensor) -> torch.Tensor:
    """Per-pixel count of 3x3 erosions survived (0 outside the mask)."""
    depth = torch.zeros(mask.shape, dtype=torch.int32)
    current = mask.bool()
    while current.any():
        depth += current.int()
        current = erode(current, 1)
    return depth


def deepest_point(mask: torch.Tensor) -> tuple[float, float] | None:
    """Pixel-centred (x, y) of the point furthest from the boundary; first in row-major order on ties."""
    if not mask.any():
        return None
    depth = depth_map(mask)
    idx = int(torch.argmax(depth.flatten()))
    row, col = divmod(idx, mask.shape[-1])
    return (col + 0.5, row + 0.5)


def component_at(mask: torch.Tensor, row: int, col: int) -> torch.Tensor:
    """8-connected component of `mask` containing (row, col); empty if that pixel is background."""
    mask = mask.bool()
    comp = torch.zeros_like(mask)
    if not mask[row, col]:
        return comp
    comp[row, col] = True
    while True:
        grown = dilate(comp, 1) & mask
        if torch.equal(grown, comp):
            return comp
        comp = grown


def resize_rgb(image: np.ndarray, size: int) -> np.ndarray:
    """Squash an RGB uint8 image to size x size (bilinear). The one resize used everywhere."""
    return np.asarray(Image.fromarray(image).convert("RGB").resize((size, size), Image.BILINEAR))


def resize_labels(labels: np.ndarray, size: int) -> np.ndarray:
    """Squash a label map (trimap or mask) to size x size with nearest neighbour."""
    return np.asarray(Image.fromarray(labels).resize((size, size), Image.NEAREST))


def to_model_input(image: np.ndarray) -> torch.Tensor:
    """uint8 H x W x 3 -> normalized float32 3 x H x W."""
    x = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    return (x - mean) / std


def logits_to_mask(logits: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
    """Upsample (h, w) logits bilinearly to out_hw, then threshold at 0 -> bool numpy mask."""
    up = F.interpolate(logits[None, None].float(), size=out_hw, mode="bilinear", align_corners=False)
    return (up[0, 0] > 0).numpy()
