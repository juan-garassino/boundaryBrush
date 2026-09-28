"""Prompt tensors for the mini-SAM: training-time sampling and inference-time encoding.

A prompt is MAX_TOKENS rows of normalized (x, y) in [0, 1] plus a token type. Clicks come
first, then the two box corners, then explicit PAD tokens (masked out of attention).
"""

from __future__ import annotations

import logging

import torch

from boundarybrush.domain.models import POSITIVE, Prompt
from boundarybrush.ops import dilate, erode

logger = logging.getLogger(__name__)

PAD, POS, NEG, BOX_TL, BOX_BR = 0, 1, 2, 3, 4
NUM_TYPES = 5
MAX_CLICKS = 8
MAX_TOKENS = MAX_CLICKS + 2


def _pad(tokens: list[tuple[float, float, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = tokens[:MAX_TOKENS]
    coords = torch.zeros(MAX_TOKENS, 2)
    types = torch.full((MAX_TOKENS,), PAD, dtype=torch.long)
    for i, (x, y, t) in enumerate(tokens):
        coords[i] = torch.tensor([x, y])
        types[i] = t
    return coords, types


def _sample_pixel(mask: torch.Tensor, gen: torch.Generator) -> tuple[float, float] | None:
    idx = mask.nonzero()
    if len(idx) == 0:
        return None
    r, c = idx[int(torch.randint(0, len(idx), (1,), generator=gen))].tolist()
    h, w = mask.shape
    return (c + 0.5) / w, (r + 0.5) / h


def _rand(gen: torch.Generator) -> float:
    return float(torch.rand(1, generator=gen))


def _clicks(target: torch.Tensor, n: int, gen: torch.Generator, ring_radius: int):
    inner = erode(target, 2)
    ring = dilate(target, ring_radius) & ~target
    tokens = []
    first = _sample_pixel(inner if inner.any() else target, gen)
    tokens.append((*first, POS))
    for _ in range(n - 1):
        if _rand(gen) < 0.5:
            xy, t = _sample_pixel(target, gen), POS
        else:
            pool = ring if (_rand(gen) < 0.5 and ring.any()) else ~target
            xy, t = _sample_pixel(pool, gen), NEG
        if xy is not None:
            tokens.append((*xy, t))
    return tokens


def _box(target: torch.Tensor, gen: torch.Generator, jitter: float):
    h, w = target.shape
    rows, cols = target.nonzero(as_tuple=True)
    x0, x1 = cols.min().item(), cols.max().item() + 1
    y0, y1 = rows.min().item(), rows.max().item() + 1
    bw, bh = x1 - x0, y1 - y0
    j = [(_rand(gen) * 2 - 1) * jitter for _ in range(4)]
    jx0, jy0 = min(max(x0 + j[0] * bw, 0), w), min(max(y0 + j[1] * bh, 0), h)
    jx1, jy1 = min(max(x1 + j[2] * bw, 0), w), min(max(y1 + j[3] * bh, 0), h)
    if jx0 < jx1 and jy0 < jy1:
        x0, y0, x1, y1 = jx0, jy0, jx1, jy1
    return [(x0 / w, y0 / h, BOX_TL), (x1 / w, y1 / h, BOX_BR)]


def sample_prompt(
    target: torch.Tensor,
    gen: torch.Generator,
    p_box: float = 0.35,
    p_box_clicks: float = 0.15,
    max_train_clicks: int = 6,
    ring_radius: int = 3,
    jitter: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a prompt for `target` (bool H x W): clicks only, a jittered box, or box + clicks.

    The first click is always positive; later clicks are positive half the time, and
    negatives come half from a ring hugging the object, like corrective clicks at eval.
    """
    target = target.bool()
    if not target.any():
        xy = _sample_pixel(~target, gen)
        return _pad([(*xy, NEG)])
    r = _rand(gen)
    if r < p_box:
        return _pad(_box(target, gen, jitter))
    if r < p_box + p_box_clicks:
        n = int(torch.randint(1, 4, (1,), generator=gen))
        return _pad(_clicks(target, n, gen, ring_radius) + _box(target, gen, jitter))
    n = int(torch.randint(1, max_train_clicks + 1, (1,), generator=gen))
    return _pad(_clicks(target, n, gen, ring_radius))


def encode_prompt(prompt: Prompt, width: int, height: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Prompt in pixel coords of a width x height image -> (coords, types) tensors."""
    points, labels = list(prompt.points), list(prompt.labels)
    if len(points) > MAX_CLICKS:
        logger.warning("keeping the latest %d of %d clicks", MAX_CLICKS, len(points))
        points, labels = points[-MAX_CLICKS:], labels[-MAX_CLICKS:]
    tokens = [
        (x / width, y / height, POS if lab == POSITIVE else NEG)
        for (x, y), lab in zip(points, labels, strict=True)
    ]
    if prompt.box is not None:
        x0, y0, x1, y1 = prompt.box
        tokens += [(x0 / width, y0 / height, BOX_TL), (x1 / width, y1 / height, BOX_BR)]
    return _pad(tokens)
