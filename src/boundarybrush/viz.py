"""Overlays, the README gallery figure and the interactive click app (matplotlib)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from boundarybrush.domain.models import POSITIVE, Prompt

logger = logging.getLogger(__name__)

MASK_COLOR = np.array([255, 64, 64], dtype=np.float32)


def overlay(
    image: np.ndarray, mask: np.ndarray, prompt: Prompt | None = None, alpha: float = 0.5
) -> np.ndarray:
    """Tint the mask red, outline its boundary, draw clicks (green +, red -) and the box."""
    out = image.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * MASK_COLOR
    edge = mask & ~_erode(mask)
    out[edge] = (255, 255, 255)
    if prompt is not None:
        h, w = mask.shape
        r = max(2, min(h, w) // 60)
        yy, xx = np.mgrid[0:h, 0:w]
        for (x, y), label in zip(prompt.points, prompt.labels, strict=True):
            disc = (xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2 <= r**2
            out[disc] = (40, 220, 90) if label == POSITIVE else (230, 40, 40)
        if prompt.box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in prompt.box)
            x1, y1 = min(x1, w) - 1, min(y1, h) - 1
            out[y0, x0 : x1 + 1] = out[y1, x0 : x1 + 1] = (60, 140, 255)
            out[y0 : y1 + 1, x0] = out[y0 : y1 + 1, x1] = (60, 140, 255)
    return out.clip(0, 255).astype(np.uint8)


def _erode(mask: np.ndarray) -> np.ndarray:
    m = np.pad(mask, 1, constant_values=False)
    return m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]


def load_image(path: Path | str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_png(image: np.ndarray, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
    return path


def gallery(rows: list[dict], path: Path | str, title: str | None = None) -> Path:
    """rows: [{"image": uint8 HxWx3, "panels": {name: (mask, prompt|None)}}] -> PNG grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(rows[0]["panels"])
    fig, axes = plt.subplots(len(rows), len(names) + 1, figsize=(2.2 * (len(names) + 1), 2.2 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        axes[r, 0].imshow(row["image"])
        for c, name in enumerate(names, start=1):
            mask, prompt = row["panels"][name]
            axes[r, c].imshow(overlay(row["image"], mask, prompt))
            if r == 0:
                axes[r, c].set_title(name, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("image", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def click_app(image: np.ndarray, segmenter, out_dir: Path | str = "out") -> None:  # pragma: no cover - GUI
    """Left click = include, right click (or `n` then click) = exclude, r = reset, s = save, q = quit."""
    import matplotlib.pyplot as plt

    segmenter.set_image(image)
    state = {"points": [], "labels": [], "negative": False, "mask": np.zeros(image.shape[:2], bool)}
    fig, ax = plt.subplots(figsize=(8, 8 * image.shape[0] / image.shape[1]))
    shown = ax.imshow(image)
    ax.axis("off")

    def redraw():
        prompt = Prompt(state["points"], state["labels"]) if state["points"] else None
        shown.set_data(overlay(image, state["mask"], prompt))
        mode = "exclude" if state["negative"] else "include"
        ax.set_title(
            f"{segmenter.backend.value} | clicks: {len(state['points'])} | next click: {mode}", fontsize=10
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        label = 0 if (event.button == 3 or state["negative"]) else 1
        state["points"].append((float(event.xdata) + 0.5, float(event.ydata) + 0.5))
        state["labels"].append(label)
        result = segmenter.predict(Prompt(state["points"], state["labels"]))
        state["mask"] = result.mask
        logger.info("score %.3f, %d px", result.score, int(result.mask.sum()))
        redraw()

    def on_key(event):
        if event.key == "n":
            state["negative"] = not state["negative"]
        elif event.key == "r":
            state.update(points=[], labels=[], mask=np.zeros(image.shape[:2], bool))
        elif event.key == "s":
            out = Path(out_dir)
            save_png(state["mask"].astype(np.uint8) * 255, out / "mask.png")
            save_png(overlay(image, state["mask"]), out / "overlay.png")
            logger.info("saved to %s", out)
        elif event.key == "q":
            plt.close(fig)
            return
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()
