from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from boundarybrush import prompts
from boundarybrush.checkpoint import build_model, save_checkpoint
from boundarybrush.config import Settings
from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import Prompt
from boundarybrush.providers.minisam_provider import MiniSAMSegmenter
from boundarybrush.providers.mock import MockSegmenter
from boundarybrush.providers.registry import get_segmenter
from boundarybrush.providers.slimsam_provider import SlimSAMSegmenter
from boundarybrush.providers.unet_provider import UNetSegmenter

IMG = np.zeros((64, 96, 3), dtype=np.uint8)  # H=64, W=96; providers work at 16x16 below


class _StubUNet(torch.nn.Module):
    """Two blobs at 16x16: rows 2-5 x cols 2-5 and rows 9-13 x cols 9-13."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        logits = torch.full((1, 1, 16, 16), -5.0)
        logits[..., 2:6, 2:6] = 5.0
        logits[..., 9:14, 9:14] = 3.0
        return logits


def _unet():
    seg = UNetSegmenter(_StubUNet(), input_size=16)
    seg.set_image(IMG)
    return seg


def _at16(mask):
    """Downsample an original-size mask back to the 16x16 grid by sampling cell centres."""
    h, w = mask.shape
    ys = ((np.arange(16) + 0.5) * h / 16).astype(int)
    xs = ((np.arange(16) + 0.5) * w / 16).astype(int)
    return mask[np.ix_(ys, xs)]


def test_predict_before_set_image_raises():
    with pytest.raises(RuntimeError):
        MockSegmenter().predict(Prompt(points=[(1, 1)], labels=[1]))


def test_set_image_validates_input():
    with pytest.raises(ValueError):
        MockSegmenter().set_image(np.zeros((8, 8), dtype=np.uint8))


def test_mock_segmenter_click_and_box():
    seg = MockSegmenter()
    seg.set_image(IMG)
    r = seg.predict(Prompt(points=[(48, 32)], labels=[1]))
    assert r.mask.shape == (64, 96) and r.mask[32, 48] and not r.mask[0, 0]
    r = seg.predict(Prompt(box=(0, 0, 10, 10)))
    assert r.mask[:10, :10].all() and r.mask.sum() == 100


def test_unet_empty_prompt_returns_everything_at_original_size():
    r = _unet().predict(Prompt())
    assert r.mask.shape == (64, 96)
    small = _at16(r.mask)
    assert small[3, 3] and small[11, 11] and not small[0, 0]


def test_unet_click_selects_component_and_negative_removes_it():
    seg = _unet()
    # cell (row 11, col 11) of 16 -> original (x = 11.5 * 96/16, y = 11.5 * 64/16)
    click = (11.5 * 6, 11.5 * 4)
    small = _at16(seg.predict(Prompt(points=[click], labels=[1])).mask)
    assert small[11, 11] and not small[3, 3]
    small = _at16(seg.predict(Prompt(points=[click], labels=[0])).mask)
    assert small[3, 3] and not small[11, 11]


def test_unet_background_click_is_empty_with_zero_score():
    r = _unet().predict(Prompt(points=[(1, 1)], labels=[1]))
    assert not r.mask.any() and r.score == 0.0


def test_unet_box_intersects():
    small = _at16(_unet().predict(Prompt(box=(0, 0, 48, 32))).mask)  # top-left quarter of the image
    assert small[3, 3] and not small[11, 11]


def test_unet_embeds_once():
    seg = _unet()
    for _ in range(3):
        seg.predict(Prompt())
    assert seg.model.calls == 1


class _StubMiniSAM(torch.nn.Module):
    def __init__(self, n_masks=2):
        super().__init__()
        self.encode_calls, self.decoded = 0, []
        self.n_masks = n_masks

    def encode_image(self, x):
        self.encode_calls += 1
        return {"x": x}

    def decode(self, feats, coords, types):
        self.decoded.append((coords.clone(), types.clone()))
        masks = torch.full((1, self.n_masks, 16, 16), -5.0)
        masks[0, 1, :8] = 5.0  # head 1 = top half
        return masks, torch.tensor([[0.2, 0.9]])


def test_minisam_provider_normalizes_coords_embeds_once_and_picks_best_head():
    stub = _StubMiniSAM()
    seg = MiniSAMSegmenter(stub, input_size=16)
    seg.set_image(IMG)
    for _ in range(3):
        r = seg.predict(Prompt(points=[(48, 16)], labels=[1]))
    assert stub.encode_calls == 1
    coords, types = stub.decoded[-1]
    assert torch.allclose(coords[0, 0], torch.tensor([0.5, 0.25]))
    assert types[0, 0] == prompts.POS and (types[0, 1:] == prompts.PAD).all()
    assert r.score == pytest.approx(0.9)
    assert r.mask[:30].all() and not r.mask[34:].any()


def test_minisam_provider_rejects_empty_prompt():
    seg = MiniSAMSegmenter(_StubMiniSAM(), input_size=16)
    seg.set_image(IMG)
    with pytest.raises(ValueError):
        seg.predict(Prompt())


class _StubImageProcessor:
    size = {"longest_edge": 1024}

    def __init__(self):
        self.post_calls = []

    def post_process_masks(self, masks, original_sizes, reshaped):
        self.post_calls.append((masks.shape, original_sizes.tolist()))
        h, w = original_sizes[0].tolist()
        n = masks.shape[2]
        out = torch.zeros(1, n, h, w, dtype=torch.bool)
        out[0, n - 1, : h // 2] = True
        return [out]


class _StubProcessor:
    def __init__(self):
        self.image_processor = _StubImageProcessor()
        self.calls = 0

    def __call__(self, images, return_tensors):
        self.calls += 1
        h, w = images.shape[:2]
        return {
            "pixel_values": torch.zeros(1, 3, 8, 8),
            "original_sizes": torch.tensor([[h, w]]),
            "reshaped_input_sizes": torch.tensor([[round(h * 1024 / max(h, w)), 1024]]),
        }


class _StubSamModel:
    def __init__(self):
        self.embed_calls, self.calls = 0, []

    def get_image_embeddings(self, pixel_values):
        self.embed_calls += 1
        return torch.zeros(1, 4, 2, 2)

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        n = 3 if kwargs["multimask_output"] else 1
        scores = torch.linspace(0.1, 0.9, n)[None, None]
        return SimpleNamespace(pred_masks=torch.zeros(1, 1, n, 4, 4), iou_scores=scores)


def _slimsam():
    model, proc = _StubSamModel(), _StubProcessor()
    seg = SlimSAMSegmenter(model, proc)
    seg.set_image(IMG)
    return seg, model, proc


def test_slimsam_embeds_once_and_scales_coords():
    seg, model, proc = _slimsam()
    for _ in range(3):
        seg.predict(Prompt(points=[(48, 32)], labels=[1]))
    assert model.embed_calls == 1 and proc.calls == 1
    kw = model.calls[-1]
    scale = 1024 / 96
    assert torch.allclose(kw["input_points"], torch.tensor([[[[48 * scale, 32 * scale]]]]))
    assert kw["input_labels"].tolist() == [[[1]]]
    assert kw["multimask_output"] is True and "input_boxes" not in kw


def test_slimsam_multimask_only_for_a_single_click():
    seg, model, _ = _slimsam()
    seg.predict(Prompt(points=[(10, 10), (20, 20)], labels=[1, 0]))
    assert model.calls[-1]["multimask_output"] is False
    seg.predict(Prompt(points=[(10, 10)], labels=[1], box=(0, 0, 50, 40)))
    assert model.calls[-1]["multimask_output"] is False
    assert model.calls[-1]["input_boxes"].shape == (1, 1, 4)


def test_slimsam_returns_best_mask_at_original_size():
    seg, _, _ = _slimsam()
    r = seg.predict(Prompt(points=[(48, 32)], labels=[1]))
    assert r.mask.shape == (64, 96) and r.mask[:32].all() and not r.mask[32:].any()
    assert r.score == pytest.approx(0.9)


def test_registry_mock_and_missing_weights(tmp_path):
    settings = Settings(weights_dir=tmp_path)
    assert get_segmenter("mock", settings=settings).backend is Backend.MOCK
    with pytest.raises(FileNotFoundError):
        get_segmenter(Backend.UNET, settings=settings)


def test_registry_loads_checkpoint_and_checks_kind(tmp_path):
    path = save_checkpoint(tmp_path / "unet.pt", kind="unet", model=build_model("unet", {"base_channels": 4}))
    seg = get_segmenter("unet", settings=Settings(weights_dir=tmp_path))
    assert isinstance(seg, UNetSegmenter)
    with pytest.raises(ValueError):
        get_segmenter("minisam", weights=path, settings=Settings(weights_dir=tmp_path))
