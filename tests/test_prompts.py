from __future__ import annotations

import torch

from boundarybrush import prompts
from boundarybrush.domain.models import Prompt


def _target(size=32):
    m = torch.zeros(size, size, dtype=torch.bool)
    m[8:24, 10:26] = True
    return m


def _pixels(coords, size):
    cols = (coords[:, 0] * size).floor().long()
    rows = (coords[:, 1] * size).floor().long()
    return rows, cols


def test_sampled_prompt_invariants():
    target = _target()
    for seed in range(200):
        gen = torch.Generator().manual_seed(seed)
        coords, types = prompts.sample_prompt(target, gen)
        assert coords.shape == (prompts.MAX_TOKENS, 2) and types.shape == (prompts.MAX_TOKENS,)
        assert ((coords >= 0) & (coords <= 1)).all()
        real = types != prompts.PAD
        assert real.any()
        # padding only at the end
        assert not (real[1:] & ~real[:-1]).any()
        rows, cols = _pixels(coords.clamp(max=1 - 1e-6), 32)
        pos, neg = types == prompts.POS, types == prompts.NEG
        assert target[rows[pos], cols[pos]].all()
        assert not target[rows[neg], cols[neg]].any()
        if (types == prompts.BOX_TL).any():
            tl = coords[types == prompts.BOX_TL][0]
            br = coords[types == prompts.BOX_BR][0]
            assert (tl < br).all()


def test_first_click_is_positive_when_no_box():
    target = _target()
    for seed in range(50):
        coords, types = prompts.sample_prompt(target, torch.Generator().manual_seed(seed))
        if types[0] in (prompts.POS, prompts.NEG):
            assert types[0] == prompts.POS


def test_empty_target_gives_only_negative_or_padding():
    coords, types = prompts.sample_prompt(torch.zeros(16, 16, dtype=torch.bool), torch.Generator())
    assert not (types == prompts.POS).any()


def test_encode_prompt_normalizes_and_orders():
    p = Prompt(points=[(16, 8), (4, 4)], labels=[1, 0], box=(0, 0, 32, 16))
    coords, types = prompts.encode_prompt(p, width=32, height=16)
    assert types[:4].tolist() == [prompts.POS, prompts.NEG, prompts.BOX_TL, prompts.BOX_BR]
    assert (types[4:] == prompts.PAD).all()
    assert torch.allclose(coords[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(coords[3], torch.tensor([1.0, 1.0]))


def test_encode_prompt_keeps_latest_clicks():
    pts = [(float(i), 1.0) for i in range(prompts.MAX_CLICKS + 3)]
    p = Prompt(points=pts, labels=[1] * len(pts))
    coords, types = prompts.encode_prompt(p, width=64, height=64)
    assert (types == prompts.POS).sum() == prompts.MAX_CLICKS
    assert torch.isclose(coords[prompts.MAX_CLICKS - 1, 0], torch.tensor((len(pts) - 1) / 64))
