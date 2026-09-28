from __future__ import annotations

import pytest
import torch

from boundarybrush import prompts
from boundarybrush.models.losses import hard_iou, minisam_loss, weighted_bce_dice
from boundarybrush.models.minisam import MiniSAM, MiniSAMConfig
from boundarybrush.models.unet import UNet, UNetConfig

TINY = MiniSAMConfig(embed_dim=32, n_heads=2, stem_channels=8, skip_channels=8)


def _click(x, y, t=prompts.POS):
    coords = torch.zeros(1, prompts.MAX_TOKENS, 2)
    types = torch.full((1, prompts.MAX_TOKENS), prompts.PAD, dtype=torch.long)
    coords[0, 0] = torch.tensor([x, y])
    types[0, 0] = t
    return coords, types


def test_unet_shapes_and_size_check():
    net = UNet(UNetConfig(base_channels=8)).eval()
    assert net(torch.randn(2, 3, 32, 32)).shape == (2, 1, 32, 32)
    with pytest.raises(ValueError):
        net(torch.randn(1, 3, 30, 30))


def test_unet_default_size_is_small():
    assert sum(p.numel() for p in UNet().parameters()) < 3_000_000


def test_minisam_shapes_single_and_multi_prompt():
    net = MiniSAM(TINY).eval()
    img = torch.randn(2, 3, 32, 32)
    coords, types = _click(0.5, 0.5)
    masks, iou = net(img, coords.expand(2, -1, -1), types.expand(2, -1))
    assert masks.shape == (2, 1, 32, 32) and iou.shape == (2, 1)
    k_coords = coords[None].expand(2, 3, -1, -1)
    k_types = types[None].expand(2, 3, -1)
    masks, iou = net(img, k_coords, k_types)
    assert masks.shape == (2, 3, 1, 32, 32) and iou.shape == (2, 3, 1)


def test_minisam_pad_tokens_do_not_change_output():
    torch.manual_seed(0)
    net = MiniSAM(TINY).eval()
    img = torch.randn(1, 3, 32, 32)
    coords, types = _click(0.3, 0.6)
    noisy = coords.clone()
    noisy[0, 1:] = torch.rand(prompts.MAX_TOKENS - 1, 2)  # garbage coords in PAD slots
    with torch.no_grad():
        a, ia = net(img, coords, types)
        b, ib = net(img, noisy, types)
    assert torch.allclose(a, b, atol=1e-5) and torch.allclose(ia, ib, atol=1e-5)


def test_minisam_moving_the_click_changes_output():
    torch.manual_seed(0)
    net = MiniSAM(TINY).eval()
    img = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, _ = net(img, *_click(0.2, 0.2))
        b, _ = net(img, *_click(0.8, 0.8))
    assert not torch.allclose(a, b)


def test_losses_are_finite_and_per_sample():
    logits = torch.randn(3, 8, 8, requires_grad=True)
    target = (torch.rand(3, 8, 8) > 0.5).float()
    loss = weighted_bce_dice(logits, target, torch.ones_like(target))
    assert loss.shape == (3,) and torch.isfinite(loss).all()
    total, stats = minisam_loss(logits[:, None], torch.zeros(3, 1), target, torch.ones_like(target))
    total.backward()
    assert logits.grad is not None and set(stats) == {"mask_loss", "iou_loss", "iou"}


def _discs(n=8, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    imgs, masks = [], []
    for _ in range(n):
        cx, cy = torch.randint(10, size - 10, (2,), generator=g).tolist()
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) < 36
        img = torch.randn(3, size, size, generator=g) * 0.2
        img[:, m] += 1.5
        imgs.append(img)
        masks.append(m.float())
    return torch.stack(imgs), torch.stack(masks)


def test_unet_overfits_discs():
    torch.manual_seed(0)
    imgs, masks = _discs()
    net = UNet(UNetConfig(base_channels=8))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(60):
        opt.zero_grad()
        loss = weighted_bce_dice(net(imgs)[:, 0], masks, torch.ones_like(masks)).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        assert hard_iou(net(imgs)[:, 0], masks).mean() > 0.9


def test_minisam_overfits_discs_with_clicks():
    torch.manual_seed(0)
    imgs, masks = _discs()
    gen = torch.Generator().manual_seed(1)
    coords, types = zip(
        *(prompts.sample_prompt(m.bool(), gen, p_box=0, p_box_clicks=0) for m in masks), strict=True
    )
    coords, types = torch.stack(coords), torch.stack(types)
    net = MiniSAM(TINY)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(120):
        opt.zero_grad()
        out, iou = net(imgs, coords, types)
        loss, _ = minisam_loss(out, iou, masks, torch.ones_like(masks))
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        out, _ = net(imgs, coords, types)
    assert hard_iou(out[:, 0], masks).mean() > 0.9
