"""Mini-SAM: a small promptable segmenter in the shape of Segment Anything (Kirillov et al., 2023).

The prompt decides *what* to segment. Pipeline:
  image -> stride-8 residual CNN (16x16x128 tokens at 128px) -> self-attention for global context
  clicks/box -> shared frozen Fourier position encoding + learned token-type embedding
  decoder: two-way attention between [iou token, mask tokens, prompt tokens] and image tokens
  masks: upscale image tokens x4, fuse a stride-2 encoder skip (as in SAM-HQ), dot with a
         per-mask hypernetwork of its token -> logits, bilinear x2 to input size
  iou head: predicts the IoU of each mask, used as the score

Ported from 009-mini-networks' MiniSAM with fixes: 3-channel input, (x, y) pixel-centred
coordinates, one position encoding shared by image grid and clicks (re-added to q/k in every
attention), token self-attention, and an explicit PAD type masked out of attention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from boundarybrush.prompts import NUM_TYPES, PAD


@dataclass(frozen=True)
class MiniSAMConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_decoder_layers: int = 2
    n_image_attn_layers: int = 1
    n_masks: int = 1
    stem_channels: int = 32
    skip_channels: int = 16

    def to_dict(self) -> dict:
        return asdict(self)


class PositionEmbeddingRandom(nn.Module):
    """Frozen Gaussian Fourier features of (x, y) in [0, 1] (Tancik et al., 2020; SAM)."""

    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.register_buffer("gaussian", scale * torch.randn(2, dim // 2))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * ((2 * coords - 1) @ self.gaussian)
        return torch.cat([proj.sin(), proj.cos()], dim=-1)

    def grid(self, h: int, w: int) -> torch.Tensor:
        """Pixel-centred encodings of an h x w grid -> [h*w, dim] in row-major order."""
        ys = (torch.arange(h, device=self.gaussian.device) + 0.5) / h
        xs = (torch.arange(w, device=self.gaussian.device) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return self(torch.stack([xx, yy], dim=-1).reshape(-1, 2))


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.short = (
            nn.Identity()
            if stride == 1 and cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.body(x) + self.short(x))


class ImageEncoder(nn.Module):
    """Stride-8 residual CNN; returns the embedding grid and the stride-2 stem features."""

    def __init__(self, cfg: MiniSAMConfig):
        super().__init__()
        s = cfg.stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, s, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(s), nn.GELU()
        )
        self.stages = nn.Sequential(
            ResBlock(s, 2 * s, stride=2),
            ResBlock(2 * s, 4 * s, stride=2),
            ResBlock(4 * s, 4 * s),
        )
        self.neck = nn.Conv2d(4 * s, cfg.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stem(x)
        return self.neck(self.stages(stem)), stem


class Attention(nn.Module):
    """Multi-head attention where positional encodings go into queries and keys only."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, q, k, v, q_pe, k_pe, key_padding_mask=None):
        out, _ = self.attn(q + q_pe, k + k_pe, v, key_padding_mask=key_padding_mask, need_weights=False)
        return out


def _mlp(dim_in: int, hidden: int, dim_out: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    dims = [dim_in] + [hidden] * (layers - 1) + [dim_out]
    for i in range(layers):
        mods.append(nn.Linear(dims[i], dims[i + 1]))
        if i < layers - 1:
            mods.append(nn.GELU())
    return nn.Sequential(*mods)


class TwoWayBlock(nn.Module):
    """SAM two-way block: token self-attn -> tokens->image -> MLP -> image->tokens."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.self_attn, self.n1 = Attention(dim, heads), nn.LayerNorm(dim)
        self.t2i, self.n2 = Attention(dim, heads), nn.LayerNorm(dim)
        self.mlp, self.n3 = _mlp(dim, 4 * dim, dim), nn.LayerNorm(dim)
        self.i2t, self.n4 = Attention(dim, heads), nn.LayerNorm(dim)

    def forward(self, tokens, image, tok_pe, img_pe, pad_mask):
        tokens = self.n1(tokens + self.self_attn(tokens, tokens, tokens, tok_pe, tok_pe, pad_mask))
        tokens = self.n2(tokens + self.t2i(tokens, image, image, tok_pe, img_pe))
        tokens = self.n3(tokens + self.mlp(tokens))
        image = self.n4(image + self.i2t(image, tokens, tokens, img_pe, tok_pe, pad_mask))
        return tokens, image


class MiniSAM(nn.Module):
    def __init__(self, config: MiniSAMConfig | None = None):
        super().__init__()
        self.config = cfg = config or MiniSAMConfig()
        d = cfg.embed_dim
        self.encoder = ImageEncoder(cfg)
        self.pe = PositionEmbeddingRandom(d)
        self.image_attn = nn.ModuleList(
            nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(cfg.n_image_attn_layers)
        )
        self.type_embed = nn.Embedding(NUM_TYPES, d)
        self.output_tokens = nn.Parameter(torch.randn(1 + cfg.n_masks, d) * 0.02)  # [iou, masks...]
        self.decoder = nn.ModuleList(TwoWayBlock(d, cfg.n_heads) for _ in range(cfg.n_decoder_layers))
        self.final_attn, self.final_norm = Attention(d, cfg.n_heads), nn.LayerNorm(d)
        up = cfg.skip_channels
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(d, d // 4, 2, stride=2),
            nn.GroupNorm(1, d // 4),
            nn.GELU(),
            nn.ConvTranspose2d(d // 4, up, 2, stride=2),
            nn.GELU(),
        )
        self.skip_proj = nn.Conv2d(cfg.stem_channels, up, 1)
        self.hyper = nn.ModuleList(_mlp(d, d, up, layers=3) for _ in range(cfg.n_masks))
        self.iou_head = _mlp(d, d, cfg.n_masks, layers=3)

    def encode_image(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """image [B, 3, H, W] normalized -> cached features for any number of prompts."""
        grid, stem = self.encoder(image)
        b, d, h, w = grid.shape
        tokens = grid.flatten(2).transpose(1, 2)
        img_pe = self.pe.grid(h, w)[None]
        for layer in self.image_attn:
            tokens = layer(tokens + img_pe) - img_pe
        return {
            "tokens": tokens,
            "stem": stem,
            "hw": torch.tensor([h, w]),
            "size": torch.tensor(image.shape[-2:]),
        }

    def embed_prompt(self, coords: torch.Tensor, types: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pad = types == PAD
        tokens = self.pe(coords).masked_fill(pad[..., None], 0.0) + self.type_embed(types)
        return tokens, pad

    def decode(self, feats: dict[str, torch.Tensor], coords: torch.Tensor, types: torch.Tensor):
        """coords [N, P, 2], types [N, P] with N == len(feats['tokens'])
        -> (mask logits [N, n_masks, H, W], iou_pred [N, n_masks])."""
        image, stem = feats["tokens"], feats["stem"]
        h, w = feats["hw"].tolist()
        n = image.shape[0]
        prompt, pad = self.embed_prompt(coords, types)
        out = self.output_tokens[None].expand(n, -1, -1)
        tokens = torch.cat([out, prompt], dim=1)
        pad_mask = torch.cat([torch.zeros(n, out.shape[1], dtype=torch.bool, device=pad.device), pad], dim=1)
        tok_pe = tokens
        img_pe = self.pe.grid(h, w)[None].expand(n, -1, -1)
        for block in self.decoder:
            tokens, image = block(tokens, image, tok_pe, img_pe, pad_mask)
        tokens = self.final_norm(tokens + self.final_attn(tokens, image, image, tok_pe, img_pe))

        grid = image.transpose(1, 2).reshape(n, -1, h, w)
        up = self.upscale(grid) + self.skip_proj(stem)
        masks = torch.stack(
            [torch.einsum("nc,nchw->nhw", mlp(tokens[:, 1 + i]), up) for i, mlp in enumerate(self.hyper)],
            dim=1,
        )
        size = tuple(feats["size"].tolist())
        masks = F.interpolate(masks, size=size, mode="bilinear", align_corners=False)
        return masks, self.iou_head(tokens[:, 0])

    def forward(self, image: torch.Tensor, coords: torch.Tensor, types: torch.Tensor):
        """image [B,3,H,W]; coords [B,P,2] or [B,K,P,2] (K prompt sets share one encoder pass).

        Returns masks [B,(K,)n_masks,H,W] logits and iou_pred [B,(K,)n_masks].
        """
        feats = self.encode_image(image)
        if coords.ndim == 3:
            return self.decode(feats, coords, types)
        b, k = coords.shape[:2]
        rep = {
            "tokens": feats["tokens"].repeat_interleave(k, 0),
            "stem": feats["stem"].repeat_interleave(k, 0),
            "hw": feats["hw"],
            "size": feats["size"],
        }
        masks, iou = self.decode(rep, coords.flatten(0, 1), types.flatten(0, 1))
        return masks.unflatten(0, (b, k)), iou.unflatten(0, (b, k))
