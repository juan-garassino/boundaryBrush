"""Plain U-Net (Ronneberger et al., 2015) returning logits, sized for 128px on a CPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 3
    base_channels: int = 16
    depth: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, config: UNetConfig | None = None):
        super().__init__()
        self.config = config or UNetConfig()
        c, d = self.config.base_channels, self.config.depth
        widths = [c * 2**i for i in range(d + 1)]  # 16, 32, 64, 128, 256
        self.inc = _double_conv(self.config.in_channels, widths[0])
        self.downs = nn.ModuleList(
            nn.Sequential(nn.MaxPool2d(2), _double_conv(widths[i], widths[i + 1])) for i in range(d)
        )
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2) for i in reversed(range(d))
        )
        self.up_convs = nn.ModuleList(_double_conv(2 * widths[i], widths[i]) for i in reversed(range(d)))
        self.head = nn.Conv2d(widths[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 3, H, W] (H, W divisible by 2**depth) -> logits [B, 1, H, W]."""
        factor = 2**self.config.depth
        if x.shape[-1] % factor or x.shape[-2] % factor:
            raise ValueError(f"input size {tuple(x.shape[-2:])} must be divisible by {factor}")
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))
        x = skips.pop()
        for up, conv in zip(self.ups, self.up_convs, strict=True):
            x = conv(torch.cat([skips.pop(), up(x)], dim=1))
        return self.head(x)
