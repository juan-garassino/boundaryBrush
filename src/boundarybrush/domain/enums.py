from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    UNET = "unet"
    MINISAM = "minisam"
    SLIMSAM = "slimsam"
    MOCK = "mock"
