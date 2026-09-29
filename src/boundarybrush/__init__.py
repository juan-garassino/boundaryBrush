"""boundaryBrush — promptable segmentation behind one predictor API.

    from boundarybrush import Prompt, get_segmenter
    seg = get_segmenter("slimsam")          # or "unet", "minisam", "mock"
    seg.set_image(rgb_uint8)                 # embed once
    result = seg.predict(Prompt(points=[(120, 80)], labels=[1]))  # mask + score
"""

from __future__ import annotations

__version__ = "0.1.0"

from boundarybrush.domain.enums import Backend  # noqa: E402
from boundarybrush.domain.models import MaskResult, Prompt  # noqa: E402
from boundarybrush.providers.base import Segmenter  # noqa: E402
from boundarybrush.providers.registry import get_segmenter  # noqa: E402

__all__ = ["Backend", "MaskResult", "Prompt", "Segmenter", "__version__", "get_segmenter"]
