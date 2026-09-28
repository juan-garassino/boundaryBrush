from __future__ import annotations

import numpy as np
import pytest

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import MaskResult, Prompt


def test_backend_values_are_lowercase():
    assert [b.value for b in Backend] == ["unet", "minisam", "slimsam", "mock"]


def test_empty_prompt():
    p = Prompt()
    assert p.is_empty
    assert p.points == () and p.box is None


def test_prompt_accepts_lists_and_normalizes_to_tuples():
    p = Prompt(points=[[10, 20], [30.5, 40]], labels=[1, 0], box=[1, 2, 50, 60])
    assert p.points == ((10.0, 20.0), (30.5, 40.0))
    assert p.labels == (1, 0)
    assert p.box == (1.0, 2.0, 50.0, 60.0)
    assert not p.is_empty


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(points=[(1, 2)], labels=[]),  # length mismatch
        dict(points=[(1, 2)], labels=[2]),  # label out of {0, 1}
        dict(points=[(-1, 2)], labels=[1]),  # negative coordinate
        dict(box=(10, 10, 5, 20)),  # x0 >= x1
        dict(box=(10, 10, 20, 10)),  # y0 >= y1
        dict(points=[(1, 2, 3)], labels=[1]),  # not (x, y)
    ],
)
def test_prompt_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        Prompt(**kwargs)


def test_prompt_within_bounds():
    p = Prompt(points=[(99.5, 49.5)], labels=[1], box=(0, 0, 100, 50))
    p.check_within(width=100, height=50)
    with pytest.raises(ValueError):
        Prompt(points=[(100.5, 10)], labels=[1]).check_within(width=100, height=50)


def test_prompt_scaled():
    p = Prompt(points=[(10, 20)], labels=[1], box=(0, 0, 40, 80)).scaled(0.5, 0.25)
    assert p.points == ((5.0, 5.0),)
    assert p.box == (0.0, 0.0, 20.0, 20.0)


def test_mask_result_requires_2d_bool():
    r = MaskResult(mask=np.zeros((4, 5), dtype=bool), score=0.5)
    assert r.mask.shape == (4, 5)
    with pytest.raises(ValueError):
        MaskResult(mask=np.zeros((4, 5), dtype=np.uint8), score=0.5)
    with pytest.raises(ValueError):
        MaskResult(mask=np.zeros((1, 4, 5), dtype=bool), score=0.5)
